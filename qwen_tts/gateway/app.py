# coding=utf-8
"""
FastAPI gateway for Qwen3 TTS — no Gradio dependency at runtime.

Start with env QWEN_TTS_MODEL_PATH set, or use `python -m qwen_tts.gateway`.
"""

from __future__ import annotations

import asyncio
import base64
import gc
import json
import os
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Callable, Dict, Generator, Optional

import numpy as np
import torch
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from qwen_tts import Qwen3TTSModel

from .audio_io import read_wav_from_base64, read_wav_from_upload, wav_bytes_from_array
from .config import GatewaySettings
from .prompt_io import load_voice_clone_items_from_pt


def _dtype_from_str(s: str):
    s = (s or "").strip().lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {s}")


def _load_model(settings: GatewaySettings) -> Qwen3TTSModel:
    dtype = _dtype_from_str(settings.dtype)
    attn = "flash_attention_2" if settings.flash_attn else None
    return Qwen3TTSModel.from_pretrained(
        settings.model_path,
        device_map=settings.device,
        dtype=dtype,
        attn_implementation=attn,
    )


def _unload_tts_model(tts: Qwen3TTSModel) -> None:
    """Drop references and release GPU memory (best-effort)."""
    try:
        del tts.model
    except Exception:
        pass
    try:
        del tts.processor
    except Exception:
        pass
    try:
        del tts
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _assert_admin_token(
    settings: GatewaySettings,
    authorization: Optional[str],
    x_admin_token: Optional[str],
) -> None:
    if not settings.admin_token:
        raise HTTPException(
            status_code=501,
            detail="Admin API disabled. Set QWEN_TTS_ADMIN_TOKEN to enable /v1/admin/unload.",
        )
    token: Optional[str] = None
    if x_admin_token:
        token = x_admin_token.strip()
    elif authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    if not token or token != settings.admin_token:
        raise HTTPException(status_code=403, detail="Invalid or missing admin token.")


def _merge_gen_kwargs(extra: Dict[str, Any]) -> Dict[str, Any]:
    """Merge optional gen params; keep explicit False for bools (e.g. do_sample=False)."""
    allowed = {
        "max_new_tokens",
        "temperature",
        "top_k",
        "top_p",
        "repetition_penalty",
        "subtalker_top_k",
        "subtalker_top_p",
        "subtalker_temperature",
        "non_streaming_mode",
        "do_sample",
    }
    out: Dict[str, Any] = {}
    for k, v in extra.items():
        if k not in allowed:
            continue
        if v is None:
            continue
        out[k] = v
    return out


class GenOptions(BaseModel):
    """Optional generation parameters forwarded to the model."""

    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    subtalker_top_k: Optional[int] = None
    subtalker_top_p: Optional[float] = None
    subtalker_temperature: Optional[float] = None
    non_streaming_mode: bool = False
    do_sample: Optional[bool] = None


class CustomVoiceBody(BaseModel):
    text: str
    speaker: str
    language: str = "Auto"
    instruct: Optional[str] = None
    gen: GenOptions = Field(default_factory=GenOptions)


class VoiceDesignBody(BaseModel):
    text: str
    instruct: str
    language: str = "Auto"
    gen: GenOptions = Field(default_factory=GenOptions)


class VoiceCloneJsonBody(BaseModel):
    text: str
    language: str = "Auto"
    ref_text: Optional[str] = None
    x_vector_only: bool = False
    ref_audio_base64: str = Field(..., description="WAV (or soundfile-readable) bytes, base64-encoded.")
    gen: GenOptions = Field(default_factory=GenOptions)


class VoiceCloneStreamBody(BaseModel):
    text: str
    language: str = "Auto"
    ref_text: Optional[str] = None
    x_vector_only: bool = False
    ref_audio_base64: str
    emit_every_frames: int = 8
    decode_window_frames: int = 80
    overlap_samples: int = 0
    gen: GenOptions = Field(default_factory=GenOptions)


def create_app(settings: Optional[GatewaySettings] = None) -> FastAPI:
    if settings is None:
        settings = GatewaySettings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model_loaded = True
        app.state._active_inferences = 0
        app.state._inference_cond = threading.Condition()
        app.state.cancel_generation = threading.Event()
        app.state.tts = _load_model(settings)
        app.state.settings = settings
        yield
        if getattr(app.state, "tts", None) is not None:
            _unload_tts_model(app.state.tts)
            app.state.tts = None
        app.state.model_loaded = False

    app = FastAPI(
        title="Qwen3 TTS Gateway",
        version="1.0.0",
        lifespan=lifespan,
    )
    # Browsers forbid credentials + wildcard origin; use credentials only with explicit origins.
    _cors_origins = settings.cors_origins
    _allow_cred = _cors_origins != ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=_allow_cred,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _tts() -> Qwen3TTSModel:
        if not getattr(app.state, "model_loaded", True):
            raise HTTPException(status_code=503, detail="Model unloaded; restart the gateway process to load again.")
        t = getattr(app.state, "tts", None)
        if t is None:
            raise HTTPException(status_code=503, detail="Model unloaded; restart the gateway process to load again.")
        return t

    @contextmanager
    def _inference_scope():
        if not getattr(app.state, "model_loaded", True) or app.state.tts is None:
            raise HTTPException(
                status_code=503,
                detail="Model unloaded; restart the gateway process to load again.",
            )
        app.state._active_inferences += 1
        try:
            yield
        finally:
            app.state._active_inferences -= 1
            with app.state._inference_cond:
                app.state._inference_cond.notify_all()

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model_loaded": bool(getattr(app.state, "model_loaded", True) and getattr(app.state, "tts", None)),
        }

    @app.get("/v1/meta")
    def meta():
        m = _tts().model
        out = {
            "model_path": settings.model_path,
            "device": settings.device,
            "dtype": settings.dtype,
            "tts_model_type": getattr(m, "tts_model_type", None),
            "tokenizer_type": getattr(m, "tokenizer_type", None),
        }
        if callable(getattr(m, "get_supported_languages", None)):
            try:
                out["supported_languages"] = m.get_supported_languages()
            except Exception:
                out["supported_languages"] = None
        if callable(getattr(m, "get_supported_speakers", None)):
            try:
                out["supported_speakers"] = m.get_supported_speakers()
            except Exception:
                out["supported_speakers"] = None
        return out

    @app.post("/v1/tts/custom_voice")
    def custom_voice(body: CustomVoiceBody):
        with _inference_scope():
            if _tts().model.tts_model_type != "custom_voice":
                raise HTTPException(
                    status_code=400,
                    detail="Current model is not CustomVoice; load Qwen3-TTS-*-CustomVoice.",
                )
            g = body.gen.model_dump(exclude_none=True)
            kwargs = _merge_gen_kwargs(g)
            try:
                wavs, sr = _tts().generate_custom_voice(
                    text=body.text.strip(),
                    speaker=body.speaker,
                    language=body.language,
                    instruct=(body.instruct or "").strip() or None,
                    **kwargs,
                )
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
            return Response(content=wav_bytes_from_array(wavs[0], sr), media_type="audio/wav")

    @app.post("/v1/tts/voice_design")
    def voice_design(body: VoiceDesignBody):
        with _inference_scope():
            if _tts().model.tts_model_type != "voice_design":
                raise HTTPException(
                    status_code=400,
                    detail="Current model is not VoiceDesign; load Qwen3-TTS-*-VoiceDesign.",
                )
            g = body.gen.model_dump(exclude_none=True)
            kwargs = _merge_gen_kwargs(g)
            try:
                wavs, sr = _tts().generate_voice_design(
                    text=body.text.strip(),
                    instruct=body.instruct.strip(),
                    language=body.language,
                    **kwargs,
                )
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
            return Response(content=wav_bytes_from_array(wavs[0], sr), media_type="audio/wav")

    @app.post("/v1/tts/voice_clone")
    async def voice_clone(
        text: str = Form(...),
        language: str = Form("Auto"),
        ref_text: Optional[str] = Form(None),
        x_vector_only: bool = Form(False),
        ref_audio: UploadFile = File(...),
    ):
        raw = await ref_audio.read()
        try:
            wav, sr = read_wav_from_upload(raw)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid audio file: {e}") from e
        with _inference_scope():
            if _tts().model.tts_model_type != "base":
                raise HTTPException(
                    status_code=400,
                    detail="Current model is not Base; load Qwen3-TTS-*-Base for voice clone.",
                )
            kwargs = _merge_gen_kwargs({})
            model = _tts()

            def _run():
                return model.generate_voice_clone(
                    text=text.strip(),
                    language=language,
                    ref_audio=(wav, sr),
                    ref_text=(ref_text.strip() if ref_text else None),
                    x_vector_only_mode=x_vector_only,
                    **kwargs,
                )

            try:
                wavs, out_sr = await asyncio.to_thread(_run)
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
            return Response(content=wav_bytes_from_array(wavs[0], out_sr), media_type="audio/wav")

    @app.post("/v1/tts/voice_clone/json")
    def voice_clone_json(body: VoiceCloneJsonBody):
        try:
            wav, sr = read_wav_from_base64(body.ref_audio_base64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid ref_audio_base64: {e}") from e
        with _inference_scope():
            if _tts().model.tts_model_type != "base":
                raise HTTPException(
                    status_code=400,
                    detail="Current model is not Base; load Qwen3-TTS-*-Base for voice clone.",
                )
            g = body.gen.model_dump(exclude_none=True)
            kwargs = _merge_gen_kwargs(g)
            try:
                wavs, out_sr = _tts().generate_voice_clone(
                    text=body.text.strip(),
                    language=body.language,
                    ref_audio=(wav, sr),
                    ref_text=(body.ref_text.strip() if body.ref_text else None),
                    x_vector_only_mode=body.x_vector_only,
                    **kwargs,
                )
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
            return Response(content=wav_bytes_from_array(wavs[0], out_sr), media_type="audio/wav")

    @app.post("/v1/tts/voice_clone/prompt")
    async def voice_clone_prompt(
        text: str = Form(...),
        language: str = Form("Auto"),
        voice_prompt: UploadFile = File(..., description="voice_clone_prompt_*.pt from Gradio demo"),
    ):
        fd, path = None, None
        try:
            import tempfile

            suffix = os.path.splitext(voice_prompt.filename or "")[1] or ".pt"
            fd, path = tempfile.mkstemp(prefix="vcp_", suffix=suffix)
            os.close(fd)
            fd = None
            content = await voice_prompt.read()
            with open(path, "wb") as f:
                f.write(content)
            items = load_voice_clone_items_from_pt(path)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        finally:
            if path and os.path.isfile(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

        with _inference_scope():
            if _tts().model.tts_model_type != "base":
                raise HTTPException(
                    status_code=400,
                    detail="Current model is not Base; load Qwen3-TTS-*-Base.",
                )
            kwargs = _merge_gen_kwargs({})
            model = _tts()

            def _run_prompt():
                return model.generate_voice_clone(
                    text=text.strip(),
                    language=language,
                    voice_clone_prompt=items,
                    **kwargs,
                )

            try:
                wavs, out_sr = await asyncio.to_thread(_run_prompt)
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
            return Response(content=wav_bytes_from_array(wavs[0], out_sr), media_type="audio/wav")

    def _sse_voice_clone_stream(
        body: VoiceCloneStreamBody,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> Generator[bytes, None, None]:
        g = body.gen.model_dump(exclude_none=True)
        kwargs = _merge_gen_kwargs(g)
        try:
            wav, sr = read_wav_from_base64(body.ref_audio_base64)
        except Exception as e:
            raise ValueError(f"Invalid ref_audio_base64: {e}") from e

        for chunk, chunk_sr in _tts().stream_generate_voice_clone(
            text=body.text.strip(),
            language=body.language,
            ref_audio=(wav, sr),
            ref_text=(body.ref_text.strip() if body.ref_text else None),
            x_vector_only_mode=body.x_vector_only,
            emit_every_frames=body.emit_every_frames,
            decode_window_frames=body.decode_window_frames,
            overlap_samples=body.overlap_samples,
            cancel_check=cancel_check,
            **kwargs,
        ):
            payload = {
                "sample_rate": int(chunk_sr),
                "pcm_b64": base64.b64encode(np.asarray(chunk, dtype=np.float32).tobytes()).decode("ascii"),
            }
            yield f"event: chunk\ndata: {json.dumps(payload)}\n\n".encode()
        yield b"event: done\ndata: {}\n\n"

    @app.post("/v1/tts/voice_clone/stream")
    async def voice_clone_stream_sse(request: Request, body: VoiceCloneStreamBody):
        if _tts().model.tts_model_type != "base":
            raise HTTPException(
                status_code=400,
                detail="Current model is not Base; load Qwen3-TTS-*-Base.",
            )
        if not hasattr(_tts(), "stream_generate_voice_clone"):
            raise HTTPException(
                status_code=501,
                detail="Streaming is not available in this install (needs streaming-capable fork).",
            )
        try:
            read_wav_from_base64(body.ref_audio_base64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid ref_audio_base64: {e}") from e

        disconnected = threading.Event()

        def cancel_check() -> bool:
            if disconnected.is_set():
                return True
            cg = getattr(app.state, "cancel_generation", None)
            return bool(cg is not None and cg.is_set())

        async def watch_disconnect() -> None:
            try:
                while True:
                    if await request.is_disconnected():
                        disconnected.set()
                        return
                    await asyncio.sleep(0.05)
            except asyncio.CancelledError:
                disconnected.set()
                raise

        watch_task = asyncio.create_task(watch_disconnect())

        def stream_gen() -> Generator[bytes, None, None]:
            try:
                with _inference_scope():
                    yield from _sse_voice_clone_stream(body, cancel_check=cancel_check)
            finally:
                watch_task.cancel()

        return StreamingResponse(
            stream_gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/v1/admin/unload")
    async def admin_unload(
        authorization: Optional[str] = Header(None),
        x_admin_token: Optional[str] = Header(None),
    ):
        _assert_admin_token(settings, authorization, x_admin_token)
        app.state.cancel_generation.set()
        deadline = time.monotonic() + 120.0
        with app.state._inference_cond:
            while app.state._active_inferences > 0:
                if time.monotonic() >= deadline:
                    raise HTTPException(
                        status_code=503,
                        detail="Timeout waiting for in-flight inference to stop; try again.",
                    )
                remaining = max(0.0, deadline - time.monotonic())
                app.state._inference_cond.wait(timeout=min(1.0, remaining))
        tts = getattr(app.state, "tts", None)
        if tts is not None:
            _unload_tts_model(tts)
        app.state.tts = None
        app.state.model_loaded = False
        app.state.cancel_generation.clear()
        return {"ok": True, "detail": "Model unloaded and GPU cache cleared. Restart the process to load again."}

    return app


# Uvicorn: export QWEN_TTS_MODEL_PATH=...  then  uvicorn qwen_tts.gateway.app:app
if os.environ.get("QWEN_TTS_MODEL_PATH", "").strip():
    app = create_app()
else:
    app = None  # type: ignore[assignment]
