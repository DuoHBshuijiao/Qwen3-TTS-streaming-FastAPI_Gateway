# coding=utf-8
"""
FastAPI gateway for Qwen3 TTS — no Gradio dependency at runtime.

Start with env QWEN_TTS_MODEL_PATH set, or use `python -m qwen_tts.gateway`.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, Generator, Optional

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
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
        app.state.tts = _load_model(settings)
        app.state.settings = settings
        yield

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

    def tts() -> Qwen3TTSModel:
        return app.state.tts

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/v1/meta")
    def meta():
        m = tts().model
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
        if tts().model.tts_model_type != "custom_voice":
            raise HTTPException(
                status_code=400,
                detail="Current model is not CustomVoice; load Qwen3-TTS-*-CustomVoice.",
            )
        g = body.gen.model_dump(exclude_none=True)
        kwargs = _merge_gen_kwargs(g)
        try:
            wavs, sr = tts().generate_custom_voice(
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
        if tts().model.tts_model_type != "voice_design":
            raise HTTPException(
                status_code=400,
                detail="Current model is not VoiceDesign; load Qwen3-TTS-*-VoiceDesign.",
            )
        g = body.gen.model_dump(exclude_none=True)
        kwargs = _merge_gen_kwargs(g)
        try:
            wavs, sr = tts().generate_voice_design(
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
        if tts().model.tts_model_type != "base":
            raise HTTPException(
                status_code=400,
                detail="Current model is not Base; load Qwen3-TTS-*-Base for voice clone.",
            )
        raw = await ref_audio.read()
        try:
            wav, sr = read_wav_from_upload(raw)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid audio file: {e}") from e
        kwargs = _merge_gen_kwargs({})
        model = tts()

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
        if tts().model.tts_model_type != "base":
            raise HTTPException(
                status_code=400,
                detail="Current model is not Base; load Qwen3-TTS-*-Base for voice clone.",
            )
        try:
            wav, sr = read_wav_from_base64(body.ref_audio_base64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid ref_audio_base64: {e}") from e
        g = body.gen.model_dump(exclude_none=True)
        kwargs = _merge_gen_kwargs(g)
        try:
            wavs, out_sr = tts().generate_voice_clone(
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
        if tts().model.tts_model_type != "base":
            raise HTTPException(
                status_code=400,
                detail="Current model is not Base; load Qwen3-TTS-*-Base.",
            )
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

        kwargs = _merge_gen_kwargs({})
        model = tts()

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

    def _sse_voice_clone_stream(body: VoiceCloneStreamBody) -> Generator[bytes, None, None]:
        g = body.gen.model_dump(exclude_none=True)
        kwargs = _merge_gen_kwargs(g)
        try:
            wav, sr = read_wav_from_base64(body.ref_audio_base64)
        except Exception as e:
            raise ValueError(f"Invalid ref_audio_base64: {e}") from e

        for chunk, chunk_sr in tts().stream_generate_voice_clone(
            text=body.text.strip(),
            language=body.language,
            ref_audio=(wav, sr),
            ref_text=(body.ref_text.strip() if body.ref_text else None),
            x_vector_only_mode=body.x_vector_only,
            emit_every_frames=body.emit_every_frames,
            decode_window_frames=body.decode_window_frames,
            overlap_samples=body.overlap_samples,
            **kwargs,
        ):
            payload = {
                "sample_rate": int(chunk_sr),
                "pcm_b64": base64.b64encode(np.asarray(chunk, dtype=np.float32).tobytes()).decode("ascii"),
            }
            yield f"event: chunk\ndata: {json.dumps(payload)}\n\n".encode()
        yield b"event: done\ndata: {}\n\n"

    @app.post("/v1/tts/voice_clone/stream")
    def voice_clone_stream_sse(body: VoiceCloneStreamBody):
        if tts().model.tts_model_type != "base":
            raise HTTPException(
                status_code=400,
                detail="Current model is not Base; load Qwen3-TTS-*-Base.",
            )
        if not hasattr(tts(), "stream_generate_voice_clone"):
            raise HTTPException(
                status_code=501,
                detail="Streaming is not available in this install (needs streaming-capable fork).",
            )
        try:
            gen = _sse_voice_clone_stream(body)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        return StreamingResponse(
            gen,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return app


# Uvicorn: export QWEN_TTS_MODEL_PATH=...  then  uvicorn qwen_tts.gateway.app:app
if os.environ.get("QWEN_TTS_MODEL_PATH", "").strip():
    app = create_app()
else:
    app = None  # type: ignore[assignment]
