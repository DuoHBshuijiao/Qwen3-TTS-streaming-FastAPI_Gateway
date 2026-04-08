# coding=utf-8
"""Run: python -m qwen_tts.gateway --model <hf_id_or_path> [--host 127.0.0.1] [--port 8080]"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    p = argparse.ArgumentParser(description="Qwen3 TTS FastAPI gateway (no Gradio)")
    p.add_argument(
        "--model",
        "-m",
        default=os.environ.get("QWEN_TTS_MODEL_PATH"),
        help="HF repo id or local path (or set QWEN_TTS_MODEL_PATH)",
    )
    p.add_argument("--host", default=os.environ.get("QWEN_TTS_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("QWEN_TTS_PORT", "8080")))
    p.add_argument("--device", default=os.environ.get("QWEN_TTS_DEVICE", "cuda:0"))
    p.add_argument("--dtype", default=os.environ.get("QWEN_TTS_DTYPE", "bfloat16"))
    p.add_argument(
        "--reload",
        action="store_true",
        help="Uvicorn auto-reload (dev only; loads model in subprocess)",
    )
    args = p.parse_args()
    if not args.model or not str(args.model).strip():
        print("error: pass --model or set QWEN_TTS_MODEL_PATH", file=sys.stderr)
        sys.exit(2)
    os.environ["QWEN_TTS_MODEL_PATH"] = str(args.model).strip()
    os.environ["QWEN_TTS_DEVICE"] = args.device
    os.environ["QWEN_TTS_DTYPE"] = args.dtype

    import uvicorn

    uvicorn.run(
        "qwen_tts.gateway.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
