# coding=utf-8
"""Environment / CLI configuration for the HTTP gateway."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional


def _split_csv(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


@dataclass
class GatewaySettings:
    """Loaded once when the app starts (see create_app)."""

    model_path: str
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    flash_attn: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    # If set, POST /v1/admin/unload requires Authorization: Bearer <token> (or X-Admin-Token).
    admin_token: Optional[str] = None

    @classmethod
    def from_env(cls) -> "GatewaySettings":
        path = os.environ.get("QWEN_TTS_MODEL_PATH", "").strip()
        if not path:
            raise RuntimeError(
                "Set QWEN_TTS_MODEL_PATH to a Hugging Face repo id or local model directory "
                "(e.g. Qwen/Qwen3-TTS-12Hz-1.7B-Base)."
            )
        device = os.environ.get("QWEN_TTS_DEVICE", "cuda:0").strip()
        dtype = os.environ.get("QWEN_TTS_DTYPE", "bfloat16").strip().lower()
        flash = os.environ.get("QWEN_TTS_FLASH_ATTN", "1").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        cors = os.environ.get("QWEN_TTS_CORS_ORIGINS", "*").strip()
        origins = ["*"] if cors == "*" else _split_csv(cors)
        admin = os.environ.get("QWEN_TTS_ADMIN_TOKEN", "").strip()
        admin_token = admin if admin else None
        return cls(
            model_path=path,
            device=device,
            dtype=dtype,
            flash_attn=flash,
            cors_origins=origins,
            admin_token=admin_token,
        )
