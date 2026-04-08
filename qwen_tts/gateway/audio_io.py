# coding=utf-8
"""WAV encode/decode helpers for the gateway."""

from __future__ import annotations

import base64
import io
from typing import Tuple

import numpy as np
import soundfile as sf


def wav_bytes_from_array(wav: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    x = np.asarray(wav, dtype=np.float32)
    if x.ndim > 1:
        x = np.mean(x, axis=-1)
    sf.write(buf, x, int(sample_rate), format="WAV", subtype="PCM_16")
    return buf.getvalue()


def read_wav_from_upload(content: bytes) -> Tuple[np.ndarray, int]:
    buf = io.BytesIO(content)
    wav, sr = sf.read(buf, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=-1)
    return wav.astype(np.float32), int(sr)


def read_wav_from_base64(b64: str) -> Tuple[np.ndarray, int]:
    raw = base64.b64decode(b64)
    return read_wav_from_upload(raw)
