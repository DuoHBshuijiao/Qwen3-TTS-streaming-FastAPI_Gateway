# coding=utf-8
"""Load serialized voice-clone prompt files (.pt) compatible with the Gradio demo."""

from __future__ import annotations

from typing import List

import torch

from qwen_tts import VoiceClonePromptItem


def load_voice_clone_items_from_pt(path: str) -> List[VoiceClonePromptItem]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # PyTorch < 2.0
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "items" not in payload:
        raise ValueError("Invalid prompt file: expected dict with 'items' key.")
    items_raw = payload["items"]
    if not isinstance(items_raw, list) or len(items_raw) == 0:
        raise ValueError("Empty or invalid 'items' in prompt file.")

    items: List[VoiceClonePromptItem] = []
    for d in items_raw:
        if not isinstance(d, dict):
            raise ValueError("Invalid item in prompt file.")
        ref_code = d.get("ref_code", None)
        if ref_code is not None and not torch.is_tensor(ref_code):
            ref_code = torch.tensor(ref_code)
        ref_spk = d.get("ref_spk_embedding", None)
        if ref_spk is None:
            raise ValueError("Missing ref_spk_embedding in prompt item.")
        if not torch.is_tensor(ref_spk):
            ref_spk = torch.tensor(ref_spk)
        items.append(
            VoiceClonePromptItem(
                ref_code=ref_code,
                ref_spk_embedding=ref_spk,
                x_vector_only_mode=bool(d.get("x_vector_only_mode", False)),
                icl_mode=bool(d.get("icl_mode", not bool(d.get("x_vector_only_mode", False)))),
                ref_text=d.get("ref_text", None),
            )
        )
    return items
