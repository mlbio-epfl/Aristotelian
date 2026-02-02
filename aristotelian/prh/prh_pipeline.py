"""PRH pipeline: activation extraction for text/vision and alignment helpers."""

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
import torch

from .prh_models import load_text_model, load_vision_model


def _pool_tokens(x: torch.Tensor, *, mode: str = "mean") -> torch.Tensor:
    if x.dim() == 3:
        if mode == "mean":
            return x.mean(dim=1)
        if mode == "cls":
            return x[:, 0, :]
        if mode == "last":
            return x[:, -1, :]
    return x


def _pool_tokens_masked(
    x: torch.Tensor, mask: torch.Tensor, *, mode: str
) -> torch.Tensor:
    if mode == "mean":
        m = mask.unsqueeze(-1)
        return (x * m).sum(dim=1) / (m.sum(dim=1) + 1e-8)
    if mode == "cls":
        return x[:, 0, :]
    if mode == "last":
        return x[:, -1, :]
    raise ValueError(f"unknown pooling mode: {mode}")


def collect_text_activations(
    texts: Iterable[str],
    *,
    model_name: Optional[str] = None,
    tokenizer=None,
    model=None,
    device: str = "cpu",
    layers: Optional[List[int]] = None,
    max_length: int | None = None,
    pool: str = "mean",
    batch_size: Optional[int] = None,
) -> List[np.ndarray]:
    """Collect text activations (per layer) as numpy arrays [n, d]."""
    if model is None or tokenizer is None:
        if model_name is None:
            raise ValueError(
                "model_name is required when tokenizer/model are not provided"
            )
        tokenizer, model = load_text_model(model_name, device=device)

    texts_list = list(texts)
    if not texts_list:
        return []
    if batch_size is None:
        batch_size = len(texts_list)
    batches = [
        texts_list[i : i + batch_size] for i in range(0, len(texts_list), batch_size)
    ]

    acts = None
    for batch in batches:
        enc = tokenizer(
            batch,
            padding=True,
            truncation=max_length is not None,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        hidden = out.hidden_states
        mask = enc.get("attention_mask")

        if layers is None:
            layers = list(range(len(hidden)))
        if acts is None:
            acts = [[] for _ in layers]

        for out_idx, layer_idx in enumerate(layers):
            if mask is None:
                pooled = _pool_tokens(hidden[layer_idx], mode=pool)
            else:
                pooled = _pool_tokens_masked(hidden[layer_idx], mask, mode=pool)
            acts[out_idx].append(pooled.detach().cpu())

    if acts is None:
        return []
    return [torch.cat(layer_chunks, dim=0).numpy() for layer_chunks in acts]


def collect_vision_activations(
    images: Iterable[torch.Tensor],
    *,
    model_name: Optional[str] = None,
    model=None,
    device: str = "cpu",
    layers: Optional[List[int]] = None,
    pool: str = "mean",
    batch_size: Optional[int] = None,
    transform=None,
) -> List[np.ndarray]:
    """Collect vision activations (per layer) as numpy arrays [n, d]."""
    if model is None:
        if model_name is None:
            raise ValueError("model_name is required when model is not provided")
        model = load_vision_model(model_name, device=device)

    imgs_list = list(images)
    if not imgs_list:
        return []
    if transform is not None:
        imgs_list = [transform(img) for img in imgs_list]
    if batch_size is None:
        batch_size = len(imgs_list)
    batches = [
        imgs_list[i : i + batch_size] for i in range(0, len(imgs_list), batch_size)
    ]

    acts = None
    for batch in batches:
        imgs = torch.stack(batch).to(device)
        with torch.no_grad():
            if hasattr(model, "get_intermediate_layers"):
                hs = model.get_intermediate_layers(imgs, n=layers if layers else None)
                hidden = list(hs)
            elif hasattr(model, "forward_features"):
                hidden = [model.forward_features(imgs)]
            else:
                output = model(imgs)
                if isinstance(output, dict):
                    hidden = list(output.values())
                else:
                    hidden = [output]

        if layers is None:
            layers = list(range(len(hidden)))
        if acts is None:
            acts = [[] for _ in layers]

        for out_idx, layer_idx in enumerate(layers):
            pooled = _pool_tokens(hidden[layer_idx], mode=pool)
            acts[out_idx].append(pooled.detach().cpu())

    if acts is None:
        return []
    return [torch.cat(layer_chunks, dim=0).numpy() for layer_chunks in acts]
