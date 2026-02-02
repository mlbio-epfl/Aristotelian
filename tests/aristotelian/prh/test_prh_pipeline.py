from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pytest
import torch

from aristotelian.prh.prh_pipeline import (
    _pool_tokens,
    _pool_tokens_masked,
    collect_text_activations,
    collect_vision_activations,
)


class DummyFeatureExtractor:
    def __init__(self, seq_len: int = 5, hidden: int = 7):
        self.seq_len = seq_len
        self.hidden = hidden

    def __call__(self, imgs: torch.Tensor):
        batch = imgs.shape[0]
        return {
            "block0": torch.randn(batch, self.seq_len, self.hidden),
            "block1": torch.randn(batch, self.seq_len, self.hidden),
        }


class DummyIntermediateModel:
    def __init__(self, seq_len: int = 4, hidden: int = 6):
        self.seq_len = seq_len
        self.hidden = hidden

    def get_intermediate_layers(self, imgs: torch.Tensor, n=None):
        batch = imgs.shape[0]
        layers = 2 if n is None else len(n)
        return tuple(
            torch.randn(batch, self.seq_len, self.hidden) for _ in range(layers)
        )


class DummyForwardFeaturesModel:
    def __init__(self, seq_len: int = 3, hidden: int = 5):
        self.seq_len = seq_len
        self.hidden = hidden

    def forward_features(self, imgs: torch.Tensor):
        batch = imgs.shape[0]
        return torch.randn(batch, self.seq_len, self.hidden)


def test_collect_vision_activations_with_dict_output():
    images = [torch.randn(3, 2, 2) for _ in range(4)]
    model = DummyFeatureExtractor()

    acts = collect_vision_activations(
        images,
        model=model,
        device="cpu",
        pool="cls",
        batch_size=2,
    )

    assert len(acts) == 2
    assert acts[0].shape == (4, 7)
    assert acts[1].shape == (4, 7)


def test_collect_vision_activations_with_intermediate_layers():
    images = [torch.randn(3, 2, 2) for _ in range(3)]
    model = DummyIntermediateModel()

    acts = collect_vision_activations(
        images,
        model=model,
        device="cpu",
        pool="mean",
        batch_size=3,
    )

    assert len(acts) == 2
    assert acts[0].shape == (3, 6)
    assert acts[1].shape == (3, 6)


def test_collect_vision_activations_with_forward_features():
    images = [torch.randn(3, 2, 2) for _ in range(2)]
    model = DummyForwardFeaturesModel()

    acts = collect_vision_activations(
        images,
        model=model,
        device="cpu",
        pool="last",
        batch_size=2,
    )

    assert len(acts) == 1
    assert acts[0].shape == (2, 5)


def test_collect_vision_activations_empty_input_returns_empty():
    acts = collect_vision_activations([], model=DummyFeatureExtractor(), device="cpu")
    assert acts == []


def test_collect_vision_activations_requires_model_or_name():
    with pytest.raises(ValueError, match="model_name is required"):
        collect_vision_activations([], model=None, model_name=None)


# --- Text activation tests ---


@dataclass
class DummyTextOutput:
    """Mimics HuggingFace transformer output with hidden_states."""

    hidden_states: Tuple[torch.Tensor, ...]


class DummyTextModel:
    """Deterministic dummy text model for testing."""

    def __init__(self, num_layers: int = 3, hidden_dim: int = 8, seed: int = 42):
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.seed = seed
        self._call_count = 0

    def __call__(self, input_ids, attention_mask=None, **kwargs):
        batch_size, seq_len = input_ids.shape
        # Use deterministic generation based on input content
        gen = torch.Generator()
        gen.manual_seed(self.seed + input_ids.sum().item())
        hidden_states = tuple(
            torch.randn(batch_size, seq_len, self.hidden_dim, generator=gen)
            for _ in range(self.num_layers)
        )
        return DummyTextOutput(hidden_states=hidden_states)

    def to(self, device):
        return self

    def eval(self):
        return self

    def parameters(self):
        return iter([torch.zeros(1)])


class DummyTokenizer:
    """Deterministic dummy tokenizer for testing."""

    def __init__(self, vocab_size: int = 100):
        self.vocab_size = vocab_size
        self.pad_token = "[PAD]"
        self.padding_side = "left"

    def __call__(
        self,
        texts,
        padding=True,
        truncation=False,
        max_length=None,
        return_tensors="pt",
    ):
        # Deterministic tokenization: hash each text to get consistent token ids
        batch_ids = []
        max_len = 0
        for text in texts:
            # Create deterministic token ids based on text content
            ids = [
                hash(text + str(i)) % self.vocab_size
                for i in range(len(text.split()) + 2)
            ]
            batch_ids.append(ids)
            max_len = max(max_len, len(ids))

        # Pad to max length
        padded = []
        masks = []
        for ids in batch_ids:
            pad_len = max_len - len(ids)
            padded.append([0] * pad_len + ids)  # left padding
            masks.append([0] * pad_len + [1] * len(ids))

        return {
            "input_ids": torch.tensor(padded),
            "attention_mask": torch.tensor(masks),
        }


def _collect_text_activations_reference(
    texts,
    tokenizer,
    model,
    device: str = "cpu",
    layers=None,
    max_length=None,
    pool: str = "mean",
    batch_size=None,
) -> List[np.ndarray]:
    """Reference implementation (old algorithm) that accumulates all hidden states."""
    texts_list = list(texts)
    if batch_size is None:
        batch_size = len(texts_list)
    batches = [
        texts_list[i : i + batch_size] for i in range(0, len(texts_list), batch_size)
    ]

    # Old approach: accumulate all hidden states first
    hidden_batches = []
    mask_batches = []
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
        hidden_batches.append(out.hidden_states)
        mask_batches.append(enc.get("attention_mask"))

    hidden = hidden_batches[0]
    if layers is None:
        layers = list(range(len(hidden)))
    acts = []
    for idx in layers:
        pooled_list = []
        for hb, mb in zip(hidden_batches, mask_batches):
            if mb is None:
                pooled = _pool_tokens(hb[idx], mode=pool)
            else:
                pooled = _pool_tokens_masked(hb[idx], mb, mode=pool)
            pooled_list.append(pooled.detach().cpu())
        acts.append(torch.cat(pooled_list, dim=0).numpy())
    return acts


class TestCollectTextActivationsEquivalence:
    """Test that the memory-optimized implementation matches the reference."""

    def test_equivalence_mean_pooling(self):
        """Test equivalence with mean pooling."""
        texts = ["hello world", "this is a test", "another sentence here", "short"]
        tokenizer = DummyTokenizer()
        model = DummyTextModel(num_layers=4, hidden_dim=16)

        # Get results from both implementations
        reference = _collect_text_activations_reference(
            texts, tokenizer, model, pool="mean", batch_size=2
        )

        # Reset model state for fair comparison (use same seed)
        model2 = DummyTextModel(num_layers=4, hidden_dim=16)
        optimized = collect_text_activations(
            texts, tokenizer=tokenizer, model=model2, pool="mean", batch_size=2
        )

        assert len(reference) == len(optimized)
        for ref, opt in zip(reference, optimized):
            np.testing.assert_allclose(ref, opt, rtol=1e-5, atol=1e-6)

    def test_equivalence_cls_pooling(self):
        """Test equivalence with CLS token pooling."""
        texts = ["hello world", "testing cls", "more text"]
        tokenizer = DummyTokenizer()
        model = DummyTextModel(num_layers=3, hidden_dim=8)

        reference = _collect_text_activations_reference(
            texts, tokenizer, model, pool="cls", batch_size=1
        )

        model2 = DummyTextModel(num_layers=3, hidden_dim=8)
        optimized = collect_text_activations(
            texts, tokenizer=tokenizer, model=model2, pool="cls", batch_size=1
        )

        assert len(reference) == len(optimized)
        for ref, opt in zip(reference, optimized):
            np.testing.assert_allclose(ref, opt, rtol=1e-5, atol=1e-6)

    def test_equivalence_last_pooling(self):
        """Test equivalence with last token pooling."""
        texts = ["one", "two words", "three word sentence", "four words in here"]
        tokenizer = DummyTokenizer()
        model = DummyTextModel(num_layers=5, hidden_dim=12)

        reference = _collect_text_activations_reference(
            texts, tokenizer, model, pool="last", batch_size=2
        )

        model2 = DummyTextModel(num_layers=5, hidden_dim=12)
        optimized = collect_text_activations(
            texts, tokenizer=tokenizer, model=model2, pool="last", batch_size=2
        )

        assert len(reference) == len(optimized)
        for ref, opt in zip(reference, optimized):
            np.testing.assert_allclose(ref, opt, rtol=1e-5, atol=1e-6)

    def test_equivalence_single_batch(self):
        """Test equivalence when all samples fit in one batch."""
        texts = ["a", "b", "c"]
        tokenizer = DummyTokenizer()
        model = DummyTextModel(num_layers=2, hidden_dim=4)

        reference = _collect_text_activations_reference(
            texts, tokenizer, model, pool="mean", batch_size=None
        )

        model2 = DummyTextModel(num_layers=2, hidden_dim=4)
        optimized = collect_text_activations(
            texts, tokenizer=tokenizer, model=model2, pool="mean", batch_size=None
        )

        assert len(reference) == len(optimized)
        for ref, opt in zip(reference, optimized):
            np.testing.assert_allclose(ref, opt, rtol=1e-5, atol=1e-6)

    def test_equivalence_many_small_batches(self):
        """Test equivalence with batch_size=1 (many small batches)."""
        texts = ["text one", "text two", "text three", "text four", "text five"]
        tokenizer = DummyTokenizer()
        model = DummyTextModel(num_layers=3, hidden_dim=6)

        reference = _collect_text_activations_reference(
            texts, tokenizer, model, pool="mean", batch_size=1
        )

        model2 = DummyTextModel(num_layers=3, hidden_dim=6)
        optimized = collect_text_activations(
            texts, tokenizer=tokenizer, model=model2, pool="mean", batch_size=1
        )

        assert len(reference) == len(optimized)
        for ref, opt in zip(reference, optimized):
            np.testing.assert_allclose(ref, opt, rtol=1e-5, atol=1e-6)


class TestCollectTextActivationsBasic:
    """Basic functionality tests for collect_text_activations."""

    def test_output_shapes(self):
        """Test that output shapes are correct."""
        texts = ["hello", "world", "test"]
        tokenizer = DummyTokenizer()
        model = DummyTextModel(num_layers=4, hidden_dim=8)

        acts = collect_text_activations(
            texts, tokenizer=tokenizer, model=model, pool="mean", batch_size=2
        )

        assert len(acts) == 4  # num_layers
        for layer_acts in acts:
            assert layer_acts.shape == (3, 8)  # (num_texts, hidden_dim)

    def test_empty_input(self):
        """Test handling of empty input."""
        texts = []
        tokenizer = DummyTokenizer()
        model = DummyTextModel()

        acts = collect_text_activations(
            texts, tokenizer=tokenizer, model=model, pool="mean"
        )

        assert acts == []

    def test_specific_layers(self):
        """Test extracting specific layers only."""
        texts = ["hello", "world"]
        tokenizer = DummyTokenizer()
        model = DummyTextModel(num_layers=5, hidden_dim=4)

        acts = collect_text_activations(
            texts, tokenizer=tokenizer, model=model, pool="mean", layers=[0, 2, 4]
        )

        assert len(acts) == 3  # Only requested layers
        for layer_acts in acts:
            assert layer_acts.shape == (2, 4)

    def test_requires_model_or_name(self):
        with pytest.raises(ValueError, match="model_name is required"):
            collect_text_activations(["hello"], model=None, tokenizer=None)
