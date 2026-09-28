"""The temporal classifier, attention pooling, and checkpointing."""

import pytest
import torch

from models.attention_layer import AttentionPooling, masked_mean_pool
from models.bert_temporal_model import BertTemporalClassifier
from models.model_utils import (
    count_parameters,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
    set_seed,
)


@pytest.fixture(scope="module")
def model():
    set_seed(0)
    return BertTemporalClassifier.tiny_for_testing(num_temporal_features=8)


@pytest.fixture
def batch():
    return {
        "input_ids": torch.randint(0, 100, (4, 12)),
        "attention_mask": torch.ones(4, 12, dtype=torch.long),
        "temporal_features": torch.randn(4, 8),
        "labels": torch.tensor([0, 1, 0, 1]),
    }


class TestAttentionPooling:
    def test_output_shape(self):
        pooled, weights = AttentionPooling(16)(torch.randn(3, 7, 16))
        assert pooled.shape == (3, 16) and weights.shape == (3, 7)

    def test_weights_sum_to_one(self):
        _, weights = AttentionPooling(16)(torch.randn(3, 7, 16))
        assert torch.allclose(weights.sum(-1), torch.ones(3), atol=1e-5)

    def test_padding_receives_zero_weight(self):
        mask = torch.ones(2, 6, dtype=torch.long)
        mask[0, 3:] = 0
        _, weights = AttentionPooling(16)(torch.randn(2, 6, 16), mask)
        assert torch.allclose(weights[0, 3:], torch.zeros(3), atol=1e-6)
        assert torch.allclose(weights[0, :3].sum(), torch.tensor(1.0), atol=1e-5)

    def test_fully_masked_row_does_not_produce_nan(self):
        # softmax over an all-masked row is the classic NaN source here.
        pooled, weights = AttentionPooling(16)(
            torch.randn(1, 5, 16), torch.zeros(1, 5, dtype=torch.long)
        )
        assert not torch.isnan(pooled).any() and not torch.isnan(weights).any()

    def test_pooled_output_ignores_padding_content(self):
        # Changing the padded positions must not change the pooled vector.
        layer = AttentionPooling(16)
        states = torch.randn(1, 6, 16)
        mask = torch.tensor([[1, 1, 1, 0, 0, 0]])
        first, _ = layer(states, mask)
        states[:, 3:] = 99.0
        second, _ = layer(states, mask)
        assert torch.allclose(first, second, atol=1e-5)

    def test_rejects_nonpositive_hidden_size(self):
        with pytest.raises(ValueError, match="hidden_size"):
            AttentionPooling(0)

    def test_masked_mean_pool_ignores_padding(self):
        states = torch.ones(1, 4, 3)
        states[:, 2:] = 100.0
        mask = torch.tensor([[1, 1, 0, 0]])
        assert torch.allclose(masked_mean_pool(states, mask), torch.ones(1, 3))


class TestBertTemporalClassifier:
    def test_forward_shapes(self, model, batch):
        out = model(**batch)
        assert out.logits.shape == (4, 2)
        assert out.loss is not None and out.loss.ndim == 0

    def test_loss_is_omitted_without_labels(self, model, batch):
        batch.pop("labels")
        assert model(**batch).loss is None

    def test_gradients_reach_both_branches(self, batch):
        set_seed(1)
        fresh = BertTemporalClassifier.tiny_for_testing(num_temporal_features=8)
        fresh(**batch).loss.backward()
        assert fresh.classifier.weight.grad is not None
        assert fresh.temporal_mlp[0].weight.grad is not None
        assert fresh.temporal_mlp[0].weight.grad.abs().sum() > 0

    def test_temporal_features_change_the_output(self, model, batch):
        model.eval()
        with torch.no_grad():
            a = model(**{**batch, "temporal_features": torch.zeros(4, 8)}).logits
            b = model(**{**batch, "temporal_features": torch.ones(4, 8)}).logits
        assert not torch.allclose(a, b)

    def test_text_only_ablation_needs_no_temporal_features(self, batch):
        ablation = BertTemporalClassifier.tiny_for_testing(num_temporal_features=0)
        batch.pop("temporal_features")
        assert ablation(**batch).logits.shape == (4, 2)
        assert ablation.temporal_mlp is None

    def test_missing_temporal_features_raises_a_clear_error(self, model, batch):
        batch.pop("temporal_features")
        with pytest.raises(ValueError, match="temporal_features=None"):
            model(**batch)

    def test_wrong_temporal_width_raises(self, model, batch):
        batch["temporal_features"] = torch.randn(4, 3)
        with pytest.raises(ValueError, match="Expected 8 temporal features"):
            model(**batch)

    def test_attention_weights_returned_on_request(self, model, batch):
        assert model(**batch, return_attention=True).attention_weights.shape == (4, 12)
        assert model(**batch).attention_weights is None

    def test_mean_pooling_variant_runs(self, batch):
        pooled = BertTemporalClassifier.tiny_for_testing(
            num_temporal_features=8, use_attention_pooling=False
        )
        assert pooled.pooler is None
        assert pooled(**batch).logits.shape == (4, 2)

    def test_freeze_bert_leaves_the_head_trainable(self, batch):
        frozen = BertTemporalClassifier.tiny_for_testing(num_temporal_features=8,
                                                         freeze_bert=True)
        assert all(not p.requires_grad for p in frozen.bert.parameters())
        assert count_parameters(frozen, trainable_only=True) > 0
        frozen(**batch).loss.backward()
        assert frozen.classifier.weight.grad is not None

    def test_unfreeze_restores_gradients(self):
        frozen = BertTemporalClassifier.tiny_for_testing(freeze_bert=True)
        frozen.unfreeze_encoder()
        assert all(p.requires_grad for p in frozen.bert.parameters())

    def test_padding_does_not_change_predictions(self, model):
        # Same content, different amounts of padding -> same logits.
        model.eval()
        ids = torch.randint(1, 100, (1, 5))
        padded = torch.cat([ids, torch.zeros(1, 7, dtype=torch.long)], dim=1)
        mask = torch.cat([torch.ones(1, 5), torch.zeros(1, 7)], dim=1).long()
        features = torch.randn(1, 8)
        with torch.no_grad():
            short = model(ids, torch.ones(1, 5, dtype=torch.long),
                          temporal_features=features).logits
            long = model(padded, mask, temporal_features=features).logits
        assert torch.allclose(short, long, atol=1e-4)

    def test_from_config(self):
        from training.config import Config
        cfg = Config(num_labels=3, temporal_hidden_size=4)
        built = BertTemporalClassifier.from_config(
            cfg, encoder=BertTemporalClassifier.tiny_for_testing().bert
        )
        assert built.num_labels == 3


class TestModelUtils:
    def test_set_seed_is_reproducible(self):
        set_seed(123)
        first = torch.randn(5)
        set_seed(123)
        second = torch.randn(5)
        assert torch.equal(first, second)

    def test_resolve_device_cpu(self):
        assert resolve_device("cpu").type == "cpu"

    def test_resolve_device_falls_back_instead_of_raising(self):
        # Asking for cuda on a CPU-only box must warn and continue.
        assert resolve_device("cuda").type in ("cuda", "cpu")

    def test_checkpoint_roundtrip_preserves_weights(self, tmp_path, model):
        path = save_checkpoint(model, tmp_path / "ck.pth", epoch=2,
                               metrics={"accuracy": 0.7})
        restored = BertTemporalClassifier.tiny_for_testing(num_temporal_features=8)
        restored, meta = load_checkpoint(restored, path)
        assert meta["epoch"] == 2 and meta["metrics"]["accuracy"] == 0.7
        for a, b in zip(model.state_dict().values(), restored.state_dict().values()):
            assert torch.allclose(a, b)

    def test_loads_a_bare_state_dict(self, tmp_path, model):
        # The original training loop saved model.state_dict() directly; those
        # checkpoints must still load.
        path = tmp_path / "legacy.pth"
        torch.save(model.state_dict(), path)
        restored = BertTemporalClassifier.tiny_for_testing(num_temporal_features=8)
        restored, meta = load_checkpoint(restored, path)
        assert meta == {}

    def test_missing_checkpoint_raises(self, tmp_path, model):
        with pytest.raises(FileNotFoundError):
            load_checkpoint(model, tmp_path / "nope.pth")

    def test_count_parameters(self, model):
        assert count_parameters(model, trainable_only=False) > 0
