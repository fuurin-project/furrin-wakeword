import pytest
import torch

from oww_distill.classifier import load_wake_classifier
from oww_distill.conv_attention import ConvAttentionHead


def test_tiny_conv_attention_shape_and_size() -> None:
    model = ConvAttentionHead(layer_dim=16, n_blocks=1, n_heads=4)
    output = model(torch.zeros(3, 16, 96))

    assert output.shape == (3,)
    assert model.deploy_parameters == 7569


def test_attention_heads_must_divide_layer_width() -> None:
    with pytest.raises(ValueError, match="divide"):
        ConvAttentionHead(layer_dim=18, n_heads=4)


def test_torch_checkpoint_runtime_matches_model(tmp_path) -> None:
    model = ConvAttentionHead(layer_dim=16, n_blocks=1, n_heads=4).eval()
    checkpoint = tmp_path / "head.pt"
    torch.save(
        {
            "architecture": {
                "type": "conv_attention",
                "n_timesteps": 16,
                "embedding_dim": 96,
                "layer_dim": 16,
                "n_blocks": 1,
                "n_heads": 4,
            },
            "model_state": model.state_dict(),
            "feature_mean": torch.zeros(96),
            "feature_scale": torch.ones(96),
            "threshold": 0.4,
            "name": "hey_pico",
        },
        checkpoint,
    )
    embeddings = torch.randn(16, 96)
    with torch.inference_mode():
        expected = float(torch.sigmoid(model(embeddings[None]))[0])

    runtime = load_wake_classifier(checkpoint)

    assert runtime.threshold == pytest.approx(0.4)
    assert runtime.score(embeddings.numpy()) == pytest.approx(expected, abs=1e-6)
