import numpy as np
import pytest
import torch
import torch.nn.functional as F

from faceapp.calibrate import expected_calibration_error, fit_temperature, nll
from faceapp.evaluate import classification_metrics
from faceapp.losses import inverse_frequency_weights, mixup_cutmix, soft_cross_entropy
from faceapp.models import FaceClassifier


def test_soft_ce_matches_torch_for_onehot():
    torch.manual_seed(0)
    logits, y = torch.randn(8, 4), torch.randint(0, 4, (8,))
    onehot = F.one_hot(y, 4).float()
    w = torch.tensor([1.0, 2.0, 0.5, 1.5])
    assert soft_cross_entropy(logits, onehot) == pytest.approx(F.cross_entropy(logits, y).item(), abs=1e-5)
    assert soft_cross_entropy(logits, onehot, 0.1) == pytest.approx(
        F.cross_entropy(logits, y, label_smoothing=0.1).item(), abs=1e-5)
    assert soft_cross_entropy(logits, onehot, 0.0, w) == pytest.approx(
        F.cross_entropy(logits, y, weight=w).item(), abs=1e-5)


def test_inverse_frequency_weights():
    w = inverse_frequency_weights([10, 30, 0, 20])
    assert w.tolist() == pytest.approx([60 / 40, 60 / 120, 1.0, 60 / 80])


@pytest.mark.parametrize("cutmix_prob", [0.0, 1.0])
def test_mixup_cutmix_targets_are_distributions(cutmix_prob):
    x = torch.randn(6, 3, 32, 32)
    t = F.one_hot(torch.arange(6) % 4, 4).float()
    x2, t2 = mixup_cutmix(x, t, alpha=0.4, cutmix_prob=cutmix_prob, rng=np.random.RandomState(0))
    assert x2.shape == x.shape
    assert torch.allclose(t2.sum(1), torch.ones(6))


def test_metrics_macro_f1():
    m = classification_metrics([0, 0, 1, 1, 2, 3], [0, 1, 1, 1, 2, 0])
    assert m["accuracy"] == pytest.approx(4 / 6)
    f1 = [0.5, 0.8, 1.0, 0.0]  # dog: P=1/2 R=1/2, cat: P=2/3 R=1, rabbit: 1, fox: 0
    assert m["macro_f1"] == pytest.approx(sum(f1) / 4)


def test_temperature_scaling_reduces_overconfidence():
    torch.manual_seed(0)
    y = torch.randint(0, 4, (400,))
    logits = torch.randn(400, 4)
    logits[torch.arange(400), y] += 1.0
    logits *= 5  # 과신 모델 흉내
    t = fit_temperature(logits, y)
    assert t > 1.5
    assert nll(logits, y, t) < nll(logits, y)
    assert expected_calibration_error((logits / t).softmax(1), y) < expected_calibration_error(logits.softmax(1), y)


@pytest.mark.parametrize("name", ["resnet18", "simple_cnn"])
def test_model_forward_shapes_and_freeze(name):
    m = FaceClassifier(name, pretrained=False).eval()
    logits, emb = m.forward_with_features(torch.randn(2, 3, 96, 96))
    assert logits.shape == (2, 4) and emb.shape == (2, m.feature_dim)
    m.set_backbone_trainable(False)
    assert not any(p.requires_grad for p in m.backbone.parameters())
    assert all(p.requires_grad for p in m.head.parameters())
