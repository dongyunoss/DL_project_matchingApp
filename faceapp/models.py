"""분류 모델: ImageNet 사전학습 백본(transfer learning) + 4클래스 head, 그리고 비교용 SimpleCNN."""
import torch
import torch.nn as nn
import torchvision

from . import NUM_CLASSES

PRETRAINED_BACKBONES = ("resnet18", "resnet34", "efficientnet_b0", "mobilenet_v3_small")
MODEL_NAMES = (*PRETRAINED_BACKBONES, "simple_cnn")


class SimpleCNN(nn.Module):
    """처음부터 학습하는 CNN 베이스라인 (transfer learning 효과 비교용)."""

    def __init__(self, channels=(32, 64, 128, 256)):
        super().__init__()
        layers, c_in = [], 3
        for c in channels:
            layers.append(nn.Sequential(
                nn.Conv2d(c_in, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ))
            c_in = c
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = c_in

    def forward(self, x):
        return torch.flatten(self.pool(self.features(x)), 1)


def _build_backbone(name: str, pretrained: bool):
    """(backbone, feature_dim, grad_cam_target_layer) 반환. backbone 출력은 pooled feature 벡터."""
    if name == "simple_cnn":
        m = SimpleCNN()
        return m, m.out_dim, m.features[-1]
    if name not in PRETRAINED_BACKBONES:
        raise ValueError(f"지원하지 않는 모델: {name} (가능: {MODEL_NAMES})")
    m = torchvision.models.get_model(name, weights="DEFAULT" if pretrained else None)
    if name.startswith("resnet"):
        dim, target = m.fc.in_features, m.layer4
        m.fc = nn.Identity()
    elif name == "efficientnet_b0":
        dim, target = m.classifier[1].in_features, m.features[-1]
        m.classifier = nn.Identity()
    else:  # mobilenet_v3_small
        dim, target = m.classifier[0].in_features, m.features[-1]
        m.classifier = nn.Identity()
    return m, dim, target


class FaceClassifier(nn.Module):
    """backbone → (embedding) → dropout → linear(4).

    - forward(x): logits
    - forward_with_features(x): (logits, embedding)  # 임베딩은 매칭 확장/시각화용
    - target_layer: Grad-CAM 대상 conv 블록
    """

    def __init__(self, name="resnet18", pretrained=True, dropout=0.3, num_classes=NUM_CLASSES):
        super().__init__()
        self.name = name
        self.backbone, self.feature_dim, self.target_layer = _build_backbone(name, pretrained)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.feature_dim, num_classes))

    def forward_with_features(self, x):
        feats = self.backbone(x)
        return self.head(feats), feats

    def forward(self, x):
        return self.forward_with_features(x)[0]

    def set_backbone_trainable(self, trainable: bool) -> None:
        """fine-tuning 단계 제어. freeze 시 BN 통계도 고정하려면 train()에서 backbone.eval() 호출."""
        for p in self.backbone.parameters():
            p.requires_grad = trainable

    def param_groups(self, lr: float, backbone_lr_factor: float, weight_decay: float):
        """Discriminative learning rate: 사전학습 backbone은 작은 lr, 새 head는 큰 lr."""
        factor = 1.0 if self.name == "simple_cnn" else backbone_lr_factor
        return [
            {"params": self.backbone.parameters(), "lr": lr * factor, "weight_decay": weight_decay},
            {"params": self.head.parameters(), "lr": lr, "weight_decay": weight_decay},
        ]


def build_model(cfg: dict, pretrained: bool | None = None) -> FaceClassifier:
    m = cfg["model"]
    return FaceClassifier(
        name=m["name"],
        pretrained=m.get("pretrained", True) if pretrained is None else pretrained,
        dropout=m.get("dropout", 0.3),
    )


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def save_checkpoint(path, model: FaceClassifier, cfg: dict, **extra) -> None:
    torch.save({
        "model_state": model.state_dict(),
        "model_name": model.name,
        "dropout": cfg["model"].get("dropout", 0.3),
        "config": cfg,
        "temperature": 1.0,
        **extra,
    }, path)


def load_checkpoint(path, device="cpu") -> tuple[FaceClassifier, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = FaceClassifier(ckpt["model_name"], pretrained=False, dropout=ckpt.get("dropout", 0.3))
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), ckpt
