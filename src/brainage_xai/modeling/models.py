"""Two predefined ImageNet ResNet18 regressors with a shared slice backbone."""

import random

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)


class BrainAgeModel(nn.Module):
    def __init__(self, kind, mean_age, pretrained_state=None):
        super().__init__()
        if kind not in ("A", "B"):
            raise ValueError("Only predefined models A and B are allowed")
        self.kind = kind
        self.backbone = resnet18(weights=None)
        if pretrained_state is not None:
            self.backbone.load_state_dict(pretrained_state, strict=True)
        self.backbone.fc = nn.Identity()
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        for p in self.backbone.layer4.parameters():
            p.requires_grad_(True)
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                for p in module.parameters():
                    p.requires_grad_(False)
        self.head = nn.Sequential(
            nn.Linear(512 if kind == "A" else 1536, 128), nn.ReLU(), nn.Linear(128, 1)
        )
        nn.init.normal_(self.head[-1].weight, std=0.001)
        nn.init.zeros_(self.head[-1].bias)
        self.register_buffer("mean_age", torch.tensor(float(mean_age)))

    def train(self, mode=True):
        super().train(mode)
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        return self

    def prefix(self, x):
        b = self.backbone
        x = b.maxpool(b.relu(b.bn1(b.conv1(x))))
        return b.layer3(b.layer2(b.layer1(x)))

    def from_maps(self, maps):
        n, slices = maps.shape[:2]
        expected = 1 if self.kind == "A" else 15
        if slices != expected:
            raise ValueError("Wrong predefined number of input slices")
        features = self.backbone.layer4(maps.flatten(0, 1))
        features = self.backbone.avgpool(features).flatten(1).reshape(n, slices, 512)
        if self.kind == "B":
            features = features.reshape(n, 3, 5, 512).mean(2).flatten(1)
        else:
            features = features[:, 0]
        return self.head(features).squeeze(-1) + self.mean_age

    def forward(self, x):
        n, slices = x.shape[:2]
        maps = self.prefix(x.flatten(0, 1))
        return self.from_maps(maps.reshape(n, slices, *maps.shape[1:]))


def save_checkpoint(path, model, metadata):
    torch.save(
        {
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "metadata": metadata,
        },
        path,
    )


def load_checkpoint(path, device="cpu"):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    meta = saved["metadata"]
    model = BrainAgeModel(meta["kind"], meta["mean_age"])
    model.load_state_dict(saved["model"], strict=True)
    return model.to(device).eval(), meta


def grad_cam(model, inputs):
    """Positive Grad-CAM per original slice, normalized independently for display."""
    activations = []

    def capture(module, args, result):
        result.retain_grad()
        activations.append(result)

    model.eval()
    handle = model.backbone.layer4.register_forward_hook(capture)
    try:
        model.zero_grad(set_to_none=True)
        prediction = model(inputs.requires_grad_(True))
        prediction.sum().backward()
        a = activations[0]
        weights = a.grad.mean((2, 3), keepdim=True)
        cam = (weights * a).sum(1).relu()
        cam = torch.nn.functional.interpolate(
            cam[:, None], inputs.shape[-2:], mode="bilinear", align_corners=False
        )[:, 0]
        maxima = cam.flatten(1).max(1).values[:, None, None]
        cam = cam / maxima.clamp_min(1e-12)
        return cam.detach().cpu().numpy(), prediction.detach().cpu().numpy()
    finally:
        handle.remove()
