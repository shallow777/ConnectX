"""AlphaZero 策略-价值网络 (policy-value network) 与损失函数.

结构与 AlphaZero 论文一致 (按 ConnectX 规模缩小): 卷积 stem + 若干残差块,
然后分出两个 head:
- policy head: 输出每列的落子 logits;
- value head:  输出当前局面的胜率估计, tanh 压到 [-1, 1]。
损失 = policy 交叉熵 + value MSE + L2 正则。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class NetworkConfig:
    rows: int = 6
    columns: int = 7
    channels: int = 64
    residual_blocks: int = 3
    l2_weight: float = 1e-4


class ResidualBlock(nn.Module):
    """标准残差块: conv-bn-relu-conv-bn + skip connection."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class AlphaZeroNet(nn.Module):
    """残差策略-价值网络; forward 返回 (policy logits, value)."""

    def __init__(
        self,
        rows: int = 6,
        columns: int = 7,
        channels: int = 64,
        residual_blocks: int = 3,
    ) -> None:
        super().__init__()
        self.rows = rows
        self.columns = columns
        self.channels = channels
        self.residual_blocks = residual_blocks

        self.stem = nn.Sequential(
            nn.Conv2d(2, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(residual_blocks)])

        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * rows * columns, columns),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(rows * columns, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.blocks(x)
        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return logits, value

    @property
    def config(self) -> NetworkConfig:
        return NetworkConfig(
            rows=self.rows,
            columns=self.columns,
            channels=self.channels,
            residual_blocks=self.residual_blocks,
        )


def masked_policy_loss(logits: torch.Tensor, target_policy: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    """交叉熵 loss, target 是 MCTS 访问次数归一化后的分布; 非法列掩掉再 softmax."""
    masked_logits = logits.masked_fill(~action_mask.bool(), -1e9)
    log_probs = F.log_softmax(masked_logits, dim=-1)
    return -(target_policy * log_probs).sum(dim=-1).mean()


def alphazero_loss(
    logits: torch.Tensor,
    values: torch.Tensor,
    target_policy: torch.Tensor,
    target_value: torch.Tensor,
    action_mask: torch.Tensor,
    model: nn.Module,
    l2_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """总损失 = policy CE + value MSE + l2_weight * ||theta||^2 (论文公式)."""
    policy_loss = masked_policy_loss(logits, target_policy, action_mask)
    value_loss = F.mse_loss(values, target_value)
    l2_loss = torch.zeros((), device=values.device)
    for parameter in model.parameters():
        l2_loss = l2_loss + parameter.pow(2).sum()
    loss = policy_loss + value_loss + l2_weight * l2_loss
    return loss, {
        "loss": float(loss.detach().cpu()),
        "policy_loss": float(policy_loss.detach().cpu()),
        "value_loss": float(value_loss.detach().cpu()),
        "l2_loss": float(l2_loss.detach().cpu()),
    }


@torch.no_grad()
def predict_policy_value(
    model: AlphaZeroNet,
    encoded_state: np.ndarray,
    action_mask: np.ndarray,
    device: str | torch.device = "cpu",
) -> tuple[np.ndarray, float]:
    """单局面推理: 返回 (掩码后归一化的 policy 分布, value 标量), 供 MCTS 调用."""
    model.eval()
    tensor = torch.as_tensor(encoded_state, dtype=torch.float32, device=device).unsqueeze(0)
    logits, value = model(tensor)
    logits_np = logits.squeeze(0).detach().cpu().numpy()
    mask = np.asarray(action_mask, dtype=bool)
    logits_np[~mask] = -1e9
    logits_np = logits_np - np.max(logits_np[mask])
    exp_logits = np.zeros_like(logits_np, dtype=np.float64)
    exp_logits[mask] = np.exp(logits_np[mask])
    policy = exp_logits / exp_logits.sum()
    return policy.astype(np.float32), float(value.item())


def save_checkpoint(
    path: str | Path,
    model: AlphaZeroNet,
    optimizer: torch.optim.Optimizer | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "network_config": model.config.__dict__,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> tuple[AlphaZeroNet, dict[str, Any]]:
    payload = torch.load(path, map_location=map_location)
    config = dict(payload.get("network_config", {}))
    config.pop("l2_weight", None)
    model = AlphaZeroNet(**config)
    model.load_state_dict(payload["model_state_dict"])
    model.to(map_location)
    model.eval()
    return model, payload
