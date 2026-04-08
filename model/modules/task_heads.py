"""
任务预测头 (Phase 1 V3)

用于 Phase 3 下游任务微调：
- RegressionHead: CPP 渗透性预测
- ClassificationHead: CPP 分类等
"""

import torch
import torch.nn as nn
from typing import List


class RegressionHead(nn.Module):
    """
    回归任务头

    用于：CPP 渗透性预测、结合亲和力预测等
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dims: List[int] = [512, 256],
        dropout: float = 0.2,
        activation: str = "relu"
    ):
        super().__init__()

        self.input_dim = input_dim

        # Build MLP
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU() if activation == "relu" else nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Final regression layer
        layers.append(nn.Linear(prev_dim, 1))

        self.mlp = nn.Sequential(*layers)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters (重要修复)"""
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                # 最后一层小初始化，减小初始输出方差
                if module.out_features == 1:
                    nn.init.normal_(module.weight, mean=0, std=0.01)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                else:
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, input_dim] pooled features

        Returns:
            [batch] regression predictions
        """
        logits = self.mlp(x)  # [batch, 1]
        return logits.squeeze(-1)  # [batch]


class ClassificationHead(nn.Module):
    """
    分类任务头

    用于：CPP 二分类、多类别分类等
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dims: List[int] = [512, 256],
        num_classes: int = 2,
        dropout: float = 0.2,
        activation: str = "relu"
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes

        # Build MLP
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU() if activation == "relu" else nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Final classification layer
        layers.append(nn.Linear(prev_dim, num_classes))

        self.mlp = nn.Sequential(*layers)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters"""
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, input_dim] pooled features

        Returns:
            [batch, num_classes] logits (before softmax)
        """
        return self.mlp(x)


def test_task_heads():
    """Test task heads"""
    print("Testing Task Heads...")

    batch_size = 4
    input_dim = 768

    # Test RegressionHead
    print("\n1. Testing RegressionHead:")
    reg_head = RegressionHead(input_dim=768, hidden_dims=[512, 256])
    x = torch.randn(batch_size, input_dim)
    out = reg_head(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")
    print(f"   Output values: {out}")
    print(f"   Parameters: {sum(p.numel() for p in reg_head.parameters()):,}")

    # Test ClassificationHead
    print("\n2. Testing ClassificationHead:")
    cls_head = ClassificationHead(input_dim=768, hidden_dims=[512, 256], num_classes=2)
    out = cls_head(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")
    print(f"   Parameters: {sum(p.numel() for p in cls_head.parameters()):,}")


if __name__ == "__main__":
    test_task_heads()
