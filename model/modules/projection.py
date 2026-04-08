"""
双投影头模块 (Phase 1 V3)

设计：
- ProteinProjection: 768 → 640 (对齐 ESM2)
- MolecularProjection: 768 → 768 (对齐 ChemBERTa)
"""

import torch
import torch.nn as nn
from typing import List


class ProteinProjection(nn.Module):
    """
    蛋白质投影头

    目的：将 SMILES 特征投影到 ESM2 的 640 维空间
    核心：学习 SMILES 中的蛋白质语义（肽段序列信息）
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dims: List[int] = [768, 640],
        output_dim: int = 640,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Build MLP
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims[:-1]):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU() if activation == "gelu" else nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Final projection
        layers.append(nn.Linear(prev_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, 768] SMILES pooled features

        Returns:
            [batch, 640] projected features for ESM2 alignment
        """
        return self.mlp(x)


class MolecularProjection(nn.Module):
    """
    分子投影头

    目的：将 SMILES 特征投影到 ChemBERTa 的 768 维空间
    核心：保持分子结构信息
    """

    def __init__(
        self,
        input_dim: int = 768,
        output_dim: int = 768,
        use_identity: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_identity = use_identity

        if use_identity and input_dim == output_dim:
            # 维度相同，使用轻量投影（LayerNorm + 残差）
            self.projection = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Dropout(dropout)
            )
        else:
            # 维度不同，使用 MLP
            self.projection = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, 768] SMILES pooled features

        Returns:
            [batch, 768] projected features for ChemBERTa alignment
        """
        if self.use_identity and self.input_dim == self.output_dim:
            # Identity + normalization
            return self.projection(x)
        else:
            return self.projection(x)


class DualProjectionHead(nn.Module):
    """
    双投影头容器

    包含：
    - Protein projection (768 → 640)
    - Molecular projection (768 → 768)
    """

    def __init__(
        self,
        input_dim: int = 768,
        esm2_dim: int = 640,
        chemberta_dim: int = 768,
        dropout: float = 0.1
    ):
        super().__init__()

        self.protein_proj = ProteinProjection(
            input_dim=input_dim,
            hidden_dims=[768, 640],
            output_dim=esm2_dim,
            dropout=dropout
        )

        self.molecular_proj = MolecularProjection(
            input_dim=input_dim,
            output_dim=chemberta_dim,
            use_identity=True,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [batch, 768] SMILES pooled features

        Returns:
            dict: {
                'protein_proj': [batch, 640],
                'molecular_proj': [batch, 768]
            }
        """
        return {
            'protein_proj': self.protein_proj(x),
            'molecular_proj': self.molecular_proj(x)
        }


def test_projections():
    """Test projection heads"""
    print("Testing Projection Heads...")

    batch_size = 4
    input_dim = 768

    # Test individual projections
    print("\n1. Testing ProteinProjection (768 → 640):")
    protein_proj = ProteinProjection(input_dim=768, output_dim=640)
    x = torch.randn(batch_size, input_dim)
    out = protein_proj(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")
    print(f"   Parameters: {sum(p.numel() for p in protein_proj.parameters()):,}")

    print("\n2. Testing MolecularProjection (768 → 768):")
    molecular_proj = MolecularProjection(input_dim=768, output_dim=768, use_identity=True)
    out = molecular_proj(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")
    print(f"   Parameters: {sum(p.numel() for p in molecular_proj.parameters()):,}")

    print("\n3. Testing DualProjectionHead:")
    dual_proj = DualProjectionHead(input_dim=768, esm2_dim=640, chemberta_dim=768)
    outputs = dual_proj(x)
    print(f"   Protein proj shape: {outputs['protein_proj'].shape}")
    print(f"   Molecular proj shape: {outputs['molecular_proj'].shape}")
    print(f"   Total parameters: {sum(p.numel() for p in dual_proj.parameters()):,}")


if __name__ == "__main__":
    test_projections()
