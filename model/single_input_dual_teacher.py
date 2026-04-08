"""
单输入双教师模型 (Phase 1 V3)

核心设计理念：
- 输入：SMILES only（不再需要 sequence 输入）
- 架构：增强版 SMILES encoder + 双投影头
- 目标：从 SMILES 学习蛋白质语义（via ESM2）+ 分子结构（via ChemBERTa）

优势：
- Phase 2/3 推理时架构完全一致（只处理 SMILES）
- 参数集中在核心模块（~48M/63M，97% 有效利用）
- SMILES encoder 已包含蛋白质知识，无需重新学习
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from .modules.smiles_encoder import EnhancedSMILESEncoder
from .modules.projection import ProteinProjection, MolecularProjection
from .modules.task_heads import RegressionHead, ClassificationHead
from .mlm_heads import SMILESMLMHead


class SingleInputDualTeacher(nn.Module):
    """
    单输入双教师知识蒸馏模型（Phase 1 V3 专用架构）

    输入：SMILES only
    教师：ESM2（640维，蛋白质语义）+ ChemBERTa（768维，分子结构）
    核心：SMILES encoder 学习包含蛋白质语义的分子表示

    适用场景：
    - Phase 1: 标准AA的SMILES → 学习双重知识
    - Phase 2: ncAA的SMILES → 强化分子知识
    - Phase 3: ncAA的SMILES → 任务预测
    """

    def __init__(
        self,
        smiles_vocab_size: int = 37,
        smiles_embed_dim: int = 768,
        smiles_num_layers: int = 12,
        smiles_num_heads: int = 12,
        smiles_max_length: int = 768,
        esm2_dim: int = 640,
        chemberta_dim: int = 768,
        graph_attention: Optional[dict] = None,
        contrastive_proj_cfg: Optional[dict] = None,
        fusion_cfg: Optional[dict] = None,
        task_type: Optional[str] = None,
        task_num_classes: Optional[int] = None,
        use_smiles_mlm: bool = True,
        dropout: float = 0.1,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()

        self.smiles_vocab_size = smiles_vocab_size
        self.smiles_embed_dim = smiles_embed_dim
        self.smiles_num_layers = smiles_num_layers
        self.esm2_dim = esm2_dim
        self.chemberta_dim = chemberta_dim
        self.task_type = task_type
        self.use_smiles_mlm = use_smiles_mlm
        self.graph_attention = graph_attention or {}
        self.contrastive_proj_cfg = contrastive_proj_cfg or {}
        self.fusion_cfg = fusion_cfg or {}

        # 1. 核心模块：增强版 SMILES 编码器
        self.smiles_encoder = EnhancedSMILESEncoder(
            vocab_size=smiles_vocab_size,
            d_model=smiles_embed_dim,
            nhead=smiles_num_heads,
            num_layers=smiles_num_layers,
            dim_feedforward=smiles_embed_dim * 4,
            dropout=dropout,
            max_seq_len=smiles_max_length,
            use_gradient_checkpointing=use_gradient_checkpointing,
            graph_attention=self.graph_attention,
        )

        # 2. 双投影头
        self.protein_proj = ProteinProjection(
            input_dim=smiles_embed_dim,
            hidden_dims=[smiles_embed_dim, esm2_dim],
            output_dim=esm2_dim,
            dropout=dropout
        )

        self.molecular_proj = MolecularProjection(
            input_dim=smiles_embed_dim,
            output_dim=chemberta_dim,
            use_identity=True,
            dropout=dropout
        )

        # 2.2 可选：预训练内融合（Residual distill / "stacking in pretrain"）
        self.fusion_enabled = bool(self.fusion_cfg.get("enabled", False))
        if self.fusion_enabled:
            mode = str(self.fusion_cfg.get("mode", "residual_distill")).strip().lower()
            if mode != "residual_distill":
                raise ValueError(f"Unsupported fusion.mode: {mode}")

            hs_source = str(self.fusion_cfg.get("hs_source", "molecular_proj")).strip().lower()
            proj_dim = int(self.fusion_cfg.get("proj_dim", chemberta_dim))

            if hs_source == "molecular_proj":
                if int(proj_dim) != int(chemberta_dim):
                    raise ValueError(
                        f"fusion.proj_dim must equal chemberta_dim when hs_source=molecular_proj "
                        f"(proj_dim={proj_dim}, chemberta_dim={chemberta_dim})"
                    )
                self.fusion_smiles_to_target = None
            elif hs_source in {"smiles_repr", "smiles"}:
                self.fusion_smiles_to_target = nn.Linear(smiles_embed_dim, proj_dim)
            else:
                raise ValueError(f"Unsupported fusion.hs_source: {hs_source}")

            self.fusion_prot_to_target = nn.Linear(esm2_dim, proj_dim)
            nn.init.zeros_(self.fusion_prot_to_target.weight)
            if self.fusion_prot_to_target.bias is not None:
                nn.init.zeros_(self.fusion_prot_to_target.bias)

        # 2.1 可选：对比式投影头（解耦 InfoNCE 使用）
        if self.contrastive_proj_cfg.get("enabled", False):
            proj_dim = int(self.contrastive_proj_cfg.get("output_dim", smiles_embed_dim))
            hidden_dim = int(self.contrastive_proj_cfg.get("hidden_dim", smiles_embed_dim))
            proj_dropout = float(self.contrastive_proj_cfg.get("dropout", dropout))
            self.contrastive_proj = nn.Sequential(
                nn.Linear(smiles_embed_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(proj_dropout),
                nn.Linear(hidden_dim, proj_dim),
            )

        # 3. SMILES MLM 头（自监督）
        if use_smiles_mlm:
            self.smiles_mlm_head = SMILESMLMHead(
                hidden_size=smiles_embed_dim,
                vocab_size=smiles_vocab_size
            )

        # 4. 任务头（Phase 3 用）
        if task_type == "regression":
            self.task_head = RegressionHead(
                input_dim=smiles_embed_dim,
                hidden_dims=[512, 256],
                dropout=dropout
            )
        elif task_type == "classification":
            self.task_head = ClassificationHead(
                input_dim=smiles_embed_dim,
                hidden_dims=[512, 256],
                num_classes=task_num_classes or 2,
                dropout=dropout
            )

    def forward(
        self,
        smiles_input_ids: torch.Tensor,
        smiles_attention_mask: Optional[torch.Tensor] = None,
        graph_distances: Optional[torch.Tensor] = None,
        return_all: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播（单输入模式）

        Args:
            smiles_input_ids: [batch, smiles_len] SMILES token IDs
            smiles_attention_mask: [batch, smiles_len] attention mask
            return_all: 是否返回所有中间结果（MLM logits等）

        Returns:
            outputs: dict 包含：
                - 'smiles_repr': [batch, 768] SMILES pooled features (CLS)
                - 'smiles_mean_pool': [batch, 768] SMILES masked mean pooling
                - 'protein_proj': [batch, 640] 用于对齐 ESM2
                - 'molecular_proj': [batch, 768] 用于对齐 ChemBERTa
                - 'smiles_mlm_logits': [batch, seq_len, vocab_size] (如果 return_all=True)
                - 'task_pred': [batch] or [batch, num_classes] (如果有任务头)
        """
        # 1. SMILES 编码
        smiles_output = self.smiles_encoder(
            smiles_input_ids,
            smiles_attention_mask,
            graph_distances=graph_distances,
        )
        # smiles_output: [batch, smiles_len, 768]

        # 2. CLS Pooling
        smiles_pooled = smiles_output[:, 0, :]  # [batch, 768]

        # 2.1 Masked mean pooling (keep consistent with Phase3 extractor)
        if smiles_attention_mask is None:
            smiles_mean_pool = smiles_output.mean(dim=1)
        else:
            mask = smiles_attention_mask.to(dtype=smiles_output.dtype).unsqueeze(-1)
            denom = mask.sum(dim=1).clamp_min(1.0)
            smiles_mean_pool = (smiles_output * mask).sum(dim=1) / denom

        # 3. 双投影
        protein_proj = self.protein_proj(smiles_pooled)    # [batch, 640]
        molecular_proj = self.molecular_proj(smiles_pooled)  # [batch, 768]

        outputs = {
            'smiles_repr': smiles_pooled,
            'smiles_mean_pool': smiles_mean_pool,
            'protein_proj': protein_proj,
            'molecular_proj': molecular_proj,
        }

        if self.fusion_enabled:
            hs_source = str(self.fusion_cfg.get("hs_source", "molecular_proj")).strip().lower()
            if hs_source == "molecular_proj":
                fusion_hs = molecular_proj
            else:
                if self.fusion_smiles_to_target is None:
                    raise RuntimeError("fusion_smiles_to_target is not initialized.")
                fusion_hs = self.fusion_smiles_to_target(smiles_pooled)

            fusion_hp = self.fusion_prot_to_target(protein_proj)
            fusion_hf = fusion_hs + fusion_hp
            outputs.update(
                {
                    "fusion_hs": fusion_hs,
                    "fusion_hp": fusion_hp,
                    "fusion_hf": fusion_hf,
                }
            )

        if hasattr(self, "contrastive_proj"):
            outputs["contrastive_proj"] = self.contrastive_proj(smiles_pooled)

        # 4. MLM logits（训练时或 return_all=True）
        if return_all and self.use_smiles_mlm:
            outputs['smiles_mlm_logits'] = self.smiles_mlm_head(smiles_output)

        # 5. 任务预测（Phase 3 用）
        if hasattr(self, 'task_head'):
            outputs['task_pred'] = self.task_head(smiles_pooled)

        return outputs

    def get_num_params(self, trainable_only: bool = False):
        """
        获取模型参数量

        Args:
            trainable_only: 是否只统计可训练参数

        Returns:
            dict: 各模块参数量
        """
        def count_params(module):
            if trainable_only:
                return sum(p.numel() for p in module.parameters() if p.requires_grad)
            else:
                return sum(p.numel() for p in module.parameters())

        params = {
            'smiles_encoder': count_params(self.smiles_encoder),
            'protein_proj': count_params(self.protein_proj),
            'molecular_proj': count_params(self.molecular_proj),
        }

        if self.fusion_enabled:
            if getattr(self, "fusion_smiles_to_target", None) is not None:
                params["fusion_smiles_to_target"] = count_params(self.fusion_smiles_to_target)
            params["fusion_prot_to_target"] = count_params(self.fusion_prot_to_target)

        if hasattr(self, 'smiles_mlm_head'):
            params['smiles_mlm_head'] = count_params(self.smiles_mlm_head)

        if hasattr(self, 'task_head'):
            params['task_head'] = count_params(self.task_head)

        params['total'] = sum(params.values())

        return params

    def print_model_info(self):
        """打印模型信息"""
        params = self.get_num_params(trainable_only=False)

        print("\n" + "="*70)
        print("SingleInputDualTeacher Model Information")
        print("="*70)
        print(f"SMILES Encoder: {self.smiles_num_layers} layers, {self.smiles_embed_dim} dim")
        print(f"  - Parameters: {params['smiles_encoder'] / 1e6:.2f}M")
        print(f"\nProtein Projection ({self.smiles_embed_dim} → {self.esm2_dim}):")
        print(f"  - Parameters: {params['protein_proj'] / 1e6:.2f}M")
        print(f"\nMolecular Projection ({self.smiles_embed_dim} → {self.chemberta_dim}):")
        print(f"  - Parameters: {params['molecular_proj'] / 1e6:.2f}M")

        if self.fusion_enabled:
            proj_dim = int(self.fusion_cfg.get("proj_dim", self.chemberta_dim))
            print(f"\nFusion (residual_distill):")
            if "fusion_smiles_to_target" in params:
                print(f"  - Smiles→Target ({self.smiles_embed_dim} → {proj_dim}): {params['fusion_smiles_to_target'] / 1e6:.2f}M")
            print(f"  - Prot→Target ({self.esm2_dim} → {proj_dim}): {params['fusion_prot_to_target'] / 1e6:.2f}M")

        if 'smiles_mlm_head' in params:
            print(f"\nSMILES MLM Head:")
            print(f"  - Parameters: {params['smiles_mlm_head'] / 1e6:.2f}M")

        if 'task_head' in params:
            print(f"\nTask Head ({self.task_type}):")
            print(f"  - Parameters: {params['task_head'] / 1e6:.2f}M")

        print(f"\nTotal Parameters: {params['total'] / 1e6:.2f}M")
        print("="*70 + "\n")


def test_single_input_dual_teacher():
    """Test the model"""
    print("Testing SingleInputDualTeacher...")

    configs = [
        {"num_layers": 12, "name": "Standard (12-layer)"},
        {"num_layers": 16, "name": "Deep (16-layer)"},
    ]

    for config in configs:
        print(f"\n{'='*70}")
        print(f"Config: {config['name']}")
        print(f"{'='*70}")

        model = SingleInputDualTeacher(
            smiles_vocab_size=37,
            smiles_embed_dim=768,
            smiles_num_layers=config['num_layers'],
            smiles_num_heads=12,
            smiles_max_length=768,
            esm2_dim=640,
            chemberta_dim=768,
            use_smiles_mlm=True
        )

        model.print_model_info()

        # Forward pass test
        batch_size = 4
        seq_len = 100
        smiles_input_ids = torch.randint(0, 37, (batch_size, seq_len))
        smiles_attention_mask = torch.ones(batch_size, seq_len)

        with torch.no_grad():
            outputs = model(smiles_input_ids, smiles_attention_mask, return_all=True)

        print("\nForward Pass Results:")
        print(f"  SMILES repr: {outputs['smiles_repr'].shape}")
        print(f"  Protein proj: {outputs['protein_proj'].shape}")
        print(f"  Molecular proj: {outputs['molecular_proj'].shape}")
        print(f"  SMILES MLM logits: {outputs['smiles_mlm_logits'].shape}")


if __name__ == "__main__":
    test_single_input_dual_teacher()
