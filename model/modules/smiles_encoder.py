"""
增强版 SMILES Transformer 编码器 (Phase 1 V3)

关键改进：
- 支持 12/16 层深度（vs 旧版 6层）
- 支持 768 维嵌入（vs 旧版 512维）
- 优化显存使用
"""

import torch
import torch.nn as nn
from typing import Optional
import math


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)"""

    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len
        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None):
        """
        Args:
            x: Tensor of shape [batch, seq_len, dim]
        Returns:
            cos, sin: Rotary embeddings for position encoding
        """
        if seq_len is None:
            seq_len = x.shape[1]

        if seq_len != self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()[None, :, :]
            self._sin_cached = emb.sin()[None, :, :]

        return self._cos_cached[:, :seq_len, :], self._sin_cached[:, :seq_len, :]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Helper function for applying rotary embeddings"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply rotary position embedding to q and k"""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层（支持RoPE + 可选图偏置）"""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        # Multi-head attention
        self.norm1 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

        # Feed-forward network
        self.norm2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, d_model)
        self.ffn_dropout = nn.Dropout(dropout)

        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rope_emb: Optional[tuple] = None,
        attn_bias: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x: [batch, seq_len, d_model]
            attention_mask: [batch, seq_len]
            rope_emb: (cos, sin) from RoPE
            attn_bias: [batch, nhead, seq_len, seq_len] or [batch, 1, seq_len, seq_len]
        """
        # 1. Self-attention
        residual = x
        x = self.norm1(x)

        batch_size, seq_len, _ = x.shape

        # Project Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)

        # Apply RoPE
        if rope_emb is not None:
            cos, sin = rope_emb
            # cos, sin: [1, seq_len, d_model]
            # Expand to [batch, nhead, seq_len, head_dim]
            cos = cos.view(1, seq_len, self.nhead, self.head_dim).transpose(1, 2).expand(batch_size, -1, -1, -1)
            sin = sin.view(1, seq_len, self.nhead, self.head_dim).transpose(1, 2).expand(batch_size, -1, -1, -1)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_bias is not None:
            attn_scores = attn_scores + attn_bias

        if attention_mask is not None:
            # attention_mask: [batch, seq_len], 0 for padding
            mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        attn_output = self.out_proj(attn_output)

        x = residual + attn_output

        # 2. Feed-forward network
        residual = x
        x = self.norm2(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.ffn_dropout(x)
        x = self.fc2(x)
        x = self.ffn_dropout(x)

        x = residual + x

        return x


class EnhancedSMILESEncoder(nn.Module):
    """
    增强版 SMILES Transformer 编码器 (Phase 1 V3)

    设计目标：
    - 更深（12/16层）学习更复杂的分子表示
    - 更宽（768维）对齐下游任务需求
    - 集中参数在核心模块（~47M/63M）

    Architecture:
        - Embedding: vocab_size → 768
        - RoPE 位置编码
        - 12/16 层 Transformer
        - LayerNorm
        - CLS pooling
    """

    def __init__(
        self,
        vocab_size: int = 37,
        d_model: int = 768,
        nhead: int = 12,
        num_layers: int = 12,
        dim_feedforward: int = 3072,  # 4 * d_model
        dropout: float = 0.1,
        max_seq_len: int = 768,
        padding_idx: int = 0,
        use_gradient_checkpointing: bool = False,
        graph_attention: Optional[dict] = None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
        if padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[padding_idx].fill_(0)

        # RoPE 位置编码
        self.rope = RotaryPositionalEmbedding(d_model, max_seq_len)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu"
            )
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)

        # Graph attention bias (optional)
        self.graph_attention_cfg = graph_attention or {}
        self.graph_attention_enabled = bool(self.graph_attention_cfg.get("enabled", False))
        self.graph_attention_max_distance = int(self.graph_attention_cfg.get("max_distance", 8))
        self.graph_attention_bias_mode = str(self.graph_attention_cfg.get("bias_mode", "per_head"))
        self.graph_attention_bias_mode = self.graph_attention_bias_mode.lower()
        if self.graph_attention_enabled:
            if self.graph_attention_bias_mode not in {"per_head", "scalar"}:
                raise ValueError(f"Unsupported graph_attention.bias_mode: {self.graph_attention_bias_mode}")
            bias_dim = self.nhead if self.graph_attention_bias_mode == "per_head" else 1
            self.graph_bias_embed = nn.Embedding(self.graph_attention_max_distance + 2, bias_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        graph_distances: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            input_ids: [batch, seq_len] SMILES token IDs
            attention_mask: [batch, seq_len] 1 for valid tokens, 0 for padding

        Returns:
            output: [batch, seq_len, d_model] encoded SMILES features
        """
        batch_size, seq_len = input_ids.shape

        # 1. Embedding
        x = self.embedding(input_ids)  # [batch, seq_len, d_model]

        # 2. RoPE embeddings
        cos, sin = self.rope(x)
        rope_emb = (cos, sin)

        # 2.1 Graph attention bias (optional)
        attn_bias = None
        if self.graph_attention_enabled:
            if graph_distances is None:
                raise ValueError("graph_distances is required when graph_attention is enabled.")
            # Embedding expects int32/int64 indices; int16 will raise on GPU.
            graph_distances = graph_distances.to(dtype=torch.long)
            graph_distances = graph_distances.clamp_max(self.graph_attention_max_distance + 1)
            bias = self.graph_bias_embed(graph_distances)  # [B, L, L, bias_dim] or [B, L, L, 1]
            if self.graph_attention_bias_mode == "per_head":
                attn_bias = bias.permute(0, 3, 1, 2)  # [B, nhead, L, L]
            else:
                attn_bias = bias.permute(0, 3, 1, 2)  # [B, 1, L, L]

        # 3. Transformer layers
        for layer in self.layers:
            if self.use_gradient_checkpointing and self.training:
                # Use gradient checkpointing to save memory
                x = torch.utils.checkpoint.checkpoint(
                    layer,
                    x,
                    attention_mask,
                    rope_emb,
                    attn_bias,
                    use_reentrant=False
                )
            else:
                x = layer(x, attention_mask, rope_emb, attn_bias)

        # 4. Final norm
        x = self.final_norm(x)

        return x

    def get_num_params(self):
        """Return number of parameters"""
        return sum(p.numel() for p in self.parameters())


def test_smiles_encoder():
    """Test the Enhanced SMILES Encoder"""
    print("Testing EnhancedSMILESEncoder...")

    # Test configurations
    configs = [
        {"num_layers": 12, "d_model": 768, "nhead": 12},  # Standard
        {"num_layers": 16, "d_model": 768, "nhead": 12},  # Deep
    ]

    for i, config in enumerate(configs):
        print(f"\nConfig {i+1}: {config['num_layers']} layers, {config['d_model']} dim")

        encoder = EnhancedSMILESEncoder(
            vocab_size=37,
            **config,
            max_seq_len=768
        )

        # Forward pass
        batch_size = 4
        seq_len = 100
        input_ids = torch.randint(0, 37, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        output = encoder(input_ids, attention_mask)

        print(f"  Input shape: {input_ids.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Parameters: {encoder.get_num_params() / 1e6:.2f}M")
        print(f"  CLS features: {output[:, 0, :].shape}")


if __name__ == "__main__":
    test_smiles_encoder()
