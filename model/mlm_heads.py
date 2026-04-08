"""
SMILES MLM 预测头 (Phase 1 V3)

用于自监督预训练阶段的 SMILES Masked Language Modeling
"""

import torch
import torch.nn as nn


class SMILESMLMHead(nn.Module):
    """
    SMILES Masked Language Model 预测头

    Architecture:
        - Transform: Linear(hidden_size, hidden_size) + GELU + LayerNorm
        - Decoder: Linear(hidden_size, vocab_size)
    """

    def __init__(
        self,
        hidden_size: int = 768,
        vocab_size: int = 37
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # Transform layer
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Decoder layer
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=True)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters"""
        nn.init.normal_(self.dense.weight, std=0.02)
        nn.init.zeros_(self.dense.bias)
        nn.init.normal_(self.decoder.weight, std=0.02)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size] encoder outputs

        Returns:
            [batch, seq_len, vocab_size] MLM logits
        """
        # Transform
        x = self.dense(hidden_states)
        x = self.activation(x)
        x = self.layer_norm(x)

        # Decode to vocab
        logits = self.decoder(x)

        return logits


def mask_smiles_tokens(
    input_ids: torch.Tensor,
    mask_prob: float = 0.15,
    vocab_size: int = 37,
    mask_token_id: int = 3,  # [MASK]
    pad_token_id: int = 0,
    special_token_ids: list = [0, 1, 2, 3]  # [PAD], [CLS], [SEP], [MASK]
):
    """
    BERT-style masking for SMILES tokens

    Strategy:
    - 15% of tokens are selected for masking
    - Of those selected:
        - 80% are replaced with [MASK]
        - 10% are replaced with random token
        - 10% are kept unchanged

    Args:
        input_ids: [batch, seq_len] input token IDs
        mask_prob: probability of masking (default: 0.15)
        vocab_size: SMILES vocabulary size
        mask_token_id: ID for [MASK] token
        pad_token_id: ID for [PAD] token
        special_token_ids: IDs for special tokens (not to be masked)

    Returns:
        masked_input_ids: [batch, seq_len] with some tokens masked
        labels: [batch, seq_len] with -100 for non-masked, original token for masked
    """
    batch_size, seq_len = input_ids.shape

    # Clone input
    masked_input_ids = input_ids.clone()
    labels = torch.full_like(input_ids, -100)  # -100 = ignore in loss

    for i in range(batch_size):
        # Create mask probability matrix
        prob_matrix = torch.full((seq_len,), mask_prob)

        # Don't mask special tokens and padding
        for special_id in special_token_ids:
            prob_matrix[input_ids[i] == special_id] = 0

        # Select tokens to mask
        masked_indices = torch.bernoulli(prob_matrix).bool()

        # Store original tokens as labels
        labels[i, masked_indices] = input_ids[i, masked_indices]

        # Apply masking strategy
        # 80% of the time, replace with [MASK]
        indices_replaced = torch.bernoulli(torch.full((seq_len,), 0.8)).bool() & masked_indices
        masked_input_ids[i, indices_replaced] = mask_token_id

        # 10% of the time, replace with random token
        indices_random = torch.bernoulli(torch.full((seq_len,), 0.5)).bool() & masked_indices & ~indices_replaced
        random_tokens = torch.randint(low=len(special_token_ids), high=vocab_size, size=(seq_len,))
        masked_input_ids[i, indices_random] = random_tokens[indices_random]

        # Remaining 10% are kept unchanged

    return masked_input_ids, labels


def test_mlm_head():
    """Test SMILES MLM Head"""
    print("Testing SMILESMLMHead...")

    batch_size = 4
    seq_len = 100
    hidden_size = 768
    vocab_size = 37

    # Test MLM head
    mlm_head = SMILESMLMHead(hidden_size=hidden_size, vocab_size=vocab_size)

    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    logits = mlm_head(hidden_states)

    print(f"Input shape: {hidden_states.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Parameters: {sum(p.numel() for p in mlm_head.parameters()):,}")

    # Test masking
    print("\nTesting mask_smiles_tokens:")
    input_ids = torch.randint(4, vocab_size, (batch_size, seq_len))  # Skip special tokens
    masked_ids, labels = mask_smiles_tokens(input_ids, mask_prob=0.15, vocab_size=vocab_size)

    print(f"Original IDs: {input_ids[0, :20]}")
    print(f"Masked IDs:   {masked_ids[0, :20]}")
    print(f"Labels:       {labels[0, :20]}")
    print(f"Masked positions: {(labels[0] != -100).sum().item()} / {seq_len} ({(labels[0] != -100).sum().item() / seq_len * 100:.1f}%)")


if __name__ == "__main__":
    test_mlm_head()
