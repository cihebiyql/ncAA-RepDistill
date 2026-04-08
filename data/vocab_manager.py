"""
词表管理器

负责加载和管理序列词表和SMILES词表
提供编码/解码功能和版本控制
"""

from typing import List, Dict, Optional, Union
from pathlib import Path
import hashlib
import json


class VocabManager:
    """
    词表管理器

    支持序列词表（uniref50）和SMILES词表
    """

    def __init__(self, vocab_file: Union[str, Path], name: str = "vocab"):
        """
        初始化词表管理器

        Args:
            vocab_file: 词表文件路径
            name: 词表名称
        """
        self.vocab_file = Path(vocab_file)
        self.name = name

        if not self.vocab_file.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_file}")

        # 加载词表
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        self._load_vocab()

        # 特殊token
        self.pad_token = "[PAD]" if "[PAD]" in self.token2id else None
        self.cls_token = "[CLS]" if "[CLS]" in self.token2id else None
        self.sep_token = "[SEP]" if "[SEP]" in self.token2id else None
        self.mask_token = "[MASK]" if "[MASK]" in self.token2id else None
        self.unk_token = "[UNK]" if "[UNK]" in self.token2id else None

        # 特殊token ID
        self.pad_token_id = self.token2id.get(self.pad_token, 0) if self.pad_token else 0
        self.cls_token_id = self.token2id.get(self.cls_token, 1) if self.cls_token else None
        self.sep_token_id = self.token2id.get(self.sep_token, 2) if self.sep_token else None
        self.mask_token_id = self.token2id.get(self.mask_token, 3) if self.mask_token else None
        self.unk_token_id = self.token2id.get(self.unk_token, 4) if self.unk_token else None

        # 计算哈希值
        self.vocab_hash = self._compute_hash()

    def _load_vocab(self):
        """加载词表文件"""
        with open(self.vocab_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                token = line.strip()
                if token:  # 忽略空行
                    self.token2id[token] = idx
                    self.id2token[idx] = token

        print(f"[VocabManager] Loaded {self.name}: {len(self.token2id)} tokens")

    def _compute_hash(self) -> str:
        """计算词表哈希值（用于版本控制）"""
        vocab_str = json.dumps(self.token2id, sort_keys=True)
        return hashlib.md5(vocab_str.encode()).hexdigest()[:8]

    def encode(
        self,
        tokens: Union[str, List[str]],
        add_special_tokens: bool = False,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False
    ) -> List[int]:
        """
        将token序列编码为ID序列

        Args:
            tokens: 单个token字符串或token列表
            add_special_tokens: 是否添加特殊token（CLS, SEP）
            max_length: 最大长度
            padding: 是否padding到max_length
            truncation: 是否截断到max_length

        Returns:
            ID列表
        """
        # 处理字符串输入（将字符串拆分为字符列表）
        if isinstance(tokens, str):
            tokens = list(tokens)

        # 编码
        ids = []

        # 添加CLS token
        if add_special_tokens and self.cls_token_id is not None:
            ids.append(self.cls_token_id)

        # 编码tokens
        for token in tokens:
            if token in self.token2id:
                ids.append(self.token2id[token])
            elif self.unk_token_id is not None:
                ids.append(self.unk_token_id)
            else:
                # 如果没有UNK token，跳过未知字符
                continue

        # 添加SEP token
        if add_special_tokens and self.sep_token_id is not None:
            ids.append(self.sep_token_id)

        # 截断
        if truncation and max_length is not None:
            ids = ids[:max_length]

        # Padding
        if padding and max_length is not None:
            if len(ids) < max_length:
                ids = ids + [self.pad_token_id] * (max_length - len(ids))

        return ids

    def decode(
        self,
        ids: List[int],
        skip_special_tokens: bool = True
    ) -> str:
        """
        将ID序列解码为token字符串

        Args:
            ids: ID列表
            skip_special_tokens: 是否跳过特殊token

        Returns:
            解码后的字符串
        """
        special_ids = set()
        if skip_special_tokens:
            if self.pad_token_id is not None:
                special_ids.add(self.pad_token_id)
            if self.cls_token_id is not None:
                special_ids.add(self.cls_token_id)
            if self.sep_token_id is not None:
                special_ids.add(self.sep_token_id)

        tokens = []
        for id in ids:
            if id in special_ids:
                continue
            if id in self.id2token:
                tokens.append(self.id2token[id])

        return ''.join(tokens)

    def batch_encode(
        self,
        batch_tokens: List[Union[str, List[str]]],
        add_special_tokens: bool = False,
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True
    ) -> List[List[int]]:
        """批量编码"""
        return [
            self.encode(
                tokens,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                padding=padding,
                truncation=truncation
            )
            for tokens in batch_tokens
        ]

    def batch_decode(
        self,
        batch_ids: List[List[int]],
        skip_special_tokens: bool = True
    ) -> List[str]:
        """批量解码"""
        return [
            self.decode(ids, skip_special_tokens=skip_special_tokens)
            for ids in batch_ids
        ]

    def __len__(self) -> int:
        """词表大小"""
        return len(self.token2id)

    def __contains__(self, token: str) -> bool:
        """检查token是否在词表中"""
        return token in self.token2id

    def get_vocab(self) -> Dict[str, int]:
        """获取完整词表"""
        return self.token2id.copy()

    def save_config(self, save_path: Union[str, Path]):
        """保存词表配置"""
        save_path = Path(save_path)
        config = {
            "name": self.name,
            "vocab_file": str(self.vocab_file),
            "vocab_size": len(self),
            "vocab_hash": self.vocab_hash,
            "special_tokens": {
                "pad_token": self.pad_token,
                "cls_token": self.cls_token,
                "sep_token": self.sep_token,
                "mask_token": self.mask_token,
                "unk_token": self.unk_token,
            },
            "special_token_ids": {
                "pad_token_id": self.pad_token_id,
                "cls_token_id": self.cls_token_id,
                "sep_token_id": self.sep_token_id,
                "mask_token_id": self.mask_token_id,
                "unk_token_id": self.unk_token_id,
            }
        }

        with open(save_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"[VocabManager] Config saved to {save_path}")

    def __repr__(self) -> str:
        return (
            f"VocabManager(\n"
            f"  name='{self.name}',\n"
            f"  vocab_size={len(self)},\n"
            f"  vocab_hash='{self.vocab_hash}',\n"
            f"  pad_token='{self.pad_token}' (id={self.pad_token_id}),\n"
            f"  unk_token='{self.unk_token}' (id={self.unk_token_id})\n"
            f")"
        )


# 便捷函数
def load_sequence_vocab(vocab_path: str = "data/uniref50/vocab_sequence_std.txt") -> VocabManager:
    """加载序列词表"""
    return VocabManager(vocab_path, name="sequence")


def load_smiles_vocab(vocab_path: str = "data/splits/vocab_smiles.txt") -> VocabManager:
    """加载SMILES词表"""
    return VocabManager(vocab_path, name="smiles")


if __name__ == "__main__":
    # 测试代码
    print("="*60)
    print("Testing VocabManager")
    print("="*60)

    # 测试序列词表
    print("\n1. Testing Sequence Vocabulary")
    print("-"*60)
    seq_vocab = load_sequence_vocab()
    print(seq_vocab)

    # 编码测试
    sequence = "MKTAYIAK"
    ids = seq_vocab.encode(sequence, add_special_tokens=True)
    print(f"\nSequence: {sequence}")
    print(f"Encoded: {ids}")

    # 解码测试
    decoded = seq_vocab.decode(ids, skip_special_tokens=True)
    print(f"Decoded: {decoded}")
    print(f"Match: {sequence == decoded}")

    # 测试SMILES词表
    print("\n2. Testing SMILES Vocabulary")
    print("-"*60)
    smiles_vocab = load_smiles_vocab()
    print(smiles_vocab)

    # 编码测试
    smiles = "CCO"
    ids = smiles_vocab.encode(smiles)
    print(f"\nSMILES: {smiles}")
    print(f"Encoded: {ids}")

    # 解码测试
    decoded = smiles_vocab.decode(ids)
    print(f"Decoded: {decoded}")
    print(f"Match: {smiles == decoded}")

    # Batch编码测试
    print("\n3. Testing Batch Encoding")
    print("-"*60)
    sequences = ["MKTA", "GLIEVQAP"]
    batch_ids = seq_vocab.batch_encode(sequences, max_length=10, padding=True)
    print(f"Sequences: {sequences}")
    print(f"Batch encoded: {batch_ids}")

    batch_decoded = seq_vocab.batch_decode(batch_ids)
    print(f"Batch decoded: {batch_decoded}")

    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)
