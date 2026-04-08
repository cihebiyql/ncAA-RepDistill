"""
数据加载器构建模块 (Phase 1 V3)
"""

import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import sys
import random

import numpy as np

# Import dataset and vocab manager
sys.path.insert(0, str(Path(__file__).parent))
from dataset import SingleInputDataset
from vocab_manager import VocabManager


def collate_and_trim_smiles_batch(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate a batch (default_collate) then trim padding on SMILES sequence dimensions.

    This reduces O(L^2) attention cost without changing outputs for valid tokens,
    because padding tokens are masked out by `smiles_attention_mask`.
    """
    collated: Dict[str, Any] = default_collate(batch)

    # View1 (main) trim
    mask = collated.get("smiles_attention_mask")
    if torch.is_tensor(mask) and mask.dim() == 2:
        max_len = int(mask.sum(dim=1).max().item())
        max_len = max(1, max_len)

        for key in ("smiles_input_ids", "smiles_attention_mask", "smiles_mlm_labels"):
            t = collated.get(key)
            if torch.is_tensor(t) and t.dim() == 2 and t.size(1) > max_len:
                collated[key] = t[:, :max_len]

        gd = collated.get("graph_distances")
        if torch.is_tensor(gd) and gd.dim() == 3 and gd.size(1) > max_len:
            collated["graph_distances"] = gd[:, :max_len, :max_len]

    # View2 trim (consistency)
    mask2 = collated.get("smiles_attention_mask_view2")
    if torch.is_tensor(mask2) and mask2.dim() == 2:
        max_len2 = int(mask2.sum(dim=1).max().item())
        max_len2 = max(1, max_len2)

        for key in ("smiles_input_ids_view2", "smiles_attention_mask_view2"):
            t = collated.get(key)
            if torch.is_tensor(t) and t.dim() == 2 and t.size(1) > max_len2:
                collated[key] = t[:, :max_len2]

        gd2 = collated.get("graph_distances_view2")
        if torch.is_tensor(gd2) and gd2.dim() == 3 and gd2.size(1) > max_len2:
            collated["graph_distances_view2"] = gd2[:, :max_len2, :max_len2]

    return collated


def _seed_worker(worker_id: int) -> None:
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return
    seed = int(worker_info.seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def build_dataloaders(config: Dict) -> Tuple[Dict[str, DataLoader], VocabManager, VocabManager]:
    """
    构建训练和验证数据加载器

    Args:
        config: 配置字典，包含：
            - data.paths.vocab_smiles: SMILES 词表路径
            - data.paths.train_data: 训练数据路径
            - data.paths.val_data: 验证数据路径
            - data.batch_size: 批次大小
            - data.num_workers: 数据加载线程数
            - data.augmentation: 数据增强配置

    Returns:
        dataloaders: {'train': train_loader, 'val': val_loader}
        smiles_vocab: SMILES 词表管理器
        None: 占位符（兼容旧版接口）
    """
    # 1. 加载词表
    smiles_vocab_path = config['data']['paths']['vocab_smiles']
    smiles_vocab = VocabManager(smiles_vocab_path)
    print(f"[DataLoader] SMILES vocab size: {len(smiles_vocab)}")

    # 2. 数据配置
    batch_size = config['data']['batch_size']
    num_workers = config['data'].get('num_workers', 4)
    pin_memory = config['data'].get('pin_memory', True)

    # SMILES 最大长度
    smiles_max_length = config['student']['smiles_encoder'].get('max_length', 768)

    # 数据增强配置
    augmentation_config = config['data'].get('augmentation', {})
    random_smiles_config = augmentation_config.get('random_smiles', {})
    smiles_mask_config = augmentation_config.get('smiles_mask', {})
    consistency_config = config.get("data", {}).get("consistency") or (config.get("loss") or {}).get("consistency") or {}
    graph_attention_config = (config.get("student") or {}).get("graph_attention") or {}
    val_consistency_config = None
    if isinstance(consistency_config, dict) and bool(consistency_config.get("enabled", False)):
        # Use deterministic SMILES variants for validation to reduce checkpoint selection noise.
        val_consistency_config = dict(consistency_config)
        val_consistency_config.setdefault("deterministic", True)

    # MLM masking 配置
    apply_smiles_masking = smiles_mask_config.get('enabled', False)
    mask_prob = smiles_mask_config.get('mask_prob', 0.15)

    # 3. 可选：任务监督（P4: 单任务；P5: 多任务轻监督）
    task_supervision_cfg = (config.get("loss") or {}).get("task_supervision") or {}
    task_supervision_enabled = bool(task_supervision_cfg.get("enabled", False))
    label_column = None
    normalization_config = None
    task_columns: Optional[List[str]] = None
    task_supervision_config: Optional[Dict[str, Any]] = None
    if task_supervision_enabled:
        label_source = str(task_supervision_cfg.get("label_source") or "").strip().lower()
        if label_source in {"morgan_fp", "morgan", "ecfp", "fingerprint", "fp"}:
            task_supervision_config = task_supervision_cfg
        else:
            tasks = task_supervision_cfg.get("tasks")
            if isinstance(tasks, (list, tuple)) and tasks:
                task_columns = [str(t) for t in tasks]
            elif isinstance(tasks, str) and tasks.strip():
                task_columns = [tasks.strip()]
            else:
                label_column = task_supervision_cfg.get("label_column") or "Permeability"
                normalization_config = task_supervision_cfg.get("normalization")

    # 3.1 可选：第二个 proxy（用于双 proxy：例如 rdkit6 + fp）
    task_supervision_aux_cfg = (config.get("loss") or {}).get("task_supervision_aux") or {}
    task_supervision_aux_enabled = bool(task_supervision_aux_cfg.get("enabled", False))
    task_supervision_aux_config: Optional[Dict[str, Any]] = None
    if task_supervision_aux_enabled:
        aux_label_source = str(task_supervision_aux_cfg.get("label_source") or "").strip().lower()
        if aux_label_source in {"morgan_fp", "morgan", "ecfp", "fingerprint", "fp"}:
            task_supervision_aux_config = task_supervision_aux_cfg
        else:
            # Allow aux supervision from CSV columns (e.g., rdkit2d descriptors).
            task_supervision_aux_config = task_supervision_aux_cfg

    # 4. 创建训练集
    train_data_path = config['data']['paths']['train_data']
    train_dataset = SingleInputDataset(
        data_path=train_data_path,
        smiles_vocab_manager=smiles_vocab,
        smiles_max_length=smiles_max_length,
        apply_smiles_masking=apply_smiles_masking,
        mask_prob=mask_prob,
        random_smiles_config=random_smiles_config,
        consistency_config=consistency_config,
        graph_attention_config=graph_attention_config,
        label_column=label_column,
        normalization_config=normalization_config,
        task_columns=task_columns,
        task_supervision_config=task_supervision_config,
        task_supervision_aux_config=task_supervision_aux_config,
        split_name='train'
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # 避免最后一个 batch 太小
        persistent_workers=bool(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        collate_fn=collate_and_trim_smiles_batch,
    )

    print(f"[DataLoader] Train: {len(train_dataset)} samples, {len(train_loader)} batches")

    # 5. 创建验证集
    dataloaders = {'train': train_loader}

    val_data_path = config['data']['paths'].get('val_data')
    if val_data_path:
        val_dataset = SingleInputDataset(
            data_path=val_data_path,
            smiles_vocab_manager=smiles_vocab,
            smiles_max_length=smiles_max_length,
            apply_smiles_masking=False,  # 验证时不 mask
            mask_prob=0.0,
            random_smiles_config=None,  # 验证时不增强
            consistency_config=val_consistency_config,
            graph_attention_config=graph_attention_config,
            label_column=label_column,
            normalization_config=normalization_config,
            task_columns=task_columns,
            task_supervision_config=task_supervision_config,
            task_supervision_aux_config=task_supervision_aux_config,
            split_name='val'
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=bool(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
            worker_init_fn=_seed_worker if num_workers > 0 else None,
            collate_fn=collate_and_trim_smiles_batch,
        )

        dataloaders['val'] = val_loader
        print(f"[DataLoader] Val: {len(val_dataset)} samples, {len(val_loader)} batches")

    return dataloaders, smiles_vocab, None


def test_dataloader():
    """Test dataloader building"""
    print("Testing build_dataloaders...")

    # Dummy config
    config = {
        'data': {
            'paths': {
                'vocab_smiles': 'data/ncaa_adaptation_v2/vocab_smiles.txt',
                'train_data': 'data/uniref_lt30_c30/train_sequences.csv',
                'val_data': 'data/uniref_lt30_c30/val_sequences.csv'
            },
            'batch_size': 4,
            'num_workers': 0,
            'pin_memory': False,
            'augmentation': {
                'random_smiles': {
                    'enabled': True,
                    'mix_ratio': 0.25
                },
                'smiles_mask': {
                    'enabled': True,
                    'mask_prob': 0.15
                }
            }
        },
        'student': {
            'smiles_encoder': {
                'max_length': 100
            }
        }
    }

    try:
        dataloaders, smiles_vocab, _ = build_dataloaders(config)

        print(f"\nDataloaders created:")
        for split, loader in dataloaders.items():
            print(f"  {split}: {len(loader)} batches")

        # Test one batch
        batch = next(iter(dataloaders['train']))
        print(f"\nSample batch:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            elif isinstance(value, list):
                print(f"  {key}: list of {len(value)} items")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_dataloader()
