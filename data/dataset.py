"""
单输入数据集 (Phase 1 V3)

关键设计：
- 只编码 SMILES 列（不再编码 sequence）
- 保留 sequence_id 用于从缓存加载 ESM2 特征
- 支持 Random SMILES 增强
- 支持 SMILES MLM masking
- 支持 Phase2 proxy：Morgan/ECFP 指纹多标签监督（BCE）
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys
import random
import re
import os
import json
import hashlib
from collections import OrderedDict
import numpy as np

try:
    from rdkit import Chem  # type: ignore
    from rdkit.Chem import rdFingerprintGenerator  # type: ignore
    from rdkit import DataStructs  # type: ignore
except Exception:  # noqa: BLE001
    Chem = None
    rdFingerprintGenerator = None
    DataStructs = None

try:
    import fcntl  # type: ignore
except Exception:  # noqa: BLE001
    fcntl = None

# Import vocab manager
sys.path.insert(0, str(Path(__file__).parent))
from vocab_manager import VocabManager

# Import MLM masking
sys.path.insert(0, str(Path(__file__).parent.parent))
from model.mlm_heads import mask_smiles_tokens


class RandomSMILESProvider:
    """Random SMILES 增强提供器（RDKit doRandom=True）"""

    def __init__(
        self,
        mix_ratio: float = 0.25,
        num_variants: int = 2,
        max_cache_size: int = 100_000,
        max_tries_per_variant: int = 20,
    ):
        self.mix_ratio = mix_ratio
        self.num_variants = num_variants
        self.max_cache_size = max_cache_size
        self.max_tries_per_variant = max_tries_per_variant
        self._cache: "OrderedDict[str, List[str]]" = OrderedDict()

    def sample(self, sequence_id: str, canonical_smiles: str) -> str:
        """
        采样 SMILES 变体

        - 以 mix_ratio 的概率返回 random SMILES（等价 SMILES 写法）
        - 其余返回 canonical SMILES
        """
        if self.mix_ratio <= 0 or self.num_variants <= 0:
            return canonical_smiles

        if Chem is None:
            return canonical_smiles

        if random.random() >= self.mix_ratio:
            return canonical_smiles

        variants = self._get_variants(canonical_smiles)
        if not variants:
            return canonical_smiles
        return random.choice(variants)

    def sample_variant(self, canonical_smiles: str, deterministic: bool = False) -> str:
        """始终尝试返回一个 SMILES 变体（若失败则回退原始）。

        Args:
            canonical_smiles: Canonical SMILES string.
            deterministic: When True, choose a stable variant per canonical_smiles
                (useful for validation to reduce metric noise).
        """
        if self.num_variants <= 0:
            return canonical_smiles
        if Chem is None:
            return canonical_smiles
        variants = self._get_variants(canonical_smiles)
        if not variants:
            return canonical_smiles
        if deterministic:
            h = hashlib.md5(canonical_smiles.encode("utf-8")).hexdigest()
            idx = int(h, 16) % len(variants)
            return variants[idx]
        return random.choice(variants)

    def _get_variants(self, canonical_smiles: str) -> List[str]:
        if canonical_smiles in self._cache:
            variants = self._cache.pop(canonical_smiles)
            self._cache[canonical_smiles] = variants
            return variants

        variants = self._generate_variants(canonical_smiles)
        self._cache[canonical_smiles] = variants
        if len(self._cache) > self.max_cache_size:
            self._cache.popitem(last=False)
        return variants

    def _generate_variants(self, canonical_smiles: str) -> List[str]:
        mol = Chem.MolFromSmiles(canonical_smiles)  # type: ignore[attr-defined]
        if mol is None:
            return []

        variants = set()
        max_tries = max(1, int(self.num_variants) * int(self.max_tries_per_variant))
        for _ in range(max_tries):
            s = Chem.MolToSmiles(mol, doRandom=True, isomericSmiles=True)  # type: ignore[attr-defined]
            if not s:
                continue
            if s == canonical_smiles:
                continue
            variants.add(s)
            if len(variants) >= int(self.num_variants):
                break

        return sorted(variants)


class SingleInputDataset(Dataset):
    """
    单输入数据集（只加载 SMILES）

    适用场景：
    - Phase 1: 标准AA SMILES → 学习双重知识
    - Phase 2: ncAA SMILES → 强化分子知识
    - Phase 3: ncAA SMILES → 任务预测
    """

    def __init__(
        self,
        data_path: str,
        smiles_vocab_manager: VocabManager,
        smiles_max_length: int = 768,
        apply_smiles_masking: bool = False,
        mask_prob: float = 0.15,
        random_smiles_config: Optional[Dict] = None,
        consistency_config: Optional[Dict] = None,
        graph_attention_config: Optional[Dict] = None,
        label_column: Optional[str] = None,
        normalization_config: Optional[Dict] = None,
        task_columns: Optional[List[str]] = None,
        task_supervision_config: Optional[Dict[str, Any]] = None,
        task_supervision_aux_config: Optional[Dict[str, Any]] = None,
        split_name: str = "train"
    ):
        """
        Args:
            data_path: CSV 数据文件路径
            smiles_vocab_manager: SMILES 词表管理器
            smiles_max_length: SMILES 最大长度
            apply_smiles_masking: 是否应用 MLM masking
            mask_prob: Masking 概率
            random_smiles_config: Random SMILES 配置
            label_column: 标签列名（Phase 3 用）
            normalization_config: 标签归一化配置（Phase 3 用）
                例如：{'enabled': True, 'type': 'zscore', 'mean': -5.87, 'std': 1.08}
                或：  {'enabled': True, 'type': 'minmax', 'min': 0, 'max': 10}
            split_name: 数据集拆分名称（用于缓存加载）
        """
        self.data_path = Path(data_path)
        self.smiles_vocab = smiles_vocab_manager
        self.smiles_max_length = smiles_max_length
        self.apply_smiles_masking = apply_smiles_masking
        self.mask_prob = mask_prob
        self.label_column = label_column
        self.task_columns = list(task_columns) if task_columns else None
        self.task_supervision_config = task_supervision_config or {}
        self.task_supervision_aux_config = task_supervision_aux_config or {}
        self.task_columns_aux: Optional[List[str]] = None
        self.label_column_aux: Optional[str] = None
        self.split_name = split_name
        self.consistency_cfg = consistency_config or {}
        self.consistency_enabled = bool(self.consistency_cfg.get("enabled", False))
        self.graph_attention_cfg = graph_attention_config or {}
        self.graph_attention_enabled = bool(self.graph_attention_cfg.get("enabled", False))
        self.graph_attention_max_distance = int(self.graph_attention_cfg.get("max_distance", 8))
        self.graph_cache_size = int(self.graph_attention_cfg.get("cache_size", 50_000))

        # 标签归一化配置
        self.normalize_label = False
        self.norm_type = None
        self.norm_params = {}
        if normalization_config and normalization_config.get('enabled'):
            self.normalize_label = True
            self.norm_type = normalization_config.get('type', 'zscore')
            if self.norm_type == 'zscore':
                self.norm_params['mean'] = normalization_config.get('mean', 0.0)
                self.norm_params['std'] = normalization_config.get('std', 1.0)
                print(f"[Dataset] Label normalization enabled: zscore (mean={self.norm_params['mean']:.4f}, std={self.norm_params['std']:.4f})")
            elif self.norm_type == 'minmax':
                self.norm_params['min'] = normalization_config.get('min', 0.0)
                self.norm_params['max'] = normalization_config.get('max', 1.0)
                print(f"[Dataset] Label normalization enabled: minmax (min={self.norm_params['min']:.4f}, max={self.norm_params['max']:.4f})")

        # 加载数据
        self.df = pd.read_csv(data_path)
        print(f"[Dataset] Loaded {len(self.df)} samples from {data_path}")

        # 统一解析 SMILES 列名（避免 __getitem__ 每次判断）
        self.smiles_col: str = ""
        for col in ("smiles", "SMILES", "canonical_smiles"):
            if col in self.df.columns:
                self.smiles_col = col
                break
        if not self.smiles_col:
            raise KeyError("无法在数据集中找到 SMILES 列（期望 'smiles' / 'SMILES' / 'canonical_smiles'）")

        # 多任务监督列检查（仅检查列是否存在；缺失值由 mask 处理）
        if self.task_columns:
            missing_cols = [c for c in self.task_columns if c not in self.df.columns]
            if missing_cols:
                raise KeyError(f"Task columns missing in dataset {data_path}: {missing_cols}")
        if self.task_columns_aux:
            missing_cols_aux = [c for c in self.task_columns_aux if c not in self.df.columns]
            if missing_cols_aux:
                raise KeyError(f"Aux task columns missing in dataset {data_path}: {missing_cols_aux}")

        # Phase2 proxy: Morgan/ECFP 指纹多标签（无需 CSV 标签列）
        self.fingerprint_cfg = self._parse_fingerprint_cfg(self.task_supervision_config)
        if self.fingerprint_cfg is not None and self.task_columns is not None:
            raise ValueError("Fingerprint supervision cannot be enabled together with task_columns.")

        # Phase2 proxy (aux): Morgan/ECFP 指纹多标签（用于双 proxy：rdkit6 + fp）
        self.fingerprint_aux_cfg = self._parse_fingerprint_cfg(self.task_supervision_aux_config)
        if self.fingerprint_aux_cfg is None and self.task_supervision_aux_config:
            tasks_aux = self.task_supervision_aux_config.get("tasks")
            if isinstance(tasks_aux, (list, tuple)) and tasks_aux:
                self.task_columns_aux = [str(t) for t in tasks_aux]
            elif isinstance(tasks_aux, str) and tasks_aux.strip():
                self.task_columns_aux = [tasks_aux.strip()]
            else:
                self.label_column_aux = (
                    self.task_supervision_aux_config.get("label_column") or "Permeability"
                )

        self._fp_labels: Optional[np.ndarray] = None
        self._fp_valid: Optional[np.ndarray] = None
        self._fp_bit_mask: Optional[np.ndarray] = None
        self._fp_pos_weight: Optional[np.ndarray] = None
        self._fp_bit_mask_tensor: Optional[torch.Tensor] = None
        self._fp_zero_labels_tensor: Optional[torch.Tensor] = None
        self._fp_zero_mask_tensor: Optional[torch.Tensor] = None
        if self.fingerprint_cfg is not None:
            self._init_fingerprint_cache()
            n_bits = int(self.fingerprint_cfg["n_bits"])
            if self._fp_bit_mask is not None:
                self._fp_bit_mask_tensor = torch.from_numpy(self._fp_bit_mask).to(dtype=torch.float32)
            self._fp_zero_labels_tensor = torch.zeros((n_bits,), dtype=torch.uint8)
            self._fp_zero_mask_tensor = torch.zeros((n_bits,), dtype=torch.float32)

        self._fp_labels_aux: Optional[np.ndarray] = None
        self._fp_valid_aux: Optional[np.ndarray] = None
        self._fp_bit_mask_aux: Optional[np.ndarray] = None
        self._fp_pos_weight_aux: Optional[np.ndarray] = None
        self._fp_bit_mask_tensor_aux: Optional[torch.Tensor] = None
        self._fp_zero_labels_tensor_aux: Optional[torch.Tensor] = None
        self._fp_zero_mask_tensor_aux: Optional[torch.Tensor] = None
        if self.fingerprint_aux_cfg is not None:
            self._init_fingerprint_aux_cache()
            n_bits_aux = int(self.fingerprint_aux_cfg["n_bits"])
            if self._fp_bit_mask_aux is not None:
                self._fp_bit_mask_tensor_aux = torch.from_numpy(self._fp_bit_mask_aux).to(dtype=torch.float32)
            self._fp_zero_labels_tensor_aux = torch.zeros((n_bits_aux,), dtype=torch.uint8)
            self._fp_zero_mask_tensor_aux = torch.zeros((n_bits_aux,), dtype=torch.float32)

        # Random SMILES 增强
        if random_smiles_config and random_smiles_config.get('enabled'):
            self.random_smiles_provider = RandomSMILESProvider(
                mix_ratio=random_smiles_config.get('mix_ratio', 0.25),
                num_variants=random_smiles_config.get('num_variants', 2),
                max_cache_size=random_smiles_config.get('max_cache_size', 100_000),
                max_tries_per_variant=random_smiles_config.get('max_tries_per_variant', 20),
            )
            print(f"[Dataset] Random SMILES enabled (mix_ratio={random_smiles_config.get('mix_ratio', 0.25)})")
        else:
            self.random_smiles_provider = None

        # Consistency SMILES provider（不依赖 mix_ratio）
        if self.consistency_enabled:
            self.consistency_smiles_provider = RandomSMILESProvider(
                mix_ratio=1.0,
                num_variants=int(self.consistency_cfg.get("num_variants", 2)),
                max_cache_size=int(self.consistency_cfg.get("max_cache_size", 50_000)),
                max_tries_per_variant=int(self.consistency_cfg.get("max_tries_per_variant", 20)),
            )
        else:
            self.consistency_smiles_provider = None

        # Graph attention distance cache
        if self.graph_attention_enabled:
            if Chem is None:
                raise RuntimeError("Graph attention requires RDKit, but rdkit is not available.")
            self._graph_dist_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()

    @staticmethod
    def _parse_fingerprint_cfg(cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        label_source = str(cfg.get("label_source") or "").strip().lower()
        if label_source not in {"morgan_fp", "morgan", "ecfp", "fingerprint", "fp"}:
            return None

        fp_cfg = cfg.get("fingerprint") or {}
        if not isinstance(fp_cfg, dict):
            fp_cfg = {}

        radius = int(fp_cfg.get("radius", 2))
        n_bits = int(fp_cfg.get("n_bits", 1024))
        use_features = bool(fp_cfg.get("use_features", False))
        use_chirality = bool(fp_cfg.get("use_chirality", True))
        cache_dir = str(fp_cfg.get("cache_dir") or "data_cache/fingerprints")
        overwrite_cache = bool(fp_cfg.get("overwrite_cache", False))
        drop_constant_bits = bool(fp_cfg.get("drop_constant_bits", True))

        bce_cfg = cfg.get("bce") or {}
        if not isinstance(bce_cfg, dict):
            bce_cfg = {}
        pos_weight_clip_max = float(bce_cfg.get("pos_weight_clip_max", 50.0))
        pos_weight_clip_min = float(bce_cfg.get("pos_weight_clip_min", 0.1))

        if radius <= 0:
            raise ValueError(f"fingerprint.radius must be > 0, got {radius}")
        if n_bits <= 0:
            raise ValueError(f"fingerprint.n_bits must be > 0, got {n_bits}")
        if pos_weight_clip_max <= 0:
            raise ValueError(f"bce.pos_weight_clip_max must be > 0, got {pos_weight_clip_max}")
        if pos_weight_clip_min <= 0:
            raise ValueError(f"bce.pos_weight_clip_min must be > 0, got {pos_weight_clip_min}")

        return {
            "label_source": label_source,
            "radius": radius,
            "n_bits": n_bits,
            "use_features": use_features,
            "use_chirality": use_chirality,
            "cache_dir": cache_dir,
            "overwrite_cache": overwrite_cache,
            "drop_constant_bits": drop_constant_bits,
            "pos_weight_clip_max": pos_weight_clip_max,
            "pos_weight_clip_min": pos_weight_clip_min,
        }

    def _resolve_repo_root(self) -> Path:
        # dataset.py lives in <repo>/data/, so parents[1] is the project root.
        return Path(__file__).resolve().parents[1]

    def _fingerprint_cache_path_for_cfg(self, fingerprint_cfg: Dict[str, Any]) -> Path:
        repo_root = self._resolve_repo_root()

        cache_dir = Path(str(fingerprint_cfg["cache_dir"]))
        if not cache_dir.is_absolute():
            cache_dir = repo_root / cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

        data_path = self.data_path
        if not data_path.is_absolute():
            data_path = (repo_root / data_path).resolve()
        else:
            data_path = data_path.resolve()

        try:
            mtime = data_path.stat().st_mtime
        except FileNotFoundError:
            mtime = 0.0

        payload = {
            "data_path": str(data_path),
            "mtime": mtime,
            "smiles_col": self.smiles_col,
            "radius": int(fingerprint_cfg["radius"]),
            "n_bits": int(fingerprint_cfg["n_bits"]),
            "use_features": bool(fingerprint_cfg["use_features"]),
            "use_chirality": bool(fingerprint_cfg["use_chirality"]),
            "drop_constant_bits": bool(fingerprint_cfg["drop_constant_bits"]),
        }
        key = hashlib.md5(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()  # noqa: S324

        stem = data_path.stem.replace(" ", "_")
        fname = (
            f"{stem}.morgan_r{payload['radius']}_b{payload['n_bits']}_"
            f"feat{int(payload['use_features'])}_chi{int(payload['use_chirality'])}_"
            f"drop{int(payload['drop_constant_bits'])}.{key[:10]}.npz"
        )
        return cache_dir / fname

    def _fingerprint_cache_path(self) -> Path:
        assert self.fingerprint_cfg is not None
        return self._fingerprint_cache_path_for_cfg(self.fingerprint_cfg)

    def _fingerprint_cache_path_aux(self) -> Path:
        assert self.fingerprint_aux_cfg is not None
        return self._fingerprint_cache_path_for_cfg(self.fingerprint_aux_cfg)

    def _init_fingerprint_cache(self) -> None:
        if Chem is None or rdFingerprintGenerator is None or DataStructs is None:
            raise RuntimeError("Fingerprint supervision requires RDKit, but rdkit is not available.")
        if fcntl is None:
            raise RuntimeError("Fingerprint supervision requires fcntl for file locking, but it is not available.")

        assert self.fingerprint_cfg is not None
        cache_path = self._fingerprint_cache_path()
        lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")

        with lock_path.open("w", encoding="utf-8") as lock_f:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)  # type: ignore[attr-defined]

            overwrite = bool(self.fingerprint_cfg.get("overwrite_cache", False))
            if cache_path.exists() and not overwrite:
                self._load_fingerprint_cache(cache_path)
                return

            print(
                "[Fingerprint] Building cache "
                f"(radius={self.fingerprint_cfg['radius']}, n_bits={self.fingerprint_cfg['n_bits']}) -> {cache_path}"
            )
            fp, valid = self._compute_morgan_fingerprint_matrix(
                radius=int(self.fingerprint_cfg["radius"]),
                n_bits=int(self.fingerprint_cfg["n_bits"]),
                use_features=bool(self.fingerprint_cfg["use_features"]),
                use_chirality=bool(self.fingerprint_cfg["use_chirality"]),
            )
            bit_mask, pos_weight = self._compute_fp_bit_mask_and_pos_weight(fp, valid, self.fingerprint_cfg)

            tmp_path = cache_path.with_name(cache_path.stem + f".tmp.{os.getpid()}.npz")
            np.savez_compressed(
                tmp_path,
                fp=fp,
                valid=valid,
                bit_mask=bit_mask,
                pos_weight=pos_weight,
                meta=json.dumps(self.fingerprint_cfg, ensure_ascii=False),
            )
            os.replace(tmp_path, cache_path)

            self._fp_labels = fp
            self._fp_valid = valid
            self._fp_bit_mask = bit_mask
            self._fp_pos_weight = pos_weight

            print(
                "[Fingerprint] Cache ready "
                f"(valid={int(valid.sum())}/{len(valid)}, active_bits={int(bit_mask.sum())}/{len(bit_mask)})"
            )

    def _load_fingerprint_cache(self, cache_path: Path) -> None:
        with np.load(cache_path, allow_pickle=False) as data:
            self._fp_labels = data["fp"]
            self._fp_valid = data["valid"]
            self._fp_bit_mask = data["bit_mask"]
            self._fp_pos_weight = data["pos_weight"]
        if self._fp_valid is not None and self._fp_bit_mask is not None:
            valid_count = int(self._fp_valid.sum())
            active_bits = int((self._fp_bit_mask > 0).sum())
            print(
                f"[Fingerprint] Loaded cache {cache_path} "
                f"(valid={valid_count}/{len(self._fp_valid)}, active_bits={active_bits}/{len(self._fp_bit_mask)})"
            )

    def _init_fingerprint_aux_cache(self) -> None:
        if Chem is None or rdFingerprintGenerator is None or DataStructs is None:
            raise RuntimeError("Fingerprint supervision requires RDKit, but rdkit is not available.")
        if fcntl is None:
            raise RuntimeError("Fingerprint supervision requires fcntl for file locking, but it is not available.")

        assert self.fingerprint_aux_cfg is not None
        cache_path = self._fingerprint_cache_path_aux()
        lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")

        with lock_path.open("w", encoding="utf-8") as lock_f:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)  # type: ignore[attr-defined]

            overwrite = bool(self.fingerprint_aux_cfg.get("overwrite_cache", False))
            if cache_path.exists() and not overwrite:
                self._load_fingerprint_aux_cache(cache_path)
                return

            print(
                "[FingerprintAux] Building cache "
                f"(radius={self.fingerprint_aux_cfg['radius']}, n_bits={self.fingerprint_aux_cfg['n_bits']}) -> {cache_path}"
            )
            fp, valid = self._compute_morgan_fingerprint_matrix(
                radius=int(self.fingerprint_aux_cfg["radius"]),
                n_bits=int(self.fingerprint_aux_cfg["n_bits"]),
                use_features=bool(self.fingerprint_aux_cfg["use_features"]),
                use_chirality=bool(self.fingerprint_aux_cfg["use_chirality"]),
            )
            bit_mask, pos_weight = self._compute_fp_bit_mask_and_pos_weight(fp, valid, self.fingerprint_aux_cfg)

            tmp_path = cache_path.with_name(cache_path.stem + f".tmp.{os.getpid()}.npz")
            np.savez_compressed(
                tmp_path,
                fp=fp,
                valid=valid,
                bit_mask=bit_mask,
                pos_weight=pos_weight,
                meta=json.dumps(self.fingerprint_aux_cfg, ensure_ascii=False),
            )
            os.replace(tmp_path, cache_path)

            self._fp_labels_aux = fp
            self._fp_valid_aux = valid
            self._fp_bit_mask_aux = bit_mask
            self._fp_pos_weight_aux = pos_weight

            print(
                "[FingerprintAux] Cache ready "
                f"(valid={int(valid.sum())}/{len(valid)}, active_bits={int(bit_mask.sum())}/{len(bit_mask)})"
            )

    def _load_fingerprint_aux_cache(self, cache_path: Path) -> None:
        with np.load(cache_path, allow_pickle=False) as data:
            self._fp_labels_aux = data["fp"]
            self._fp_valid_aux = data["valid"]
            self._fp_bit_mask_aux = data["bit_mask"]
            self._fp_pos_weight_aux = data["pos_weight"]
        if self._fp_valid_aux is not None and self._fp_bit_mask_aux is not None:
            valid_count = int(self._fp_valid_aux.sum())
            active_bits = int((self._fp_bit_mask_aux > 0).sum())
            print(
                f"[FingerprintAux] Loaded cache {cache_path} "
                f"(valid={valid_count}/{len(self._fp_valid_aux)}, active_bits={active_bits}/{len(self._fp_bit_mask_aux)})"
            )

    def _compute_morgan_fingerprint_matrix(
        self,
        radius: int,
        n_bits: int,
        use_features: bool,
        use_chirality: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert Chem is not None and rdFingerprintGenerator is not None and DataStructs is not None
        n = len(self.df)
        fp = np.zeros((n, n_bits), dtype=np.uint8)
        valid = np.zeros((n,), dtype=np.uint8)

        atom_inv_gen = None
        if use_features:
            atom_inv_gen = rdFingerprintGenerator.GetMorganFeatureAtomInvGen()  # type: ignore[attr-defined]
        gen = rdFingerprintGenerator.GetMorganGenerator(  # type: ignore[attr-defined]
            radius=radius,
            includeChirality=use_chirality,
            fpSize=n_bits,
            atomInvariantsGenerator=atom_inv_gen,
        )

        smiles_series = self.df[self.smiles_col]
        tmp = np.zeros((n_bits,), dtype=np.uint8)
        for i in range(n):
            s = smiles_series.iat[i]
            if s is None or (isinstance(s, float) and np.isnan(s)):
                continue
            smiles = str(s)
            mol = Chem.MolFromSmiles(smiles)  # type: ignore[attr-defined]
            if mol is None:
                continue
            try:
                bv = gen.GetFingerprint(mol)
            except Exception:  # noqa: BLE001
                continue
            tmp.fill(0)
            DataStructs.ConvertToNumpyArray(bv, tmp)  # type: ignore[attr-defined]
            fp[i] = tmp
            valid[i] = 1

        return fp, valid

    def _compute_fp_bit_mask_and_pos_weight(
        self, fp: np.ndarray, valid: np.ndarray, fingerprint_cfg: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_bits = int(fingerprint_cfg["n_bits"])
        if fp.ndim != 2 or fp.shape[1] != n_bits:
            raise ValueError(f"Fingerprint matrix shape mismatch: fp.shape={fp.shape}, expected n_bits={n_bits}")
        valid_count = int(valid.sum())
        if valid_count <= 0:
            bit_mask = np.zeros((n_bits,), dtype=np.float32)
            pos_weight = np.ones((n_bits,), dtype=np.float32)
            return bit_mask, pos_weight

        fp_valid = fp[valid.astype(bool)]
        pos_counts = fp_valid.sum(axis=0).astype(np.float32)
        neg_counts = float(valid_count) - pos_counts

        if bool(fingerprint_cfg.get("drop_constant_bits", True)):
            active = (pos_counts > 0.0) & (pos_counts < float(valid_count))
        else:
            active = np.ones((n_bits,), dtype=bool)

        bit_mask = active.astype(np.float32)

        eps = 1e-6
        pos_weight = np.ones((n_bits,), dtype=np.float32)
        pos_weight[active] = neg_counts[active] / (pos_counts[active] + eps)

        clip_min = float(fingerprint_cfg.get("pos_weight_clip_min", 0.1))
        clip_max = float(fingerprint_cfg.get("pos_weight_clip_max", 50.0))
        pos_weight = np.clip(pos_weight, clip_min, clip_max)

        return bit_mask, pos_weight

    def __len__(self):
        return len(self.df)

    def _token_spans(self, smiles: str) -> Optional[List[Tuple[int, int, str]]]:
        pattern = re.compile(
            r"(\\[[^\\[\\]]+\\]|Br|Cl|Si|Se|Na|Ca|Li|Mg|Al|Zn|[B-IK-PR-Z]|[bcnops]|\\(|\\)|=|#|-|\\+|\\\\|/|\\.|:|~|@|\\?|>|\\*|\\$|%\\d\\d|\\d)"
        )
        spans: List[Tuple[int, int, str]] = []
        idx = 0
        for match in pattern.finditer(smiles):
            if match.start() != idx:
                return None
            spans.append((match.start(), match.end(), match.group()))
            idx = match.end()
        if idx != len(smiles):
            return None
        return spans

    def _build_char_atom_mapping(self, smiles: str) -> Optional[Tuple[List[int], int]]:
        spans = self._token_spans(smiles)
        if spans is None:
            return None

        atom_set = {
            "B", "C", "N", "O", "P", "S", "F", "I", "Cl", "Br", "Si", "Se",
            "Na", "Ca", "Li", "Mg", "Al", "Zn", "K", "H",
        }
        aromatic_set = {"b", "c", "n", "o", "p", "s"}
        mapping = [-1] * len(smiles)
        atom_idx = 0
        for start, end, token in spans:
            is_atom = token.startswith("[") or token in atom_set or token in aromatic_set
            if is_atom:
                for pos in range(start, end):
                    mapping[pos] = atom_idx
                atom_idx += 1
        return mapping, atom_idx

    def _get_graph_distances(self, smiles: str) -> np.ndarray:
        cache_key = f"{smiles}|{self.smiles_max_length}|{self.graph_attention_max_distance}"
        if cache_key in self._graph_dist_cache:
            graph_dist = self._graph_dist_cache.pop(cache_key)
            self._graph_dist_cache[cache_key] = graph_dist
            return graph_dist

        mol = Chem.MolFromSmiles(smiles)  # type: ignore[attr-defined]
        if mol is None:
            return np.zeros((self.smiles_max_length, self.smiles_max_length), dtype=np.int16)

        mapping_info = self._build_char_atom_mapping(smiles)
        if mapping_info is None:
            return np.zeros((self.smiles_max_length, self.smiles_max_length), dtype=np.int16)
        mapping, atom_count = mapping_info
        if atom_count != mol.GetNumAtoms():
            return np.zeros((self.smiles_max_length, self.smiles_max_length), dtype=np.int16)

        dist = Chem.GetDistanceMatrix(mol).astype(np.int16)  # type: ignore[attr-defined]

        has_sep = self.smiles_vocab.sep_token_id is not None
        max_smiles_len = self.smiles_max_length - (1 + (1 if has_sep else 0))
        mapping = mapping[:max_smiles_len]

        token_to_atom = np.full((self.smiles_max_length,), -1, dtype=np.int16)
        token_to_atom[1:1 + len(mapping)] = np.array(mapping, dtype=np.int16)

        ai = token_to_atom[:, None]
        aj = token_to_atom[None, :]
        mask = (ai >= 0) & (aj >= 0)

        dist_tok = np.zeros((self.smiles_max_length, self.smiles_max_length), dtype=np.int16)
        if mask.any():
            dist_clipped = np.minimum(dist, self.graph_attention_max_distance).astype(np.int16)
            dist_tok[mask] = dist_clipped[ai[mask], aj[mask]]

        dist_tok = np.where(mask, np.minimum(dist_tok, self.graph_attention_max_distance) + 1, 0).astype(np.int16)

        self._graph_dist_cache[cache_key] = dist_tok
        if len(self._graph_dist_cache) > self.graph_cache_size:
            self._graph_dist_cache.popitem(last=False)
        return dist_tok

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        # 1. 获取 SMILES（可能是随机变体）
        smiles = row.get(self.smiles_col)
        if smiles is None or pd.isna(smiles):
            raise KeyError(f"SMILES is missing at idx={idx} (col={self.smiles_col})")
        smiles = str(smiles)

        sequence_id = row.get('sequence_id', f'sample_{idx}')

        smiles_view1 = smiles
        if self.random_smiles_provider:
            smiles_view1 = self.random_smiles_provider.sample(sequence_id, smiles)

        smiles_view2 = None
        if self.consistency_enabled:
            if self.consistency_smiles_provider is not None:
                deterministic = bool(self.consistency_cfg.get("deterministic", False))
                smiles_view2 = self.consistency_smiles_provider.sample_variant(smiles, deterministic=deterministic)
            else:
                smiles_view2 = smiles_view1

        # 2. 编码 SMILES
        smiles_tokens = list(smiles_view1)
        smiles_input_ids = self.smiles_vocab.encode(
            smiles_tokens,
            add_special_tokens=True,
            max_length=self.smiles_max_length,
            padding='max_length',
            truncation=True
        )

        # 3. Attention mask
        smiles_attention_mask = [1 if token_id != 0 else 0 for token_id in smiles_input_ids]

        # 4. MLM masking（可选）
        smiles_mlm_labels = None
        if self.apply_smiles_masking:
            smiles_input_ids_tensor = torch.tensor(smiles_input_ids)
            smiles_input_ids_masked, smiles_mlm_labels = mask_smiles_tokens(
                smiles_input_ids_tensor.unsqueeze(0),
                mask_prob=self.mask_prob,
                vocab_size=len(self.smiles_vocab)
            )
            smiles_input_ids = smiles_input_ids_masked.squeeze(0).tolist()
            smiles_mlm_labels = smiles_mlm_labels.squeeze(0)

        # 5. 构建输出
        # 5. 构建输出
        # row_index:
        # - 对于 ncAA CPP 数据，优先使用 CycPeptMPDB_ID 作为稳定ID，方便对齐ChemBERTa缓存
        # - 否则回退到行索引 idx
        if 'CycPeptMPDB_ID' in self.df.columns:
            row_index_value = row['CycPeptMPDB_ID']
        else:
            row_index_value = idx

        output = {
            'smiles_input_ids': torch.tensor(smiles_input_ids, dtype=torch.long),
            'smiles_attention_mask': torch.tensor(smiles_attention_mask, dtype=torch.long),
            'sequence_id': sequence_id,  # 关键：用于加载 ESM2 缓存
            'row_index': row_index_value
        }

        if self.graph_attention_enabled:
            graph_dist = self._get_graph_distances(smiles_view1)
            output['graph_distances'] = torch.from_numpy(graph_dist)

        if self.consistency_enabled and smiles_view2 is not None:
            view2_tokens = list(smiles_view2)
            view2_ids = self.smiles_vocab.encode(
                view2_tokens,
                add_special_tokens=True,
                max_length=self.smiles_max_length,
                padding='max_length',
                truncation=True
            )
            view2_mask = [1 if token_id != 0 else 0 for token_id in view2_ids]
            output['smiles_input_ids_view2'] = torch.tensor(view2_ids, dtype=torch.long)
            output['smiles_attention_mask_view2'] = torch.tensor(view2_mask, dtype=torch.long)
            if self.graph_attention_enabled:
                graph_dist_v2 = self._get_graph_distances(smiles_view2)
                output['graph_distances_view2'] = torch.from_numpy(graph_dist_v2)

        # 6. MLM labels
        if smiles_mlm_labels is not None:
            output['smiles_mlm_labels'] = smiles_mlm_labels

        # 7. 任务标签（Phase 3 用）
        if self.label_column and self.label_column in row:
            label = row[self.label_column]
            if pd.notna(label):
                label_value = float(label)

                # 应用归一化
                if self.normalize_label:
                    if self.norm_type == 'zscore':
                        # z = (x - mean) / std
                        label_value = (label_value - self.norm_params['mean']) / self.norm_params['std']
                    elif self.norm_type == 'minmax':
                        # x_norm = (x - min) / (max - min)
                        label_range = self.norm_params['max'] - self.norm_params['min']
                        label_value = (label_value - self.norm_params['min']) / label_range

                output['task_label'] = torch.tensor(label_value, dtype=torch.float32)

        # 8. 多任务监督标签（Phase2/Phase3 的轻监督扩展用）
        if self.task_columns:
            labels: List[float] = []
            masks: List[float] = []
            for col in self.task_columns:
                val = row.get(col)
                if val is None or pd.isna(val):
                    labels.append(0.0)
                    masks.append(0.0)
                else:
                    labels.append(float(val))
                    masks.append(1.0)
            output["task_labels"] = torch.tensor(labels, dtype=torch.float32)
            output["task_mask"] = torch.tensor(masks, dtype=torch.float32)

        # 9. 指纹多标签监督（Phase2 proxy）
        if self.fingerprint_cfg is not None and self._fp_labels is not None:
            valid = self._fp_valid is not None and int(self._fp_valid[idx]) == 1
            if not valid:
                output["task_labels"] = self._fp_zero_labels_tensor
                output["task_mask"] = self._fp_zero_mask_tensor
            else:
                output["task_labels"] = torch.from_numpy(self._fp_labels[idx]).to(dtype=torch.uint8)
                if self._fp_bit_mask_tensor is None:
                    raise RuntimeError("Fingerprint bit mask tensor is not initialized.")
                output["task_mask"] = self._fp_bit_mask_tensor

        # 10. 指纹多标签监督（Phase2 proxy aux：用于双 proxy 的 fp 正则）
        if self.fingerprint_aux_cfg is not None and self._fp_labels_aux is not None:
            valid_aux = self._fp_valid_aux is not None and int(self._fp_valid_aux[idx]) == 1
            if not valid_aux:
                output["task_labels_aux"] = self._fp_zero_labels_tensor_aux
                output["task_mask_aux"] = self._fp_zero_mask_tensor_aux
            else:
                output["task_labels_aux"] = torch.from_numpy(self._fp_labels_aux[idx]).to(dtype=torch.uint8)
                if self._fp_bit_mask_tensor_aux is None:
                    raise RuntimeError("Aux fingerprint bit mask tensor is not initialized.")
                output["task_mask_aux"] = self._fp_bit_mask_tensor_aux

        # 11. Aux 任务监督标签（非指纹：来自 CSV 列）
        if self.fingerprint_aux_cfg is None:
            if self.task_columns_aux:
                labels_aux: List[float] = []
                masks_aux: List[float] = []
                for col in self.task_columns_aux:
                    val = row.get(col)
                    if val is None or pd.isna(val):
                        labels_aux.append(0.0)
                        masks_aux.append(0.0)
                    else:
                        labels_aux.append(float(val))
                        masks_aux.append(1.0)
                output["task_labels_aux"] = torch.tensor(labels_aux, dtype=torch.float32)
                output["task_mask_aux"] = torch.tensor(masks_aux, dtype=torch.float32)
            elif self.label_column_aux and self.label_column_aux in row:
                val = row[self.label_column_aux]
                if pd.notna(val):
                    output["task_label_aux"] = torch.tensor(float(val), dtype=torch.float32)

        return output

    @property
    def task_pos_weight(self) -> Optional[torch.Tensor]:
        if self._fp_pos_weight is None:
            return None
        return torch.from_numpy(self._fp_pos_weight).to(dtype=torch.float32)

    @property
    def task_aux_pos_weight(self) -> Optional[torch.Tensor]:
        if self._fp_pos_weight_aux is None:
            return None
        return torch.from_numpy(self._fp_pos_weight_aux).to(dtype=torch.float32)


def test_dataset():
    """Test SingleInputDataset"""
    print("Testing SingleInputDataset...")

    # Create a dummy dataset
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("sequence_id,sequence,smiles,Permeability\n")
        f.write("seq_001,MFANLSQDKD,CC(C)C[C@H](N)C(=O)O,0.5\n")
        f.write("seq_002,IYTGMKAGLL,CC(C)[C@@H](N)C(=O)O,0.3\n")
        f.write("seq_003,QDKDQKNNGH,CC(C)C[C@H](N)C(=O)O,0.7\n")
        temp_file = f.name

    try:
        # Create vocab manager
        vocab_file = "data/ncaa_adaptation_v2/vocab_smiles.txt"
        if not Path(vocab_file).exists():
            print(f"Warning: {vocab_file} not found, using dummy vocab")
            # Create dummy vocab
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as vf:
                vf.write("[PAD]\n[CLS]\n[SEP]\n[MASK]\n")
                vf.write("C\nN\nO\nS\n(\n)\n=\n[\n]\n@\n+\n-\n1\n2\n3\n")
                vocab_file = vf.name

        smiles_vocab = VocabManager(vocab_file)

        # Create dataset
        dataset = SingleInputDataset(
            data_path=temp_file,
            smiles_vocab_manager=smiles_vocab,
            smiles_max_length=100,
            apply_smiles_masking=True,
            mask_prob=0.15,
            random_smiles_config={'enabled': True, 'mix_ratio': 0.25},
            label_column='Permeability'
        )

        print(f"\nDataset size: {len(dataset)}")

        # Test sample
        sample = dataset[0]
        print("\nSample 0:")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  {key}: {value}")

    finally:
        # Cleanup
        os.unlink(temp_file)


if __name__ == "__main__":
    test_dataset()
