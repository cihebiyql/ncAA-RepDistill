"""Utility classes for loading precomputed teacher feature caches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch


class TeacherFeatureCacheSplit:
    """Holds cached teacher features for a single split."""

    def __init__(self, features: torch.Tensor, ids: Sequence[str]):
        self.features = features
        self.id_to_index: Dict[str, int] = {}
        for idx, sample_id in enumerate(ids):
            self.id_to_index[str(sample_id)] = idx

    def get(self, sample_ids: Sequence[str]) -> Tuple[Optional[torch.Tensor], List[str]]:
        indices: List[int] = []
        missing: List[str] = []

        for sample_id in sample_ids:
            key = str(sample_id)
            if key not in self.id_to_index:
                missing.append(key)
                continue
            indices.append(self.id_to_index[key])

        if missing:
            return None, missing

        index_tensor = torch.tensor(indices, dtype=torch.long)
        return self.features.index_select(0, index_tensor), []


class TeacherFeatureCacheManager:
    """Lazy loader for teacher feature caches across splits."""

    def __init__(
        self,
        cache_path: str,
        id_keys: Optional[Iterable[str]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.cache_root = Path(cache_path)
        self.id_keys = list(id_keys or ["sequence_ids", "sample_ids"])
        self.device = device

        self.split_files = self._discover_split_files()
        self.loaded_splits: Dict[str, TeacherFeatureCacheSplit] = {}

    def _discover_split_files(self) -> Dict[str, Path]:
        if self.cache_root.is_file():
            split_name = self._extract_split_name(self.cache_root.name)
            return {split_name: self.cache_root}

        split_files: Dict[str, Path] = {}
        for path in self.cache_root.glob("*_features.pt"):
            split_name = self._extract_split_name(path.name)
            split_files[split_name] = path
        return split_files

    @staticmethod
    def _extract_split_name(filename: str) -> str:
        name = filename.replace(".pt", "")
        if name.endswith("_features"):
            name = name[:-9]
        return name

    def _load_split(self, split: str) -> None:
        if split not in self.split_files or split in self.loaded_splits:
            return

        cache_file = self.split_files[split]
        data = torch.load(cache_file, map_location="cpu")

        features = data.get("pooled_features")
        if features is None:
            features = data.get("features")
            if features is not None and features.ndim == 3:
                features = features[:, 0, :]

        if features is None:
            raise ValueError(f"Cache file {cache_file} is missing pooled features")

        id_list = None
        for key in self.id_keys:
            if key in data:
                id_list = data[key]
                break

        if id_list is None:
            raise ValueError(f"Cache file {cache_file} missing identifier columns {self.id_keys}")

        tensor = features.detach().clone().contiguous()
        self.loaded_splits[split] = TeacherFeatureCacheSplit(tensor, id_list)

    def get(self, split: str, sample_ids: Sequence[str]) -> Tuple[Optional[torch.Tensor], List[str]]:
        if split not in self.split_files:
            return None, list(sample_ids)

        if split not in self.loaded_splits:
            self._load_split(split)

        cache_split = self.loaded_splits.get(split)
        if cache_split is None:
            return None, list(sample_ids)

        features, missing = cache_split.get(sample_ids)
        if features is None:
            return None, missing

        if self.device is not None:
            features = features.to(self.device)
        return features, missing


__all__ = ["TeacherFeatureCacheManager"]

