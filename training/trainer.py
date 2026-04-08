"""
双教师蒸馏训练器 (Phase 1 V3)

核心训练循环，包含：
- 梯度累积
- EMA
- 学习率调度
- Checkpoint 保存
- 验证评估
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import time
import json
from typing import Dict, List, Optional
import sys
import random

import numpy as np
import pandas as pd

# Import model and losses
sys.path.insert(0, str(Path(__file__).parent.parent))
from model.single_input_dual_teacher import SingleInputDualTeacher
from model.losses import compute_dual_teacher_loss, contrastive_infonce_loss
from data.dataloader import build_dataloaders

# Import utils
from utils.teacher_cache import TeacherFeatureCacheManager
from utils.ema import ExponentialMovingAverage
from utils.early_stopping import StepEarlyStopping

# Import evaluator
from training.evaluator import DualTeacherEvaluator
from math import inf


class DualTeacherTrainer:
    """
    双教师蒸馏训练器（Phase 1 V3）

    负责：
    - 初始化模型、优化器、调度器
    - 训练循环（梯度累积、EMA更新）
    - 验证评估
    - Checkpoint 保存
    - 日志记录
    """

    def __init__(self, config: Dict):
        self.config = config
        self._maybe_set_seed()
        self.device = torch.device(f"cuda:{config['device']['gpu_id']}" if torch.cuda.is_available() else "cpu")

        # 设置输出目录
        self.output_dir = Path(config['training']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)

        print(f"\n{'='*70}")
        print(f"Initializing DualTeacherTrainer")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Output: {self.output_dir}")

        # 初始化模型
        self._init_model()

        # 初始化数据加载器
        self._init_dataloaders()
        self._maybe_init_task_supervision_pos_weight()
        self._maybe_init_task_supervision_aux_pos_weight()

        # 初始化教师特征缓存
        self._init_teacher_caches()

        # 初始化优化器和调度器
        self._init_optimizer()

        # 初始化 EMA
        self._init_ema()

        # 可选：Phase2→Phase3 联动（用 Phase3-val R² 选 ckpt / 早停）
        self.phase3_probe_cfg = (self.config.get("phase3_probe") or {}) if isinstance(self.config, dict) else {}
        self.phase3_probe_loaders = None
        self.phase3_early_stopper: Optional[StepEarlyStopping] = None
        self._init_phase3_probe()

        # 初始化评估器
        self.evaluator = DualTeacherEvaluator(device=self.device)

        # 训练状态
        self.global_step = 0
        self.current_epoch = 0
        self.best_metric_name, self.best_metric_mode = self._get_best_metric_config()
        eval_cfg = self.config.get("evaluation", {}) or {}
        best_from_epoch = eval_cfg.get("best_from_epoch")
        if best_from_epoch is None:
            best_from_epoch = eval_cfg.get("best_min_epoch", 0)
        self.best_from_epoch = int(best_from_epoch or 0)
        self.best_metric = inf if self.best_metric_mode == "min" else -inf  # 用于保存最佳模型
        self.start_epoch = 0

        # 保存配置
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

        # 尝试断点恢复
        self._maybe_resume_checkpoint()

        print(f"{'='*70}\n")

    def _maybe_set_seed(self) -> None:
        seed = self.config.get("seed")
        if seed is None:
            seed = (self.config.get("training") or {}).get("seed")
        if seed is None:
            return

        seed_int = int(seed)
        random.seed(seed_int)
        np.random.seed(seed_int % (2**32))
        torch.manual_seed(seed_int)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_int)
        print(f"[Seed] Global seed set to: {seed_int}")

    def _get_best_metric_config(self):
        """读取配置中的最佳指标设置（用于保存 best checkpoint）"""
        eval_cfg = self.config.get("evaluation", {}) or {}
        best_metric_name = eval_cfg.get("best_metric", "protein_spearman")

        # 优先使用 minimize / best_metric_mode
        if eval_cfg.get("minimize") is True:
            best_metric_mode = "min"
        else:
            best_metric_mode = eval_cfg.get("best_metric_mode") or eval_cfg.get("mode") or "max"

        if best_metric_mode not in {"min", "max"}:
            best_metric_mode = "max"

        return best_metric_name, best_metric_mode

    def _init_phase3_probe(self) -> None:
        cfg = self.phase3_probe_cfg or {}
        if not cfg.get("enabled", False):
            return

        feature_type = cfg.get("feature_type", "molecular_proj")
        if feature_type not in {"molecular_proj", "smiles_repr"}:
            raise ValueError(f"Unsupported phase3_probe.feature_type: {feature_type}")

        batch_size = int(cfg.get("batch_size", 64))
        data_num_workers = int((self.config.get("data") or {}).get("num_workers", 0))
        num_workers = int(cfg.get("num_workers", data_num_workers))
        cfg["batch_size"] = batch_size
        cfg["num_workers"] = num_workers
        cfg["alpha_grid"] = cfg.get("alpha_grid") or [1e-4, 1e-3, 1e-2, 1e-1, 1.0]

        es_cfg = cfg.get("early_stopping") or {}
        if es_cfg.get("enabled", True):
            patience_epochs = int(es_cfg.get("patience_epochs", 4))
            min_delta = float(es_cfg.get("min_delta", 1e-4))
            self.phase3_early_stopper = StepEarlyStopping(
                patience_steps=patience_epochs,
                mode="max",
                min_delta=min_delta,
            )
            print(f"[Phase3Probe] Early stopping enabled (patience={patience_epochs}, min_delta={min_delta})")

        print(
            "[Phase3Probe] Enabled "
            f"(feature_type={feature_type}, batch_size={batch_size}, alpha_grid={cfg['alpha_grid']})"
        )

    def _build_phase3_probe_dataloaders(self):
        from data.dataloader import collate_and_trim_smiles_batch
        from data.dataset import SingleInputDataset

        cfg = self.phase3_probe_cfg or {}
        data_cfg = self.config.get("data", {}) or {}
        paths_cfg = data_cfg.get("paths", {}) or {}

        train_data = cfg.get("train_data") or paths_cfg.get("train_data")
        val_data = cfg.get("val_data") or paths_cfg.get("val_data")
        if not train_data or not val_data:
            raise ValueError("phase3_probe requires train_data and val_data")

        smiles_max_length = int(self.config.get("student", {}).get("smiles_encoder", {}).get("max_length", 768))
        graph_attention_config = (self.config.get("student") or {}).get("graph_attention") or {}
        batch_size = int(cfg.get("batch_size", 64))
        num_workers = int(cfg.get("num_workers", data_cfg.get("num_workers", 0)))

        train_ds = SingleInputDataset(
            data_path=str(train_data),
            smiles_vocab_manager=self.smiles_vocab,
            smiles_max_length=smiles_max_length,
            apply_smiles_masking=False,
            mask_prob=0.0,
            random_smiles_config=None,
            consistency_config=None,
            graph_attention_config=graph_attention_config,
            label_column="Permeability",
            normalization_config=None,
            split_name="phase3_probe_train",
        )
        val_ds = SingleInputDataset(
            data_path=str(val_data),
            smiles_vocab_manager=self.smiles_vocab,
            smiles_max_length=smiles_max_length,
            apply_smiles_masking=False,
            mask_prob=0.0,
            random_smiles_config=None,
            consistency_config=None,
            graph_attention_config=graph_attention_config,
            label_column="Permeability",
            normalization_config=None,
            split_name="phase3_probe_val",
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=bool(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
            collate_fn=collate_and_trim_smiles_batch,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=bool(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
            collate_fn=collate_and_trim_smiles_batch,
        )
        return {"train": train_loader, "val": val_loader}

    @torch.no_grad()
    def _extract_phase3_probe_features(self, dataloader: DataLoader, feature_type: str):
        features = []
        labels = []
        self.model.eval()

        for batch in dataloader:
            smiles_input_ids = batch["smiles_input_ids"].to(self.device, non_blocking=True)
            smiles_attention_mask = batch["smiles_attention_mask"].to(self.device, non_blocking=True)
            graph_distances = batch.get("graph_distances")
            if graph_distances is not None:
                graph_distances = graph_distances.to(self.device, non_blocking=True)
            y = batch["task_label"].to(self.device, non_blocking=True)

            outputs = self.model(
                smiles_input_ids,
                smiles_attention_mask,
                graph_distances=graph_distances,
                return_all=False,
            )
            x = outputs[feature_type]

            features.append(x.detach().float().cpu())
            labels.append(y.detach().float().cpu())

        x_all = torch.cat(features, dim=0)
        y_all = torch.cat(labels, dim=0)
        return x_all, y_all

    @torch.no_grad()
    def _phase3_probe_ridge_val(self, x_train: torch.Tensor, y_train: torch.Tensor, x_val: torch.Tensor, y_val: torch.Tensor):
        cfg = self.phase3_probe_cfg or {}
        alpha_grid = [float(a) for a in (cfg.get("alpha_grid") or [1e-4, 1e-3, 1e-2, 1e-1, 1.0])]

        mean = x_train.mean(dim=0, keepdim=True)
        std = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
        x_train = (x_train - mean) / std
        x_val = (x_val - mean) / std

        device = self.device
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        x_val = x_val.to(device)
        y_val = y_val.to(device)

        n, d = x_train.shape
        ones_train = torch.ones((n, 1), device=device)
        x_train_aug = torch.cat([x_train, ones_train], dim=1)  # bias
        xtx = x_train_aug.T @ x_train_aug
        xty = x_train_aug.T @ y_train.unsqueeze(-1)
        eye = torch.eye(d + 1, device=device)

        best = {
            "val_rmse": float("inf"),
            "val_r2": float("-inf"),
            "best_alpha": float(alpha_grid[0]),
        }

        ones_val = torch.ones((x_val.shape[0], 1), device=device)
        x_val_aug = torch.cat([x_val, ones_val], dim=1)

        y_val_mean = y_val.mean()
        ss_tot = torch.sum((y_val - y_val_mean) ** 2).clamp_min(1e-8)

        for alpha in alpha_grid:
            w = torch.linalg.solve(xtx + float(alpha) * eye, xty)
            pred = (x_val_aug @ w).squeeze(-1)
            mse = torch.mean((pred - y_val) ** 2)
            rmse = torch.sqrt(mse).item()

            ss_res = torch.sum((y_val - pred) ** 2)
            r2 = (1.0 - (ss_res / ss_tot)).item()
            if rmse < best["val_rmse"]:
                best["val_rmse"] = float(rmse)
                best["val_r2"] = float(r2)
                best["best_alpha"] = float(alpha)

        return best

    def _run_phase3_probe(self) -> Dict[str, float]:
        cfg = self.phase3_probe_cfg or {}
        if not cfg.get("enabled", False):
            return {}

        feature_type = cfg.get("feature_type", "molecular_proj")
        use_ema = bool(cfg.get("use_ema", False))

        if self.phase3_probe_loaders is None:
            self.phase3_probe_loaders = self._build_phase3_probe_dataloaders()

        if use_ema and self.ema is not None:
            self.ema.apply_shadow(self.model)

        try:
            x_train, y_train = self._extract_phase3_probe_features(
                self.phase3_probe_loaders["train"], feature_type
            )
            x_val, y_val = self._extract_phase3_probe_features(
                self.phase3_probe_loaders["val"], feature_type
            )
            best = self._phase3_probe_ridge_val(x_train, y_train, x_val, y_val)
        finally:
            if use_ema and self.ema is not None:
                self.ema.restore(self.model)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return {
            "phase3_val_rmse": float(best["val_rmse"]),
            "phase3_val_r2": float(best["val_r2"]),
            "phase3_best_alpha": float(best["best_alpha"]),
        }

    def _init_model(self):
        """初始化学生模型"""
        print("\n[Init] Creating student model...")

        use_grad_checkpoint = self.config['training'].get('gradient_checkpointing', {}).get('enabled', False)
        student_cfg = self.config.get('student', {})
        fusion_cfg = self.config.get("fusion") or {}
        if not isinstance(fusion_cfg, dict):
            fusion_cfg = {}

        self.model = SingleInputDualTeacher(
            smiles_vocab_size=student_cfg['smiles_encoder']['vocab_size'],
            smiles_embed_dim=student_cfg['smiles_encoder']['embed_dim'],
            smiles_num_layers=student_cfg['smiles_encoder']['num_layers'],
            smiles_num_heads=student_cfg['smiles_encoder']['num_heads'],
            smiles_max_length=student_cfg['smiles_encoder']['max_length'],
            esm2_dim=student_cfg['projection']['esm2_dim'],
            chemberta_dim=student_cfg['projection']['chemberta_dim'],
            graph_attention=student_cfg.get("graph_attention"),
            contrastive_proj_cfg=student_cfg.get("contrastive_proj"),
            fusion_cfg=fusion_cfg,
            use_smiles_mlm=student_cfg.get('use_smiles_mlm', True),
            dropout=student_cfg['smiles_encoder'].get('dropout', 0.1),
            use_gradient_checkpointing=use_grad_checkpoint
        )

        self.model.to(self.device)
        self.model.print_model_info()

        # 可选：从已有 checkpoint 加载学生模型权重（用于 Phase 2/3 初始化）
        resume_from = student_cfg.get('resume_from')
        if resume_from:
            resume_path = Path(resume_from)
            if not resume_path.exists():
                print(f"[Init] Warning: student.resume_from not found: {resume_path}")
            else:
                print(f"\n[Init] Loading student weights from {resume_path} ...")
                checkpoint = torch.load(resume_path, map_location=self.device)
                state_dict = checkpoint.get('model_state_dict', checkpoint)
                missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
                print(f"[Init] Loaded student weights (missing={len(missing)}, unexpected={len(unexpected)})")

        # 可选：P4（Phase2 引入轻量 Permeability 监督项）
        self._init_task_supervision()
        self._init_task_supervision_aux()

        # 可选：Phase2 漂移控制（冻结 SMILES encoder 的部分层）
        self._maybe_freeze_smiles_encoder()

        # Cache consistency loss config on the model so evaluator can compute val consistency loss
        # without accessing the full trainer config.
        consistency_cfg = ((self.config.get("loss") or {}).get("consistency") or {})
        if isinstance(consistency_cfg, dict):
            setattr(self.model, "_consistency_cfg", consistency_cfg)

    def _get_fusion_cfg(self) -> Dict:
        cfg = self.config.get("fusion") or {}
        if not isinstance(cfg, dict):
            return {}
        return cfg

    def _resolve_fusion_weight(self, epoch: int) -> float:
        cfg = self._get_fusion_cfg()
        if not bool(cfg.get("enabled", False)):
            return 0.0

        start_epoch = int(cfg.get("start_epoch", 0))
        if epoch < start_epoch:
            return 0.0
        return float(cfg.get("loss_weight", 0.0))

    def _maybe_freeze_smiles_encoder_for_fusion(self, epoch: int) -> None:
        cfg = self._get_fusion_cfg()
        if not bool(cfg.get("enabled", False)):
            return
        if not bool(cfg.get("freeze_smiles", False)):
            return

        start_epoch = int(cfg.get("start_epoch", 0))
        if epoch < start_epoch:
            return
        if getattr(self, "_fusion_smiles_frozen", False):
            return

        enc = getattr(self.model, "smiles_encoder", None)
        if enc is None:
            print("[Fusion] Warning: model has no smiles_encoder; skip fusion.freeze_smiles.")
            self._fusion_smiles_frozen = True
            return

        frozen_params = 0
        frozen_tensors = 0
        for p in enc.parameters(recurse=True):
            if not p.requires_grad:
                continue
            frozen_tensors += 1
            frozen_params += p.numel()
            p.requires_grad = False

        self._fusion_smiles_frozen = True
        print(
            f"[Fusion] freeze_smiles enabled at epoch {epoch + 1}: "
            f"frozen_tensors={frozen_tensors}, frozen_params={frozen_params / 1e6:.2f}M"
        )

    def _compute_fusion_residual_distill_loss(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_features: Dict[str, torch.Tensor],
        fusion_weight: float,
    ) -> torch.Tensor:
        if fusion_weight <= 0:
            return torch.tensor(0.0, device=self.device)

        cfg = self._get_fusion_cfg()
        mode = str(cfg.get("mode", "residual_distill")).strip().lower()
        if mode != "residual_distill":
            raise ValueError(f"Unsupported fusion.mode: {mode}")

        target = str(cfg.get("target", "geminimol")).strip().lower()
        if target not in {"geminimol", "chemberta", "molecular_teacher"}:
            raise ValueError(f"Unsupported fusion.target: {target}")

        t = teacher_features.get("chemberta")
        if t is None:
            return torch.tensor(0.0, device=self.device)

        if "fusion_hs" not in student_outputs or "fusion_hp" not in student_outputs:
            raise KeyError("Fusion enabled but model outputs missing fusion_hs/fusion_hp.")

        hs = student_outputs["fusion_hs"]
        hp = student_outputs["fusion_hp"]

        if bool(cfg.get("stop_grad_hs", True)) or bool(cfg.get("freeze_smiles", False)):
            hs = hs.detach()

        residual = t.to(hp.device) - hs
        return F.mse_loss(hp, residual)

    def _init_task_supervision(self) -> None:
        cfg = (self.config.get("loss") or {}).get("task_supervision") or {}
        if not isinstance(cfg, dict):
            cfg = {}

        self.task_supervision_cfg = cfg
        enabled = bool(cfg.get("enabled", False))
        if not enabled:
            self.task_supervision_head = None
            self.task_supervision_tasks = []
            self.task_supervision_alpha = None
            self.task_supervision_norm = None
            self.task_supervision_loss_type = "mse"
            self.task_supervision_pos_weight = None
            self.task_supervision_output_dim = 0
            return

        label_source = str(cfg.get("label_source") or "").strip().lower()
        is_fingerprint = label_source in {"morgan_fp", "morgan", "ecfp", "fingerprint", "fp"}

        loss_type = str(cfg.get("loss_type") or ("bce" if is_fingerprint else "mse")).strip().lower()
        if loss_type not in {"mse", "bce"}:
            raise ValueError(f"Unsupported loss.task_supervision.loss_type: {loss_type}")
        self.task_supervision_loss_type = loss_type
        self.task_supervision_pos_weight = None

        task_names: List[str] = []
        if is_fingerprint:
            fp_cfg = cfg.get("fingerprint") or {}
            if not isinstance(fp_cfg, dict):
                fp_cfg = {}
            output_dim = int(fp_cfg.get("n_bits", 1024))
            if output_dim <= 0:
                raise ValueError(f"loss.task_supervision.fingerprint.n_bits must be > 0, got {output_dim}")
        else:
            tasks = cfg.get("tasks")
            if isinstance(tasks, (list, tuple)) and tasks:
                task_names = [str(t) for t in tasks]
            elif isinstance(tasks, str) and tasks.strip():
                task_names = [tasks.strip()]
            else:
                label_column = cfg.get("label_column") or "Permeability"
                task_names = [str(label_column)]
            output_dim = len(task_names)

        self.task_supervision_tasks = task_names
        self.task_supervision_output_dim = int(output_dim)

        feature_type = cfg.get("feature_type") or "molecular_proj"
        if feature_type not in {"molecular_proj", "smiles_repr", "smiles_mean_pool"}:
            raise ValueError(f"Unsupported loss.task_supervision.feature_type: {feature_type}")

        head_type = cfg.get("head_type") or "linear"
        if head_type not in {"linear", "mlp1"}:
            raise ValueError(f"Unsupported loss.task_supervision.head_type: {head_type}")

        dropout = float(cfg.get("dropout", 0.0))
        hidden_dim = int(cfg.get("mlp_hidden_dim", 256))

        if feature_type == "molecular_proj":
            input_dim = int((self.config.get("student") or {}).get("projection", {}).get("chemberta_dim", 768))
        else:
            input_dim = int((self.config.get("student") or {}).get("smiles_encoder", {}).get("embed_dim", 768))

        if head_type == "linear":
            if dropout > 0:
                head = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(input_dim, output_dim),
                )
            else:
                head = nn.Linear(input_dim, output_dim)
        else:
            head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )

        for module in head.modules():
            if isinstance(module, nn.Linear):
                if module.out_features == output_dim:
                    nn.init.normal_(module.weight, mean=0.0, std=0.01)
                else:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        setattr(self.model, "p4_task_head", head)
        head.to(self.device)

        self.task_supervision_head = head

        raw_task_weights = cfg.get("task_weights") or {}
        weights: List[float] = []
        if task_names and isinstance(raw_task_weights, dict) and raw_task_weights:
            for name in task_names:
                w = raw_task_weights.get(name, 1.0)
                weights.append(float(w))
        else:
            weights = [1.0 for _ in range(output_dim)]

        weight_sum = float(sum(weights))
        if weight_sum <= 0:
            weights = [1.0 / max(1, output_dim) for _ in range(output_dim)]
        else:
            weights = [w / weight_sum for w in weights]
        self.task_supervision_alpha = torch.tensor(weights, dtype=torch.float32, device=self.device)

        norm_cfg = cfg.get("normalization") or {}
        self.task_supervision_norm = None
        if loss_type == "bce" and isinstance(norm_cfg, dict) and bool(norm_cfg.get("enabled", False)):
            raise ValueError("loss.task_supervision.normalization is not supported for BCE fingerprint supervision.")

        if loss_type == "mse" and isinstance(norm_cfg, dict) and bool(norm_cfg.get("enabled", False)):
            norm_type = str(norm_cfg.get("type") or "zscore").lower()
            if norm_type != "zscore":
                raise ValueError(f"Unsupported loss.task_supervision.normalization.type: {norm_type}")

            train_path = ((self.config.get("data") or {}).get("paths") or {}).get("train_data")
            if not train_path:
                raise KeyError("loss.task_supervision.normalization enabled but data.paths.train_data is missing.")

            df = pd.read_csv(train_path, usecols=task_names)
            means: List[float] = []
            stds: List[float] = []
            for col in task_names:
                s = df[col].astype("float64")
                s = s.dropna()
                if len(s) == 0:
                    means.append(0.0)
                    stds.append(1.0)
                    continue
                mean = float(s.mean())
                std = float(s.std(ddof=0))
                if not (std > 0):
                    std = 1.0
                means.append(mean)
                stds.append(std)

            self.task_supervision_norm = {
                "type": "zscore",
                "mean": torch.tensor(means, dtype=torch.float32, device=self.device),
                "std": torch.tensor(stds, dtype=torch.float32, device=self.device),
            }

        # Attach task supervision metadata to the head for evaluator usage.
        setattr(head, "_task_feature_type", feature_type)
        setattr(head, "_task_loss_type", loss_type)
        setattr(head, "_task_alpha", self.task_supervision_alpha)
        setattr(head, "_task_norm", self.task_supervision_norm)
        setattr(head, "_task_pos_weight", self.task_supervision_pos_weight)

        print(
            "[TaskSup] Task supervision enabled: "
            f"label_source={label_source or 'csv'}, loss_type={loss_type}, "
            f"feature_type={feature_type}, head_type={head_type}, "
            f"tasks={output_dim}, input_dim={input_dim}, dropout={dropout}, hidden_dim={hidden_dim}"
        )
        if is_fingerprint:
            print(f"[TaskSup] fingerprint bits={output_dim}")
        else:
            print(f"[TaskSup] tasks={task_names}")
            print(f"[TaskSup] task_weights(normed)={dict(zip(task_names, weights))}")
        if self.task_supervision_norm is not None:
            mean = self.task_supervision_norm["mean"].detach().cpu().tolist()
            std = self.task_supervision_norm["std"].detach().cpu().tolist()
            print(f"[TaskSup] normalization=zscore (train) mean/std:")
            for name, m, s in zip(task_names, mean, std):
                print(f"  - {name}: mean={m:.6g}, std={s:.6g}")

    def _init_task_supervision_aux(self) -> None:
        cfg = (self.config.get("loss") or {}).get("task_supervision_aux") or {}
        if not isinstance(cfg, dict):
            cfg = {}

        self.task_supervision_aux_cfg = cfg
        enabled = bool(cfg.get("enabled", False))
        if not enabled:
            self.task_supervision_aux_head = None
            self.task_supervision_aux_tasks = []
            self.task_supervision_aux_alpha = None
            self.task_supervision_aux_norm = None
            self.task_supervision_aux_loss_type = "mse"
            self.task_supervision_aux_pos_weight = None
            self.task_supervision_aux_output_dim = 0
            return

        label_source = str(cfg.get("label_source") or "").strip().lower()
        is_fingerprint = label_source in {"morgan_fp", "morgan", "ecfp", "fingerprint", "fp"}

        loss_type = str(cfg.get("loss_type") or ("bce" if is_fingerprint else "mse")).strip().lower()
        if loss_type not in {"mse", "bce"}:
            raise ValueError(f"Unsupported loss.task_supervision_aux.loss_type: {loss_type}")
        self.task_supervision_aux_loss_type = loss_type
        self.task_supervision_aux_pos_weight = None

        task_names: List[str] = []
        if is_fingerprint:
            fp_cfg = cfg.get("fingerprint") or {}
            if not isinstance(fp_cfg, dict):
                fp_cfg = {}
            output_dim = int(fp_cfg.get("n_bits", 1024))
            if output_dim <= 0:
                raise ValueError(f"loss.task_supervision_aux.fingerprint.n_bits must be > 0, got {output_dim}")
        else:
            tasks = cfg.get("tasks")
            if isinstance(tasks, (list, tuple)) and tasks:
                task_names = [str(t) for t in tasks]
            elif isinstance(tasks, str) and tasks.strip():
                task_names = [tasks.strip()]
            else:
                label_column = cfg.get("label_column") or "Permeability"
                task_names = [str(label_column)]
            output_dim = len(task_names)

        self.task_supervision_aux_tasks = task_names
        self.task_supervision_aux_output_dim = int(output_dim)

        feature_type = cfg.get("feature_type") or "molecular_proj"
        if feature_type not in {"molecular_proj", "smiles_repr", "smiles_mean_pool"}:
            raise ValueError(f"Unsupported loss.task_supervision_aux.feature_type: {feature_type}")

        head_type = cfg.get("head_type") or "linear"
        if head_type not in {"linear", "mlp1"}:
            raise ValueError(f"Unsupported loss.task_supervision_aux.head_type: {head_type}")

        dropout = float(cfg.get("dropout", 0.0))
        hidden_dim = int(cfg.get("mlp_hidden_dim", 256))

        if feature_type == "molecular_proj":
            input_dim = int((self.config.get("student") or {}).get("projection", {}).get("chemberta_dim", 768))
        else:
            input_dim = int((self.config.get("student") or {}).get("smiles_encoder", {}).get("embed_dim", 768))

        if head_type == "linear":
            if dropout > 0:
                head = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(input_dim, output_dim),
                )
            else:
                head = nn.Linear(input_dim, output_dim)
        else:
            head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )

        for module in head.modules():
            if isinstance(module, nn.Linear):
                if module.out_features == output_dim:
                    nn.init.normal_(module.weight, mean=0.0, std=0.01)
                else:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        setattr(self.model, "p4_task_head_aux", head)
        head.to(self.device)

        self.task_supervision_aux_head = head

        raw_task_weights = cfg.get("task_weights") or {}
        weights: List[float] = []
        if task_names and isinstance(raw_task_weights, dict) and raw_task_weights:
            for name in task_names:
                w = raw_task_weights.get(name, 1.0)
                weights.append(float(w))
        else:
            weights = [1.0 for _ in range(output_dim)]

        weight_sum = float(sum(weights))
        if weight_sum <= 0:
            weights = [1.0 / max(1, output_dim) for _ in range(output_dim)]
        else:
            weights = [w / weight_sum for w in weights]
        self.task_supervision_aux_alpha = torch.tensor(weights, dtype=torch.float32, device=self.device)

        norm_cfg = cfg.get("normalization") or {}
        self.task_supervision_aux_norm = None
        if loss_type == "bce" and isinstance(norm_cfg, dict) and bool(norm_cfg.get("enabled", False)):
            raise ValueError("loss.task_supervision_aux.normalization is not supported for BCE fingerprint supervision.")

        if loss_type == "mse" and isinstance(norm_cfg, dict) and bool(norm_cfg.get("enabled", False)):
            norm_type = str(norm_cfg.get("type") or "zscore").lower()
            if norm_type != "zscore":
                raise ValueError(f"Unsupported loss.task_supervision_aux.normalization.type: {norm_type}")

            train_path = ((self.config.get("data") or {}).get("paths") or {}).get("train_data")
            if not train_path:
                raise KeyError("loss.task_supervision_aux.normalization enabled but data.paths.train_data is missing.")

            df = pd.read_csv(train_path, usecols=task_names)
            means: List[float] = []
            stds: List[float] = []
            for col in task_names:
                s = df[col].astype("float64")
                s = s.dropna()
                if len(s) == 0:
                    means.append(0.0)
                    stds.append(1.0)
                    continue
                mean = float(s.mean())
                std = float(s.std(ddof=0))
                if not (std > 0):
                    std = 1.0
                means.append(mean)
                stds.append(std)

            self.task_supervision_aux_norm = {
                "type": "zscore",
                "mean": torch.tensor(means, dtype=torch.float32, device=self.device),
                "std": torch.tensor(stds, dtype=torch.float32, device=self.device),
            }

        # Attach task supervision metadata to the head for evaluator usage.
        setattr(head, "_task_feature_type", feature_type)
        setattr(head, "_task_loss_type", loss_type)
        setattr(head, "_task_alpha", self.task_supervision_aux_alpha)
        setattr(head, "_task_norm", self.task_supervision_aux_norm)
        setattr(head, "_task_pos_weight", None)

        print(
            "[TaskSupAux] Task supervision enabled: "
            f"label_source={label_source or 'csv'}, loss_type={loss_type}, "
            f"feature_type={feature_type}, head_type={head_type}, "
            f"tasks={output_dim}, input_dim={input_dim}, dropout={dropout}, hidden_dim={hidden_dim}"
        )
        if is_fingerprint:
            print(f"[TaskSupAux] fingerprint bits={output_dim}")
        else:
            print(f"[TaskSupAux] tasks={task_names}")
            print(f"[TaskSupAux] task_weights(normed)={dict(zip(task_names, weights))}")
        if self.task_supervision_aux_norm is not None:
            mean = self.task_supervision_aux_norm["mean"].detach().cpu().tolist()
            std = self.task_supervision_aux_norm["std"].detach().cpu().tolist()
            print(f"[TaskSupAux] normalization=zscore (train) mean/std:")
            for name, m, s in zip(task_names, mean, std):
                print(f"  - {name}: mean={m:.6g}, std={s:.6g}")

    def _maybe_freeze_smiles_encoder(self) -> None:
        cfg = (self.config.get("training") or {}).get("freeze_smiles_encoder") or {}
        if not cfg.get("enabled", False):
            return

        freeze_embedding = bool(cfg.get("freeze_embedding", True))
        freeze_layers = int(cfg.get("freeze_layers", 0))
        freeze_final_norm = bool(cfg.get("freeze_final_norm", False))
        freeze_epochs = cfg.get("freeze_epochs", None)

        enc = getattr(self.model, "smiles_encoder", None)
        if enc is None:
            print("[Freeze] Warning: model has no smiles_encoder; skip freezing.")
            return

        num_layers = getattr(enc, "num_layers", None)
        if num_layers is None:
            num_layers = len(getattr(enc, "layers", []))

        if freeze_layers < 0:
            freeze_layers = 0
        if num_layers is not None:
            freeze_layers = min(freeze_layers, int(num_layers))

        frozen_params = 0
        frozen_tensors = 0

        def _freeze_module(module: nn.Module) -> None:
            nonlocal frozen_params, frozen_tensors
            for p in module.parameters(recurse=True):
                frozen_tensors += 1
                frozen_params += p.numel()
                p.requires_grad = False

        if freeze_embedding and hasattr(enc, "embedding"):
            _freeze_module(enc.embedding)

        if freeze_layers > 0 and hasattr(enc, "layers"):
            for layer in enc.layers[:freeze_layers]:
                _freeze_module(layer)

        if freeze_final_norm and hasattr(enc, "final_norm"):
            _freeze_module(enc.final_norm)

        # 记录冻结配置（用于两段式冻结：训练中自动解冻）
        self._freeze_smiles_encoder_cfg = {
            "freeze_embedding": freeze_embedding,
            "freeze_layers": freeze_layers,
            "freeze_final_norm": freeze_final_norm,
            "freeze_epochs": freeze_epochs,
        }
        self._smiles_encoder_unfrozen = False

        print(
            "[Freeze] SMILES encoder frozen: "
            f"embedding={freeze_embedding}, layers=0..{freeze_layers - 1 if freeze_layers else -1} "
            f"(total_layers={num_layers}), final_norm={freeze_final_norm}; "
            f"frozen_tensors={frozen_tensors}, frozen_params={frozen_params / 1e6:.2f}M"
        )
        if freeze_epochs is not None:
            try:
                freeze_epochs_int = int(freeze_epochs)
            except Exception:  # noqa: BLE001
                freeze_epochs_int = None

            if freeze_epochs_int is None:
                print("[Freeze] Two-stage enabled: freeze_epochs is set but not an int; will unfreeze at epoch 1 start.")
            elif freeze_epochs_int <= 0:
                print("[Freeze] Two-stage enabled: freeze_epochs<=0; will unfreeze immediately at epoch 1 start.")
            else:
                print(f"[Freeze] Two-stage enabled: will unfreeze at epoch {freeze_epochs_int + 1} start.")

    def _maybe_unfreeze_smiles_encoder(self, epoch: int) -> None:
        cfg = getattr(self, "_freeze_smiles_encoder_cfg", None) or {}
        freeze_epochs = cfg.get("freeze_epochs", None)
        if freeze_epochs is None:
            return

        try:
            freeze_epochs_int = int(freeze_epochs)
        except Exception:  # noqa: BLE001
            freeze_epochs_int = 0

        if epoch < freeze_epochs_int:
            return
        if getattr(self, "_smiles_encoder_unfrozen", False):
            return

        enc = getattr(self.model, "smiles_encoder", None)
        if enc is None:
            print("[Freeze] Warning: model has no smiles_encoder; skip unfreezing.")
            self._smiles_encoder_unfrozen = True
            return

        freeze_embedding = bool(cfg.get("freeze_embedding", True))
        freeze_layers = int(cfg.get("freeze_layers", 0))
        freeze_final_norm = bool(cfg.get("freeze_final_norm", False))

        unfrozen_params = 0
        unfrozen_tensors = 0

        def _unfreeze_module(module: nn.Module) -> None:
            nonlocal unfrozen_params, unfrozen_tensors
            for p in module.parameters(recurse=True):
                if p.requires_grad:
                    continue
                unfrozen_tensors += 1
                unfrozen_params += p.numel()
                p.requires_grad = True

        if freeze_embedding and hasattr(enc, "embedding"):
            _unfreeze_module(enc.embedding)

        if freeze_layers > 0 and hasattr(enc, "layers"):
            for layer in enc.layers[:freeze_layers]:
                _unfreeze_module(layer)

        if freeze_final_norm and hasattr(enc, "final_norm"):
            _unfreeze_module(enc.final_norm)

        self._smiles_encoder_unfrozen = True
        print(
            "[Freeze] SMILES encoder unfrozen (two-stage): "
            f"embedding={freeze_embedding}, layers=0..{freeze_layers - 1 if freeze_layers else -1} "
            f"final_norm={freeze_final_norm}; unfrozen_tensors={unfrozen_tensors}, "
            f"unfrozen_params={unfrozen_params / 1e6:.2f}M"
        )

    def _init_dataloaders(self):
        """初始化数据加载器"""
        print("\n[Init] Creating dataloaders...")
        self.dataloaders, self.smiles_vocab, _ = build_dataloaders(self.config)

    def _maybe_init_task_supervision_pos_weight(self) -> None:
        cfg = getattr(self, "task_supervision_cfg", {}) or {}
        if not cfg.get("enabled", False):
            return

        loss_type = str(getattr(self, "task_supervision_loss_type", "mse")).lower()
        if loss_type != "bce":
            return

        bce_cfg = cfg.get("bce") or {}
        if not isinstance(bce_cfg, dict):
            bce_cfg = {}
        use_pos_weight = bool(bce_cfg.get("use_pos_weight", True))
        if not use_pos_weight:
            return

        train_loader = (getattr(self, "dataloaders", {}) or {}).get("train")
        if train_loader is None:
            return

        ds = getattr(train_loader, "dataset", None)
        pos_weight = getattr(ds, "task_pos_weight", None)
        if pos_weight is None:
            print("[TaskSup] BCE pos_weight not found on dataset; will run without pos_weight.")
            return

        if not torch.is_tensor(pos_weight):
            pos_weight = torch.tensor(pos_weight, dtype=torch.float32)

        output_dim = int(getattr(self, "task_supervision_output_dim", pos_weight.numel()))
        if int(pos_weight.numel()) != int(output_dim):
            raise ValueError(
                f"task_pos_weight dim mismatch: got {int(pos_weight.numel())}, expected {output_dim}."
            )

        self.task_supervision_pos_weight = pos_weight.to(self.device, non_blocking=True)
        head = getattr(self.model, "p4_task_head", None)
        if head is not None:
            setattr(head, "_task_pos_weight", self.task_supervision_pos_weight)
        print(f"[TaskSup] BCE pos_weight enabled (dim={output_dim})")

    def _maybe_init_task_supervision_aux_pos_weight(self) -> None:
        cfg = getattr(self, "task_supervision_aux_cfg", {}) or {}
        if not cfg.get("enabled", False):
            return

        loss_type = str(getattr(self, "task_supervision_aux_loss_type", "mse")).lower()
        if loss_type != "bce":
            return

        bce_cfg = cfg.get("bce") or {}
        if not isinstance(bce_cfg, dict):
            bce_cfg = {}
        use_pos_weight = bool(bce_cfg.get("use_pos_weight", True))
        if not use_pos_weight:
            return

        train_loader = (getattr(self, "dataloaders", {}) or {}).get("train")
        if train_loader is None:
            return

        ds = getattr(train_loader, "dataset", None)
        pos_weight = getattr(ds, "task_aux_pos_weight", None)
        if pos_weight is None:
            print("[TaskSupAux] BCE pos_weight not found on dataset; will run without pos_weight.")
            return

        if not torch.is_tensor(pos_weight):
            pos_weight = torch.tensor(pos_weight, dtype=torch.float32)

        output_dim = int(getattr(self, "task_supervision_aux_output_dim", pos_weight.numel()))
        if int(pos_weight.numel()) != int(output_dim):
            raise ValueError(
                f"task_aux_pos_weight dim mismatch: got {int(pos_weight.numel())}, expected {output_dim}."
            )

        self.task_supervision_aux_pos_weight = pos_weight.to(self.device, non_blocking=True)
        head = getattr(self.model, "p4_task_head_aux", None)
        if head is not None:
            setattr(head, "_task_pos_weight", self.task_supervision_aux_pos_weight)
        print(f"[TaskSupAux] BCE pos_weight enabled (dim={output_dim})")

    def _init_teacher_caches(self):
        """初始化教师特征缓存"""
        print("\n[Init] Loading teacher feature caches...")
        self.teacher_caches = {}

        teacher_config = self.config.get('teacher', {})

        for name in ['esm2', 'chemberta']:
            cfg = teacher_config.get(name, {})
            if not cfg.get('use_feature_cache'):
                continue

            cache_path = cfg.get('feature_cache_path')
            if not cache_path:
                continue

            try:
                id_keys = cfg.get('id_keys')
                if not id_keys:
                    # 默认键：ESM2 用 sequence_id；ChemBERTa 用 row_index / sample_ids
                    if name == 'esm2':
                        id_keys = ['sequence_ids', 'sequence_id']
                    else:
                        id_keys = ['sample_ids', 'row_index', 'row_indices']
                self.teacher_caches[name] = TeacherFeatureCacheManager(cache_path, id_keys=id_keys)
                print(f"  {name}: {cache_path}")
            except Exception as e:
                print(f"  Warning: Failed to load {name} cache: {e}")

    def _init_optimizer(self):
        """初始化优化器和调度器"""
        print("\n[Init] Creating optimizer and scheduler...")

        opt_config = self.config['training']['optimizer']

        optimizer_type = str(opt_config.get("type", "adamw")).lower()
        base_lr = float(opt_config['lr'])
        weight_decay = float(opt_config.get('weight_decay', 0.01))
        betas = opt_config.get('betas', [0.9, 0.999])

        # 当启用“两段式冻结→解冻”时，冻结参数也需要进入 optimizer param_groups，
        # 否则解冻后不会被更新（optimizer 里没有这些 params）。
        freeze_cfg = (self.config.get("training") or {}).get("freeze_smiles_encoder") or {}
        include_frozen_in_optim = bool(freeze_cfg.get("enabled", False)) and freeze_cfg.get("freeze_epochs") is not None

        # 可选：分组学习率（discriminative lr）
        dlr_cfg = opt_config.get("discriminative_lr") or {}
        dlr_enabled = bool(dlr_cfg.get("enabled", False))
        backbone_lr_scale = float(dlr_cfg.get("backbone_lr_scale", 0.1))

        if optimizer_type == "adamw":
            if dlr_enabled:
                backbone_params = []
                head_params = []
                for name, p in self.model.named_parameters():
                    if not p.requires_grad and not include_frozen_in_optim:
                        continue
                    if name.startswith("smiles_encoder."):
                        backbone_params.append(p)
                    else:
                        head_params.append(p)

                if not backbone_params:
                    print("[DLR] Warning: no trainable backbone params found; fallback to single group.")
                    param_groups = [p for p in self.model.parameters() if p.requires_grad]
                else:
                    param_groups = [
                        {"params": backbone_params, "lr": base_lr * backbone_lr_scale},
                        {"params": head_params, "lr": base_lr},
                    ]

                self.optimizer = optim.AdamW(
                    param_groups,
                    lr=base_lr,
                    weight_decay=weight_decay,
                    betas=betas,
                )
                print(f"  Discriminative LR: enabled (backbone_lr_scale={backbone_lr_scale})")
            else:
                if include_frozen_in_optim:
                    trainable_params = list(self.model.parameters())
                else:
                    trainable_params = [p for p in self.model.parameters() if p.requires_grad]
                self.optimizer = optim.AdamW(
                    trainable_params,
                    lr=base_lr,
                    weight_decay=weight_decay,
                    betas=betas,
                )
        elif optimizer_type == "muon":
            if dlr_enabled:
                print("[Muon] Warning: discriminative_lr not supported for Muon; will ignore.")

            from utils.muon_optimizer import Muon

            muon_cfg = opt_config.get("muon") or {}
            momentum = float(muon_cfg.get("momentum", 0.95))
            nesterov = bool(muon_cfg.get("nesterov", True))
            ns_steps = int(muon_cfg.get("ns_steps", 5))
            muon_lr_scale = float(muon_cfg.get("muon_lr_scale", 0.2))
            adamw_betas = muon_cfg.get("adamw_betas", (0.9, 0.95))
            adamw_eps = float(muon_cfg.get("adamw_eps", 1e-8))

            exclude_names = muon_cfg.get(
                "exclude_names",
                ["embedding", "smiles_mlm_head", "task_head", "p4_task_head", "p4_task_head_aux"],
            )
            if isinstance(exclude_names, str):
                exclude_names = [exclude_names]

            muon_params = []
            adamw_params = []
            for name, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                use_muon = p.ndim == 2 and not any(ex in name for ex in exclude_names)
                if use_muon:
                    muon_params.append(p)
                else:
                    adamw_params.append(p)

            self.optimizer = Muon(
                lr=base_lr,
                weight_decay=weight_decay,
                muon_params=muon_params,
                adamw_params=adamw_params,
                momentum=momentum,
                nesterov=nesterov,
                ns_steps=ns_steps,
                muon_lr_scale=muon_lr_scale,
                adamw_betas=adamw_betas,
                adamw_eps=adamw_eps,
            )
            print(
                f"  Muon enabled: muon_params={len(muon_params)}, adamw_params={len(adamw_params)}, "
                f"ns_steps={ns_steps}, muon_lr_scale={muon_lr_scale}"
            )
        else:
            raise ValueError(f"Unsupported optimizer.type: {optimizer_type}")

        # 学习率调度器
        scheduler_config = opt_config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'cosine')
        warmup_steps = scheduler_config.get('warmup_steps', 5000)
        min_lr = scheduler_config.get('min_lr', 1e-6)

        num_training_steps = len(self.dataloaders['train']) * self.config['training']['epochs']

        if scheduler_type == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

            # Warmup scheduler
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_steps
            )

            # Cosine scheduler
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=num_training_steps - warmup_steps,
                eta_min=min_lr
            )

            # Sequential scheduler
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps]
            )
        else:
            self.scheduler = None

        print(f"  Optimizer: {optimizer_type} (lr={base_lr}, weight_decay={weight_decay})")
        print(f"  Scheduler: {scheduler_type} (warmup={warmup_steps})")

        # 混合精度训练
        self.use_amp = self.config['training'].get('mixed_precision', {}).get('enabled', False)
        self.use_grad_scaler = self.use_amp and self.config['training'].get('mixed_precision', {}).get('use_grad_scaler', False)
        if self.use_amp:
            if self.use_grad_scaler:
                self.scaler = torch.cuda.amp.GradScaler()
                print(f"  Mixed Precision: Enabled (bf16) with GradScaler")
            else:
                self.scaler = None
                print(f"  Mixed Precision: Enabled (bf16) without GradScaler")
        else:
            self.scaler = None

    def _init_ema(self):
        """初始化 EMA"""
        ema_config = self.config['training'].get('ema', {})
        if ema_config.get('enabled', False):
            self.ema = ExponentialMovingAverage(
                self.model,
                decay=ema_config.get('decay', 0.999)
            )
            print(f"\n[Init] EMA enabled (decay={ema_config.get('decay', 0.999)})")
        else:
            self.ema = None

    def _resolve_loss_weights(self, epoch: int) -> Dict[str, float]:
        """根据当前 epoch 计算损失权重（支持渐进式）"""
        loss_weights_cfg = self.config['loss']['weights']
        if isinstance(loss_weights_cfg, dict) and loss_weights_cfg.get('type') == 'progressive':
            stages = loss_weights_cfg.get('stages', [])
            for stage in stages:
                start, end = stage['epochs']
                if start <= epoch < end:
                    return stage['weights']
            return stages[-1]['weights'] if stages else {}
        return loss_weights_cfg

    def _maybe_resume_checkpoint(self):
        """从 checkpoint 恢复训练状态"""
        resume_path = self.config['training'].get('resume_from')
        if not resume_path:
            return

        resume_path = Path(resume_path)
        if not resume_path.exists():
            print(f"[Resume] Warning: resume checkpoint {resume_path} not found, skip.")
            return

        print(f"\n[Resume] Loading checkpoint from {resume_path} ...")
        checkpoint = torch.load(resume_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.global_step = checkpoint.get('global_step', 0)
        self.current_epoch = checkpoint.get('epoch', 0)
        self.start_epoch = self.current_epoch + 1
        self.best_metric = checkpoint.get('best_metric', -inf)

        if self.ema and 'ema_state_dict' in checkpoint:
            # 直接覆盖 shadow_params
            shadow_params = checkpoint['ema_state_dict']
            self.ema.shadow_params = {k: v.to(self.device) for k, v in shadow_params.items()}

        print(f"[Resume] Resumed at epoch {self.current_epoch+1}, global_step {self.global_step}, best_metric {self.best_metric:.4f}")

    def _load_teacher_features(
        self,
        batch: Dict,
        split_name: str
    ) -> Dict[str, torch.Tensor]:
        """
        加载教师特征（从缓存）

        Args:
            batch: 数据批次
            split_name: 'train' or 'val'

        Returns:
            teacher_features: {'esm2': [batch, 640], 'chemberta': [batch, 768]}
        """
        teacher_features = {}
        sequence_ids = batch['sequence_id']
        row_indices = batch.get('row_index')  # For ChemBERTa

        # 加载 ESM2 特征（使用 sequence_id）
        if 'esm2' in self.teacher_caches:
            esm2_feats, missing = self.teacher_caches['esm2'].get(split_name, sequence_ids)
            if esm2_feats is not None:
                teacher_features['esm2'] = esm2_feats.to(self.device)
            else:
                print(f"Warning: ESM2 cache miss for {len(missing)} samples")

        # 加载 ChemBERTa/Geminimol 特征
        if 'chemberta' in self.teacher_caches:
            chem_cache = self.teacher_caches['chemberta']
            # 根据cache的id_keys选择使用sequence_id还是row_index
            use_sequence_ids = any(k in chem_cache.id_keys for k in ['sequence_ids', 'sequence_id'])

            if use_sequence_ids:
                chem_ids = sequence_ids
            else:
                if row_indices is None:
                    chem_ids = []
                else:
                    chem_ids = [str(idx.item()) if torch.is_tensor(idx) else str(idx) for idx in row_indices]

            if chem_ids:
                chemberta_feats, missing = chem_cache.get(split_name, chem_ids)
                if chemberta_feats is not None:
                    teacher_features['chemberta'] = chemberta_feats.to(self.device)
                else:
                    print(f"Warning: ChemBERTa cache miss for {len(missing)} samples")

        return teacher_features

    def train_epoch(self, epoch: int):
        """训练一个 epoch"""
        self.model.train()

        total_loss = 0.0
        total_protein_loss = 0.0
        total_molecular_loss = 0.0
        total_self_loss = 0.0
        total_rkd_loss = 0.0
        total_contrastive_loss = 0.0
        total_consistency_loss = 0.0
        total_task_loss = 0.0
        total_task_fp_loss = 0.0
        total_fusion_loss = 0.0
        num_batches = 0

        gradient_accumulation_steps = self.config['training'].get('gradient_accumulation_steps', 1)
        log_interval = self.config['training'].get('log_interval', 100)
        max_grad_norm = self.config['training'].get('max_grad_norm', 1.0)

        # 损失权重（考虑渐进式）
        loss_weights = self._resolve_loss_weights(epoch)
        print(f"[Epoch {epoch+1}] Using loss weights: {loss_weights}")
        task_weight = float(loss_weights.get("task", 0.0))
        task_fp_weight = float(loss_weights.get("task_fp", 0.0))
        consistency_weight = float(loss_weights.get("consistency", 0.0))
        fusion_weight = float(self._resolve_fusion_weight(epoch))
        if fusion_weight > 0:
            cfg = self._get_fusion_cfg()
            print(
                f"[Epoch {epoch+1}] Fusion enabled: mode={cfg.get('mode','residual_distill')}, "
                f"target={cfg.get('target','geminimol')}, weight={fusion_weight}, "
                f"stop_grad_hs={cfg.get('stop_grad_hs', True)}, freeze_smiles={cfg.get('freeze_smiles', False)}"
            )

        start_time = time.time()

        for batch_idx, batch in enumerate(self.dataloaders['train']):
            # 移动到设备
            smiles_input_ids = batch['smiles_input_ids'].to(self.device)
            smiles_attention_mask = batch['smiles_attention_mask'].to(self.device)
            smiles_mlm_labels = batch.get('smiles_mlm_labels')
            if smiles_mlm_labels is not None:
                smiles_mlm_labels = smiles_mlm_labels.to(self.device)
            graph_distances = batch.get("graph_distances")
            if graph_distances is not None:
                graph_distances = graph_distances.to(self.device, non_blocking=True)

            # 加载教师特征
            teacher_features = self._load_teacher_features(batch, split_name='train')

            # 混合精度前向传播和损失计算
            if self.use_amp:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    student_outputs = self.model(
                        smiles_input_ids,
                        smiles_attention_mask,
                        graph_distances=graph_distances,
                        return_all=True,
                    )
                    losses = compute_dual_teacher_loss(
                        student_outputs=student_outputs,
                        teacher_esm2_features=teacher_features.get('esm2'),
                        teacher_chemberta_features=teacher_features.get('chemberta'),
                        weights=loss_weights,
                        mlm_labels=smiles_mlm_labels,
                        vocab_size=len(self.smiles_vocab),
                        loss_cfg=self.config.get("loss"),
                    )
                    consistency_loss = torch.tensor(0.0, device=self.device)
                    if consistency_weight > 0:
                        view2_ids = batch.get("smiles_input_ids_view2")
                        view2_mask = batch.get("smiles_attention_mask_view2")
                        if view2_ids is not None and view2_mask is not None:
                            view2_ids = view2_ids.to(self.device, non_blocking=True)
                            view2_mask = view2_mask.to(self.device, non_blocking=True)
                            view2_graph = batch.get("graph_distances_view2")
                            if view2_graph is not None:
                                view2_graph = view2_graph.to(self.device, non_blocking=True)
                            outputs_view2 = self.model(
                                view2_ids,
                                view2_mask,
                                graph_distances=view2_graph,
                                return_all=False,
                            )
                            consistency_loss = self._compute_consistency_loss(student_outputs, outputs_view2)
                    task_loss = self._compute_task_supervision_loss(student_outputs, batch, task_weight)
                    task_fp_loss = self._compute_task_supervision_aux_loss(student_outputs, batch, task_fp_weight)
                    fusion_loss = self._compute_fusion_residual_distill_loss(student_outputs, teacher_features, fusion_weight)
                    losses["task"] = task_loss
                    losses["task_fp"] = task_fp_loss
                    losses["consistency"] = consistency_loss
                    losses["fusion"] = fusion_loss
                    losses["total"] = (
                        losses["total"]
                        + task_weight * task_loss
                        + task_fp_weight * task_fp_loss
                        + consistency_weight * consistency_loss
                        + fusion_weight * fusion_loss
                    )
                loss = losses['total'] / gradient_accumulation_steps
                if self.use_grad_scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
            else:
                student_outputs = self.model(
                    smiles_input_ids,
                    smiles_attention_mask,
                    graph_distances=graph_distances,
                    return_all=True,
                )
                losses = compute_dual_teacher_loss(
                    student_outputs=student_outputs,
                    teacher_esm2_features=teacher_features.get('esm2'),
                    teacher_chemberta_features=teacher_features.get('chemberta'),
                    weights=loss_weights,
                    mlm_labels=smiles_mlm_labels,
                    vocab_size=len(self.smiles_vocab),
                    loss_cfg=self.config.get("loss"),
                )
                consistency_loss = torch.tensor(0.0, device=self.device)
                if consistency_weight > 0:
                    view2_ids = batch.get("smiles_input_ids_view2")
                    view2_mask = batch.get("smiles_attention_mask_view2")
                    if view2_ids is not None and view2_mask is not None:
                        view2_ids = view2_ids.to(self.device, non_blocking=True)
                        view2_mask = view2_mask.to(self.device, non_blocking=True)
                        view2_graph = batch.get("graph_distances_view2")
                        if view2_graph is not None:
                            view2_graph = view2_graph.to(self.device, non_blocking=True)
                        outputs_view2 = self.model(
                            view2_ids,
                            view2_mask,
                            graph_distances=view2_graph,
                            return_all=False,
                        )
                        consistency_loss = self._compute_consistency_loss(student_outputs, outputs_view2)
                task_loss = self._compute_task_supervision_loss(student_outputs, batch, task_weight)
                task_fp_loss = self._compute_task_supervision_aux_loss(student_outputs, batch, task_fp_weight)
                fusion_loss = self._compute_fusion_residual_distill_loss(student_outputs, teacher_features, fusion_weight)
                losses["task"] = task_loss
                losses["task_fp"] = task_fp_loss
                losses["consistency"] = consistency_loss
                losses["fusion"] = fusion_loss
                losses["total"] = (
                    losses["total"]
                    + task_weight * task_loss
                    + task_fp_weight * task_fp_loss
                    + consistency_weight * consistency_loss
                    + fusion_weight * fusion_loss
                )
                loss = losses['total'] / gradient_accumulation_steps
                loss.backward()

            # 累积统计
            total_loss += losses['total'].item()
            total_protein_loss += losses['protein'].item()
            total_molecular_loss += losses['molecular'].item()
            total_self_loss += losses.get('self', torch.tensor(0.0)).item()
            total_rkd_loss += losses.get("rkd", torch.tensor(0.0)).item()
            total_contrastive_loss += losses.get("contrastive", torch.tensor(0.0)).item()
            total_consistency_loss += losses.get("consistency", torch.tensor(0.0)).item()
            total_task_loss += losses.get('task', torch.tensor(0.0)).item()
            total_task_fp_loss += losses.get("task_fp", torch.tensor(0.0)).item()
            total_fusion_loss += losses.get("fusion", torch.tensor(0.0)).item()
            num_batches += 1

            # 优化器更新
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if self.use_amp and self.use_grad_scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    self.optimizer.step()

                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad()

                # EMA 更新
                if self.ema:
                    self.ema.update(self.model)

                self.global_step += 1

                # 日志输出
                if self.global_step % log_interval == 0:
                    avg_loss = total_loss / num_batches
                    avg_protein = total_protein_loss / num_batches
                    avg_molecular = total_molecular_loss / num_batches
                    avg_self = total_self_loss / num_batches
                    avg_rkd = total_rkd_loss / num_batches
                    avg_ctr = total_contrastive_loss / num_batches
                    avg_cons = total_consistency_loss / num_batches
                    avg_task = total_task_loss / num_batches
                    avg_task_fp = total_task_fp_loss / num_batches
                    avg_fuse = total_fusion_loss / num_batches
                    lr = self.optimizer.param_groups[0]['lr']

                    print(f"[Epoch {epoch+1}] Step {self.global_step}: "
                          f"Loss={avg_loss:.4f} (P={avg_protein:.4f}, M={avg_molecular:.4f}, S={avg_self:.4f}, R={avg_rkd:.4f}, C={avg_ctr:.4f}, K={avg_cons:.4f}, T={avg_task:.4f}, F={avg_task_fp:.4f}, U={avg_fuse:.4f}), "
                          f"LR={lr:.2e}")

        # Epoch 结束
        epoch_loss = total_loss / num_batches
        elapsed = time.time() - start_time

        print(f"\n[Epoch {epoch+1}] Train Loss: {epoch_loss:.4f} ({elapsed:.1f}s)")

        return epoch_loss

    def _compute_task_supervision_loss(
        self,
        student_outputs: Dict[str, torch.Tensor],
        batch: Dict,
        task_weight: float,
    ) -> torch.Tensor:
        cfg = getattr(self, "task_supervision_cfg", {}) or {}
        if task_weight <= 0 or not cfg.get("enabled", False):
            return torch.tensor(0.0, device=self.device)

        head = getattr(self, "task_supervision_head", None)
        if head is None:
            raise RuntimeError("P4 enabled but task_supervision_head is missing.")

        feature_type = cfg.get("feature_type") or "molecular_proj"
        if feature_type not in student_outputs:
            raise KeyError(f"P4 feature_type not in student outputs: {feature_type}")

        x = student_outputs[feature_type]
        if bool(cfg.get("detach_features", False)):
            x = x.detach()

        pred = head(x)
        loss_type = str(getattr(self, "task_supervision_loss_type", "mse")).lower()

        # 多任务：batch['task_labels'] shape [B, T] + batch['task_mask'] shape [B, T]
        if "task_labels" in batch:
            y = batch["task_labels"].to(self.device, non_blocking=True)
            mask = batch.get("task_mask")
            if mask is None:
                mask = torch.isfinite(y).to(dtype=torch.float32)
            else:
                mask = mask.to(self.device, non_blocking=True)

            if pred.ndim == 1:
                pred = pred.unsqueeze(-1)

            if loss_type == "bce":
                y = y.to(dtype=torch.float32).clamp(0.0, 1.0)
                mask = mask.to(dtype=torch.float32)

                bce_cfg = cfg.get("bce") or {}
                if not isinstance(bce_cfg, dict):
                    bce_cfg = {}
                use_pos_weight = bool(bce_cfg.get("use_pos_weight", True))
                pos_weight = getattr(self, "task_supervision_pos_weight", None) if use_pos_weight else None

                per_elem = F.binary_cross_entropy_with_logits(
                    pred.to(dtype=torch.float32),
                    y,
                    pos_weight=pos_weight,
                    reduction="none",
                )
                per_elem = per_elem * mask
                denom = mask.sum(dim=0).clamp_min(1.0)
                per_task = per_elem.sum(dim=0) / denom

                alpha = getattr(self, "task_supervision_alpha", None)
                if alpha is None:
                    return per_task.mean()
                return (per_task * alpha).sum()

            if loss_type != "mse":
                raise ValueError(f"Unsupported task supervision loss_type: {loss_type}")

            norm = getattr(self, "task_supervision_norm", None)
            if norm is not None and norm.get("type") == "zscore":
                mean = norm["mean"]
                std = norm["std"]
                y = (y - mean) / std

            se = (pred - y) ** 2
            se = se * mask
            denom = mask.sum(dim=0).clamp_min(1.0)
            per_task_mse = se.sum(dim=0) / denom

            alpha = getattr(self, "task_supervision_alpha", None)
            if alpha is None:
                return per_task_mse.mean()
            return (per_task_mse * alpha).sum()

        # 单任务（P4 兼容）：batch['task_label'] shape [B]
        if "task_label" not in batch:
            raise KeyError(
                "Task supervision enabled but batch has no 'task_label'/'task_labels'. "
                "Check dataloader config (label_column or tasks)."
            )

        y1 = batch["task_label"].to(self.device, non_blocking=True)
        pred1 = pred.squeeze(-1)
        if loss_type == "bce":
            y1 = y1.to(dtype=torch.float32).clamp(0.0, 1.0)
            return F.binary_cross_entropy_with_logits(pred1.to(dtype=torch.float32), y1)
        return F.mse_loss(pred1, y1)

    def _compute_task_supervision_aux_loss(
        self,
        student_outputs: Dict[str, torch.Tensor],
        batch: Dict,
        task_weight: float,
    ) -> torch.Tensor:
        cfg = getattr(self, "task_supervision_aux_cfg", {}) or {}
        if task_weight <= 0 or not cfg.get("enabled", False):
            return torch.tensor(0.0, device=self.device)

        head = getattr(self, "task_supervision_aux_head", None)
        if head is None:
            raise RuntimeError("task_supervision_aux enabled but task_supervision_aux_head is missing.")

        feature_type = cfg.get("feature_type") or "molecular_proj"
        if feature_type not in student_outputs:
            raise KeyError(f"task_supervision_aux feature_type not in student outputs: {feature_type}")

        x = student_outputs[feature_type]
        if bool(cfg.get("detach_features", False)):
            x = x.detach()

        pred = head(x)
        loss_type = str(getattr(self, "task_supervision_aux_loss_type", "mse")).lower()

        # 多任务：batch['task_labels_aux'] shape [B, T] + batch['task_mask_aux'] shape [B, T]
        if "task_labels_aux" in batch:
            y = batch["task_labels_aux"].to(self.device, non_blocking=True)
            mask = batch.get("task_mask_aux")
            if mask is None:
                mask = torch.isfinite(y).to(dtype=torch.float32)
            else:
                mask = mask.to(self.device, non_blocking=True)

            if pred.ndim == 1:
                pred = pred.unsqueeze(-1)

            if loss_type == "bce":
                y = y.to(dtype=torch.float32).clamp(0.0, 1.0)
                mask = mask.to(dtype=torch.float32)

                bce_cfg = cfg.get("bce") or {}
                if not isinstance(bce_cfg, dict):
                    bce_cfg = {}
                use_pos_weight = bool(bce_cfg.get("use_pos_weight", True))
                pos_weight = getattr(self, "task_supervision_aux_pos_weight", None) if use_pos_weight else None

                per_elem = F.binary_cross_entropy_with_logits(
                    pred.to(dtype=torch.float32),
                    y,
                    pos_weight=pos_weight,
                    reduction="none",
                )
                per_elem = per_elem * mask
                denom = mask.sum(dim=0).clamp_min(1.0)
                per_task = per_elem.sum(dim=0) / denom

                alpha = getattr(self, "task_supervision_aux_alpha", None)
                if alpha is None:
                    return per_task.mean()
                return (per_task * alpha).sum()

            if loss_type != "mse":
                raise ValueError(f"Unsupported task_supervision_aux loss_type: {loss_type}")

            norm = getattr(self, "task_supervision_aux_norm", None)
            if norm is not None and norm.get("type") == "zscore":
                mean = norm["mean"]
                std = norm["std"]
                y = (y - mean) / std

            se = (pred - y) ** 2
            se = se * mask
            denom = mask.sum(dim=0).clamp_min(1.0)
            per_task_mse = se.sum(dim=0) / denom

            alpha = getattr(self, "task_supervision_aux_alpha", None)
            if alpha is None:
                return per_task_mse.mean()
            return (per_task_mse * alpha).sum()

        # 单任务（兼容）：batch['task_label_aux'] shape [B]
        if "task_label_aux" not in batch:
            raise KeyError(
                "task_supervision_aux enabled but batch has no 'task_label_aux'/'task_labels_aux'. "
                "Check dataset/dataloader wiring."
            )

        y1 = batch["task_label_aux"].to(self.device, non_blocking=True)
        pred1 = pred.squeeze(-1)
        if loss_type == "bce":
            y1 = y1.to(dtype=torch.float32).clamp(0.0, 1.0)
            return F.binary_cross_entropy_with_logits(pred1.to(dtype=torch.float32), y1)
        return F.mse_loss(pred1, y1)

    def _compute_consistency_loss(
        self,
        outputs_view1: Dict[str, torch.Tensor],
        outputs_view2: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        cfg = (self.config.get("loss") or {}).get("consistency") or {}
        if not cfg.get("enabled", False):
            return torch.tensor(0.0, device=self.device)

        feature_key = cfg.get("apply_to", "smiles_repr")
        if feature_key not in outputs_view1 or feature_key not in outputs_view2:
            raise KeyError(f"Consistency feature not found: {feature_key}")

        z1 = outputs_view1[feature_key]
        z2 = outputs_view2[feature_key]

        method = str(cfg.get("method", "cosine")).lower()
        if method == "cosine":
            return 1.0 - F.cosine_similarity(z1, z2, dim=1).mean()
        if method == "mse":
            return F.mse_loss(z1, z2)
        if method == "infonce":
            tau = float(cfg.get("tau", 0.07))
            symmetric = bool(cfg.get("symmetric", True))
            return contrastive_infonce_loss(z1, z2, tau=tau, symmetric=symmetric)

        raise ValueError(f"Unsupported consistency.method: {method}")

    @torch.no_grad()
    def validate(self):
        """验证"""
        if 'val' not in self.dataloaders:
            return {}

        print("\n[Validation] Evaluating...")

        current_loss_weights = dict(self._resolve_loss_weights(self.current_epoch) or {})
        current_loss_weights["fusion"] = float(self._resolve_fusion_weight(self.current_epoch))

        use_ema_for_eval = self.ema is not None and self.config['training'].get('ema', {}).get('use_for_eval', True)
        if use_ema_for_eval:
            self.ema.apply_shadow(self.model)

        metrics = self.evaluator.evaluate(
            model=self.model,
            dataloader=self.dataloaders['val'],
            teacher_cache_manager=self.teacher_caches,
            split_name='val',
            loss_weights=current_loss_weights,
            vocab_size=len(self.smiles_vocab),
            use_mlm=self.config['student'].get('use_smiles_mlm', True)
        )

        if use_ema_for_eval:
            self.ema.restore(self.model)

        # 可选：Phase3-val Ridge probe（用 R² 选 ckpt / 早停）
        if self.phase3_probe_cfg and self.phase3_probe_cfg.get("enabled", False):
            try:
                phase3_metrics = self._run_phase3_probe()
                if phase3_metrics:
                    metrics.update(phase3_metrics)
                    print(
                        "\n[Phase3Probe] "
                        f"val_rmse={phase3_metrics['phase3_val_rmse']:.4f} "
                        f"val_r2={phase3_metrics['phase3_val_r2']:.4f} "
                        f"best_alpha={phase3_metrics['phase3_best_alpha']}"
                    )
            except Exception as e:  # noqa: BLE001
                print(f"[Phase3Probe] Warning: failed to compute probe metrics: {e}")

        self.evaluator.print_metrics(metrics, prefix="Val")

        return metrics

    def save_checkpoint(self, name: str):
        """保存 checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"{name}.pt"

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'epoch': self.current_epoch,
            'config': self.config,
            'best_metric': self.best_metric
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if self.use_grad_scaler and self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        if self.ema:
            checkpoint['ema_state_dict'] = {k: v.detach().cpu() for k, v in self.ema.shadow_params.items()}

        # Note: EMA shadow params are not saved separately in this version
        # They are stored in the model's state_dict during training

        torch.save(checkpoint, checkpoint_path)
        print(f"[Checkpoint] Saved: {checkpoint_path}")

    def save_best_model(self, metrics: Dict[str, float]):
        """保存最佳模型"""
        metric_value = metrics.get(self.best_metric_name)
        if metric_value is None:
            print(f"[Best Model] Warning: metric '{self.best_metric_name}' not found in validation metrics.")
            return

        if self.best_from_epoch and (self.current_epoch + 1) < self.best_from_epoch:
            print(
                f"[Best Model] Skip best update at epoch {self.current_epoch+1} "
                f"(best_from_epoch={self.best_from_epoch})"
            )
            return

        improved = (
            metric_value < self.best_metric if self.best_metric_mode == "min" else metric_value > self.best_metric
        )

        if improved:
            self.best_metric = metric_value
            self.save_checkpoint("best")
            print(f"[Best Model] Saved ({self.best_metric_name}={metric_value:.4f}, mode={self.best_metric_mode})")

    def train(self):
        """完整训练流程"""
        num_epochs = self.config['training']['epochs']
        val_interval = self.config['training'].get('val_interval_epochs', 5)
        save_interval = self.config['training'].get('save_interval_epochs', 10)

        print(f"\n{'='*70}")
        print(f"Starting Training: {num_epochs} epochs")
        print(f"{'='*70}\n")

        stopped_early = False
        last_validated_epoch = None  # 1-based epoch index
        for epoch in range(self.start_epoch, num_epochs):
            self.current_epoch = epoch

            # 可选：Phase2 漂移控制（两段式冻结→解冻）
            self._maybe_unfreeze_smiles_encoder(epoch)
            self._maybe_freeze_smiles_encoder_for_fusion(epoch)

            # 训练
            _train_loss = self.train_epoch(epoch)

            # 验证
            if (epoch + 1) % val_interval == 0:
                metrics = self.validate()
                last_validated_epoch = epoch + 1
                if metrics:
                    self.save_best_model(metrics)
                    if self.phase3_early_stopper is not None and "phase3_val_r2" in metrics:
                        if self.phase3_early_stopper.update(float(metrics["phase3_val_r2"]), step=epoch + 1):
                            print(
                                "[Phase3Probe] Early stopping triggered "
                                f"at epoch {epoch + 1} (best_epoch={self.phase3_early_stopper.best_step}, "
                                f"best_phase3_val_r2={self.phase3_early_stopper.best_metric:.4f})"
                            )
                            stopped_early = True
                            break

            # 保存 checkpoint
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(f'epoch_{epoch+1}')

        # 训练结束
        print(f"\n{'='*70}")
        print("Training Completed!" if not stopped_early else "Training Completed (Early Stopped)!")
        print(f"{'='*70}\n")

        # 最终验证：
        # 若最后一个 epoch 已经做过 validate，则无需重复跑一遍（省时间，尤其是 phase3_probe 很重）
        final_metrics = {}
        if last_validated_epoch != (self.current_epoch + 1):
            final_metrics = self.validate()
        self.save_checkpoint('final')

        return final_metrics


def test_trainer():
    """Test trainer (requires full setup)"""
    print("Trainer module loaded successfully!")


if __name__ == "__main__":
    test_trainer()
