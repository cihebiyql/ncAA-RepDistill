"""Exponential Moving Average utilities for model weights."""

from __future__ import annotations

from typing import Dict, Optional

import torch


class ExponentialMovingAverage:
    """Maintain an exponential moving average of model parameters."""

    def __init__(
        self,
        model: torch.nn.Module,
        decay: float = 0.999,
        device: Optional[torch.device] = None,
    ) -> None:
        if decay <= 0.0 or decay >= 1.0:
            raise ValueError("EMA decay must be in (0, 1)")

        self.decay = decay
        self.device = device
        self.shadow_params: Dict[str, torch.Tensor] = {}
        self.backup_params: Dict[str, torch.Tensor] = {}

        self._init_shadow(model)

    def _init_shadow(self, model: torch.nn.Module) -> None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                data = param.detach().clone()
                if self.device is not None:
                    data = data.to(self.device)
                self.shadow_params[name] = data

    def update(self, model: torch.nn.Module) -> None:
        """Update EMA weights from the current model parameters."""

        with torch.no_grad():
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if name not in self.shadow_params:
                    self.shadow_params[name] = param.detach().clone()
                shadow = self.shadow_params[name]
                new_average = shadow.mul(self.decay).add(param.detach(), alpha=1 - self.decay)
                self.shadow_params[name] = new_average

    def apply_shadow(self, model: torch.nn.Module) -> None:
        """Swap model weights with their EMA counterparts."""

        if self.backup_params:
            raise RuntimeError("EMA shadow already applied. Call restore() before reapplying.")

        with torch.no_grad():
            for name, param in model.named_parameters():
                if not param.requires_grad or name not in self.shadow_params:
                    continue
                self.backup_params[name] = param.detach().clone()
                param.data.copy_(self.shadow_params[name].to(param.device))

    def restore(self, model: torch.nn.Module) -> None:
        """Restore original model parameters after apply_shadow."""

        if not self.backup_params:
            return

        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.backup_params:
                    param.data.copy_(self.backup_params[name].to(param.device))

        self.backup_params.clear()


__all__ = ["ExponentialMovingAverage"]

