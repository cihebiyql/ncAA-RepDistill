"""
Muon optimizer（带 AdamW 兜底）：
- 对 2D 权重矩阵做 momentum + Newton-Schulz 正交化更新（Muon）
- 对其余参数（bias/LayerNorm/embedding/head）做 AdamW 更新

说明：
本实现是面向本项目的最小可用版本，默认适配 bf16/AMP 训练。
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

import torch


@torch.no_grad()
def zeropower_via_newtonschulz5(g: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Newton-Schulz iteration (quintic) for approximate orthogonalization.

    输入 g 为 2D 梯度矩阵，输出形状相同的“近似正交更新”。
    """
    if g.ndim != 2:
        raise ValueError(f"Expected 2D tensor, got shape={tuple(g.shape)}")

    a, b, c = (3.4445, -4.7750, 2.0315)
    x = g.to(dtype=torch.bfloat16)
    transposed = False
    if x.size(0) > x.size(1):
        x = x.t()
        transposed = True

    x = x / (x.norm() + 1e-7)
    for _ in range(int(steps)):
        aa = x @ x.t()
        bb = b * aa + c * (aa @ aa)
        x = a * x + bb @ x

    if transposed:
        x = x.t()
    return x


class Muon(torch.optim.Optimizer):
    """
    Muon（MomentUm Orthogonalized by Newton-Schulz）+ AdamW fallback.

    - muon_params: 使用 Muon 更新的参数（必须是 2D 矩阵参数）
    - adamw_params: 使用 AdamW 更新的参数（可包含 1D/2D 参数）
    """

    def __init__(
        self,
        *,
        lr: float,
        weight_decay: float,
        muon_params: Iterable[torch.nn.Parameter],
        adamw_params: Iterable[torch.nn.Parameter],
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        muon_lr_scale: float = 0.2,
        adamw_betas: Sequence[float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
    ) -> None:
        betas = tuple(float(x) for x in adamw_betas)
        if len(betas) != 2:
            raise ValueError("adamw_betas must have length 2")

        defaults = dict(
            lr=float(lr),
            wd=float(weight_decay),
            momentum=float(momentum),
            nesterov=bool(nesterov),
            ns_steps=int(ns_steps),
            muon_lr_scale=float(muon_lr_scale),
            adamw_betas=betas,
            adamw_eps=float(adamw_eps),
        )

        muon_params_list = list(muon_params)
        adamw_params_list = list(adamw_params)
        params: List[torch.nn.Parameter] = [*muon_params_list, *adamw_params_list]
        super().__init__(params, defaults)

        for p in muon_params_list:
            if p.ndim != 2:
                raise ValueError(f"Muon params must be 2D, got ndim={p.ndim}")
            self.state[p]["use_muon"] = True
        for p in adamw_params_list:
            self.state[p]["use_muon"] = False

    @staticmethod
    def _adjust_lr_for_muon(lr: float, param_shape: Tuple[int, int], scale: float) -> float:
        a, b = param_shape[:2]
        return float(lr) * float(scale) * math.sqrt(float(max(a, b)))

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            wd = float(group["wd"])
            momentum = float(group["momentum"])
            nesterov = bool(group["nesterov"])
            ns_steps = int(group["ns_steps"])
            muon_lr_scale = float(group["muon_lr_scale"])
            beta1, beta2 = group["adamw_betas"]
            eps = float(group["adamw_eps"])

            # 1) Muon for 2D params
            for p in group["params"]:
                state = self.state[p]
                if not state.get("use_muon", False):
                    continue
                g = p.grad
                if g is None:
                    continue

                gg = g
                if gg.ndim != 2:
                    gg = gg.view(gg.size(0), -1)

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(gg)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(gg)
                update = gg.add(buf, alpha=momentum) if nesterov else buf

                u = zeropower_via_newtonschulz5(update, steps=ns_steps)
                adjusted_lr = self._adjust_lr_for_muon(lr, p.shape, muon_lr_scale)

                if wd > 0:
                    p.mul_(1 - lr * wd)
                p.add_(u, alpha=-adjusted_lr)

            # 2) AdamW fallback
            for p in group["params"]:
                state = self.state[p]
                if state.get("use_muon", False):
                    continue
                g = p.grad
                if g is None:
                    continue

                if "step" not in state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(g)
                    state["exp_avg_sq"] = torch.zeros_like(g)

                state["step"] += 1
                step = int(state["step"])
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                denom = exp_avg_sq.sqrt().add_(eps)
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                if wd > 0:
                    p.mul_(1 - lr * wd)
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

