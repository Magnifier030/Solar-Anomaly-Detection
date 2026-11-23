"""Parameter and running-stat policies independent of the training engine."""

from __future__ import annotations

from typing import Any

from torch import nn


def _keep_eval(module: nn.Module, inputs: Any) -> None:
    # The generic engine calls model.train() each epoch. Reapply eval before
    # a frozen subtree executes, so its BN buffers and dropout remain fixed.
    module.eval()


def apply_finetuning_policy(model: nn.Module, settings: dict[str, Any]) -> dict[str, Any]:
    mode = settings.get("mode", "full")
    names = settings.get("modules", [])
    freeze_stats = settings.get("freeze_running_stats", True)
    if mode not in {"full", "freeze_modules"}:
        raise ValueError("finetuning.mode must be full or freeze_modules")
    if not isinstance(names, list) or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("finetuning.modules must be a list of nonempty module names")
    if not isinstance(freeze_stats, bool):
        raise ValueError("freeze_running_stats must be boolean")
    if mode == "full" and names:
        raise ValueError("full finetuning cannot specify frozen modules")
    if mode == "freeze_modules" and not names:
        raise ValueError("freeze_modules requires module names")
    selected = []
    for name in names:
        try:
            selected.append(model.get_submodule(name))
        except AttributeError as error:
            raise ValueError(f"Unknown frozen module: {name}") from error
    frozen_ids = {id(p) for module in selected for p in module.parameters()}
    if selected and not any(id(p) not in frozen_ids for p in model.parameters()):
        raise ValueError("Finetuning must leave trainable parameters")
    for module in selected:
        module.requires_grad_(False)
        if freeze_stats:
            module.eval()
            module.register_forward_pre_hook(_keep_eval)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "mode": mode,
        "frozen_modules": names,
        "frozen_running_stats": bool(selected) and freeze_stats,
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
        "total_parameters": total,
    }
