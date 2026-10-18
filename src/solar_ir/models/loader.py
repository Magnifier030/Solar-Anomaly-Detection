"""Dynamic model factory loading."""

from __future__ import annotations

import importlib
from typing import Any, cast

from torch import nn


def load_model(config: dict[str, Any]) -> nn.Module:
    factory_path = str(config["factory"])
    module_name, separator, attribute = factory_path.partition(":")
    if not separator:
        raise ValueError("model.factory must use package.module:factory syntax")
    factory = getattr(importlib.import_module(module_name), attribute)
    return cast(nn.Module, factory(config))
