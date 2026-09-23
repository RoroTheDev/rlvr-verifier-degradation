"""GSM8K/MATH-500 decontamination pipeline.

The public API is loaded lazily so importing :mod:`decontamination_pipeline`
does not download datasets or require the optional Qwen dependencies.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "GSM8K_DATASET": "GSM8K_DATASET",
    "MATH500_DATASET": "MATH500_DATASET",
    "DEFAULT_QWEN_MODEL": "DEFAULT_QWEN_MODEL",
    "QwenJudge": "QwenJudge",
    "minhash": "minhash",
    "normalize": "normalize",
    "run": "run",
    "main": "main",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve pipeline symbols only when they are actually requested."""
    if name not in _EXPORTS:
        raise AttributeError(f"Module {__name__!r} has no attribute {name!r}")
    module = import_module(".decontamination_pipeline", __name__)
    return getattr(module, _EXPORTS[name])


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))