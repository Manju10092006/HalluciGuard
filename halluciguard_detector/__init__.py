"""HalluciGuard reference-grounded detector and orchestration adapter."""

from importlib import import_module

__all__ = ["Detector", "DetectorAgent"]
__version__ = "0.1.0"


def __getattr__(name):
    # Importing evidence/input helpers must not initialize torch/transformers.
    # Public class imports remain compatible and load their dependencies on use.
    if name not in __all__:
        raise AttributeError(name)
    module = ".agent" if name == "DetectorAgent" else ".detector"
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value
