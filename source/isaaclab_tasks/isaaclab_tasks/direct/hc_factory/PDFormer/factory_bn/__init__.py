"""Factory bottleneck prediction package built on PDFormer + ST-GNN Point Process."""

__all__ = ["BNPDFormer"]


def __getattr__(name: str):
    if name == "BNPDFormer":
        from .model import BNPDFormer

        return BNPDFormer
    raise AttributeError(name)
