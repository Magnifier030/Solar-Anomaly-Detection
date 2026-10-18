"""Dataset readers and deterministic split utilities."""

from solar_ir.datasets.labelme import ThermalPolygonDataset, build_group_split

__all__ = ["ThermalPolygonDataset", "build_group_split"]
