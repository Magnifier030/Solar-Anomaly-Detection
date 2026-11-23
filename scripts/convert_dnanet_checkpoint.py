"""Convert an author's legacy checkpoint to tensor-only weights, with restricted loading."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from solar_ir.models.dnanet import create_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    # The author's tar includes a NumPy float64 metric. Allow only its constructors;
    # retain weights_only=True rather than enabling arbitrary pickle deserialization.
    from numpy._core.multiarray import scalar

    with torch.serialization.safe_globals(
        [
            (scalar, "numpy.core.multiarray.scalar"),
            np.dtype,
            np.dtypes.Float64DType,
        ]
    ):
        checkpoint = torch.load(args.source, map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    model = create_model({})
    model.load_state_dict(state, strict=True)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.destination)
    print(f"Saved strictly validated ResNet-18 / channel_size=three weights: {args.destination}")


if __name__ == "__main__":
    main()
