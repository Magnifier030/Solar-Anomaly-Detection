"""Convert binary masks to submission-compatible polygons."""

from __future__ import annotations

from typing import Any

import numpy as np


def mask_to_polygons(mask: Any, threshold: float = 0.5) -> list[list[list[float]]]:
    """Extract external contours from a 2-D probability or binary mask."""
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - depends on optional eval extra
        raise RuntimeError("Install the evaluation extra to use mask-to-polygon conversion") from error
    array = np.asarray(mask, dtype=np.float32)
    binary = (array >= threshold).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[list[list[float]]] = []
    for contour in contours:
        if cv2.contourArea(contour) <= 0:
            continue
        polygons.append([[float(point[0][0]), float(point[0][1])] for point in contour])
    return polygons
