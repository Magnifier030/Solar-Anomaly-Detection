"""Convert native-resolution masks to bounded competition polygons."""

from __future__ import annotations

import numpy as np


def submission_polygons(mask: np.ndarray, max_vertices: int = 100) -> list[list[list[float]]]:
    """Keep external positive-area contours; simplify only above the vertex limit."""
    import cv2

    if max_vertices < 3:
        raise ValueError("max_vertices must be at least 3")
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) <= 0:
            continue
        epsilon = 0.25
        simplified = contour
        while len(simplified) > max_vertices:
            simplified = cv2.approxPolyDP(contour, epsilon, True)
            epsilon *= 1.5
        if len(simplified) < 3 or cv2.contourArea(simplified) <= 0:
            raise ValueError("Contour simplification lost positive-area geometry")
        polygons.append([[float(p[0][0]), float(p[0][1])] for p in simplified])
    return polygons
