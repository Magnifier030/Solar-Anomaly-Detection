"""Perspective-rectify a module polygon to an upright size x size crop.

Axis-aligned bbox crops of tilted modules carry black corners and neighbouring module
edges, which dominate within-frame outlier scores. Each polygon is reduced to four
corners (the polygon itself when it has four points, otherwise its minimum-area
rectangle), ordered TL, TR, BR, BL, rotated so the long side is vertical, and warped.
"""

import cv2
import numpy as np


def corners(points):
    pts = np.asarray(points, np.float32)
    if len(pts) != 4:
        pts = cv2.boxPoints(cv2.minAreaRect(pts)).astype(np.float32)
    center = pts.mean(0)
    angle = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angle)]  # clockwise in image coordinates, starting near TL
    start = np.argmin(pts.sum(1))
    pts = np.roll(pts, -start, axis=0)  # TL, TR, BR, BL
    width = np.linalg.norm(pts[1] - pts[0])
    height = np.linalg.norm(pts[3] - pts[0])
    if width > height:
        pts = np.roll(pts, -1, axis=0)  # make the long side vertical
    return pts


def rectify(image, points, size, inset=0.0):
    src = corners(points)
    # Corners land `inset` outside the crop, so only the module interior is kept.
    m = inset * size
    dst = np.array([[-m, -m], [size - 1 + m, -m], [size - 1 + m, size - 1 + m], [-m, size - 1 + m]], np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, matrix, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
