from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from solar_ir.datasets.labelme import ThermalPolygonDataset
from solar_ir.inference import TestImages as InferenceImages
from solar_ir.postprocess.submission import submission_polygons


def test_submission_contours_keep_small_objects_and_native_coordinates():
    mask = np.zeros((512, 640), dtype=np.uint8)
    mask[400:410, 600:620] = 255
    mask[10:12, 20:22] = 255
    polygons = submission_polygons(mask)
    assert len(polygons) == 2
    assert max(point[0] for p in polygons for point in p) == 619
    assert max(point[1] for p in polygons for point in p) == 409
    assert submission_polygons(np.zeros_like(mask)) == []


def test_submission_vertex_limit():
    import cv2

    mask = np.zeros((512, 640), dtype=np.uint8)
    cv2.circle(mask, (320, 256), 170, 255, -1)
    polygons = submission_polygons(mask)
    assert len(polygons) == 1
    assert 3 <= len(polygons[0]) <= 100
    assert cv2.contourArea(np.array(polygons[0], dtype=np.float32)) > 0


def test_test_loader_matches_validation_preprocessing(tmp_path):
    key = "test/IR/image/example.jpg"
    path = tmp_path / key
    path.parent.mkdir(parents=True)
    Image.fromarray(np.arange(100, dtype=np.uint8).reshape(10, 10)).save(path)
    loader = InferenceImages(
        [key], tmp_path, {"image_size": [32, 32], "preprocessing": "grayscale"}
    )
    tensor, actual_key, width, height = loader[0]
    transform = ThermalPolygonDataset([], tmp_path, (32, 32), preprocessing="grayscale")
    with Image.open(path) as source:
        image = source.convert("L")
        expected, _ = transform._prepare_tensors(image, Image.new("L", image.size))
    assert torch.equal(tensor, expected)
    assert actual_key == key and (width, height) == (10, 10)
