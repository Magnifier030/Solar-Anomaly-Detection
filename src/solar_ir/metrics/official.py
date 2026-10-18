"""Official competition localization metric (F1_TIoU).

The implementation follows the downloaded evaluation kit: recall and precision
are accumulated over the complete validation subset, duplicate predictions are
zeroed after matching, and class averaging excludes classes absent from both
ground truth and predictions. Text scoring is intentionally a separate plugin.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any


Point = Sequence[float]
PolygonPoints = Sequence[Point]
CLASS_NAMES = ("solar_panel", "hotspot", "crack")


def _polygon(points: PolygonPoints) -> Any:
    from shapely.geometry import Polygon

    if len(points) < 3:
        return None
    polygon = Polygon(points)
    if polygon.is_empty or not polygon.is_valid or polygon.area <= 0:
        return None
    return polygon


def _intersection_area(first: Any, second: Any) -> float:
    if first is None or second is None:
        return 0.0
    return float(first.intersection(second).area)


def _iou(first: Any, second: Any, intersection: float | None = None) -> float:
    if first is None or second is None:
        return 0.0
    overlap = _intersection_area(first, second) if intersection is None else intersection
    union = float(first.area + second.area - overlap)
    return overlap / union if union > 0 else 0.0


def _class_components(ground_truth: list[Any], predictions: list[Any]) -> tuple[float, float, int, int]:
    recall_sum = 0.0
    for truth in ground_truth:
        overlaps = [_intersection_area(truth, prediction) for prediction in predictions]
        if not overlaps or max(overlaps) <= 0:
            continue
        index = max(range(len(overlaps)), key=lambda position: overlaps[position])
        overlap = overlaps[index]
        recall_sum += _iou(truth, predictions[index], overlap) * overlap / float(truth.area)

    raw_precision: list[float] = []
    matched_truth: list[int | None] = []
    for prediction in predictions:
        overlaps = [_intersection_area(prediction, truth) for truth in ground_truth]
        if not overlaps or max(overlaps) <= 0:
            raw_precision.append(0.0)
            matched_truth.append(None)
            continue
        index = max(range(len(overlaps)), key=lambda position: overlaps[position])
        overlap = overlaps[index]
        other_union_area = 0.0
        others = [truth for position, truth in enumerate(ground_truth) if position != index and truth is not None]
        if others:
            other_union_area = float(prediction.intersection(_union(others)).difference(ground_truth[index]).area)
        penalty = other_union_area / float(prediction.area) if prediction.area > 0 else 0.0
        raw_precision.append(_iou(prediction, ground_truth[index], overlap) * (1.0 - penalty))
        matched_truth.append(index)

    precision = list(raw_precision)
    for truth_index in {index for index in matched_truth if index is not None}:
        candidates = [position for position, index in enumerate(matched_truth) if index == truth_index]
        best = max(candidates, key=lambda position: (raw_precision[position], -position))
        for position in candidates:
            if position != best:
                precision[position] = 0.0

    return recall_sum, sum(precision), len(ground_truth), len(predictions)


def _union(polygons: list[Any]) -> Any:
    from shapely.ops import unary_union

    return unary_union(polygons)


def f1_tiou(samples: Iterable[Mapping[str, Any]], classes: Sequence[str] = CLASS_NAMES) -> dict[str, float]:
    """Score polygon records using the official localization definition."""
    per_class: dict[str, float] = {}
    for class_name in classes:
        recall_sum = precision_sum = 0.0
        truth_count = prediction_count = 0
        for sample in samples:
            truth_map = sample.get("ground_truth", {})
            prediction_map = sample.get("prediction", {})
            truths = [_polygon(points) for points in truth_map.get(class_name, [])]
            predictions = [_polygon(points) for points in prediction_map.get(class_name, [])]
            # Invalid polygons remain in the denominator, matching the official
            # evaluator; they simply cannot overlap anything.
            recall, precision, gt_count, prediction_count_for_image = _class_components(truths, predictions)
            recall_sum += recall
            precision_sum += precision
            truth_count += gt_count
            prediction_count += prediction_count_for_image
        if truth_count or prediction_count:
            recall = recall_sum / truth_count if truth_count else 0.0
            precision = precision_sum / prediction_count if prediction_count else 0.0
            per_class[f"TIoU_{class_name}"] = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
    values = list(per_class.values())
    per_class["F1_TIoU"] = sum(values) / len(values) if values else 0.0
    return per_class


class OfficialLocalizationMetric:
    """Configurable metric plugin for validation and later submission audits."""

    name = "f1_tiou"

    def __init__(self, classes: Sequence[str] = CLASS_NAMES) -> None:
        self.classes = tuple(classes)

    def __call__(self, samples: Iterable[Mapping[str, Any]]) -> dict[str, float]:
        return f1_tiou(samples, self.classes)


def create_metric(config: dict[str, Any] | None = None) -> OfficialLocalizationMetric:
    config = config or {}
    return OfficialLocalizationMetric(tuple(config.get("classes", CLASS_NAMES)))
