"""Photo-level splitting policies, independent of image loading and models."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Mapping


def photo_stratified_split(
    frames: list[str],
    group_by_frame: Mapping[str, str],
    positive_by_frame: Mapping[str, bool],
    validation_fraction: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Split individual photos within each group/label stratum.

    Groups balance scene coverage; they are deliberately allowed across splits.
    Singleton strata stay in training. Physical panel identity is not available.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if len(set(frames)) != len(frames):
        raise ValueError("Photo split requires unique frame IDs")
    strata: dict[tuple[str, bool], list[str]] = defaultdict(list)
    for frame in sorted(frames):
        strata[(group_by_frame[frame], positive_by_frame[frame])].append(frame)
    generator = random.Random(seed)
    validation: set[str] = set()
    for key in sorted(strata):
        candidates = strata[key].copy()
        if len(candidates) < 2:
            continue
        generator.shuffle(candidates)
        count = min(len(candidates) - 1, max(1, round(len(candidates) * validation_fraction)))
        validation.update(candidates[:count])
    train = [frame for frame in sorted(frames) if frame not in validation]
    valid = [frame for frame in sorted(frames) if frame in validation]
    if not train or not valid:
        raise ValueError("Photo stratification produced an empty split; more samples are needed")
    return train, valid
