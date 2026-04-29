from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from PIL import Image


def assess_image_quality(image: Image.Image) -> Dict[str, Any]:
    rgb_image = image.convert("RGB")
    rgb_array = np.asarray(rgb_image, dtype=np.float32)
    gray_array = np.asarray(rgb_image.convert("L"), dtype=np.float32)

    sharpness_raw = _laplacian_variance(gray_array)
    contrast_raw = float(gray_array.std())
    dynamic_range_raw = float(np.percentile(gray_array, 95) - np.percentile(gray_array, 5))
    brightness_raw = float(gray_array.mean())
    saturation_raw = float(np.mean(rgb_array.max(axis=2) - rgb_array.min(axis=2)))

    sharpness = _normalize_ratio(sharpness_raw, 120.0)
    contrast = _normalize_ratio(contrast_raw, 48.0)
    dynamic_range = max(0.0, min(1.0, dynamic_range_raw / 255.0))
    exposure = max(0.0, 1.0 - min(abs((brightness_raw / 255.0) - 0.55) / 0.55, 1.0))
    color_pop = _normalize_ratio(saturation_raw, 40.0)

    score = (
        0.35 * sharpness
        + 0.25 * contrast
        + 0.20 * dynamic_range
        + 0.10 * exposure
        + 0.10 * color_pop
    )

    checks: Dict[str, bool] = {
        "sharpness": sharpness >= 0.35,
        "contrast": contrast >= 0.35,
        "dynamic_range": dynamic_range >= 0.30,
        "exposure": exposure >= 0.45,
    }
    passed_checks: List[str] = [name for name, passed in checks.items() if passed]

    return {
        "score": float(max(0.0, min(1.0, score))),
        "check_count": len(passed_checks),
        "passed_checks": passed_checks,
        "metrics": {
            "sharpness": sharpness,
            "contrast": contrast,
            "dynamic_range": dynamic_range,
            "exposure": exposure,
            "color_pop": color_pop,
        },
    }


def _laplacian_variance(gray_array: np.ndarray) -> float:
    if gray_array.size == 0:
        return 0.0

    padded = np.pad(gray_array, 1, mode="edge")
    laplacian = (
        -4.0 * padded[1:-1, 1:-1]
        + padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
    )
    return float(laplacian.var())


def _normalize_ratio(value: float, scale: float) -> float:
    if value <= 0:
        return 0.0
    return float(value / (value + scale))
