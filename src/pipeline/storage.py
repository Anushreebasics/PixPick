from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from PIL import Image


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def detect_mime_type(image_path: Path) -> str:
    ext = image_path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext == ".webp":
        return "image/webp"
    return "image/png"


def save_image(path: Path, image_bytes: bytes) -> None:
    ensure_dir(path.parent)
    path.write_bytes(image_bytes)


def image_dimensions(image_bytes: bytes) -> tuple[int, int]:
    from io import BytesIO

    with Image.open(BytesIO(image_bytes)) as img:
        return img.width, img.height


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
