import csv
from pathlib import Path
from typing import List

from .types import ProductRecord


REQUIRED_COLUMNS = {"product_id", "category", "image_path", "style"}


def load_manifest(path: Path) -> List[ProductRecord]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

        records: List[ProductRecord] = []
        for row in reader:
            records.append(
                ProductRecord(
                    product_id=row["product_id"].strip(),
                    category=row["category"].strip().lower(),
                    image_path=row["image_path"].strip(),
                    style=row["style"].strip().lower(),
                    product_title=(row.get("product_title") or "").strip(),
                )
            )
    return records
