import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from .types import ProductRecord


REQUIRED_COLUMNS = {"product_id", "category", "image_path", "style"}
TERMINAL_STATUSES = {"passed", "manual_review", "errors"}


def load_manifest(path: Path, skip_completed: bool = True) -> List[ProductRecord]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

        records: List[ProductRecord] = []
        for row_index, row in enumerate(reader):
            status = (row.get("status") or "").strip().lower()
            if skip_completed and status in TERMINAL_STATUSES:
                continue

            records.append(
                ProductRecord(
                    product_id=row["product_id"].strip(),
                    category=row["category"].strip().lower(),
                    image_path=row["image_path"].strip(),
                    style=row["style"].strip().lower(),
                    product_title=(row.get("product_title") or "").strip(),
                    manifest_row_index=row_index,
                    manifest_status=status,
                )
            )
    return records


def update_manifest(path: Path, record: ProductRecord, status: str) -> None:
    normalized_status = status.strip().lower()
    if not normalized_status:
        return

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if "status" not in fieldnames:
        fieldnames.append("status")
    if "updated_at" not in fieldnames:
        fieldnames.append("updated_at")

    if record.manifest_row_index < 0 or record.manifest_row_index >= len(rows):
        return

    rows[record.manifest_row_index]["status"] = normalized_status
    rows[record.manifest_row_index]["updated_at"] = datetime.now(timezone.utc).isoformat()

    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    temp_path.replace(path)
