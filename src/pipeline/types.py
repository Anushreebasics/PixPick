from dataclasses import dataclass, field
from typing import List


@dataclass
class ProductRecord:
    product_id: str
    category: str
    image_path: str
    style: str
    product_title: str = ""
    manifest_row_index: int = -1
    manifest_status: str = ""


@dataclass
class ValidationResult:
    passed: bool
    score: float
    failure_types: List[str] = field(default_factory=list)
    corrective_constraints: List[str] = field(default_factory=list)
    rationale: str = ""


@dataclass
class GenerationAttempt:
    attempt: int
    prompt: str
    generated_image_bytes: bytes
    validation: ValidationResult
