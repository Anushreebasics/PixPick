from __future__ import annotations

from .types import ValidationResult


def parse_validation(raw: dict, score_threshold: float) -> ValidationResult:
    passed = bool(raw.get("passed", False))
    score = float(raw.get("score", 0.0))

    failure_types = [str(item).strip().lower() for item in raw.get("failure_types", []) if str(item).strip()]
    corrective_constraints = [
        str(item).strip() for item in raw.get("corrective_constraints", []) if str(item).strip()
    ]

    rationale = str(raw.get("rationale", "")).strip()

    # Gate on threshold even if the model marks it as passed.
    final_passed = passed and score >= score_threshold

    return ValidationResult(
        passed=final_passed,
        score=score,
        failure_types=failure_types,
        corrective_constraints=corrective_constraints,
        rationale=rationale,
    )
