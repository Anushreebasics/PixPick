from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class Settings:
    backend: str
    input_root: Path
    manifest_path: Path
    output_root: Path
    manual_review_root: Path
    max_retries: int
    worker_count: int
    request_timeout_seconds: int
    retry_backoff_seconds: int
    generation_model: str
    validation_model: str
    generation_temperature: float
    generation_candidate_count: int
    generation_output_mime_type: str
    validation_score_threshold: float
    validation_required_checks: List[str]
    replicate_model_ref: str = "google/gemini-2.5-flash-image"
    replicate_strength: float = 0.7
    replicate_num_inference_steps: int = 24
    replicate_guidance_scale: float = 3.5
    local_model_name: str = "runwayml/stable-diffusion-v1-5"
    local_seed: int = 42


@dataclass
class PromptConfig:
    system_prompt: str
    default_style: str
    categories: Dict[str, str]
    failure_constraints: Dict[str, List[str]]


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_settings(path: str) -> Settings:
    raw = _load_yaml(Path(path))
    return Settings(
        backend=str(raw.get("backend", "gemini")).lower(),
        input_root=Path(raw["input_root"]),
        manifest_path=Path(raw["manifest_path"]),
        output_root=Path(raw["output_root"]),
        manual_review_root=Path(raw["manual_review_root"]),
        max_retries=int(raw["max_retries"]),
        worker_count=int(raw["worker_count"]),
        request_timeout_seconds=int(raw["request_timeout_seconds"]),
        retry_backoff_seconds=int(raw["retry_backoff_seconds"]),
        generation_model=str(raw["models"]["generation_model"]),
        validation_model=str(raw["models"]["validation_model"]),
        generation_temperature=float(raw["generation"]["temperature"]),
        generation_candidate_count=int(raw["generation"]["candidate_count"]),
        generation_output_mime_type=str(raw["generation"]["output_mime_type"]),
        validation_score_threshold=float(raw["validation"]["score_threshold"]),
        validation_required_checks=list(raw["validation"]["required_checks"]),
        replicate_model_ref=str(raw.get("replicate", {}).get("model_ref", "google/gemini-2.5-flash-image")),
        replicate_strength=float(raw.get("replicate", {}).get("strength", 0.7)),
        replicate_num_inference_steps=int(raw.get("replicate", {}).get("num_inference_steps", 24)),
        replicate_guidance_scale=float(raw.get("replicate", {}).get("guidance_scale", 3.5)),
        local_model_name=str(raw.get("local", {}).get("model_name", "runwayml/stable-diffusion-v1-5")),
        local_seed=int(raw.get("local", {}).get("seed", 42)),
    )


def load_prompt_config(path: str) -> PromptConfig:
    raw = _load_yaml(Path(path))
    return PromptConfig(
        system_prompt=str(raw["system_prompt"]),
        default_style=str(raw["default_style"]),
        categories=dict(raw.get("categories", {})),
        failure_constraints=dict(raw.get("failure_constraints", {})),
    )
