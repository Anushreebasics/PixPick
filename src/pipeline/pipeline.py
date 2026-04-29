from __future__ import annotations

import concurrent.futures
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

from .config import PromptConfig, Settings
from .manifest import load_manifest, update_manifest
from .prompting import PromptBuilder
from .qa import parse_validation
from .storage import detect_mime_type, image_dimensions, save_image, write_json
from .types import GenerationAttempt, ProductRecord


class BatchPipeline:
    def __init__(self, settings: Settings, prompt_config: PromptConfig, gemini_client) -> None:
        self.settings = settings
        self.prompt_builder = PromptBuilder(prompt_config)
        self.client = gemini_client
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(self, limit: Optional[int] = None) -> Dict[str, int]:
        records = load_manifest(self.settings.manifest_path, skip_completed=True)
        if limit is not None:
            records = records[:limit]

        self.logger.info(
            "Loaded %d pending records from manifest (completed rows are skipped for resume).",
            len(records),
        )

        counters = {"passed": 0, "manual_review": 0, "errors": 0}

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.settings.worker_count) as pool:
            futures = {pool.submit(self._process_record, record): record for record in records}
            for future in concurrent.futures.as_completed(futures):
                record = futures[future]
                result = future.result()
                update_manifest(self.settings.manifest_path, record, result)
                counters[result] += 1

        self.logger.info("Pipeline finished: %s", counters)
        return counters

    def _process_record(self, record: ProductRecord) -> str:
        image_path = Path(record.image_path)
        if not image_path.exists():
            self.logger.error("Missing image for %s at %s", record.product_id, image_path)
            return "errors"

        source_bytes = image_path.read_bytes()
        source_mime = detect_mime_type(image_path)

        base_prompt = self.prompt_builder.build_initial_prompt(record)

        attempts: List[GenerationAttempt] = []

        # Generate multiple variations and pick the best
        try:
            result = self.client.generate_multiple_images(
                source_image_bytes=source_bytes,
                prompt=base_prompt,
                num_variations=8,
                source_mime_type=source_mime,
            )
            
            generated_bytes = result['best_image_bytes']
            best_score = result['best_score']
            variation_idx = result['variation_index']
            best_check_count = result.get('check_counts', [0])[variation_idx]
            best_passed_checks = result.get('passed_checks', [[]])[variation_idx]
            
            raw_validation = {
                "passed": best_score > 0.75 and best_check_count >= 3,
                "score": best_score,
                "failure_types": [],
                "corrective_constraints": [],
                "rationale": (
                    f"Best quality variation from {result['num_variations']} variations "
                    f"(index: {variation_idx}, checks: {', '.join(best_passed_checks) if best_passed_checks else 'none'})"
                ),
            }
            
            validation = parse_validation(raw_validation, self.settings.validation_score_threshold)
            attempts.append(
                GenerationAttempt(
                    attempt=1,
                    prompt=base_prompt,
                    generated_image_bytes=generated_bytes,
                    validation=validation,
                )
            )
            
            if validation.passed:
                self._persist_success(record, attempts[-1], extra_metadata={
                    'variations_count': result['num_variations'],
                    'best_variation_index': variation_idx,
                    'all_scores': result['scores'],
                    'all_check_counts': result.get('check_counts', []),
                })
                self.logger.info(
                    "PASS product=%s variations=%d quality_score=%.2f best_idx=%d checks=%d",
                    record.product_id,
                    result['num_variations'],
                    best_score,
                    variation_idx,
                    best_check_count,
                )
                return "passed"
            
        except Exception as exc:
            self.logger.exception(
                "Multi-variation generation failed for product=%s error=%s",
                record.product_id,
                exc,
            )
            self._persist_manual_review(record, attempts)
            return "manual_review"

        self._persist_manual_review(record, attempts)
        self.logger.warning("MANUAL_REVIEW product=%s after variation generation", record.product_id)
        return "manual_review"

    def _persist_success(self, record: ProductRecord, attempt: GenerationAttempt, extra_metadata: Optional[Dict] = None) -> None:
        out_dir = self.settings.output_root / record.category / record.product_id
        image_path = out_dir / f"{record.style}.png"
        meta_path = out_dir / f"{record.style}.json"

        save_image(image_path, attempt.generated_image_bytes)
        width, height = image_dimensions(attempt.generated_image_bytes)

        metadata = {
            "product_id": record.product_id,
            "category": record.category,
            "style": record.style,
            "source_image": record.image_path,
            "status": "passed",
            "attempt": attempt.attempt,
            "validation": asdict(attempt.validation),
            "output_image": str(image_path),
            "dimensions": {"width": width, "height": height},
        }
        
        if extra_metadata:
            metadata.update(extra_metadata)
        
        write_json(meta_path, metadata)

    def _persist_manual_review(self, record: ProductRecord, attempts: List[GenerationAttempt]) -> None:
        review_dir = self.settings.manual_review_root / record.category / record.product_id

        history_payload = {
            "product_id": record.product_id,
            "category": record.category,
            "style": record.style,
            "source_image": record.image_path,
            "status": "manual_review",
            "attempts": [
                {
                    "attempt": attempt.attempt,
                    "validation": asdict(attempt.validation),
                    "prompt": attempt.prompt,
                }
                for attempt in attempts
            ],
        }

        write_json(review_dir / f"{record.style}.json", history_payload)

        if attempts:
            save_image(review_dir / f"{record.style}_latest.png", attempts[-1].generated_image_bytes)
