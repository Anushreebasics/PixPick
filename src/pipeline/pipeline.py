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
from .local_backend import LocalImageBackend


class BatchPipeline:
    def __init__(self, settings: Settings, prompt_config: PromptConfig, gemini_client) -> None:
        self.settings = settings
        self.prompt_builder = PromptBuilder(prompt_config)
        self.client = gemini_client
        # Always have a local backend to generate variations locally.
        self.local_client = LocalImageBackend(model_name=self.settings.local_model_name, seed=self.settings.local_seed)
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
            # Stage 1: generate 8 local variations and collect their scores
            local_result = self.local_client.generate_multiple_images(
                source_image_bytes=source_bytes,
                prompt=base_prompt,
                num_variations=8,
                source_mime_type=source_mime,
            )

            # Pick top-2 candidates by (check_count, score)
            indices = list(range(len(local_result.get('scores', []))))
            indices.sort(key=lambda i: (local_result.get('check_counts', [0])[i], local_result.get('scores', [0])[i]), reverse=True)
            top2 = indices[:2] if indices else [0]

            # Stage 2: ask Gemini (or gemini_client) to compare the top-2 variations and act as QA
            chosen_idx = local_result.get('variation_index', 0)
            chosen_bytes = local_result.get('best_image_bytes')
            try:
                if len(top2) >= 2:
                    a_idx, b_idx = top2[0], top2[1]
                    a_bytes = local_result.get('variations', [None])[a_idx]
                    b_bytes = local_result.get('variations', [None])[b_idx]
                    compare = self.client.compare_images(source_bytes, a_bytes, b_bytes, base_prompt)
                    # choose winner based on Gemini/local verdict
                    winner = int(compare.get('winner_index', 0))
                    winner_global_idx = [a_idx, b_idx][winner]
                    chosen_idx = winner_global_idx
                    chosen_bytes = local_result.get('variations', [None])[chosen_idx]
                    # Use Gemini's passed verdict if available, else fallback to local
                    raw_validation = {
                        'passed': bool(compare.get('passed', False)),
                        'score': 0.0,
                        'failure_types': [],
                        'corrective_constraints': [],
                        'rationale': compare.get('rationale', 'gemini_comparison'),
                    }
                else:
                    # Fallback to local-best
                    chosen_bytes = local_result.get('best_image_bytes')
                    chosen_idx = local_result.get('variation_index', 0)
                    raw_validation = {
                        'passed': local_result.get('best_score', 0.0) > 0.75 and (local_result.get('check_counts', [0])[chosen_idx] >= 3),
                        'score': local_result.get('best_score', 0.0),
                        'failure_types': [],
                        'corrective_constraints': [],
                        'rationale': (
                            f"Best quality variation from {local_result.get('num_variations', 8)} variations "
                            f"(index: {chosen_idx})"
                        ),
                    }
            except Exception as exc:
                self.logger.warning("Comparison via client failed, falling back to local pick: %s", exc)
                chosen_bytes = local_result.get('best_image_bytes')
                chosen_idx = local_result.get('variation_index', 0)
                raw_validation = {
                    'passed': local_result.get('best_score', 0.0) > 0.75 and (local_result.get('check_counts', [0])[chosen_idx] >= 3),
                    'score': local_result.get('best_score', 0.0),
                    'failure_types': [],
                    'corrective_constraints': [],
                    'rationale': (
                        f"Best quality variation from {local_result.get('num_variations', 8)} variations "
                        f"(index: {chosen_idx})"
                    ),
                }

            generated_bytes = chosen_bytes
            best_score = local_result.get('scores', [0.0])[chosen_idx] if local_result.get('scores') else 0.0
            variation_idx = chosen_idx
            best_check_count = local_result.get('check_counts', [0])[variation_idx] if local_result.get('check_counts') else 0
            best_passed_checks = local_result.get('passed_checks', [[]])[variation_idx] if local_result.get('passed_checks') else []
            
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
                    'variations_count': local_result.get('num_variations', 8),
                    'best_variation_index': variation_idx,
                    'all_scores': local_result.get('scores', []),
                    'all_check_counts': local_result.get('check_counts', []),
                })
                self.logger.info(
                    "PASS product=%s variations=%d quality_score=%.2f best_idx=%d checks=%d",
                    record.product_id,
                    local_result.get('num_variations', 8),
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
