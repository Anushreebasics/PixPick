from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from src.pipeline.config import load_prompt_config, load_settings
from src.pipeline.gemini_backend import GeminiImageBackend
from src.pipeline.local_backend import LocalImageBackend
from src.pipeline.pipeline import BatchPipeline
from src.pipeline.manifest import load_manifest, TERMINAL_STATUSES


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch AI product image generation pipeline")
    parser.add_argument("--settings", default="config/settings.yaml", help="Path to settings YAML")
    parser.add_argument("--prompts", default="templates/prompts.yaml", help="Path to prompt templates YAML")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for number of records")
    parser.add_argument("--dry-run", action="store_true", help="Validate manifest and image paths; estimate throughput without generating")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def _resolve_config_path(raw_path: str, fallback_dir: str) -> str:
    path = Path(raw_path)
    if path.exists():
        return str(path)

    fallback_path = Path(fallback_dir) / path.name
    if fallback_path.exists():
        return str(fallback_path)

    return raw_path


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    load_dotenv()

    settings_path = _resolve_config_path(args.settings, "config")
    prompts_path = _resolve_config_path(args.prompts, "templates")

    settings = load_settings(settings_path)
    prompt_config = load_prompt_config(prompts_path)

    local_client = LocalImageBackend(
        model_name=settings.local_model_name,
        seed=settings.local_seed,
    )

    gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if gemini_api_key:
        gemini_model = os.getenv("GEMINI_IMAGE_MODEL", "").strip() or settings.generation_model or "gemini-2.5-flash-image"
        client = GeminiImageBackend(
            api_key=gemini_api_key,
            model_name=gemini_model,
            timeout_seconds=settings.request_timeout_seconds,
            local_fallback=local_client,
        )
        logging.getLogger("main").info("Using backend: gemini (model=%s) with local fallback", gemini_model)
    else:
        client = local_client
        logging.getLogger("main").info("Using backend: local (no GEMINI_API_KEY found)")

    pipeline = BatchPipeline(settings=settings, prompt_config=prompt_config, gemini_client=client)
    if args.dry_run:
        logger = logging.getLogger("main.dry-run")
        manifest_path = settings.manifest_path
        logger.info("Dry-run: validating manifest at %s", manifest_path)
        try:
            all_records = load_manifest(manifest_path, skip_completed=False)
        except Exception as e:
            logger.error("Manifest validation failed: %s", e)
            return 2

        missing_paths = []
        pending = 0
        for r in all_records:
            if (r.manifest_status or "").lower() not in TERMINAL_STATUSES:
                pending += 1

            img_path = Path(r.image_path)
            resolved = img_path if img_path.is_absolute() else (settings.input_root / img_path)
            if not resolved.exists():
                missing_paths.append(str(resolved))

        backend_type = "gemini" if gemini_api_key else "local"
        avg_seconds = 5.0 if backend_type == "gemini" else 2.0
        workers = max(1, int(settings.worker_count))
        est_seconds = (pending * avg_seconds) / workers if pending else 0
        logger.info("Manifest rows: %d; pending: %d; missing image files: %d", len(all_records), pending, len(missing_paths))
        if missing_paths:
            logger.warning("Missing image paths (showing up to 10): %s", missing_paths[:10])
        logger.info("Estimated wall time at %d workers (avg %.1fs/img): %.1fs", workers, avg_seconds, est_seconds)
        return 0

    summary = pipeline.run(limit=args.limit)

    logging.getLogger("main").info("Summary: %s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
