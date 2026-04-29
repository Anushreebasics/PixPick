# AI Product Image Batch Pipeline (Local)

Automates stylized product image generation at scale using deterministic PIL-based image transformations. Generates **8 variations per product**, scores them with perceptual quality checks, and selects the strongest candidate.

## What It Solves

- Processes large catalogs (`~5,000` products) with structured batch processing
- **No API keys required** — fully local, offline operation
- Generates **8 image variations** per product with category-specific styling
- Selects the **best variation** using perceptual quality scoring and QA check counts
- Category-aware prompts ensure consistency (electronics, clothing, home, etc.)
- Routes outputs to `data/output/<category>/<product_id>/`
- Tracks all variation scores in metadata JSON
- Resume capability via manifest status tracking (skip completed rows, reprocess manually flagged items)

## Pipeline Flow

1. Read product records from `data/manifest.csv`
2. Build category-aware prompts from `templates/prompts.yaml`
3. **Generate 8 stylized image variations** using PIL deterministic transformations (different strength/seed combinations)
4. Score each variation with perceptual quality metrics:
   - **Sharpness** (Laplacian variance)
   - **Contrast** (pixel std dev)
   - **Dynamic range** (95th–5th percentile brightness)
   - **Exposure balance** (proximity to 0.55 brightness)
   - **Color pop** (saturation)
5. **Select best variation**: sort by QA check count (higher is better), then by quality score
6. Validate against thresholds (score > 0.75 and ≥ 3 QA checks)
7. If pass: save to `data/output/<category>/<product_id>/<style>.png` + metadata
8. If fail: route to `data/manual_review/<category>/<product_id>/` for human review

## Setup

1. Create virtual environment and install deps:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. **No API keys needed!** Local backend requires only standard libraries + PIL + numpy

3. Prepare input data:

- Put source images under `data/input/`
- Update `data/manifest.csv` with product details

Required CSV columns:
- `product_id`
- `category`
- `image_path`
- `style`

Optional:
- `product_title`

## Run

```bash
# Full run (resumes from manifest; skips completed rows automatically)
python main.py

# Validate manifest and estimate throughput without processing
python main.py --dry-run

# Test with first 5 products
python main.py --limit 5 --log-level INFO

# Shorthand (auto-resolves config files)
python main.py --settings settings.yaml --prompts prompts.yaml --limit 2
```

**Optional flags:**
- `--limit N` — process only first N products
- `--dry-run` — validate manifest, check image paths, estimate throughput (no generation)
- `--log-level DEBUG` — verbose logging
- `--settings <path>` — custom settings file
- `--prompts <path>` — custom prompts file

**Resume & Checkpoint:**
- Manifest rows with `status` = `passed` or `manual_review` are automatically skipped (resume on rerun)
- To reprocess a product, edit its row in `data/manifest.csv` and clear the `status` field, then re-run

## How It Works: Multi-Variation Selection

For each product:

1. **Generate 8 variations** with varying strength/seed combinations using PIL-based deterministic transforms:
   - Variation 0: strength 0.7
   - Variation 1: strength 0.72
   - ... (increments of 0.02, through 0.84)

2. **Apply category-specific styling**:
   - **Electronics**: brightness ↑, saturation ↑, sharp, cool-tone grading
   - **Clothing**: warm tones, vibrant colors, fashion glow
   - **Home**: inviting warmth, luxury aesthetic, mood lighting
   - **Generic**: balanced enhancements

3. **Score each variation** with perceptual quality metrics:
   - **Sharpness**: Laplacian variance (edge detection); threshold ≥ 0.35
   - **Contrast**: pixel standard deviation; threshold ≥ 0.35
   - **Dynamic range**: 95th–5th percentile brightness; threshold ≥ 0.30
   - **Exposure**: normalized brightness distance from ideal 0.55 (uses asymmetric divisor); threshold ≥ 0.45
   - **Color pop**: mean saturation (channel max – min); threshold ≥ 0.20
   - **Overall score**: weighted combination of all metrics (0–1 range)

4. **Select best variation**:
   - Sort by: (QA check count, quality score) in descending order
   - Example: if variation 0 passes 4 checks with score 0.82, and variation 3 passes 3 checks with score 0.85, variation 0 wins
   - Save all scores in metadata for audit trail

## Output Structure

**Success (Passed validation):**
```
data/output/<category>/<product_id>/
  ├── hero.png           ← Best stylized image
  └── hero.json          ← Metadata with variation scores
```

**Example metadata:**
```json
{
  "product_id": "SKU-1001",
  "category": "electronics",
  "style": "hero",
  "source_image": "data/input/earbuds.jpg",
  "status": "passed",
  "attempt": 1,
  "validation": {
    "passed": true,
    "score": 0.85,
    "rationale": "Best quality variation from 8 variations (index: 0, checks: sharpness, contrast, dynamic_range, exposure, color_pop)"
  },
  "output_image": "data/output/electronics/SKU-1001/hero.png",
  "dimensions": {"width": 1200, "height": 1200},
  "variations_count": 8,
  "best_variation_index": 0,
  "all_scores": [0.85, 0.82, 0.79, 0.81, 0.78, 0.80, 0.77, 0.76],
  "all_check_counts": [5, 4, 4, 5, 3, 4, 3, 3]
}
```

**Manual Review (Failed validation):**
```
data/manual_review/<category>/<product_id>/
  ├── hero.json          ← Attempt history
  └── hero_latest.png    ← Last generated image
```

## Configuration

Edit `config/settings.yaml` to customize:

```yaml
backend: local                          # Always local (no API calls)
input_root: data/input
manifest_path: data/manifest.csv
output_root: data/output
manual_review_root: data/manual_review
max_retries: 3                          # Unused (local uses multi-variation instead)
worker_count: 6                         # Parallel workers (CPU threads)
request_timeout_seconds: 120            # Timeout (unused in local mode)
validation_score_threshold: 0.78        # Min perceptual quality score (0–1)
validation_required_checks:             # QA checks that must pass (≥ 3 required)
  - sharpness                           # ≥ 0.35
  - contrast                            # ≥ 0.35
  - dynamic_range                       # ≥ 0.30
  - exposure                            # ≥ 0.45
  - color_pop                           # ≥ 0.20
local:
  model_name: runwayml/stable-diffusion-v1-5
  seed: 42
```

## Performance & Throughput

- **Speed**: ~2 seconds per product (8 variations × PIL transforms, no GPU needed)
- **Expected throughput**: 1500–1800 images/hour (6 workers)
- **Parallelism**: configurable worker threads (default 6, adjust for your CPU cores)
- **Memory**: minimal (PIL in-memory only; one image at a time per worker)
- **Dependencies**: no external APIs, no internet required

**Throughput Estimation:**
Run `python main.py --dry-run` to validate manifest and estimate wall time for your dataset before processing 5,000+ products.

**Scaling tips:**
- Increase `worker_count` in `config/settings.yaml` for faster batch processing
- Run on machines with 4+ CPU cores for optimal throughput
- Use `--limit` flag for testing before full batch runs
- Use `--log-level INFO` to track progress, `DEBUG` for verbose diagnostics

## Architecture

```
main.py (CLI entry, manifest validation)
  ↓
BatchPipeline (orchestration, resume logic)
  ├─→ manifest.py (load CSV, skip completed rows, atomic updates)
  ├─→ prompting.py (build category-aware prompts)
  ├─→ LocalImageBackend (generate 8 variations locally)
  │   ├─→ _stylize_electronics() / _clothing() / _home()
  │   └─→ generate_multiple_images() ← calls quality.py per variation
  ├─→ quality.py (perceptual scoring: sharpness, contrast, dynamic_range, exposure, color_pop)
  └─→ storage.py (save outputs & metadata)
```

**Single-Stage Workflow:**
- Generate 8 deterministic PIL-based variations
- Score all variations with perceptual quality metrics
- Rank by QA check count → quality score and select best

## File Structure

```
pyproject/
├── main.py                    # CLI entry (--dry-run, --limit, --log-level)
├── config/
│   ├── settings.yaml          # Pipeline config (backend, thresholds, workers)
├── templates/
│   └── prompts.yaml           # Category-specific styling prompts
├── data/
│   ├── manifest.csv           # Product metadata (product_id, category, image_path, style, status)
│   ├── input/                 # Source product images
│   ├── output/                # Generated images (passed validation)
│   └── manual_review/         # Failed outputs (manual review pending)
├── src/pipeline/
│   ├── local_backend.py       # PIL-based image generation engine
│   ├── gemini_backend.py      # Gemini API client + fallback
│   ├── pipeline.py            # Orchestration & resume logic
│   ├── quality.py             # Perceptual quality scoring
│   ├── config.py              # Config/prompt parsing
│   ├── prompting.py           # Prompt building
│   ├── manifest.py            # CSV loading & atomic updates
│   ├── qa.py                  # Validation wrapper
│   ├── storage.py             # I/O utilities
│   └── types.py               # Data structures
└── requirements.txt           # Dependencies
```

## Dependencies

- **pillow** — Image processing (crop, resize, enhance)
- **pyyaml** — Config/prompt loading
- **numpy** — Perceptual quality metrics and image statistics
- **python-dotenv** — Environment variables (optional, for future extensions)

Install all:
```bash
pip install -r requirements.txt
```

## Example Usage

```bash
# Validate manifest before running full batch
python main.py --dry-run

# Test with 2 products
python main.py --limit 2 --log-level INFO

# Process all pending products (resume from manifest)
python main.py

# Debug specific categories
python main.py --limit 10 --log-level DEBUG
```

**Expected output (local backend):**
```
2026-04-29 15:05:58,001 | INFO | main | Using backend: local (no GEMINI_API_KEY found)
2026-04-29 15:05:58,002 | INFO | BatchPipeline | Loaded 2 pending records from manifest (completed rows are skipped for resume).
2026-04-29 15:06:01,234 | INFO | BatchPipeline | PASS product=SKU-1001 variations=8 quality_score=0.85 best_idx=0 checks=5
2026-04-29 15:06:02,567 | INFO | BatchPipeline | MANUAL_REVIEW product=SKU-2001 after variation generation
2026-04-29 15:06:02,568 | INFO | BatchPipeline | Pipeline finished: {'passed': 1, 'manual_review': 1, 'errors': 0}
```

**Dry-run output:**
```
2026-04-29 15:05:58,001 | INFO | main.dry-run | Dry-run: validating manifest at data/manifest.csv
2026-04-29 15:05:58,002 | INFO | main.dry-run | Manifest rows: 2; pending: 1; missing image files: 0
2026-04-29 15:05:58,003 | INFO | main.dry-run | Estimated wall time at 6 workers (avg 2.0s/img): 0.3s
```

