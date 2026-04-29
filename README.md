# AI Product Image Batch Pipeline (Local)

Automates stylized product image generation at scale using deterministic PIL-based image transformations. Generates **8 variations per product**, compares them to the source image, and selects the best match.

## What It Solves

- Processes large catalogs (`~5,000` products) with structured batch processing
- **No API keys required** — fully local, offline operation
- Generates **8 image variations** per product with category-specific styling
- Selects **best match** using image similarity (MSE metric)
- Category-aware prompts ensure consistency (electronics, clothing, home, etc.)
- Routes outputs to `data/output/<category>/<product_id>/`
- Tracks all variation scores in metadata JSON

## Pipeline Flow

1. Read product records from `data/manifest.csv`
2. Build category-aware prompts from `config/prompts.yaml`
3. **Generate 8 stylized image variations** (different strength/seed combinations)
4. Compare each variation to source image (MSE similarity metric)
5. **Select best match** (highest similarity score)
6. Validate: score > threshold (default 0.75)
7. If pass: save to `data/output/<category>/<product_id>/<style>.png` + metadata
8. If fail: route to manual review

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
# Full run with all products in manifest
python main.py --settings config/settings.yaml --prompts config/prompts.yaml

# Test with first 5 products
python main.py --limit 5 --log-level INFO

# Shorthand (auto-resolves config files)
python main.py --settings settings.yaml --prompts prompts.yaml --limit 2
```

**Optional flags:**
- `--limit N` — process only first N products
- `--log-level DEBUG` — verbose logging
- `--settings <path>` — custom settings file
- `--prompts <path>` — custom prompts file

## How It Works: Multi-Variation Selection

For each product:

1. **Generate 8 variations** with varying strength/seed:
   - Variation 0: strength 0.7
   - Variation 1: strength 0.72
   - ... (through 0.84)
   
2. **Apply category-specific styling**:
   - **Electronics**: brightness ↑, saturation ↑, sharp, cool-tone grading
   - **Clothing**: warm tones, vibrant colors, fashion glow
   - **Home**: inviting warmth, luxury aesthetic, mood lighting
   - **Generic**: balanced enhancements

3. **Compare to source** using MSE (Mean Squared Error):
   - Lower MSE = higher similarity
   - Converted to 0.0-1.0 score: `similarity = 1 - (mse / max_mse)`

4. **Select best match**:
   - Example scores: `[0.996, 0.996, 0.995, 0.995, 0.995, 0.994, 0.994, 0.993]`
   - Pick variation with highest score
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
  "validation": {
    "passed": true,
    "score": 0.9961894154548645,
    "rationale": "Best match from 8 variations (index: 0)"
  },
  "output_image": "data/output/electronics/SKU-1001/hero.png",
  "dimensions": {"width": 1200, "height": 1200},
  "variations_count": 8,
  "best_variation_index": 0,
  "all_scores": [0.996, 0.996, 0.996, 0.995, 0.995, 0.995, 0.995, 0.995]
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
backend: local                    # Always local (no API calls)
max_retries: 3                    # Unused (local has multi-variation instead)
worker_count: 6                   # Parallel workers (CPU threads)
validation_score_threshold: 0.75  # Min similarity to pass
output_root: data/output
manual_review_root: data/manual_review
```

## Performance & Throughput

- **Speed**: ~1-2 seconds per product (PIL operations only, no GPU needed)
- **Expected throughput**: 500-1000 images/day easily achievable
- **Parallelism**: 6 worker threads (configurable)
- **Memory**: Minimal (PIL in-memory only)
- **Dependencies**: No external APIs, no internet required

**Scaling tips:**
- Increase `worker_count` for faster batch processing
- Run on machines with 4+ CPU cores for optimal throughput
- Use `--limit` for testing before full batch runs

## Architecture

```
main.py (CLI entry)
  ↓
BatchPipeline (orchestration)
  ├─→ manifest.py (load CSV)
  ├─→ prompting.py (build prompts)
  ├─→ LocalImageBackend (generate 8 variations)
  │   ├─→ _stylize_electronics()
  │   ├─→ _stylize_clothing()
  │   ├─→ _stylize_home()
  │   └─→ _compute_image_similarity() [MSE]
  ├─→ qa.py (validate scores)
  └─→ storage.py (save outputs)
```

## File Structure

```
pyproject/
├── main.py                    # CLI entry
├── config/
│   ├── settings.yaml          # Pipeline config
├── config/
│   └── prompts.yaml           # Category prompts
├── data/
│   ├── manifest.csv           # Product metadata
│   ├── input/                 # Source product images
│   ├── output/                # Generated images (passed)
│   └── manual_review/         # Failed outputs
├── src/pipeline/
│   ├── local_backend.py       # Image generation engine
│   ├── pipeline.py            # Orchestration
│   ├── config.py              # Config parsing
│   ├── prompting.py           # Prompt building
│   ├── manifest.py            # CSV loading
│   ├── qa.py                  # Validation
│   ├── storage.py             # I/O utilities
│   └── types.py               # Data structures
└── requirements.txt           # Dependencies
```

## Dependencies

- **pillow** — Image processing (crop, resize, enhance)
- **pyyaml** — Config/prompt loading
- **numpy** — Histogram/MSE computation
- **python-dotenv** — Environment variables (optional, for future extensions)

Install all:
```bash
pip install -r requirements.txt
```

## Example Usage

```bash
# Test with 2 products
python main.py --limit 2 --log-level INFO

# Process all products
python main.py

# Debug specific categories
python main.py --limit 10 --log-level DEBUG
```

**Expected output:**
```
2026-04-29 15:05:58 | INFO | main | Using backend: local
2026-04-29 15:05:58 | INFO | BatchPipeline | Loaded 2 records from manifest.
2026-04-29 15:06:01 | INFO | BatchPipeline | PASS product=SKU-1001 variations=8 best_score=1.00 best_idx=0
2026-04-29 15:06:02 | INFO | BatchPipeline | PASS product=SKU-2001 variations=8 best_score=0.98 best_idx=0
2026-04-29 15:06:02 | INFO | BatchPipeline | Pipeline finished: {'passed': 2, 'manual_review': 0, 'errors': 0}
```

