# Subterra v2 (Model-Only)

This folder is a clean, “model-only” extraction meant for a fresh SubterraAI v2 rebuild later.

What’s included:
- Minimal Python package for **inference** (U-Net + YOLO segmentation) and metric extraction.
- A small CLI to run predictions on a folder of images.

What’s *not* included:
- Django backend, database, API, or the React frontend
- Model weight files (`.pth` / `.pt`) — these are large; point the CLI at your existing checkpoints.

## Quickstart

1) Create an environment and install deps:
```bash
cd subterra_v2
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Run prediction on a directory of images:
```bash
python -m subterra_model.cli predict \
  --model unet \
  --checkpoint ../media/models/unet_saved.pth \
  --input ../batch_processing/input \
  --output ./outputs/unet_run \
  --recursive
```

YOLO example:
```bash
python -m subterra_model.cli predict \
  --model yolo \
  --checkpoint ../media/models/yolo_saved.pt \
  --input ../batch_processing/input \
  --output ./outputs/yolo_run \
  --recursive
```

Outputs:
- `outputs/.../mask/` binary masks (`.png`)
- `outputs/.../compare/` (optional) side-by-side image+mask (`.png`)
- `outputs/.../measurements.csv` per-image root metrics

## FastAPI Web Service

For programmatic access, use the included FastAPI service:

1) Start the API server (automatically loads models from `saved_models/` folder):
```bash
python api.py
```

2) The API provides endpoints for single and batch image analysis. See `API_README.md` for detailed documentation.

## Next.js Human-in-the-loop GUI

This repo includes a simple Next.js GUI (`gui/`) for the full loop (currently U-Net only):
- Annotation (correct masks and save pairs)
- Re-training (fine-tune from current model + versioning)
- Phenotyping (batch metrics + CSV export)
- Phenotyping supports a Mini-rhizotron mode (tube/genotype/depth/timepoint metadata)
- Genotyping (upload markers + map against phenotypes)
- API (keys + automatic ingestion)

1) Start the API:
```bash
source .venv/bin/activate
python api.py
```

2) Start the GUI (in another terminal):
```bash
cd gui
npm install
npm run dev
```

3) Open `http://localhost:3000`, upload images, correct masks, then save pairs to `data/annotations/`.

### Re-training + versioning

After you save some corrected pairs, go to the GUI “Re-training” page to start a fine-tune job.

Artifacts:
- Annotations: `data/annotations/<id>/...`
- Model registry: `data/models/registry.json`
- Versioned checkpoints: `data/models/unet/<version>/unet.pth`

### Mini-rhizotron phenotyping

In `Phenotyping`, select “Mini-rhizotron” to attach metadata to each image (stored in the `analyses.extra.meta` JSON):
- `tube_id`, `genotype`, `depth`, `depth_length_cm`, `timepoint`, `session_label`, `session_time`
- camera calibration info (e.g. CI-600 scan DPI and `pixel_to_cm`) which drives the `scaling_factor` used in metric calculations

Tip: include tokens in filenames to auto-fill, e.g. `tube01_depth1_time1.png` (parses `tube01` and `depth1`).

## Docker Compose (API + GUI + Postgres)

`docker compose up --build` starts:
- API: `http://localhost:8001`
- GUI: `http://localhost:3000`
- Genotype/mapping service: `http://localhost:8002`
- Postgres: `localhost:5432` (db: `subterra`, user: `subterra`, password: `subterra`)

The API persists:
- Annotations + analyses + model versions + training jobs in Postgres
- Files on disk under `data/annotations/` and `data/models/`

The genotyping service persists:
- Marker tables + mapping results in the same Postgres database (shared `DATABASE_URL`)

Genotyping methods (selectable in the GUI):
- GWAS (single-marker linear regression + p-values, optional FDR/BH or Bonferroni)
- QTL (ANOVA across genotype groups)
- QTL (LOD score from regression vs null)

## Model details

See `subterra_v2/MODEL_DETAILS.md`.
