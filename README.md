# Subterra v2 — Human-in-the-loop Root Phenotyping Platform

Subterra v2 is a local-first, Docker-friendly stack for:

- Segmenting roots with a U-Net model (and optional YOLO inference support in the API)
- Human-in-the-loop correction (annotation) with a Next.js GUI
- Fine-tuning / retraining from corrected masks with versioning
- Phenotyping export (CSV) with Minirhizotron metadata (tube/genotype/depth/time/session)
- Genotype mapping (GWAS/QTL-style association) via a separate service

This repo is intended for research workflows and iterative model improvement.

## Architecture

- **GUI (Next.js)**: `gui/` → `http://localhost:3000`
  - `/` dashboard
  - `/phenotype` batch inference + preview + CSV export + send-to-annotation
  - `/annotate` brush-based mask correction + save pairs
  - `/retrain` start U-Net fine-tune jobs + version list
  - `/genotype` upload marker tables + run mapping (GWAS/QTL-like)
  - `/api` manage API keys (admin)
- **Root analysis API (FastAPI)**: `api.py` → `http://localhost:8001`
  - inference (`/analyze`, `/batch-analyze`)
  - annotation storage (`/annotations`)
  - training jobs + versioning (`/train/unet`, `/models`, `/dashboard`)
  - ingestion + API keys (`/ingest/analysis`, `/api-keys/*`)
- **Genotype mapping service (FastAPI)**: `genotype_service/app.py` → `http://localhost:8002`
  - marker storage + mapping runs (`/markers/upload`, `/mapping/run`)
- **Database**
  - Docker Compose uses Postgres (recommended)
  - Local dev can fall back to SQLite if `DATABASE_URL` is not set

## Quickstart (recommended): Docker Compose

1) Ensure you have model weights available:
- U-Net checkpoint should exist at `subterra_model/models/saved_models/unet_saved.pth`
- (Optional) YOLO checkpoint at `subterra_model/models/saved_models/yolo_saved.pt`

2) Start everything:
```bash
docker compose up --build
```

3) Open the GUI:
- `http://localhost:3000`

### What Docker starts

- GUI: `http://localhost:3000`
- API: `http://localhost:8001`
- Genotype service: `http://localhost:8002`
- Postgres: `localhost:5432` (db: `subterra`, user: `subterra`, password: `subterra`)

Data volumes:
- Annotations: `data/annotations/`
- Versioned models + registry: `data/models/`
- Postgres volume: `postgres_data` (Docker-managed)

## Local development (no Docker)

### API

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python api.py
```

### GUI

```bash
cd gui
npm install
npm run dev
```

Open `http://localhost:3000`.

### Genotype service (optional)

```bash
cd genotype_service
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

## The human-in-the-loop workflow

1) **Phenotype** (`/phenotype`)
- Upload images → the API runs U-Net inference → metrics calculated → preview image + predicted mask side-by-side
- Export metrics to CSV
- Click “Send to Annotation” to open the same image in `/annotate` with the suggested mask

2) **Annotate** (`/annotate`)
- Brush-correct the predicted mask (Add / Erase)
- Save pairs → stored under `data/annotations/<annotation_id>/` and also recorded in the database
- The API enforces a safety check: the saved mask **must match** the original image size in pixels (prevents training errors)

3) **Re-train** (`/retrain`)
- Start a fine-tune job from the current U-Net checkpoint
- Each job produces a new version: `unet_v0001`, `unet_v0002`, …
- The current version is used for new inference immediately (hot-reloaded in the API)

## Phenotyping modes

### Standard

Use the numeric `Scaling Factor` directly (cm-per-pixel or any project-defined scaling unit you use in `subterra_model/utils/root_analysis.py`).

### Minirhizotron (recommended for tubes/depth sessions)

Minirhizotron mode adds metadata fields and uses camera calibration to derive scaling.

Metadata stored per image (in analysis records and/or annotations):
- `tube_id`, `genotype`
- `depth`, `depth_length_cm`
- `timepoint`, `session_label`, `session_time`
- `camera_model`, `camera_dpi`, `pixel_to_cm`

Camera (current default option):
- **CID CI-600 In-Situ Root Imager**
- The UI supports **DPI-based** conversion to `pixel_to_cm` and a **manual override**
- Link: https://cid-inc.com/plant-science-tools/root-measurement-plants/ci-600-in-situ-root-imager/#Specifications

Filename parsing helpers:
- `tube01_depth1_...png` auto-fills `tube_id=tube01`, `depth=1`

## Training “perfectness” (avoiding pixel/camera mistakes)

To keep training consistent:

- Always annotate on the original resolution; the API rejects mask/image size mismatches at save time (`POST /annotations`)
- In `/annotate`, enable Minirhizotron capture metadata so camera/DPI/pixel-to-cm are saved with each annotation
- In `/retrain`, use:
  - **Camera Filter** → select “CID CI-600 In-Situ Root Imager” to avoid mixing cameras
  - **Use Only Corrected** → trains only on masks you actually edited (recommended)
  - **Aspect Ratio: Preserve + pad** → avoids distortion when resizing to square training sizes

## Model versioning

Artifacts:
- Annotation pairs: `data/annotations/<id>/`
- Model registry: `data/models/registry.json`
- Versioned checkpoints: `data/models/unet/unet_v####/unet.pth`

The API keeps a database table `model_versions` with `is_current=true` for the active checkpoint.

## Genotype mapping (GWAS/QTL-style)

The genotype service stores a marker table and computes associations between a chosen phenotype metric and each marker.

Supported methods:
- `pearson` correlation
- `linear` regression (single-marker; p-value for slope)
- `anova` one-way ANOVA across genotype groups (rounded marker values)
- `lod` QTL-style LOD score (regression vs null)

P-value adjustment:
- `none`, `bonferroni`, `bh` (Benjamini–Hochberg FDR)

Important: mapping joins phenotypes to genotypes by **sample_id = analysis filename**. Keep your markers CSV first column aligned with the filenames in the analyses table.

## API keys + ingestion

- Admin creates keys via `/api-keys` (requires `SUBTERRA_ADMIN_TOKEN`)
- Clients authenticate with `X-API-Key` or `Authorization: Bearer ...`
- You can ingest externally computed phenotype rows via `POST /ingest/analysis`

See `API_README.md` for endpoint details.

## Repo layout

- `api.py` FastAPI root analysis API
- `subterra_model/` inference, metrics, DB models, training
- `gui/` Next.js human-in-the-loop GUI
- `genotype_service/` genotype + mapping microservice
- `data/` persistent artifacts (annotations, versioned models)

## Troubleshooting

- **Postgres “locale not found” warning**: harmless on Alpine images; you can ignore it.
- **API inference fails**: ensure `subterra_model/models/saved_models/unet_saved.pth` exists or set `SUBTERRA_UNET_CHECKPOINT`.
- **Training finds 0 samples**: save some annotations first; if using “Use Only Corrected”, ensure you actually edited masks before saving.

## Model internals

- Model details: `MODEL_DETAILS.md`
- Training code: `subterra_model/training/unet_finetune.py`
