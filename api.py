from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import List, Optional
from uuid import uuid4

import cv2
import numpy as np
import torch
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from subterra_model.db import SessionLocal, init_db
from subterra_model.db_models import AnalysisRow, AnnotationRow, ApiKeyRow, ModelVersionRow, TrainJobRow
from subterra_model.loading import load_model
from subterra_model.utils.masks import threshold as threshold_mask
from subterra_model.utils.root_analysis import calculate_metrics


class AnalysisRequest(BaseModel):
    model_type: str = "unet"  # "unet" or "yolo"
    threshold_area: int = 50
    scaling_factor: float = 1.0
    confidence_threshold: float = 0.3


class AnalysisResult(BaseModel):
    root_count: int
    average_root_diameter: float
    total_root_length: float
    total_root_area: float
    total_root_volume: float
    mask_image_base64: str
    original_image_base64: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    models_loaded: List[str]
    device: str


app = FastAPI(
    title="Subterra Root Analysis API",
    description="API for analyzing root images using U-Net and YOLO segmentation models",
    version="1.0.0"
)

# CORS (for local GUI dev at localhost:3000)
cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ALLOW_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model storage
models = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ANNOTATIONS_DIR = Path(os.getenv("SUBTERRA_ANNOTATIONS_DIR", "data/annotations"))
MODELS_DIR = Path(os.getenv("SUBTERRA_MODELS_DIR", "data/models"))
REGISTRY_PATH = MODELS_DIR / "registry.json"

_registry_lock = Lock()
_train_lock = Lock()
_executor = ThreadPoolExecutor(max_workers=1)
_train_jobs: dict[str, dict] = {}


def _db_session() -> Session:
    return SessionLocal()


def _api_key_secret() -> str:
    return os.getenv("SUBTERRA_API_KEY_SECRET", "dev-secret-change-me")


def _hash_api_key(token: str) -> str:
    data = f"{token}:{_api_key_secret()}".encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _extract_api_key_header(authorization: Optional[str], x_api_key: Optional[str]) -> Optional[str]:
    if x_api_key:
        return x_api_key.strip()
    if authorization:
        value = authorization.strip()
        if value.lower().startswith("bearer "):
            return value.split(" ", 1)[1].strip()
    return None


def _require_api_key() -> bool:
    return os.getenv("SUBTERRA_REQUIRE_API_KEY", "0").lower() in {"1", "true", "yes"}


def _verify_api_key(token: str) -> Optional[ApiKeyRow]:
    if not token:
        return None
    hashed = _hash_api_key(token)
    db = _db_session()
    try:
        row = (
            db.execute(select(ApiKeyRow).where(ApiKeyRow.hashed_key == hashed, ApiKeyRow.revoked.is_(False)))
            .scalars()
            .first()
        )
        if row:
            row.last_used_at = datetime.now(timezone.utc)
            db.commit()
        return row
    except Exception:
        db.rollback()
        return None
    finally:
        db.close()


def require_api_key(
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> ApiKeyRow:
    token = _extract_api_key_header(authorization, x_api_key)
    row = _verify_api_key(token or "")
    if not row:
        raise HTTPException(status_code=401, detail="Missing or invalid API key")
    return row


def maybe_require_api_key(
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> Optional[ApiKeyRow]:
    if not _require_api_key():
        return None
    return require_api_key(authorization=authorization, x_api_key=x_api_key)


def _require_admin(admin_token: Optional[str]) -> None:
    expected = os.getenv("SUBTERRA_ADMIN_TOKEN", "")
    if not expected:
        # If no admin token is configured, do not allow key creation/revoke by default.
        raise HTTPException(status_code=403, detail="Admin token not configured (set SUBTERRA_ADMIN_TOKEN)")
    if not admin_token or admin_token != expected:
        raise HTTPException(status_code=403, detail="Invalid admin token")


def _current_unet_version_id(db: Session) -> str:
    row = (
        db.execute(
            select(ModelVersionRow)
            .where(ModelVersionRow.model_type == "unet", ModelVersionRow.is_current.is_(True))
            .order_by(ModelVersionRow.created_at.desc())
        )
        .scalars()
        .first()
    )
    return row.version_id if row else ""


def _next_unet_version_id_db(db: Session) -> str:
    version_ids = db.execute(select(ModelVersionRow.version_id).where(ModelVersionRow.model_type == "unet")).scalars().all()
    max_n = 0
    for vid in version_ids:
        m = re.match(r"unet_v(\d+)$", str(vid))
        if m:
            max_n = max(max_n, int(m.group(1)))
    return f"unet_v{max_n + 1:04d}"


def _bootstrap_db(db: Session) -> None:
    """
    Initialize DB with a base U-Net version and import existing artifacts if present.
    """
    # Ensure a "base" U-Net version exists so versioning can start from current checkpoint.
    has_any = db.execute(select(func.count(ModelVersionRow.id))).scalar_one()
    if int(has_any) == 0:
        base_ckpt = os.getenv("SUBTERRA_UNET_CHECKPOINT") or "subterra_model/models/saved_models/unet_saved.pth"
        if os.path.exists(base_ckpt):
            db.add(
                ModelVersionRow(
                    version_id="unet_v0000",
                    model_type="unet",
                    checkpoint_path=str(base_ckpt),
                    base_checkpoint_path="",
                    annotations_dir="",
                    train_config={},
                    metrics={},
                    is_current=True,
                )
            )
            db.commit()

    # Import existing registry.json (if any) into DB (best-effort).
    if REGISTRY_PATH.exists():
        try:
            registry = _load_registry()
            versions = registry.get("unet", {}).get("versions", []) or []
            current = registry.get("unet", {}).get("current")
            existing = set(
                db.execute(select(ModelVersionRow.version_id).where(ModelVersionRow.model_type == "unet")).scalars().all()
            )
            for v in versions:
                vid = str(v.get("id", ""))
                if not vid or vid in existing:
                    continue
                ckpt = str(v.get("checkpoint_path", ""))
                if not ckpt:
                    continue
                db.add(
                    ModelVersionRow(
                        version_id=vid,
                        model_type="unet",
                        checkpoint_path=ckpt,
                        base_checkpoint_path=str(v.get("base_checkpoint_path") or ""),
                        annotations_dir=str(v.get("annotations_dir") or ""),
                        train_config=dict(v.get("train_config") or {}),
                        metrics=dict(v.get("metrics") or {}),
                        is_current=(vid == current),
                    )
                )
            db.commit()
        except Exception:
            db.rollback()

    # Import existing saved annotations into DB (best-effort, only when table empty).
    anno_count = db.execute(select(func.count(AnnotationRow.id))).scalar_one()
    if int(anno_count) == 0 and ANNOTATIONS_DIR.exists():
        for child in ANNOTATIONS_DIR.iterdir():
            if not child.is_dir():
                continue
            meta = child / "meta.json"
            if not meta.exists():
                continue
            try:
                payload = json.loads(meta.read_text("utf-8"))
            except Exception:
                continue

            annotation_id = str(payload.get("annotation_id") or child.name)
            image_name = str(payload.get("image_filename") or "")
            mask_name = str(payload.get("mask_filename") or "")
            if not image_name or not mask_name:
                continue
            image_path = child / image_name
            mask_path = child / mask_name
            if not image_path.exists() or not mask_path.exists():
                continue

            db.add(
                AnnotationRow(
                    annotation_id=annotation_id,
                    original_filename=str(payload.get("original_filename") or ""),
                    image_path=str(image_path),
                    mask_path=str(mask_path),
                    meta=payload if isinstance(payload, dict) else {},
                )
            )
        try:
            db.commit()
        except Exception:
            db.rollback()

def _safe_filename(filename: str) -> str:
    basename = os.path.basename(filename or "")
    basename = re.sub(r"[^A-Za-z0-9._-]+", "_", basename).strip("._")
    return basename or "file"


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


class AnnotationSaveResponse(BaseModel):
    annotation_id: str
    image_path: str
    mask_path: str
    metadata_path: str


class ModelVersion(BaseModel):
    id: str
    created_at: str
    checkpoint_path: str
    base_checkpoint_path: Optional[str] = None
    annotations_dir: str
    train_config: dict
    metrics: dict


class ModelsResponse(BaseModel):
    unet_current: Optional[str] = None
    unet_versions: List[ModelVersion] = []


class DashboardCounts(BaseModel):
    annotations: int
    analyses: int
    model_versions: int
    train_jobs: int
    api_keys: int


class DashboardAnnotation(BaseModel):
    annotation_id: str
    created_at: str
    original_filename: str


class DashboardAnalysis(BaseModel):
    id: int
    filename: str
    created_at: str
    model_type: str
    model_version: str
    root_count: int
    total_root_length: float


class DashboardJob(BaseModel):
    job_id: str
    status: str
    created_at: str
    planned_version_id: str
    produced_version_id: str


class DashboardResponse(BaseModel):
    counts: DashboardCounts
    current_unet_version: Optional[str] = None
    recent_annotations: List[DashboardAnnotation] = []
    recent_analyses: List[DashboardAnalysis] = []
    recent_train_jobs: List[DashboardJob] = []


class ApiKeyCreateRequest(BaseModel):
    name: str = ""


class ApiKeyCreateResponse(BaseModel):
    key_id: str
    name: str
    created_at: str
    api_key: str


class ApiKeyListItem(BaseModel):
    key_id: str
    name: str
    created_at: str
    last_used_at: Optional[str] = None
    revoked: bool


class ApiKeyListResponse(BaseModel):
    keys: List[ApiKeyListItem]


class ApiKeyRevokeResponse(BaseModel):
    key_id: str
    revoked: bool


class IngestAnalysisRequest(BaseModel):
    filename: str
    model_type: str = "unet"
    model_version: str = ""
    threshold_area: int = 0
    scaling_factor: float = 1.0
    confidence_threshold: float = 0.0
    root_count: int
    average_root_diameter: float
    total_root_length: float
    total_root_area: float
    total_root_volume: float
    extra: dict = {}


class TrainUNetRequest(BaseModel):
    epochs: int = 3
    batch_size: int = 2
    lr: float = 1e-4
    image_size: int = 512
    base_checkpoint: Optional[str] = None
    camera_model: Optional[str] = None
    only_corrected: bool = False
    preserve_aspect_ratio: bool = True


class TrainJobResponse(BaseModel):
    job_id: str
    status: str
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None
    planned_version_id: Optional[str] = None
    produced_version_id: Optional[str] = None
    log: List[str] = []


class AnnotationStatsResponse(BaseModel):
    total_annotations: int
    by_camera_model: dict[str, int]
    missing_camera_model: int


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_camera_model_from_meta(meta: dict) -> str:
    if not isinstance(meta, dict):
        return ""
    if isinstance(meta.get("camera_model"), str):
        return meta.get("camera_model") or ""
    camera = meta.get("camera") or {}
    if isinstance(camera, dict) and isinstance(camera.get("camera_model"), str):
        return camera.get("camera_model") or ""
    nested = meta.get("meta") or {}
    if isinstance(nested, dict) and isinstance(nested.get("camera_model"), str):
        return nested.get("camera_model") or ""
    mini = meta.get("minirhizotron") or {}
    if isinstance(mini, dict) and isinstance(mini.get("camera_model"), str):
        return mini.get("camera_model") or ""
    return ""


def _load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        return {"unet": {"current": None, "versions": []}}
    try:
        return json.loads(REGISTRY_PATH.read_text("utf-8"))
    except Exception:
        return {"unet": {"current": None, "versions": []}}


def _save_registry(registry: dict) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2), encoding="utf-8")


def _next_unet_version_id(registry: dict) -> str:
    versions = registry.get("unet", {}).get("versions", [])
    max_n = 0
    for v in versions:
        vid = str(v.get("id", ""))
        m = re.match(r"unet_v(\d+)$", vid)
        if m:
            max_n = max(max_n, int(m.group(1)))
    return f"unet_v{max_n + 1:04d}"


def _resolve_unet_checkpoint_path() -> Optional[str]:
    env_path = os.getenv("SUBTERRA_UNET_CHECKPOINT")
    if env_path and os.path.exists(env_path):
        return env_path

    try:
        db = _db_session()
        try:
            row = (
                db.execute(
                    select(ModelVersionRow)
                    .where(ModelVersionRow.model_type == "unet", ModelVersionRow.is_current.is_(True))
                    .order_by(ModelVersionRow.created_at.desc())
                )
                .scalars()
                .first()
            )
            if row and row.checkpoint_path and os.path.exists(row.checkpoint_path):
                return row.checkpoint_path
        finally:
            db.close()
    except Exception:
        pass

    with _registry_lock:
        registry = _load_registry()
        current = registry.get("unet", {}).get("current")
        if current:
            for v in registry.get("unet", {}).get("versions", []):
                if v.get("id") == current and v.get("checkpoint_path") and os.path.exists(v["checkpoint_path"]):
                    return v["checkpoint_path"]

    default_path = "subterra_model/models/saved_models/unet_saved.pth"
    return default_path if os.path.exists(default_path) else None


def load_saved_models():
    """Load the saved models on startup"""
    global models

    model_paths = {
        "unet": _resolve_unet_checkpoint_path(),
        "yolo": "subterra_model/models/saved_models/yolo_saved.pt",
    }

    for model_name, model_path in model_paths.items():
        if model_path and os.path.exists(model_path):
            try:
                models[model_name] = load_model(model_name, model_path, device=device)
                print(f"Loaded {model_name} model successfully")
            except Exception as e:
                print(f"Failed to load {model_name} model: {e}")
        else:
            print(f"Model file not found: {model_path}")


def preprocess_image(image: Image.Image, max_size: int = 1024) -> torch.Tensor:
    """Preprocess PIL image for model inference with automatic resizing for large images"""
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Resize large images to prevent memory issues
    width, height = image.size
    if max(width, height) > max_size:
        # Calculate new dimensions maintaining aspect ratio
        if width > height:
            new_width = max_size
            new_height = int(height * max_size / width)
        else:
            new_height = max_size
            new_width = int(width * max_size / height)

        print(f"Resizing image from {width}x{height} to {new_width}x{new_height}")
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Convert to numpy array and normalize to [0, 1]
    image_array = np.array(image).astype(np.float32) / 255.0

    # Convert to tensor and add batch dimension
    tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)

    return tensor


def postprocess_mask(mask_tensor: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    """Convert model output to binary mask"""
    mask = mask_tensor.squeeze().cpu().numpy()
    if mask.ndim == 3:
        mask = mask[0]  # Take first channel if multi-channel
    binary_mask = (mask > threshold).astype(np.uint8) * 255
    return binary_mask


def yolo_predict(model, image: torch.Tensor, confidence_threshold: float) -> np.ndarray:
    """Run YOLO inference and combine masks"""
    # Convert tensor back to PIL for YOLO
    image_np = image.squeeze().permute(1, 2, 0).cpu().numpy()
    image_pil = Image.fromarray((image_np * 255).astype(np.uint8))

    # Run prediction
    results = model.predict(image_pil, conf=confidence_threshold)

    if len(results) == 0 or results[0].masks is None:
        return np.zeros((image.shape[2], image.shape[3]), dtype=np.uint8)

    # Get masks and combine them
    masks = results[0].masks.data.cpu().numpy()
    if masks.ndim == 3:
        # Combine multiple instance masks
        combined_mask = np.max(masks, axis=0)
    else:
        combined_mask = masks

    # Convert to binary mask
    binary_mask = (combined_mask > 0).astype(np.uint8) * 255
    return binary_mask


def encode_image_to_base64(image_array: np.ndarray) -> str:
    """Encode numpy array image to base64 string"""
    if image_array.ndim == 2:
        # Grayscale mask
        image_pil = Image.fromarray(image_array.astype(np.uint8), mode='L')
    else:
        # RGB image
        image_pil = Image.fromarray(image_array.astype(np.uint8))

    buffer = io.BytesIO()
    image_pil.save(buffer, format='PNG')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{image_base64}"


@app.on_event("startup")
async def startup_event():
    """Load models when the application starts"""
    try:
        init_db()
    except Exception as e:
        # Don't prevent the API from starting if Postgres isn't ready yet.
        print(f"DB init failed (continuing without DB): {e}")
    load_saved_models()
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with _registry_lock:
        if not REGISTRY_PATH.exists():
            _save_registry({"unet": {"current": None, "versions": []}})
    try:
        db = _db_session()
        try:
            _bootstrap_db(db)
        finally:
            db.close()
    except Exception as e:
        print(f"DB bootstrap skipped: {e}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check the health of the API and loaded models"""
    loaded_models = list(models.keys())
    return HealthResponse(
        status="healthy" if loaded_models else "unhealthy",
        models_loaded=loaded_models,
        device=str(device)
    )


@app.get("/dashboard", response_model=DashboardResponse)
async def get_dashboard():
    try:
        db = _db_session()
        try:
            annotations_count = int(db.execute(select(func.count(AnnotationRow.id))).scalar_one())
            analyses_count = int(db.execute(select(func.count(AnalysisRow.id))).scalar_one())
            model_versions_count = int(db.execute(select(func.count(ModelVersionRow.id))).scalar_one())
            train_jobs_count = int(db.execute(select(func.count(TrainJobRow.id))).scalar_one())
            api_keys_count = int(db.execute(select(func.count(ApiKeyRow.id))).scalar_one())

            current_unet = _current_unet_version_id(db) or None

            recent_annotations = (
                db.execute(select(AnnotationRow).order_by(AnnotationRow.created_at.desc()).limit(10)).scalars().all()
            )
            recent_analyses = (
                db.execute(select(AnalysisRow).order_by(AnalysisRow.created_at.desc()).limit(10)).scalars().all()
            )
            recent_jobs = db.execute(select(TrainJobRow).order_by(TrainJobRow.created_at.desc()).limit(10)).scalars().all()

            return DashboardResponse(
                counts=DashboardCounts(
                    annotations=annotations_count,
                    analyses=analyses_count,
                    model_versions=model_versions_count,
                    train_jobs=train_jobs_count,
                    api_keys=api_keys_count,
                ),
                current_unet_version=current_unet,
                recent_annotations=[
                    DashboardAnnotation(
                        annotation_id=a.annotation_id,
                        created_at=a.created_at.isoformat(),
                        original_filename=a.original_filename,
                    )
                    for a in recent_annotations
                ],
                recent_analyses=[
                    DashboardAnalysis(
                        id=a.id,
                        filename=a.filename,
                        created_at=a.created_at.isoformat(),
                        model_type=a.model_type,
                        model_version=a.model_version,
                        root_count=a.root_count,
                        total_root_length=a.total_root_length,
                    )
                    for a in recent_analyses
                ],
                recent_train_jobs=[
                    DashboardJob(
                        job_id=j.job_id,
                        status=j.status,
                        created_at=j.created_at.isoformat(),
                        planned_version_id=j.planned_version_id,
                        produced_version_id=j.produced_version_id,
                    )
                    for j in recent_jobs
                ],
            )
        finally:
            db.close()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"DB unavailable: {e}")


@app.post("/api-keys", response_model=ApiKeyCreateResponse)
async def create_api_key(req: ApiKeyCreateRequest, x_admin_token: Optional[str] = Header(default=None, alias="X-Admin-Token")):
    _require_admin(x_admin_token)

    raw = f"stk_{secrets.token_urlsafe(32)}"
    key_id = uuid4().hex
    hashed = _hash_api_key(raw)

    db = _db_session()
    try:
        db.add(ApiKeyRow(key_id=key_id, name=req.name.strip(), hashed_key=hashed, revoked=False))
        db.commit()
        return ApiKeyCreateResponse(key_id=key_id, name=req.name.strip(), created_at=_utc_now_iso(), api_key=raw)
    finally:
        db.close()


@app.get("/api-keys", response_model=ApiKeyListResponse)
async def list_api_keys(x_admin_token: Optional[str] = Header(default=None, alias="X-Admin-Token")):
    _require_admin(x_admin_token)
    db = _db_session()
    try:
        rows = db.execute(select(ApiKeyRow).order_by(ApiKeyRow.created_at.desc())).scalars().all()
        return ApiKeyListResponse(
            keys=[
                ApiKeyListItem(
                    key_id=r.key_id,
                    name=r.name,
                    created_at=r.created_at.isoformat(),
                    last_used_at=r.last_used_at.isoformat() if r.last_used_at else None,
                    revoked=bool(r.revoked),
                )
                for r in rows
            ]
        )
    finally:
        db.close()


@app.post("/api-keys/{key_id}/revoke", response_model=ApiKeyRevokeResponse)
async def revoke_api_key(key_id: str, x_admin_token: Optional[str] = Header(default=None, alias="X-Admin-Token")):
    _require_admin(x_admin_token)
    db = _db_session()
    try:
        row = db.execute(select(ApiKeyRow).where(ApiKeyRow.key_id == key_id)).scalars().first()
        if not row:
            raise HTTPException(status_code=404, detail="Key not found")
        row.revoked = True
        db.commit()
        return ApiKeyRevokeResponse(key_id=key_id, revoked=True)
    finally:
        db.close()


@app.post("/ingest/analysis")
async def ingest_analysis(req: IngestAnalysisRequest, api_key: ApiKeyRow = Depends(require_api_key)):
    # Store externally computed phenotypes/metrics (no model inference).
    db = _db_session()
    try:
        db.add(
            AnalysisRow(
                filename=req.filename,
                model_type=req.model_type,
                model_version=req.model_version,
                threshold_area=req.threshold_area,
                scaling_factor=req.scaling_factor,
                confidence_threshold=req.confidence_threshold,
                root_count=req.root_count,
                average_root_diameter=req.average_root_diameter,
                total_root_length=req.total_root_length,
                total_root_area=req.total_root_area,
                total_root_volume=req.total_root_volume,
                extra={**(req.extra or {}), "ingested_via": "api_key", "api_key_id": api_key.key_id},
            )
        )
        db.commit()
        return {"success": True}
    finally:
        db.close()


@app.get("/models", response_model=ModelsResponse)
async def get_models():
    try:
        db = _db_session()
        try:
            current = _current_unet_version_id(db) or None
            rows = (
                db.execute(
                    select(ModelVersionRow)
                    .where(ModelVersionRow.model_type == "unet")
                    .order_by(ModelVersionRow.created_at.asc())
                )
                .scalars()
                .all()
            )
            versions = [
                ModelVersion(
                    id=r.version_id,
                    created_at=r.created_at.isoformat(),
                    checkpoint_path=r.checkpoint_path,
                    base_checkpoint_path=r.base_checkpoint_path or None,
                    annotations_dir=r.annotations_dir,
                    train_config=r.train_config or {},
                    metrics=r.metrics or {},
                )
                for r in rows
            ]
            return ModelsResponse(unet_current=current, unet_versions=versions)
        finally:
            db.close()
    except Exception:
        with _registry_lock:
            registry = _load_registry()
        versions = registry.get("unet", {}).get("versions", []) or []
        current = registry.get("unet", {}).get("current")
        parsed_versions: List[ModelVersion] = []
        for v in versions:
            try:
                parsed_versions.append(ModelVersion(**v))
            except Exception:
                continue
        return ModelsResponse(unet_current=current, unet_versions=parsed_versions)


def _set_job(job_id: str, **updates) -> None:
    with _train_lock:
        job = _train_jobs.get(job_id)
        if not job:
            return
        job.update(updates)
        persisted = dict(job)

    try:
        db = _db_session()
        try:
            row = db.execute(select(TrainJobRow).where(TrainJobRow.job_id == job_id)).scalars().first()
            if not row:
                return
            if "status" in persisted and persisted["status"] is not None:
                row.status = str(persisted["status"])
            if "error" in persisted and persisted["error"] is not None:
                row.error = str(persisted["error"] or "")
            if "planned_version_id" in persisted and persisted["planned_version_id"] is not None:
                row.planned_version_id = str(persisted["planned_version_id"] or "")
            if "produced_version_id" in persisted and persisted["produced_version_id"] is not None:
                row.produced_version_id = str(persisted["produced_version_id"] or "")
            if persisted.get("started_at"):
                row.started_at = datetime.fromisoformat(persisted["started_at"])
            if persisted.get("finished_at"):
                row.finished_at = datetime.fromisoformat(persisted["finished_at"])
            db.commit()
        finally:
            db.close()
    except Exception:
        pass


def _append_job_log(job_id: str, line: str) -> None:
    with _train_lock:
        job = _train_jobs.get(job_id)
        if not job:
            return
        job.setdefault("log", []).append(line)
        job["log"] = job["log"][-200:]
        log = list(job["log"])

    try:
        db = _db_session()
        try:
            row = db.execute(select(TrainJobRow).where(TrainJobRow.job_id == job_id)).scalars().first()
            if not row:
                return
            row.log = log
            db.commit()
        finally:
            db.close()
    except Exception:
        pass


def _run_unet_finetune_job(job_id: str, request: TrainUNetRequest) -> None:
    _set_job(job_id, status="running", started_at=_utc_now_iso())
    _append_job_log(job_id, f"Starting U-Net fine-tune on device={device}")

    try:
        from subterra_model.training.unet_finetune import finetune_unet_from_annotations

        base_ckpt = request.base_checkpoint or _resolve_unet_checkpoint_path()
        if not base_ckpt or not os.path.exists(base_ckpt):
            raise RuntimeError("Base checkpoint not found; set SUBTERRA_UNET_CHECKPOINT or add a trained version.")

        try:
            db = _db_session()
            try:
                version_id = _next_unet_version_id_db(db)
            finally:
                db.close()
        except Exception:
            with _registry_lock:
                registry = _load_registry()
            version_id = _next_unet_version_id(registry)

        version_dir = MODELS_DIR / "unet" / version_id
        checkpoint_path = version_dir / "unet.pth"

        _append_job_log(job_id, f"Base checkpoint: {base_ckpt}")
        _append_job_log(job_id, f"Output checkpoint: {checkpoint_path}")
        _append_job_log(job_id, f"Annotations dir: {ANNOTATIONS_DIR}")

        metrics = finetune_unet_from_annotations(
            annotations_dir=ANNOTATIONS_DIR,
            base_checkpoint_path=base_ckpt,
            output_checkpoint_path=checkpoint_path,
            device=device,
            epochs=request.epochs,
            batch_size=request.batch_size,
            lr=request.lr,
            image_size=request.image_size,
            camera_model=request.camera_model,
            only_corrected=bool(request.only_corrected),
            preserve_aspect_ratio=bool(request.preserve_aspect_ratio),
            log_fn=lambda s: _append_job_log(job_id, s),
        )

        version_record = {
            "id": version_id,
            "created_at": _utc_now_iso(),
            "checkpoint_path": str(checkpoint_path),
            "base_checkpoint_path": str(base_ckpt),
            "annotations_dir": str(ANNOTATIONS_DIR),
            "train_config": request.model_dump() if hasattr(request, "model_dump") else request.dict(),
            "metrics": metrics,
        }

        try:
            db = _db_session()
            try:
                # mark all previous as not current
                prev = (
                    db.execute(select(ModelVersionRow).where(ModelVersionRow.model_type == "unet", ModelVersionRow.is_current.is_(True)))
                    .scalars()
                    .all()
                )
                for r in prev:
                    r.is_current = False
                db.add(
                    ModelVersionRow(
                        version_id=version_id,
                        model_type="unet",
                        checkpoint_path=str(checkpoint_path),
                        base_checkpoint_path=str(base_ckpt),
                        annotations_dir=str(ANNOTATIONS_DIR),
                        train_config=version_record.get("train_config") or {},
                        metrics=metrics,
                        is_current=True,
                    )
                )
                db.commit()
            finally:
                db.close()
        except Exception:
            pass

        # Keep registry.json updated for backward compatibility
        with _registry_lock:
            registry = _load_registry()
            registry.setdefault("unet", {}).setdefault("versions", []).append(version_record)
            registry.setdefault("unet", {})["current"] = version_id
            _save_registry(registry)

        # Hot-reload UNet in memory for immediate inference
        models["unet"] = load_model("unet", str(checkpoint_path), device=device)

        _append_job_log(job_id, f"Training complete. Current version set to {version_id}")
        _set_job(job_id, status="succeeded", finished_at=_utc_now_iso(), produced_version_id=version_id)
    except Exception as e:
        _append_job_log(job_id, f"FAILED: {e}")
        _set_job(job_id, status="failed", finished_at=_utc_now_iso(), error=str(e))


@app.post("/train/unet", response_model=TrainJobResponse)
async def train_unet(req: TrainUNetRequest, _auth: Optional[ApiKeyRow] = Depends(maybe_require_api_key)):
    # prevent concurrent training
    try:
        db = _db_session()
        try:
            running = db.execute(select(func.count(TrainJobRow.id)).where(TrainJobRow.status == "running")).scalar_one()
            if int(running) > 0:
                raise HTTPException(status_code=409, detail="A training job is already running")
            planned_version = _next_unet_version_id_db(db)
        finally:
            db.close()
    except HTTPException:
        raise
    except Exception:
        with _train_lock:
            if any(j.get("status") == "running" for j in _train_jobs.values()):
                raise HTTPException(status_code=409, detail="A training job is already running")
        with _registry_lock:
            registry = _load_registry()
            planned_version = _next_unet_version_id(registry)

    job_id = uuid4().hex
    job = {
        "job_id": job_id,
        "status": "queued",
        "created_at": _utc_now_iso(),
        "started_at": None,
        "finished_at": None,
        "error": None,
        "planned_version_id": planned_version,
        "produced_version_id": None,
        "log": [f"Queued job {job_id} (planned {planned_version})"],
    }
    with _train_lock:
        _train_jobs[job_id] = job
    try:
        db = _db_session()
        try:
            db.add(
                TrainJobRow(
                    job_id=job_id,
                    status="queued",
                    error="",
                    planned_version_id=str(planned_version or ""),
                    produced_version_id="",
                    log=job["log"],
                )
            )
            db.commit()
        finally:
            db.close()
    except Exception:
        pass

    _executor.submit(_run_unet_finetune_job, job_id, req)
    return TrainJobResponse(**job)


@app.get("/train/jobs", response_model=List[TrainJobResponse])
async def list_train_jobs():
    try:
        db = _db_session()
        try:
            rows = db.execute(select(TrainJobRow).order_by(TrainJobRow.created_at.desc())).scalars().all()
            return [
                TrainJobResponse(
                    job_id=r.job_id,
                    status=r.status,
                    created_at=r.created_at.isoformat(),
                    started_at=r.started_at.isoformat() if r.started_at else None,
                    finished_at=r.finished_at.isoformat() if r.finished_at else None,
                    error=r.error or None,
                    planned_version_id=r.planned_version_id or None,
                    produced_version_id=r.produced_version_id or None,
                    log=[str(x) for x in (r.log or [])],
                )
                for r in rows
            ]
        finally:
            db.close()
    except Exception:
        with _train_lock:
            jobs = list(_train_jobs.values())
        jobs.sort(key=lambda j: j.get("created_at", ""), reverse=True)
        return [TrainJobResponse(**j) for j in jobs]


@app.get("/train/jobs/{job_id}", response_model=TrainJobResponse)
async def get_train_job(job_id: str):
    try:
        db = _db_session()
        try:
            r = db.execute(select(TrainJobRow).where(TrainJobRow.job_id == job_id)).scalars().first()
            if not r:
                raise HTTPException(status_code=404, detail="Job not found")
            return TrainJobResponse(
                job_id=r.job_id,
                status=r.status,
                created_at=r.created_at.isoformat(),
                started_at=r.started_at.isoformat() if r.started_at else None,
                finished_at=r.finished_at.isoformat() if r.finished_at else None,
                error=r.error or None,
                planned_version_id=r.planned_version_id or None,
                produced_version_id=r.produced_version_id or None,
                log=[str(x) for x in (r.log or [])],
            )
        finally:
            db.close()
    except HTTPException:
        raise
    except Exception:
        with _train_lock:
            job = _train_jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return TrainJobResponse(**job)


@app.post("/analyze", response_model=AnalysisResult)
def analyze_root_image(
    file: UploadFile = File(...),
    model_type: str = "unet",
    threshold_area: int = 50,
    scaling_factor: float = 1.0,
    confidence_threshold: float = 0.3,
    metadata_json: Optional[str] = Form(None),
    _auth: Optional[ApiKeyRow] = Depends(maybe_require_api_key),
):
    """
    Analyze a root image using the specified model.

    - **file**: Image file to analyze (PNG, JPG, JPEG)
    - **model_type**: Model to use ("unet" or "yolo")
    - **threshold_area**: Minimum area threshold for root detection
    - **scaling_factor**: Scaling factor for metric calculations
    - **confidence_threshold**: Confidence threshold for YOLO model (ignored for U-Net)
    """

    # Validate model type
    if model_type not in ["unet", "yolo"]:
        raise HTTPException(status_code=400, detail="model_type must be 'unet' or 'yolo'")

    # Check if model is loaded
    if model_type not in models:
        raise HTTPException(status_code=503, detail=f"{model_type} model not loaded")

    try:
        print(f"Processing image: {file.filename}, model: {model_type}")

        # Read and validate image
        contents = file.file.read()
        image = Image.open(io.BytesIO(contents))
        print(f"Image loaded: {image.size}, mode: {image.mode}")

        # Preprocess image
        tensor_image = preprocess_image(image)
        print(f"Tensor shape: {tensor_image.shape}")

        # Run inference
        model = models[model_type]
        print(f"Running {model_type} inference...")

        if model_type == "unet":
            with torch.no_grad():
                print("Starting U-Net forward pass...")
                output = model.model(tensor_image.to(device))
                print(f"U-Net output shape: {output.shape}")
                mask = postprocess_mask(output)
                print(f"Mask shape: {mask.shape}")
        else:  # yolo
            print("Starting YOLO inference...")
            mask = yolo_predict(model.model, tensor_image, confidence_threshold)
            print(f"YOLO mask shape: {mask.shape}")

        # Apply area thresholding
        if threshold_area > 0:
            print(f"Applying area thresholding with threshold: {threshold_area}")
            mask = threshold_mask(mask, threshold_area)

        # Calculate metrics
        print("Calculating metrics...")
        metrics = calculate_metrics(mask, scaling_factor)
        print(f"Metrics calculated: {metrics}")

        # Convert images to base64
        mask_base64 = encode_image_to_base64(mask)
        original_image_array = np.array(image)
        original_base64 = encode_image_to_base64(original_image_array)

        result = AnalysisResult(
            **metrics,
            mask_image_base64=mask_base64,
            original_image_base64=original_base64
        )

        metadata: dict = {}
        if metadata_json:
            try:
                parsed = json.loads(metadata_json)
                if isinstance(parsed, dict):
                    metadata = parsed
            except json.JSONDecodeError:
                metadata = {"metadata_json_error": "Invalid JSON", "metadata_json_raw": metadata_json}
        # Persist analysis metadata + metrics (best-effort)
        try:
            db = _db_session()
            try:
                model_version = _current_unet_version_id(db) if model_type == "unet" else ""
                db.add(
                    AnalysisRow(
                        filename=str(file.filename or ""),
                        model_type=str(model_type),
                        model_version=str(model_version or ""),
                        threshold_area=int(threshold_area),
                        scaling_factor=float(scaling_factor),
                        confidence_threshold=float(confidence_threshold),
                        root_count=int(metrics.get("root_count", 0)),
                        average_root_diameter=float(metrics.get("average_root_diameter", 0.0)),
                        total_root_length=float(metrics.get("total_root_length", 0.0)),
                        total_root_area=float(metrics.get("total_root_area", 0.0)),
                        total_root_volume=float(metrics.get("total_root_volume", 0.0)),
                        extra={"image_size": list(getattr(image, "size", ())), "meta": metadata},
                    )
                )
                db.commit()
            finally:
                db.close()
        except Exception:
            pass

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/batch-analyze")
def batch_analyze_root_images(
    files: List[UploadFile] = File(...),
    model_type: str = "unet",
    threshold_area: int = 50,
    scaling_factor: float = 1.0,
    confidence_threshold: float = 0.3,
    metadata_json: Optional[List[str]] = Form(None),
    _auth: Optional[ApiKeyRow] = Depends(maybe_require_api_key),
):
    """
    Analyze multiple root images in batch.

    Returns a list of analysis results for each image.
    """
    results = []
    for idx, file in enumerate(files):
        try:
            meta = None
            if metadata_json and idx < len(metadata_json):
                meta = metadata_json[idx]
            result = analyze_root_image(
                file=file,
                model_type=model_type,
                threshold_area=threshold_area,
                scaling_factor=scaling_factor,
                confidence_threshold=confidence_threshold,
                metadata_json=meta,
            )
            results.append({
                "filename": file.filename,
                "success": True,
                "result": result
            })
        except Exception as e:
            print(f"Error processing {file.filename}: {e}")
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })

    return {"results": results}


@app.get("/annotations/stats", response_model=AnnotationStatsResponse)
async def annotation_stats(_auth: Optional[ApiKeyRow] = Depends(maybe_require_api_key)):
    """
    Lightweight summary of saved annotations (camera metadata coverage).
    """
    total = 0
    by_camera: dict[str, int] = {}
    missing = 0

    try:
        if ANNOTATIONS_DIR.exists():
            for child in sorted(ANNOTATIONS_DIR.iterdir()):
                if not child.is_dir():
                    continue
                meta_path = child / "meta.json"
                if not meta_path.exists():
                    continue
                try:
                    meta = json.loads(meta_path.read_text("utf-8"))
                except Exception:
                    meta = {}
                total += 1
                cam = _extract_camera_model_from_meta(meta).strip()
                if not cam:
                    missing += 1
                else:
                    by_camera[cam] = by_camera.get(cam, 0) + 1
    except Exception:
        pass

    return AnnotationStatsResponse(total_annotations=total, by_camera_model=by_camera, missing_camera_model=missing)


@app.post("/annotations", response_model=AnnotationSaveResponse)
async def save_annotation_pair(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    original_filename: str = Form(""),
    metadata_json: Optional[str] = Form(None),
    _auth: Optional[ApiKeyRow] = Depends(maybe_require_api_key),
):
    """
    Save an (image, mask) pair to disk for human-in-the-loop fine-tuning.

    Writes under `SUBTERRA_ANNOTATIONS_DIR` (default: `data/annotations/`) and stores `meta.json`.
    """
    annotation_id = uuid4().hex
    image_name = _safe_filename(original_filename or image.filename or "image")
    mask_name = _safe_filename(mask.filename or "mask.png")

    annotation_dir = ANNOTATIONS_DIR / annotation_id
    image_bytes = await image.read()
    mask_bytes = await mask.read()

    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
        msk = Image.open(io.BytesIO(mask_bytes))
        msk.load()
        img_w, img_h = img.size
        msk_w, msk_h = msk.size
        if (img_w, img_h) != (msk_w, msk_h):
            raise HTTPException(
                status_code=400,
                detail=f"Mask size {msk_w}x{msk_h} must match image size {img_w}x{img_h}",
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image or mask: {e}")

    annotation_dir.mkdir(parents=True, exist_ok=True)

    image_path = annotation_dir / image_name
    mask_path = annotation_dir / mask_name
    metadata_path = annotation_dir / "meta.json"

    _write_bytes(image_path, image_bytes)
    _write_bytes(mask_path, mask_bytes)

    metadata: dict = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "annotation_id": annotation_id,
        "original_filename": original_filename or image.filename or "",
        "image_filename": image_name,
        "mask_filename": mask_name,
        "image_width_px": img_w,
        "image_height_px": img_h,
        "mask_width_px": msk_w,
        "mask_height_px": msk_h,
    }
    if metadata_json:
        try:
            metadata.update(json.loads(metadata_json))
        except json.JSONDecodeError:
            metadata["metadata_json_error"] = "Invalid JSON; stored raw string"
            metadata["metadata_json_raw"] = metadata_json

    _write_bytes(metadata_path, json.dumps(metadata, indent=2).encode("utf-8"))

    # Persist annotation record (best-effort)
    try:
        db = _db_session()
        try:
            db.add(
                AnnotationRow(
                    annotation_id=annotation_id,
                    original_filename=str(metadata.get("original_filename") or ""),
                    image_path=str(image_path),
                    mask_path=str(mask_path),
                    meta=metadata,
                )
            )
            db.commit()
        finally:
            db.close()
    except Exception:
        pass

    return AnnotationSaveResponse(
        annotation_id=annotation_id,
        image_path=str(image_path),
        mask_path=str(mask_path),
        metadata_path=str(metadata_path),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
