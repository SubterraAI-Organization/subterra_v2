from __future__ import annotations

import csv
import io
import math
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scipy import stats
from sqlalchemy import DateTime, Float, Integer, String, UniqueConstraint, create_engine, func, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/subterra_genotype.sqlite3")
if DATABASE_URL.startswith("sqlite:"):
    Path("data").mkdir(parents=True, exist_ok=True)


def _connect_args(url: str) -> dict:
    if url.startswith("sqlite:"):
        return {"check_same_thread": False}
    return {}


engine = create_engine(DATABASE_URL, pool_pre_ping=True, connect_args=_connect_args(DATABASE_URL))
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


class GenotypeMarker(Base):
    __tablename__ = "genotype_markers"
    __table_args__ = (UniqueConstraint("name", name="uq_genotype_markers_name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, nullable=False)


class GenotypeSample(Base):
    __tablename__ = "genotype_samples"
    __table_args__ = (UniqueConstraint("sample_id", name="uq_genotype_samples_sample_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sample_id: Mapped[str] = mapped_column(String(512), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, nullable=False)


class GenotypeValue(Base):
    __tablename__ = "genotype_values"
    __table_args__ = (UniqueConstraint("sample_id", "marker_id", name="uq_genotype_values_sample_marker"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sample_id: Mapped[str] = mapped_column(String(512), nullable=False)
    marker_id: Mapped[int] = mapped_column(Integer, nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, nullable=False)


class MappingResult(Base):
    __tablename__ = "mapping_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, nullable=False)

    phenotype_field: Mapped[str] = mapped_column(String(128), nullable=False)
    marker_name: Mapped[str] = mapped_column(String(256), nullable=False)
    n: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    pearson_r: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)


app = FastAPI(
    title="Subterra Genotype Mapping Service",
    description="Stores genetic marker tables and maps phenotype metrics to markers",
    version="0.1.0",
)

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


@app.on_event("startup")
def _startup() -> None:
    Base.metadata.create_all(bind=engine)


class HealthResponse(BaseModel):
    status: str
    database_url: str


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", database_url=DATABASE_URL)


class UploadMarkersResponse(BaseModel):
    samples_upserted: int
    markers_upserted: int
    values_upserted: int


def _parse_marker_csv(contents: bytes) -> Tuple[List[str], List[Dict[str, str]]]:
    text_data = contents.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text_data))
    if not reader.fieldnames:
        raise ValueError("CSV has no header")
    fieldnames = [f.strip() for f in reader.fieldnames if f is not None]
    if len(fieldnames) < 2:
        raise ValueError("CSV must contain at least: sample_id, <marker1>, ...")
    if fieldnames[0].lower() not in {"sample_id", "sample", "id", "filename"}:
        raise ValueError("First column must be sample_id (or sample/id/filename)")
    rows: List[Dict[str, str]] = []
    for r in reader:
        if not r:
            continue
        rows.append({(k or "").strip(): (v or "").strip() for k, v in r.items()})
    return fieldnames, rows


@app.post("/markers/upload", response_model=UploadMarkersResponse)
async def upload_markers(file: UploadFile = File(...)) -> UploadMarkersResponse:
    contents = await file.read()
    try:
        fieldnames, rows = _parse_marker_csv(contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    sample_col = fieldnames[0]
    marker_names = fieldnames[1:]

    with SessionLocal() as db:
        # markers
        existing_markers = {
            m.name: m
            for m in db.execute(select(GenotypeMarker).where(GenotypeMarker.name.in_(marker_names))).scalars().all()
        }
        markers_upserted = 0
        for name in marker_names:
            if name not in existing_markers:
                m = GenotypeMarker(name=name)
                db.add(m)
                existing_markers[name] = m
                markers_upserted += 1

        # samples
        sample_ids = [r.get(sample_col, "") for r in rows if r.get(sample_col, "")]
        existing_samples = {
            s.sample_id: s
            for s in db.execute(select(GenotypeSample).where(GenotypeSample.sample_id.in_(sample_ids))).scalars().all()
        }
        samples_upserted = 0
        for sid in sample_ids:
            if sid and sid not in existing_samples:
                s = GenotypeSample(sample_id=sid)
                db.add(s)
                existing_samples[sid] = s
                samples_upserted += 1

        db.flush()

        # values: use upsert style (delete existing then insert) for simplicity
        values_upserted = 0
        for r in rows:
            sid = r.get(sample_col, "")
            if not sid:
                continue
            for name in marker_names:
                raw = r.get(name, "")
                if raw == "":
                    continue
                try:
                    val = float(raw)
                except ValueError:
                    continue
                marker_id = existing_markers[name].id
                if marker_id is None:
                    continue
                db.execute(
                    text(
                        "INSERT INTO genotype_values (sample_id, marker_id, value, created_at) "
                        "VALUES (:sample_id, :marker_id, :value, :created_at) "
                        "ON CONFLICT (sample_id, marker_id) DO UPDATE SET value = EXCLUDED.value"
                    ),
                    {"sample_id": sid, "marker_id": int(marker_id), "value": float(val), "created_at": _utc_now()},
                )
                values_upserted += 1

        db.commit()

    return UploadMarkersResponse(
        samples_upserted=samples_upserted,
        markers_upserted=markers_upserted,
        values_upserted=values_upserted,
    )


class MappingRunRequest(BaseModel):
    phenotype_field: str = "total_root_length"
    method: str = "linear"  # pearson | linear | anova | lod
    p_adjust: str = "bh"  # none | bonferroni | bh
    max_markers: int = 5000
    min_n: int = 6


class MappingRow(BaseModel):
    marker_name: str
    n: int
    effect: Optional[float] = None
    p_value: Optional[float] = None
    p_adjusted: Optional[float] = None
    r2: Optional[float] = None
    lod: Optional[float] = None


class MappingRunResponse(BaseModel):
    phenotype_field: str
    method: str
    p_adjust: str
    rows: List[MappingRow]


def _adjust_pvalues(p_values: List[Optional[float]], method: str) -> List[Optional[float]]:
    p = [v for v in p_values]
    if method == "none":
        return p

    idx = [i for i, v in enumerate(p) if v is not None]
    if not idx:
        return p

    if method == "bonferroni":
        m = len(idx)
        for i in idx:
            p[i] = min(1.0, float(p[i]) * m)  # type: ignore[arg-type]
        return p

    if method != "bh":
        raise ValueError("Unsupported p_adjust method")

    # Benjamini–Hochberg FDR
    sorted_idx = sorted(idx, key=lambda i: float(p[i]))  # type: ignore[arg-type]
    q = [None] * m
    prev = 1.0
    for rank, i in enumerate(reversed(sorted_idx), start=1):
        # reverse traversal to enforce monotonicity
        # rank in reversed order means i has larger p; compute on original rank:
        orig_rank = len(sorted_idx) - rank + 1
        pv = float(p[i])  # type: ignore[arg-type]
        val = min(prev, pv * len(sorted_idx) / orig_rank)
        prev = val
        q[i] = min(1.0, val)
    for i in idx:
        if q[i] is not None:
            p[i] = float(q[i])
    return p


def _as_float_list(vals: List[float]) -> Tuple[List[float], bool]:
    xs = [float(v) for v in vals]
    var = float(stats.tvar(xs)) if len(xs) >= 2 else 0.0
    return xs, var > 0


def _assoc_pearson(xs: List[float], ys: List[float]) -> Optional[MappingRow]:
    n = len(xs)
    if n < 3:
        return None
    xs_f, okx = _as_float_list(xs)
    ys_f, oky = _as_float_list(ys)
    if not okx or not oky:
        return None
    r, p = stats.pearsonr(xs_f, ys_f)
    return MappingRow(marker_name="", n=n, effect=float(r), p_value=float(p), r2=float(r * r))


def _assoc_linear(xs: List[float], ys: List[float]) -> Optional[MappingRow]:
    # y = b0 + b1*x; p-value for b1
    n = len(xs)
    if n < 3:
        return None
    xs_f, okx = _as_float_list(xs)
    ys_f, oky = _as_float_list(ys)
    if not okx or not oky:
        return None

    x = xs_f
    y = ys_f
    mx = sum(x) / n
    my = sum(y) / n
    sxx = sum((xi - mx) ** 2 for xi in x)
    if sxx == 0:
        return None
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    b1 = sxy / sxx
    b0 = my - b1 * mx
    resid = [yi - (b0 + b1 * xi) for xi, yi in zip(x, y)]
    rss = sum(r * r for r in resid)
    tss = sum((yi - my) ** 2 for yi in y)
    df = n - 2
    if df <= 0:
        return None
    sigma2 = rss / df
    se_b1 = (sigma2 / sxx) ** 0.5
    if se_b1 == 0:
        return None
    t_stat = b1 / se_b1
    p = float(2 * stats.t.sf(abs(t_stat), df=df))
    r2 = 0.0 if tss == 0 else float(1 - rss / tss)
    return MappingRow(marker_name="", n=n, effect=float(b1), p_value=p, r2=r2)


def _assoc_anova(xs: List[float], ys: List[float]) -> Optional[MappingRow]:
    # one-way ANOVA across genotype groups (requires at least 2 groups)
    n = len(xs)
    if n < 3:
        return None
    groups: Dict[int, List[float]] = {}
    for x, y in zip(xs, ys):
        try:
            g = int(round(float(x)))
        except Exception:
            continue
        groups.setdefault(g, []).append(float(y))
    if len(groups) < 2:
        return None
    arrays = [groups[k] for k in sorted(groups.keys()) if len(groups[k]) >= 2]
    if len(arrays) < 2:
        return None
    f_stat, p = stats.f_oneway(*arrays)
    return MappingRow(marker_name="", n=n, effect=float(f_stat), p_value=float(p))


def _assoc_lod(xs: List[float], ys: List[float]) -> Optional[MappingRow]:
    # QTL-style LOD from regression y~x vs null mean-only; also provide F-test p-value
    n = len(xs)
    if n < 3:
        return None
    xs_f, okx = _as_float_list(xs)
    ys_f, oky = _as_float_list(ys)
    if not okx or not oky:
        return None
    x = xs_f
    y = ys_f
    my = sum(y) / n
    rss0 = sum((yi - my) ** 2 for yi in y)

    mx = sum(x) / n
    sxx = sum((xi - mx) ** 2 for xi in x)
    if sxx == 0:
        return None
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    b1 = sxy / sxx
    b0 = my - b1 * mx
    rss1 = sum((yi - (b0 + b1 * xi)) ** 2 for xi, yi in zip(x, y))
    if rss1 <= 0 or rss0 <= 0:
        return None
    lod = float((n / 2.0) * math.log10(rss0 / rss1))
    # F test
    df1 = 1
    df2 = n - 2
    if df2 <= 0:
        return MappingRow(marker_name="", n=n, lod=lod)
    f = ((rss0 - rss1) / df1) / (rss1 / df2) if rss1 > 0 else 0.0
    p = float(stats.f.sf(f, df1, df2)) if f >= 0 else None
    return MappingRow(marker_name="", n=n, lod=lod, p_value=p, effect=float(f))


def _load_phenotypes(db: Session, field: str) -> Dict[str, float]:
    # Reads from the API's `analyses` table (created by api.py). This service expects to share the same Postgres DB.
    # We join on filename; users should keep sample_id in markers CSV equal to analysis filename.
    if field not in {
        "root_count",
        "average_root_diameter",
        "total_root_length",
        "total_root_area",
        "total_root_volume",
    }:
        raise ValueError("Unsupported phenotype_field")

    # Attempt exact filename match first.
    # NOTE: This assumes the API service has created the `analyses` table.
    rows = db.execute(text(f"SELECT filename, {field} FROM analyses ORDER BY created_at DESC")).all()
    out: Dict[str, float] = {}
    for filename, value in rows:
        if filename and filename not in out and value is not None:
            try:
                out[str(filename)] = float(value)
            except Exception:
                continue
    return out


@app.post("/mapping/run", response_model=MappingRunResponse)
def run_mapping(req: MappingRunRequest) -> MappingRunResponse:
    with SessionLocal() as db:
        try:
            phen = _load_phenotypes(db, req.phenotype_field)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read phenotypes from DB: {e}")

        if not phen:
            raise HTTPException(status_code=400, detail="No phenotypes found in DB (run phenotyping first)")

        marker_rows = db.execute(select(GenotypeMarker).order_by(GenotypeMarker.id.asc()).limit(req.max_markers)).scalars().all()
        if not marker_rows:
            raise HTTPException(status_code=400, detail="No genotype markers found (upload markers CSV first)")

        method = req.method.lower().strip()
        if method not in {"pearson", "linear", "anova", "lod"}:
            raise HTTPException(status_code=400, detail="method must be one of: pearson, linear, anova, lod")
        p_adjust = req.p_adjust.lower().strip()
        if p_adjust not in {"none", "bonferroni", "bh"}:
            raise HTTPException(status_code=400, detail="p_adjust must be one of: none, bonferroni, bh")

        # For each marker, gather sample overlap and compute correlation.
        results: List[MappingRow] = []
        for marker in marker_rows:
            values = db.execute(
                select(GenotypeValue.sample_id, GenotypeValue.value).where(GenotypeValue.marker_id == marker.id)
            ).all()
            xs: List[float] = []
            ys: List[float] = []
            for sid, x in values:
                if sid in phen:
                    xs.append(float(x))
                    ys.append(float(phen[sid]))
            if len(xs) < req.min_n:
                continue
            row: Optional[MappingRow]
            if method == "pearson":
                row = _assoc_pearson(xs, ys)
            elif method == "linear":
                row = _assoc_linear(xs, ys)
            elif method == "anova":
                row = _assoc_anova(xs, ys)
            else:  # lod
                row = _assoc_lod(xs, ys)
            if row is None:
                continue
            payload = row.model_dump() if hasattr(row, "model_dump") else row.dict()  # pydantic v2/v1 compat
            payload["marker_name"] = marker.name
            results.append(MappingRow(**payload))

        pvals = [r.p_value for r in results]
        try:
            padj = _adjust_pvalues(pvals, p_adjust)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        for r, q in zip(results, padj):
            r.p_adjusted = q

        if method == "pearson":
            results.sort(key=lambda r: abs(r.effect or 0.0), reverse=True)
        else:
            results.sort(key=lambda r: (r.p_adjusted if r.p_adjusted is not None else 1.0, r.p_value if r.p_value is not None else 1.0))

        # persist top results
        try:
            for row in results[:200]:
                db.add(
                    MappingResult(
                        phenotype_field=req.phenotype_field,
                        marker_name=row.marker_name,
                        n=row.n,
                        pearson_r=float(row.effect or 0.0),
                    )
                )
            db.commit()
        except Exception:
            db.rollback()

        return MappingRunResponse(phenotype_field=req.phenotype_field, method=method, p_adjust=p_adjust, rows=results[:200])


class MappingHistoryItem(BaseModel):
    created_at: str
    phenotype_field: str
    marker_name: str
    n: int
    effect: float


@app.get("/mapping/history", response_model=List[MappingHistoryItem])
def mapping_history(limit: int = 50) -> List[MappingHistoryItem]:
    with SessionLocal() as db:
        rows = db.execute(select(MappingResult).order_by(MappingResult.created_at.desc()).limit(limit)).scalars().all()
        return [
            MappingHistoryItem(
                created_at=r.created_at.isoformat(),
                phenotype_field=r.phenotype_field,
                marker_name=r.marker_name,
                n=r.n,
                effect=r.pearson_r,
            )
            for r in rows
        ]


class GenotypeStats(BaseModel):
    samples: int
    markers: int
    values: int


@app.get("/stats", response_model=GenotypeStats)
def stats() -> GenotypeStats:
    with SessionLocal() as db:
        samples = int(db.execute(select(func.count(GenotypeSample.id))).scalar_one())
        markers = int(db.execute(select(func.count(GenotypeMarker.id))).scalar_one())
        values = int(db.execute(select(func.count(GenotypeValue.id))).scalar_one())
        return GenotypeStats(samples=samples, markers=markers, values=values)
