"use client";

import React, { useEffect, useMemo, useState } from "react";
import { ProgressBar } from "../_components/ProgressBar";

const GENO_BASE = process.env.NEXT_PUBLIC_GENOTYPE_BASE_URL ?? "http://localhost:8002";
const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8001";

type Stats = { samples: number; markers: number; values: number };
type UploadResp = { samples_upserted: number; markers_upserted: number; values_upserted: number };
type MappingRow = {
  marker_name: string;
  n: number;
  effect?: number | null;
  p_value?: number | null;
  p_adjusted?: number | null;
  r2?: number | null;
  lod?: number | null;
};
type MappingRunResp = { phenotype_field: string; method: string; p_adjust: string; rows: MappingRow[] };
type HistoryItem = { created_at: string; phenotype_field: string; marker_name: string; n: number; effect: number };

function fmtR(r: number) {
  if (!Number.isFinite(r)) return "—";
  return r.toFixed(4);
}

function fmtP(p: number | null | undefined) {
  if (p == null || !Number.isFinite(p)) return "—";
  if (p === 0) return "0";
  if (p < 1e-4) return p.toExponential(2);
  return p.toFixed(6);
}

export default function Page() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [error, setError] = useState<string | null>(null);

  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<UploadResp | null>(null);

  const [phenotypeField, setPhenotypeField] = useState<string>("total_root_length");
  const [method, setMethod] = useState<string>("linear");
  const [pAdjust, setPAdjust] = useState<string>("bh");
  const [running, setRunning] = useState(false);
  const [runResult, setRunResult] = useState<MappingRunResp | null>(null);

  async function refresh() {
    try {
      const [s, h] = await Promise.all([
        fetch(`${GENO_BASE}/stats`).then((r) => (r.ok ? r.json() : null)),
        fetch(`${GENO_BASE}/mapping/history?limit=25`).then((r) => (r.ok ? r.json() : []))
      ]);
      setStats(s);
      setHistory(h);
    } catch {
      // ignore; error shown on interactions
    }
  }

  useEffect(() => {
    void refresh();
    const id = setInterval(refresh, 5000);
    return () => clearInterval(id);
  }, []);

  async function uploadCsv(file: File) {
    setError(null);
    setUploadResult(null);
    setUploading(true);
    try {
      const form = new FormData();
      form.append("file", file, file.name);
      const res = await fetch(`${GENO_BASE}/markers/upload`, { method: "POST", body: form });
      if (!res.ok) throw new Error(await res.text());
      setUploadResult((await res.json()) as UploadResp);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setUploading(false);
    }
  }

  async function runMapping() {
    setError(null);
    setRunning(true);
    setRunResult(null);
    try {
      const res = await fetch(`${GENO_BASE}/mapping/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ phenotype_field: phenotypeField, method, p_adjust: pAdjust })
      });
      if (!res.ok) throw new Error(await res.text());
      setRunResult((await res.json()) as MappingRunResp);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  }

  const top10 = useMemo(() => (runResult?.rows ?? []).slice(0, 10), [runResult]);
  const effectLabel =
    method === "pearson" ? "r" : method === "linear" ? "beta" : method === "anova" ? "F" : "F";

  return (
    <main style={{ display: "grid", gap: 16 }}>
      <header className="panel">
        <h1 style={{ margin: 0, fontSize: 18 }}>Genotyping / Mapping</h1>
        <p style={{ margin: "6px 0 0", color: "var(--muted)", fontSize: 13 }}>
          Upload genetic marker CSV, then map stored phenotypes (from the API analyses table) to markers.
        </p>
        <div className="hint" style={{ marginTop: 8 }}>
          Genotype service: {GENO_BASE} · API: {API_BASE}
        </div>
        {error ? <div style={{ marginTop: 10, color: "var(--danger)", fontSize: 13 }}>{error}</div> : null}
      </header>

      <section className="cards">
        <div className="panel">
          <div className="panelTitle">Samples</div>
          <div style={{ fontSize: 22 }}>{stats?.samples ?? "—"}</div>
          <div className="hint">Unique sample_id rows</div>
        </div>
        <div className="panel">
          <div className="panelTitle">Markers</div>
          <div style={{ fontSize: 22 }}>{stats?.markers ?? "—"}</div>
          <div className="hint">Marker columns</div>
        </div>
        <div className="panel">
          <div className="panelTitle">Values</div>
          <div style={{ fontSize: 22 }}>{stats?.values ?? "—"}</div>
          <div className="hint">sample×marker entries</div>
        </div>
        <div className="panel">
          <div className="panelTitle">Phenotypes</div>
          <div style={{ fontSize: 22 }}>DB-backed</div>
          <div className="hint">From `analyses` in Postgres</div>
        </div>
      </section>

      <section className="grid">
        <div className="panel">
          <div className="panelTitle">1) Upload Genetic Marker CSV</div>
          <ProgressBar active={uploading} label={uploading ? "Uploading…" : undefined} reserveSpace />
          <div className="hint">
            CSV format: first column `sample_id` (or `filename`), then marker columns with numeric values (0/1/2 or any float).
            Sample IDs should match the phenotype `filename` values stored by Subterra.
          </div>
          <div className="row" style={{ marginTop: 12 }}>
            <input
              className="input"
              type="file"
              accept=".csv,text/csv"
              disabled={uploading}
              onChange={(e) => {
                const f = e.currentTarget.files?.[0];
                if (!f) return;
                void uploadCsv(f);
              }}
            />
          </div>
          {uploadResult ? (
            <div className="hint" style={{ marginTop: 10 }}>
              Upserted: samples {uploadResult.samples_upserted}, markers {uploadResult.markers_upserted}, values {uploadResult.values_upserted}
            </div>
          ) : null}
        </div>

        <div className="panel">
          <div className="panelTitle">2) Run Mapping</div>
          <ProgressBar active={running} label={running ? "Running mapping…" : undefined} reserveSpace />
          <div className="row" style={{ marginTop: 10 }}>
            <div className="field">
              <div className="label">Phenotype Field</div>
              <select className="select" value={phenotypeField} onChange={(e) => setPhenotypeField(e.target.value)}>
                <option value="total_root_length">total_root_length</option>
                <option value="root_count">root_count</option>
                <option value="average_root_diameter">average_root_diameter</option>
                <option value="total_root_area">total_root_area</option>
                <option value="total_root_volume">total_root_volume</option>
              </select>
            </div>
            <div className="field">
              <div className="label">Method</div>
              <select className="select" value={method} onChange={(e) => setMethod(e.target.value)}>
                <option value="linear">GWAS (linear regression)</option>
                <option value="anova">QTL (ANOVA across genotypes)</option>
                <option value="lod">QTL (LOD score)</option>
                <option value="pearson">Correlation (Pearson)</option>
              </select>
            </div>
            <div className="field">
              <div className="label">P adjust</div>
              <select className="select" value={pAdjust} onChange={(e) => setPAdjust(e.target.value)}>
                <option value="bh">FDR (BH)</option>
                <option value="bonferroni">Bonferroni</option>
                <option value="none">None</option>
              </select>
            </div>
            <button className="btn btnPrimary" onClick={() => void runMapping()} disabled={running || uploading}>
              Run
            </button>
          </div>

          {top10.length ? (
            <div style={{ marginTop: 12, overflow: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                <thead>
                  <tr>
                    {["marker", "n", effectLabel, "p", "p_adj", "r2", "lod"].map((h) => (
                      <th
                        key={h}
                        style={{
                          textAlign: "left",
                          padding: "8px 6px",
                          color: "var(--muted)",
                          borderBottom: "1px solid var(--border)"
                        }}
                      >
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {top10.map((r) => (
                    <tr key={r.marker_name}>
                      <td style={{ padding: "8px 6px", borderBottom: "1px solid var(--border)" }}>{r.marker_name}</td>
                      <td style={{ padding: "8px 6px", borderBottom: "1px solid var(--border)" }}>{r.n}</td>
                      <td style={{ padding: "8px 6px", borderBottom: "1px solid var(--border)" }}>
                        {r.effect == null ? "—" : fmtR(r.effect)}
                      </td>
                      <td style={{ padding: "8px 6px", borderBottom: "1px solid var(--border)" }}>{fmtP(r.p_value ?? null)}</td>
                      <td style={{ padding: "8px 6px", borderBottom: "1px solid var(--border)" }}>{fmtP(r.p_adjusted ?? null)}</td>
                      <td style={{ padding: "8px 6px", borderBottom: "1px solid var(--border)" }}>
                        {r.r2 == null ? "—" : fmtR(r.r2)}
                      </td>
                      <td style={{ padding: "8px 6px", borderBottom: "1px solid var(--border)" }}>
                        {r.lod == null ? "—" : fmtR(r.lod)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="hint" style={{ marginTop: 12 }}>
              Run mapping to see top associated markers.
            </div>
          )}
        </div>
      </section>

      <section className="panel">
        <div className="panelTitle">Recent Mapping History</div>
        {history.length ? (
          <div style={{ overflow: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
              <thead>
                <tr>
                  {["created_at", "phenotype", "marker", "n", "effect"].map((h) => (
                    <th
                      key={h}
                      style={{
                        textAlign: "left",
                        padding: "8px 6px",
                        color: "var(--muted)",
                        borderBottom: "1px solid var(--border)"
                      }}
                    >
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {history.map((r, idx) => (
                  <tr key={`${r.created_at}-${r.marker_name}-${idx}`}>
                    <td style={{ padding: "8px 6px", borderBottom: "1px solid var(--border)" }}>{r.created_at}</td>
                    <td style={{ padding: "8px 6px", borderBottom: "1px solid var(--border)" }}>{r.phenotype_field}</td>
                    <td style={{ padding: "8px 6px", borderBottom: "1px solid var(--border)" }}>{r.marker_name}</td>
                    <td style={{ padding: "8px 6px", borderBottom: "1px solid var(--border)" }}>{r.n}</td>
                    <td style={{ padding: "8px 6px", borderBottom: "1px solid var(--border)" }}>{fmtR(r.effect)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="hint">No mapping runs yet.</div>
        )}
      </section>
    </main>
  );
}
