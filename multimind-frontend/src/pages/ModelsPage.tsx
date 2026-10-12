import { useState, type FormEvent } from "react";
import {
  CompareResponse,
  ModelHealth,
  checkHealth,
  getModels,
  postCompare,
} from "../api/client";
import { toApiError, useApi } from "../api/useApi";
import BackendDown from "../components/BackendDown";

export default function ModelsPage() {
  const { state: modelsState, reload } = useApi(getModels);

  const [health, setHealth] = useState<Record<string, ModelHealth> | null>(null);
  const [healthBusy, setHealthBusy] = useState(false);
  const [healthError, setHealthError] = useState<string | null>(null);

  const [prompt, setPrompt] = useState("");
  const [selected, setSelected] = useState<string[]>([]);
  const [compareBusy, setCompareBusy] = useState(false);
  const [compareError, setCompareError] = useState<string | null>(null);
  const [compareResult, setCompareResult] = useState<CompareResponse | null>(null);

  if (modelsState.status === "loading") return <p className="muted">Loading models...</p>;
  if (modelsState.status === "error") {
    return (
      <>
        <h1 className="page-header">Models</h1>
        <BackendDown backend="gateway" error={modelsState.error} />
      </>
    );
  }

  const models = Object.entries(modelsState.data.models);

  const runHealthCheck = async () => {
    setHealthBusy(true);
    setHealthError(null);
    try {
      setHealth(await checkHealth());
    } catch (err) {
      setHealthError(toApiError(err).detail);
    } finally {
      setHealthBusy(false);
    }
  };

  const toggleSelected = (name: string) => {
    setSelected((current) =>
      current.includes(name) ? current.filter((m) => m !== name) : [...current, name],
    );
  };

  const runCompare = async (e: FormEvent) => {
    e.preventDefault();
    if (!prompt.trim() || selected.length === 0 || compareBusy) return;
    setCompareBusy(true);
    setCompareError(null);
    setCompareResult(null);
    try {
      setCompareResult(await postCompare({ prompt: prompt.trim(), models: selected }));
    } catch (err) {
      setCompareError(toApiError(err).detail);
    } finally {
      setCompareBusy(false);
    }
  };

  return (
    <>
      <h1 className="page-header">Models</h1>
      <p className="page-sub">Catalog, health, and side-by-side comparison via the gateway.</p>

      <div className="card">
        <div className="form-row" style={{ marginBottom: "0.75rem", justifyContent: "space-between" }}>
          <h2 style={{ margin: 0 }}>Catalog</h2>
          <div style={{ display: "flex", gap: "0.5rem" }}>
            <button className="secondary" onClick={reload}>
              Refresh
            </button>
            <button onClick={runHealthCheck} disabled={healthBusy}>
              {healthBusy ? "Checking..." : "Check health"}
            </button>
          </div>
        </div>
        {healthError ? (
          <div className="banner danger" role="alert">
            Health check failed: {healthError}
          </div>
        ) : null}
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>Status</th>
                <th>Backing model</th>
                <th className="num">Temperature</th>
                <th>Health</th>
              </tr>
            </thead>
            <tbody>
              {models.map(([name, info]) => {
                const h = health?.[name];
                return (
                  <tr key={name}>
                    <td>{name}</td>
                    <td>
                      <span className={`pill ${info.status === "available" ? "ok" : "danger"}`}>
                        {info.status}
                      </span>
                    </td>
                    <td>{info.config.model_name}</td>
                    <td className="num">{info.config.temperature}</td>
                    <td>
                      {h ? (
                        <span className={`pill ${h.is_healthy ? "ok" : "danger"}`}>
                          {h.is_healthy ? "healthy" : (h.error_message ?? "unhealthy")}
                          {h.latency_ms != null ? ` (${h.latency_ms.toFixed(0)}ms)` : ""}
                        </span>
                      ) : (
                        <span className="muted">not checked</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card">
        <h2>Compare</h2>
        <p className="page-sub" style={{ marginBottom: "0.75rem" }}>
          Run the same prompt against multiple models. Unavailable models are skipped server-side.
        </p>
        <form onSubmit={runCompare}>
          <div className="field">
            <label htmlFor="compare-prompt">Prompt</label>
            <textarea
              id="compare-prompt"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Write a limerick about vector databases."
            />
          </div>
          <div className="field">
            <label>Models</label>
            <div className="checkbox-grid">
              {models.map(([name]) => (
                <label key={name}>
                  <input
                    type="checkbox"
                    checked={selected.includes(name)}
                    onChange={() => toggleSelected(name)}
                  />
                  {name}
                </label>
              ))}
            </div>
          </div>
          <button type="submit" disabled={compareBusy || !prompt.trim() || selected.length === 0}>
            {compareBusy ? "Comparing..." : "Compare"}
          </button>
        </form>

        {compareError ? (
          <div className="banner danger" role="alert" style={{ marginTop: "1rem" }}>
            {compareError}
          </div>
        ) : null}

        {compareResult ? (
          <div className="grid-2" style={{ marginTop: "1rem" }}>
            {Object.entries(compareResult.responses).map(([name, resp]) => (
              <div className="card" key={name}>
                <h2>{name}</h2>
                <pre className="output-panel">{resp.content}</pre>
              </div>
            ))}
          </div>
        ) : null}
      </div>
    </>
  );
}
