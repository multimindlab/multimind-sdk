import { FormEvent, useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { getProjects, getRuns, tokenTotal } from "../../api/platform";
import { useApi } from "../../api/useApi";
import EmptyState from "../../components/EmptyState";
import PlatformDown from "../../components/PlatformDown";
import Thumb from "../../components/Thumb";
import { fmtNum, fmtTimestamp, fmtUSD } from "../../components/format";

const PAGE_SIZE = 50;
const RUN_TYPES = ["llm", "chain", "tool"];
const REFRESH_MS = 10000;

export function fmtLatency(ms: number | null): string {
  if (ms === null) return "-";
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)}s` : `${Math.round(ms)}ms`;
}

export default function TracesPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const project = searchParams.get("project") ?? "";
  const [status, setStatus] = useState("");
  const [runType, setRunType] = useState("");
  const [q, setQ] = useState("");
  const [searchText, setSearchText] = useState("");
  const [offset, setOffset] = useState(0);
  const [autoRefresh, setAutoRefresh] = useState(false);

  const projectsState = useApi(getProjects).state;
  const projects = projectsState.status === "ready" ? projectsState.data.projects : [];

  const { state, reload } = useApi(
    () =>
      getRuns({
        project: project || undefined,
        status: status || undefined,
        run_type: runType || undefined,
        q: q || undefined,
        limit: PAGE_SIZE,
        offset,
      }),
    [project, status, runType, q, offset],
  );

  useEffect(() => {
    if (!autoRefresh) return;
    const timer = setInterval(reload, REFRESH_MS);
    return () => clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoRefresh]);

  const submitSearch = (e: FormEvent) => {
    e.preventDefault();
    setQ(searchText.trim());
    setOffset(0);
  };

  const data = state.status === "ready" ? state.data : null;

  return (
    <>
      <h1 className="page-header">Traces</h1>
      <p className="page-sub">Runs recorded by the platform server (newest first).</p>

      <div className="card">
        <form className="form-row" style={{ marginBottom: "0.9rem" }} onSubmit={submitSearch}>
          <div>
            <label htmlFor="trace-project">Project</label>
            <select
              id="trace-project"
              value={project}
              onChange={(e) => {
                setSearchParams(e.target.value ? { project: e.target.value } : {});
                setOffset(0);
              }}
            >
              <option value="">All projects</option>
              {projects.map((p) => (
                <option key={p.name} value={p.name}>
                  {p.name}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label htmlFor="trace-status">Status</label>
            <select
              id="trace-status"
              value={status}
              onChange={(e) => {
                setStatus(e.target.value);
                setOffset(0);
              }}
            >
              <option value="">All</option>
              <option value="success">Success</option>
              <option value="error">Error</option>
            </select>
          </div>
          <div>
            <label htmlFor="trace-type">Run type</label>
            <select
              id="trace-type"
              value={runType}
              onChange={(e) => {
                setRunType(e.target.value);
                setOffset(0);
              }}
            >
              <option value="">All</option>
              {RUN_TYPES.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label htmlFor="trace-search">Search</label>
            <input
              id="trace-search"
              type="text"
              value={searchText}
              placeholder="name or model"
              onChange={(e) => setSearchText(e.target.value)}
            />
          </div>
          <button type="submit" className="secondary">
            Search
          </button>
          <label className="check-inline" style={{ paddingBottom: "0.55rem" }}>
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
            />
            Auto-refresh (10s)
          </label>
        </form>

        {state.status === "loading" ? <p className="muted">Loading runs...</p> : null}
        {state.status === "error" ? <PlatformDown error={state.error} /> : null}
        {data ? (
          data.runs.length === 0 ? (
            <EmptyState
              title="No runs match."
              hint="Runs appear once the SDK posts traces to the platform server."
            />
          ) : (
            <>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th />
                      <th>Name</th>
                      <th>Type</th>
                      <th>Model</th>
                      <th className="num">Latency</th>
                      <th className="num">Tokens</th>
                      <th className="num">Cost</th>
                      <th>Feedback</th>
                      <th>Time</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.runs.map((run) => {
                      const tokens = tokenTotal(run.tokens);
                      return (
                        <tr key={run.id} className={run.status === "error" ? "row-blocked" : ""}>
                          <td>
                            <span
                              className={`status-dot ${run.status}`}
                              role="img"
                              aria-label={run.status}
                            />
                          </td>
                          <td>
                            <Link to={`/traces/${encodeURIComponent(run.id)}`}>{run.name}</Link>
                          </td>
                          <td>
                            <span className="pill">{run.run_type}</span>
                          </td>
                          <td>{run.model ?? "-"}</td>
                          <td className="num">{fmtLatency(run.latency_ms)}</td>
                          <td className="num">{tokens === null ? "-" : fmtNum(tokens)}</td>
                          <td className="num">{run.cost === null ? "-" : fmtUSD(run.cost)}</td>
                          <td>
                            {run.feedback_score === null ? (
                              <span className="muted">-</span>
                            ) : (
                              <Thumb dir={run.feedback_score >= 0 ? "up" : "down"} active />
                            )}
                          </td>
                          <td>{fmtTimestamp(run.start_time)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              <div className="pagination">
                <button
                  className="secondary"
                  disabled={offset === 0}
                  onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
                >
                  Previous
                </button>
                <span className="muted">
                  {offset + 1}-{Math.min(offset + PAGE_SIZE, data.total)} of {fmtNum(data.total)}
                </span>
                <button
                  className="secondary"
                  disabled={offset + PAGE_SIZE >= data.total}
                  onClick={() => setOffset(offset + PAGE_SIZE)}
                >
                  Next
                </button>
              </div>
            </>
          )
        ) : null}
      </div>
    </>
  );
}
