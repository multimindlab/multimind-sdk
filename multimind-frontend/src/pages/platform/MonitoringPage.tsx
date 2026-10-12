import { useState } from "react";
import { getProjects, getStats } from "../../api/platform";
import { useApi } from "../../api/useApi";
import EmptyState from "../../components/EmptyState";
import LineChart from "../../components/LineChart";
import MultiLineChart from "../../components/MultiLineChart";
import PlatformDown from "../../components/PlatformDown";
import StackedBarChart from "../../components/StackedBarChart";
import { fmtNum, fmtUSD } from "../../components/format";

const DAYS = 30;

export default function MonitoringPage() {
  const [project, setProject] = useState("");

  const projectsState = useApi(getProjects).state;
  const projects = projectsState.status === "ready" ? projectsState.data.projects : [];

  const { state } = useApi(() => getStats(project || undefined, DAYS), [project]);

  return (
    <>
      <h1 className="page-header">Monitoring</h1>
      <p className="page-sub">Last {DAYS} days of platform activity.</p>

      <div className="card">
        <div className="form-row">
          <div>
            <label htmlFor="mon-project">Project</label>
            <select id="mon-project" value={project} onChange={(e) => setProject(e.target.value)}>
              <option value="">All projects</option>
              {projects.map((p) => (
                <option key={p.name} value={p.name}>
                  {p.name}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {state.status === "loading" ? <p className="muted">Loading stats...</p> : null}
      {state.status === "error" ? <PlatformDown error={state.error} /> : null}
      {state.status === "ready" ? (
        state.data.daily.length === 0 ? (
          <EmptyState title="No stats yet." hint="Charts appear once runs are recorded." />
        ) : (
          (() => {
            const daily = state.data.daily;
            const labels = daily.map((d) => d.date.slice(5));
            return (
              <div className="grid-2">
                <div className="card">
                  <h2>Runs per day</h2>
                  <StackedBarChart
                    ariaLabel="Runs per day"
                    points={daily.map((d) => ({
                      label: d.date.slice(5),
                      ok: Math.max(d.runs - d.errors, 0),
                      error: d.errors,
                    }))}
                  />
                </div>
                <div className="card">
                  <h2>Cost per day</h2>
                  <LineChart
                    ariaLabel="Cost per day"
                    formatValue={fmtUSD}
                    points={daily.map((d) => ({ label: d.date.slice(5), value: d.cost }))}
                  />
                </div>
                <div className="card">
                  <h2>Latency (p50 / p95)</h2>
                  <MultiLineChart
                    ariaLabel="Latency p50 and p95"
                    labels={labels}
                    formatValue={(v) => `${Math.round(v)}ms`}
                    series={[
                      { name: "p50", color: "var(--accent)", values: daily.map((d) => d.p50_latency_ms) },
                      { name: "p95", color: "var(--warn)", values: daily.map((d) => d.p95_latency_ms) },
                    ]}
                  />
                </div>
                <div className="card">
                  <h2>Token volume</h2>
                  <LineChart
                    ariaLabel="Token volume"
                    formatValue={fmtNum}
                    points={daily.map((d) => ({ label: d.date.slice(5), value: d.tokens }))}
                  />
                </div>
              </div>
            );
          })()
        )
      ) : null}
    </>
  );
}
