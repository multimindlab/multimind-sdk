import { Link } from "react-router-dom";
import { getProjects } from "../../api/platform";
import { useApi } from "../../api/useApi";
import EmptyState from "../../components/EmptyState";
import PlatformDown from "../../components/PlatformDown";
import { fmtNum, fmtTimestamp, fmtUSD } from "../../components/format";

export default function ProjectsPage() {
  const { state } = useApi(getProjects);

  if (state.status === "loading") return <p className="muted">Loading projects...</p>;
  if (state.status === "error") {
    return (
      <>
        <h1 className="page-header">Projects</h1>
        <PlatformDown error={state.error} />
      </>
    );
  }

  const projects = [...state.data.projects].sort((a, b) => b.run_count - a.run_count);

  return (
    <>
      <h1 className="page-header">Projects</h1>
      <p className="page-sub">Tracing projects on this platform server. Click one to see its runs.</p>

      {projects.length === 0 ? (
        <EmptyState
          title="No projects yet."
          hint="Projects appear once the SDK logs its first traced run."
        />
      ) : (
        <div className="card">
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Project</th>
                  <th className="num">Runs</th>
                  <th>Last activity</th>
                  <th className="num">Total cost</th>
                  <th>Error rate</th>
                </tr>
              </thead>
              <tbody>
                {projects.map((p) => {
                  const errPct = Math.min(p.error_rate * 100, 100);
                  return (
                    <tr key={p.name}>
                      <td>
                        <Link to={`/traces?project=${encodeURIComponent(p.name)}`}>{p.name}</Link>
                      </td>
                      <td className="num">{fmtNum(p.run_count)}</td>
                      <td>{p.last_run_at ? fmtTimestamp(p.last_run_at) : "-"}</td>
                      <td className="num">{fmtUSD(p.total_cost)}</td>
                      <td>
                        <div style={{ display: "flex", alignItems: "center", gap: "0.6rem" }}>
                          <div className="share-bar error-bar" aria-hidden="true">
                            <span style={{ width: `${errPct}%` }} />
                          </div>
                          <span className="muted">{errPct.toFixed(1)}%</span>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </>
  );
}
