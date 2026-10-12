import { getCosts, getSummary } from "../api/client";
import { useApi } from "../api/useApi";
import BackendDown from "../components/BackendDown";
import BarChart from "../components/BarChart";
import LineChart from "../components/LineChart";
import { fmtNum, fmtUSD } from "../components/format";

export default function OverviewPage() {
  const { state } = useApi(() => Promise.all([getSummary(), getCosts()]));

  if (state.status === "loading") return <p className="muted">Loading overview...</p>;
  if (state.status === "error") {
    return (
      <>
        <h1 className="page-header">Overview</h1>
        <BackendDown backend="dashboard" error={state.error} />
      </>
    );
  }

  const [summary, costs] = state.data;
  const audit = summary.audit.all_time;
  const last24 = summary.audit.last_24h;
  const spendPoints = costs.daily.slice(-30).map((d) => ({ label: d.date.slice(5), value: d.cost }));
  const piiBars = Object.entries(audit.by_type)
    .sort((a, b) => b[1] - a[1])
    .map(([label, value]) => ({ label, value }));

  return (
    <>
      <h1 className="page-header">Overview</h1>
      <p className="page-sub">Governance posture across audit, spend, and guardrails.</p>

      {summary.budget?.exceeded ? (
        <div className="banner danger" role="alert">
          Budget exceeded: {fmtUSD(summary.budget.spent)} spent of a {fmtUSD(summary.budget.max_cost)}{" "}
          ceiling ({summary.budget.basis}).
        </div>
      ) : null}

      <div className="tiles">
        <div className="tile">
          <div className="tile-label">PII events</div>
          <div className="tile-value">{fmtNum(audit.events)}</div>
          <div className="tile-hint">{fmtNum(last24.events)} in last 24h</div>
        </div>
        <div className={audit.blocked > 0 ? "tile danger" : "tile"}>
          <div className="tile-label">Blocked requests</div>
          <div className="tile-value">{fmtNum(audit.blocked)}</div>
          <div className="tile-hint">{fmtNum(last24.blocked)} in last 24h</div>
        </div>
        <div className="tile">
          <div className="tile-label">Total spend</div>
          <div className="tile-value">{fmtUSD(summary.costs.total_cost)}</div>
          <div className="tile-hint">{fmtUSD(summary.costs.last_24h_cost)} in last 24h</div>
        </div>
        <div className="tile">
          <div className="tile-label">Models in use</div>
          <div className="tile-value">{fmtNum(summary.costs.models.length)}</div>
          <div className="tile-hint">
            {summary.costs.models.length > 0 ? summary.costs.models.join(", ") : "no calls logged"}
          </div>
        </div>
      </div>

      <div className="grid-2">
        <div className="card">
          <h2>Spend, last 30 days</h2>
          {costs.available && spendPoints.length > 0 ? (
            <LineChart points={spendPoints} formatValue={fmtUSD} ariaLabel="Daily spend, last 30 days" />
          ) : (
            <p className="muted">No cost records yet. Spend appears once calls are tracked to costs.jsonl.</p>
          )}
        </div>
        <div className="card">
          <h2>PII detections by type</h2>
          {summary.audit.available && piiBars.length > 0 ? (
            <BarChart bars={piiBars} formatValue={fmtNum} ariaLabel="PII detections by type" />
          ) : (
            <p className="muted">No audit records yet. Detections appear once the guard writes audit.jsonl.</p>
          )}
        </div>
      </div>
    </>
  );
}
