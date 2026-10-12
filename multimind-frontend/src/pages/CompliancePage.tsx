import { useState, type FormEvent } from "react";
import {
  DashboardMetrics,
  configureAlerts,
  getAlerts,
  getDashboardMetrics,
  getHealthcareUseCases,
  getRegulations,
} from "../api/compliance";
import { toApiError, useApi } from "../api/useApi";
import BackendDown from "../components/BackendDown";

const TIME_RANGES = ["24h", "7d", "30d"];

export default function CompliancePage() {
  const { state: regulationsState } = useApi(getRegulations);
  const { state: useCasesState } = useApi(getHealthcareUseCases);

  const [orgId, setOrgId] = useState("demo-org");
  const [timeRange, setTimeRange] = useState("7d");
  const [metrics, setMetrics] = useState<DashboardMetrics | null>(null);
  const [alerts, setAlerts] = useState<Record<string, unknown>[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [rulesJson, setRulesJson] = useState('{\n  "severity_threshold": "medium"\n}');
  const [configureBusy, setConfigureBusy] = useState(false);
  const [configureMessage, setConfigureMessage] = useState<string | null>(null);

  if (regulationsState.status === "loading" || useCasesState.status === "loading") {
    return <p className="muted">Loading compliance catalog...</p>;
  }
  if (regulationsState.status === "error") {
    return (
      <>
        <h1 className="page-header">Compliance Center</h1>
        <BackendDown backend="gateway" error={regulationsState.error} />
      </>
    );
  }

  const regulations = regulationsState.status === "ready" ? regulationsState.data : [];
  const useCases = useCasesState.status === "ready" ? useCasesState.data : [];

  const loadOrg = async (e: FormEvent) => {
    e.preventDefault();
    if (!orgId.trim() || busy) return;
    setBusy(true);
    setError(null);
    try {
      const [m, a] = await Promise.all([
        getDashboardMetrics(orgId.trim(), timeRange),
        getAlerts(orgId.trim()),
      ]);
      setMetrics(m);
      setAlerts(a);
    } catch (err) {
      setError(toApiError(err).detail);
    } finally {
      setBusy(false);
    }
  };

  const onConfigureAlerts = async (e: FormEvent) => {
    e.preventDefault();
    if (!orgId.trim() || configureBusy) return;
    let rules: Record<string, unknown>;
    try {
      rules = JSON.parse(rulesJson);
    } catch {
      setConfigureMessage("Alert rules must be valid JSON.");
      return;
    }
    setConfigureBusy(true);
    setConfigureMessage(null);
    try {
      const res = await configureAlerts(orgId.trim(), rules);
      setConfigureMessage(res.message);
    } catch (err) {
      setConfigureMessage(toApiError(err).detail);
    } finally {
      setConfigureBusy(false);
    }
  };

  return (
    <>
      <h1 className="page-header">Compliance Center</h1>
      <p className="page-sub">Regulation catalog, per-organization dashboard, and alert rules.</p>

      <div className="banner warn" role="status">
        The dashboard and alerts endpoints below create their compliance state fresh per
        request (no persistence between calls yet), so metrics will typically read zero until
        that's backed by storage. Deep compliance scans (monitor/report) aren't wired into this
        UI: they currently fail server-side by design rather than fabricate a result &mdash; see
        <code> ModelWatermarking</code> in <code>multimind/compliance/advanced.py</code>.
      </div>

      <div className="grid-2">
        <div className="card">
          <h2>Supported regulations</h2>
          <div className="checkbox-grid" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(120px, 1fr))" }}>
            {regulations.map((r) => (
              <span key={r} className="pill">
                {r}
              </span>
            ))}
          </div>
        </div>
        <div className="card">
          <h2>Healthcare use cases</h2>
          <div className="checkbox-grid" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))" }}>
            {useCases.map((u) => (
              <span key={u} className="pill">
                {u}
              </span>
            ))}
          </div>
        </div>
      </div>

      <div className="card">
        <h2>Organization dashboard</h2>
        <form className="form-row" onSubmit={loadOrg} style={{ marginBottom: "1rem" }}>
          <div className="field" style={{ marginBottom: 0 }}>
            <label htmlFor="org-id">Organization ID</label>
            <input
              id="org-id"
              type="text"
              value={orgId}
              onChange={(e) => setOrgId(e.target.value)}
            />
          </div>
          <div className="field" style={{ marginBottom: 0 }}>
            <label htmlFor="org-range">Time range</label>
            <select id="org-range" value={timeRange} onChange={(e) => setTimeRange(e.target.value)}>
              {TIME_RANGES.map((r) => (
                <option key={r} value={r}>
                  {r}
                </option>
              ))}
            </select>
          </div>
          <button type="submit" disabled={busy || !orgId.trim()}>
            {busy ? "Loading..." : "Load"}
          </button>
        </form>

        {error ? (
          <div className="banner danger" role="alert">
            {error}
          </div>
        ) : null}

        {metrics ? (
          <>
            <div className="tiles">
              <div className="tile">
                <div className="tile-label">Total checks</div>
                <div className="tile-value">{metrics.total_checks}</div>
              </div>
              <div className="tile">
                <div className="tile-label">Passed</div>
                <div className="tile-value">{metrics.passed_checks}</div>
              </div>
              <div className="tile danger">
                <div className="tile-label">Failed</div>
                <div className="tile-value">{metrics.failed_checks}</div>
              </div>
              <div className="tile">
                <div className="tile-label">Compliance score</div>
                <div className="tile-value">{(metrics.compliance_score * 100).toFixed(0)}%</div>
              </div>
            </div>

            <h2>Active alerts</h2>
            {!alerts || alerts.length === 0 ? (
              <p className="muted">No active alerts.</p>
            ) : (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Status</th>
                      <th>Severity</th>
                      <th>Detail</th>
                    </tr>
                  </thead>
                  <tbody>
                    {alerts.map((a, i) => (
                      <tr key={i}>
                        <td>{String(a.status ?? "-")}</td>
                        <td>{String(a.severity ?? "-")}</td>
                        <td>{JSON.stringify(a)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </>
        ) : (
          <p className="muted">Enter an organization ID and load to see its metrics.</p>
        )}
      </div>

      <div className="card">
        <h2>Configure alert rules</h2>
        <form onSubmit={onConfigureAlerts}>
          <div className="field">
            <label htmlFor="alert-rules">Alert rules (JSON)</label>
            <textarea
              id="alert-rules"
              value={rulesJson}
              onChange={(e) => setRulesJson(e.target.value)}
            />
          </div>
          <button type="submit" disabled={configureBusy || !orgId.trim()}>
            {configureBusy ? "Saving..." : "Save alert rules"}
          </button>
        </form>
        {configureMessage ? <p className="muted">{configureMessage}</p> : null}
      </div>
    </>
  );
}
