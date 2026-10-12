import { useState } from "react";
import { getAudit } from "../api/client";
import { useApi } from "../api/useApi";
import BackendDown from "../components/BackendDown";
import EmptyState from "../components/EmptyState";
import { fmtNum, fmtTimestamp } from "../components/format";

const PAGE_SIZE = 50;

export default function AuditPage() {
  const [type, setType] = useState("");
  const [blocked, setBlocked] = useState<"all" | "yes" | "no">("all");
  const [offset, setOffset] = useState(0);

  const { state } = useApi(
    () =>
      getAudit({
        limit: PAGE_SIZE,
        offset,
        type: type || undefined,
        blocked: blocked === "all" ? undefined : blocked === "yes",
      }),
    [type, blocked, offset],
  );

  const data = state.status === "ready" ? state.data : null;

  return (
    <>
      <h1 className="page-header">Audit Trail</h1>
      <p className="page-sub">PII events recorded by the compliance guard (newest first).</p>

      <div className="card">
        <div className="form-row" style={{ marginBottom: "0.9rem" }}>
          <div>
            <label htmlFor="audit-type">PII type</label>
            <select
              id="audit-type"
              value={type}
              onChange={(e) => {
                setType(e.target.value);
                setOffset(0);
              }}
            >
              <option value="">All types</option>
              {(data?.pii_types ?? []).map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label htmlFor="audit-blocked">Blocked</label>
            <select
              id="audit-blocked"
              value={blocked}
              onChange={(e) => {
                setBlocked(e.target.value as "all" | "yes" | "no");
                setOffset(0);
              }}
            >
              <option value="all">All events</option>
              <option value="yes">Blocked only</option>
              <option value="no">Not blocked</option>
            </select>
          </div>
        </div>

        {state.status === "loading" ? <p className="muted">Loading audit records...</p> : null}
        {state.status === "error" ? <BackendDown backend="dashboard" error={state.error} /> : null}
        {data ? (
          data.records.length === 0 ? (
            <EmptyState
              title="No audit records match."
              hint="Events appear here once a guarded model writes to audit.jsonl."
            />
          ) : (
            <>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Timestamp</th>
                      <th>Direction</th>
                      <th>Method</th>
                      <th>PII types</th>
                      <th className="num">Count</th>
                      <th>Strategy</th>
                      <th>Blocked</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.records.map((r, i) => (
                      <tr key={`${r.timestamp}-${i}`} className={r.blocked ? "row-blocked" : ""}>
                        <td>{fmtTimestamp(r.timestamp)}</td>
                        <td>{r.direction}</td>
                        <td>{r.method}</td>
                        <td>
                          {Object.entries(r.pii_types ?? {})
                            .map(([t, n]) => `${t} (${n})`)
                            .join(", ") || "-"}
                        </td>
                        <td className="num">{fmtNum(r.count)}</td>
                        <td>{r.strategy}</td>
                        <td>
                          {r.blocked ? <span className="pill danger">blocked</span> : <span className="muted">no</span>}
                        </td>
                      </tr>
                    ))}
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
