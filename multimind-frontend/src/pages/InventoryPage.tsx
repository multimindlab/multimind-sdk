import { useCallback, useEffect, useState } from "react";
import { InventoryFinding, InventoryResponse, getInventory } from "../api/client";
import { LoadState, toApiError } from "../api/useApi";
import BackendDown from "../components/BackendDown";
import EmptyState from "../components/EmptyState";
import { fmtNum, fmtTimestamp } from "../components/format";

function groupByProvider(findings: InventoryFinding[]): Array<[string, InventoryFinding[]]> {
  const groups = new Map<string, InventoryFinding[]>();
  for (const f of findings) {
    const list = groups.get(f.provider) ?? [];
    list.push(f);
    groups.set(f.provider, list);
  }
  return [...groups.entries()].sort((a, b) => b[1].length - a[1].length);
}

export default function InventoryPage() {
  const [state, setState] = useState<LoadState<InventoryResponse>>({ status: "loading" });
  const [rescanning, setRescanning] = useState(false);

  const load = useCallback(async (refresh: boolean) => {
    if (refresh) setRescanning(true);
    else setState({ status: "loading" });
    try {
      const data = await getInventory(refresh);
      setState({ status: "ready", data });
    } catch (err) {
      setState({ status: "error", error: toApiError(err) });
    } finally {
      setRescanning(false);
    }
  }, []);

  useEffect(() => {
    load(false);
  }, [load]);

  if (state.status === "loading") return <p className="muted">Scanning project...</p>;
  if (state.status === "error") {
    return (
      <>
        <h1 className="page-header">Shadow-AI Inventory</h1>
        <BackendDown backend="dashboard" error={state.error} />
      </>
    );
  }

  const data = state.data;
  const groups = groupByProvider(data.findings);
  const hardcoded = data.risks.hardcoded_keys;

  return (
    <>
      <h1 className="page-header">Shadow-AI Inventory</h1>
      <p className="page-sub">
        AI usage discovered in {data.root} ({fmtNum(data.summary.scanned_files)} files scanned,
        last scan {fmtTimestamp(data.scanned_at)}).
      </p>

      <div className="form-row" style={{ marginBottom: "1rem" }}>
        <button onClick={() => load(true)} disabled={rescanning}>
          {rescanning ? "Rescanning..." : "Rescan project"}
        </button>
        {data.note ? <span className="muted">{data.note}</span> : null}
      </div>

      {hardcoded.length > 0 ? (
        <div className="banner danger" role="alert">
          <strong>
            {fmtNum(hardcoded.length)} hardcoded API key{hardcoded.length === 1 ? "" : "s"} found.
          </strong>{" "}
          Move them to environment variables:
          <ul style={{ margin: "0.35rem 0 0", paddingLeft: "1.2rem" }}>
            {hardcoded.map((f, i) => (
              <li key={i}>
                <code>
                  {f.file ?? "unknown file"}
                  {f.line != null ? `:${f.line}` : ""}
                </code>{" "}
                ({f.provider})
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      {data.risks.external_data_flow_providers.length > 0 ? (
        <div className="banner warn">
          Data flows to external providers: {data.risks.external_data_flow_providers.join(", ")}
        </div>
      ) : null}

      {groups.length === 0 ? (
        <EmptyState title="No AI usage found in this project." />
      ) : (
        groups.map(([provider, findings]) => (
          <div className="card" key={provider}>
            <h2>
              {provider} <span className="pill">{fmtNum(findings.length)}</span>
            </h2>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Kind</th>
                    <th>Evidence</th>
                    <th>Location</th>
                    <th>Data flow</th>
                  </tr>
                </thead>
                <tbody>
                  {findings.map((f, i) => (
                    <tr key={i} className={f.kind === "hardcoded_key" ? "row-blocked" : ""}>
                      <td>{f.kind}</td>
                      <td>
                        <code>{f.evidence}</code>
                      </td>
                      <td>
                        {f.file ? (
                          <code>
                            {f.file}
                            {f.line != null ? `:${f.line}` : ""}
                          </code>
                        ) : (
                          <span className="muted">-</span>
                        )}
                      </td>
                      <td>{f.data_flow ?? <span className="muted">unknown</span>}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ))
      )}
    </>
  );
}
