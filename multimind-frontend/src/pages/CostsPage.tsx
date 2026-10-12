import { getCosts } from "../api/client";
import { useApi } from "../api/useApi";
import BackendDown from "../components/BackendDown";
import EmptyState from "../components/EmptyState";
import LineChart from "../components/LineChart";
import { fmtNum, fmtUSD } from "../components/format";

export default function CostsPage() {
  const { state } = useApi(getCosts);

  if (state.status === "loading") return <p className="muted">Loading costs...</p>;
  if (state.status === "error") {
    return (
      <>
        <h1 className="page-header">Costs</h1>
        <BackendDown backend="dashboard" error={state.error} />
      </>
    );
  }

  const data = state.data;
  if (!data.available) {
    return (
      <>
        <h1 className="page-header">Costs</h1>
        <EmptyState
          title="No cost data yet."
          hint="Spend appears once tracked calls are logged to costs.jsonl."
        />
      </>
    );
  }

  const tags = Object.entries(data.chargeback.by_tag).sort((a, b) => b[1].cost - a[1].cost);
  const models = Object.entries(data.by_model).sort((a, b) => b[1].cost - a[1].cost);
  const spendPoints = data.daily.map((d) => ({ label: d.date.slice(5), value: d.cost }));

  return (
    <>
      <h1 className="page-header">Costs</h1>
      <p className="page-sub">
        {fmtNum(data.chargeback.calls)} calls, {fmtUSD(data.chargeback.total_cost)} total
        {data.chargeback.period ? ` in ${data.chargeback.period}` : ""}.
      </p>

      <div className="card">
        <h2>Daily spend</h2>
        <LineChart points={spendPoints} formatValue={fmtUSD} ariaLabel="Daily spend" />
      </div>

      <div className="card">
        <h2>Chargeback by tag</h2>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Tag</th>
                <th className="num">Calls</th>
                <th className="num">Tokens</th>
                <th className="num">Cost</th>
                <th>Share</th>
              </tr>
            </thead>
            <tbody>
              {tags.map(([tag, g]) => (
                <tr key={tag}>
                  <td>{tag}</td>
                  <td className="num">{fmtNum(g.calls)}</td>
                  <td className="num">{fmtNum(g.total_tokens)}</td>
                  <td className="num">{fmtUSD(g.cost)}</td>
                  <td>
                    <div style={{ display: "flex", alignItems: "center", gap: "0.6rem" }}>
                      <div className="share-bar" aria-hidden="true">
                        <span style={{ width: `${Math.min(g.share_pct, 100)}%` }} />
                      </div>
                      <span className="muted">{g.share_pct.toFixed(1)}%</span>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card">
        <h2>Per model</h2>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th className="num">Calls</th>
                <th className="num">Input tokens</th>
                <th className="num">Output tokens</th>
                <th className="num">Cost</th>
              </tr>
            </thead>
            <tbody>
              {models.map(([model, g]) => (
                <tr key={model}>
                  <td>{model}</td>
                  <td className="num">{fmtNum(g.calls)}</td>
                  <td className="num">{fmtNum(g.input_tokens)}</td>
                  <td className="num">{fmtNum(g.output_tokens)}</td>
                  <td className="num">{fmtUSD(g.cost)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}
