import { useState, type FormEvent } from "react";
import { RedactionStrategy, ScanTextResponse, scanText } from "../api/client";
import { toApiError } from "../api/useApi";
import BackendDown from "../components/BackendDown";
import { ApiError } from "../api/client";
import { fmtNum } from "../components/format";

const STRATEGIES: RedactionStrategy[] = ["mask", "hash", "remove"];

export default function ScanPage() {
  const [text, setText] = useState("");
  const [strategy, setStrategy] = useState<RedactionStrategy>("mask");
  const [pending, setPending] = useState(false);
  const [result, setResult] = useState<ScanTextResponse | null>(null);
  const [error, setError] = useState<ApiError | null>(null);

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setPending(true);
    setError(null);
    try {
      setResult(await scanText({ text, strategy }));
    } catch (err) {
      setResult(null);
      setError(toApiError(err));
    } finally {
      setPending(false);
    }
  };

  return (
    <>
      <h1 className="page-header">PII Scan</h1>
      <p className="page-sub">Scan any text for PII and preview the redacted result.</p>

      <form className="card" onSubmit={onSubmit} aria-label="Scan text">
        <div className="field">
          <label htmlFor="scan-text">Text to scan</label>
          <textarea
            id="scan-text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Paste a prompt, log line, or document excerpt..."
          />
        </div>
        <div className="form-row">
          <div>
            <label htmlFor="scan-strategy">Redaction strategy</label>
            <select
              id="scan-strategy"
              value={strategy}
              onChange={(e) => setStrategy(e.target.value as RedactionStrategy)}
            >
              {STRATEGIES.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </div>
          <button type="submit" disabled={pending || text.trim() === ""}>
            {pending ? "Scanning..." : "Scan text"}
          </button>
        </div>
      </form>

      {error ? <BackendDown backend="dashboard" error={error} /> : null}

      {result ? (
        <>
          <div className="card">
            <h2>Findings ({fmtNum(result.total)})</h2>
            {result.total === 0 ? (
              <p className="muted">No PII found.</p>
            ) : (
              <>
                <p className="muted">
                  {Object.entries(result.counts)
                    .map(([t, n]) => `${t}: ${n}`)
                    .join(", ")}
                </p>
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Type</th>
                        <th className="num">Start</th>
                        <th className="num">End</th>
                        <th>Matched text</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.findings.map((f, i) => (
                        <tr key={i}>
                          <td>{f.type}</td>
                          <td className="num">{f.start}</td>
                          <td className="num">{f.end}</td>
                          <td>
                            <code>{f.text}</code>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            )}
          </div>
          <div className="card">
            <h2>Redacted output</h2>
            {result.redacted != null ? (
              <pre className="output-panel">{result.redacted}</pre>
            ) : (
              <p className="muted">No redacted output returned.</p>
            )}
          </div>
        </>
      ) : null}
    </>
  );
}
