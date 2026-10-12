import { FormEvent, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { getRun, postFeedback, PlatformSpan, tokenTotal } from "../../api/platform";
import { toApiError, useApi } from "../../api/useApi";
import PlatformDown from "../../components/PlatformDown";
import Thumb from "../../components/Thumb";
import Waterfall from "../../components/Waterfall";
import { fmtNum, fmtTimestamp, fmtUSD } from "../../components/format";
import { fmtLatency } from "./TracesPage";

function Field({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="span-field">
      <span className="muted">{label}</span>
      <span>{value}</span>
    </div>
  );
}

function SpanPanel({ span, onClose }: { span: PlatformSpan; onClose: () => void }) {
  const tokens = tokenTotal(span.tokens);
  return (
    <div className="card span-panel" aria-label="Span details">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
        <h2>{span.name}</h2>
        <button className="secondary" onClick={onClose}>
          Close
        </button>
      </div>
      <Field label="Type" value={span.run_type} />
      <Field label="Status" value={span.status} />
      {span.model ? <Field label="Model" value={span.model} /> : null}
      {span.provider ? <Field label="Provider" value={span.provider} /> : null}
      <Field label="Start" value={fmtTimestamp(span.start_time)} />
      {span.end_time ? <Field label="End" value={fmtTimestamp(span.end_time)} /> : null}
      {span.latency_ms !== undefined && span.latency_ms !== null ? (
        <Field label="Latency" value={fmtLatency(span.latency_ms)} />
      ) : null}
      {tokens !== null ? <Field label="Tokens" value={fmtNum(tokens)} /> : null}
      {span.cost !== undefined && span.cost !== null ? (
        <Field label="Cost" value={fmtUSD(span.cost)} />
      ) : null}
      {span.input_chars !== undefined && span.input_chars !== null ? (
        <Field label="Input chars" value={fmtNum(span.input_chars)} />
      ) : null}
      {span.output_chars !== undefined && span.output_chars !== null ? (
        <Field label="Output chars" value={fmtNum(span.output_chars)} />
      ) : null}
      {span.input_sha256 ? <Field label="Input sha256" value={span.input_sha256} /> : null}
      {span.output_sha256 ? <Field label="Output sha256" value={span.output_sha256} /> : null}
      {span.error ? (
        <div className="banner danger" role="alert">
          {span.error}
        </div>
      ) : null}
      {span.inputs !== undefined && span.inputs !== null ? (
        <>
          <h3 className="span-io-title">Inputs</h3>
          <pre className="output-panel">{JSON.stringify(span.inputs, null, 2)}</pre>
        </>
      ) : null}
      {span.outputs !== undefined && span.outputs !== null ? (
        <>
          <h3 className="span-io-title">Outputs</h3>
          <pre className="output-panel">{JSON.stringify(span.outputs, null, 2)}</pre>
        </>
      ) : null}
    </div>
  );
}

function FeedbackWidget({ runId, onSubmitted }: { runId: string; onSubmitted: () => void }) {
  const [score, setScore] = useState<-1 | 1 | null>(null);
  const [comment, setComment] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState(false);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (score === null || busy) return;
    setBusy(true);
    setError(null);
    try {
      await postFeedback(runId, comment.trim() ? { score, comment: comment.trim() } : { score });
      setDone(true);
      setScore(null);
      setComment("");
      onSubmitted();
    } catch (err) {
      setError(toApiError(err).detail);
    } finally {
      setBusy(false);
    }
  };

  return (
    <form onSubmit={submit}>
      <div className="form-row" style={{ alignItems: "center" }}>
        <button
          type="button"
          className={`secondary thumb-btn${score === 1 ? " selected" : ""}`}
          aria-pressed={score === 1}
          aria-label="rate up"
          onClick={() => setScore(score === 1 ? null : 1)}
        >
          <Thumb dir="up" active={score === 1} />
        </button>
        <button
          type="button"
          className={`secondary thumb-btn${score === -1 ? " selected" : ""}`}
          aria-pressed={score === -1}
          aria-label="rate down"
          onClick={() => setScore(score === -1 ? null : -1)}
        >
          <Thumb dir="down" active={score === -1} />
        </button>
        <input
          type="text"
          value={comment}
          placeholder="Optional comment"
          aria-label="Feedback comment"
          onChange={(e) => setComment(e.target.value)}
          style={{ flex: 1 }}
        />
        <button type="submit" disabled={score === null || busy}>
          {busy ? "Sending..." : "Submit feedback"}
        </button>
      </div>
      {done ? <p className="muted">Feedback recorded.</p> : null}
      {error ? (
        <div className="banner danger" role="alert">
          {error}
        </div>
      ) : null}
    </form>
  );
}

export default function TraceDetailPage() {
  const { id = "" } = useParams();
  const { state, reload } = useApi(() => getRun(id), [id]);
  const [selected, setSelected] = useState<PlatformSpan | null>(null);

  if (state.status === "loading") return <p className="muted">Loading trace...</p>;
  if (state.status === "error") {
    return (
      <>
        <h1 className="page-header">Trace</h1>
        <PlatformDown error={state.error} />
      </>
    );
  }

  const { run, spans, feedback } = state.data;
  const tokens = tokenTotal(run.tokens);

  return (
    <>
      <p style={{ margin: "0 0 0.5rem" }}>
        <Link to="/traces">&larr; All traces</Link>
      </p>
      <h1 className="page-header">
        {run.name} <span className={`pill ${run.status === "error" ? "danger" : "ok"}`}>{run.status}</span>
      </h1>
      <p className="page-sub">
        {run.project} &middot; {run.run_type} &middot; started {fmtTimestamp(run.start_time)}
      </p>

      <div className="tiles">
        <div className="tile">
          <div className="tile-label">Latency</div>
          <div className="tile-value">{fmtLatency(run.latency_ms)}</div>
        </div>
        <div className="tile">
          <div className="tile-label">Cost</div>
          <div className="tile-value">{run.cost === null ? "-" : fmtUSD(run.cost)}</div>
        </div>
        <div className="tile">
          <div className="tile-label">Tokens</div>
          <div className="tile-value">{tokens === null ? "-" : fmtNum(tokens)}</div>
        </div>
        <div className="tile">
          <div className="tile-label">Model</div>
          <div className="tile-value" style={{ fontSize: 16 }}>
            {run.model ?? "-"}
          </div>
          <div className="tile-hint">{run.provider ?? ""}</div>
        </div>
      </div>

      {run.tags.length > 0 ? (
        <p style={{ marginTop: 0 }}>
          {run.tags.map((t) => (
            <span key={t} className="pill" style={{ marginRight: "0.4rem" }}>
              {t}
            </span>
          ))}
        </p>
      ) : null}

      {run.error ? (
        <div className="banner danger" role="alert">
          {run.error}
        </div>
      ) : null}

      <div className="trace-layout">
        <div className="card" style={{ flex: 2, minWidth: 0 }}>
          <h2>Waterfall</h2>
          <Waterfall
            spans={spans}
            selectedId={selected?.id ?? null}
            onSelect={(span) => setSelected(span)}
          />
        </div>
        {selected ? <SpanPanel span={selected} onClose={() => setSelected(null)} /> : null}
      </div>

      <div className="card">
        <h2>Feedback</h2>
        <FeedbackWidget runId={run.id} onSubmitted={reload} />
        {feedback.length > 0 ? (
          <ul className="feedback-list">
            {feedback.map((f, i) => (
              <li key={`${f.created_at}-${i}`}>
                <Thumb dir={f.score >= 0 ? "up" : "down"} active />
                <span>{f.comment ?? <span className="muted">no comment</span>}</span>
                <span className="muted">{fmtTimestamp(f.created_at)}</span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="muted" style={{ marginBottom: 0 }}>
            No feedback yet.
          </p>
        )}
      </div>
    </>
  );
}
