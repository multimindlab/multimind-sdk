import { useEffect, useState, type FormEvent } from "react";
import { GuardrailsConfig, RedactionStrategy, getGuardrails, putGuardrails } from "../api/client";
import { toApiError, useApi } from "../api/useApi";
import BackendDown from "../components/BackendDown";

const STRATEGIES: RedactionStrategy[] = ["mask", "hash", "remove"];

interface Toast {
  kind: "ok" | "danger";
  text: string;
  hint?: string;
}

export default function GuardrailsPage() {
  const { state } = useApi(getGuardrails);

  const [strategy, setStrategy] = useState<RedactionStrategy>("mask");
  const [blockOn, setBlockOn] = useState<string[]>([]);
  const [budget, setBudget] = useState("");
  const [scanOutput, setScanOutput] = useState(true);
  const [saving, setSaving] = useState(false);
  const [toast, setToast] = useState<Toast | null>(null);

  useEffect(() => {
    if (state.status === "ready") {
      const cfg = state.data.config;
      setStrategy(cfg.strategy);
      setBlockOn(cfg.block_on);
      setBudget(cfg.budget_max_cost != null ? String(cfg.budget_max_cost) : "");
      setScanOutput(cfg.scan_output);
    }
  }, [state]);

  useEffect(() => {
    if (!toast) return;
    const timer = setTimeout(() => setToast(null), 6000);
    return () => clearTimeout(timer);
  }, [toast]);

  if (state.status === "loading") return <p className="muted">Loading guardrails...</p>;
  if (state.status === "error") {
    return (
      <>
        <h1 className="page-header">Guardrails</h1>
        <BackendDown backend="dashboard" error={state.error} />
      </>
    );
  }

  const { pii_types, path, exists } = state.data;

  const toggleBlockOn = (type: string) => {
    setBlockOn((current) =>
      current.includes(type) ? current.filter((t) => t !== type) : [...current, type],
    );
  };

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    const budgetValue = budget.trim() === "" ? null : Number(budget);
    if (budgetValue !== null && (!Number.isFinite(budgetValue) || budgetValue <= 0)) {
      setToast({ kind: "danger", text: "Budget must be a positive number (or empty for no limit)." });
      return;
    }
    const config: GuardrailsConfig = {
      strategy,
      block_on: blockOn,
      budget_max_cost: budgetValue,
      scan_output: scanOutput,
    };
    setSaving(true);
    try {
      const res = await putGuardrails(config);
      setToast({
        kind: "ok",
        text: `Guardrails saved to ${res.path}.`,
        hint: `Restart the proxy to apply: ${res.apply_hint}`,
      });
    } catch (err) {
      const apiErr = toApiError(err);
      setToast({ kind: "danger", text: `Save failed: ${apiErr.detail}` });
    } finally {
      setSaving(false);
    }
  };

  return (
    <>
      <h1 className="page-header">Guardrails</h1>
      <p className="page-sub">
        No-code guardrails config at <code>{path}</code>
        {exists ? "" : " (not created yet; saving will create it)"}. The guard proxy loads it via{" "}
        <code>multimind serve --config {path}</code>.
      </p>

      <form className="card" onSubmit={onSubmit} aria-label="Guardrails configuration">
        <div className="field">
          <label htmlFor="gr-strategy">Redaction strategy</label>
          <select
            id="gr-strategy"
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

        <div className="field">
          <label>Block requests containing</label>
          <div className="checkbox-grid">
            {pii_types.map((t) => (
              <label key={t}>
                <input
                  type="checkbox"
                  checked={blockOn.includes(t)}
                  onChange={() => toggleBlockOn(t)}
                />
                {t}
              </label>
            ))}
          </div>
        </div>

        <div className="field">
          <label htmlFor="gr-budget">Budget ceiling (USD, empty for no limit)</label>
          <input
            id="gr-budget"
            type="number"
            min="0"
            step="0.01"
            value={budget}
            onChange={(e) => setBudget(e.target.value)}
            placeholder="e.g. 25.00"
          />
        </div>

        <div className="field">
          <label className="check-inline" htmlFor="gr-scan-output">
            <input
              id="gr-scan-output"
              type="checkbox"
              checked={scanOutput}
              onChange={(e) => setScanOutput(e.target.checked)}
            />
            Also scan model output for PII
          </label>
        </div>

        <button type="submit" disabled={saving}>
          {saving ? "Saving..." : "Save guardrails"}
        </button>
      </form>

      {toast ? (
        <div className={`toast ${toast.kind}`} role={toast.kind === "danger" ? "alert" : "status"}>
          <div>{toast.text}</div>
          {toast.hint ? <div className="toast-hint">{toast.hint}</div> : null}
        </div>
      ) : null}
    </>
  );
}
