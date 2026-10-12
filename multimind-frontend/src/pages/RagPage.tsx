import { useRef, useState, type FormEvent } from "react";
import {
  GenerateResponse,
  QueryResponse,
  addDocuments,
  clearDocuments,
  generateAnswer,
  getDocumentCount,
  login,
  queryDocuments,
  switchModel,
  uploadFile,
} from "../api/rag";
import { clearRagToken, setRagToken, useRagToken } from "../api/ragAuth";
import { toApiError, useApi } from "../api/useApi";
import BackendDown from "../components/BackendDown";

function RagLogin() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!username.trim() || !password || busy) return;
    setBusy(true);
    setError(null);
    try {
      const res = await login(username.trim(), password);
      setRagToken(res.access_token);
    } catch (err) {
      const apiErr = toApiError(err);
      if (apiErr.status === 503) {
        setError("JWT auth is not configured on the RAG backend; requests should already work unauthenticated.");
      } else {
        setError(`Sign in failed (${apiErr.status}): ${apiErr.detail}`);
      }
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="card" style={{ maxWidth: 420 }}>
      <h2>Sign in</h2>
      <form onSubmit={submit}>
        <div className="field">
          <label htmlFor="rag-user">Username</label>
          <input id="rag-user" value={username} onChange={(e) => setUsername(e.target.value)} />
        </div>
        <div className="field">
          <label htmlFor="rag-pass">Password</label>
          <input
            id="rag-pass"
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
        </div>
        {error ? (
          <div className="banner danger" role="alert">
            {error}
          </div>
        ) : null}
        <button type="submit" disabled={busy || !username.trim() || !password}>
          {busy ? "Signing in..." : "Sign in"}
        </button>
      </form>
    </div>
  );
}

export default function RagPage() {
  const token = useRagToken();
  const { state: countState, reload: reloadCount } = useApi(getDocumentCount);

  const [text, setText] = useState("");
  const [addBusy, setAddBusy] = useState(false);
  const [addMessage, setAddMessage] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  const [query, setQuery] = useState("");
  const [queryBusy, setQueryBusy] = useState(false);
  const [queryError, setQueryError] = useState<string | null>(null);
  const [queryResult, setQueryResult] = useState<QueryResponse | null>(null);

  const [question, setQuestion] = useState("");
  const [genBusy, setGenBusy] = useState(false);
  const [genError, setGenError] = useState<string | null>(null);
  const [genResult, setGenResult] = useState<GenerateResponse | null>(null);

  const [modelType, setModelType] = useState<"openai" | "anthropic">("openai");
  const [modelName, setModelName] = useState("gpt-3.5-turbo");
  const [switchBusy, setSwitchBusy] = useState(false);
  const [switchMessage, setSwitchMessage] = useState<string | null>(null);

  const [showLogin, setShowLogin] = useState(false);

  const needsAuth = countState.status === "error" && countState.error.status === 401;

  if (countState.status === "loading") return <p className="muted">Connecting to the RAG API...</p>;
  if (countState.status === "error" && !needsAuth) {
    return (
      <>
        <h1 className="page-header">RAG Workbench</h1>
        <BackendDown backend="gateway" error={countState.error} />
        <p className="muted" style={{ marginTop: "0.5rem" }}>
          Start it as part of the backend aggregator: <code>multimind backend</code>
        </p>
      </>
    );
  }

  const addText = async (e: FormEvent) => {
    e.preventDefault();
    if (!text.trim() || addBusy) return;
    setAddBusy(true);
    setAddMessage(null);
    try {
      const res = await addDocuments([{ text: text.trim() }]);
      setAddMessage(`Added ${res.total} document(s).`);
      setText("");
      reloadCount();
    } catch (err) {
      setAddMessage(`Failed: ${toApiError(err).detail}`);
    } finally {
      setAddBusy(false);
    }
  };

  const addFile = async () => {
    const file = fileInput.current?.files?.[0];
    if (!file || addBusy) return;
    setAddBusy(true);
    setAddMessage(null);
    try {
      const res = await uploadFile(file);
      setAddMessage(`Added ${res.total} document(s) from ${file.name}.`);
      if (fileInput.current) fileInput.current.value = "";
      reloadCount();
    } catch (err) {
      setAddMessage(`Failed: ${toApiError(err).detail}`);
    } finally {
      setAddBusy(false);
    }
  };

  const runQuery = async (e: FormEvent) => {
    e.preventDefault();
    if (!query.trim() || queryBusy) return;
    setQueryBusy(true);
    setQueryError(null);
    setQueryResult(null);
    try {
      setQueryResult(await queryDocuments(query.trim()));
    } catch (err) {
      setQueryError(toApiError(err).detail);
    } finally {
      setQueryBusy(false);
    }
  };

  const runGenerate = async (e: FormEvent) => {
    e.preventDefault();
    if (!question.trim() || genBusy) return;
    setGenBusy(true);
    setGenError(null);
    setGenResult(null);
    try {
      setGenResult(await generateAnswer(question.trim()));
    } catch (err) {
      setGenError(toApiError(err).detail);
    } finally {
      setGenBusy(false);
    }
  };

  const onClear = async () => {
    if (!window.confirm("Clear all indexed documents?")) return;
    try {
      await clearDocuments();
      reloadCount();
    } catch (err) {
      setAddMessage(`Failed: ${toApiError(err).detail}`);
    }
  };

  const onSwitchModel = async (e: FormEvent) => {
    e.preventDefault();
    if (!modelName.trim() || switchBusy) return;
    setSwitchBusy(true);
    setSwitchMessage(null);
    try {
      const res = await switchModel(modelType, modelName.trim());
      setSwitchMessage(res.message);
    } catch (err) {
      setSwitchMessage(toApiError(err).detail);
    } finally {
      setSwitchBusy(false);
    }
  };

  return (
    <>
      <h1 className="page-header">RAG Workbench</h1>
      <p className="page-sub">Document ingestion, retrieval, and grounded generation.</p>

      {needsAuth ? (
        <div className="banner danger" role="alert">
          Authentication is required on the RAG backend.{" "}
          {token ? "Your token was rejected." : "Sign in below."}
        </div>
      ) : null}
      {needsAuth && !token ? (
        showLogin ? (
          <RagLogin />
        ) : (
          <button onClick={() => setShowLogin(true)}>Sign in</button>
        )
      ) : null}
      {token ? (
        <p className="muted">
          Signed in.{" "}
          <button
            className="secondary"
            onClick={() => {
              clearRagToken();
              reloadCount();
            }}
          >
            Sign out
          </button>
        </p>
      ) : null}

      {!needsAuth || token ? (
        <>
          <div className="card">
            <div className="form-row" style={{ justifyContent: "space-between" }}>
              <div>
                <h2 style={{ margin: 0 }}>Index</h2>
                <p className="muted" style={{ margin: 0 }}>
                  {countState.status === "ready"
                    ? `${countState.data.count} document(s) indexed`
                    : "..."}
                </p>
              </div>
              <div style={{ display: "flex", gap: "0.5rem" }}>
                <button className="secondary" onClick={reloadCount}>
                  Refresh
                </button>
                <button className="secondary" onClick={onClear}>
                  Clear all
                </button>
              </div>
            </div>
          </div>

          <div className="grid-2">
            <div className="card">
              <h2>Add text</h2>
              <form onSubmit={addText}>
                <div className="field">
                  <label htmlFor="rag-text">Document text</label>
                  <textarea id="rag-text" value={text} onChange={(e) => setText(e.target.value)} />
                </div>
                <button type="submit" disabled={addBusy || !text.trim()}>
                  {addBusy ? "Adding..." : "Add document"}
                </button>
              </form>
              <h2 style={{ marginTop: "1.25rem" }}>Upload file</h2>
              <div className="form-row">
                <input ref={fileInput} type="file" />
                <button onClick={addFile} disabled={addBusy}>
                  Upload
                </button>
              </div>
              {addMessage ? <p className="muted">{addMessage}</p> : null}
            </div>

            <div className="card">
              <h2>Switch model</h2>
              <form onSubmit={onSwitchModel}>
                <div className="field">
                  <label htmlFor="rag-model-type">Provider</label>
                  <select
                    id="rag-model-type"
                    value={modelType}
                    onChange={(e) => setModelType(e.target.value as "openai" | "anthropic")}
                  >
                    <option value="openai">OpenAI</option>
                    <option value="anthropic">Anthropic</option>
                  </select>
                </div>
                <div className="field">
                  <label htmlFor="rag-model-name">Model name</label>
                  <input
                    id="rag-model-name"
                    value={modelName}
                    onChange={(e) => setModelName(e.target.value)}
                  />
                </div>
                <button type="submit" disabled={switchBusy || !modelName.trim()}>
                  {switchBusy ? "Switching..." : "Switch"}
                </button>
              </form>
              {switchMessage ? <p className="muted">{switchMessage}</p> : null}
            </div>
          </div>

          <div className="card">
            <h2>Query</h2>
            <form className="form-row" onSubmit={runQuery}>
              <input
                aria-label="Query"
                style={{ flex: 1 }}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="What is MultiMind?"
              />
              <button type="submit" disabled={queryBusy || !query.trim()}>
                {queryBusy ? "Searching..." : "Search"}
              </button>
            </form>
            {queryError ? (
              <div className="banner danger" role="alert">
                {queryError}
              </div>
            ) : null}
            {queryResult ? (
              <div className="table-wrap" style={{ marginTop: "0.75rem" }}>
                <table>
                  <thead>
                    <tr>
                      <th>Text</th>
                      <th className="num">Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {queryResult.documents.map((d, i) => (
                      <tr key={i}>
                        <td>{d.text}</td>
                        <td className="num">{d.score != null ? d.score.toFixed(3) : "-"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : null}
          </div>

          <div className="card">
            <h2>Generate</h2>
            <form className="form-row" onSubmit={runGenerate}>
              <input
                aria-label="Question"
                style={{ flex: 1 }}
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Summarize the indexed documents."
              />
              <button type="submit" disabled={genBusy || !question.trim()}>
                {genBusy ? "Generating..." : "Ask"}
              </button>
            </form>
            {genError ? (
              <div className="banner danger" role="alert">
                {genError}
              </div>
            ) : null}
            {genResult ? (
              <>
                <pre className="output-panel" style={{ marginTop: "0.75rem" }}>
                  {genResult.text}
                </pre>
                <p className="muted">Sourced from {genResult.documents.length} document(s).</p>
              </>
            ) : null}
          </div>
        </>
      ) : null}
    </>
  );
}
