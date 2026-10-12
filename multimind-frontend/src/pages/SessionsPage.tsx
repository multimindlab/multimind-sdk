import { useEffect, useRef, useState, type FormEvent } from "react";
import {
  SessionDetail,
  addSessionMessage,
  createSession,
  deleteSession,
  getModels,
  getSession,
  listSessions,
} from "../api/client";
import { toApiError, useApi } from "../api/useApi";
import BackendDown from "../components/BackendDown";
import { fmtTimestamp } from "../components/format";

// The assistant reply is generated in the background on the server (see
// POST /v1/sessions/{id}/messages); poll the session a few times after
// sending so the reply appears without a manual refresh.
const POLL_ATTEMPTS = 6;
const POLL_INTERVAL_MS = 1500;

export default function SessionsPage() {
  const { state: modelsState } = useApi(getModels);
  const { state: sessionsState, reload: reloadSessions } = useApi(listSessions);

  const [newModel, setNewModel] = useState("");
  const [newSystemPrompt, setNewSystemPrompt] = useState("");
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);

  const [activeId, setActiveId] = useState<string | null>(null);
  const [detail, setDetail] = useState<SessionDetail | null>(null);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [sending, setSending] = useState(false);
  const pollTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (pollTimer.current) clearTimeout(pollTimer.current);
    };
  }, []);

  const loadDetail = async (sessionId: string) => {
    setDetailError(null);
    try {
      setDetail(await getSession(sessionId));
    } catch (err) {
      setDetailError(toApiError(err).detail);
    }
  };

  const openSession = async (sessionId: string) => {
    setActiveId(sessionId);
    setDetail(null);
    await loadDetail(sessionId);
  };

  const pollForReply = (sessionId: string, attemptsLeft: number) => {
    if (attemptsLeft <= 0) return;
    pollTimer.current = setTimeout(async () => {
      await loadDetail(sessionId);
      pollForReply(sessionId, attemptsLeft - 1);
    }, POLL_INTERVAL_MS);
  };

  const onCreate = async (e: FormEvent) => {
    e.preventDefault();
    if (!newModel || creating) return;
    setCreating(true);
    setCreateError(null);
    try {
      const session = await createSession({
        model: newModel,
        system_prompt: newSystemPrompt.trim() || undefined,
      });
      setNewSystemPrompt("");
      reloadSessions();
      await openSession(session.session_id);
    } catch (err) {
      setCreateError(toApiError(err).detail);
    } finally {
      setCreating(false);
    }
  };

  const onSend = async (e: FormEvent) => {
    e.preventDefault();
    if (!activeId || !draft.trim() || sending) return;
    const content = draft.trim();
    setDraft("");
    setSending(true);
    try {
      await addSessionMessage(activeId, { role: "user", content });
      await loadDetail(activeId);
      pollForReply(activeId, POLL_ATTEMPTS);
    } catch (err) {
      setDetailError(toApiError(err).detail);
    } finally {
      setSending(false);
    }
  };

  const onDelete = async (sessionId: string) => {
    try {
      await deleteSession(sessionId);
      if (activeId === sessionId) {
        setActiveId(null);
        setDetail(null);
      }
      reloadSessions();
    } catch (err) {
      setDetailError(toApiError(err).detail);
    }
  };

  if (sessionsState.status === "loading") return <p className="muted">Loading sessions...</p>;
  if (sessionsState.status === "error") {
    return (
      <>
        <h1 className="page-header">Sessions</h1>
        <BackendDown backend="gateway" error={sessionsState.error} />
      </>
    );
  }

  const models = modelsState.status === "ready" ? Object.keys(modelsState.data.models) : [];
  const sessions = sessionsState.data;

  return (
    <>
      <h1 className="page-header">Sessions</h1>
      <p className="page-sub">Persistent chat sessions via the gateway (in-memory, per process).</p>

      <div className="grid-2">
        <div className="card">
          <h2>New session</h2>
          <form onSubmit={onCreate}>
            <div className="field">
              <label htmlFor="session-model">Model</label>
              <select
                id="session-model"
                value={newModel}
                onChange={(e) => setNewModel(e.target.value)}
              >
                <option value="">Select a model...</option>
                {models.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            </div>
            <div className="field">
              <label htmlFor="session-system">System prompt (optional)</label>
              <input
                id="session-system"
                type="text"
                value={newSystemPrompt}
                onChange={(e) => setNewSystemPrompt(e.target.value)}
                placeholder="You are a helpful assistant."
              />
            </div>
            {createError ? (
              <div className="banner danger" role="alert">
                {createError}
              </div>
            ) : null}
            <button type="submit" disabled={creating || !newModel}>
              {creating ? "Creating..." : "Create session"}
            </button>
          </form>

          <h2 style={{ marginTop: "1.25rem" }}>All sessions</h2>
          {sessions.length === 0 ? (
            <p className="muted">No sessions yet.</p>
          ) : (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Model</th>
                    <th className="num">Messages</th>
                    <th>Updated</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {sessions.map((s) => (
                    <tr
                      key={s.session_id}
                      className={s.session_id === activeId ? "row-blocked" : undefined}
                      style={{ cursor: "pointer" }}
                      onClick={() => openSession(s.session_id)}
                    >
                      <td>{s.model}</td>
                      <td className="num">{s.message_count}</td>
                      <td>{fmtTimestamp(s.updated_at)}</td>
                      <td>
                        <button
                          className="secondary"
                          onClick={(e) => {
                            e.stopPropagation();
                            onDelete(s.session_id);
                          }}
                        >
                          Delete
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        <div className="card">
          <h2>Conversation</h2>
          {!activeId ? (
            <p className="muted">Select or create a session to view its messages.</p>
          ) : detail === null ? (
            <p className="muted">Loading...</p>
          ) : (
            <>
              {detailError ? (
                <div className="banner danger" role="alert">
                  {detailError}
                </div>
              ) : null}
              <div className="chat-thread" aria-label="Conversation">
                {detail.messages.length === 0 ? (
                  <p className="muted">No messages yet.</p>
                ) : (
                  detail.messages.map((m, i) => (
                    <div key={i} className={`chat-msg ${m.role === "user" ? "user" : "assistant"}`}>
                      {m.content}
                      <span className="chat-meta">
                        {m.model ?? m.role} &middot; {fmtTimestamp(m.timestamp)}
                      </span>
                    </div>
                  ))
                )}
              </div>
              <form className="chat-compose" onSubmit={onSend}>
                <textarea
                  aria-label="Message"
                  value={draft}
                  onChange={(e) => setDraft(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      onSend(e);
                    }
                  }}
                  placeholder="Type a message"
                />
                <button type="submit" disabled={sending || !draft.trim()}>
                  Send
                </button>
              </form>
            </>
          )}
        </div>
      </div>
    </>
  );
}
