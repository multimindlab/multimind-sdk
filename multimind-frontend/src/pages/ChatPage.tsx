import { useEffect, useMemo, useRef, useState } from "react";
import { ApiError, ChatMessage, getModels, isBackendDown, postChat } from "../api/client";
import { toApiError, useApi } from "../api/useApi";
import EmptyState from "../components/EmptyState";

export default function ChatPage() {
  const { state: modelsState } = useApi(getModels);

  const [model, setModel] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const threadRef = useRef<HTMLDivElement>(null);

  const models = useMemo(
    () => (modelsState.status === "ready" ? Object.entries(modelsState.data.models) : []),
    [modelsState],
  );

  useEffect(() => {
    if (models.length > 0 && model === "") {
      const firstAvailable = models.find(([, info]) => info.status === "available");
      setModel(firstAvailable ? firstAvailable[0] : models[0][0]);
    }
  }, [models, model]);

  useEffect(() => {
    // scrollTop assignment instead of scrollTo() for jsdom compatibility
    if (threadRef.current) threadRef.current.scrollTop = threadRef.current.scrollHeight;
  }, [messages, pending]);

  const send = async () => {
    const content = draft.trim();
    if (content === "" || model === "" || pending) return;
    const userMessage: ChatMessage = { role: "user", content };
    const history = [...messages, userMessage];
    setMessages(history);
    setDraft("");
    setPending(true);
    setError(null);
    try {
      const reply = await postChat({ model, messages: history, temperature: 0.7 });
      setMessages([...history, { role: "assistant", content: reply.content, model: reply.model }]);
    } catch (err) {
      setError(toApiError(err));
    } finally {
      setPending(false);
    }
  };

  if (modelsState.status === "loading") {
    return <p className="muted">Connecting to the gateway...</p>;
  }

  if (modelsState.status === "error") {
    const err = modelsState.error;
    return (
      <>
        <h1 className="page-header">Chat</h1>
        {isBackendDown(err) ? (
          <EmptyState
            title="The gateway backend is not reachable."
            hint="Start it with:"
            command="multimind backend"
          />
        ) : (
          <>
            <div className="banner danger" role="alert">
              Gateway error ({err.status}): {err.detail}
            </div>
            <EmptyState
              title="Chat needs at least one configured provider."
              hint="Set a provider API key (for example OPENAI_API_KEY or ANTHROPIC_API_KEY) in the gateway environment, then restart:"
              command="multimind backend"
            />
          </>
        )}
      </>
    );
  }

  return (
    <>
      <h1 className="page-header">Chat</h1>
      <p className="page-sub">Talk to a model through the MultiMind gateway.</p>

      <div className="card">
        <div className="form-row" style={{ marginBottom: "0.75rem" }}>
          <div>
            <label htmlFor="chat-model">Model</label>
            <select id="chat-model" value={model} onChange={(e) => setModel(e.target.value)}>
              {models.map(([name, info]) => (
                <option key={name} value={name} disabled={info.status !== "available"}>
                  {name} ({info.config.model_name}){info.status !== "available" ? " - unavailable" : ""}
                </option>
              ))}
            </select>
          </div>
          <button
            className="secondary"
            onClick={() => {
              setMessages([]);
              setError(null);
            }}
            disabled={messages.length === 0}
          >
            Clear conversation
          </button>
        </div>

        {error ? (
          <div className="banner danger" role="alert">
            Chat failed ({error.status === 0 ? "gateway unreachable" : error.status}): {error.detail}
          </div>
        ) : null}

        <div className="chat-thread" ref={threadRef} aria-label="Conversation">
          {messages.length === 0 ? (
            <p className="muted">No messages yet. Say hello below.</p>
          ) : (
            messages.map((m, i) => (
              <div key={i} className={`chat-msg ${m.role === "user" ? "user" : "assistant"}`}>
                {m.content}
                {m.role === "assistant" && m.model ? (
                  <span className="chat-meta">{m.model}</span>
                ) : null}
              </div>
            ))
          )}
          {pending ? <div className="chat-msg assistant muted">Thinking...</div> : null}
        </div>

        <div className="chat-compose">
          <textarea
            aria-label="Message"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
            placeholder="Type a message (Enter to send, Shift+Enter for newline)"
          />
          <button onClick={send} disabled={pending || draft.trim() === "" || model === ""}>
            Send
          </button>
        </div>
      </div>
    </>
  );
}
