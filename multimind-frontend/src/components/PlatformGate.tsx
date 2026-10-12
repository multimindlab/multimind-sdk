import { FormEvent, ReactNode, useState } from "react";
import { ApiError, isBackendDown } from "../api/client";
import { verifyPlatformKey } from "../api/platform";
import { setPlatformKey, usePlatformKey } from "../api/platformAuth";
import { toApiError } from "../api/useApi";

function loginError(err: ApiError): string {
  if (err.status === 401) return "Invalid API key.";
  if (err.status === 503) return `Platform server has no API key configured: ${err.detail}`;
  if (isBackendDown(err)) return "The platform backend is not reachable. Start it on port 8600.";
  return `Login failed (${err.status}): ${err.detail}`;
}

export function PlatformLogin() {
  const [key, setKey] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!key.trim() || busy) return;
    setBusy(true);
    setError(null);
    try {
      // Verify against a lightweight authed endpoint before storing anything.
      await verifyPlatformKey(key.trim());
      setPlatformKey(key.trim());
    } catch (err) {
      setError(loginError(toApiError(err)));
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <h1 className="page-header">Platform sign in</h1>
      <p className="page-sub">
        The observability platform requires an API key (single-tenant; one key for the whole
        server).
      </p>
      <div className="card" style={{ maxWidth: 480 }}>
        <form onSubmit={submit}>
          <div className="field">
            <label htmlFor="platform-key">API key</label>
            <input
              id="platform-key"
              type="password"
              value={key}
              autoComplete="off"
              placeholder="MULTIMIND_PLATFORM_KEY value"
              onChange={(e) => setKey(e.target.value)}
            />
          </div>
          {error ? (
            <div className="banner danger" role="alert">
              {error}
            </div>
          ) : null}
          <button type="submit" disabled={busy || !key.trim()}>
            {busy ? "Verifying..." : "Sign in"}
          </button>
        </form>
      </div>
    </>
  );
}

/** Guards the platform section only; governance pages stay ungated. */
export default function PlatformGate({ children }: { children: ReactNode }) {
  const key = usePlatformKey();
  if (!key) return <PlatformLogin />;
  return <>{children}</>;
}
