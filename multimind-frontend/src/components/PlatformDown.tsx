import { ApiError, isBackendDown } from "../api/client";
import EmptyState from "./EmptyState";

/** Down-state / error banner for the observability platform backend. */
export default function PlatformDown({ error }: { error: ApiError }) {
  if (isBackendDown(error)) {
    return (
      <EmptyState
        title="The platform backend is not reachable."
        hint="Start it on port 8600 with:"
        command="MULTIMIND_PLATFORM_KEY=... python -m backend.app"
      />
    );
  }
  if (error.status === 401) {
    return (
      <div className="banner warn" role="alert">
        Session expired: the stored API key was rejected. Sign in again.
      </div>
    );
  }
  return (
    <div className="banner danger" role="alert">
      Request failed ({error.status}): {error.detail}
    </div>
  );
}
