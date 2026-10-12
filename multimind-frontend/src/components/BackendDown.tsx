import { ApiError, isBackendDown } from "../api/client";
import EmptyState from "./EmptyState";

interface Props {
  backend: "dashboard" | "gateway";
  error: ApiError;
}

const COMMANDS = {
  dashboard: "multimind dashboard",
  gateway: "multimind backend",
};

const NAMES = {
  dashboard: "governance dashboard",
  gateway: "gateway",
};

/** Renders a friendly down-state, or the API error detail for real errors. */
export default function BackendDown({ backend, error }: Props) {
  if (isBackendDown(error)) {
    return (
      <EmptyState
        title={`The ${NAMES[backend]} backend is not reachable.`}
        hint="Start it with:"
        command={COMMANDS[backend]}
      />
    );
  }
  return (
    <div className="banner danger" role="alert">
      Request failed ({error.status}): {error.detail}
    </div>
  );
}
