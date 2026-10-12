import { DependencyList, useEffect, useState } from "react";
import { ApiError } from "./client";

export type LoadState<T> =
  | { status: "loading" }
  | { status: "ready"; data: T }
  | { status: "error"; error: ApiError };

export function toApiError(err: unknown): ApiError {
  return err instanceof ApiError ? err : new ApiError(0, err instanceof Error ? err.message : String(err));
}

/** Load data on mount (and whenever deps change); reload() refetches. */
export function useApi<T>(load: () => Promise<T>, deps: DependencyList = []) {
  const [state, setState] = useState<LoadState<T>>({ status: "loading" });
  const [tick, setTick] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setState({ status: "loading" });
    load().then(
      (data) => {
        if (!cancelled) setState({ status: "ready", data });
      },
      (err) => {
        if (!cancelled) setState({ status: "error", error: toApiError(err) });
      },
    );
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, tick]);

  return { state, setState, reload: () => setTick((t) => t + 1) };
}
