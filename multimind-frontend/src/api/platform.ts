// Typed client for the observability platform API (port 8600), reached through
// the Vite proxy: /platform/api/v1/* -> :8600/api/v1/*, /platform-health -> /health.
// Every /api/v1/* call carries the X-API-Key header from the auth store; a 401
// clears the stored key so the platform section falls back to the login view.

import { ApiError } from "./client";
import { clearPlatformKey, getPlatformKey } from "./platformAuth";

// ---------------------------------------------------------------------------
// Types (mirror the platform API contract)

export interface PlatformProject {
  name: string;
  run_count: number;
  last_run_at: string | null;
  total_cost: number;
  error_rate: number;
}

export interface ProjectsResponse {
  projects: PlatformProject[];
}

export type RunStatus = "success" | "error";

export interface TokenUsage {
  input?: number;
  output?: number;
  estimated?: boolean;
}

/** The server stores tokens as {input, output, estimated}; tolerate a plain total too. */
export type RunTokens = TokenUsage | number | null;

export function tokenTotal(tokens: RunTokens | undefined): number | null {
  if (tokens === null || tokens === undefined) return null;
  if (typeof tokens === "number") return tokens;
  return (tokens.input ?? 0) + (tokens.output ?? 0);
}

export interface PlatformRun {
  id: string;
  project: string;
  name: string;
  run_type: string;
  start_time: string;
  end_time: string | null;
  status: RunStatus;
  error: string | null;
  model: string | null;
  provider: string | null;
  input_chars: number | null;
  output_chars: number | null;
  input_sha256?: string | null;
  output_sha256?: string | null;
  inputs?: unknown;
  outputs?: unknown;
  tokens: RunTokens;
  cost: number | null;
  tags: string[];
  latency_ms: number | null;
  feedback_score: number | null;
}

export interface RunsResponse {
  runs: PlatformRun[];
  total: number;
}

/** Flat span; parent_id links it into the trace tree. */
export interface PlatformSpan {
  id: string;
  parent_id: string | null;
  name: string;
  run_type: string;
  start_time: string;
  end_time: string | null;
  status: RunStatus;
  error?: string | null;
  model?: string | null;
  provider?: string | null;
  input_chars?: number | null;
  output_chars?: number | null;
  input_sha256?: string | null;
  output_sha256?: string | null;
  inputs?: unknown;
  outputs?: unknown;
  tokens?: RunTokens;
  cost?: number | null;
  latency_ms?: number | null;
}

export interface RunFeedback {
  score: number;
  comment: string | null;
  created_at: string;
}

export interface RunDetailResponse {
  run: PlatformRun;
  spans: PlatformSpan[];
  feedback: RunFeedback[];
}

export interface FeedbackRequest {
  score: -1 | 1;
  comment?: string;
}

export interface DailyStat {
  date: string;
  runs: number;
  errors: number;
  cost: number;
  tokens: number;
  p50_latency_ms: number;
  p95_latency_ms: number;
}

export interface StatsResponse {
  daily: DailyStat[];
}

export interface PlatformHealth {
  status: string;
}

// ---------------------------------------------------------------------------
// Request plumbing

async function errorDetail(res: Response): Promise<string> {
  let body: unknown;
  try {
    body = await res.json();
  } catch {
    return `${res.status} ${res.statusText}`;
  }
  if (body && typeof body === "object" && "detail" in body) {
    const detail = (body as { detail: unknown }).detail;
    if (typeof detail === "string") return detail;
    return JSON.stringify(detail);
  }
  return `${res.status} ${res.statusText}`;
}

async function rawRequest<T>(path: string, apiKey: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`/platform${path}`, {
      ...init,
      headers: { ...(init?.headers as Record<string, string>), "X-API-Key": apiKey },
    });
  } catch (err) {
    throw new ApiError(0, err instanceof Error ? err.message : "Network error");
  }
  if (!res.ok) {
    throw new ApiError(res.status, await errorDetail(res));
  }
  try {
    return (await res.json()) as T;
  } catch {
    // 201/204 responses may have no body.
    return undefined as T;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  try {
    return await rawRequest<T>(path, getPlatformKey() ?? "", init);
  } catch (err) {
    // Stored key rejected: drop it so the auth gate shows the login view.
    if (err instanceof ApiError && err.status === 401) clearPlatformKey();
    throw err;
  }
}

function jsonInit(method: string, body: unknown): RequestInit {
  return {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
}

// ---------------------------------------------------------------------------
// Endpoints

export function getPlatformHealth(): Promise<PlatformHealth> {
  return fetch("/platform-health").then((res) => {
    if (!res.ok) throw new ApiError(res.status, res.statusText);
    return res.json() as Promise<PlatformHealth>;
  });
}

/** Authed probe used by the login view; does NOT touch the stored key on 401. */
export function verifyPlatformKey(candidate: string): Promise<ProjectsResponse> {
  return rawRequest<ProjectsResponse>("/api/v1/projects", candidate);
}

export function getProjects(): Promise<ProjectsResponse> {
  return request<ProjectsResponse>("/api/v1/projects");
}

export interface RunsQuery {
  project?: string;
  status?: string;
  run_type?: string;
  q?: string;
  limit?: number;
  offset?: number;
}

export function getRuns(query: RunsQuery = {}): Promise<RunsResponse> {
  const params = new URLSearchParams();
  if (query.project) params.set("project", query.project);
  if (query.status) params.set("status", query.status);
  if (query.run_type) params.set("run_type", query.run_type);
  if (query.q) params.set("q", query.q);
  if (query.limit !== undefined) params.set("limit", String(query.limit));
  if (query.offset !== undefined) params.set("offset", String(query.offset));
  const qs = params.toString();
  return request<RunsResponse>(qs ? `/api/v1/runs?${qs}` : "/api/v1/runs");
}

export function getRun(id: string): Promise<RunDetailResponse> {
  return request<RunDetailResponse>(`/api/v1/runs/${encodeURIComponent(id)}`);
}

export function postFeedback(id: string, body: FeedbackRequest): Promise<void> {
  return request<void>(`/api/v1/runs/${encodeURIComponent(id)}/feedback`, jsonInit("POST", body));
}

export function getStats(project?: string, days = 30): Promise<StatsResponse> {
  const params = new URLSearchParams();
  if (project) params.set("project", project);
  params.set("days", String(days));
  return request<StatsResponse>(`/api/v1/stats?${params.toString()}`);
}
