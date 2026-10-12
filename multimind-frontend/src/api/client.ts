// Typed API client for the MultiMind backends, reached through the Vite
// dev-server proxy (see vite.config.ts):
//   Dashboard API (multimind dashboard, :8501)          -> /api/*, /db-health
//   Gateway API   (multimind backend, :8080, /gateway)  -> /v1/*,  /gw-health

// ---------------------------------------------------------------------------
// Shared

export class ApiError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

/** status 0 means the backend could not be reached at all. */
export function isBackendDown(err: unknown): boolean {
  return err instanceof ApiError && (err.status === 0 || err.status === 502 || err.status === 504);
}

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

async function doFetch(path: string, init?: RequestInit): Promise<Response> {
  let res: Response;
  try {
    res = await fetch(path, init);
  } catch (err) {
    throw new ApiError(0, err instanceof Error ? err.message : "Network error");
  }
  if (!res.ok) {
    throw new ApiError(res.status, await errorDetail(res));
  }
  return res;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await doFetch(path, init);
  return (await res.json()) as T;
}

function jsonInit(method: string, body: unknown): RequestInit {
  return {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
}

// ---------------------------------------------------------------------------
// Dashboard API types (mirrors multimind/dashboard/server.py)

export interface HealthResponse {
  status: string;
  version: string;
}

export interface AuditStats {
  events: number;
  pii_detections: number;
  blocked: number;
  by_type: Record<string, number>;
}

export interface BudgetStatus {
  max_cost: number;
  spent: number;
  remaining: number;
  exceeded: boolean;
  basis: string;
}

export interface SummaryResponse {
  audit: {
    available: boolean;
    all_time: AuditStats;
    last_24h: AuditStats;
  };
  costs: {
    available: boolean;
    total_cost: number;
    last_24h_cost: number;
    calls: number;
    models: string[];
  };
  budget: BudgetStatus | null;
}

/** One line of the ComplianceGuard JSONL audit trail. */
export interface AuditRecord {
  timestamp: string;
  direction: string;
  method: string;
  pii_types: Record<string, number>;
  count: number;
  strategy: string;
  blocked: boolean;
  tags: string[];
}

export interface AuditResponse {
  total: number;
  limit: number;
  offset: number;
  pii_types: string[];
  records: AuditRecord[];
}

export interface CostGroup {
  calls: number;
  input_tokens: number;
  output_tokens: number;
  cost: number;
}

export interface ChargebackGroup extends CostGroup {
  total_tokens: number;
  estimated_calls: number;
  unpriced_calls: number;
  share_pct: number;
}

export interface DailySpend {
  date: string;
  cost: number;
  calls: number;
}

export interface CostsResponse {
  available: boolean;
  chargeback: {
    period: string | null;
    calls: number;
    total_cost: number;
    by_tag: Record<string, ChargebackGroup>;
  };
  by_model: Record<string, CostGroup>;
  daily: DailySpend[];
}

export interface InventoryFinding {
  kind: string; // dependency | api_url | env_read | hardcoded_key | env_set
  provider: string;
  evidence: string;
  file: string | null;
  line: number | null;
  data_flow: string | null;
}

export interface InventoryResponse {
  scanned_at: string;
  cached: boolean;
  root: string;
  findings: InventoryFinding[];
  summary: {
    findings: number;
    by_provider: Record<string, number>;
    by_kind: Record<string, number>;
    by_data_flow: Record<string, number>;
    scanned_files: number;
    skipped_files: number;
  };
  risks: {
    external_data_flow_providers: string[];
    hardcoded_keys: InventoryFinding[];
  };
  note?: string;
}

export type RedactionStrategy = "mask" | "hash" | "remove";

export interface GuardrailsConfig {
  strategy: RedactionStrategy;
  block_on: string[];
  budget_max_cost: number | null;
  scan_output: boolean;
}

export interface GuardrailsGetResponse {
  path: string;
  exists: boolean;
  pii_types: string[];
  config: GuardrailsConfig;
}

export interface GuardrailsPutResponse {
  path: string;
  exists: boolean;
  config: GuardrailsConfig;
  apply_hint: string;
}

export interface ScanTextRequest {
  text: string;
  strategy?: RedactionStrategy;
}

export interface ScanFinding {
  type: string;
  start: number;
  end: number;
  text: string;
}

export interface ScanTextResponse {
  findings: ScanFinding[];
  counts: Record<string, number>;
  total: number;
  redacted: string | null;
}

export type EvidenceFormat = "md" | "html";

// ---------------------------------------------------------------------------
// Gateway API types (mirrors multimind/gateway/api.py)

export interface GatewayModelInfo {
  status: "available" | "unavailable";
  config: {
    model_name: string;
    temperature: number;
    max_tokens: number | null;
  };
}

export interface ModelsResponse {
  models: Record<string, GatewayModelInfo>;
}

export interface ChatMessage {
  role: string;
  content: string;
  model?: string | null;
  metadata?: Record<string, unknown>;
}

export interface ChatRequest {
  messages: ChatMessage[];
  model: string;
  temperature?: number;
  max_tokens?: number | null;
}

export interface ModelResponse {
  content: string;
  model: string;
  usage: Record<string, number> | null;
  finish_reason: string | null;
  timestamp: string;
}

export interface CompareRequest {
  prompt: string;
  models: string[];
  temperature?: number;
  max_tokens?: number | null;
}

export interface CompareResponse {
  responses: Record<string, ModelResponse>;
}

export interface ModelHealth {
  is_healthy: boolean;
  last_check: string;
  error_message: string | null;
  latency_ms: number | null;
  uptime_percentage: number;
}

export interface MetricsResponse {
  metrics: Record<string, unknown>;
  health: Record<string, ModelHealth>;
}

export interface SessionCreateRequest {
  model: string;
  system_prompt?: string | null;
  metadata?: Record<string, unknown>;
}

export interface SessionResponse {
  session_id: string;
  model: string;
  created_at: string;
  updated_at: string;
  message_count: number;
}

export interface SessionMessage {
  role: string;
  content: string;
  model?: string | null;
  timestamp: string;
  metadata?: Record<string, unknown>;
}

export interface SessionDetail {
  session_id: string;
  model: string;
  created_at: string;
  updated_at: string;
  messages: SessionMessage[];
}

// ---------------------------------------------------------------------------
// Dashboard API functions

export function getDashboardHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/db-health");
}

export function getSummary(): Promise<SummaryResponse> {
  return request<SummaryResponse>("/api/summary");
}

export interface AuditQuery {
  limit?: number;
  offset?: number;
  type?: string;
  blocked?: boolean;
}

export function getAudit(query: AuditQuery = {}): Promise<AuditResponse> {
  const params = new URLSearchParams();
  if (query.limit !== undefined) params.set("limit", String(query.limit));
  if (query.offset !== undefined) params.set("offset", String(query.offset));
  if (query.type !== undefined && query.type !== "") params.set("type", query.type);
  if (query.blocked !== undefined) params.set("blocked", String(query.blocked));
  const qs = params.toString();
  return request<AuditResponse>(qs ? `/api/audit?${qs}` : "/api/audit");
}

export function getCosts(): Promise<CostsResponse> {
  return request<CostsResponse>("/api/costs");
}

export function getInventory(refresh = false): Promise<InventoryResponse> {
  return request<InventoryResponse>(refresh ? "/api/inventory?refresh=true" : "/api/inventory");
}

export function getGuardrails(): Promise<GuardrailsGetResponse> {
  return request<GuardrailsGetResponse>("/api/guardrails");
}

export function putGuardrails(config: GuardrailsConfig): Promise<GuardrailsPutResponse> {
  return request<GuardrailsPutResponse>("/api/guardrails", jsonInit("PUT", config));
}

export function scanText(body: ScanTextRequest): Promise<ScanTextResponse> {
  return request<ScanTextResponse>("/api/scan-text", jsonInit("POST", body));
}

/** Fetch the evidence report; the caller decides to download or preview it. */
export async function getEvidenceReport(
  format: EvidenceFormat,
): Promise<{ blob: Blob; filename: string }> {
  const res = await doFetch(`/api/evidence-report?format=${format}`);
  const disposition = res.headers.get("Content-Disposition") ?? "";
  const match = /filename="?([^";]+)"?/.exec(disposition);
  const filename = match ? match[1] : `evidence-report.${format}`;
  return { blob: await res.blob(), filename };
}

// ---------------------------------------------------------------------------
// Gateway API functions

export function getGatewayHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/gw-health");
}

export function getModels(): Promise<ModelsResponse> {
  return request<ModelsResponse>("/v1/models");
}

export function postChat(body: ChatRequest): Promise<ModelResponse> {
  return request<ModelResponse>("/v1/chat", jsonInit("POST", body));
}

export function postCompare(body: CompareRequest): Promise<CompareResponse> {
  return request<CompareResponse>("/v1/compare", jsonInit("POST", body));
}

export function getMetrics(model?: string): Promise<MetricsResponse> {
  return request<MetricsResponse>(model ? `/v1/metrics?model=${encodeURIComponent(model)}` : "/v1/metrics");
}

export function checkHealth(model?: string): Promise<Record<string, ModelHealth>> {
  return request<Record<string, ModelHealth>>(
    "/v1/health/check" + (model ? `?model=${encodeURIComponent(model)}` : ""),
    { method: "POST" },
  );
}

export function createSession(body: SessionCreateRequest): Promise<SessionResponse> {
  return request<SessionResponse>("/v1/sessions", jsonInit("POST", body));
}

export function listSessions(): Promise<SessionResponse[]> {
  return request<SessionResponse[]>("/v1/sessions");
}

export function getSession(sessionId: string): Promise<SessionDetail> {
  return request<SessionDetail>(`/v1/sessions/${encodeURIComponent(sessionId)}`);
}

/** Fire-and-forget: the assistant reply is generated in the background on
 * the server; poll getSession() to see it appear. */
export function addSessionMessage(
  sessionId: string,
  message: ChatMessage,
): Promise<{ status: string }> {
  return request<{ status: string }>(
    `/v1/sessions/${encodeURIComponent(sessionId)}/messages`,
    jsonInit("POST", message),
  );
}

export function deleteSession(sessionId: string): Promise<{ status: string }> {
  return request<{ status: string }>(`/v1/sessions/${encodeURIComponent(sessionId)}`, {
    method: "DELETE",
  });
}
