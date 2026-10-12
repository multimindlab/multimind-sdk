// Typed client for multimind/gateway/rag_api.py, reached through the Vite
// dev-server proxy at /rag (see vite.config.ts), which forwards to the
// backend aggregator's /rag mount unrewritten (rag_api's own routes are
// unprefixed, so /rag/documents on the wire hits rag_api's /documents).
//
// Auth is optional server-side (open unless JWT_SECRET or API_KEYS is
// configured — see rag_api.py _auth_configured); when a token is present in
// ragAuth it is sent as a Bearer header, otherwise requests go out plain.

import { ApiError } from "./client";
import { getRagToken } from "./ragAuth";

function authHeaders(): Record<string, string> {
  const token = getRagToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, {
      ...init,
      headers: { ...authHeaders(), ...(init.headers ?? {}) },
    });
  } catch (err) {
    throw new ApiError(0, err instanceof Error ? err.message : "Network error");
  }
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body && typeof body === "object" && "detail" in body) {
        detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
      }
    } catch {
      // keep the status-line fallback
    }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as T;
}

export interface HealthResponse {
  status: string;
  version: string;
}

export function getRagHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/rag-health");
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
}

/** POST /rag/token expects form-encoded username/password, not JSON. */
export async function login(username: string, password: string): Promise<TokenResponse> {
  const body = new URLSearchParams({ username, password });
  return request<TokenResponse>("/rag/token", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body,
  });
}

export interface DocumentInput {
  text: string;
  metadata?: Record<string, unknown>;
}

export interface DocumentsAddResponse {
  documents: { text: string; metadata: Record<string, unknown> }[];
  total: number;
}

export function addDocuments(documents: DocumentInput[]): Promise<DocumentsAddResponse> {
  return request<DocumentsAddResponse>("/rag/documents", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ documents }),
  });
}

export function uploadFile(file: File): Promise<DocumentsAddResponse> {
  const form = new FormData();
  form.append("file", file);
  return request<DocumentsAddResponse>("/rag/files", { method: "POST", body: form });
}

export interface QueryResult {
  text: string;
  metadata: Record<string, unknown>;
  score: number | null;
}

export interface QueryResponse {
  documents: QueryResult[];
  total: number;
}

export function queryDocuments(query: string, topK = 3): Promise<QueryResponse> {
  return request<QueryResponse>("/rag/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k: topK }),
  });
}

export interface GenerateResponse {
  text: string;
  documents: QueryResult[];
}

export function generateAnswer(query: string, topK = 3): Promise<GenerateResponse> {
  return request<GenerateResponse>("/rag/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k: topK }),
  });
}

export function clearDocuments(): Promise<{ message: string }> {
  return request<{ message: string }>("/rag/documents", { method: "DELETE" });
}

export function getDocumentCount(): Promise<{ count: number }> {
  return request<{ count: number }>("/rag/documents/count");
}

export function switchModel(
  modelType: "openai" | "anthropic",
  modelName: string,
): Promise<{ message: string }> {
  const body = new URLSearchParams({ model_type: modelType, model_name: modelName });
  return request<{ message: string }>("/rag/models/switch", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body,
  });
}
