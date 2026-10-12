// Typed client for the subset of multimind/gateway/compliance_api.py that is
// actually functional. Reached through the Vite dev-server proxy at /v1/compliance
// (see vite.config.ts), which forwards to the backend aggregator's /gateway mount.
//
// NOT wired here on purpose: POST /monitor, POST /report, and POST /example/{type}.
// /monitor and /report unconditionally raise NotImplementedError server-side
// (ModelWatermarking.watermark_model refuses to fabricate a watermark — see
// multimind/compliance/advanced.py) and always return HTTP 500. /example/{type}
// imports from the examples/ tree, which is not part of the installed package.
// Building UI against endpoints that are guaranteed to fail would just move the
// dishonesty from the backend into the frontend.

import { ApiError } from "./client";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, init);
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

export function getRegulations(): Promise<string[]> {
  return request<string[]>("/v1/compliance/regulations");
}

export function getHealthcareUseCases(): Promise<string[]> {
  return request<string[]>("/v1/compliance/healthcare/use-cases");
}

export interface DashboardMetrics {
  total_checks: number;
  passed_checks: number;
  failed_checks: number;
  compliance_score: number;
  recent_issues: Record<string, unknown>[];
  trend_data: Record<string, number[]>;
  alerts: Record<string, unknown>[];
}

export function getDashboardMetrics(
  organizationId: string,
  timeRange = "7d",
): Promise<DashboardMetrics> {
  const params = new URLSearchParams({ organization_id: organizationId, time_range: timeRange });
  return request<DashboardMetrics>(`/v1/compliance/dashboard?${params}`);
}

export interface ComplianceAlert {
  status?: string;
  severity?: string;
  [key: string]: unknown;
}

export function getAlerts(
  organizationId: string,
  status = "active",
): Promise<ComplianceAlert[]> {
  const params = new URLSearchParams({ organization_id: organizationId, status });
  return request<ComplianceAlert[]>(`/v1/compliance/alerts?${params}`);
}

export function configureAlerts(
  organizationId: string,
  alertRules: Record<string, unknown>,
): Promise<{ status: string; message: string }> {
  const params = new URLSearchParams({ organization_id: organizationId });
  return request<{ status: string; message: string }>(
    `/v1/compliance/alerts/configure?${params}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(alertRules),
    },
  );
}
