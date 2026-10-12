import { vi } from "vitest";

/** Minimal Response stand-in covering everything the API client touches. */
export function jsonResponse(
  body: unknown,
  status = 200,
  headers: Record<string, string> = {},
): Response {
  const lookup = new Map(Object.entries(headers).map(([k, v]) => [k.toLowerCase(), v]));
  const text = typeof body === "string" ? body : JSON.stringify(body);
  // Plain object instead of a real Blob: jsdom's Blob lacks .text()
  const fakeBlob = { text: async () => text, size: text.length } as unknown as Blob;
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: String(status),
    headers: { get: (name: string) => lookup.get(name.toLowerCase()) ?? null },
    json: async () => body,
    blob: async () => fakeBlob,
  } as unknown as Response;
}

/** Stub global fetch with a URL-routing handler; returns the mock for asserts. */
export function stubFetch(
  handler: (url: string, init?: RequestInit) => Response | Promise<Response>,
) {
  const mock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) =>
    handler(String(input), init),
  );
  vi.stubGlobal("fetch", mock);
  return mock;
}

export function requestBody(init: RequestInit | undefined): unknown {
  return JSON.parse(String(init?.body));
}
