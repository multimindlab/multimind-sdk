import { ApiError, getAudit, getSummary } from "./client";
import { jsonResponse, stubFetch } from "../test/helpers";

const emptyAudit = { total: 0, limit: 10, offset: 20, pii_types: [], records: [] };

describe("api client", () => {
  it("builds the audit query string from all filters", async () => {
    const fetchMock = stubFetch(() => jsonResponse(emptyAudit));
    await getAudit({ limit: 10, offset: 20, type: "email", blocked: true });
    expect(String(fetchMock.mock.calls[0][0])).toBe(
      "/api/audit?limit=10&offset=20&type=email&blocked=true",
    );

    await getAudit();
    expect(String(fetchMock.mock.calls[1][0])).toBe("/api/audit");
  });

  it("throws ApiError with status and detail on HTTP errors", async () => {
    stubFetch(() => jsonResponse({ detail: "Could not parse guardrails file" }, 500));
    const err = await getSummary().catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).status).toBe(500);
    expect((err as ApiError).detail).toBe("Could not parse guardrails file");
  });

  it("maps network failures to ApiError with status 0", async () => {
    stubFetch(() => {
      throw new TypeError("fetch failed");
    });
    const err = await getSummary().catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).status).toBe(0);
  });
});
