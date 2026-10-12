import { ApiError } from "./client";
import { getProjects, getRuns, tokenTotal } from "./platform";
import { getPlatformKey, setPlatformKey } from "./platformAuth";
import { jsonResponse, stubFetch } from "../test/helpers";

const emptyRuns = { runs: [], total: 0 };

beforeEach(() => {
  sessionStorage.clear();
});

describe("platform api client", () => {
  it("attaches the stored key as X-API-Key and hits the /platform proxy", async () => {
    setPlatformKey("secret-key");
    const fetchMock = stubFetch(() => jsonResponse({ projects: [] }));
    await getProjects();

    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toBe("/platform/api/v1/projects");
    expect((init?.headers as Record<string, string>)["X-API-Key"]).toBe("secret-key");
  });

  it("builds the runs query string from all filters", async () => {
    setPlatformKey("secret-key");
    const fetchMock = stubFetch(() => jsonResponse(emptyRuns));
    await getRuns({
      project: "demo",
      status: "error",
      run_type: "llm",
      q: "gpt",
      limit: 50,
      offset: 100,
    });
    expect(String(fetchMock.mock.calls[0][0])).toBe(
      "/platform/api/v1/runs?project=demo&status=error&run_type=llm&q=gpt&limit=50&offset=100",
    );

    await getRuns();
    expect(String(fetchMock.mock.calls[1][0])).toBe("/platform/api/v1/runs");
  });

  it("clears the stored key when the server answers 401", async () => {
    setPlatformKey("stale-key");
    stubFetch(() => jsonResponse({ detail: "Invalid API key" }, 401));

    const err = await getProjects().catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).status).toBe(401);
    expect(getPlatformKey()).toBeNull();
  });

  it("sums token objects and passes through plain totals", () => {
    expect(tokenTotal({ input: 120, output: 80, estimated: false })).toBe(200);
    expect(tokenTotal(1234)).toBe(1234);
    expect(tokenTotal(null)).toBeNull();
    expect(tokenTotal(undefined)).toBeNull();
  });
});
