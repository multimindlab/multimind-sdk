import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import TracesPage from "../TracesPage";
import { setPlatformKey } from "../../../api/platformAuth";
import { jsonResponse, stubFetch } from "../../../test/helpers";

function run(overrides: Record<string, unknown> = {}) {
  return {
    id: "run-1",
    project: "checkout-bot",
    name: "summarize-order",
    run_type: "llm",
    start_time: "2026-07-06T10:00:00+00:00",
    end_time: "2026-07-06T10:00:01+00:00",
    status: "success",
    error: null,
    model: "gpt-4o",
    provider: "openai",
    input_chars: 900,
    output_chars: 300,
    tokens: { input: 120, output: 80, estimated: false },
    cost: 0.0125,
    tags: ["prod"],
    latency_ms: 1000,
    feedback_score: 1,
    ...overrides,
  };
}

const projects = {
  projects: [
    { name: "checkout-bot", run_count: 1, last_run_at: null, total_cost: 0, error_rate: 0 },
  ],
};

function stubTraces(runs: unknown[], total = runs.length) {
  return stubFetch((url) => {
    if (url.includes("/api/v1/projects")) return jsonResponse(projects);
    return jsonResponse({ runs, total });
  });
}

beforeEach(() => {
  sessionStorage.clear();
  setPlatformKey("k");
});

function renderPage(initialEntry = "/traces") {
  return render(
    <MemoryRouter initialEntries={[initialEntry]}>
      <TracesPage />
    </MemoryRouter>,
  );
}

describe("TracesPage", () => {
  it("renders runs with type chip, tokens, cost, and status", async () => {
    stubTraces([run(), run({ id: "run-2", name: "lookup", status: "error", feedback_score: null })]);
    renderPage();

    expect(await screen.findByRole("link", { name: "summarize-order" })).toBeTruthy();
    expect(screen.getAllByText("llm").length).toBeGreaterThan(0);
    expect(screen.getAllByText("200").length).toBe(2); // token object summed per run
    expect(screen.getAllByText("$0.0125").length).toBe(2);
    expect(screen.getByRole("img", { name: "error" })).toBeTruthy();
  });

  it("passes status, run_type, and search filters as query parameters", async () => {
    const fetchMock = stubTraces([run()]);
    const user = userEvent.setup();
    renderPage();
    await screen.findByRole("link", { name: "summarize-order" });

    await user.selectOptions(screen.getByLabelText("Status"), "error");
    await waitFor(() => {
      expect(fetchMock.mock.calls.some(([url]) => String(url).includes("status=error"))).toBe(true);
    });

    await user.selectOptions(screen.getByLabelText("Run type"), "llm");
    await waitFor(() => {
      expect(fetchMock.mock.calls.some(([url]) => String(url).includes("run_type=llm"))).toBe(true);
    });

    await user.type(screen.getByLabelText("Search"), "gpt{Enter}");
    await waitFor(() => {
      expect(
        fetchMock.mock.calls.some(
          ([url]) => String(url).includes("q=gpt") && String(url).includes("offset=0"),
        ),
      ).toBe(true);
    });
  });

  it("filters by project taken from the URL and paginates", async () => {
    const fetchMock = stubTraces(
      Array.from({ length: 50 }, (_, i) => run({ id: `run-${i}`, name: `run ${i}` })),
      120,
    );
    const user = userEvent.setup();
    renderPage("/traces?project=checkout-bot");

    await screen.findByText("1-50 of 120");
    expect(
      fetchMock.mock.calls.some(([url]) => String(url).includes("project=checkout-bot")),
    ).toBe(true);

    await user.click(screen.getByRole("button", { name: "Next" }));
    await waitFor(() => {
      expect(fetchMock.mock.calls.some(([url]) => String(url).includes("offset=50"))).toBe(true);
    });
  });
});
