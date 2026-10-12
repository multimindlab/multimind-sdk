import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import MonitoringPage from "../MonitoringPage";
import { setPlatformKey } from "../../../api/platformAuth";
import { jsonResponse, stubFetch } from "../../../test/helpers";

const stats = {
  daily: [
    {
      date: "2026-07-05",
      runs: 40,
      errors: 4,
      cost: 1.25,
      tokens: 52000,
      p50_latency_ms: 800,
      p95_latency_ms: 2400,
    },
    {
      date: "2026-07-06",
      runs: 55,
      errors: 0,
      cost: 2.1,
      tokens: 61000,
      p50_latency_ms: 750,
      p95_latency_ms: 1900,
    },
  ],
};

const projects = {
  projects: [
    { name: "checkout-bot", run_count: 95, last_run_at: null, total_cost: 3.35, error_rate: 0.04 },
  ],
};

function stubMonitoring() {
  return stubFetch((url) => {
    if (url.includes("/api/v1/projects")) return jsonResponse(projects);
    return jsonResponse(stats);
  });
}

beforeEach(() => {
  sessionStorage.clear();
  setPlatformKey("k");
});

describe("MonitoringPage", () => {
  it("renders all four 30-day charts from the stats series", async () => {
    stubMonitoring();
    render(<MonitoringPage />);

    expect(await screen.findByRole("img", { name: "Runs per day" })).toBeTruthy();
    expect(screen.getByRole("img", { name: "Cost per day" })).toBeTruthy();
    expect(screen.getByRole("img", { name: "Latency p50 and p95" })).toBeTruthy();
    expect(screen.getByRole("img", { name: "Token volume" })).toBeTruthy();
    expect(screen.getByText("p50")).toBeTruthy();
    expect(screen.getByText("p95")).toBeTruthy();
    // Axis labels come from the mocked daily dates.
    expect(screen.getAllByText("07-05").length).toBeGreaterThan(0);
    expect(screen.getAllByText("07-06").length).toBeGreaterThan(0);
  });

  it("requests stats for the selected project", async () => {
    const fetchMock = stubMonitoring();
    const user = userEvent.setup();
    render(<MonitoringPage />);
    await screen.findByRole("img", { name: "Runs per day" });

    await user.selectOptions(screen.getByLabelText("Project"), "checkout-bot");
    await waitFor(() => {
      expect(
        fetchMock.mock.calls.some(
          ([url]) => String(url).includes("project=checkout-bot") && String(url).includes("days=30"),
        ),
      ).toBe(true);
    });
  });
});
