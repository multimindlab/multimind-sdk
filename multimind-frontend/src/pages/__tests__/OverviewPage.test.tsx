import { render, screen } from "@testing-library/react";
import OverviewPage from "../OverviewPage";
import { jsonResponse, stubFetch } from "../../test/helpers";

const summary = {
  audit: {
    available: true,
    all_time: { events: 42, pii_detections: 61, blocked: 5, by_type: { email: 30, ssn: 12 } },
    last_24h: { events: 7, pii_detections: 9, blocked: 1, by_type: { email: 6 } },
  },
  costs: {
    available: true,
    total_cost: 12.34,
    last_24h_cost: 1.5,
    calls: 88,
    models: ["gpt-4o-mini", "llama3"],
  },
  budget: null,
};

const costs = {
  available: true,
  chargeback: { period: null, calls: 88, total_cost: 12.34, by_tag: {} },
  by_model: {},
  daily: [
    { date: "2026-07-01", cost: 1.2, calls: 4 },
    { date: "2026-07-02", cost: 2.4, calls: 8 },
  ],
};

describe("OverviewPage", () => {
  it("renders stat tiles and charts from the summary and costs APIs", async () => {
    stubFetch((url) => {
      if (url.startsWith("/api/summary")) return jsonResponse(summary);
      if (url.startsWith("/api/costs")) return jsonResponse(costs);
      throw new Error(`unexpected fetch: ${url}`);
    });
    render(<OverviewPage />);

    expect(await screen.findByText("42")).toBeTruthy(); // PII events
    expect(screen.getByText("5")).toBeTruthy(); // blocked
    expect(screen.getByText("$12.34")).toBeTruthy(); // spend
    expect(screen.getByText("2")).toBeTruthy(); // models in use
    expect(screen.getByText("gpt-4o-mini, llama3")).toBeTruthy();
    expect(screen.getByRole("img", { name: "Daily spend, last 30 days" })).toBeTruthy();
    expect(screen.getByRole("img", { name: "PII detections by type" })).toBeTruthy();
  });

  it("shows the dashboard down state when the backend is unreachable", async () => {
    stubFetch(() => {
      throw new TypeError("fetch failed");
    });
    render(<OverviewPage />);

    expect(await screen.findByText(/not reachable/)).toBeTruthy();
    expect(screen.getByText("multimind dashboard")).toBeTruthy();
  });
});
