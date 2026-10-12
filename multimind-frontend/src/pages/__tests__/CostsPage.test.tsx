import { render, screen } from "@testing-library/react";
import CostsPage from "../CostsPage";
import { jsonResponse, stubFetch } from "../../test/helpers";

const costs = {
  available: true,
  chargeback: {
    period: null,
    calls: 10,
    total_cost: 8.0,
    by_tag: {
      "team-a": {
        calls: 6,
        input_tokens: 1200,
        output_tokens: 800,
        total_tokens: 2000,
        cost: 5.0,
        estimated_calls: 0,
        unpriced_calls: 0,
        share_pct: 62.5,
      },
      "(untagged)": {
        calls: 4,
        input_tokens: 700,
        output_tokens: 300,
        total_tokens: 1000,
        cost: 3.0,
        estimated_calls: 1,
        unpriced_calls: 0,
        share_pct: 37.5,
      },
    },
  },
  by_model: {
    "gpt-4o": { calls: 10, input_tokens: 1900, output_tokens: 1100, cost: 8.0 },
  },
  daily: [{ date: "2026-07-01", cost: 8.0, calls: 10 }],
};

describe("CostsPage", () => {
  it("renders chargeback shares and the per-model table", async () => {
    stubFetch(() => jsonResponse(costs));
    render(<CostsPage />);

    expect(await screen.findByText("team-a")).toBeTruthy();
    expect(screen.getByText("62.5%")).toBeTruthy();
    expect(screen.getByText("(untagged)")).toBeTruthy();
    expect(screen.getByText("gpt-4o")).toBeTruthy();
    expect(screen.getByRole("img", { name: "Daily spend" })).toBeTruthy();
  });

  it("shows an empty state when no cost log exists", async () => {
    stubFetch(() =>
      jsonResponse({
        available: false,
        chargeback: { period: null, calls: 0, total_cost: 0.0, by_tag: {} },
        by_model: {},
        daily: [],
      }),
    );
    render(<CostsPage />);

    expect(await screen.findByText("No cost data yet.")).toBeTruthy();
  });
});
