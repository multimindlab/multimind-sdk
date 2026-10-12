import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import TraceDetailPage from "../TraceDetailPage";
import { setPlatformKey } from "../../../api/platformAuth";
import { jsonResponse, requestBody, stubFetch } from "../../../test/helpers";

// 4s trace window: root spans all of it, llm child covers 1s..2s (25% wide at 25%).
const detail = {
  run: {
    id: "run-1",
    project: "checkout-bot",
    name: "summarize-order",
    run_type: "chain",
    start_time: "2026-07-06T10:00:00.000Z",
    end_time: "2026-07-06T10:00:04.000Z",
    status: "success",
    error: null,
    model: "gpt-4o",
    provider: "openai",
    input_chars: 900,
    output_chars: 300,
    tokens: { input: 120, output: 80, estimated: false },
    cost: 0.0125,
    tags: ["prod", "v2"],
    latency_ms: 4000,
    feedback_score: null,
  },
  spans: [
    {
      id: "s-root",
      parent_id: null,
      name: "chain-root",
      run_type: "chain",
      start_time: "2026-07-06T10:00:00.000Z",
      end_time: "2026-07-06T10:00:04.000Z",
      status: "success",
    },
    {
      id: "s-llm",
      parent_id: "s-root",
      name: "llm-call",
      run_type: "llm",
      start_time: "2026-07-06T10:00:01.000Z",
      end_time: "2026-07-06T10:00:02.000Z",
      status: "success",
      model: "gpt-4o",
      input_chars: 900,
      output_chars: 300,
      input_sha256: "abc123",
      inputs: { prompt: "hello" },
    },
  ],
  feedback: [{ score: 1, comment: "solid", created_at: "2026-07-06T11:00:00+00:00" }],
};

beforeEach(() => {
  sessionStorage.clear();
  setPlatformKey("k");
});

function renderDetail(handler = () => jsonResponse(detail)) {
  const fetchMock = stubFetch((url, init) => {
    if (url.includes("/feedback") && init?.method === "POST") return jsonResponse({}, 201);
    return handler();
  });
  render(
    <MemoryRouter initialEntries={["/traces/run-1"]}>
      <Routes>
        <Route path="/traces/:id" element={<TraceDetailPage />} />
      </Routes>
    </MemoryRouter>,
  );
  return fetchMock;
}

describe("TraceDetailPage", () => {
  it("renders the waterfall with bars positioned and scaled by span times", async () => {
    renderDetail();

    await screen.findByText("chain-root");
    const rootBar = screen.getByTestId("wf-bar-s-root");
    expect(rootBar.style.left).toBe("0%");
    expect(rootBar.style.width).toBe("100%");

    const llmBar = screen.getByTestId("wf-bar-s-llm");
    expect(llmBar.style.left).toBe("25%");
    expect(llmBar.style.width).toBe("25%");

    // Child is indented under its parent.
    const rootName = screen.getByRole("button", { name: /chain-root/ });
    const llmName = screen.getByRole("button", { name: /llm-call/ });
    expect(parseFloat(llmName.style.paddingLeft)).toBeGreaterThan(
      parseFloat(rootName.style.paddingLeft),
    );
  });

  it("opens the span side panel with fields on click", async () => {
    const user = userEvent.setup();
    renderDetail();

    await user.click(await screen.findByRole("button", { name: /llm-call/ }));
    const panel = screen.getByLabelText("Span details");
    expect(panel.textContent).toContain("Input chars");
    expect(panel.textContent).toContain("900");
    expect(panel.textContent).toContain("abc123");
    expect(panel.textContent).toContain('"prompt": "hello"');
    // Outputs absent on the fixture, so the section is not rendered.
    expect(screen.queryByText("Outputs")).toBeNull();
  });

  it("posts thumbs feedback with the comment", async () => {
    const user = userEvent.setup();
    const fetchMock = renderDetail();

    await user.click(await screen.findByRole("button", { name: "rate up" }));
    await user.type(screen.getByLabelText("Feedback comment"), "great answer");
    await user.click(screen.getByRole("button", { name: "Submit feedback" }));

    const call = fetchMock.mock.calls.find(
      ([url, init]) => String(url).includes("/api/v1/runs/run-1/feedback") && init?.method === "POST",
    );
    expect(call).toBeTruthy();
    expect(requestBody(call![1])).toEqual({ score: 1, comment: "great answer" });
  });

  it("shows the run header stats and existing feedback", async () => {
    renderDetail();

    expect(await screen.findByText("summarize-order")).toBeTruthy();
    // "4.00s" shows in the latency tile and again as the root span duration.
    expect(screen.getAllByText("4.00s").length).toBeGreaterThan(0);
    expect(screen.getByText("$0.0125")).toBeTruthy();
    expect(screen.getByText("200")).toBeTruthy();
    expect(screen.getByText("gpt-4o")).toBeTruthy();
    expect(screen.getByText("prod")).toBeTruthy();
    expect(screen.getByText("solid")).toBeTruthy();
  });
});
