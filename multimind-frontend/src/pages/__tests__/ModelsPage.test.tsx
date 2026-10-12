import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import ModelsPage from "../ModelsPage";
import { jsonResponse, stubFetch } from "../../test/helpers";

const models = {
  models: {
    openai: {
      status: "available",
      config: { model_name: "gpt-4o", temperature: 0.7, max_tokens: null },
    },
    anthropic: {
      status: "unavailable",
      config: { model_name: "claude-3-sonnet", temperature: 0.7, max_tokens: null },
    },
  },
};

describe("ModelsPage", () => {
  it("renders the catalog", async () => {
    stubFetch((url) => {
      if (url === "/v1/models") return jsonResponse(models);
      throw new Error(`unexpected fetch: ${url}`);
    });
    render(<ModelsPage />);

    expect(await screen.findByText("openai", { selector: "td" })).toBeTruthy();
    expect(screen.getByText("gpt-4o")).toBeTruthy();
    expect(screen.getByText("unavailable")).toBeTruthy();
  });

  it("runs a compare and shows per-model output", async () => {
    stubFetch((url) => {
      if (url === "/v1/models") return jsonResponse(models);
      if (url === "/v1/compare") {
        return jsonResponse({
          responses: {
            openai: {
              content: "A limerick about vectors",
              model: "openai",
              usage: null,
              finish_reason: "stop",
              timestamp: "2026-07-06T10:00:00",
            },
          },
        });
      }
      throw new Error(`unexpected fetch: ${url}`);
    });
    const user = userEvent.setup();
    render(<ModelsPage />);

    await user.type(await screen.findByLabelText("Prompt"), "Write a limerick");
    await user.click(screen.getByRole("checkbox", { name: "openai" }));
    await user.click(screen.getByRole("button", { name: "Compare" }));

    expect(await screen.findByText("A limerick about vectors")).toBeTruthy();
  });

  it("shows the gateway-down state", async () => {
    stubFetch(() => {
      throw new Error("network down");
    });
    render(<ModelsPage />);
    expect(await screen.findByText("multimind backend")).toBeTruthy();
  });
});
