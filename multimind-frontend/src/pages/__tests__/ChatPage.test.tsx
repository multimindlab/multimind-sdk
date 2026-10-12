import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import ChatPage from "../ChatPage";
import { jsonResponse, requestBody, stubFetch } from "../../test/helpers";

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

const reply = {
  content: "Hello from the model",
  model: "openai",
  usage: { total_tokens: 12 },
  finish_reason: "stop",
  timestamp: "2026-07-06T10:00:00",
};

describe("ChatPage", () => {
  it("posts the conversation to /v1/chat and renders the reply", async () => {
    const fetchMock = stubFetch((url) => {
      if (url === "/v1/models") return jsonResponse(models);
      if (url === "/v1/chat") return jsonResponse(reply);
      throw new Error(`unexpected fetch: ${url}`);
    });
    const user = userEvent.setup();
    render(<ChatPage />);

    await user.type(await screen.findByLabelText("Message"), "Hi there");
    await user.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByText("Hello from the model")).toBeTruthy();
    const chatCall = fetchMock.mock.calls.find(([url]) => String(url) === "/v1/chat");
    expect(chatCall).toBeTruthy();
    expect(requestBody(chatCall![1])).toEqual({
      messages: [{ role: "user", content: "Hi there" }],
      model: "openai",
      temperature: 0.7,
    });
  });

  it("shows the keyless empty state when the gateway has no providers", async () => {
    stubFetch((url) => {
      if (url === "/v1/models") {
        return jsonResponse(
          { detail: "No models are properly configured. Please check your API keys." },
          503,
        );
      }
      throw new Error(`unexpected fetch: ${url}`);
    });
    render(<ChatPage />);

    const banner = await screen.findByRole("alert");
    expect(banner.textContent).toContain("No models are properly configured");
    expect(screen.getByText("multimind backend")).toBeTruthy();
  });

  it("surfaces the API error detail when a chat request fails", async () => {
    stubFetch((url) => {
      if (url === "/v1/models") return jsonResponse(models);
      if (url === "/v1/chat") {
        return jsonResponse({ detail: "Model openai is not available" }, 400);
      }
      throw new Error(`unexpected fetch: ${url}`);
    });
    const user = userEvent.setup();
    render(<ChatPage />);

    await user.type(await screen.findByLabelText("Message"), "Hi");
    await user.click(screen.getByRole("button", { name: "Send" }));

    const banner = await screen.findByRole("alert");
    expect(banner.textContent).toContain("Model openai is not available");
  });
});
