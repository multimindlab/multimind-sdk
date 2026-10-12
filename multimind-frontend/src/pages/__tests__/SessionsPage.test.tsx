import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import SessionsPage from "../SessionsPage";
import { jsonResponse, stubFetch } from "../../test/helpers";

const models = {
  models: {
    openai: { status: "available", config: { model_name: "gpt-4o", temperature: 0.7, max_tokens: null } },
  },
};

const sessions = [
  {
    session_id: "sess-1",
    model: "openai",
    created_at: "2026-07-06T10:00:00",
    updated_at: "2026-07-06T10:05:00",
    message_count: 2,
  },
];

const detail = {
  session_id: "sess-1",
  model: "openai",
  created_at: "2026-07-06T10:00:00",
  updated_at: "2026-07-06T10:05:00",
  messages: [
    { role: "user", content: "Hi", timestamp: "2026-07-06T10:00:00" },
    { role: "assistant", content: "Hello!", model: "openai", timestamp: "2026-07-06T10:00:05" },
  ],
};

describe("SessionsPage", () => {
  it("lists sessions and opens a conversation", async () => {
    stubFetch((url) => {
      if (url === "/v1/models") return jsonResponse(models);
      if (url === "/v1/sessions") return jsonResponse(sessions);
      if (url === "/v1/sessions/sess-1") return jsonResponse(detail);
      throw new Error(`unexpected fetch: ${url}`);
    });
    const user = userEvent.setup();
    render(<SessionsPage />);

    const row = await screen.findByText("openai", { selector: "td" });
    await user.click(row);

    expect(await screen.findByText("Hi")).toBeTruthy();
    expect(await screen.findByText("Hello!")).toBeTruthy();
  });

  it("creates a session", async () => {
    stubFetch((url, init) => {
      if (url === "/v1/models") return jsonResponse(models);
      if (url === "/v1/sessions" && (!init || init.method === undefined)) return jsonResponse([]);
      if (url === "/v1/sessions" && init?.method === "POST") {
        return jsonResponse({
          session_id: "sess-new",
          model: "openai",
          created_at: "2026-07-06T10:00:00",
          updated_at: "2026-07-06T10:00:00",
          message_count: 0,
        });
      }
      if (url === "/v1/sessions/sess-new") {
        return jsonResponse({
          session_id: "sess-new",
          model: "openai",
          created_at: "2026-07-06T10:00:00",
          updated_at: "2026-07-06T10:00:00",
          messages: [],
        });
      }
      throw new Error(`unexpected fetch: ${url}`);
    });
    const user = userEvent.setup();
    render(<SessionsPage />);

    await user.selectOptions(await screen.findByLabelText("Model"), "openai");
    await user.click(screen.getByRole("button", { name: "Create session" }));

    expect(await screen.findByText("No messages yet.")).toBeTruthy();
  });
});
