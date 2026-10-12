import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import RagPage from "../RagPage";
import { jsonResponse, stubFetch } from "../../test/helpers";

describe("RagPage", () => {
  it("shows the document count and adds a document", async () => {
    let count = 0;
    stubFetch((url) => {
      if (url === "/rag/documents/count") return jsonResponse({ count });
      if (url === "/rag/documents") {
        count = 1;
        return jsonResponse({ documents: [{ text: "hello", metadata: {} }], total: 1 });
      }
      throw new Error(`unexpected fetch: ${url}`);
    });
    const user = userEvent.setup();
    render(<RagPage />);

    expect(await screen.findByText("0 document(s) indexed")).toBeTruthy();

    await user.type(screen.getByLabelText("Document text"), "MultiMind is an AI SDK.");
    await user.click(screen.getByRole("button", { name: "Add document" }));

    expect(await screen.findByText("Added 1 document(s).")).toBeTruthy();
    expect(await screen.findByText("1 document(s) indexed")).toBeTruthy();
  });

  it("runs a query and shows results", async () => {
    stubFetch((url) => {
      if (url === "/rag/documents/count") return jsonResponse({ count: 3 });
      if (url === "/rag/query") {
        return jsonResponse({
          documents: [{ text: "MultiMind is an AI SDK.", metadata: {}, score: 0.92 }],
          total: 1,
        });
      }
      throw new Error(`unexpected fetch: ${url}`);
    });
    const user = userEvent.setup();
    render(<RagPage />);

    await user.type(await screen.findByLabelText("Query"), "What is MultiMind?");
    await user.click(screen.getByRole("button", { name: "Search" }));

    expect(await screen.findByText("MultiMind is an AI SDK.")).toBeTruthy();
    expect(screen.getByText("0.920")).toBeTruthy();
  });

  it("shows a sign-in prompt when the backend requires auth", async () => {
    stubFetch((url) => {
      if (url === "/rag/documents/count") {
        return jsonResponse({ detail: "Authentication required" }, 401);
      }
      throw new Error(`unexpected fetch: ${url}`);
    });
    render(<RagPage />);

    expect(await screen.findByRole("button", { name: "Sign in" })).toBeTruthy();
  });
});
