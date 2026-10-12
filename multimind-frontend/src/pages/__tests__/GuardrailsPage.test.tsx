import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import GuardrailsPage from "../GuardrailsPage";
import { jsonResponse, requestBody, stubFetch } from "../../test/helpers";

const getResponse = {
  path: "guardrails.json",
  exists: true,
  pii_types: ["email", "phone", "ssn", "credit_card"],
  config: { strategy: "mask", block_on: ["ssn"], budget_max_cost: null, scan_output: true },
};

describe("GuardrailsPage", () => {
  it("PUTs the edited config and shows the success toast with the restart hint", async () => {
    const fetchMock = stubFetch((url, init) => {
      if (init?.method === "PUT") {
        return jsonResponse({
          path: "guardrails.json",
          exists: true,
          config: requestBody(init),
          apply_hint: "multimind serve --config guardrails.json",
        });
      }
      if (url.startsWith("/api/guardrails")) return jsonResponse(getResponse);
      throw new Error(`unexpected fetch: ${url}`);
    });
    const user = userEvent.setup();
    render(<GuardrailsPage />);

    await user.selectOptions(await screen.findByLabelText("Redaction strategy"), "hash");
    await user.click(screen.getByLabelText("email"));
    await user.type(screen.getByLabelText(/Budget ceiling/), "12.5");
    await user.click(screen.getByRole("button", { name: "Save guardrails" }));

    expect(await screen.findByText(/Guardrails saved to guardrails.json/)).toBeTruthy();
    expect(screen.getByText(/Restart the proxy to apply: multimind serve --config/)).toBeTruthy();

    const putCall = fetchMock.mock.calls.find(([, init]) => init?.method === "PUT");
    expect(putCall).toBeTruthy();
    expect(requestBody(putCall![1])).toEqual({
      strategy: "hash",
      block_on: ["ssn", "email"],
      budget_max_cost: 12.5,
      scan_output: true,
    });
  });

  it("shows an error toast when the PUT is rejected", async () => {
    stubFetch((url, init) => {
      if (init?.method === "PUT") {
        return jsonResponse({ detail: "Unknown redaction strategy: 'x'" }, 422);
      }
      if (url.startsWith("/api/guardrails")) return jsonResponse(getResponse);
      throw new Error(`unexpected fetch: ${url}`);
    });
    const user = userEvent.setup();
    render(<GuardrailsPage />);

    await user.click(await screen.findByRole("button", { name: "Save guardrails" }));

    const toast = await screen.findByRole("alert");
    expect(toast.textContent).toContain("Save failed");
    expect(toast.textContent).toContain("Unknown redaction strategy");
  });
});
