import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import ScanPage from "../ScanPage";
import { jsonResponse, requestBody, stubFetch } from "../../test/helpers";

const scanResult = {
  findings: [{ type: "email", start: 8, end: 25, text: "alice@example.com" }],
  counts: { email: 1 },
  total: 1,
  redacted: "Contact [EMAIL] for details",
};

describe("ScanPage", () => {
  it("posts the text and renders findings plus the redacted output", async () => {
    const fetchMock = stubFetch((url) => {
      if (url === "/api/scan-text") return jsonResponse(scanResult);
      throw new Error(`unexpected fetch: ${url}`);
    });
    const user = userEvent.setup();
    render(<ScanPage />);

    await user.type(screen.getByLabelText("Text to scan"), "Contact alice@example.com for details");
    await user.selectOptions(screen.getByLabelText("Redaction strategy"), "hash");
    await user.click(screen.getByRole("button", { name: "Scan text" }));

    expect(await screen.findByText("alice@example.com")).toBeTruthy();
    expect(screen.getByText("Findings (1)")).toBeTruthy();
    expect(screen.getByText("Contact [EMAIL] for details")).toBeTruthy();

    expect(requestBody(fetchMock.mock.calls[0][1])).toEqual({
      text: "Contact alice@example.com for details",
      strategy: "hash",
    });
  });

  it("surfaces API errors from the scan endpoint", async () => {
    stubFetch(() => jsonResponse({ detail: "scan failed" }, 500));
    const user = userEvent.setup();
    render(<ScanPage />);

    await user.type(screen.getByLabelText("Text to scan"), "hello");
    await user.click(screen.getByRole("button", { name: "Scan text" }));

    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("500");
    expect(alert.textContent).toContain("scan failed");
  });
});
