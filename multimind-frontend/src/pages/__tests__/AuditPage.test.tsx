import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import AuditPage from "../AuditPage";
import { jsonResponse, stubFetch } from "../../test/helpers";

function record(overrides: Record<string, unknown> = {}) {
  return {
    timestamp: "2026-07-06T10:00:00+00:00",
    direction: "input",
    method: "chat",
    pii_types: { email: 2 },
    count: 2,
    strategy: "mask",
    blocked: false,
    tags: ["[EMAIL:ab12cd34]"],
    ...overrides,
  };
}

function auditResponse(records: unknown[]) {
  return {
    total: records.length,
    limit: 50,
    offset: 0,
    pii_types: ["email", "phone", "ssn"],
    records,
  };
}

describe("AuditPage", () => {
  it("renders records and highlights blocked rows", async () => {
    stubFetch(() =>
      jsonResponse(
        auditResponse([record(), record({ blocked: true, pii_types: { ssn: 1 }, count: 1 })]),
      ),
    );
    render(<AuditPage />);

    expect(await screen.findByText("email (2)")).toBeTruthy();
    const blockedPill = screen.getByText("blocked");
    expect(blockedPill.closest("tr")?.className).toContain("row-blocked");
    expect(screen.getByText("email (2)").closest("tr")?.className).not.toContain("row-blocked");
  });

  it("passes type and blocked filters as query parameters", async () => {
    const fetchMock = stubFetch(() => jsonResponse(auditResponse([record()])));
    const user = userEvent.setup();
    render(<AuditPage />);
    await screen.findByText("email (2)");

    await user.selectOptions(screen.getByLabelText("PII type"), "email");
    await waitFor(() => {
      expect(
        fetchMock.mock.calls.some(([url]) => String(url).includes("type=email")),
      ).toBe(true);
    });

    await user.selectOptions(screen.getByLabelText("Blocked"), "yes");
    await waitFor(() => {
      expect(
        fetchMock.mock.calls.some(
          ([url]) => String(url).includes("blocked=true") && String(url).includes("offset=0"),
        ),
      ).toBe(true);
    });
  });

  it("shows the dashboard down state on network failure", async () => {
    stubFetch(() => {
      throw new TypeError("fetch failed");
    });
    render(<AuditPage />);

    expect(await screen.findByText(/not reachable/)).toBeTruthy();
    expect(screen.getByText("multimind dashboard")).toBeTruthy();
  });
});
