import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import InventoryPage from "../InventoryPage";
import { jsonResponse, stubFetch } from "../../test/helpers";

const findings = [
  {
    kind: "dependency",
    provider: "openai",
    evidence: "openai>=1.0",
    file: "pyproject.toml",
    line: 12,
    data_flow: "external",
  },
  {
    kind: "hardcoded_key",
    provider: "openai",
    evidence: "OPENAI_API_KEY assigned a literal",
    file: "scripts/run.py",
    line: 3,
    data_flow: "external",
  },
  {
    kind: "env_read",
    provider: "anthropic",
    evidence: "os.getenv('ANTHROPIC_API_KEY')",
    file: "app/llm.py",
    line: 7,
    data_flow: "external",
  },
];

function inventory(extra: Record<string, unknown> = {}) {
  return {
    scanned_at: "2026-07-06T10:00:00+00:00",
    cached: true,
    root: "/repo",
    findings,
    summary: {
      findings: findings.length,
      by_provider: { openai: 2, anthropic: 1 },
      by_kind: { dependency: 1, hardcoded_key: 1, env_read: 1 },
      by_data_flow: { external: 3 },
      scanned_files: 120,
      skipped_files: 0,
    },
    risks: {
      external_data_flow_providers: ["anthropic", "openai"],
      hardcoded_keys: [findings[1]],
    },
    ...extra,
  };
}

describe("InventoryPage", () => {
  it("groups findings by provider and shows the hardcoded-key risk banner", async () => {
    stubFetch(() => jsonResponse(inventory()));
    render(<InventoryPage />);

    expect(await screen.findByText("openai")).toBeTruthy();
    expect(screen.getByText("anthropic")).toBeTruthy();
    const banner = screen.getByRole("alert");
    expect(banner.textContent).toContain("1 hardcoded API key");
    expect(banner.textContent).toContain("scripts/run.py:3");
  });

  it("rescans with refresh=true when the button is clicked", async () => {
    const fetchMock = stubFetch(() => jsonResponse(inventory()));
    const user = userEvent.setup();
    render(<InventoryPage />);
    await screen.findByText("openai");

    await user.click(screen.getByRole("button", { name: "Rescan project" }));
    await waitFor(() => {
      expect(
        fetchMock.mock.calls.some(([url]) => String(url) === "/api/inventory?refresh=true"),
      ).toBe(true);
    });
  });

  it("shows the dashboard down state on network failure", async () => {
    stubFetch(() => {
      throw new TypeError("fetch failed");
    });
    render(<InventoryPage />);

    expect(await screen.findByText(/not reachable/)).toBeTruthy();
  });
});
