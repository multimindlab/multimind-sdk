import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import EvidencePage from "../EvidencePage";
import { jsonResponse, stubFetch } from "../../test/helpers";

describe("EvidencePage", () => {
  beforeEach(() => {
    Object.assign(URL, {
      createObjectURL: vi.fn(() => "blob:mock"),
      revokeObjectURL: vi.fn(),
    });
  });

  it("downloads the report in the selected format", async () => {
    const clickSpy = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => {});
    const fetchMock = stubFetch((url) => {
      if (url === "/api/evidence-report?format=md") {
        return jsonResponse("# Evidence Report", 200, {
          "Content-Disposition": 'attachment; filename="evidence-report.md"',
        });
      }
      throw new Error(`unexpected fetch: ${url}`);
    });
    const user = userEvent.setup();
    render(<EvidencePage />);

    await user.selectOptions(screen.getByLabelText("Format"), "md");
    await user.click(screen.getByRole("button", { name: "Download report" }));

    await waitFor(() => {
      expect(String(fetchMock.mock.calls[0][0])).toBe("/api/evidence-report?format=md");
      expect(URL.createObjectURL).toHaveBeenCalledTimes(1);
      expect(clickSpy).toHaveBeenCalledTimes(1);
    });
  });

  it("previews the HTML report inline in a sandboxed iframe", async () => {
    stubFetch((url) => {
      if (url === "/api/evidence-report?format=html") {
        return jsonResponse("<h1>Evidence</h1>", 200, {
          "Content-Disposition": 'attachment; filename="evidence-report.html"',
        });
      }
      throw new Error(`unexpected fetch: ${url}`);
    });
    const user = userEvent.setup();
    render(<EvidencePage />);

    await user.click(screen.getByRole("button", { name: "Preview inline" }));

    const frame = (await screen.findByTitle("Evidence report preview")) as HTMLIFrameElement;
    expect(frame.getAttribute("sandbox")).toBe("");
    expect(frame.getAttribute("srcdoc")).toContain("<h1>Evidence</h1>");
  });

  it("shows the dashboard down state when the download fails", async () => {
    stubFetch(() => {
      throw new TypeError("fetch failed");
    });
    const user = userEvent.setup();
    render(<EvidencePage />);

    await user.click(screen.getByRole("button", { name: "Download report" }));
    expect(await screen.findByText(/not reachable/)).toBeTruthy();
    expect(screen.getByText("multimind dashboard")).toBeTruthy();
  });
});
