import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import CompliancePage from "../CompliancePage";
import { jsonResponse, stubFetch } from "../../test/helpers";

const regulations = ["GDPR", "HIPAA", "CCPA"];
const useCases = ["medical_diagnosis", "telemedicine"];

const metrics = {
  total_checks: 10,
  passed_checks: 8,
  failed_checks: 2,
  compliance_score: 0.8,
  recent_issues: [],
  trend_data: {},
  alerts: [],
};

describe("CompliancePage", () => {
  it("renders regulations and healthcare use cases", async () => {
    stubFetch((url) => {
      if (url === "/v1/compliance/regulations") return jsonResponse(regulations);
      if (url === "/v1/compliance/healthcare/use-cases") return jsonResponse(useCases);
      throw new Error(`unexpected fetch: ${url}`);
    });
    render(<CompliancePage />);

    expect(await screen.findByText("GDPR")).toBeTruthy();
    expect(screen.getByText("medical_diagnosis")).toBeTruthy();
  });

  it("loads an organization's dashboard metrics", async () => {
    stubFetch((url) => {
      if (url === "/v1/compliance/regulations") return jsonResponse(regulations);
      if (url === "/v1/compliance/healthcare/use-cases") return jsonResponse(useCases);
      if (url.startsWith("/v1/compliance/dashboard")) return jsonResponse(metrics);
      if (url.startsWith("/v1/compliance/alerts")) return jsonResponse([]);
      throw new Error(`unexpected fetch: ${url}`);
    });
    const user = userEvent.setup();
    render(<CompliancePage />);

    await screen.findByText("GDPR");
    await user.click(screen.getByRole("button", { name: "Load" }));

    expect(await screen.findByText("80%")).toBeTruthy();
  });
});
