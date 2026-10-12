import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import ProjectsPage from "../ProjectsPage";
import { setPlatformKey } from "../../../api/platformAuth";
import { jsonResponse, stubFetch } from "../../../test/helpers";

const projects = {
  projects: [
    {
      name: "checkout-bot",
      run_count: 420,
      last_run_at: "2026-07-06T10:00:00+00:00",
      total_cost: 12.5,
      error_rate: 0.125,
    },
    {
      name: "search-agent",
      run_count: 7,
      last_run_at: null,
      total_cost: 0,
      error_rate: 0,
    },
  ],
};

beforeEach(() => {
  sessionStorage.clear();
  setPlatformKey("k");
});

describe("ProjectsPage", () => {
  it("renders projects with counts, cost, and error rate, linking to traces", async () => {
    stubFetch(() => jsonResponse(projects));
    render(
      <MemoryRouter>
        <ProjectsPage />
      </MemoryRouter>,
    );

    const link = await screen.findByRole("link", { name: "checkout-bot" });
    expect(link.getAttribute("href")).toBe("/traces?project=checkout-bot");
    expect(screen.getByText("420")).toBeTruthy();
    expect(screen.getByText("$12.50")).toBeTruthy();
    expect(screen.getByText("12.5%")).toBeTruthy();
    expect(screen.getByText("search-agent")).toBeTruthy();
  });

  it("shows an empty state when there are no projects", async () => {
    stubFetch(() => jsonResponse({ projects: [] }));
    render(
      <MemoryRouter>
        <ProjectsPage />
      </MemoryRouter>,
    );

    expect(await screen.findByText("No projects yet.")).toBeTruthy();
  });
});
