import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import PlatformGate from "../PlatformGate";
import ProjectsPage from "../../pages/platform/ProjectsPage";
import { getPlatformKey, setPlatformKey } from "../../api/platformAuth";
import { jsonResponse, stubFetch } from "../../test/helpers";

beforeEach(() => {
  sessionStorage.clear();
});

describe("PlatformGate", () => {
  it("verifies the key on login, stores it, and shows the gated content", async () => {
    const fetchMock = stubFetch(() => jsonResponse({ projects: [] }));
    const user = userEvent.setup();
    render(
      <PlatformGate>
        <div>gated content</div>
      </PlatformGate>,
    );

    await user.type(screen.getByLabelText("API key"), "my-key");
    await user.click(screen.getByRole("button", { name: "Sign in" }));

    expect(await screen.findByText("gated content")).toBeTruthy();
    expect(getPlatformKey()).toBe("my-key");
    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toBe("/platform/api/v1/projects");
    expect((init?.headers as Record<string, string>)["X-API-Key"]).toBe("my-key");
  });

  it("rejects a wrong key without storing it", async () => {
    stubFetch(() => jsonResponse({ detail: "Invalid API key" }, 401));
    const user = userEvent.setup();
    render(
      <PlatformGate>
        <div>gated content</div>
      </PlatformGate>,
    );

    await user.type(screen.getByLabelText("API key"), "wrong");
    await user.click(screen.getByRole("button", { name: "Sign in" }));

    expect(await screen.findByText("Invalid API key.")).toBeTruthy();
    expect(screen.queryByText("gated content")).toBeNull();
    expect(getPlatformKey()).toBeNull();
  });

  it("explains a 503 when the server has no key configured", async () => {
    stubFetch(() =>
      jsonResponse({ detail: "Platform key not configured. Set MULTIMIND_PLATFORM_KEY." }, 503),
    );
    const user = userEvent.setup();
    render(
      <PlatformGate>
        <div>gated content</div>
      </PlatformGate>,
    );

    await user.type(screen.getByLabelText("API key"), "any");
    await user.click(screen.getByRole("button", { name: "Sign in" }));

    expect(await screen.findByText(/MULTIMIND_PLATFORM_KEY/)).toBeTruthy();
    expect(getPlatformKey()).toBeNull();
  });

  it("drops a stale key on 401 and falls back to the login view", async () => {
    setPlatformKey("stale");
    stubFetch(() => jsonResponse({ detail: "Invalid API key" }, 401));
    render(
      <MemoryRouter>
        <PlatformGate>
          <ProjectsPage />
        </PlatformGate>
      </MemoryRouter>,
    );

    // The gated page's first authed call 401s, which signs the user out.
    expect(await screen.findByLabelText("API key")).toBeTruthy();
    expect(getPlatformKey()).toBeNull();
  });
});
