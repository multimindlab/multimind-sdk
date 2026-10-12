import { NavLink, Route, Routes } from "react-router-dom";
import { clearPlatformKey, usePlatformKey } from "./api/platformAuth";
import BackendStatus from "./components/BackendStatus";
import PlatformGate from "./components/PlatformGate";
import AuditPage from "./pages/AuditPage";
import ChatPage from "./pages/ChatPage";
import CompliancePage from "./pages/CompliancePage";
import CostsPage from "./pages/CostsPage";
import EvidencePage from "./pages/EvidencePage";
import GuardrailsPage from "./pages/GuardrailsPage";
import InventoryPage from "./pages/InventoryPage";
import ModelsPage from "./pages/ModelsPage";
import OverviewPage from "./pages/OverviewPage";
import RagPage from "./pages/RagPage";
import ScanPage from "./pages/ScanPage";
import SessionsPage from "./pages/SessionsPage";
import MonitoringPage from "./pages/platform/MonitoringPage";
import ProjectsPage from "./pages/platform/ProjectsPage";
import TraceDetailPage from "./pages/platform/TraceDetailPage";
import TracesPage from "./pages/platform/TracesPage";

const GOVERNANCE_LINKS = [
  { to: "/", label: "Overview" },
  { to: "/audit", label: "Audit Trail" },
  { to: "/costs", label: "Costs" },
  { to: "/inventory", label: "Inventory" },
  { to: "/guardrails", label: "Guardrails" },
  { to: "/scan", label: "Scan" },
  { to: "/evidence", label: "Evidence" },
  { to: "/compliance", label: "Compliance Center" },
];

const GATEWAY_LINKS = [
  { to: "/chat", label: "Chat" },
  { to: "/models", label: "Models" },
  { to: "/sessions", label: "Sessions" },
  { to: "/rag", label: "RAG Workbench" },
];

const PLATFORM_LINKS = [
  { to: "/projects", label: "Projects" },
  { to: "/traces", label: "Traces" },
  { to: "/monitoring", label: "Monitoring" },
];

export default function App() {
  const platformKey = usePlatformKey();

  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true" />
          <span>
            MultiMind
            <small>Governance Console</small>
          </span>
        </div>
        <nav aria-label="Main">
          <span className="nav-group">Governance</span>
          {GOVERNANCE_LINKS.map((link) => (
            <NavLink key={link.to} to={link.to} end={link.to === "/"}>
              {link.label}
            </NavLink>
          ))}
          <span className="nav-group">Gateway</span>
          {GATEWAY_LINKS.map((link) => (
            <NavLink key={link.to} to={link.to}>
              {link.label}
            </NavLink>
          ))}
          <span className="nav-group">Platform</span>
          {PLATFORM_LINKS.map((link) => (
            <NavLink key={link.to} to={link.to}>
              {link.label}
            </NavLink>
          ))}
          {platformKey ? (
            <button type="button" className="signout" onClick={clearPlatformKey}>
              Sign out
            </button>
          ) : null}
        </nav>
        <BackendStatus />
      </aside>
      <main className="main">
        <Routes>
          <Route path="/" element={<OverviewPage />} />
          <Route path="/audit" element={<AuditPage />} />
          <Route path="/costs" element={<CostsPage />} />
          <Route path="/inventory" element={<InventoryPage />} />
          <Route path="/guardrails" element={<GuardrailsPage />} />
          <Route path="/scan" element={<ScanPage />} />
          <Route path="/evidence" element={<EvidencePage />} />
          <Route path="/compliance" element={<CompliancePage />} />
          <Route path="/chat" element={<ChatPage />} />
          <Route path="/models" element={<ModelsPage />} />
          <Route path="/sessions" element={<SessionsPage />} />
          <Route path="/rag" element={<RagPage />} />
          <Route
            path="/projects"
            element={
              <PlatformGate>
                <ProjectsPage />
              </PlatformGate>
            }
          />
          <Route
            path="/traces"
            element={
              <PlatformGate>
                <TracesPage />
              </PlatformGate>
            }
          />
          <Route
            path="/traces/:id"
            element={
              <PlatformGate>
                <TraceDetailPage />
              </PlatformGate>
            }
          />
          <Route
            path="/monitoring"
            element={
              <PlatformGate>
                <MonitoringPage />
              </PlatformGate>
            }
          />
        </Routes>
      </main>
    </div>
  );
}
