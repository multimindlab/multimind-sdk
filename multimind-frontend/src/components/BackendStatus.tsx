import { useEffect, useState } from "react";
import { getDashboardHealth, getGatewayHealth } from "../api/client";
import { getPlatformHealth } from "../api/platform";

const POLL_MS = 15000;

type Status = "unknown" | "up" | "down";

function Row({ name, status }: { name: string; status: Status }) {
  const text = status === "up" ? "online" : status === "down" ? "offline" : "checking";
  return (
    <div className="status-row">
      <span className={`status-dot ${status === "unknown" ? "" : status}`} aria-hidden="true" />
      <span>
        {name}: {text}
      </span>
    </div>
  );
}

export default function BackendStatus() {
  const [dashboard, setDashboard] = useState<Status>("unknown");
  const [gateway, setGateway] = useState<Status>("unknown");
  const [platform, setPlatform] = useState<Status>("unknown");

  useEffect(() => {
    let cancelled = false;
    async function poll() {
      const [db, gw, pf] = await Promise.allSettled([
        getDashboardHealth(),
        getGatewayHealth(),
        getPlatformHealth(),
      ]);
      if (cancelled) return;
      setDashboard(db.status === "fulfilled" ? "up" : "down");
      setGateway(gw.status === "fulfilled" ? "up" : "down");
      setPlatform(pf.status === "fulfilled" ? "up" : "down");
    }
    poll();
    const timer = setInterval(poll, POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  return (
    <div className="backend-status" aria-label="Backend status">
      <Row name="Dashboard" status={dashboard} />
      <Row name="Gateway" status={gateway} />
      <Row name="Platform" status={platform} />
    </div>
  );
}
