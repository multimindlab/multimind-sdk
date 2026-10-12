export interface StackPoint {
  label: string;
  ok: number;
  error: number;
}

interface Props {
  points: StackPoint[];
  ariaLabel?: string;
}

const W = 640;
const H = 220;
const PAD = { top: 14, right: 14, bottom: 26, left: 52 };

/** Vertical stacked bars (ok + error segments), hand-rolled SVG. */
export default function StackedBarChart({ points, ariaLabel }: Props) {
  if (points.length === 0) {
    return <p className="muted">No data yet.</p>;
  }

  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;
  const max = Math.max(...points.map((p) => p.ok + p.error), 0) || 1;
  const step = innerW / points.length;
  const barW = Math.max(step * 0.65, 1);
  const yOf = (v: number) => PAD.top + innerH - (v / max) * innerH;

  return (
    <div className="chart-box">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={ariaLabel ?? "Stacked bar chart"}>
        {[0.5, 1].map((f) => (
          <g key={f}>
            <line
              x1={PAD.left}
              x2={W - PAD.right}
              y1={yOf(max * f)}
              y2={yOf(max * f)}
              stroke="var(--grid)"
            />
            <text
              x={PAD.left - 8}
              y={yOf(max * f) + 4}
              textAnchor="end"
              fontSize="11"
              fill="var(--ink-3)"
            >
              {Math.round(max * f).toLocaleString("en-US")}
            </text>
          </g>
        ))}
        <line
          x1={PAD.left}
          x2={W - PAD.right}
          y1={PAD.top + innerH}
          y2={PAD.top + innerH}
          stroke="var(--border)"
        />
        {points.map((p, i) => {
          const x = PAD.left + i * step + (step - barW) / 2;
          const total = p.ok + p.error;
          const yTop = yOf(total);
          const yErrTop = yOf(p.error);
          return (
            <g key={p.label}>
              <title>{`${p.label}: ${total} runs, ${p.error} errors`}</title>
              {p.ok > 0 ? (
                <rect x={x} y={yTop} width={barW} height={(p.ok / max) * innerH} fill="var(--accent)" />
              ) : null}
              {p.error > 0 ? (
                <rect x={x} y={yErrTop} width={barW} height={(p.error / max) * innerH} fill="var(--danger)" />
              ) : null}
            </g>
          );
        })}
        <text x={PAD.left} y={H - 8} textAnchor="start" fontSize="11" fill="var(--ink-3)">
          {points[0].label}
        </text>
        {points.length > 1 ? (
          <text x={W - PAD.right} y={H - 8} textAnchor="end" fontSize="11" fill="var(--ink-3)">
            {points[points.length - 1].label}
          </text>
        ) : null}
      </svg>
      <div className="chart-legend">
        <span>
          <i className="legend-swatch" style={{ background: "var(--accent)" }} /> runs
        </span>
        <span>
          <i className="legend-swatch" style={{ background: "var(--danger)" }} /> errors
        </span>
      </div>
    </div>
  );
}
