import { useRef, useState, type MouseEvent as ReactMouseEvent } from "react";

export interface LinePoint {
  label: string;
  value: number;
}

interface Props {
  points: LinePoint[];
  formatValue?: (v: number) => string;
  ariaLabel?: string;
}

const W = 640;
const H = 220;
const PAD = { top: 14, right: 14, bottom: 26, left: 52 };

/** Single-series SVG line chart with crosshair + tooltip hover layer. */
export default function LineChart({ points, formatValue = String, ariaLabel }: Props) {
  const boxRef = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<number | null>(null);

  if (points.length === 0) {
    return <p className="muted">No data yet.</p>;
  }

  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;
  const max = Math.max(...points.map((p) => p.value), 0);
  const yMax = max > 0 ? max * 1.1 : 1;

  const x = (i: number) =>
    PAD.left + (points.length === 1 ? innerW / 2 : (i / (points.length - 1)) * innerW);
  const y = (v: number) => PAD.top + innerH - (v / yMax) * innerH;

  const path = points.map((p, i) => `${i === 0 ? "M" : "L"}${x(i)},${y(p.value)}`).join(" ");
  const area = `${path} L${x(points.length - 1)},${PAD.top + innerH} L${x(0)},${PAD.top + innerH} Z`;
  const gridYs = [0.25, 0.5, 0.75, 1].map((f) => PAD.top + innerH - f * innerH);

  const onMove = (e: ReactMouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * W;
    let best = 0;
    let bestDist = Infinity;
    points.forEach((_, i) => {
      const d = Math.abs(x(i) - px);
      if (d < bestDist) {
        bestDist = d;
        best = i;
      }
    });
    setHover(best);
  };

  const hovered = hover === null ? null : points[hover];

  return (
    <div className="chart-box" ref={boxRef}>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={ariaLabel ?? "Line chart"}
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
      >
        {gridYs.map((gy) => (
          <line key={gy} x1={PAD.left} x2={W - PAD.right} y1={gy} y2={gy} stroke="var(--grid)" />
        ))}
        <line
          x1={PAD.left}
          x2={W - PAD.right}
          y1={PAD.top + innerH}
          y2={PAD.top + innerH}
          stroke="var(--border)"
        />
        {[0.5, 1].map((f) => (
          <text
            key={f}
            x={PAD.left - 8}
            y={PAD.top + innerH - f * innerH + 4}
            textAnchor="end"
            fontSize="11"
            fill="var(--ink-3)"
          >
            {formatValue((yMax * f * 10) / 10)}
          </text>
        ))}
        <text x={x(0)} y={H - 8} textAnchor="start" fontSize="11" fill="var(--ink-3)">
          {points[0].label}
        </text>
        {points.length > 1 ? (
          <text
            x={x(points.length - 1)}
            y={H - 8}
            textAnchor="end"
            fontSize="11"
            fill="var(--ink-3)"
          >
            {points[points.length - 1].label}
          </text>
        ) : null}
        <path d={area} fill="var(--accent-soft)" />
        <path d={path} fill="none" stroke="var(--accent)" strokeWidth="2" strokeLinejoin="round" />
        {hover !== null && hovered ? (
          <g>
            <line
              x1={x(hover)}
              x2={x(hover)}
              y1={PAD.top}
              y2={PAD.top + innerH}
              stroke="var(--ink-3)"
              strokeDasharray="3 3"
            />
            <circle
              cx={x(hover)}
              cy={y(hovered.value)}
              r={4.5}
              fill="var(--accent)"
              stroke="var(--surface)"
              strokeWidth="2"
            />
          </g>
        ) : null}
      </svg>
      {hover !== null && hovered ? (
        <div
          className="chart-tip"
          style={{ left: `${(x(hover) / W) * 100}%`, top: `${(y(hovered.value) / H) * 100}%` }}
        >
          <span className="tip-label">{hovered.label}</span>{" "}
          <span className="tip-value">{formatValue(hovered.value)}</span>
        </div>
      ) : null}
    </div>
  );
}
