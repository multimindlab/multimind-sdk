import { useState } from "react";

export interface Bar {
  label: string;
  value: number;
}

interface Props {
  bars: Bar[];
  formatValue?: (v: number) => string;
  ariaLabel?: string;
}

const W = 640;
const ROW_H = 28;
const BAR_H = 14;
const LABEL_W = 120;
const PAD_RIGHT = 56;

/** Single-measure horizontal SVG bar chart with rounded data-ends. */
export default function BarChart({ bars, formatValue = String, ariaLabel }: Props) {
  const [hover, setHover] = useState<number | null>(null);

  if (bars.length === 0) {
    return <p className="muted">No data yet.</p>;
  }

  const height = bars.length * ROW_H + 8;
  const innerW = W - LABEL_W - PAD_RIGHT;
  const max = Math.max(...bars.map((b) => b.value), 0) || 1;

  return (
    <div className="chart-box">
      <svg viewBox={`0 0 ${W} ${height}`} role="img" aria-label={ariaLabel ?? "Bar chart"}>
        <line x1={LABEL_W} x2={LABEL_W} y1={0} y2={height - 4} stroke="var(--border)" />
        {bars.map((bar, i) => {
          const yTop = i * ROW_H + (ROW_H - BAR_H) / 2 + 4;
          const w = Math.max((bar.value / max) * innerW, 2);
          return (
            <g
              key={bar.label}
              onMouseEnter={() => setHover(i)}
              onMouseLeave={() => setHover(null)}
            >
              {/* Oversized hit target for the hover layer */}
              <rect x={0} y={i * ROW_H + 4} width={W} height={ROW_H} fill="transparent" />
              <text
                x={LABEL_W - 8}
                y={yTop + BAR_H - 3}
                textAnchor="end"
                fontSize="12"
                fill="var(--ink-2)"
              >
                {bar.label}
              </text>
              {/* Rounded data-end, square baseline end */}
              <path
                d={`M${LABEL_W},${yTop} h${Math.max(w - 4, 0)} a4,4 0 0 1 4,4 v${BAR_H - 8} a4,4 0 0 1 -4,4 h-${Math.max(w - 4, 0)} Z`}
                fill="var(--accent)"
                opacity={hover === null || hover === i ? 1 : 0.45}
              />
              <text
                x={LABEL_W + w + 6}
                y={yTop + BAR_H - 3}
                fontSize="11"
                fill="var(--ink-3)"
              >
                {formatValue(bar.value)}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
