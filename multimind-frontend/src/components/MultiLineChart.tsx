export interface LineSeries {
  name: string;
  color: string;
  values: number[];
}

interface Props {
  labels: string[];
  series: LineSeries[];
  formatValue?: (v: number) => string;
  ariaLabel?: string;
}

const W = 640;
const H = 220;
const PAD = { top: 14, right: 14, bottom: 26, left: 52 };

/** Multi-series SVG line chart sharing one x-axis and y-scale. */
export default function MultiLineChart({ labels, series, formatValue = String, ariaLabel }: Props) {
  if (labels.length === 0 || series.length === 0) {
    return <p className="muted">No data yet.</p>;
  }

  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;
  const max = Math.max(...series.flatMap((s) => s.values), 0);
  const yMax = max > 0 ? max * 1.1 : 1;

  const x = (i: number) =>
    PAD.left + (labels.length === 1 ? innerW / 2 : (i / (labels.length - 1)) * innerW);
  const y = (v: number) => PAD.top + innerH - (v / yMax) * innerH;

  return (
    <div className="chart-box">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={ariaLabel ?? "Line chart"}>
        {[0.25, 0.5, 0.75, 1].map((f) => (
          <line
            key={f}
            x1={PAD.left}
            x2={W - PAD.right}
            y1={PAD.top + innerH - f * innerH}
            y2={PAD.top + innerH - f * innerH}
            stroke="var(--grid)"
          />
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
            {formatValue(yMax * f)}
          </text>
        ))}
        <text x={x(0)} y={H - 8} textAnchor="start" fontSize="11" fill="var(--ink-3)">
          {labels[0]}
        </text>
        {labels.length > 1 ? (
          <text x={x(labels.length - 1)} y={H - 8} textAnchor="end" fontSize="11" fill="var(--ink-3)">
            {labels[labels.length - 1]}
          </text>
        ) : null}
        {series.map((s) => (
          <path
            key={s.name}
            d={s.values
              .slice(0, labels.length)
              .map((v, i) => `${i === 0 ? "M" : "L"}${x(i)},${y(v)}`)
              .join(" ")}
            fill="none"
            stroke={s.color}
            strokeWidth="2"
            strokeLinejoin="round"
          />
        ))}
      </svg>
      <div className="chart-legend">
        {series.map((s) => (
          <span key={s.name}>
            <i className="legend-swatch" style={{ background: s.color }} /> {s.name}
          </span>
        ))}
      </div>
    </div>
  );
}
