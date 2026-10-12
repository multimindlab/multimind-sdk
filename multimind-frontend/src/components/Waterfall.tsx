import { PlatformSpan } from "../api/platform";

interface Props {
  spans: PlatformSpan[];
  selectedId: string | null;
  onSelect: (span: PlatformSpan) => void;
}

interface Row {
  span: PlatformSpan;
  depth: number;
  leftPct: number;
  widthPct: number;
  durationMs: number;
}

const MIN_WIDTH_PCT = 0.5;

function toMs(iso: string | null | undefined): number | null {
  if (!iso) return null;
  const t = new Date(iso).getTime();
  return Number.isNaN(t) ? null : t;
}

/** Depth-first order (children under their parent, sorted by start time). */
export function buildWaterfallRows(spans: PlatformSpan[]): Row[] {
  const byId = new Map(spans.map((s) => [s.id, s]));
  const children = new Map<string | null, PlatformSpan[]>();
  for (const s of spans) {
    const parent = s.parent_id !== null && byId.has(s.parent_id) ? s.parent_id : null;
    const list = children.get(parent) ?? [];
    list.push(s);
    children.set(parent, list);
  }
  for (const list of children.values()) {
    list.sort((a, b) => (toMs(a.start_time) ?? 0) - (toMs(b.start_time) ?? 0));
  }

  const starts = spans.map((s) => toMs(s.start_time)).filter((t): t is number => t !== null);
  const ends = spans
    .map((s) => toMs(s.end_time) ?? toMs(s.start_time))
    .filter((t): t is number => t !== null);
  const min = starts.length ? Math.min(...starts) : 0;
  const max = ends.length ? Math.max(...ends) : min;
  const total = Math.max(max - min, 1);

  const rows: Row[] = [];
  const seen = new Set<string>();
  const walk = (parent: string | null, depth: number) => {
    for (const span of children.get(parent) ?? []) {
      if (seen.has(span.id)) continue;
      seen.add(span.id);
      const start = toMs(span.start_time) ?? min;
      const end = toMs(span.end_time) ?? start;
      const durationMs = Math.max(end - start, 0);
      rows.push({
        span,
        depth,
        leftPct: ((start - min) / total) * 100,
        widthPct: Math.max((durationMs / total) * 100, MIN_WIDTH_PCT),
        durationMs,
      });
      walk(span.id, depth + 1);
    }
  };
  walk(null, 0);
  return rows;
}

function fmtMs(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)}s` : `${Math.round(ms)}ms`;
}

/** Horizontal trace waterfall: bars scaled by start/end, indented by depth. */
export default function Waterfall({ spans, selectedId, onSelect }: Props) {
  const rows = buildWaterfallRows(spans);
  if (rows.length === 0) {
    return <p className="muted">No spans recorded for this run.</p>;
  }

  return (
    <div className="waterfall" role="list" aria-label="Trace spans">
      {rows.map(({ span, depth, leftPct, widthPct, durationMs }) => (
        <div
          key={span.id}
          role="listitem"
          className={`wf-row${selectedId === span.id ? " selected" : ""}`}
          onClick={() => onSelect(span)}
        >
          <button
            type="button"
            className="wf-name"
            style={{ paddingLeft: `${0.4 + depth * 1.1}rem` }}
            onClick={(e) => {
              e.stopPropagation();
              onSelect(span);
            }}
          >
            <span className={`status-dot ${span.status}`} aria-hidden="true" />
            {span.name}
          </button>
          <div className="wf-track" aria-hidden="true">
            <span
              className={`wf-bar ${span.status}`}
              data-testid={`wf-bar-${span.id}`}
              style={{ left: `${leftPct}%`, width: `${widthPct}%` }}
            />
          </div>
          <span className="wf-duration muted">{fmtMs(durationMs)}</span>
        </div>
      ))}
    </div>
  );
}
