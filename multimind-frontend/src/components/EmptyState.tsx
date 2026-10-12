interface Props {
  title: string;
  hint?: string;
  command?: string;
}

/** Friendly placeholder when a backend is down or has no data yet. */
export default function EmptyState({ title, hint, command }: Props) {
  return (
    <div className="empty-state" role="status">
      <p>{title}</p>
      {hint ? <p className="muted">{hint}</p> : null}
      {command ? <code>{command}</code> : null}
    </div>
  );
}
