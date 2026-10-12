interface Props {
  dir: "up" | "down";
  active?: boolean;
}

/** Hand-rolled thumbs up/down glyph; colored when active. */
export default function Thumb({ dir, active = false }: Props) {
  const color = active ? (dir === "up" ? "var(--ok)" : "var(--danger)") : "var(--ink-3)";
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 14 14"
      role="img"
      aria-label={dir === "up" ? "thumbs up" : "thumbs down"}
    >
      <g transform={dir === "down" ? "rotate(180 7 7)" : undefined} fill={color}>
        <rect x="1" y="6" width="2.6" height="6" rx="0.6" />
        <path d="M4.6 12V6.6L7 1.6c1 .1 1.6.9 1.4 1.9L8 5.4h3.2c.9 0 1.5.8 1.3 1.7l-.9 3.6c-.2.8-.8 1.3-1.6 1.3z" />
      </g>
    </svg>
  );
}
