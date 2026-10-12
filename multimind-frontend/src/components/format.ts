export function fmtUSD(value: number): string {
  if (value === 0) return "$0.00";
  return value >= 1 ? `$${value.toFixed(2)}` : `$${value.toFixed(4)}`;
}

export function fmtNum(value: number): string {
  return value.toLocaleString("en-US");
}

export function fmtTimestamp(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString();
}
