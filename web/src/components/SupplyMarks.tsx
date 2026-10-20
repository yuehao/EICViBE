// Marks on a power supply's current (ADR 0014): where its magnets' transfer functions come from,
// a placeholder marked as one, since its current is not the real magnet's; and red where the design
// asks more current than the supply's limits give, a design the supply cannot run.

import { formatNumber } from "../model/format";

const SOURCE_TITLES: Record<string, string> = {
  placeholder: "A placeholder transfer function, not this magnet's: the current shown is not the real one",
  design: "From the magnet's design: its coil and aperture, as its designers give them",
  measured: "From measurements of the magnet",
};

/** Where a supply's transfer functions come from: placeholder (marked), design or measured. */
export function SourceBadge({ source }: { source: string | null | undefined }) {
  if (!source) return null;
  return (
    <span className={`badge ${source}`} data-testid="source-badge" title={SOURCE_TITLES[source] ?? "Where the transfer function comes from"}>
      {source}
    </span>
  );
}

/** Whether a value lies outside its limits. */
export function outside(value: number, limits: readonly number[] | null | undefined): boolean {
  if (!limits || limits.length < 2) return false;
  const [low, high] = limits as [number, number];
  const margin = 1e-9 * Math.max(Math.abs(low), Math.abs(high), 1);
  return value < low - margin || value > high + margin;
}

/** "beyond its supply's limits 0 … 150 A", for a design current a supply cannot run. */
export function BeyondLimits({ current, limits, unit }: { current: number; limits: readonly number[]; unit: string }) {
  return (
    <div className="beyond" data-testid="beyond-limits" title="The design asks more current than this supply's limits give: the supply cannot run it">
      design needs {formatNumber(current, 4)} {unit}, beyond its supply's limits {formatNumber(limits[0]!, 4)} … {formatNumber(limits[1]!, 4)} {unit}
    </div>
  );
}
