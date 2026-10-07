// Number formatting for tables, tooltips and axes.

/** Up to ``digits`` significant digits, plain notation for 1e-4 <= |v| < 1e6, exponent otherwise. */
export function formatNumber(value: number | null | undefined, digits = 6): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "";
  if (value === 0) return "0";
  const magnitude = Math.abs(value);
  if (magnitude >= 1e6 || magnitude < 1e-4) return value.toExponential(Math.max(digits - 1, 0)).replace(/\.?0+e/, "e");
  return String(Number(value.toPrecision(digits)));
}

/** Fixed decimals for positions in metres. */
export function formatMetres(value: number, decimals = 4): string {
  return value.toFixed(decimals);
}

/** Axis label for a tick at ``value`` with spacing ``step``. */
export function formatTick(value: number, step: number): string {
  const decimals = Math.max(0, -Math.floor(Math.log10(step) + 1e-9));
  return value.toFixed(decimals);
}

export function formatEnergy(eV: number | null | undefined): string {
  if (eV === null || eV === undefined) return "";
  for (const [scale, unit] of [
    [1e12, "TeV"],
    [1e9, "GeV"],
    [1e6, "MeV"],
    [1e3, "keV"],
  ] as const)
    if (Math.abs(eV) >= scale) return `${formatNumber(eV / scale, 5)} ${unit}`;
  return `${formatNumber(eV, 5)} eV`;
}
