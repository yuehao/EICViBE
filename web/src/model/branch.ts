// A branch prepared for drawing and lookup: the API's parallel columns as
// typed arrays, plus what every view derives from them once.

import type { BranchData, BranchSummary, Column, LatticeSummary } from "../api/client";
import { kindStyle } from "./kinds";
import { SegmentIndex } from "./spatial";

export interface BranchModel {
  id: string;
  name: string;
  order: number;
  periodic: boolean;
  length: number;
  n: number;
  ids: string[];
  names: string[];
  kinds: string[];
  elements: string[];
  s: Float64Array;
  lengths: Float64Array;
  columns: Column[];
  values: Record<string, (number | null)[]>;
  devices: (string | null)[];
  forkTo: (string | null)[];
  components: (string | null)[];
  /** +1 above the axis (focusing), -1 below, 0 centred. */
  polarity: Int8Array;
  /** Floor path of occurrence i: points offsets[i] .. offsets[i+1]-1, in (z, x). */
  offsets: Int32Array;
  z: Float64Array;
  x: Float64Array;
  theta: Float64Array;
  /** Occurrence of each floor point. */
  occurrenceOfPoint: Int32Array;
  index: Map<string, number>;
  segments: SegmentIndex;
}

const MAIN_STRENGTH: Record<string, string[]> = {
  Quadrupole: ["Kn1L"],
  Sextupole: ["Kn2L"],
  Octupole: ["Kn3L"],
  Multipole: ["Kn1L", "Kn2L", "Kn3L"],
};

export function branchModel(data: BranchData, summary: BranchSummary, order: number): BranchModel {
  const { table, floor } = data;
  const n = table.id.length;
  const offsets = Int32Array.from(floor.offsets);
  const z = Float64Array.from(floor.z);
  const x = Float64Array.from(floor.x);
  const occurrenceOfPoint = new Int32Array(z.length);
  for (let i = 0; i < n; i++) occurrenceOfPoint.fill(i, offsets[i]!, offsets[i + 1]!);
  const polarity = new Int8Array(n);
  for (let i = 0; i < n; i++) {
    const kind = table.kind[i]!;
    if (kindStyle(kind).lane !== "polarity") continue;
    for (const key of MAIN_STRENGTH[kind] ?? []) {
      const value = table.values[key]?.[i];
      if (value) {
        polarity[i] = value > 0 ? 1 : -1;
        break;
      }
    }
  }
  const index = new Map<string, number>();
  table.id.forEach((id, i) => index.set(id, i));
  return {
    id: summary.id,
    name: summary.name,
    order,
    periodic: summary.kind === "periodic",
    length: summary.length,
    n,
    ids: table.id,
    names: table.name,
    kinds: table.kind,
    elements: table.element,
    s: Float64Array.from(table.s),
    lengths: Float64Array.from(table.length),
    columns: table.columns,
    values: table.values,
    devices: table.device,
    forkTo: table.fork_to,
    components: table.component,
    polarity,
    offsets,
    z,
    x,
    theta: Float64Array.from(floor.theta),
    occurrenceOfPoint,
    index,
    segments: SegmentIndex.build(z, x, offsets),
  };
}

/** Index of the last occurrence starting at or before s (binary search). */
export function occurrenceAt(branch: BranchModel, s: number): number {
  let lo = 0;
  let hi = branch.n - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (branch.s[mid]! <= s) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

/** The value that best describes an occurrence's strength, e.g. "Kn1 = 0.2929 1/m^2". */
export function strengthText(branch: BranchModel, i: number, format: (v: number) => string): string {
  const parts: string[] = [];
  for (const column of branch.columns) {
    const value = branch.values[column.key]?.[i];
    if (value === null || value === undefined) continue;
    if (column.key.endsWith("L") && branch.values[column.key.slice(0, -1)]?.[i] != null) continue;
    parts.push(`${column.label} = ${format(value)}${column.unit ? ` ${column.unit}` : ""}`);
  }
  return parts.join(", ");
}

/** A loop closes when its two ends agree to 1 um and 1 urad; the server reports every loop it checks. */
export function isOpen(closure: { position_error: number; angle_error: number }): boolean {
  return closure.position_error > 1e-6 || closure.angle_error > 1e-6;
}

/** Branch ids in beam order: from the root along forks (by s), then any others. */
export function branchOrder(summary: Pick<LatticeSummary, "root_branch" | "branches" | "forks">): string[] {
  const order: string[] = [];
  const visit = (id: string) => {
    if (order.includes(id)) return;
    order.push(id);
    summary.forks
      .filter((f) => f.from_branch === id)
      .sort((a, b) => a.s_from - b.s_from)
      .forEach((f) => visit(f.to_branch));
  };
  if (summary.root_branch) visit(summary.root_branch);
  summary.branches.forEach((b) => visit(b.id));
  return order;
}
