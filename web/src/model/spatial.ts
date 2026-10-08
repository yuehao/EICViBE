// A uniform grid over floor path segments, for hit testing under the cursor
// without scanning every element.

const SPAN = 1 << 16;
const HALF = SPAN >> 1;

export class SegmentIndex {
  private constructor(
    private readonly cell: number,
    private readonly cells: Map<number, number[]>,
    private readonly u: Float64Array,
    private readonly v: Float64Array,
    /** End point of the segment starting at each indexed point. */
    private readonly end: Int32Array,
  ) {}

  /**
   * Index a set of paths: path i has points offsets[i] .. offsets[i+1]-1 (as
   * in BranchFloor), and each pair of neighbouring points is a segment. A
   * path of one point is indexed as that point.
   */
  static build(u: Float64Array, v: Float64Array, offsets: Int32Array): SegmentIndex {
    let uMin = Infinity, uMax = -Infinity, vMin = Infinity, vMax = -Infinity;
    for (let j = 0; j < u.length; j++) {
      uMin = Math.min(uMin, u[j]!);
      uMax = Math.max(uMax, u[j]!);
      vMin = Math.min(vMin, v[j]!);
      vMax = Math.max(vMax, v[j]!);
    }
    const cell = Math.max(Math.max(uMax - uMin, vMax - vMin, 1) / 512, 0.25);
    const cells = new Map<number, number[]>();
    const end = new Int32Array(u.length);
    const add = (j: number, k: number) => {
      end[j] = k;
      const i0 = Math.floor(Math.min(u[j]!, u[k]!) / cell), i1 = Math.floor(Math.max(u[j]!, u[k]!) / cell);
      const k0 = Math.floor(Math.min(v[j]!, v[k]!) / cell), k1 = Math.floor(Math.max(v[j]!, v[k]!) / cell);
      for (let a = i0; a <= i1; a++)
        for (let b = k0; b <= k1; b++) {
          const key = (a + HALF) * SPAN + (b + HALF);
          const list = cells.get(key);
          if (list) list.push(j);
          else cells.set(key, [j]);
        }
    };
    for (let i = 0; i + 1 < offsets.length; i++) {
      const first = offsets[i]!, stop = offsets[i + 1]!;
      if (stop - first === 1) add(first, first);
      for (let j = first; j + 1 < stop; j++) add(j, j + 1);
    }
    return new SegmentIndex(cell, cells, u, v, end);
  }

  /**
   * The indexed segment nearest to (pu, pv) within ``radius``: its start point
   * and distance, or null. ``score`` ranks candidates (lowest wins); by
   * default the distance.
   */
  nearest(
    pu: number,
    pv: number,
    radius: number,
    score: (point: number, distance: number) => number = (_, d) => d,
  ): { point: number; distance: number } | null {
    const { cell, u, v, end } = this;
    const i0 = Math.floor((pu - radius) / cell), i1 = Math.floor((pu + radius) / cell);
    const k0 = Math.floor((pv - radius) / cell), k1 = Math.floor((pv + radius) / cell);
    let best = -1, bestScore = Infinity, bestDistance = Infinity;
    const visit = (j: number) => {
      const k = end[j]!;
      const d = segmentDistance(pu, pv, u[j]!, v[j]!, u[k]!, v[k]!);
      if (d > radius) return;
      const s = score(j, d);
      if (s < bestScore) {
        bestScore = s;
        best = j;
        bestDistance = d;
      }
    };
    if ((i1 - i0 + 1) * (k1 - k0 + 1) > this.cells.size) {
      for (const list of this.cells.values()) list.forEach(visit);
    } else {
      for (let a = i0; a <= i1; a++)
        for (let b = k0; b <= k1; b++) this.cells.get((a + HALF) * SPAN + (b + HALF))?.forEach(visit);
    }
    return best < 0 ? null : { point: best, distance: bestDistance };
  }
}

/** Distance from (px, py) to the segment (ax, ay)-(bx, by). */
export function segmentDistance(px: number, py: number, ax: number, ay: number, bx: number, by: number): number {
  const dx = bx - ax, dy = by - ay;
  const length2 = dx * dx + dy * dy;
  let t = length2 > 0 ? ((px - ax) * dx + (py - ay) * dy) / length2 : 0;
  t = Math.max(0, Math.min(1, t));
  return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
}
