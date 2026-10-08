// Element list of the current branch, without drifts. Only the rows in view
// are rendered (fixed row height), so a branch of any length scrolls
// smoothly. Filtering, sorting and kind toggles are per browser; nothing here
// talks to the server.

import { memo, useEffect, useMemo, useRef, useState, type CSSProperties } from "react";

import type { BranchModel } from "../model/branch";
import { formatNumber } from "../model/format";
import { isDrift, KIND_ORDER, swatch } from "../model/kinds";
import { occurrenceAt } from "../model/branch";
import { createStore, useStore, useStoreSelect, type Store, type Target } from "../model/store";
import type { SRange } from "../model/viewport";

const ROW = 24;
const OVERSCAN = 8;
const HIDDEN_KEY = "eicvibe.table.hiddenKinds";

type SortKey = "index" | "name" | "lattice" | "kind" | "s" | "length" | "device" | string;

interface Props {
  branch: BranchModel | null;
  latticeName: string;
  selection: Target | null;
  hover: Store<Target | null>;
  range: Store<SRange | null>;
  query: string;
  onQuery(query: string): void;
  onSelect(target: Target): void;
}

const NO_RANGE = createStore<SRange | null>(null);
const haystacks = new WeakMap<BranchModel, string[]>();

function haystack(branch: BranchModel): string[] {
  let found = haystacks.get(branch);
  if (!found) {
    found = branch.names.map((name, i) =>
      [branch.eicNames[i] ?? "", name, branch.elements[i], branch.kinds[i], branch.devices[i] ?? "", branch.components[i] ?? ""]
        .join(" ")
        .toLowerCase(),
    );
    haystacks.set(branch, found);
  }
  return found;
}

function loadHidden(): Set<string> {
  try {
    const saved = window.localStorage.getItem(HIDDEN_KEY);
    if (saved) return new Set(JSON.parse(saved) as string[]);
  } catch {
    // storage unavailable
  }
  return new Set();
}

export function ElementTable(props: Props) {
  const { branch, latticeName, selection, hover, range, query, onQuery, onSelect } = props;
  const [hidden, setHidden] = useState<Set<string>>(loadHidden);
  const [sort, setSort] = useState<{ key: SortKey; descending: boolean }>({ key: "index", descending: false });
  const [follow, setFollow] = useState(false);
  const [scrollTop, setScrollTop] = useState(0);
  const [viewport, setViewport] = useState(400);
  const scroller = useRef<HTMLDivElement | null>(null);
  // Subscribed to the linear view's range only while following it, so panning there costs nothing here.
  const followed = useStore(follow ? range : NO_RANGE);

  useEffect(() => {
    try {
      window.localStorage.setItem(HIDDEN_KEY, JSON.stringify([...hidden]));
    } catch {
      // storage unavailable
    }
  }, [hidden]);

  const listed = useMemo(() => branch?.kinds.reduce((n, kind) => n + Number(!isDrift(kind)), 0) ?? 0, [branch]);
  const counts = useMemo(() => {
    const found = new Map<string, number>();
    branch?.kinds.forEach((kind) => isDrift(kind) || found.set(kind, (found.get(kind) ?? 0) + 1));
    return [...found.entries()].sort(([a], [b]) => KIND_ORDER.indexOf(a) - KIND_ORDER.indexOf(b));
  }, [branch]);

  const rows = useMemo(() => {
    if (!branch) return new Int32Array(0);
    const terms = query.toLowerCase().split(/\s+/).filter(Boolean);
    const text = terms.length ? haystack(branch) : null;
    let first = 0, last = branch.n - 1;
    if (followed) {
      first = occurrenceAt(branch, followed.s0);
      last = occurrenceAt(branch, followed.s1);
    }
    const out: number[] = [];
    for (let i = first; i <= last; i++) {
      const kind = branch.kinds[i]!;
      if (isDrift(kind) || hidden.has(kind)) continue;
      if (text && !terms.every((term) => text[i]!.includes(term))) continue;
      out.push(i);
    }
    if (sort.key !== "index") {
      const value = sortValue(branch, sort.key);
      out.sort((a, b) => {
        const va = value(a), vb = value(b);
        if (va === null) return vb === null ? a - b : 1;
        if (vb === null) return -1;
        const order = va < vb ? -1 : va > vb ? 1 : a - b;
        return sort.descending ? -order : order;
      });
    } else if (sort.descending) out.reverse();
    return Int32Array.from(out);
  }, [branch, query, hidden, sort, followed]);

  useEffect(() => {
    const element = scroller.current;
    if (!element) return;
    const observer = new ResizeObserver(() => setViewport(element.clientHeight));
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  // Bring the selection into view when it is chosen elsewhere.
  useEffect(() => {
    const element = scroller.current;
    if (!element || !selection || !branch || selection.branch !== branch.id) return;
    const position = rows.indexOf(selection.index);
    if (position < 0) return;
    const top = position * ROW, bottom = top + ROW + ROW; // header row
    if (top < element.scrollTop || bottom > element.scrollTop + element.clientHeight)
      element.scrollTop = Math.max(0, top - element.clientHeight / 3);
  }, [selection, branch, rows]);

  if (!branch) return <div className="table-empty">No branch loaded</div>;

  const columns = branch.columns;
  const template = `52px minmax(110px, 1fr) minmax(120px, 1.2fr) 112px 92px 76px ${columns.map(() => "92px").join(" ")} minmax(120px, 1fr)`;
  const first = Math.max(0, Math.floor(scrollTop / ROW) - OVERSCAN);
  const last = Math.min(rows.length, Math.ceil((scrollTop + viewport) / ROW) + OVERSCAN);
  const header = (key: SortKey, label: string, title?: string, numeric = false) => (
    <button
      type="button"
      className={`th${numeric ? " num" : ""}${sort.key === key ? " sorted" : ""}`}
      title={title ? `${title} (click to sort)` : "Click to sort"}
      onClick={() =>
        setSort((s) => (s.key !== key ? { key, descending: false } : s.descending ? { key: "index", descending: false } : { key, descending: true }))
      }
    >
      {label}
      {sort.key === key ? (sort.descending ? " ▾" : " ▴") : ""}
    </button>
  );

  return (
    <div className="table">
      <div className="table-tools">
        <input
          type="search"
          placeholder="Filter: name, kind, device…"
          value={query}
          onChange={(e) => onQuery(e.target.value)}
          aria-label="Filter elements"
        />
        <label title="Show only the elements in the linear view">
          <input type="checkbox" checked={follow} onChange={(e) => setFollow(e.target.checked)} /> In view
        </label>
        <span className="count">
          {rows.length} of {listed}
        </span>
        <button type="button" onClick={() => exportCsv(branch, rows, latticeName)} title="Download the rows shown as CSV">
          CSV
        </button>
      </div>
      <div className="chips">
        {counts.map(([kind, count]) => (
          <button
            key={kind}
            type="button"
            className={`chip${hidden.has(kind) ? " off" : ""}`}
            aria-pressed={!hidden.has(kind)}
            onClick={() =>
              setHidden((h) => {
                const next = new Set(h);
                if (next.has(kind)) next.delete(kind);
                else next.add(kind);
                return next;
              })
            }
          >
            <i style={{ background: swatch(kind) }} />
            {kind} <span>{count}</span>
          </button>
        ))}
      </div>
      <div className="table-scroll" ref={scroller} onScroll={(e) => setScrollTop(e.currentTarget.scrollTop)} data-testid="element-table">
        <div className="tr head" style={{ gridTemplateColumns: template }}>
          {header("index", "#", "Position in the branch", true)}
          {header("name", "name", "EIC name (EIC-SEG-RSI-009); the lattice name where there is none")}
          {header("lattice", "lattice name", "Name in the lattice (design) files")}
          {header("kind", "kind")}
          {header("s", "s [m]", "s at the entrance", true)}
          {header("length", "L [m]", "Length", true)}
          {columns.map((c) => (
            <span key={c.key}>{header(c.key, `${c.label}${c.unit ? ` [${c.unit}]` : ""}`, c.description, true)}</span>
          ))}
          {header("device", "device", "Devices from the machine registry")}
        </div>
        <div style={{ height: rows.length * ROW, position: "relative" }}>
          {Array.from({ length: last - first }, (_, k) => {
            const i = rows[first + k]!;
            return (
              <Row
                key={i}
                branch={branch}
                i={i}
                top={(first + k) * ROW}
                template={template}
                selected={selection?.branch === branch.id && selection.index === i}
                hover={hover}
                onSelect={onSelect}
              />
            );
          })}
        </div>
      </div>
    </div>
  );
}

interface RowProps {
  branch: BranchModel;
  i: number;
  top: number;
  template: string;
  selected: boolean;
  hover: Store<Target | null>;
  onSelect(target: Target): void;
}

const Row = memo(function Row({ branch, i, top, template, selected, hover, onSelect }: RowProps) {
  const target = { branch: branch.id, index: i };
  const hovered = useStoreSelect(hover, (h) => h !== null && h.branch === branch.id && h.index === i);
  const style: CSSProperties = { gridTemplateColumns: template, transform: `translateY(${top}px)` };
  return (
    <div
      className={`tr${selected ? " selected" : ""}${hovered ? " hovered" : ""}`}
      style={style}
      onClick={() => onSelect(target)}
      onMouseEnter={() => hover.set(target)}
      onMouseLeave={() => hover.set(null)}
      role="row"
      aria-selected={selected}
    >
      <span className="num muted">{i}</span>
      <span className="name">{branch.labels[i]}</span>
      <span className="muted" title={branch.elements[i] !== branch.names[i] ? `element ${branch.elements[i]}` : undefined}>
        {branch.names[i]}
      </span>
      <span>
        <i className="swatch" style={{ background: swatch(branch.kinds[i]!) }} />
        {branch.kinds[i]}
      </span>
      <span className="num">{branch.s[i]!.toFixed(4)}</span>
      <span className="num">{formatNumber(branch.lengths[i], 6)}</span>
      {branch.columns.map((c) => (
        <span key={c.key} className="num">
          {formatNumber(branch.values[c.key]?.[i], 6)}
        </span>
      ))}
      <span className="device" title={branch.devices[i] ?? undefined}>
        {branch.devices[i] ?? (branch.components[i] ? `shared: ${branch.components[i]}` : "")}
      </span>
    </div>
  );
});

function sortValue(branch: BranchModel, key: SortKey): (i: number) => number | string | null {
  switch (key) {
    case "name":
      return (i) => branch.labels[i]!;
    case "lattice":
      return (i) => branch.names[i]!;
    case "kind":
      return (i) => branch.kinds[i]!;
    case "s":
      return (i) => branch.s[i]!;
    case "length":
      return (i) => branch.lengths[i]!;
    case "device":
      return (i) => branch.devices[i] ?? null;
    default:
      return (i) => branch.values[key]?.[i] ?? null;
  }
}

function csvField(value: string | number | null | undefined): string {
  if (value === null || value === undefined) return "";
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function exportCsv(branch: BranchModel, rows: Int32Array, latticeName: string) {
  const head = ["index", "eic name", "lattice name", "kind", "element", "s [m]", "L [m]", ...branch.columns.map((c) => `${c.key}${c.unit ? ` [${c.unit}]` : ""}`), "device", "id"];
  const lines = [head.map(csvField).join(",")];
  for (const i of rows)
    lines.push(
      [i, branch.eicNames[i], branch.names[i], branch.kinds[i], branch.elements[i], branch.s[i], branch.lengths[i], ...branch.columns.map((c) => branch.values[c.key]?.[i]), branch.devices[i], branch.ids[i]]
        .map(csvField)
        .join(","),
    );
  const url = URL.createObjectURL(new Blob([lines.join("\n") + "\n"], { type: "text/csv" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = `${latticeName}_${branch.name}.csv`;
  link.click();
  URL.revokeObjectURL(url);
}
