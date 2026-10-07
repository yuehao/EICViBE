import { useMemo, useRef, useState } from "react";

import { api, type LatticeEntry, type LatticeSummary } from "../api/client";
import { isOpen, type BranchModel } from "../model/branch";
import { formatEnergy, formatNumber } from "../model/format";
import { BRANCH_COLORS, isDrift, swatch } from "../model/kinds";
import type { Target } from "../model/store";

interface Props {
  entries: LatticeEntry[] | null;
  source: string | null;
  digest: string | null;
  summary: LatticeSummary | null;
  branchOrder: string[];
  models: Map<string, BranchModel>;
  current: string | null;
  onSource(source: string): void;
  onRevision(digest: string): void;
  onBranch(branch: string): void;
  onFind(target: Target): void;
}

export function TopBar(props: Props) {
  const { entries, source, digest, summary, branchOrder, models, current } = props;
  const entry = entries?.find((e) => e.source === source);
  const newer = entry && digest && entry.digest !== digest ? entry.digest : null;
  const branches = new Map(summary?.branches.map((b) => [b.id, b]));
  const [notesOpen, setNotesOpen] = useState(false);
  const open = summary?.closures.filter(isOpen) ?? [];
  const notes = open.length + (summary?.diagnostics.length ?? 0);

  return (
    <header className="topbar">
      <div className="brand">
        <strong>EICViBE</strong> <span>lattice viewer</span>
      </div>
      <select
        value={source ?? ""}
        onChange={(e) => props.onSource(e.target.value)}
        aria-label="Lattice"
        disabled={!entries?.length}
      >
        {entries?.map((e) => (
          <option key={e.source} value={e.source}>
            {e.source} ({e.occurrences} elements)
          </option>
        ))}
      </select>
      {digest ? (
        <span className="revision" title={`Revision sha256:${digest}${entry ? `, loaded ${entry.loaded_at}` : ""}`}>
          rev {digest.slice(0, 10)}
        </span>
      ) : null}
      {newer ? (
        <button type="button" className="notice" onClick={() => props.onRevision(newer)}>
          A newer revision is available: switch
        </button>
      ) : null}

      <nav className="branches" aria-label="Branches">
        {branchOrder.map((id, k) => {
          const b = branches.get(id);
          if (!b) return null;
          return (
            <button
              key={id}
              type="button"
              className={id === current ? "active" : ""}
              aria-pressed={id === current}
              onClick={() => props.onBranch(id)}
              title={`${b.kind}, ${formatNumber(b.length, 6)} m, ${b.occurrences} elements${b.reference ? `, ${b.reference.species} ${formatEnergy(b.reference.pc)}` : ""}${models.has(id) ? "" : " (loading)"}`}
            >
              <i style={{ background: BRANCH_COLORS[k % BRANCH_COLORS.length] }} />
              {b.name}
              {b.kind === "periodic" ? " ⟳" : ""}
            </button>
          );
        })}
      </nav>

      <Search models={models} branchOrder={branchOrder} onFind={props.onFind} />

      <div className="spacer" />
      {summary ? (
        <div className="notes-wrap">
          <button
            type="button"
            className={open.length ? "warn" : ""}
            onClick={() => setNotesOpen((o) => !o)}
            aria-expanded={notesOpen}
            title="Floor loops and diagnostics"
          >
            Notes {notes}
          </button>
          {notesOpen ? (
            <div className="popover" role="dialog">
              <h3>Floor loops</h3>
              {open.length === 0 ? (
                <p className="muted">
                  {summary.closures.length
                    ? `Every loop closes (${summary.closures.length} checked, to 1 µm and 1 µrad).`
                    : "No branches form a loop."}
                </p>
              ) : null}
              {open.map((c) => {
                const fork = summary.forks.find((f) => f.id === c.fork);
                return (
                  <p key={c.fork}>
                    <button
                      type="button"
                      className="link"
                      onClick={() => {
                        const model = fork && models.get(fork.from_branch);
                        const index = model?.index.get(c.fork);
                        if (model && index !== undefined) props.onFind({ branch: model.id, index });
                        setNotesOpen(false);
                      }}
                    >
                      {fork?.name ?? c.fork}
                    </button>
                    : the line arrives {formatNumber(c.position_error * 1e3, 4)} mm and {formatNumber(c.angle_error * 1e3, 4)} mrad away
                    from where the other branch places it.
                  </p>
                );
              })}
              <h3>Diagnostics</h3>
              {summary.diagnostics.length === 0 ? <p className="muted">None.</p> : null}
              <ul className="plain">
                {summary.diagnostics.map((d, k) => (
                  <li key={k}>
                    <code>{d.code}</code> {d.message}
                  </li>
                ))}
              </ul>
              <h3>Elements</h3>
              <p className="kinds">
                {Object.entries(summary.kinds).map(([kind, count]) => (
                  <span key={kind}>
                    <i style={{ background: swatch(kind) }} />
                    {kind} {count}
                  </span>
                ))}
              </p>
              {summary.devices ? (
                <p className="muted">
                  {summary.devices} devices, {summary.channels} channels in the registry
                </p>
              ) : (
                <p className="muted">No machine registry: device names are not shown.</p>
              )}
            </div>
          ) : null}
        </div>
      ) : null}
      {digest ? (
        <a className="button" href={api.downloadUrl(digest)} title="This revision as canonical JSON, for Lattice.from_json">
          Download
        </a>
      ) : null}
    </header>
  );
}

function Search({ models, branchOrder, onFind }: { models: Map<string, BranchModel>; branchOrder: string[]; onFind(t: Target): void }) {
  const [text, setText] = useState("");
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const input = useRef<HTMLInputElement | null>(null);
  const results = useMemo(() => {
    const needle = text.trim().toLowerCase();
    if (needle.length < 2) return [];
    const found: (Target & { name: string; branchName: string; kind: string; exact: boolean })[] = [];
    for (const id of branchOrder) {
      const model = models.get(id);
      if (!model) continue;
      for (let i = 0; i < model.n && found.length < 200; i++) {
        if (isDrift(model.kinds[i]!)) continue;
        const name = model.names[i]!;
        const lower = name.toLowerCase();
        const device = model.devices[i]?.toLowerCase() ?? "";
        if (lower.includes(needle) || device.includes(needle))
          found.push({ branch: id, index: i, name, branchName: model.name, kind: model.kinds[i]!, exact: lower === needle });
      }
    }
    return found.sort((a, b) => Number(b.exact) - Number(a.exact)).slice(0, 40);
  }, [text, models, branchOrder]);

  const choose = (k: number) => {
    const r = results[k];
    if (!r) return;
    onFind({ branch: r.branch, index: r.index });
    setOpen(false);
    input.current?.blur();
  };

  return (
    <div className="search">
      <input
        ref={input}
        type="search"
        placeholder="Find element or device…"
        value={text}
        aria-label="Find element"
        onChange={(e) => {
          setText(e.target.value);
          setOpen(true);
          setActive(0);
        }}
        onFocus={() => setOpen(true)}
        onBlur={() => window.setTimeout(() => setOpen(false), 150)}
        onKeyDown={(e) => {
          if (e.key === "ArrowDown") setActive((a) => Math.min(a + 1, results.length - 1));
          else if (e.key === "ArrowUp") setActive((a) => Math.max(a - 1, 0));
          else if (e.key === "Enter") choose(active);
          else if (e.key === "Escape") setOpen(false);
          else return;
          e.preventDefault();
        }}
      />
      {open && results.length ? (
        <ul className="results" role="listbox">
          {results.map((r, k) => (
            <li
              key={`${r.branch}:${r.index}`}
              role="option"
              aria-selected={k === active}
              className={k === active ? "active" : ""}
              onMouseDown={(e) => {
                e.preventDefault();
                choose(k);
              }}
            >
              <i className="swatch" style={{ background: swatch(r.kind) }} />
              {r.name} <span className="muted">{r.branchName}</span>
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}
