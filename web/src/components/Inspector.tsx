// Everything about the selected occurrence: parameters with units, floor
// pose, where a fork leads, other traversals of a shared component, and the
// devices and channels (PV names) that drive or read it.

import { useEffect, useState } from "react";

import { api, type OccurrenceDetail, type Pose } from "../api/client";
import { formatNumber } from "../model/format";
import { swatch } from "../model/kinds";
import type { MachineEntry } from "../api/va";
import type { Auth } from "./Auth";
import { Setpoints } from "./Setpoints";

interface Props {
  digest: string;
  occurrence: string | null;
  branchName(id: string): string;
  /** Name and kind of an occurrence in a loaded branch. */
  describe(id: string): { name: string; kind: string; branch: string } | null;
  onGo(occurrence: string): void;
  onGoBranch(branch: string): void;
  /** The virtual accelerator's machine the viewer works with, if any, and who is logged in. */
  machine?: MachineEntry | null;
  auth?: Auth;
}

export function Inspector({ digest, occurrence, branchName, describe, onGo, onGoBranch, machine, auth }: Props) {
  const [detail, setDetail] = useState<OccurrenceDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!occurrence) {
      setDetail(null);
      setError(null);
      return;
    }
    let live = true;
    setError(null);
    api.occurrence(digest, occurrence).then(
      (d) => live && setDetail(d),
      (e: Error) => live && setError(e.message),
    );
    return () => {
      live = false;
    };
  }, [digest, occurrence]);

  if (!occurrence) return <div className="inspector empty">Select an element in any view to inspect it.</div>;
  if (error) return <div className="inspector empty">Cannot load {occurrence}: {error}</div>;
  if (!detail || detail.id !== occurrence) return <div className="inspector empty">Loading…</div>;

  return (
    <div className="inspector" data-testid="inspector">
      <header>
        <h2>
          <i className="swatch" style={{ background: swatch(detail.kind) }} />
          {detail.eic_name ?? detail.name}
        </h2>
        <div className="muted">
          {detail.eic_name ? <>lattice name <code>{detail.name}</code> · </> : null}
          {detail.kind}
          {detail.element !== detail.name ? <> · element <code>{detail.element}</code></> : null} · branch{" "}
          <button type="button" className="link" onClick={() => onGoBranch(detail.branch)}>
            {branchName(detail.branch)}
          </button>{" "}
          #{detail.index}
        </div>
      </header>

      <section>
        <table className="kv">
          <tbody>
            <tr>
              <th>s</th>
              <td>
                {detail.s_enter.toFixed(6)} → {detail.s_exit.toFixed(6)} m
              </td>
            </tr>
            <tr>
              <th>length</th>
              <td>{formatNumber(detail.length, 9)} m</td>
            </tr>
            {detail.tags.length ? (
              <tr>
                <th>tags</th>
                <td>{detail.tags.join(", ")}</td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </section>

      {detail.fork ? (
        <section>
          <h3>Fork</h3>
          <p>
            Leads to{" "}
            <button type="button" className="link" onClick={() => onGoBranch(detail.fork!.to_branch)}>
              {branchName(detail.fork.to_branch)}
            </button>
            {detail.fork.to_occurrence ? (
              <>
                {" "}at the {detail.fork.to_side} of{" "}
                <button type="button" className="link" onClick={() => onGo(detail.fork!.to_occurrence!)}>
                  {describe(detail.fork.to_occurrence)?.name ?? detail.fork.to_occurrence}
                </button>
              </>
            ) : (
              " at its start"
            )}
          </p>
        </section>
      ) : null}

      {detail.component ? (
        <section>
          <h3>Shared component {detail.component.name}</h3>
          <p className="muted">
            One magnet passed by several beam lines. Each line models it for its own beam, so the kinds can differ:
            a septum is a bend for the beam it injects or extracts, and a pulsed-off kicker for the stored beam. The
            floor plan joins these lines at the magnet's pole faces.
          </p>
          <ul className="plain">
            {detail.component.traversals.map((t) => {
              const other = t.occurrence === detail.id ? { name: detail.name, kind: detail.kind } : describe(t.occurrence);
              return (
                <li key={t.occurrence}>
                  {t.role}:{" "}
                  {t.occurrence === detail.id ? (
                    <strong>{detail.name}</strong>
                  ) : (
                    <button type="button" className="link" onClick={() => onGo(t.occurrence)}>
                      {other?.name ?? t.occurrence}
                    </button>
                  )}{" "}
                  {other ? (
                    <>
                      <i className="swatch" style={{ background: swatch(other.kind) }} />
                      {other.kind}{" "}
                    </>
                  ) : null}
                  <span className="muted">in {branchName(t.branch)}</span>
                </li>
              );
            })}
          </ul>
        </section>
      ) : null}

      {detail.groups.map((group) => (
        <section key={group.group}>
          <h3>{group.group}</h3>
          {group.parameters.length === 0 ? (
            <p className="muted">all defaults</p>
          ) : (
            <table className="kv">
              <tbody>
                {group.parameters.map((p) => (
                  <tr key={p.name} title={p.description ?? undefined}>
                    <th>
                      {p.name}
                      {p.overridden ? <span className="badge" title="Set on this occurrence">occ</span> : null}
                    </th>
                    <td>
                      {renderValue(p.value)} {p.unit ? <span className="muted">{p.unit}</span> : null}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>
      ))}

      {machine && auth ? <Setpoints occurrence={detail.id} machine={machine} auth={auth} /> : null}

      {detail.devices.length ? (
        <section>
          <h3>Devices</h3>
          {detail.devices.map((device) => (
            <div key={device.id} className="device-card">
              <div>
                <strong>{device.name}</strong> <span className="muted">{device.type.replaceAll("_", " ")}</span>
              </div>
              <table className="kv channels">
                <tbody>
                  {device.channels.map((c) => (
                    <tr key={c.name}>
                      <th>
                        <code>{c.name}</code>
                        <button type="button" className="copy" title="Copy the PV name" onClick={() => copy(c.name)}>
                          ⧉
                        </button>
                      </th>
                      <td>
                        {c.role}
                        {c.unit ? ` · ${c.unit}` : ""}
                        {c.control_limits ? ` · ${formatNumber(c.control_limits[0], 4)} … ${formatNumber(c.control_limits[1], 4)}` : ""}
                        {c.parameter ? <div className="muted">{drives(c, detail, describe)}</div> : null}
                        {c.observable ? <div className="muted">reads {c.observable.replaceAll("_", " ")}</div> : null}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {device.power.map((power) => (
                <PowerCircuit key={power.circuit} power={power} detail={detail} describe={describe} />
              ))}
            </div>
          ))}
        </section>
      ) : null}

      <section>
        <h3>Floor</h3>
        <table className="kv poses">
          <thead>
            <tr>
              <th />
              <th>x [m]</th>
              <th>y [m]</th>
              <th>z [m]</th>
              <th>θ</th>
              <th>φ</th>
              <th>ψ [rad]</th>
            </tr>
          </thead>
          <tbody>
            {(["entrance", "exit"] as const).map((end) => (
              <PoseRow key={end} label={end} pose={detail[end]} />
            ))}
          </tbody>
        </table>
      </section>
    </div>
  );
}

function PoseRow({ label, pose }: { label: string; pose: Pose }) {
  return (
    <tr>
      <th>{label}</th>
      {[pose.x, pose.y, pose.z].map((v, k) => (
        <td key={k}>{v.toFixed(6)}</td>
      ))}
      {[pose.theta, pose.phi, pose.psi].map((v, k) => (
        <td key={k + 3}>{formatNumber(v, 7)}</td>
      ))}
    </tr>
  );
}

type Power = OccurrenceDetail["devices"][number]["power"][number];

/** A supply's circuit as it concerns the magnet shown: string or alone, what it sets, transfer function. */
function PowerCircuit({ power, detail, describe }: { power: Power; detail: OccurrenceDetail; describe: Props["describe"] }) {
  const names = (ids: string[]) => {
    const shown = ids.slice(0, 6).map((id) => describe(id)?.name ?? id);
    return shown.join(", ") + (ids.length > 6 ? ` and ${ids.length - 6} more` : "");
  };
  return (
    <div className="power">
      <div>
        {power.series ? (
          <>
            <strong>string {power.name}</strong> <span className="muted">· {power.magnets} magnets in series</span>
          </>
        ) : (
          <strong>powered on its own</strong>
        )}
      </div>
      {power.parameter ? (
        <>
          <div>
            sets <code>{power.parameter}</code>
            {power.polarity === -1 ? <span className="muted"> · wired reversed</span> : null}
          </div>
          {power.transfer ? (
            <div>
              {power.transfer.summary}{" "}
              <span className={`badge ${power.transfer.source}`} title="Where the transfer function comes from">
                {power.transfer.source}
              </span>
            </div>
          ) : null}
          {power.design_current != null ? (
            <div className="muted">
              design: {formatNumber(power.design_current, 6)} {power.transfer?.engineering_unit ?? ""}
            </div>
          ) : null}
        </>
      ) : (
        <div className="muted">sets {names(power.occurrences.filter((o) => o !== detail.id))}, not this traversal</div>
      )}
    </div>
  );
}

/** What a setting drives, said from the point of view of the occurrence shown. */
function drives(
  channel: { parameter?: string | null; targets: string[] },
  detail: OccurrenceDetail,
  describe: Props["describe"],
): string {
  const here = channel.targets.some((t) => t === detail.id || t === detail.element_id);
  const others = channel.targets
    .filter((t) => t !== detail.id && t !== detail.element_id)
    .map((t) => {
      const found = describe(t);
      return found ? `${found.name} (${found.branch})` : t;
    });
  if (!others.length) return `drives ${channel.parameter}`;
  if (here) return `drives ${channel.parameter} here and of ${others.join(", ")}`;
  return `drives ${channel.parameter} of ${others.join(", ")}, not of this traversal`;
}

function renderValue(value: unknown): string {
  if (typeof value === "number") return formatNumber(value, 10);
  if (typeof value === "string" || typeof value === "boolean") return String(value);
  return JSON.stringify(value);
}

function copy(text: string) {
  navigator.clipboard?.writeText(text).catch(() => undefined);
}
