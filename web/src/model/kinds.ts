// How each element kind is drawn. Colours and linear-view heights follow the
// earlier EICViBE viewer (machine_portal/element_colors.py and the beamline
// strip): a muted steel-blue magnet family, darkening with multipole order,
// with quadrupoles drawn lower than there so dipoles stand out. Each colour
// has a lighter twin for the dark theme.

export type Lane = "straddle" | "polarity" | "tick" | "none";

export interface KindStyle {
  /** Light-theme colour. */
  color: string;
  /** Dark-theme colour. */
  dark: string;
  /** Full width on the floor plan, m. 0: drawn as a dot, not a band. */
  floorWidth: number;
  /** Block height in the linear view, as a fraction of a dipole's. */
  height: number;
  /** "polarity": above the axis when focusing, below when defocusing. */
  lane: Lane;
  /** Preference when hit testing a crowded area; 0 is never picked. */
  priority: number;
}

const STYLES: Record<string, KindStyle> = {
  Bend: { color: "#5B7FA6", dark: "#86A9D1", floorWidth: 0.4, height: 1.0, lane: "straddle", priority: 3 },
  Quadrupole: { color: "#4A6D8C", dark: "#7397B8", floorWidth: 0.5, height: 0.6, lane: "polarity", priority: 3 },
  Sextupole: { color: "#284153", dark: "#5F8299", floorWidth: 0.3, height: 0.5, lane: "polarity", priority: 3 },
  Octupole: { color: "#192935", dark: "#4E6678", floorWidth: 0.3, height: 0.42, lane: "polarity", priority: 3 },
  Multipole: { color: "#192935", dark: "#4E6678", floorWidth: 0.3, height: 0.42, lane: "polarity", priority: 3 },
  RFCavity: { color: "#7A6E9E", dark: "#A397C9", floorWidth: 0.5, height: 0.8, lane: "straddle", priority: 3 },
  CrabCavity: { color: "#7A6E9E", dark: "#A397C9", floorWidth: 0.5, height: 0.8, lane: "straddle", priority: 3 },
  Kicker: { color: "#9E6060", dark: "#C98989", floorWidth: 0.25, height: 0.5, lane: "straddle", priority: 3 },
  Solenoid: { color: "#4E8A8A", dark: "#76B3B3", floorWidth: 0.4, height: 0.75, lane: "straddle", priority: 3 },
  Instrument: { color: "#6A7D6A", dark: "#93A893", floorWidth: 0, height: 0.35, lane: "tick", priority: 2 },
  BeamBeam: { color: "#5A6473", dark: "#8A95A6", floorWidth: 0, height: 0.5, lane: "tick", priority: 2 },
  Fork: { color: "#C2185B", dark: "#F06292", floorWidth: 0, height: 1.0, lane: "tick", priority: 2 },
  Marker: { color: "#7A7A7A", dark: "#A3A3A3", floorWidth: 0, height: 0.25, lane: "tick", priority: 1 },
  Patch: { color: "#A07848", dark: "#C9A070", floorWidth: 0, height: 0.4, lane: "tick", priority: 1 },
  ReferenceChange: { color: "#A07848", dark: "#C9A070", floorWidth: 0, height: 0.4, lane: "tick", priority: 1 },
  BeginningEle: { color: "#7A7A7A", dark: "#A3A3A3", floorWidth: 0, height: 0.25, lane: "none", priority: 1 },
  Drift: { color: "#9AA0A6", dark: "#5B6270", floorWidth: 0, height: 0, lane: "none", priority: 0 },
};

const UNKNOWN: KindStyle = { color: "#5A6473", dark: "#8A95A6", floorWidth: 0.3, height: 0.5, lane: "straddle", priority: 2 };

export function kindStyle(kind: string): KindStyle {
  return STYLES[kind] ?? UNKNOWN;
}

export function kindColor(kind: string, dark: boolean): string {
  const style = kindStyle(kind);
  return dark ? style.dark : style.color;
}

/** CSS colour of a kind for HTML swatches; follows the colour scheme (see installKindColors). */
export function swatch(kind: string): string {
  return `var(--kind-${kind}, ${kindStyle(kind).color})`;
}

/** Define --kind-<Kind> colours for both colour schemes, once per page. */
export function installKindColors(): void {
  if (document.getElementById("kind-colors")) return;
  const rules = (dark: boolean) =>
    Object.keys(STYLES)
      .map((kind) => `--kind-${kind}: ${kindColor(kind, dark)};`)
      .join(" ");
  const style = document.createElement("style");
  style.id = "kind-colors";
  style.textContent = `:root { ${rules(false)} } @media (prefers-color-scheme: dark) { :root { ${rules(true)} } }`;
  document.head.append(style);
}

/** Drifts are spacing, not elements: they are neither listed, highlighted nor picked. */
export function isDrift(kind: string): boolean {
  return kind === "Drift";
}

/** Kinds in the order legends and filters list them. */
export const KIND_ORDER = Object.keys(STYLES);

/** Colours for branch centre lines, by branch order. */
export const BRANCH_COLORS = ["#5b6b8c", "#7a5c99", "#4f8a6b", "#9a6a3a", "#3f7f9f", "#8c5b6b", "#6b7f3f"];
