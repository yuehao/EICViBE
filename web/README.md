# EICViBE web client

The lattice viewer served by `eicvibe-web` (Python package `eicvibe.api`, see `docs/adr/0004-lattice-viewer.md`).

```sh
npm install
npm run build            # type-check and build into ../src/eicvibe/api/static
uv run eicvibe-web --demo --open                       # a small made-up machine
uv run eicvibe-web /path/to/EIN/machine --open         # a machine folder (ADR 0005); edits show live
uv run eicvibe-web /path/to/EIN/machine --va va-state  # and the virtual accelerator (ADR 0008)
```

Development with hot reload: run `uv run eicvibe-web --demo` (port 8000) and `npm run dev`, then open the Vite URL. API calls are forwarded to port 8000.

| Command | What it does |
| --- | --- |
| `npm test` | Unit tests of the model code (geometry transforms, hit testing, URL state) |
| `npm run e2e` | Browser tests against `eicvibe-web --demo`, in the installed Chrome (`PW_CHANNEL=chromium` for Playwright's own browser) |
| `npm run generate-types` | Regenerate `src/api/openapi.json` and `src/api/schema.d.ts` after changing `eicvibe/api/schemas.py` |

## Using the viewer

- **Floor plan**: drag to pan, scroll or pinch to zoom, double-click or `0` to fit, arrow keys to pan. Hover shows the element; click selects it. Element names appear when you zoom in. A red circle marks a loop of branches that does not close.
- **Lattice strip and optics**: the current branch along s, as a strip of its elements directly above its optics plot, both on one s axis. The strip uses the colours and block heights of the earlier EICViBE viewer (quadrupoles lower than dipoles); focusing magnets are drawn above the axis and defocusing ones below, and a selected element is outlined. Drag in either to zoom to an s range; right click undoes the last zoom (repeatedly); shift-drag pans; the wheel zooms, or pans when scrolled sideways; double-click or "Whole branch" ("Whole route") shows all of it. Click a dashed fork flag to go to that branch.
- **Names**: elements are shown by their EIC names (EIC-SEG-RSI-009, e.g. `bar-q5`) from the machine registry. The lattice (design) name is beside them in the table, tooltips and inspector, and search finds either.
- **Table**: every element except drifts (which are also never highlighted in the plots). Filter by EIC name, lattice name, kind or device; click a header to sort; toggle kinds; "In view" limits the rows to the linear view's range; CSV downloads the rows shown.
- **Optics**: beta functions (left axis) and dispersion (right axis) of the current branch along s, and a ring's tunes, chromaticities and momentum compaction. Toggle curves with the legend; choose the engine (Xsuite, or TrackPad with TPSA maps) and fidelity on the right. Hover for the values at a point. With TrackPad the curves follow the optics inside magnets (sampled at its integrator steps); with Xsuite they join the values at element exits, exact only in drifts. A line starts from its initial conditions or from the fork it starts from, as the summary says. TrackPad's first computation starts Julia and takes about 20 s.
- **Routes**: the buttons after the branches (⇢) show a route, the beam's path through several branches (`routes.yaml` in a machine folder, ADR 0005). Every view treats it as one branch with s along the route: the strip flags the forks along it, the optics are the injected beam's, the floor plan draws its path over its branches, and an element of a branch it passes is selected on the route. ⚠ marks a route that cannot be one pass (its energy changes).
- **Optics and Orbit tabs**: the panel under the floor plan shows the optics, or the BPM orbits of the current ring or route; both share the strip above the plot, its s axis, zoom and undo.
- **Orbit** (when the server reads virtual runs: `eicvibe-web MACHINE --runs DIR`, where `eicvibe-virtual ring` or `eicvibe-virtual route` writes them; `--demo` runs a ring and a route): BPM readings along s, x above and y below, N turns apart for a ring and N shots apart for a route (each shot a single pass), the newest orbit bold and the older ones fading. "every" sets N, "last" how many orbits. While the run goes on the view follows its latest turns; Pause holds them, ◀ and ▶ step N turns through what the run keeps, Live follows again. Hover a BPM for its readings on every turn shown. A ring's data, chosen beside the run, comes as a real BPM system's does: the latest turns (the BPMs' buffer, 4096 turns), the slow orbit (one sample per second), or a capture (the buffer frozen by a kick, an injection, a beam loss, or Acquire on a virtual accelerator's machine). Every user sees the same runs.
- **Check** (a ring's run): its tracking against the model. The latest kick's capture, from the kick on (or the latest turns, or any capture), gives the measured tunes, beside the model's (the engine's optics with the run's settings) and the run's own one-turn map's; the spectra; tune and amplitude over the turns; the phase advance and beta at every BPM against the model; the mean orbit against the model's closed orbit. "tune vs amplitude" tracks single particles in the model: a large kick moves the tune, and this shows by how much it should. "Full page ↗" opens it in `check.html`, with room for the charts.
- **Virtual accelerator** (when the server runs one: `eicvibe-web MACHINE --va STATE`, ADR 0008): choose a machine in the top bar; the inspector shows the selected element's setpoints on it and the Orbit tab its BPM orbits. Viewing needs no login. Type a value and Set to change it: the first time, the viewer asks you to log in, and the change goes into the log with your name and reason. "Sandbox" makes a machine of your own with the chosen one's settings. Once logged in, your name in the top bar opens a menu to log out, make a token for notebooks or change your password. Operators and administrators manage machines, users and the log on a separate page they are told of.
- **Inspector**: parameters with units, devices and PV names (with copy buttons), fork targets, other traversals of a shared component, and floor poses.
- **Links**: the address bar always holds the current view. Copy it to show a colleague the same element.
