# EICViBE web client

The lattice viewer served by `eicvibe-web` (Python package `eicvibe.api`, see `docs/adr/0004-lattice-viewer.md`).

```sh
npm install
npm run build            # type-check and build into ../src/eicvibe/api/static
uv run eicvibe-web --demo --open                       # a small made-up machine
uv run eicvibe-web /path/to/EIN/machine --open         # a machine folder (ADR 0005); edits show live
```

Development with hot reload: run `uv run eicvibe-web --demo` (port 8000) and `npm run dev`, then open the Vite URL. API calls are forwarded to port 8000.

| Command | What it does |
| --- | --- |
| `npm test` | Unit tests of the model code (geometry transforms, hit testing, URL state) |
| `npm run e2e` | Browser tests against `eicvibe-web --demo`, in the installed Chrome (`PW_CHANNEL=chromium` for Playwright's own browser) |
| `npm run generate-types` | Regenerate `src/api/openapi.json` and `src/api/schema.d.ts` after changing `eicvibe/api/schemas.py` |

## Using the viewer

- **Floor plan**: drag to pan, scroll or pinch to zoom, double-click or `0` to fit, arrow keys to pan. Hover shows the element; click selects it. Element names appear when you zoom in. A red circle marks a loop of branches that does not close.
- **Linear view**: the current branch along s, in the colours and block heights of the earlier EICViBE viewer (quadrupoles lower than dipoles). Focusing magnets are drawn above the axis and defocusing ones below; a selected element is outlined. Scroll to zoom, drag or scroll sideways to pan. Click a dashed fork flag to go to that branch.
- **Names**: elements are shown by their EIC names (EIC-SEG-RSI-009, e.g. `bar-q5`) from the machine registry. The lattice (design) name is beside them in the table, tooltips and inspector, and search finds either.
- **Table**: every element except drifts (which are also never highlighted in the plots). Filter by EIC name, lattice name, kind or device; click a header to sort; toggle kinds; "In view" limits the rows to the linear view's range; CSV downloads the rows shown.
- **Inspector**: parameters with units, devices and PV names (with copy buttons), fork targets, other traversals of a shared component, and floor poses.
- **Links**: the address bar always holds the current view. Copy it to show a colleague the same element.
