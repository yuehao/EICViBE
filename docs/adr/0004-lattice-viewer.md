# ADR 0004: A shared web lattice viewer

Status: proposed (2026-10-07). Scope: floor geometry (`eicvibe.geometry`), the viewer API (`eicvibe.api`) and the web client (`web/`). Depends on ADR 0001; first step of REBUILD_PLAN §8 (machine explorer) and roadmap M3.

The first web screen shows one lattice revision three ways: a floor plan of every branch, a linear (synoptic) view of one branch along s, and its element table, with an inspector for the selected occurrence. Scientists use it to read optics settings and topology; engineers use it for positions, lengths, device and PV names, and CSV exports.

## Decisions

1. **The server keeps no per-user state.** Every endpoint reads an immutable revision addressed by its digest (`/api/v1/lattices/{digest}/...`). Branch, zoom, s range, selection and filters live in the browser and in the URL hash, so a link reproduces a view and nobody's browsing affects anyone else. This replaces the legacy `ViewerSession` shared by all clients (audit A01, A06, A09). There is no `POST /load`: the catalog is the set of lattice files and folders given when the server starts (audit A12).
2. **Responses are computed once per revision and served as bytes.** The catalog builds each response on first request (concurrent first requests wait for one build), keeps it with its gzip form, and revalidates with ETags. A changed file becomes a new revision; recent older revisions stay available, so a client is never switched under its feet. The client polls the listing and offers a newer revision, but never switches to it automatically. Several server processes (`--workers`) each hold their own cache, which is correct because nothing is shared.
3. **Per-occurrence data is columnar.** A branch is one response of parallel lists (table values and floor paths), which the client turns into typed arrays. For the RCS (2852 occurrences) that is 585 kB, or 95 kB gzipped.
4. **Floor geometry follows PALS and is computed by the package, not the client.** Bends use arc length, angle and `tilt_ref`, and patches use their offsets and rotations, so vertical bends and patches are placed correctly (the legacy web walk ignored both). Forks place branches in both directions: a fork places the branch it leads to, and a branch that forks into a placed branch is placed backward from it. A fork between two branches that are both already placed checks a loop and reports how well it closes (distance, angle). A fork that joins two traversals of one shared component at the same face matches the magnet's pole faces, not the reference orbits: a bend's entrance face is turned by −e1 from its orbit, its exit face by +e2, and other elements' faces are square to the orbit. This is how the EIN septum works: a rectangular magnet, a bend of −0.12 rad for the injected and extracted beams and a pulsed-off kicker for the stored beam, so the injected beam leaves it 1.75 mrad (e2) off the stored orbit. Matching orbits instead left the EIN loop open by 0.7 mm and 3.5 mrad (0.4 m × 1.75 mrad, twice); matching faces closes it exactly. Positions agree with MAD-X SURVEY to 1e-14 m.
5. **The client draws with Canvas 2D, not SVG.** This departs from REBUILD_PLAN §8, which keeps SVG until profiling shows a need. Here the floor plan shows every branch at once, which the legacy one never did, and the legacy SVG redraws its whole tree on every mouse move. Each branch becomes one `Path2D` per element kind in world coordinates. Pan and zoom change only the transform, hit testing uses a uniform grid, and hover and range updates bypass React. The table renders only its visible rows. The client was measured in headless Chrome at 1600×950 on the EIN (3.2k occurrences) and on a synthetic ring of 25k occurrences. Wheel zoom, drag pan, hover, linear-view zoom and table scrolling all held p95 frame intervals of 16.7 ms (60 fps); the largest single frame was 33 ms, in the 25k zoom. Controls, the table and the inspector remain HTML.
6. **API types are generated.** `npm run generate-types` writes the OpenAPI document of the Pydantic response models and the TypeScript types from it. A Python test fails when the committed document is out of date.

## Layout

- `eicvibe.geometry`: `floor_geometry(lattice)` (frames of every occurrence, fork placement, closures) and `element_path` (sampled bend arcs). Depends only on the domain.
- `eicvibe.api.views.RevisionViews`: response models from a lattice and optional registry. No web framework imports; usable in notebooks.
- `eicvibe.api.catalog`: discovery (`<stem>_lattice.json` paired with `<stem>_registry.json` when it describes that lattice), revisions by digest, cached payloads.
- `eicvibe.api.app`: FastAPI routes and the static client; `eicvibe-web` (`python -m eicvibe.api`) runs it.
- `web/`: React and TypeScript client built with Vite into `src/eicvibe/api/static` (not committed; build it before packaging).

## Measurements

These were taken on 2026-10-07 on a development Mac, with one server process unless stated.

- EIN branch payloads: first request 21 ms, then 1.6 ms.
- 50 simulated users loading the whole view at once (1400 requests): about 2000 requests/s, p95 47 ms. 200 users: p95 190 ms with one process, 72 ms with `--workers 4`.

## Not yet

- Authentication. The server listens on loopback by default; the API is read-only and accepts no paths, but expose it only on a trusted network until OIDC (REBUILD_PLAN §12).
- Optics. The linear view shares its s axis with the Twiss and orbit plots planned for the next steps.
- Machine modes (the BAR mode settings), the registry's naming report, and live values from the virtual accelerator (ADR 0003).
