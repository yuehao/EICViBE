# ADR 0003: Machine access through pvAccess, with a virtual accelerator service

Status: proposed (2026-10-07). Scope: the device registry (`eicvibe.domain.machine`), the virtual accelerator service, the replay service, and client access. Depends on ADR 0001. This changes the priority of REBUILD_PLAN §11 and roadmap item B14: machine access is needed long before the machine exists, because the virtual accelerator is the machine everyone works against until then.

## Decisions

1. **The contract is EPICS 7 pvAccess.** Clients read, write and monitor channels by their EIC names (`es6-q42-ps:sp`, `es6-b52:rdx`, `es9-cav2:sp:volt`). Values are Normative Types (`NTScalar`, `NTScalarArray`, `NTEnum`, `NTTable`) with value, alarm, timestamp, display and control fields. There is no second, EICViBE-specific API with its own semantics. A Python SDK and an HTTP/WebSocket bridge are conveniences on top of pvAccess, not alternatives to it.
2. **Every mode serves the same names with the same semantics.** VIRTUAL, REAL and REPLAY are separate pvAccess endpoints, selected by connection configuration (`EPICS_PVA_ADDR_LIST`, server port, or a gateway). A mode is never selected by a name prefix, because then names would differ between modes. A session is bound to one mode, and every log record names its mode.
3. **The virtual accelerator is a shared, long-running service.** Notebooks, GUIs and web pages use it at the same time, as they will use the real machine. Like a real machine, it has many readers and several writers. Writes are checked against permissions and limits, and logged.
4. **Writes to the real machine come later**, once the machine is ready. Until then the REAL endpoint is read-only, through the site gateway.
5. **Names follow EIC-SEG-RSI-009 Rev 02** (Naming Requirements, effective 2026-08-18), as described under Names below. The registry stores names: of devices and channels, and of every lattice element (`slots`, `aa-ddpp:nn` such as `ltr-q6` or `ltr-dt5`, drifts and markers included, as in the ESR naming table; bookkeeping elements such as forks and patches have none). The lattice keeps its design names, so the model still matches its source files. Viewers show the EIC name, with the design name beside it. A parser checks the syntax of every name when a registry is built, and a repository check reports abbreviations not yet approved.

```text
 notebooks (SDK)   Phoebus / scripts   web GUI ── HTTP/WebSocket bridge (optional)
        └─────────────┬──────────────────┘          │
                      │  pvAccess: EIC names, Normative Types
      ┌───────────────┼──────────────────┬──────────┘
 VIRTUAL service                    REAL (site gateway)        REPLAY service
 lattice revision + settings        production IOCs,           recorded timeline,
 + hidden errors + engine;          read-only for now          writes refused
 permissions, limits, write log
```

## Names (`eicvibe.domain.machine.naming`)

Rev 02 made the syntax EPICS compatible: `:` replaces the ADO `.` and the `_` of Rev 01, which the NameOps service and its examples still use.

- **Lattice (beamline) devices:** `aa-ddpp:nn-cc:ss`, in lower case.
  - `aa` is the machine prefix and sector (`es6`, `rcb1`).
  - `dd` is a device abbreviation chosen from the element (`q`, `qs`, `sx`, `dh`/`dv` by bend plane, `th`/`tv`/`t` for correctors, `b` for a BPM, `cav`, `sol`, `cm`).
  - `pp` is the clock coordinate: metres from the area's nominal IP to the element's exit, in the direction the sector counts, modulo the 320 m sector length, rounded half up. Rev 02 leaves this subsection empty, but the ESR naming table follows the rule for 7410 of its 7411 names; the exception is the ring's start marker.
  - `nn` numbers devices of one kind that share an area and a position, by s, from 1. A lone device has none.
- **Non-lattice devices:** `aa:bb-ddpp:zz:nn-ss`, e.g. the tune monitor `es-btun-rdx`.
- **Channels:** `<slot>-ps:sp` and `<slot>-ps:rb` for magnet supplies; `<slot>:rdx` and `<slot>:rdy` for BPMs; `:sp:h` and `:sp:v` for two-plane correctors; `<slot>:sp:volt` and `<slot>:sp:phase` for cavities. Each magnet and corrector has its own supply, as in the standard's example `es6-qf50_2-ps` ("the supply powering the second focusing quadrupole").
- **Areas:** `ring_sectors("es")` gives the ESR layout: s = 0 at IP6, IPs every 640 m (nominal; the lattice is 3834 m), even sectors counting forward from their IP and odd sectors counting back from the next one. Actual sector starts can be passed when they differ from the nominal ones.
- **Repository:** a snapshot in `eic_naming_repository.json`, built from NameOps and its SWN abbreviation database. The official repository is SEG's SharePoint page, and the deployed NameOps service (`epics-services.eic.bnl.gov:9090`) is only reachable on the BNL network.

## Device registry (`eicvibe.domain.machine`)

The lattice describes the machine's physics. The registry describes the devices the control system exposes, and how each channel relates to the lattice. It is a versioned document outside the `Lattice`, keyed by lattice ids, like `ParameterLimits` (ADR 0001): hardware changes independently of lattice revisions.

- **Device:** a power supply, BPM, RF station or similar, with a stable id, its EIC name, a type, and the lattice occurrences it belongs to. One power supply may feed a string of magnets in series or one magnet (see Power circuits below).
- **Channel:** a PV name, its device, a role, a Normative Type, unit, precision, and display and control limits.
  - Roles: **setting** (writable demand), **readback** (the device's reading of a setting), **measurement** (an observation of the beam or machine, such as a BPM position or a tune), and **status**.
  - Control limits replace the placeholder entries of `ParameterLimits` and map to the NT `control` field.
- **Binding:** a channel's link to the lattice.
  - A magnet supply's setting binds to its power circuit; its value is the circuit current.
  - Any other setting binds to a parameter path such as `RFP.voltage` on definition or occurrence ids, through a calibration.
  - A measurement binds to an observable at an occurrence or a branch, such as the closed orbit x at a BPM or the horizontal tune of a ring.
- **Calibration:** a polynomial from an engineering value to a model value, for settings that are not magnet currents (MV to V, degrees to radians).

### Power circuits and transfer functions (`eicvibe.domain.machine.power`)

A **power circuit** is one supply and the magnets it drives. A magnet powered on its own is a circuit of one. Main dipoles and quadrupole families are often **strings**: many magnets in series, one supply, one current. A trim winding on a magnet of a string is a second circuit, and the fields of all circuits that set one parameter of one magnet add (`total_strengths`). Each magnet on a circuit has:

- the traversals and lattice parameter it sets (`MagneticMultipoleP.Kn0` of a bend, `Kn1` or `Kn1L` of a quadrupole, …). A shared magnet sets several traversals; the septum sets ISEPTUM and ESEPTUM but not the BAR's SEPTUM;
- its wiring polarity, so that transfer functions stay positive as measured;
- its own **transfer function** from the supply's engineering value (A) to its field (T, T/m, …) or integrated field (T·m, T, …):
  - **polynomial:** B = c0 + c1 I + …; a linear approximation has two coefficients;
  - **table:** a measured excitation curve, interpolated linearly and extrapolated along its end segments;
  - **service:** a dedicated transfer-function service, for nonlinear magnets and hysteresis, where the field depends on the current's history. The registry names the service and the service's id of the magnet; the deployment resolves the name. The service keeps each magnet's history. A local fallback (polynomial or table) serves where no service is available, for instance offline. In code, a service is anything with `field(transfer, current)` and `current(transfer, field)`.

Strength = polarity × field / Bρ, with Bρ the reference rigidity of the magnet's branch, per length or integrated as the parameter is. The virtual accelerator runs this forward (current to strength). Setting a model strength, or comparing the model with the real machine, runs it backward within the supply's limits. A setting of a bend's `Kn0` gives its field explicitly, so whoever applies it also sets `Kn0_from_g_ref` false: then 0 A means no field. Since a string carries one current, `MachineRegistry.power_problems` reports a string whose magnets need different currents for their design values, for example after one magnet's measured curve replaced its placeholder. The viewer lists these under Notes.

Generated circuits use linear placeholder transfer functions: 1 mT/A for dipoles, 0.1 T/m per A for quadrupoles, 1 T/m² per A for sextupoles, 1 mT·m/A for correctors. `source` marks each as `placeholder` until measured curves replace them magnet by magnet.

The registry is generated for the virtual accelerator from the imported lattice plus a device description: which supplies drive which magnets, where the BPMs are, which channels exist. Later it is reconciled against the real IOC database or ChannelFinder, and any difference is an error.

## Virtual accelerator service

The first version runs over HTTP, with local accounts and shared machines plus private sandboxes (ADR 0008); pvAccess comes next, on the same write path and log.

One process publishes every registry channel with p4p's pvAccess server, and holds the virtual machine's state:

- **State:**
  - The accepted lattice revision.
  - A settings layer: the current setpoints, kept apart from structure as the tunable-settings design requires.
  - Hidden "true" errors that clients cannot read: misalignments, field and calibration errors, and BPM offsets, gains and noise. Without them, commissioning practice teaches nothing.
  - A clock.
- **Writes:**
  1. Identify the peer (pvAccess account and host).
  2. Check the peer's role against the channel's class.
  3. Check the control limits and the allowed rate of change.
  4. Append a record to the write log.
  5. Update the settings layer, and complete the put.

  A put completes when the setting is accepted; readbacks follow later, as on real hardware.
- **Physics updates:** changed settings, coalesced, go to an engine worker (Xsuite first, TrackPad later), which computes the closed orbit, optics and tunes of the true machine. The service then publishes readbacks:
  - supply readbacks, with ramp rate and noise;
  - BPM positions, with offsets and noise;
  - tunes and RF values.

  If the engine finds no stable solution, BPMs go to `INVALID` alarm, as real BPMs do without beam; no values are fabricated.
- **Concurrency:** the service has real-machine semantics, with no implicit locks. A study can take a lease on a group of devices (for example the orbit correctors), as REBUILD_PLAN §2 foresees for long-lived shared virtual-machine runs; other writers then get a refusal that names the lease holder.
- **Model workflow:** a snapshot of virtual settings can become a proposed change against the accepted revision, through the normal review and promotion workflow. Writes to the virtual machine never change the accepted revision directly.

## Replay service

A second pvAccess server publishes recorded values at a movable time cursor, controlled by playback channels outside the machine's names. The first recordings come from the virtual service: its write log plus its published values. Recordings of the real machine come later, from the EPICS Archiver Appliance. Writes to machine channels are refused with a pvAccess error. A replay snapshot can seed a new virtual instance, to ask what would have happened.

## Permissions and the write log

- **Roles:** viewer (reads only), physicist (writes settings on the virtual machine, within limits), operator (also writes status and leases), administrator (manages instances and hidden errors).
  - The virtual service enforces roles in its put handler, or a p4p gateway in front of it does.
  - pvAccess's default authentication reports an account and host but does not prove them. That is acceptable for a development service on the lab network. Stronger identity comes through the HTTP bridge (institutional login) or secure pvAccess.
- **Write log:** append-only. Each record holds:
  - id, time, mode and instance;
  - the peer's account and host;
  - channel, previous value and requested value;
  - the result of the checks, and the applied value or the reason for refusal;
  - a correlation id that groups the writes of one multi-channel change, and a free-text reason.

  It starts as a file or SQLite store owned by the service, and moves to the PostgreSQL event store of REBUILD_PLAN later. Writes are not atomic across channels, and hardware writes will never promise rollback (REBUILD_PLAN §11).
- **Real machine:** the machine's own access security governs it. EICViBE adds nothing that could bypass it.

## Client access

- **Notebooks:** an SDK (`eicvibe.machine`) on the p4p client.
  - It selects the endpoint for a mode and uses the registry for names, roles and units.
  - It offers get, put and monitor; snapshots of settings (save and restore); and model-versus-machine comparison through the calibrations.
  - It tags multi-channel puts with a correlation id.
- **Phoebus and existing EPICS scripts:** connect to the virtual or replay endpoint directly.
- **Web GUI:** an HTTP/WebSocket bridge that is itself a pvAccess client, written with FastAPI or taken from the existing PV Web Socket service. It inherits the mode of the endpoint it is configured for.

## Plan

| Step | Deliverable | Done when |
| --- | --- | --- |
| 1 | This ADR | Reviewed; the naming standard is available to encode |
| 2 | Device registry in the domain package, a generator from a lattice, EIC naming | A registry for the RCS validates against the imported lattice; done 2026-10-07 |
| 3 | Virtual accelerator v1: the static electron injector (LTR, BAR complex, RCS at injection energy, RTE) with supplies, correctors, BPMs, tunes and RF; write log and roles; SDK | An orbit-correction notebook, using only PV names, corrects hidden misalignments while a second client watches the orbit |
| 4 | HTTP/WebSocket bridge for the web GUI | Browser and notebook see the same values and timestamps |
| 5 | Replay service from virtual recordings | The orbit-correction notebook runs read-only against a recording |
| 6 | REAL, read-only, through the site gateway | The same notebook reads the real machine once IOCs exist |

The virtual accelerator first runs Xsuite in its own process. It moves behind the engine-worker protocol of roadmap M2 when that exists. Ramping (the RCS cycle) comes after the static machine.

## First machine: the electron injector complex (EIN)

`eicvibe.machines.ein.build_ein(folder)` assembles the EIN into one lattice, connected as in the legacy route notebooks. Its input is the EIN folder of the virtual accelerator project (`virtual_accelerator/EIN`):

- the MAD-X files in `EICVIBE_lattice/madx`, which `EIN_connection.ipynb` converted from the Bmad and Elegant designs of the `einlattices` checkout: LTR_main, LTR_P1, LTR_P2, ITL, BAR, ETL, the 40-cell 9 GeV RCS (RCS9GeV40CellV1) and the 10 GeV RTE;
- the preinjector beam in `einlattices/LTR`.

`EinSources` names the files. The RCS connection points are located by their neighbours, because the staged ring spells its injection point and septa all MAR.SEP: the injection point is the marker after MQA_US, and the extraction point is the marker after ES.PIP03. `EinSources(**RCSV5S1)` switches to the older RCS. `python -m eicvibe.machines.ein FOLDER` imports all this into the machine folder `FOLDER/machine` (ADR 0005), whose files people then edit: the lattice per branch, naming, power supplies, diagnostics, RF and the routes. Importing again shows how the design files differ from the lattice files and replaces them only with `--write`. The older outputs, `EICVIBE_lattice/ein_*.json` and the notebook's `preinjector_*.yaml`, were removed on 2026-10-11; `machine/build/` replaces them.

- **Names are temporary and configurable.** SEG has not assigned area prefixes to the EIN branches, so `TEMPORARY_AREAS` holds placeholders (the branch names). `ein_registry(ein, areas=..., scheme=NamingScheme(...))` replaces any prefix or signal token. Device ids derive from lattice occurrence and branch ids, not from names, so renaming changes PV names but keeps every device's identity. `ein_repository()` adds the temporary prefixes to the naming check, which then reports only the signal tokens awaiting approval.

- **The LTR is one branch, with one setting for both routes.** LTR_P1 and LTR_P2 contain the same elements as sections of LTR_main and differ only in 15 quadrupole strengths: the designers retuned the LTR for the BAR route. They were the BAR-mode settings until 2026-10-11; they are no longer imported. Instead the ITL and ETL are matched so that the BAR is transparent: the beam reaches P2 with the same optics through the BAR as past it, and one setting of the LTR serves both routes (ADR 0006). Each magnet, and each PV, exists once with one strength; choosing the BAR route or the direct route is a scenario choice.
- **The BAR attaches to the LTR with forks.** A fork after LTR_MARK_P1 leads to the ITL. The ITL injects through the septum into the BAR, the BAR extracts through it into the ETL, and the ETL lands back on the LTR after LTR_MARK_P2. The septum is one shared component with injection, circulation and extraction traversals. It is one magnet that each line models for its own beam: a bend (ISEPTUM, ESEPTUM) for the injected and extracted beams, whose field is on, and a kicker (SEPTUM) for the stored beam, for which it is pulsed off. The floor geometry joins the lines at the magnet's pole faces (ADR 0004), which closes the LTR–ITL–BAR–ETL loop. The registry generator makes no device per traversal, because one magnet must not appear as three; `ein_description` describes the magnet once (`SharedMagnet`): one element name, `bar-sptm18` with the temporary names, and one device, `bar-sptm18-ps`, both at all three traversals. Its pulsed supply setting (`:sp`, A, placeholder calibration of 1 mT/A) drives `MagneticMultipoleP.Kn0` of ISEPTUM and ESEPTUM only, so the stored beam never sees it. Those two bends state their field explicitly (`Kn0 = angle/L`, `Kn0_from_g_ref` false), so that a supply at 0 A means no field rather than the reference field.
- **Duplicated sections are cut.** LTR_main repeats the RCS injection section after MAR.INJ, and RTE_10GeV repeats the RCS extraction section up to MAR.SEP. Both copies are removed, so the beam passes each magnet once.
- **Initial conditions.** The LTR starts with the preinjector beam's Twiss parameters and floor position (beam_pre_inj.bmad). The RCS file is written at its extraction energy (9 GeV, 30 MV per cavity). As in EIN_connection.ipynb, the RCS is pinned at injection: 0.75 GeV and 1.698 MV total RF, from the ramp at t = 0 in the RCS parameter database. `build_ein(..., rcs_energy=None, rcs_rf_voltage=None)` keeps the file's values. The RTE stays at 10 GeV.
- **Power strings are assumed.** `EIN_SERIES` wires the rings' dipoles and quadrupoles in strings. Magnets of one kind with the same length and design strength (within 1 µm and 1 ppm) most likely share a supply, whatever their lattice definitions; magnets at zero strength stay on their own. In the BAR and RCS these strings are exactly the lattice families. The RCS's 156 main dipoles are on `rcs-dh111-ps`, named after the string's first magnet; its 4 weak and 2 "orig" dipoles differ in strength, so they are strings of their own. Every other magnet has its own supply, bends included. That gives 495 supplies, 720 circuits (a corrector has one per plane) and 20 strings. `ein_registry(ein, series=...)` takes the real power design when it is known.
- **Not modelled yet:** the preinjector, which exists as GPT and PARMELA input only.

## Open questions

1. Naming questions for SEG and the naming committee:
   - Approval of the signal tokens `h`, `v`, `volt` and `phase`, or the tokens to use instead.
   - Whether a lattice signal without a controller is written `<slot>:ss` (used here, the literal Rev 02 form) or `<slot>-ss` (NameOps).
   - Whether a string's supply is named after its first magnet (`rcs-dh111-ps`, used here) or otherwise.
   - Correctors: `th`/`tv` (abbreviation database) or `ks` (the ESR naming table).
   - Area prefixes for the BAR, ITL, ETL and RCS (the repository lists `ltr` and `rte`), and the RCS sections (`rcb1` …).
   - Units for BPM readings (mm here) and supply currents (A).
2. Who sets the hidden errors of a virtual instance, and whether several named instances (for example a main and a training instance) are needed from the start.
3. Update rates and noise models for supplies and BPMs, ideally from the hardware specifications.
4. Whether secure pvAccess will be available on the virtual accelerator's network, or whether login goes through the HTTP bridge only.
