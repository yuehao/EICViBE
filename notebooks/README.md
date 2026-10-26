# Notebooks

Examples of building on EICViBE in Jupyter (ADR 0016). Each runs from start to end and is meant to be
changed.

| notebook | what it does | needs |
|---|---|---|
| `01_rcs_ramp.ipynb` | ramps a virtual RCS from 0.75 to 9 GeV, then reads every turn of the ramp from its BPMs: tunes against energy, damping, charge, the bunch extracted | the EIN machine; about a minute with TrackPad |
| `02_orbit_correction.ipynb` | corrects a ring's orbit through the virtual accelerator, in a sandbox: BPMs read through the client, a response matrix from the model, a correction written by hand and by `correct_orbit`, written as one logged change, checked against the BPMs | a running virtual accelerator whose ring has errors, and a token |
| `03_bpm_data.ipynb` | kicks a virtual RCS with hidden quadrupole errors, writes its BPMs' turns to an HDF5 file and reads it back; tunes, phase advances and beta-beating from the file, against the model and the truth | the EIN machine; half a minute |

Run them with EICViBE's own environment: `uv sync`, then `uv run jupyter lab` in the checkout, or
VS Code with the checkout's `.venv`. Without `npm run build` in `web/` they run, but the viewer's
views (`ev.OpticsView`, `ev.FloorView`, `ev.OrbitView`, `ev.chart`) are missing.

They read the machine folder from `EICVIBE_MACHINE` (by default `../../virtual_accelerator/EIN/machine`,
beside the checkout) and write runs into `EICVIBE_RUNS` (by default `runs/` here, which git ignores). The
virtual accelerator's notebook takes `EICVIBE_URL` and `EICVIBE_TOKEN` (`eicvibe-va token STATE NAME`),
and `EICVIBE_VA_MACHINE` to choose its ring.
