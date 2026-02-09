# EICViBE Optics Module

This module provides **package-level standard data structures** for beam optics and dynamics.

## Purpose

The optics module defines standardized data structures that are **engine-agnostic** - they can be populated by any simulation engine (XSuite, MAD-X, Elegant, future engines, etc.).

## Key Classes

### `TwissData`

Comprehensive Twiss parameter data structure containing:
- Beta functions (βx, βy) 
- Alpha functions (αx, αy)
- Dispersion functions (Dx, Dy, D'x, D'y)
- Phase advance (μx, μy)
- Closed orbit (x, px, y, py)
- Tunes and chromaticity
- Many more optical functions

**Design Philosophy:**
- **Engine-Agnostic**: Any engine can populate this structure
- **Comprehensive**: Supports all common optical functions
- **Flexible**: Optional fields for advanced features
- **Validated**: Pydantic ensures data consistency
- **XSuite-Compatible**: Provides aliases for XSuite naming conventions

### `SimulationMode`

Enum defining simulation modes:
- `RING`: Circular multi-turn tracking with periodic boundary
- `LINAC`: Linear accelerator single-pass tracking
- `RAMPING`: Time-dependent parameter evolution

## Usage

### Engines Fill TwissData

```python
from eicvibe.optics import TwissData, SimulationMode

# Example: XSuite engine fills TwissData
def calculate_twiss_xsuite(line) -> TwissData:
    xsuite_twiss = line.twiss(method='4d')
    
    # Convert to EICViBE standard
    twiss_data = TwissData(
        s=xsuite_twiss.s,
        beta_x=xsuite_twiss.betx,
        beta_y=xsuite_twiss.bety,
        alpha_x=xsuite_twiss.alfx,
        alpha_y=xsuite_twiss.alfy,
        dx=xsuite_twiss.dx,
        # ... more fields
        simulation_mode=SimulationMode.RING,
        reference_energy=line.particle_ref.p0c[0],
        engine_name="XSuite",
        computation_method="4d"
    )
    return twiss_data

# Example: Future MAD-X engine could do the same
def calculate_twiss_madx(madx) -> TwissData:
    madx.twiss()
    twiss_table = madx.table.twiss
    
    return TwissData(
        s=twiss_table.s,
        beta_x=twiss_table.betx,
        beta_y=twiss_table.bety,
        # ...
        simulation_mode=SimulationMode.RING,
        reference_energy=madx.beam.pc * 1e9,
        engine_name="MAD-X",
        computation_method="linear"
    )
```

### GUI/Tools Consume TwissData

```python
from eicvibe.optics import TwissData
from eicvibe.visualization import create_notebook_viewer

# GUI works at EICViBE level - doesn't care which engine produced the data
twiss: TwissData = engine.calculate_twiss()  # Any engine

viewer = create_notebook_viewer(
    lattice=lattice,
    twiss=twiss,  # Engine-agnostic TwissData
    branch_name="FODO"
)
viewer.show_twiss()
```

## Backward Compatibility

For existing code, `TwissData` is also re-exported from `eicvibe.simulators.types` for backward compatibility:

```python
# Old import (still works)
from eicvibe.simulators.types import TwissData

# New recommended import (package level)
from eicvibe.optics import TwissData
```

## Architecture

```
EICViBE Package Level
  ├── optics/
  │   ├── TwissData (standard data structure)
  │   └── SimulationMode (enum)
  │
  ├── simulators/
  │   ├── xsuite_interface.py (populates TwissData)
  │   ├── future_madx.py (populates TwissData)
  │   └── future_elegant.py (populates TwissData)
  │
  └── visualization/
      └── gui_app.py (consumes TwissData)
```

This ensures:
1. **Clear ownership**: Optics module owns the data structures
2. **Engine independence**: Engines implement, don't define
3. **Easy extension**: New engines just need to populate TwissData
4. **Consistency**: All tools work with same data structure
