# EICViBE Documentation

**EIC Virtual Beam Environment** - A modular accelerator physics simulation framework.

```{toctree}
:maxdepth: 2
:caption: User Guides

guides/index
guides/lattice_design
guides/element_selection
guides/madx_integration
guides/simulation_engines
guides/matched_beams
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/index
api/machine_portal
```

## Overview

EICViBE provides a **modular, inheritance-based** architecture for accelerator lattice modeling and simulation:

- **Machine Portal** - Core lattice modeling with prototype/instance pattern
- **Simulators** - Physics engines (XSuite integration) with LINAC/RING/RAMPING modes
- **Models** - Pydantic-validated parameter groups
- **Utilities** - MAD-X import and helpers

## Quick Start

```python
from eicvibe.machine_portal import Lattice, Quadrupole, Drift

# Create a simple FODO cell
lattice = Lattice(name="fodo")
lattice.add_branch("main", branch_type="ring")

# Define elements
qf = Quadrupole(name="QF", length=0.5)
qf.add_parameter("MagneticMultipoleP", "kn1", 1.2)

qd = Quadrupole(name="QD", length=0.5)  
qd.add_parameter("MagneticMultipoleP", "kn1", -1.2)

drift = Drift(name="D", length=2.0)
```

## Installation

```bash
# Using uv (recommended)
uv add eicvibe

# Using pip
pip install eicvibe
```

## Indices and Tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
