# EICViBE - EIC Virtual Beam Environment

EICViBE is a Python package for accelerator physics simulations, specifically designed for the Electron-Ion Collider (EIC) and other particle accelerator facilities. It provides a comprehensive framework for modeling accelerator lattices, managing beam dynamics, and interfacing with simulation tools like MAD-X.

## Features

### 🎯 Core Capabilities
- **Lattice Modeling**: Comprehensive accelerator element library with inheritance-based design
- **Element Selection**: Advanced filtering with multiple criteria and relative positioning
- **Ring & Linac Support**: Full support for both linear and circular accelerator topologies
- **MAD-X Integration**: Import and optimize MAD-X lattices with drift consolidation
- **Parameter Management**: Flexible parameter grouping and inheritance system
- **Visualization**: 1D beamline and 2D floor plan plotting capabilities

### 🚀 Advanced Features
- **Relative Position Selection**: Select elements relative to reference points with negative offset support
- **Ring Wrap-around**: Seamless element selection across ring boundaries
- **Drift Consolidation**: Intelligent merging of consecutive drift spaces from MAD-X imports
- **Branch Management**: Multi-branch lattice support with different topologies
- **YAML Export/Import**: Human-readable lattice serialization

## Installation

### Prerequisites
- Python ≥ 3.11
- Optional: MAD-X installation for cpymad integration

### Install from Source
```bash
git clone https://github.com/yuehao/EICViBE.git
cd EICViBE
pip install -e .
```

### Dependencies
- `cpymad>=1.17.0` - MAD-X Python interface
- `matplotlib>=3.10.3` - Plotting and visualization
- `numpy>=2.3.1` - Numerical computations
- `pyyaml>=6.0.2` - YAML serialization
- `pytest>=8.4.0` - Testing framework

## Quick Start

### Basic Lattice Creation

```python
from eicvibe.machine_portal.lattice import Lattice
from eicvibe.machine_portal.drift import Drift
from eicvibe.machine_portal.quadrupole import Quadrupole
from eicvibe.machine_portal.parameter_group import ParameterGroup

# Create a lattice
lattice = Lattice(name="my_lattice")

# Define prototype elements
drift = Drift(name="d1", length=1.0)
quad = Quadrupole(name="qf", length=0.5)

# Add magnetic parameters to quadrupole
mag_group = ParameterGroup(name="MagneticMultipoleP", type="MagneticMultipoleP")
mag_group.add_parameter("kn1", 0.1)  # Focusing strength
quad.add_parameter_group(mag_group)

# Add elements to lattice
lattice.add_element(drift)
lattice.add_element(quad)

# Create a branch and add element instances
lattice.add_branch("main", branch_type="linac")
lattice.add_element_to_branch("main", "d1")
lattice.add_element_to_branch("main", "qf")
lattice.add_element_to_branch("main", "d1")

print(f"Total length: {lattice.get_total_path_length('main'):.1f} m")
```

### Advanced Element Selection

```python
# Select by type
quads = lattice.select_elements(element_type="Quadrupole")

# Select by name pattern
d1_elements = lattice.select_elements(name_pattern="d1")

# Select by relative position (NEW!)
nearby_elements = lattice.select_elements(
    relative_position_range=("qf_1", -0.5, 1.0),  # ±0.5m around qf_1
    branch_name="main"
)

# Combined criteria
specific_drifts = lattice.select_elements(
    element_type="Drift",
    inherit_name="d1",
    position_range=(0.0, 10.0)
)
```

### Ring Topology with Wrap-around

```python
# Create a ring branch
lattice.add_branch("ring", branch_type="ring")

# Add elements to ring
for elem_name in ["qf", "d1", "qd", "d1"]:
    lattice.add_element_to_branch("ring", elem_name)

# Select elements across ring boundary
wrap_elements = lattice.select_elements(
    relative_position_range=("last_element", -1.0, 1.0),  # Wraps around
    branch_name="ring"
)
```

### MAD-X Integration

```python
from eicvibe.utilities.madx_import import lattice_from_madx_file

# Import MAD-X lattice with drift consolidation
lattice = lattice_from_madx_file(
    "my_lattice.madx",
    lattice_name="RING",
    consolidate_drifts=True  # Merges consecutive drifts
)

print(f"Imported {len(lattice.branches['RING'])} elements")
```

## Package Structure

```
eicvibe/
├── machine_portal/          # Core lattice and element classes
│   ├── element.py          # Base element class
│   ├── lattice.py          # Lattice management with advanced selection
│   ├── parameter_group.py  # Parameter management
│   ├── drift.py           # Drift space elements
│   ├── quadrupole.py      # Quadrupole magnets
│   ├── bend.py            # Bending magnets
│   ├── monitor.py         # Beam position monitors
│   └── ...                # Other element types
├── utilities/              # Utility functions
│   ├── madx_import.py     # MAD-X import with optimization
│   └── element_types.yaml # Element type definitions
├── control/               # Control system interface
├── simulators/            # Simulation backends
└── visualization/         # Plotting and visualization
```

## Documentation

### API Reference
- [Machine Portal API](docs/api/machine_portal.md) - Core lattice classes
- [Element Types](docs/api/elements.md) - Available accelerator elements
- [Utilities](docs/api/utilities.md) - Helper functions and MAD-X integration
- [Selection Guide](docs/guides/element_selection.md) - Advanced element selection

### Guides
- [Getting Started](docs/guides/getting_started.md) - Basic usage examples
- [Lattice Design](docs/guides/lattice_design.md) - Creating accelerator lattices
- [MAD-X Import](docs/guides/madx_integration.md) - Working with MAD-X files
- [Ring vs Linac](docs/guides/topologies.md) - Handling different accelerator types

### Examples
- [Basic Lattice](examples/basic_lattice.ipynb) - Simple lattice creation
- [Element Selection Demo](examples/element_selection_demo.ipynb) - Advanced selection features
- [MAD-X Import](examples/madx_import_example.ipynb) - Real-world lattice import

## Development

### Running Tests
```bash
pytest tests/
```

### Code Style
We follow PEP 8 style guidelines with type hints for better code maintainability.

### Contributing
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes with tests
4. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **EIC Collaboration** - For accelerator design requirements
- **MAD-X Team** - For the excellent accelerator physics toolkit
- **Python Scientific Community** - For the foundational libraries

## Support

- 📧 **Issues**: [GitHub Issues](https://github.com/yuehao/EICViBE/issues)
- 📖 **Documentation**: [Full Documentation](docs/)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/yuehao/EICViBE/discussions)

---

**EICViBE** - Advancing accelerator physics simulation for the next generation of particle physics research.
