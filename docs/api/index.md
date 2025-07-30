# API Reference

## Core Modules

### Machine Portal (`eicvibe.machine_portal`)

The machine portal module provides the core classes for modeling accelerator lattices and elements.

- **[Lattice](machine_portal.md#lattice-class)** - Main lattice management class with advanced element selection
- **[Element Classes](machine_portal.md#element-classes)** - Base and specific element types (Drift, Quadrupole, Bend, etc.)
- **[Parameter Management](machine_portal.md#parameter-management)** - ParameterGroup system for element parameters

### Utilities (`eicvibe.utilities`)

Utility functions for lattice manipulation and analysis.

- **[MAD-X Import](../guides/madx_integration.md)** - Import MAD-X lattice files with cpymad integration
- **Element Type Definitions** - YAML-based element type specifications

### Control (`eicvibe.control`)

Control system interfaces and beam dynamics calculations.

### Simulators (`eicvibe.simulators`)

Integration with external simulation codes.

### Visualization (`eicvibe.visualization`)

Plotting and visualization tools for lattice analysis.

## Quick Reference

### Element Selection

```python
# Basic selection
elements = lattice.select_elements(element_type="Quadrupole")

# Advanced selection with relative positioning
elements = lattice.select_elements(
    element_type="Monitor",
    relative_position_range=("IP", -10.0, 10.0),
    branch_name="main_ring"
)
```

### Lattice Management

```python
# Create lattice
lattice = Lattice(name="accelerator")

# Add branch with topology
lattice.add_branch("main_ring", branch_type="ring")

# Add elements
lattice.add_element_to_branch("main_ring", "quad_prototype", kn1=1.5)
```

### MAD-X Integration

```python
# Import MAD-X file
lattice = import_madx_file("lattice.madx", "sequence_name")

# Set topology for advanced features
lattice.set_branch_type("main_ring", "ring")
```

## Class Hierarchy

```
Element (base)
├── Drift
├── Marker
├── Monitor
├── Quadrupole
├── Sextupole
├── Octupole
├── Bend
├── RBend
├── RFCavity
├── CrabCavity
└── Kicker

Lattice
├── branches: dict[str, list[str]]
├── elements: dict[str, Element]
├── branch_specs: dict[str, str]
└── Methods for element selection and management

ParameterGroup
├── MagneticMultipoleP (magnetic parameters)
├── GeometryP (geometric parameters)
├── RFP (RF parameters)
└── Custom parameter groups
```

For detailed documentation of specific classes and methods, see the individual module pages.
