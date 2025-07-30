# MAD-X Integration Guide

EICViBE provides seamless integration with MAD-X through the `madx_import` utility, enabling you to import existing MAD-X lattice files and leverage EICViBE's advanced selection and analysis capabilities.

## Overview

The MAD-X integration allows you to:
- Import MAD-X sequence definitions
- Automatically convert MAD-X elements to EICViBE elements
- Consolidate drift spaces for cleaner lattice representation
- Preserve element parameters and relationships
- Use EICViBE's advanced selection features on imported lattices

## Basic Usage

### Importing a MAD-X File

```python
from eicvibe.utilities.madx_import import import_madx_file

# Import a MAD-X file with automatic drift consolidation
lattice = import_madx_file("path/to/lattice.madx", sequence_name="main")

# Import without drift consolidation
lattice = import_madx_file(
    "path/to/lattice.madx", 
    sequence_name="main",
    consolidate_drifts=False
)
```

### Working with cpymad

You can also work directly with an existing cpymad Madx instance:

```python
from cpymad.madx import Madx
from eicvibe.utilities.madx_import import import_from_madx_instance

# Create or use existing Madx instance
madx = Madx()
madx.input('''
! Your MAD-X input here
BEAM, PARTICLE=ELECTRON, ENERGY=5.0;
''')

# Import the sequence
lattice = import_from_madx_instance(madx, sequence_name="mysequence")
```

## Element Mapping

The import process automatically maps MAD-X element types to EICViBE element classes:

| MAD-X Type | EICViBE Class | Parameters Imported |
|------------|---------------|-------------------|
| `DRIFT` | `Drift` | `L` → `length` |
| `QUADRUPOLE` | `Quadrupole` | `L` → `length`, `K1` → `kn1` |
| `SEXTUPOLE` | `Sextupole` | `L` → `length`, `K2` → `kn2` |
| `OCTUPOLE` | `Octupole` | `L` → `length`, `K3` → `kn3` |
| `SBEND` | `Bend` | `L` → `length`, `ANGLE` → `angle` |
| `RBEND` | `RBend` | `L` → `length`, `ANGLE` → `angle` |
| `MONITOR` | `Monitor` | `L` → `length` |
| `RFCAVITY` | `RFCavity` | `L` → `length`, `VOLT` → `voltage`, `FREQ` → `frequency` |
| `MARKER` | `Marker` | Zero length |
| `KICKER` | `Kicker` | `L` → `length`, kick parameters |
| `CRABCAVITY` | `CrabCavity` | `L` → `length`, RF parameters |

### Parameter Group Mapping

Parameters are organized into appropriate groups:

- **Magnetic parameters** → `MagneticMultipoleP` group
- **RF parameters** → `RFP` group  
- **Geometric parameters** → `GeometryP` group

## Drift Consolidation

One of the key features is automatic drift consolidation, which combines adjacent drift spaces for a cleaner lattice representation.

### How It Works

```python
# Before consolidation (MAD-X sequence):
# ELEM1: QUADRUPOLE, L=0.5
# DRIFT1: DRIFT, L=1.0  
# DRIFT2: DRIFT, L=0.5
# DRIFT3: DRIFT, L=2.0
# ELEM2: SEXTUPOLE, L=0.3

# After consolidation (EICViBE lattice):
# ELEM1: Quadrupole, length=0.5
# consolidated_drift_1: Drift, length=3.5  # 1.0 + 0.5 + 2.0
# ELEM2: Sextupole, length=0.3
```

### Controlling Consolidation

```python
# Enable consolidation (default)
lattice = import_madx_file("lattice.madx", "main", consolidate_drifts=True)

# Disable consolidation to preserve original structure
lattice = import_madx_file("lattice.madx", "main", consolidate_drifts=False)

# Check the difference
print(f"With consolidation: {len(lattice.expand_lattice())} elements")
lattice_no_consol = import_madx_file("lattice.madx", "main", consolidate_drifts=False)
print(f"Without consolidation: {len(lattice_no_consol.expand_lattice())} elements")
```

## Working with Imported Lattices

### Element Selection on Imported Lattices

Once imported, you can use all of EICViBE's selection features:

```python
# Import lattice
lattice = import_madx_file("eic_lattice.madx", "electron_ring")

# Select imported quadrupoles
quads = lattice.select_elements(element_type="Quadrupole")
print(f"Found {len(quads)} quadrupoles")

# Find focusing quadrupoles (positive K1)
focusing_quads = []
for quad in quads:
    mag_group = quad.get_parameter_group("MagneticMultipoleP")
    if mag_group and mag_group.get_parameter("kn1", 0) > 0:
        focusing_quads.append(quad)

# Use relative position selection with imported elements
ip_elements = lattice.select_elements(
    relative_position_range=("IP", -10.0, 10.0)  # 10m around interaction point
)
```

### Analyzing Imported Lattices

```python
# Get lattice statistics
total_length = lattice.get_total_path_length()
element_positions = lattice.get_element_positions()

print(f"Total lattice length: {total_length:.2f} m")
print(f"Number of elements: {len(lattice.expand_lattice())}")

# Analyze element types
from collections import Counter
element_types = [elem.type for elem in lattice.expand_lattice()]
type_counts = Counter(element_types)
print("Element type distribution:")
for elem_type, count in type_counts.items():
    print(f"  {elem_type}: {count}")

# Find magnetic elements
magnetic_elements = lattice.select_elements(element_type="Quadrupole") + \
                   lattice.select_elements(element_type="Sextupole") + \
                   lattice.select_elements(element_type="Bend")
print(f"Total magnetic elements: {len(magnetic_elements)}")
```

## Advanced Features

### Preserving MAD-X Element Names

The import process preserves original MAD-X element names:

```python
# MAD-X sequence with elements: QF.1, QF.2, QD.1, etc.
lattice = import_madx_file("lattice.madx", "main")

# Find elements by MAD-X naming convention
qf_elements = lattice.select_elements(name_pattern="QF", use_regex=True)
qd_elements = lattice.select_elements(name_pattern="QD", use_regex=True)
```

### Handling Complex Parameter Sets

```python
# Import lattice with complex elements
lattice = import_madx_file("complex_lattice.madx", "ring")

# Access imported parameters
for quad in lattice.select_elements(element_type="Quadrupole"):
    mag_group = quad.get_parameter_group("MagneticMultipoleP")
    if mag_group:
        k1 = mag_group.get_parameter("kn1", 0)
        tilt = mag_group.get_parameter("tilt", 0)
        print(f"{quad.name}: K1={k1:.3f}, tilt={tilt:.3f}")
```

### Ring vs Linac Detection

```python
# The import process can detect ring vs linac topology
lattice = import_madx_file("ring_lattice.madx", "electron_ring")

# Check detected topology
branch_type = lattice.get_branch_type("electron_ring")
print(f"Detected topology: {branch_type}")

# For rings, use wrap-around selection
if branch_type == "ring":
    # Select elements around the injection point
    injection_region = lattice.select_elements(
        relative_position_range=("injection_kicker", -5.0, 5.0)
    )
```

## Practical Examples

### Example 1: EIC Electron Ring Analysis

```python
# Import EIC electron ring
lattice = import_madx_file("eic_electron_ring.madx", "electron_ring")

# Set as ring topology for wrap-around behavior
lattice.set_branch_type("electron_ring", "ring")

# Find interaction regions
ip_markers = lattice.select_elements(name_pattern="IP", use_regex=True)
for ip in ip_markers:
    print(f"\nAnalyzing region around {ip.name}:")
    
    # Find elements within ±20m of IP
    ip_region = lattice.select_elements(
        relative_position_range=(ip.name, -20.0, 20.0)
    )
    
    # Analyze quadrupole gradients in IP region
    ip_quads = [elem for elem in ip_region if elem.type == "Quadrupole"]
    for quad in ip_quads:
        mag_group = quad.get_parameter_group("MagneticMultipoleP")
        k1 = mag_group.get_parameter("kn1", 0) if mag_group else 0
        distance = "TBD"  # Would calculate from positions
        print(f"  {quad.name}: K1={k1:.3f} at {distance}m from IP")
```

### Example 2: Lattice Comparison

```python
# Compare original and modified lattices
original = import_madx_file("original.madx", "main")
modified = import_madx_file("modified.madx", "main")

# Compare quadrupole strengths
orig_quads = original.select_elements(element_type="Quadrupole")
mod_quads = modified.select_elements(element_type="Quadrupole")

for orig, mod in zip(orig_quads, mod_quads):
    orig_k1 = orig.get_parameter_group("MagneticMultipoleP").get_parameter("kn1", 0)
    mod_k1 = mod.get_parameter_group("MagneticMultipoleP").get_parameter("kn1", 0)
    
    if abs(orig_k1 - mod_k1) > 1e-6:
        print(f"{orig.name}: {orig_k1:.6f} → {mod_k1:.6f} (Δ={mod_k1-orig_k1:.6f})")
```

### Example 3: Error Detection

```python
# Import and validate lattice
lattice = import_madx_file("lattice.madx", "main")

# Check for missing elements
required_elements = ["IP", "DUMP", "INJECTION_KICKER"]
for elem_name in required_elements:
    matches = lattice.select_elements(name_pattern=elem_name)
    if not matches:
        print(f"Warning: Required element '{elem_name}' not found")

# Check quadrupole focusing patterns
quads = lattice.select_elements(element_type="Quadrupole")
focusing_pattern = []
for quad in quads:
    mag_group = quad.get_parameter_group("MagneticMultipoleP")
    k1 = mag_group.get_parameter("kn1", 0) if mag_group else 0
    focusing_pattern.append("F" if k1 > 0 else "D" if k1 < 0 else "0")

print(f"Focusing pattern: {''.join(focusing_pattern)}")
```

## Tips and Best Practices

1. **Always specify the sequence name** when importing MAD-X files
2. **Use drift consolidation** for cleaner lattice representation
3. **Set branch topology** explicitly after import for ring behavior
4. **Validate imported parameters** by checking a few known elements
5. **Use EICViBE's selection features** to analyze imported lattices efficiently
6. **Preserve original MAD-X files** as references
7. **Test with small lattices first** to understand the mapping
8. **Check element counts** before and after consolidation
