# Element Selection User Guide

This guide demonstrates the advanced element selection capabilities in EICViBE's lattice system.

## Overview

The `select_elements()` method provides powerful filtering capabilities to find elements by multiple criteria. All criteria use AND logic, meaning elements must match ALL specified conditions.

## Basic Selection Methods

### By Element Type

```python
# Select all quadrupoles
quads = lattice.select_elements(element_type="Quadrupole")

# Select all drifts
drifts = lattice.select_elements(element_type="Drift")

# Select RF cavities
cavities = lattice.select_elements(element_type="RFCavity")
```

### By Name Pattern

```python
# Simple substring matching
focusing_elements = lattice.select_elements(name_pattern="qf")

# Wildcard patterns (use use_regex=False)
dipoles = lattice.select_elements(name_pattern="bend*")

# Regular expression matching
numbered_quads = lattice.select_elements(
    name_pattern=r"qf_\d+",  # Matches qf_1, qf_2, etc.
    use_regex=True
)
```

### By Inheritance

```python
# Find all elements inheriting from 'main_quadrupole'
quad_instances = lattice.select_elements(inherit_name="main_quadrupole")
```

### By Position Range

```python
# Select elements between 10m and 50m
middle_elements = lattice.select_elements(position_range=(10.0, 50.0))

# Select elements near the beginning
start_elements = lattice.select_elements(position_range=(0.0, 5.0))
```

### By Index Range

```python
# Select first 10 elements
first_ten = lattice.select_elements(index_range=(0, 9))

# Select elements 50-100
middle_range = lattice.select_elements(index_range=(50, 100))
```

## Advanced: Relative Position Selection

The relative position feature allows you to select elements based on their position relative to a reference element. This is especially powerful for ring lattices with wrap-around capability.

### Basic Relative Selection

```python
# Select elements within 2m after the "injection_point" marker
post_injection = lattice.select_elements(
    relative_position_range=("injection_point", 0.0, 2.0)
)

# Select elements 1m before to 1m after "main_detector"
around_detector = lattice.select_elements(
    relative_position_range=("main_detector", -1.0, 1.0)
)
```

### Negative Offsets (Back-tracing)

Negative offsets allow you to select elements **before** the reference element:

```python
# Select elements in the 5m leading up to "final_focus"
approach_region = lattice.select_elements(
    relative_position_range=("final_focus", -5.0, 0.0)
)

# Select 2m before to 3m after "interaction_point"
interaction_region = lattice.select_elements(
    relative_position_range=("interaction_point", -2.0, 3.0)
)
```

### Ring Wrap-around

For branches with `branch_type="ring"`, selection automatically wraps around the ring boundaries:

```python
# Example: Ring circumference = 100m, reference at position 95m
# Selection range: reference -2m to +5m
# This spans: 93m-95m + 0m-3m (wrapping around)
wrap_selection = lattice.select_elements(
    relative_position_range=("near_end_element", -2.0, 5.0),
    branch_name="electron_ring"  # Must be ring topology
)

# Select elements around the injection point (which might be near position 0)
injection_area = lattice.select_elements(
    relative_position_range=("injection_kicker", -3.0, 3.0),
    branch_name="main_ring"
)
```

## Combining Criteria

All selection criteria can be combined for precise filtering:

### Example 1: Specific Element Types in Regions

```python
# Find quadrupoles within 10m of the interaction point
ip_quadrupoles = lattice.select_elements(
    element_type="Quadrupole",
    relative_position_range=("interaction_point", -10.0, 10.0)
)

# Find focusing quadrupoles near the final focus
focusing_near_ff = lattice.select_elements(
    element_type="Quadrupole",
    name_pattern="qf",  # Assuming focusing quads have "qf" in name
    relative_position_range=("final_focus", -5.0, 2.0)
)
```

### Example 2: Diagnostics in Specific Sections

```python
# Find all monitors in the first arc
arc1_monitors = lattice.select_elements(
    element_type="Monitor",
    position_range=(100.0, 200.0),  # Arc1 spans 100m-200m
    branch_name="main_line"
)

# Find BPMs near each quadrupole (requires iteration)
for quad in lattice.select_elements(element_type="Quadrupole"):
    nearby_bpms = lattice.select_elements(
        element_type="Monitor",
        relative_position_range=(quad.name, -0.5, 0.5)
    )
    print(f"Near {quad.name}: {[bpm.name for bpm in nearby_bpms]}")
```

### Example 3: RF Systems

```python
# Find all RF elements in the acceleration section
rf_elements = lattice.select_elements(
    element_type="RFCavity",
    position_range=(50.0, 150.0)
)

# Find elements that might interfere with RF (within 1m of cavities)
for cavity in rf_elements:
    nearby = lattice.select_elements(
        relative_position_range=(cavity.name, -1.0, 1.0)
    )
    non_drift = [elem for elem in nearby if elem.type != "Drift"]
    if len(non_drift) > 1:  # More than just the cavity itself
        print(f"Elements near {cavity.name}: {[e.name for e in non_drift]}")
```

## Branch-Specific Selection

When working with multi-branch lattices:

```python
# Select elements from specific branches
main_quads = lattice.select_elements(
    element_type="Quadrupole",
    branch_name="main_line"
)

bypass_elements = lattice.select_elements(
    branch_name="bypass_line"
)

# Compare elements between branches
main_monitors = lattice.select_elements(
    element_type="Monitor",
    branch_name="main_line"
)
bypass_monitors = lattice.select_elements(
    element_type="Monitor", 
    branch_name="bypass_line"
)
```

## Practical Examples

### Lattice Analysis

```python
# Check quadrupole spacing
quads = lattice.select_elements(element_type="Quadrupole")
positions = lattice.get_element_positions()
quad_positions = [pos[1] for pos in positions if pos[0] in quads]
spacings = [quad_positions[i+1] - quad_positions[i] for i in range(len(quad_positions)-1)]
print(f"Quadrupole spacings: {spacings}")

# Find drift lengths in interaction region
ip_drifts = lattice.select_elements(
    element_type="Drift",
    relative_position_range=("interaction_point", -20.0, 20.0)
)
drift_lengths = [drift.length for drift in ip_drifts]
print(f"Drift lengths near IP: {drift_lengths}")
```

### Lattice Modifications

```python
# Select and modify all focusing quadrupoles
focusing_quads = lattice.select_elements(
    element_type="Quadrupole",
    name_pattern="qf"
)

for quad in focusing_quads:
    # Increase focusing strength by 5%
    magnetic_group = quad.get_parameter_group("MagneticMultipoleP")
    if magnetic_group:
        current_strength = magnetic_group.get_parameter("kn1", 0.0)
        new_strength = current_strength * 1.05
        magnetic_group.add_parameter("kn1", new_strength)
```

### Quality Assurance

```python
# Check for missing diagnostics
quads = lattice.select_elements(element_type="Quadrupole")
for quad in quads:
    # Look for monitors within 2m of each quadrupole
    nearby_monitors = lattice.select_elements(
        element_type="Monitor",
        relative_position_range=(quad.name, -2.0, 2.0)
    )
    if not nearby_monitors:
        print(f"Warning: No monitor near quadrupole {quad.name}")

# Verify element spacing in rings
if lattice.get_branch_type("main_ring") == "ring":
    # Check that elements are evenly distributed
    all_elements = lattice.select_elements(branch_name="main_ring")
    ring_length = lattice.get_total_path_length("main_ring")
    expected_spacing = ring_length / len(all_elements)
    
    # Check actual spacings using relative position
    for i, elem in enumerate(all_elements[:-1]):
        next_elem = all_elements[i+1]
        # This would need additional position calculation logic
        # but demonstrates the concept
```

## Tips and Best Practices

1. **Use relative positioning** for selections that should adapt to lattice changes
2. **Combine criteria** to be as specific as possible
3. **Check branch topology** before using ring-specific features
4. **Cache selection results** if used multiple times
5. **Use meaningful names** for reference elements in relative selections
6. **Test with wrap-around** when working with rings
7. **Validate selections** by checking the number and types of returned elements
