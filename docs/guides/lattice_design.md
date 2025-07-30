# Lattice Design Guide

This guide covers best practices for designing accelerator lattices using EICViBE, from basic element creation to complex multi-branch systems.

## Design Philosophy

EICViBE promotes a modular, inheritance-based approach to lattice design:

1. **Define prototypes** for common element types
2. **Create instances** that inherit from prototypes
3. **Organize elements** into logical branches
4. **Use topology specifications** for ring vs linac behavior
5. **Leverage selection tools** for analysis and modification

## Basic Lattice Creation

### Step 1: Create the Lattice

```python
from eicvibe.machine_portal.lattice import Lattice
from eicvibe.machine_portal.element import create_element_by_type

# Create a new lattice
lattice = Lattice(name="my_accelerator")
```

### Step 2: Define Prototype Elements

```python
# Create prototype drift
drift_proto = create_element_by_type("Drift", "standard_drift", length=1.0)
lattice.add_element(drift_proto)

# Create prototype quadrupole
quad_proto = create_element_by_type("Quadrupole", "standard_quad", length=0.5)
quad_proto.add_parameter_group(ParameterGroup("MagneticMultipoleP", "magnetic"))
quad_proto.add_parameter("MagneticMultipoleP", "kn1", 1.2)  # 1.2 m^-2
lattice.add_element(quad_proto)

# Create prototype bend
bend_proto = create_element_by_type("Bend", "standard_bend", length=2.0)
bend_proto.add_parameter_group(ParameterGroup("GeometryP", "geometric"))
bend_proto.add_parameter("GeometryP", "angle", 0.1)  # 0.1 radians
lattice.add_element(bend_proto)
```

### Step 3: Build the Lattice Structure

```python
# Create a branch with automatic instance naming
lattice.add_branch("main_line", branch_type="linac")

# Add elements to the branch (creates instances automatically)
lattice.add_element_to_branch("main_line", "standard_drift")  # → standard_drift_1
lattice.add_element_to_branch("main_line", "standard_quad", kn1=1.5)  # Override strength
lattice.add_element_to_branch("main_line", "standard_drift")  # → standard_drift_2
lattice.add_element_to_branch("main_line", "standard_quad", kn1=-1.5)  # Defocusing
lattice.add_element_to_branch("main_line", "standard_drift")  # → standard_drift_3
```

## Advanced Design Patterns

### Pattern 1: FODO Cell Design

```python
def create_fodo_cell(lattice, cell_name, quad_strength=1.2, drift_length=2.0, quad_length=0.5):
    """Create a FODO cell with focusing-drift-defocusing-drift pattern."""
    
    # Define prototypes if not already existing
    if f"{cell_name}_drift" not in lattice.elements:
        drift = create_element_by_type("Drift", f"{cell_name}_drift", length=drift_length)
        lattice.add_element(drift)
    
    if f"{cell_name}_quad_f" not in lattice.elements:
        quad_f = create_element_by_type("Quadrupole", f"{cell_name}_quad_f", length=quad_length)
        quad_f.add_parameter_group(ParameterGroup("MagneticMultipoleP", "magnetic"))
        quad_f.add_parameter("MagneticMultipoleP", "kn1", quad_strength)
        lattice.add_element(quad_f)
    
    if f"{cell_name}_quad_d" not in lattice.elements:
        quad_d = create_element_by_type("Quadrupole", f"{cell_name}_quad_d", length=quad_length)
        quad_d.add_parameter_group(ParameterGroup("MagneticMultipoleP", "magnetic"))
        quad_d.add_parameter("MagneticMultipoleP", "kn1", -quad_strength)
        lattice.add_element(quad_d)
    
    # Build the cell structure
    cell_elements = [
        f"{cell_name}_quad_f",  # Focusing quadrupole
        f"{cell_name}_drift",   # Half drift
        f"{cell_name}_quad_d",  # Defocusing quadrupole  
        f"{cell_name}_drift"    # Half drift
    ]
    
    return cell_elements

# Use the FODO cell
fodo_elements = create_fodo_cell(lattice, "arc1", quad_strength=1.5, drift_length=3.0)

# Add multiple FODO cells to a branch
lattice.add_branch("arc1", branch_type="linac")
for i in range(10):  # 10 FODO cells
    for element_name in fodo_elements:
        lattice.add_element_to_branch("arc1", element_name)
```

### Pattern 2: Interaction Region Design

```python
def create_interaction_region(lattice, ip_name="IP"):
    """Create a symmetric interaction region with final focus quadrupoles."""
    
    # Interaction point marker
    ip_marker = create_element_by_type("Marker", ip_name, length=0.0)
    lattice.add_element(ip_marker)
    
    # Final focus quadrupoles
    qf_final = create_element_by_type("Quadrupole", "qf_final", length=1.0)
    qf_final.add_parameter_group(ParameterGroup("MagneticMultipoleP", "magnetic"))
    qf_final.add_parameter("MagneticMultipoleP", "kn1", 5.0)  # Strong focusing
    lattice.add_element(qf_final)
    
    qd_final = create_element_by_type("Quadrupole", "qd_final", length=1.0)
    qd_final.add_parameter_group(ParameterGroup("MagneticMultipoleP", "magnetic"))
    qd_final.add_parameter("MagneticMultipoleP", "kn1", -5.0)  # Strong defocusing
    lattice.add_element(qd_final)
    
    # Short drifts
    short_drift = create_element_by_type("Drift", "ff_drift", length=0.5)
    lattice.add_element(short_drift)
    
    # Build interaction region
    lattice.add_branch("interaction_region", branch_type="linac")
    
    # Symmetric structure around IP
    elements = [
        "qf_final", "ff_drift", "qd_final", "ff_drift",
        ip_name,  # Interaction point at center
        "ff_drift", "qd_final", "ff_drift", "qf_final"
    ]
    
    for element_name in elements:
        lattice.add_element_to_branch("interaction_region", element_name)

# Create the interaction region
create_interaction_region(lattice, "IP1")
```

### Pattern 3: Ring Design with Injection/Extraction

```python
def create_storage_ring(lattice, circumference=1000.0):
    """Create a storage ring with injection and extraction systems."""
    
    # Calculate required bending
    n_bends = 8  # 8 bending magnets
    bend_angle = 2 * np.pi / n_bends  # Total bending / number of bends
    bend_length = 3.0
    
    # Create bend prototype
    ring_bend = create_element_by_type("Bend", "ring_bend", length=bend_length)
    ring_bend.add_parameter_group(ParameterGroup("GeometryP", "geometric"))
    ring_bend.add_parameter("GeometryP", "angle", bend_angle)
    lattice.add_element(ring_bend)
    
    # Create other prototypes
    ring_quad_f = create_element_by_type("Quadrupole", "ring_quad_f", length=0.8)
    ring_quad_f.add_parameter_group(ParameterGroup("MagneticMultipoleP", "magnetic"))
    ring_quad_f.add_parameter("MagneticMultipoleP", "kn1", 1.0)
    lattice.add_element(ring_quad_f)
    
    ring_quad_d = create_element_by_type("Quadrupole", "ring_quad_d", length=0.8)
    ring_quad_d.add_parameter_group(ParameterGroup("MagneticMultipoleP", "magnetic"))
    ring_quad_d.add_parameter("MagneticMultipoleP", "kn1", -1.0)
    lattice.add_element(ring_quad_d)
    
    # Injection kicker
    inj_kicker = create_element_by_type("Kicker", "injection_kicker", length=0.5)
    lattice.add_element(inj_kicker)
    
    # Monitors
    bpm = create_element_by_type("Monitor", "bpm", length=0.1)
    lattice.add_element(bpm)
    
    # Drifts of various lengths
    for length in [1.0, 2.0, 3.0]:
        drift = create_element_by_type("Drift", f"drift_{int(length*10)}", length=length)
        lattice.add_element(drift)
    
    # Build ring structure
    lattice.add_branch("storage_ring", branch_type="ring")
    
    # Create one sector (1/8 of ring)
    sector_pattern = [
        "ring_bend", "drift_10", "bpm", "drift_10",
        "ring_quad_f", "drift_20", 
        "ring_quad_d", "drift_30",
        "bpm"
    ]
    
    # Add 8 sectors
    for sector in range(8):
        for element_name in sector_pattern:
            # Add injection kicker in first sector
            if sector == 0 and element_name == "bpm":
                lattice.add_element_to_branch("storage_ring", "injection_kicker")
            lattice.add_element_to_branch("storage_ring", element_name)

# Create the storage ring
create_storage_ring(lattice)
```

## Multi-Branch Lattice Design

### Complex Accelerator with Multiple Lines

```python
def create_complex_accelerator(lattice):
    """Create a complex accelerator with injector, main linac, and rings."""
    
    # 1. Injector linac
    lattice.add_branch("injector", branch_type="linac")
    
    # Low energy, short quadrupoles
    inj_quad = create_element_by_type("Quadrupole", "injector_quad", length=0.2)
    inj_quad.add_parameter_group(ParameterGroup("MagneticMultipoleP", "magnetic"))
    inj_quad.add_parameter("MagneticMultipoleP", "kn1", 2.0)  # Strong gradient for low energy
    lattice.add_element(inj_quad)
    
    inj_drift = create_element_by_type("Drift", "injector_drift", length=0.5)
    lattice.add_element(inj_drift)
    
    # RF cavities for acceleration
    inj_cavity = create_element_by_type("RFCavity", "injector_cavity", length=1.0)
    inj_cavity.add_parameter_group(ParameterGroup("RFP", "rf"))
    inj_cavity.add_parameter("RFP", "voltage", 10e6)  # 10 MV
    inj_cavity.add_parameter("RFP", "frequency", 1.3e9)  # 1.3 GHz
    lattice.add_element(inj_cavity)
    
    # Build injector
    for i in range(20):  # 20-cell injector
        lattice.add_element_to_branch("injector", "injector_quad", kn1=2.0/(1+i*0.1))  # Adiabatic
        lattice.add_element_to_branch("injector", "injector_drift")
        if i % 5 == 0:  # Every 5th cell has cavity
            lattice.add_element_to_branch("injector", "injector_cavity")
    
    # 2. Main linac
    lattice.add_branch("main_linac", branch_type="linac")
    
    # Higher energy, longer elements
    main_quad = create_element_by_type("Quadrupole", "main_quad", length=0.5)
    main_quad.add_parameter_group(ParameterGroup("MagneticMultipoleP", "magnetic"))
    main_quad.add_parameter("MagneticMultipoleP", "kn1", 1.0)
    lattice.add_element(main_quad)
    
    main_cavity = create_element_by_type("RFCavity", "main_cavity", length=2.0)
    main_cavity.add_parameter_group(ParameterGroup("RFP", "rf"))
    main_cavity.add_parameter("RFP", "voltage", 50e6)  # 50 MV
    lattice.add_element(main_cavity)
    
    # 3. Electron storage ring (created earlier)
    create_storage_ring(lattice)  # This creates "storage_ring" branch
    
    # 4. Bypass line
    lattice.add_branch("bypass", branch_type="linac")
    
    bypass_bend = create_element_by_type("Bend", "bypass_bend", length=1.5)
    bypass_bend.add_parameter_group(ParameterGroup("GeometryP", "geometric"))
    bypass_bend.add_parameter("GeometryP", "angle", 0.2)  # 0.2 rad bend
    lattice.add_element(bypass_bend)

# Create the complex system
create_complex_accelerator(lattice)
```

## Design Validation and Analysis

### Consistency Checking

```python
def validate_lattice_design(lattice):
    """Validate the lattice design for common issues."""
    
    issues = []
    
    # Check all branches
    for branch_name in lattice.branches.keys():
        branch_elements = lattice.select_elements(branch_name=branch_name)
        
        # 1. Check for zero-length non-marker elements
        for elem in branch_elements:
            if elem.length == 0 and elem.type not in ["Marker"]:
                issues.append(f"Zero-length {elem.type} element: {elem.name}")
        
        # 2. Check quadrupole focusing pattern
        quads = lattice.select_elements(element_type="Quadrupole", branch_name=branch_name)
        if len(quads) > 1:
            focusing_pattern = []
            for quad in quads:
                mag_group = quad.get_parameter_group("MagneticMultipoleP")
                if mag_group:
                    k1 = mag_group.get_parameter("kn1", 0)
                    focusing_pattern.append("F" if k1 > 0 else "D" if k1 < 0 else "0")
            
            # Check for alternating pattern
            pattern_str = "".join(focusing_pattern)
            if "FF" in pattern_str or "DD" in pattern_str:
                issues.append(f"Non-alternating quadrupoles in {branch_name}: {pattern_str}")
        
        # 3. Check for missing diagnostics
        total_length = lattice.get_total_path_length(branch_name)
        monitors = lattice.select_elements(element_type="Monitor", branch_name=branch_name)
        monitor_density = len(monitors) / total_length if total_length > 0 else 0
        
        if monitor_density < 0.01:  # Less than 1 monitor per 100m
            issues.append(f"Low monitor density in {branch_name}: {monitor_density:.3f}/m")
        
        # 4. For rings, check closure
        if lattice.get_branch_type(branch_name) == "ring":
            bends = lattice.select_elements(element_type="Bend", branch_name=branch_name)
            bends += lattice.select_elements(element_type="RBend", branch_name=branch_name)
            
            total_angle = 0
            for bend in bends:
                geom_group = bend.get_parameter_group("GeometryP")
                if geom_group:
                    total_angle += geom_group.get_parameter("angle", 0)
            
            if abs(total_angle - 2*np.pi) > 0.01:  # 0.01 rad tolerance
                issues.append(f"Ring {branch_name} doesn't close: total angle = {total_angle:.3f} rad")
    
    return issues

# Validate the design
issues = validate_lattice_design(lattice)
for issue in issues:
    print(f"⚠️  {issue}")

if not issues:
    print("✅ Lattice design validation passed!")
```

### Performance Analysis

```python
def analyze_lattice_performance(lattice):
    """Analyze key performance metrics of the lattice."""
    
    for branch_name in lattice.branches.keys():
        print(f"\n=== Analysis of {branch_name} ===")
        
        # Basic statistics
        total_length = lattice.get_total_path_length(branch_name)
        all_elements = lattice.select_elements(branch_name=branch_name)
        
        print(f"Total length: {total_length:.2f} m")
        print(f"Number of elements: {len(all_elements)}")
        
        # Element type distribution
        type_counts = {}
        for elem in all_elements:
            type_counts[elem.type] = type_counts.get(elem.type, 0) + 1
        
        print("Element distribution:")
        for elem_type, count in sorted(type_counts.items()):
            print(f"  {elem_type}: {count}")
        
        # Magnetic element analysis
        quads = lattice.select_elements(element_type="Quadrupole", branch_name=branch_name)
        if quads:
            quad_strengths = []
            for quad in quads:
                mag_group = quad.get_parameter_group("MagneticMultipoleP")
                if mag_group:
                    k1 = mag_group.get_parameter("kn1", 0)
                    quad_strengths.append(abs(k1))
            
            if quad_strengths:
                print(f"Quadrupole strengths: max={max(quad_strengths):.2f}, "
                      f"avg={sum(quad_strengths)/len(quad_strengths):.2f} m⁻²")
        
        # RF system analysis
        cavities = lattice.select_elements(element_type="RFCavity", branch_name=branch_name)
        if cavities:
            total_voltage = 0
            for cavity in cavities:
                rf_group = cavity.get_parameter_group("RFP")
                if rf_group:
                    voltage = rf_group.get_parameter("voltage", 0)
                    total_voltage += voltage
            
            print(f"Total RF voltage: {total_voltage/1e6:.1f} MV")

# Analyze performance
analyze_lattice_performance(lattice)
```

## Best Practices Summary

### Design Principles

1. **Use inheritance** - Create prototypes and inherit instances
2. **Modular design** - Build reusable cell patterns
3. **Consistent naming** - Use clear, systematic element names  
4. **Parameter validation** - Check element parameters make physical sense
5. **Topology awareness** - Set branch types correctly for ring/linac behavior

### Organization Tips

1. **Group related elements** in branches
2. **Use meaningful branch names** (e.g., "injector", "arc1", "interaction_region")
3. **Document design choices** in comments
4. **Version control** lattice files
5. **Test incrementally** as you build

### Performance Considerations

1. **Use drift consolidation** for imported lattices
2. **Cache selection results** for repeated use
3. **Prefer relative positioning** for maintainable selections
4. **Validate designs early** and often
5. **Profile large lattices** for performance bottlenecks

### Common Pitfalls to Avoid

1. **Forgetting to set ring topology** for circular machines
2. **Creating excessive drift fragmentation** 
3. **Inconsistent quadrupole focusing patterns**
4. **Missing diagnostic elements**
5. **Not validating lattice closure** for rings
6. **Hardcoding positions** instead of using relative selection
