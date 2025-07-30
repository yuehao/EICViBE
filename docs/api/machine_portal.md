# Machine Portal API Reference

The Machine Portal module provides the core classes for modeling accelerator lattices and elements.

## Lattice Class

The `Lattice` class is the central component for managing accelerator lattices.

### Class: `eicvibe.machine_portal.lattice.Lattice`

```python
@dataclass
class Lattice:
    """Class representing a lattice structure in the machine portal.
    
    A lattice consists of multiple branches, each branch has its own name and 
    contains a list of element names. One branch is marked as the root branch, 
    and contains a pool of element definitions.
    """
```

#### Attributes

- `name: str` - Name of the lattice
- `branches: dict[str, list[str]]` - Dictionary mapping branch names to element name lists
- `root_branch_name: str` - Name of the root branch
- `elements: dict[str, Element]` - Pool of element definitions
- `element_occurrences: dict[str, int]` - Track occurrence count for each element name
- `branch_specs: dict[str, str]` - Branch specifications ('ring' or 'linac')

#### Key Methods

##### Branch Management

```python
def add_branch(self, branch_name: str, elements: list[Element] | list[str] | None = None, branch_type: str = "linac")
```
Add a new branch to the lattice with specified topology.

**Parameters:**
- `branch_name` - Name of the branch to add
- `elements` - List of Element objects or element names
- `branch_type` - Either 'ring' or 'linac' (default: 'linac')

```python
def set_branch_type(self, branch_name: str, branch_type: str)
def get_branch_type(self, branch_name: str) -> str
def get_all_branch_specs(self) -> dict[str, str]
```

##### Element Management

```python
def add_element(self, element: Element, check_consistency: bool = True)
```
Add an element definition to the lattice.

```python
def add_element_to_branch(self, branch_name: str, element_name: str, **overrides) -> Element
```
Add an element instance to a branch with automatic naming and optional parameter overrides.

##### Element Selection (Advanced)

```python
def select_elements(self, 
                   element_type: str = None,
                   name_pattern: str = None,
                   inherit_name: str = None,
                   position_range: tuple[float, float] = None,
                   relative_position_range: tuple[str, float, float] = None,
                   index_range: tuple[int, int] = None,
                   branch_name: str = None,
                   use_regex: bool = False) -> list[Element]
```

Select elements by multiple criteria with AND logic.

**Parameters:**
- `element_type` - Type of elements to select (e.g., 'Drift', 'Quadrupole')
- `name_pattern` - Pattern to match element names against
- `inherit_name` - Name of prototype element for inheritance filtering
- `position_range` - Tuple of (start_position, end_position) in meters
- `relative_position_range` - **NEW!** Tuple of (element_name, start_offset, end_offset)
- `index_range` - Tuple of (start_index, end_index) for element indices
- `branch_name` - Name of branch to search (None uses root branch)
- `use_regex` - Whether to treat name_pattern as regular expression

**Relative Position Selection:**
The `relative_position_range` parameter enables selection relative to a reference element:
- Format: `("reference_element", start_offset, end_offset)`
- Offsets are measured from the **start** of the reference element
- **Negative offsets** enable back-tracing from the reference element
- **Ring topology** automatically handles wrap-around at boundaries

**Examples:**
```python
# Select elements by type
quads = lattice.select_elements(element_type="Quadrupole")

# Select elements near a reference quadrupole
nearby = lattice.select_elements(
    relative_position_range=("qf_5", -1.0, 2.0),  # -1m to +2m from qf_5
    branch_name="main"
)

# Combined criteria
focusing_drifts = lattice.select_elements(
    element_type="Drift",
    relative_position_range=("focusing_quad", 0.0, 5.0),
    name_pattern="d1"
)

# Ring wrap-around selection
ring_selection = lattice.select_elements(
    relative_position_range=("last_element", -0.5, 1.0),  # Wraps around
    branch_name="ring"  # Must be ring topology
)
```

##### Utility Methods

```python
def get_total_path_length(self, branch_name: str = None) -> float
def get_element_positions(self, branch_name: str = None) -> list[tuple[Element, float, float]]
def expand_lattice(self, force_expand: bool = False) -> list[Element]
```

## Element Classes

### Base Element Class

```python
class Element:
    """Base class for all accelerator elements."""
    
    def __init__(self, name: str, type: str, length: float = 0.0, inherit: str | None = None)
```

#### Common Attributes
- `name: str` - Unique element name
- `type: str` - Element type identifier
- `length: float` - Physical length in meters
- `inherit: str | None` - Name of prototype element (if this is an instance)
- `parameters: list[ParameterGroup]` - List of parameter groups

#### Common Methods
```python
def add_parameter_group(self, group: ParameterGroup)
def get_parameter_group(self, name: str) -> ParameterGroup | None
def add_parameter(self, group_name: str, param_name: str, value)
def check_consistency(self)
```

### Specific Element Types

#### Drift
```python
class Drift(Element):
    """Drift space element for field-free regions."""
```

#### Quadrupole
```python
class Quadrupole(Element):
    """Quadrupole magnet element."""
```
**Required Parameters:**
- `MagneticMultipoleP` group with `kn1` parameter (focusing strength)

#### Bend/RBend
```python
class Bend(Element):
    """Sector bending magnet."""

class RBend(Element):
    """Rectangular bending magnet."""
```

#### Monitor
```python
class Monitor(Element):
    """Beam position monitor or diagnostic element."""
```

#### RFCavity
```python
class RFCavity(Element):
    """RF accelerating cavity."""
```

## Parameter Management

### ParameterGroup Class

```python
class ParameterGroup:
    """Container for related element parameters."""
    
    def __init__(self, name: str, type: str)
    def add_parameter(self, name: str, value)
    def get_parameter(self, name: str, default=None)
```

#### Common Parameter Groups

- **MagneticMultipoleP** - Magnetic multipole strengths
  - `kn1` - Quadrupole strength (m⁻²)
  - `kn2` - Sextupole strength (m⁻³)
  - `kn3` - Octupole strength (m⁻⁴)

- **GeometryP** - Geometric parameters
  - `angle` - Bending angle (rad)
  - `e1`, `e2` - Edge angles (rad)

- **RFP** - RF cavity parameters
  - `voltage` - Accelerating voltage (V)
  - `frequency` - RF frequency (Hz)
  - `phase` - RF phase (rad)

## Factory Functions

```python
def create_element_by_type(element_type: str, name: str, length: float = 0.0, inherit: str | None = None) -> Element
```

Creates the appropriate element subclass based on the type string.

**Supported Types:**
- `Drift`, `Quadrupole`, `Sextupole`, `Octupole`
- `Bend`, `RBend`, `Marker`, `Monitor`
- `RFCavity`, `CrabCavity`, `Kicker`

## Advanced Features

### Ring Topology Support

When a branch is specified as `branch_type="ring"`, the lattice system:
- Enables wrap-around behavior for position-based selections
- Handles circular coordinate systems automatically
- Supports element selection across the ring junction (0-position boundary)

### Element Inheritance System

Elements can inherit from prototype definitions:
- **Prototype elements** have `inherit=None` and define base parameters
- **Instance elements** have `inherit="prototype_name"` and can override parameters
- Automatic naming: instances are named `prototype_name_{occurrence_number}`

### Consistency Checking

Elements can validate their parameters:
```python
element.check_consistency()  # Raises ValueError if invalid
```

Each element type implements specific validation rules (e.g., quadrupoles require magnetic parameters).
