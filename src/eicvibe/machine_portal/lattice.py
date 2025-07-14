# Create a lattice class to represent a lattice structure in the machine portal.
# Lattice consists of a list of branches, each branch is a list of elements.
# One branch is marked as the root branch, 
# Lattice can be expanded, starting from root branch, to a list of elements, so that tracking can be done through the lattice.

from eicvibe.machine_portal.element import Element
from eicvibe.machine_portal.parameter_group import ParameterGroup
from eicvibe.machine_portal.drift import Drift
from eicvibe.machine_portal.bend import Bend
from eicvibe.machine_portal.quadrupole import Quadrupole
from eicvibe.machine_portal.marker import Marker
from dataclasses import dataclass, field
import copy


def create_element_by_type(element_type: str, name: str, length: float = 0.0, inherit: str | None = None) -> Element:
    """Factory function to create the correct element type based on element_type string."""
    element_classes = {
        'Drift': Drift,
        'Bend': Bend,
        'Quadrupole': Quadrupole,
        'Marker': Marker,
    }
    
    element_class = element_classes.get(element_type, Element)
    
    # Create element with appropriate constructor
    if element_type == 'Drift':
        return Drift(name=name, length=length, inherit=inherit)
    elif element_type == 'Bend':
        return Bend(name=name, length=length, inherit=inherit)
    elif element_type == 'Quadrupole':
        return Quadrupole(name=name, length=length, inherit=inherit)
    elif element_type == 'Marker':
        return Marker(name=name, length=length, inherit=inherit)
    else:
        return Element(name=name, type=element_type, length=length, inherit=inherit)


@dataclass
class Branch:
    """Class representing a branch in the lattice structure.
    A branch has a name and contains a list of elements."""
    name: str
    elements: list[Element] = field(default_factory=list)

    def __post_init__(self):
        """Initialize the branch with a name and elements."""
        if not self.name:
            raise ValueError("Branch must have a name.")
        if not isinstance(self.elements, list):
            raise TypeError("Elements must be a list of Element instances.")
        for element in self.elements:
            if not isinstance(element, Element):
                raise TypeError("All elements in the branch must be instances of Element.")
            
    def add_element(self, element: Element):
        """Add an element to the branch."""
        if not isinstance(element, Element):
            raise TypeError("Element must be an instance of Element.")
        self.elements.append(element)


@dataclass
class Lattice:
    """Class representing a lattice structure in the machine portal.
    A lattice consists of multiple branches, each branch has its own name and contains a list of element names.
    One branch is marked as the root branch, and contains a pool of element definitions."""
    name: str 
    branches: dict[str, list[str]] = field(default_factory=dict)  # Dictionary of branches, where key is branch name and value is a list of element names in that branch
    root_branch_name: str = ""  # Name of the root branch
    elements: dict[str, Element] = field(default_factory=dict)  # Pool of element definitions
    element_occurrences: dict[str, int] = field(default_factory=dict)  # Track occurrence count for each element name

    def __post_init__(self):
        """Initialize the lattice with a name and branches."""
        if not self.name:
            raise ValueError("Lattice must have a name.")
        if not isinstance(self.branches, dict):
            raise TypeError("Branches must be a dictionary with branch names as keys and lists of elements as values.")
        if self.root_branch_name and self.root_branch_name not in self.branches:
            raise ValueError(f"Root branch '{self.root_branch_name}' must be present in the branches.")

    def add_element(self, element: Element):
        """Add an element definition to the lattice."""
        if not isinstance(element, Element):
            raise TypeError("Element must be an instance of Element.")
        if element.name in self.elements:
            raise ValueError(f"Element '{element.name}' already exists in the lattice.")
        self.elements[element.name] = element

    def _get_next_occurrence_number(self, element_name: str) -> int:
        """Get the next occurrence number for an element name."""
        return self.element_occurrences.get(element_name, 0) + 1

    def _increment_occurrence_count(self, element_name: str):
        """Increment the occurrence count for an element name."""
        self.element_occurrences[element_name] = self._get_next_occurrence_number(element_name)

    def get_element(self, name: str) -> Element:
        """Get an element definition by name."""
        if name not in self.elements:
            raise ValueError(f"Element '{name}' not found.")
        return self.elements[name]


    
    def add_branch(self, branch_name: str, elements: list[Element] | list[str] | None = None):
        """Add a new branch to the lattice.
        Accepts either a list of Element objects or a list of element names."""
        if not branch_name:
            raise ValueError("Branch name cannot be empty.")
        if elements is None:
            elements = []
        if not isinstance(elements, list):
            raise TypeError("Elements must be a list of Element objects or element names.")
        if branch_name in self.branches:
            raise ValueError(f"Branch '{branch_name}' already exists in the lattice.")
        
        self.branches[branch_name] = []
        
        # Handle the elements based on their type
        for item in elements:
            if isinstance(item, Element):
                # If it's an Element object, add it to the element pool first
                if item.name not in self.elements:
                    self.add_element(item)
                self.add_element_to_branch(branch_name, item.name)
            elif isinstance(item, str):
                # If it's a string (element name), validate and add
                if item not in self.elements:
                    raise ValueError(f"Element '{item}' not found in the element pool.")
                self.add_element_to_branch(branch_name, item)
            else:
                raise TypeError(f"Invalid element type: {type(item)}. Must be Element object or string.")
        
        # Set as root branch if this is the first branch
        if not self.root_branch_name:
            self.root_branch_name = branch_name

    def add_element_to_branch(self, branch_name: str, element_name: str, **overrides):
        """Add an element to a branch by referencing its name in the element pool.
        Creates a copy with automatic naming: existing_name_{#of appearance}.
        Optional parameter overrides can be provided."""
        if branch_name not in self.branches:
            raise ValueError(f"Branch '{branch_name}' does not exist.")
        if element_name not in self.elements:
            raise ValueError(f"Element '{element_name}' not found in the element pool.")
        
        # Get the next occurrence number and increment the counter
        count = self._get_next_occurrence_number(element_name)
        self._increment_occurrence_count(element_name)
        
        # Create a copy with the new name
        original = self.get_element(element_name)
        new_element = create_element_by_type(
            element_type=original.type,
            name=f"{element_name}_{count}",
            length=original.length,
            inherit=element_name
        )
        
        # Copy parameter groups
        for group in original.parameters:
            new_group = copy.deepcopy(group)
            new_element.add_parameter_group(new_group)
        
        # Apply overrides
        for param_group, params in overrides.items():
            for param_name, param_value in params.items():
                new_element.add_parameter(param_group, param_name, param_value)
        
        # Add the copy to the element pool
        self.elements[new_element.name] = new_element
        
        # Add only the element name to the branch
        self.branches[branch_name].append(new_element.name)
        return new_element

    def remove_branch(self, branch_name: str):
        """Remove a branch from the lattice."""
        if branch_name not in self.branches:
            raise ValueError(f"Branch '{branch_name}' does not exist in the lattice.")
        if branch_name == self.root_branch_name:
            raise ValueError("Cannot remove the root branch.")
        
        del self.branches[branch_name]

    def set_root_branch(self, branch_name: str):
        """Set a new root branch for the lattice."""
        if branch_name not in self.branches:
            raise ValueError(f"Branch '{branch_name}' does not exist in the lattice.")
        self.root_branch_name = branch_name
    

    # Lattice expansion
    # This function expands the lattice starting from the root branch. 
    # If elements in the root branch is forked to other branches, then other branches will be added.  
    # If the element has a fork parameter group and the parameter force_expand is set to True,
    # then we will expand the forked branches, the branch name is specified in 'to_line' parameter and the starting element is specified in 'to_ele' parameter in ForkP group
    def _expand_branch(self, branch_name: str, start_element: str | None = None, force_expand: bool = False) -> list[Element]:
        """Recursively expand a branch."""
        branch_element_names = self.branches[branch_name]
        elements = []
        started = start_element is None
        
        for element_name in branch_element_names:
            element = self.elements[element_name]  # Get the actual element from pool
            
            if not started and element.name == start_element:
                started = True
            
            if started:
                elements.append(element)
                
                # Check for ForkP parameter group
                fork_group = element.get_parameter_group("ForkP")
                if fork_group is not None:
                    to_line = fork_group.get_parameter("to_line")
                    to_ele = fork_group.get_parameter("to_ele")
                    force = fork_group.get_parameter("force_expand", False)
                    
                    if (isinstance(to_line, str) and isinstance(to_ele, str) and 
                        (force_expand or (isinstance(force, bool) and force))):
                        if to_line in self.branches:
                            # Recursively expand the forked branch
                            forked_elements = self._expand_branch(to_line, to_ele, force_expand)
                            elements.extend(forked_elements)
        
        return elements

    def expand_lattice(self, force_expand: bool = False) -> list[Element]:
        """Expand the lattice starting from the root branch."""
        if not self.root_branch_name:
            raise ValueError("No root branch set for lattice expansion.")
        return self._expand_branch(self.root_branch_name, force_expand=force_expand)
    
    def plot_branch_beamline(self, branch_name: str, ax, s_start: float = 0.0, normalized_strength=None):
        """Plot a branch in 1D beamline view using element plotting methods."""
        
        if branch_name not in self.branches:
            raise ValueError(f"Branch '{branch_name}' does not exist.")
        
        s_current = s_start
        
        # Plot each element in the branch
        for element_name in self.branches[branch_name]:
            element = self.elements[element_name]
            s_current = element.plot_in_beamline(ax, s_current, normalized_strength)
        
        # Set plot properties
        ax.set_xlabel('S coordinate (m)')
        ax.set_xlim(s_start, s_current)
        ax.set_ylim(-1, 1)
        #ax.set_ylabel('Normalized strength')
        #ax.set_title(f'Beamline view: {branch_name}')
        ax.grid(True, alpha=0.3)
        
        return s_current
    
    def plot_branch_floorplan(self, branch_name: str, ax, entrance_coords=(0, 0), entrance_angle=0):
        """Plot a branch in 2D floor plan view using element plotting methods."""
        import numpy as np
        
        if branch_name not in self.branches:
            raise ValueError(f"Branch '{branch_name}' does not exist.")
        
        # Initial position and direction
        current_coords = np.array(entrance_coords)
        current_vector = np.array([np.cos(entrance_angle), np.sin(entrance_angle)])
        
        # Plot each element in the branch
        for element_name in self.branches[branch_name]:
            element = self.elements[element_name]
            current_coords, current_vector = element.plot_in_floorplan(ax, current_coords, current_vector)
        
        # Set plot properties
        ax.set_xlabel('X coordinate (m)')
        ax.set_ylabel('Y coordinate (m)')
        ax.set_title(f'Floor plan view: {branch_name}')
        ax.grid(True, alpha=0.3)
        
        return current_coords, current_vector
    
    def to_yaml_dict(self) -> dict:
        """Convert the lattice to YAML format.
        Structure: prototype elements first, then branches with inline element details."""
        result: dict = {
            'name': self.name,
            'root_branch': self.root_branch_name
        }
        
        # First, output all prototype elements (inherit=None) as a list
        prototype_elements = []
        for element_name, element in self.elements.items():
            if element.inherit is None:
                prototype_elements.append(element.to_yaml_dict())
        
        if prototype_elements:
            result['elements'] = prototype_elements
        
        # Add all branches with inline element details
        for branch_name, element_names in self.branches.items():
            branch_elements = []
            for element_name in element_names:
                element = self.elements[element_name]
                branch_elements.append(element.to_yaml_dict())
            result[branch_name] = branch_elements
        
        return result
    
    @classmethod
    def from_yaml_dict(cls, data: dict) -> 'Lattice':
        """Create a Lattice from a YAML dictionary with inline element details."""
        name = data.get('name', '')
        root_branch = data.get('root_branch', '')
        
        # Create lattice instance without setting root branch yet
        lattice = cls(name=name, root_branch_name="")
        
        # First, load prototype elements (inherit=None)
        prototypes_data = data.get('elements', [])
        for element_data in prototypes_data:
            if isinstance(element_data, dict) and len(element_data) == 1:
                element_type = list(element_data.keys())[0]
                element_info = element_data[element_type]
                
                # Create prototype element from the data
                element = create_element_by_type(
                    element_type=element_type,
                    name=element_info.get('name', ''),
                    length=element_info.get('length', 0.0),
                    inherit=None  # Prototypes have inherit=None
                )
                
                # Add parameter groups
                for param_group_name, param_data in element_info.items():
                    if param_group_name not in ['name', 'length', 'inherit']:
                        param_group = ParameterGroup(name=param_group_name, type=param_group_name)
                        if isinstance(param_data, dict):
                            for param_name, param_value in param_data.items():
                                param_group.add_parameter(param_name, param_value)
                        element.add_parameter_group(param_group)
                
                # Add prototype element to lattice
                lattice.add_element(element)
        
        # Process each branch and its inline elements
        for key, value in data.items():
            if key not in ['name', 'root_branch', 'elements']:
                # This is a branch
                branch_name = key
                branch_elements = value
                element_names = []
                
                if isinstance(branch_elements, list):
                    # Process each element in the branch
                    for element_data in branch_elements:
                        if isinstance(element_data, dict) and len(element_data) == 1:
                            element_type = list(element_data.keys())[0]
                            element_info = element_data[element_type]
                            
                            # Create element from the data
                            element = create_element_by_type(
                                element_type=element_type,
                                name=element_info.get('name', ''),
                                length=element_info.get('length', 0.0),
                                inherit=element_info.get('inherit')  # Preserve inherit info if available
                            )
                            
                            # Add parameter groups
                            for param_group_name, param_data in element_info.items():
                                if param_group_name not in ['name', 'length', 'inherit']:
                                    param_group = ParameterGroup(name=param_group_name, type=param_group_name)
                                    if isinstance(param_data, dict):
                                        for param_name, param_value in param_data.items():
                                            param_group.add_parameter(param_name, param_value)
                                    element.add_parameter_group(param_group)
                            
                            # Add element to lattice and track its name
                            lattice.add_element(element)
                            element_names.append(element.name)
                    
                    # Add the branch with the element names
                    lattice.add_branch(branch_name, element_names)
        
        # Set the root branch after all branches are created
        if root_branch:
            lattice.set_root_branch(root_branch)
        
        return lattice 