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
from eicvibe.machine_portal.sextupole import Sextupole
from eicvibe.machine_portal.octupole import Octupole
from eicvibe.machine_portal.rfcavity import RFCavity
from eicvibe.machine_portal.crabcavity import CrabCavity
from eicvibe.machine_portal.kicker import Kicker
from eicvibe.machine_portal.monitor import Monitor
from eicvibe.machine_portal.rbend import RBend
from dataclasses import dataclass, field
import copy


def create_element_by_type(element_type: str, name: str, length: float = 0.0, inherit: str | None = None) -> Element:
    """Factory function to create the correct element type based on element_type string."""
    element_classes = {
        'Drift': Drift,
        'Bend': Bend,
        'RBend': RBend,
        'Quadrupole': Quadrupole,
        'Sextupole': Sextupole,
        'Octupole': Octupole,
        'RFCavity': RFCavity,
        'CrabCavity': CrabCavity,
        'Kicker': Kicker,
        'Monitor': Monitor,
        'Marker': Marker,
    }
    
    element_class = element_classes.get(element_type, Element)
    
    # Create element with appropriate constructor
    if element_type == 'Drift':
        return Drift(name=name, length=length, inherit=inherit)
    elif element_type == 'Bend':
        return Bend(name=name, length=length, inherit=inherit)
    elif element_type == 'RBend':
        return RBend(name=name, length=length, inherit=inherit)
    elif element_type == 'Quadrupole':
        return Quadrupole(name=name, length=length, inherit=inherit)
    elif element_type == 'Sextupole':
        return Sextupole(name=name, length=length, inherit=inherit)
    elif element_type == 'Octupole':
        return Octupole(name=name, length=length, inherit=inherit)
    elif element_type == 'RFCavity':
        return RFCavity(name=name, length=length, inherit=inherit)
    elif element_type == 'CrabCavity':
        return CrabCavity(name=name, length=length, inherit=inherit)
    elif element_type == 'Kicker':
        return Kicker(name=name, length=length, inherit=inherit)
    elif element_type == 'Monitor':
        return Monitor(name=name, length=length, inherit=inherit)
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
    branch_specs: dict[str, str] = field(default_factory=dict)  # Branch specifications: 'ring' or 'linac'

    def __post_init__(self):
        """Initialize the lattice with a name and branches."""
        if not self.name:
            raise ValueError("Lattice must have a name.")
        if not isinstance(self.branches, dict):
            raise TypeError("Branches must be a dictionary with branch names as keys and lists of elements as values.")
        if self.root_branch_name and self.root_branch_name not in self.branches:
            raise ValueError(f"Root branch '{self.root_branch_name}' must be present in the branches.")

    def add_element(self, element: Element, check_consistency: bool = True):
        """Add an element definition to the lattice.
        
        Args:
            element: The element to add to the lattice.
            check_consistency: Whether to validate element consistency before adding.
                              Defaults to True for safety, can be set to False for
                              performance or when adding incomplete elements.
        
        Raises:
            TypeError: If element is not an Element instance.
            ValueError: If element name already exists or consistency check fails.
        """
        if not isinstance(element, Element):
            raise TypeError("Element must be an instance of Element.")
        if element.name in self.elements:
            raise ValueError(f"Element '{element.name}' already exists in the lattice.")
        
        # Perform consistency check if requested
        if check_consistency:
            try:
                element.check_consistency()
            except Exception as e:
                raise ValueError(f"Element '{element.name}' failed consistency check: {e}") from e
        
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


    
    def add_branch(self, branch_name: str, elements: list[Element] | list[str] | None = None, branch_type: str = "linac"):
        """Add a new branch to the lattice.
        Accepts either a list of Element objects or a list of element names.
        
        Args:
            branch_name: Name of the branch to add.
            elements: List of Element objects or element names.
            branch_type: Type of branch, either 'ring' or 'linac' (default: 'linac').
        """
        if not branch_name:
            raise ValueError("Branch name cannot be empty.")
        if elements is None:
            elements = []
        if not isinstance(elements, list):
            raise TypeError("Elements must be a list of Element objects or element names.")
        if branch_name in self.branches:
            raise ValueError(f"Branch '{branch_name}' already exists in the lattice.")
        if branch_type not in ['ring', 'linac']:
            raise ValueError("Branch type must be either 'ring' or 'linac'.")
        
        self.branches[branch_name] = []
        self.branch_specs[branch_name] = branch_type
        
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

    def get_total_path_length(self, branch_name: str = None) -> float:
        """Calculate the total path length of a branch or the root branch.
        
        Args:
            branch_name: Name of the branch to calculate length for. 
                        If None, uses the root branch.
        
        Returns:
            Total path length in meters.
            
        Raises:
            ValueError: If branch doesn't exist or no root branch is set.
        """
        if branch_name is None:
            if not self.root_branch_name:
                raise ValueError("No root branch set and no branch specified.")
            branch_name = self.root_branch_name
        
        if branch_name not in self.branches:
            raise ValueError(f"Branch '{branch_name}' does not exist in the lattice.")
        
        total_length = 0.0
        for element_name in self.branches[branch_name]:
            if element_name in self.elements:
                element = self.elements[element_name]
                total_length += element.length
        
        return total_length

    def get_all_branch_lengths(self) -> dict[str, float]:
        """Calculate the total path length for all branches in the lattice.
        
        Returns:
            Dictionary mapping branch names to their total lengths.
        """
        branch_lengths = {}
        for branch_name in self.branches:
            branch_lengths[branch_name] = self.get_total_path_length(branch_name)
        return branch_lengths
    
    def set_branch_type(self, branch_name: str, branch_type: str):
        """Set the type specification for a branch.
        
        Args:
            branch_name: Name of the branch.
            branch_type: Type of branch, either 'ring' or 'linac'.
        """
        if branch_name not in self.branches:
            raise ValueError(f"Branch '{branch_name}' does not exist in the lattice.")
        if branch_type not in ['ring', 'linac']:
            raise ValueError("Branch type must be either 'ring' or 'linac'.")
        
        self.branch_specs[branch_name] = branch_type
    
    def get_branch_type(self, branch_name: str) -> str:
        """Get the type specification for a branch.
        
        Args:
            branch_name: Name of the branch.
            
        Returns:
            Branch type ('ring' or 'linac').
        """
        if branch_name not in self.branches:
            raise ValueError(f"Branch '{branch_name}' does not exist in the lattice.")
        
        return self.branch_specs.get(branch_name, 'linac')
    
    def get_all_branch_specs(self) -> dict[str, str]:
        """Get all branch specifications.
        
        Returns:
            Dictionary mapping branch names to their types.
        """
        result = {}
        for branch_name in self.branches:
            result[branch_name] = self.get_branch_type(branch_name)
        return result
    
    # Element selection functions
    def _select_elements_by_type(self, element_type: str, branch_name: str = None) -> list[Element]:
        """Select elements by their type.
        
        Args:
            element_type: Type of elements to select (e.g., 'Drift', 'Quadrupole').
            branch_name: Name of the branch to search in. If None, uses root branch.
        
        Returns:
            List of elements matching the type criteria.
        """
        if branch_name is None:
            if not self.root_branch_name:
                raise ValueError("No root branch set and no branch specified.")
            branch_name = self.root_branch_name
        
        if branch_name not in self.branches:
            raise ValueError(f"Branch '{branch_name}' does not exist in the lattice.")
        
        selected_elements = []
        for element_name in self.branches[branch_name]:
            element = self.elements[element_name]
            if element.type == element_type:
                selected_elements.append(element)
        
        return selected_elements
    
    def _select_elements_by_name_pattern(self, name_pattern: str, branch_name: str = None, use_regex: bool = False) -> list[Element]:
        """Select elements by name pattern.
        
        Args:
            name_pattern: Pattern to match element names against.
            branch_name: Name of the branch to search in. If None, uses root branch.
            use_regex: Whether to treat name_pattern as a regular expression.
        
        Returns:
            List of elements matching the name pattern.
        """
        import re
        
        if branch_name is None:
            if not self.root_branch_name:
                raise ValueError("No root branch set and no branch specified.")
            branch_name = self.root_branch_name
        
        if branch_name not in self.branches:
            raise ValueError(f"Branch '{branch_name}' does not exist in the lattice.")
        
        selected_elements = []
        for element_name in self.branches[branch_name]:
            element = self.elements[element_name]
            if use_regex:
                if re.search(name_pattern, element.name):
                    selected_elements.append(element)
            else:
                if name_pattern in element.name:
                    selected_elements.append(element)
        
        return selected_elements
    
    def _select_elements_by_inheritance(self, inherit_name: str, branch_name: str = None) -> list[Element]:
        """Select elements that inherit from a specific prototype element.
        
        Args:
            inherit_name: Name of the prototype element that selected elements should inherit from.
            branch_name: Name of the branch to search in. If None, uses root branch.
        
        Returns:
            List of elements that inherit from the specified prototype.
        """
        if branch_name is None:
            if not self.root_branch_name:
                raise ValueError("No root branch set and no branch specified.")
            branch_name = self.root_branch_name
        
        if branch_name not in self.branches:
            raise ValueError(f"Branch '{branch_name}' does not exist in the lattice.")
        
        selected_elements = []
        for element_name in self.branches[branch_name]:
            element = self.elements[element_name]
            if element.inherit == inherit_name:
                selected_elements.append(element)
        
        return selected_elements
    
    def _select_elements_by_position_range(self, start_position: float, end_position: float, branch_name: str = None) -> list[Element]:
        """Select elements within a specific position range along the beamline.
        
        Args:
            start_position: Starting position in meters.
            end_position: Ending position in meters.
            branch_name: Name of the branch to search in. If None, uses root branch.
        
        Returns:
            List of elements whose position ranges overlap with the specified range.
        """
        if branch_name is None:
            if not self.root_branch_name:
                raise ValueError("No root branch set and no branch specified.")
            branch_name = self.root_branch_name
        
        if branch_name not in self.branches:
            raise ValueError(f"Branch '{branch_name}' does not exist in the lattice.")
        
        if start_position > end_position:
            raise ValueError("Start position must be less than or equal to end position.")
        
        selected_elements = []
        current_position = 0.0
        
        for element_name in self.branches[branch_name]:
            element = self.elements[element_name]
            element_start = current_position
            element_end = current_position + element.length
            
            # Check if element overlaps with the specified range
            if element_end >= start_position and element_start <= end_position:
                selected_elements.append(element)
            
            current_position = element_end
        
        return selected_elements
    
    def _select_elements_by_index_range(self, start_index: int, end_index: int, branch_name: str = None) -> list[Element]:
        """Select elements within a specific index range in the branch.
        
        Args:
            start_index: Starting index (inclusive, 0-based).
            end_index: Ending index (inclusive, 0-based).
            branch_name: Name of the branch to search in. If None, uses root branch.
        
        Returns:
            List of elements within the specified index range.
        """
        if branch_name is None:
            if not self.root_branch_name:
                raise ValueError("No root branch set and no branch specified.")
            branch_name = self.root_branch_name
        
        if branch_name not in self.branches:
            raise ValueError(f"Branch '{branch_name}' does not exist in the lattice.")
        
        branch_elements = self.branches[branch_name]
        
        if start_index < 0 or end_index >= len(branch_elements) or start_index > end_index:
            raise ValueError("Invalid index range.")
        
        selected_elements = []
        for i in range(start_index, end_index + 1):
            element_name = branch_elements[i]
            element = self.elements[element_name]
            selected_elements.append(element)
        
        return selected_elements
    
    def _find_element_position(self, element_name: str, branch_name: str) -> tuple[int, float]:
        """Find the index and starting position of an element in a branch.
        
        Args:
            element_name: Name of the element to find.
            branch_name: Name of the branch to search in.
            
        Returns:
            Tuple of (element_index, start_position)
            
        Raises:
            ValueError: If element is not found in the branch.
        """
        branch_elements = self.branches[branch_name]
        current_position = 0.0
        
        for i, elem_name in enumerate(branch_elements):
            element = self.elements[elem_name]
            if elem_name == element_name:
                return i, current_position
            current_position += element.length
        
        raise ValueError(f"Element '{element_name}' not found in branch '{branch_name}'.")
    
    def _calculate_ring_positions(self, branch_name: str, reference_element: str, start_offset: float, end_offset: float) -> tuple[float, float]:
        """Calculate absolute positions for ring topology with wrap-around support.
        
        Args:
            branch_name: Name of the branch.
            reference_element: Name of the reference element.
            start_offset: Start offset from reference element (can be negative).
            end_offset: End offset from reference element (can be negative).
            
        Returns:
            Tuple of (start_position, end_position) in absolute coordinates.
        """
        ref_idx, ref_pos = self._find_element_position(reference_element, branch_name)
        total_length = self.get_total_path_length(branch_name)
        
        # Calculate absolute positions with wrap-around for rings
        abs_start = (ref_pos + start_offset) % total_length
        abs_end = (ref_pos + end_offset) % total_length
        
        return abs_start, abs_end
    
    def _select_elements_in_ring_range(self, branch_name: str, start_pos: float, end_pos: float, candidate_elements: list) -> list:
        """Select elements in a ring topology, handling wrap-around cases.
        
        Args:
            branch_name: Name of the branch.
            start_pos: Start position (absolute).
            end_pos: End position (absolute).
            candidate_elements: List of (index, element) tuples to filter.
            
        Returns:
            Filtered list of (index, element) tuples.
        """
        branch_elements = self.branches[branch_name]
        total_length = self.get_total_path_length(branch_name)
        filtered_elements = []
        current_position = 0.0
        
        for i, element_name in enumerate(branch_elements):
            element = self.elements[element_name]
            element_start = current_position
            element_end = current_position + element.length
            
            # Check if this element is in our candidate list
            is_candidate = any(candidate_idx == i for candidate_idx, candidate_elem in candidate_elements)
            if not is_candidate:
                current_position = element_end
                continue
            
            # Handle wrap-around case for rings
            if start_pos <= end_pos:
                # Normal case: no wrap-around
                if element_end >= start_pos and element_start <= end_pos:
                    for candidate_idx, candidate_elem in candidate_elements:
                        if candidate_idx == i:
                            filtered_elements.append((i, candidate_elem))
                            break
            else:
                # Wrap-around case: selection crosses the 0 point
                if element_end >= start_pos or element_start <= end_pos:
                    for candidate_idx, candidate_elem in candidate_elements:
                        if candidate_idx == i:
                            filtered_elements.append((i, candidate_elem))
                            break
            
            current_position = element_end
        
        return filtered_elements

    def select_elements(self, 
                        element_type: str = None,
                        name_pattern: str = None,
                        inherit_name: str = None,
                        position_range: tuple[float, float] = None,
                        relative_position_range: tuple[str, float, float] = None,
                        index_range: tuple[int, int] = None,
                        branch_name: str = None,
                        use_regex: bool = False) -> list[Element]:
        """Select elements by multiple criteria (AND operation).
        
        Args:
            element_type: Type of elements to select.
            name_pattern: Pattern to match element names against.
            inherit_name: Name of prototype element that selected elements should inherit from.
            position_range: Tuple of (start_position, end_position) in meters.
            relative_position_range: Tuple of (element_name, start_offset, end_offset) for relative positioning.
                                   Offsets are relative to the start of the named element.
                                   Negative values allow back-tracing from the element.
            index_range: Tuple of (start_index, end_index) for element indices.
            branch_name: Name of the branch to search in. If None, uses root branch.
            use_regex: Whether to treat name_pattern as a regular expression.
        
        Returns:
            List of elements matching ALL specified criteria.
        """
        if branch_name is None:
            if not self.root_branch_name:
                raise ValueError("No root branch set and no branch specified.")
            branch_name = self.root_branch_name
        
        if branch_name not in self.branches:
            raise ValueError(f"Branch '{branch_name}' does not exist in the lattice.")
        
        # Start with all elements in the branch
        candidate_elements = []
        branch_elements = self.branches[branch_name]
        
        for i, element_name in enumerate(branch_elements):
            element = self.elements[element_name]
            candidate_elements.append((i, element))
        
        # Apply each filter criterion
        if element_type is not None:
            candidate_elements = [(i, elem) for i, elem in candidate_elements if elem.type == element_type]
        
        if name_pattern is not None:
            import re
            if use_regex:
                candidate_elements = [(i, elem) for i, elem in candidate_elements if re.search(name_pattern, elem.name)]
            else:
                candidate_elements = [(i, elem) for i, elem in candidate_elements if name_pattern in elem.name]
        
        if inherit_name is not None:
            candidate_elements = [(i, elem) for i, elem in candidate_elements if elem.inherit == inherit_name]
        
        if index_range is not None:
            start_idx, end_idx = index_range
            if start_idx < 0 or end_idx >= len(branch_elements) or start_idx > end_idx:
                raise ValueError("Invalid index range.")
            candidate_elements = [(i, elem) for i, elem in candidate_elements if start_idx <= i <= end_idx]
        
        if position_range is not None:
            start_pos, end_pos = position_range
            if start_pos > end_pos:
                raise ValueError("Start position must be less than or equal to end position.")
            
            # Calculate positions for filtering
            filtered_elements = []
            current_position = 0.0
            
            for i, element_name in enumerate(branch_elements):
                element = self.elements[element_name]
                element_start = current_position
                element_end = current_position + element.length
                
                # Check if this element is in our candidate list and overlaps with position range
                for candidate_idx, candidate_elem in candidate_elements:
                    if candidate_idx == i and element_end >= start_pos and element_start <= end_pos:
                        filtered_elements.append((i, candidate_elem))
                        break
                
                current_position = element_end
            
            candidate_elements = filtered_elements
        
        if relative_position_range is not None:
            ref_element, start_offset, end_offset = relative_position_range
            
            try:
                # Find the reference element and calculate absolute positions
                if self.branch_specs.get(branch_name, 'linac') == 'ring':
                    # For rings, handle wrap-around
                    abs_start, abs_end = self._calculate_ring_positions(branch_name, ref_element, start_offset, end_offset)
                    candidate_elements = self._select_elements_in_ring_range(branch_name, abs_start, abs_end, candidate_elements)
                else:
                    # For linacs, use simple offset calculation
                    ref_idx, ref_pos = self._find_element_position(ref_element, branch_name)
                    abs_start = ref_pos + start_offset
                    abs_end = ref_pos + end_offset
                    
                    if abs_start > abs_end:
                        raise ValueError("Start position must be less than or equal to end position for linac topology.")
                    
                    # Filter elements using absolute positions
                    filtered_elements = []
                    current_position = 0.0
                    
                    for i, element_name in enumerate(branch_elements):
                        element = self.elements[element_name]
                        element_start = current_position
                        element_end = current_position + element.length
                        
                        # Check if this element is in our candidate list and overlaps with position range
                        for candidate_idx, candidate_elem in candidate_elements:
                            if candidate_idx == i and element_end >= abs_start and element_start <= abs_end:
                                filtered_elements.append((i, candidate_elem))
                                break
                        
                        current_position = element_end
                    
                    candidate_elements = filtered_elements
                    
            except ValueError as e:
                raise ValueError(f"Error in relative position filtering: {e}")

        # Return only the elements (not the indices)
        return [elem for i, elem in candidate_elements]
    
    def get_element_positions(self, branch_name: str = None) -> list[tuple[Element, float, float]]:
        """Get the position information for all elements in a branch.
        
        Args:
            branch_name: Name of the branch to get positions for. If None, uses root branch.
        
        Returns:
            List of tuples containing (element, start_position, end_position) for each element.
        """
        if branch_name is None:
            if not self.root_branch_name:
                raise ValueError("No root branch set and no branch specified.")
            branch_name = self.root_branch_name
        
        if branch_name not in self.branches:
            raise ValueError(f"Branch '{branch_name}' does not exist in the lattice.")
        
        element_positions = []
        current_position = 0.0
        
        for element_name in self.branches[branch_name]:
            element = self.elements[element_name]
            start_position = current_position
            end_position = current_position + element.length
            element_positions.append((element, start_position, end_position))
            current_position = end_position
        
        return element_positions

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