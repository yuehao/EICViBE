# Create a lattice class to represent a lattice structure in the machine portal.
# Lattice consists of a list of branches, each branch is a list of elements.
# One branch is marked as the root branch, 
# Lattice can be expanded, starting from root branch, to a list of elements, so that tracking can be done through the lattice.

from eicvibe.machine_portal.element import Element
from eicvibe.machine_portal.parameter_group import ParameterGroup
from dataclasses import dataclass, field

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
    
    def add_branch(self, branch: 'Branch'):
        """Add another branch to this branch."""
        if not isinstance(branch, Branch):
            raise TypeError("Branch must be an instance of Branch.")
        if branch.name in [b.name for b in self.elements]:
            raise ValueError(f"Branch '{branch.name}' already exists in the elements.")
        self.elements.append(branch)



@dataclass
class Lattice:
    """Class representing a lattice structure in the machine portal.
    A lattice consists of multiple branches, each branch has its own name and contains a list of elements.
    One branch is marked as the root branch, """
    name : str 
    branches: dict[str, list[Element]] = field(default_factory=dict)  # Dictionary of branches, where key is branch name and value is a list of elements in that branch
    root_branch_name: str  # Name of the root branch

    def __post_init__(self):
        """Initialize the lattice with a name and branches."""
        if not self.name:
            raise ValueError("Lattice must have a name.")
        if not isinstance(self.branches, dict):
            raise TypeError("Branches must be a dictionary with branch names as keys and lists of elements as values.")
        if self.root_branch_name not in self.branches:
            raise ValueError(f"Root branch '{self.root_branch_name}' must be present in the branches.")

        # Ensure that the root branch is a list of elements
        if not isinstance(self.branches[self.root_branch_name], list):
            raise TypeError(f"Branch '{self.root_branch_name}' must be a list of elements.")
        
        # Set the index of the root branch for easy access
        self.root_branch_index = list(self.branches.keys()).index(self.root_branch_name)
    
    def add_branch(self, branch_name: str, elements: list[Element]):
        """Add a new branch to the lattice."""
        if not branch_name:
            raise ValueError("Branch name cannot be empty.")
        if not isinstance(elements, list):
            raise TypeError("Elements must be a list of Element instances.")
        if branch_name in self.branches:
            raise ValueError(f"Branch '{branch_name}' already exists in the lattice.")
        
        self.branches[branch_name] = elements

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
        self.root_branch_index = list(self.branches.keys()).index(branch_name)
    

    # Lattice expansion
    # This function expands the lattice starting from the root branch. 
    # If elements in the root branch is forked to other branches, then other branches will be added.  
    # If the element has a fork parameter group and the parameter force_expand is set to True,
    # then we will expand the forked branches, the branch name is specified in 'to_line' parameter and the starting element is specified in 'to_ele' parameter in ForkP group
    def expand_lattice(self, force_expand: bool = False) -> list[Element]:
        """Expand the lattice starting from the root branch."""
        expanded_elements = []
        
        for element in self.branches[self.root_branch_name]:
            if not isinstance(element, Element):
                raise TypeError("All elements in the branches must be instances of Element.")
            
            # Add the element to the expanded list
            expanded_elements.append(element)
            
            # Check for ForkP parameter group to expand further
            fork_group = element.get_parameter_group("ForkP")
            if fork_group is not None and (force_expand or fork_group.get_parameter("force_expand", False)):
                to_line = fork_group.get_parameter("to_line")
                to_ele = fork_group.get_parameter("to_ele")
                
                if to_line and to_ele:
                    if to_line in self.branches:
                        # Add elements from the specified branch starting from the to_ele element
                        for forked_element in self.branches[to_line]:
                            if forked_element.name == to_ele:
                                # Start adding from the specified element
                                expanded_elements.append(forked_element)
                                break
                            expanded_elements.append(forked_element)

        return expanded_elements
    