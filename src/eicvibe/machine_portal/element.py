# define elements of accelerator
# its parameter should be in a ParameterGroup object.
# The only exceptions are the name, type and the length of the element.
# Current the type contains:
## ACKicker : Time varying kicker element, Allowed parameter groups: 'MagneticMultipoleP', 'ApertureP', 'RFP', 'MetaP'
## BeamBeam : Colliding beam interaction element, Allowed parameter groups: 'BeamBeamP', 'ApertureP', 'MetaP'
## Bend : Dipole bending magnet, Allowed parameter groups: 'BendP', 'ApertureP', 'MetaP', 'MagneticMultipoleP'
## CrabCavity : RF crab cavity element, Allowed parameter groups: 'RFP', 'ApertureP', 'MagneticMultipoleP', 'MetaP'
## Drift : Drift space element, Allowed parameter groups: 'ApertureP', 'MetaP'
## EGun : Electron gun element, Allowed parameter groups: 'EGunP', 'ApertureP', 'MetaP'
## Instrument : Diagnostic element, Allowed parameter groups: 'InstrumentP', 'ApertureP', 'MetaP'
## Kicker : Static kicker element, Allowed parameter groups: 'KickerP', 'ApertureP', 'MetaP'
## Marker : Marker element, Allowed parameter groups: 'ApertureP', 'MetaP'
## Match : Orbit/Twiss/Dispersion matching element, Allowed parameter groups: 'MatchP', 'ApertureP', 'MetaP'
## Multipole : Multipole element, Allowed parameter groups: 'MagneticMultipoleP', 'ApertureP', 'MetaP'
## Octupole : Octupole element, Allowed parameter groups: 'MagneticMultipoleP', 'ApertureP', 'MetaP'
## Quadrupole : Quadrupole element, Allowed parameter groups: 'MagneticMultipoleP', 'ApertureP', 'MetaP'
## RFCavity : RF cavity element, Allowed parameter groups: 'RFP', 'ApertureP', 'MagneticMultipoleP', 'MetaP'
## Sextupole : Sextupole element, Allowed parameter groups: 'MagneticMultipoleP', 'ApertureP', 'MetaP'
## Solenoid : Solenoid element, Allowed parameter groups: 'SolenoidP', 'ApertureP', 'MetaP'
## Taylor : Taylor map element, Allowed parameter groups: 'TaylorP', 'ApertureP', 'MetaP'
## Undulator : Undulator element, Allowed parameter groups: 'UndulatorP', 'ApertureP', 'MetaP'

# Define the Element class for general accelerator elements and 
# each element type that inherits from it.



from . parameter_group import ParameterGroup
from dataclasses import dataclass, field
import os
import yaml


def _load_allowed_groups_from_yaml():
    yaml_path = os.path.join(os.path.dirname(__file__), 'elements.yaml')
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"elements.yaml not found at {yaml_path}")
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("elements.yaml is malformed: root should be a mapping")
    all_groups = set(data.get('All', {}).get('group', []))
    allowed = {}
    for elem_type, v in data.items():
        if elem_type == 'All':
            continue
        groups = set(v.get('group', [])) | all_groups
        allowed[elem_type] = list(groups)
    return allowed

try:
    element_type_allowed_groups = _load_allowed_groups_from_yaml()
except Exception as e:
    raise RuntimeError(f"Failed to load allowed parameter groups from elements.yaml: {e}")


@dataclass
class Element:
    """Base class for accelerator elements."""
    name: str
    type: str
    length: float = 0.0
    parameters: list[ParameterGroup] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize the element with a name, type, length and parameters."""
        if self.parameters is None:
            self.parameters = []

    def check_parameter_group(self, group_type: str, allowed_groups: list[str]) -> bool:
        """Check if the parameter group type is allowed for the element type."""
        if group_type not in allowed_groups:
            raise ValueError(f"Parameter group '{group_type}' is not allowed for element type '{self.type}'.")
        return True
        

    def add_parameter_group(self, group: ParameterGroup):
        """Add a parameter group to the element."""
        if not isinstance(group, ParameterGroup):
            raise TypeError("Parameter group must be an instance of ParameterGroup.")
        if self.check_parameter_group(group.type, element_type_allowed_groups.get(self.type, [])):
            self.parameters.append(group)
        else:
            raise ValueError(f"Parameter group '{group.type}' is not allowed for element type '{self.type}'.")
        
    def get_parameter_group(self, group_type: str) -> ParameterGroup | None:
        """Get a parameter group by type."""
        for group in self.parameters:
            if group.type == group_type:
                return group
        return None
    
    

    # add a parameter to a specific group with specific group type; check if the group already exist, if not, create a new group
    def add_parameter(self, group_type: str, parameter_name: str, value: str | float | int | list[float] | list[int]):
        """Add a parameter to a specific group."""
        group = self.get_parameter_group(group_type)
        if group is None:
            group = ParameterGroup(name=group_type, type=group_type)
            self.add_parameter_group(group)
        group.add_parameter(parameter_name, value)

    def get_parameter(self, group_type: str, parameter_name: str) -> str | float | int | list[float] | list[int] | None:
        """Get a parameter value by group type and parameter name."""
        group = self.get_parameter_group(group_type)
        if group is not None:
            return group.get_parameter(parameter_name)
        return None

        
    def remove_parameter(self, group_type: str, name: str):
        """Remove a parameter from a specific group."""
        group = self.get_parameter_group(group_type)
        if group is not None:
            group.remove_parameter(name)
    
    def get_length(self) -> float:
        """Get the length of the element."""
        return self.length
    def set_length(self, length: float):
        """Set the length of the element."""
        if length <= 0:
            raise ValueError("Length must be positive.")
        self.length = length
        
    def get_type(self) -> str:
        """Get the type of the element."""
        return self.type
    
    def get_name(self) -> str:
        """Get the name of the element."""
        return self.name
    
    def set_name(self, name: str):
        """Set the name of the element."""
        if not name:
            raise ValueError("Name cannot be empty.")
        self.name = name
    
    def __str__(self): 
        """String representation of the element."""
        return f"Element(name={self.name}, type={self.type}, length={self.length}, parameters={self.parameters})"
    def __repr__(self):
        """String representation of the element."""
        return self.__str__()
    
    # Convert element to yaml dictionary format.
    def to_yaml_dict(self) -> dict:
        """Convert the element to a dictionary in the format:
        element_type:
              name: element_name
              length: number
              parameter_group_1:
                      parameter_name: parameter_value
        """
        result = {}
        element_dict = {}
        element_dict['name'] = self.name
        element_dict['length'] = self.length
        
        # Add parameter groups using the ParameterGroup.to_yaml_dict() method
        for group in self.parameters:
            element_dict[group.type] = group.to_yaml_dict()
            
        result[self.type] = element_dict
        return result
    

    
    # Define a method to check consistance of the element, each element type should implement its own check.
    def check_consistency(self):
        pass # To be implemented by subclasses

    # Define a plot method to visualize the element in a beamline floor view. The input should be matplotlib Axes object, the entrance coordinates and the tangent vector.
    # The output should be the coordinates of the exit point and the tangent vector, Each element type should implement its own plot method.
    def plot_in_floorplan(self, ax, entrance_coords, tangent_vector):
        return entrance_coords, tangent_vector  # Default implementation does nothing, to be overridden by subclasses
    
    # Define a plot method to visualize the element in a 1-D beamline view, represented by boxes.  The input should be matplotlib Axes object, the initial s coordinate, and the optional normailzed strength.
    # If the normalized strength is not provided, all boxes will be full heigth; otherwise the height will be scaled by the normalizd strength.
    # The output should be the existing s coordinate
    def plot_in_beamline(self, ax, s_start, normalized_strength=None):
        """Plot the element in a 1-D beamline view."""
        return s_start + self.length  # Default implementation does nothing, to be overridden by subclasses
    
    # Expremental plotting function for 3-D visualization of the element, to be implemented by subclasses.
    def plot_in_3d(self, ax, entrance_coords, tangent_vector):
        return entrance_coords, tangent_vector  # Default implementation does nothing, to be overridden by subclasses
    

        


    

        
    
    



        


        

    



