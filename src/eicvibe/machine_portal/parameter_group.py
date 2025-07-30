# define class ParameterGroup for elements.
# Each parameter group has a name, a group type, and a list of parametername (string) and value (string, number or list of number) pairs .
# Parameter group also allow subgroups, which are also ParameterGroup objects.

from dataclasses import dataclass, field
import os
import yaml

def _load_allowed_parameters_from_yaml():
    """Load the allowed parameters for each parameter group type from parameters.yaml."""
    yaml_path = os.path.join(os.path.dirname(__file__), 'parameters.yaml')
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"parameters.yaml not found at {yaml_path}")
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("parameters.yaml is malformed: root should be a mapping")
    return data

try:
    parameter_group_allowed_parameters = _load_allowed_parameters_from_yaml()
except Exception as e:
    raise RuntimeError(f"Failed to load allowed parameters from parameters.yaml: {e}")


@dataclass
class ParameterGroup:
    name: str
    type: str
    parameters: dict[str, str | float | int | list[float] | list[int]] = field(default_factory=dict)
    subgroups: list['ParameterGroup'] = field(default_factory=list)

    def __post_init__(self):
        """Initialize the parameter group with a name, type, parameters and subgroups."""
        if self.parameters is None:
            self.parameters = {}
        if self.subgroups is None:
            self.subgroups = []

    def validate_parameter(self, name: str) -> bool:
        """Validate if a parameter is allowed for this parameter group type."""
        allowed_params = parameter_group_allowed_parameters.get(self.type, {})
        if not allowed_params:
            # If parameter group type is not defined in parameters.yaml, allow any parameter (backward compatibility)
            return True
        
        if name not in allowed_params:
            raise ValueError(f"Parameter '{name}' is not allowed for parameter group type '{self.type}'. "
                           f"Allowed parameters: {list(allowed_params.keys())}")
        return True

    def get_allowed_parameters(self) -> list[str]:
        """Get the list of allowed parameters for this parameter group type."""
        allowed_params = parameter_group_allowed_parameters.get(self.type, {})
        return list(allowed_params.keys())

    @staticmethod
    def get_allowed_parameters_for_type(group_type: str) -> list[str]:
        """Get the list of allowed parameters for a given parameter group type."""
        allowed_params = parameter_group_allowed_parameters.get(group_type, {})
        return list(allowed_params.keys())

    @staticmethod
    def get_all_parameter_groups() -> list[str]:
        """Get the list of all defined parameter group types."""
        return list(parameter_group_allowed_parameters.keys())

    def add_parameter(self, name: str, value: str | float | int | list[float] | list[int]):
        """Add a parameter to the group with validation."""
        self.validate_parameter(name)
        self.parameters[name] = value

    def add_subgroup(self, subgroup: 'ParameterGroup'):
        """Add a subgroup to the group."""
        self.subgroups.append(subgroup)

    def get_parameter(self, name: str) -> str | float | int | list[float] | list[int] | None:
        """Get a parameter value by name."""
        return self.parameters.get(name, None)
    
    def get_subgroup_by_name(self, name: str) -> 'ParameterGroup | None':
        """Get a subgroup by name."""
        for subgroup in self.subgroups:
            if subgroup.name == name:
                return subgroup
        return None
    
    def get_subgroup_by_type(self, type_: str) -> 'ParameterGroup | None':
        """Get a subgroup by type."""
        for subgroup in self.subgroups:
            if subgroup.type == type_:
                return subgroup
        return None
    

    
    def remove_parameter(self, name: str):
        """Remove a parameter from the group."""
        if name in self.parameters:
            del self.parameters[name]

    def remove_subgroup(self, group_name: str):
        """Remove a subgroup by name."""
        self.subgroups = [subgroup for subgroup in self.subgroups if subgroup.name != group_name]
        

    
    
    def __str__(self):
        """String representation of the parameter group."""
        return f"ParameterGroup(name={self.name}, type={self.type}, parameters={self.parameters}, subgroups={self.subgroups})"
    def __repr__(self):
        """String representation of the parameter group."""
        return self.__str__()
    
    def to_dict(self) -> dict:  
        """Convert the parameter group to a dictionary."""
        return {
            'name': self.name,
            'type': self.type,
            'parameters': self.parameters,
            'subgroups': [subgroup.to_dict() for subgroup in self.subgroups]
        }
    
    def to_yaml_dict(self) -> dict:
        """Convert the parameter group to a dictionary in YAML format.
        Handles recursive cases where parameter groups can contain subgroups.
        Format:
        parameter_group1:
               parameter_name: parameter_value
               sub_group1:
                       parameter_name: parameter_value
        """
        result: dict = {}
        
        # Add direct parameters
        for param_name, param_value in self.parameters.items():
            result[param_name] = param_value
            
        # Add subgroups recursively
        for subgroup in self.subgroups:
            subgroup_dict = subgroup.to_yaml_dict()
            result[subgroup.type] = subgroup_dict
            
        return result
    @classmethod
    def from_dict(cls, data: dict) -> 'ParameterGroup':
        """Create a ParameterGroup from a dictionary."""
        name = data.get('name', '')
        type_ = data.get('type', '')
        parameters = data.get('parameters', {})
        subgroups_data = data.get('subgroups', [])
        subgroups = [cls.from_dict(subgroup) for subgroup in subgroups_data]
        return cls(name=name, type=type_, parameters=parameters, subgroups=subgroups)
    
# Example usage:

    