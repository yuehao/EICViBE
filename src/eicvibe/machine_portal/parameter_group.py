# define class ParameterGroup for elements.
# Each parameter group has a name, a group type, and a list of parametername (string) and value (string, number or list of number) pairs .
# Parameter group also allow subgroups, which are also ParameterGroup objects.

from pydantic import Field, field_validator, model_validator
from typing import Dict, List, Union, Any, Optional
from ..models.base import PhysicsBaseModel
from ..models.parameter_groups import PARAMETER_GROUP_MODELS, create_parameter_group_model
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


class ParameterGroup(PhysicsBaseModel):
    """
    Pydantic model for accelerator element parameter groups.
    
    This model provides automatic validation of parameter names against
    the allowed parameters defined in parameters.yaml, while maintaining
    full backward compatibility with the existing API.
    """
    
    name: str = Field(..., min_length=1, description="Parameter group name")
    type: str = Field(..., min_length=1, description="Parameter group type")
    parameters: Dict[str, Union[str, float, int, List[float], List[int]]] = Field(
        default_factory=dict,
        description="Parameter name-value pairs"
    )
    subgroups: List['ParameterGroup'] = Field(
        default_factory=list,
        description="Nested parameter subgroups"
    )

    @field_validator('type')
    @classmethod
    def validate_parameter_type(cls, v):
        """Validate parameter group type against allowed types."""
        # Load from parameters.yaml for validation
        if v not in parameter_group_allowed_parameters:
            # Allow unknown types for backward compatibility, but issue warning
            import warnings
            warnings.warn(f"Unknown parameter group type: {v}. Consider adding to parameters.yaml")
        return v
    
    @field_validator('parameters')
    @classmethod
    def validate_parameter_values(cls, v, info):
        """Validate parameter values against group type specifications."""
        if info.data and 'type' in info.data:
            group_type = info.data['type']
            if group_type in parameter_group_allowed_parameters:
                allowed_params = parameter_group_allowed_parameters[group_type]
                for param_name in v.keys():
                    if param_name not in allowed_params:
                        raise ValueError(
                            f"Parameter '{param_name}' not allowed in group type '{group_type}'. "
                            f"Allowed parameters: {list(allowed_params.keys())}"
                        )
        return v
    
    @model_validator(mode='after')
    def validate_specialized_parameters(self):
        """Apply specialized validation for known parameter group types."""
        if self.type in PARAMETER_GROUP_MODELS:
            try:
                # Create specialized model to validate parameters
                specialized_model = create_parameter_group_model(self.type, **self.parameters)
                # If validation passes, store any normalized values back
                self.parameters.update(specialized_model.model_dump(exclude_unset=True))
            except Exception as e:
                # Don't fail completely for backward compatibility, but warn
                import warnings
                warnings.warn(f"Specialized validation failed for {self.type}: {e}")
        return self

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

    def get_allowed_parameters(self) -> List[str]:
        """Get the list of allowed parameters for this parameter group type."""
        allowed_params = parameter_group_allowed_parameters.get(self.type, {})
        return list(allowed_params.keys())

    @staticmethod
    def get_allowed_parameters_for_type(group_type: str) -> List[str]:
        """Get the list of allowed parameters for a given parameter group type."""
        allowed_params = parameter_group_allowed_parameters.get(group_type, {})
        return list(allowed_params.keys())

    @staticmethod
    def get_all_parameter_groups() -> List[str]:
        """Get the list of all defined parameter group types."""
        return list(parameter_group_allowed_parameters.keys())

    def add_parameter(self, name: str, value: Union[str, float, int, List[float], List[int]]):
        """Add a parameter to the group with validation."""
        self.validate_parameter(name)
        self.parameters[name] = value
        
        # Trigger re-validation with specialized model if available
        if self.type in PARAMETER_GROUP_MODELS:
            try:
                # Validate with specialized model
                create_parameter_group_model(self.type, **self.parameters)
            except Exception as e:
                # Remove the parameter if specialized validation fails
                del self.parameters[name]
                raise ValueError(f"Parameter '{name}' with value '{value}' failed validation: {e}")

    def add_subgroup(self, subgroup: 'ParameterGroup'):
        """Add a subgroup to the group."""
        if not isinstance(subgroup, ParameterGroup):
            raise TypeError("Subgroup must be an instance of ParameterGroup")
        self.subgroups.append(subgroup)

    def get_parameter(self, name: str) -> Union[str, float, int, List[float], List[int], None]:
        """Get a parameter value by name."""
        return self.parameters.get(name, None)
    
    def get_subgroup_by_name(self, name: str) -> Optional['ParameterGroup']:
        """Get a subgroup by name."""
        for subgroup in self.subgroups:
            if subgroup.name == name:
                return subgroup
        return None
    
    def get_subgroup_by_type(self, type_: str) -> Optional['ParameterGroup']:
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

    