# define class ParameterGroup for elements.
# Each parameter group has a name, a group type, and a list of parametername (string) and value (string, number or list of number) pairs .
# Parameter group also allow subgroups, which are also ParameterGroup objects.

from dataclasses import dataclass

@dataclass
class ParameterGroup:
    name: str
    type: str
    parameters: dict[str, str | float | int | list[float] | list[int]] = None
    subgroups: list['ParameterGroup'] = None

    def __post_init__(self):
        """Initialize the parameter group with a name, type, parameters and subgroups."""
        if self.parameters is None:
            self.parameters = {}
        if self.subgroups is None:
            self.subgroups = []
        
        
        

    def add_parameter(self, name: str, value: str | float | int | list[float] | list[int]):
        """Add a parameter to the group."""
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
    @classmethod
    def from_dict(cls, data: dict) -> 'ParameterGroup':
        """Create a ParameterGroup from a dictionary."""
        name = data.get('name')
        type_ = data.get('type')
        parameters = data.get('parameters', {})
        subgroups_data = data.get('subgroups', [])
        subgroups = [cls.from_dict(subgroup) for subgroup in subgroups_data]
        return cls(name=name, type=type_, parameters=parameters, subgroups=subgroups)
    
# Example usage:
if __name__ == "__main__":
    group = ParameterGroup(name="ExampleGroup")
    group.add_parameter("param1", 10)
    group.add_parameter("param2", "value")
    subgroup = ParameterGroup(name="SubGroup")
    subgroup.add_parameter("subparam1", [1.0, 2.0, 3.0])
    group.add_subgroup(subgroup)

    #print(group)
    #print(group.to_dict())
    
    new_group = ParameterGroup.from_dict(group.to_dict())
    print(new_group.to_dict())
    