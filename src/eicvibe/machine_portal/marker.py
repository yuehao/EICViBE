# Implementation of the marker element for the EICVibe machine portal.
from eicvibe.machine_portal.element import Element
from eicvibe.machine_portal.parameter_group import ParameterGroup
from dataclasses import dataclass, field

@dataclass
class Marker(Element):
    """Marker element."""
    type: str = 'Marker'
    
    def __post_init__(self):
        """Initialize the marker element with a name, type, length and parameters."""
        super().__post_init__()
        if self.length != 0:
            raise ValueError("Length of a marker element must be zero.")
        if self.type != 'Marker':
            raise ValueError("Type of a marker element must be 'Marker'.")