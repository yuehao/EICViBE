# Implementation of the marker element for the EICVibe machine portal.
from eicvibe.machine_portal.element import Element
from eicvibe.machine_portal.parameter_group import ParameterGroup
from eicvibe.models.base import PhysicsBaseModel
from pydantic import Field, field_validator
from typing import Optional

class Marker(Element):
    """Marker element with Pydantic validation.
    
    A marker is a zero-length element used to mark specific locations
    in the accelerator lattice for reference or measurement purposes.
    """
    type: str = Field(default='Marker', description="Element type")
    
    @field_validator('length')
    @classmethod
    def validate_length(cls, v):
        """Validate marker length is exactly zero."""
        if v != 0:
            raise ValueError("Length of a marker element must be zero.")
        return v
    
    @field_validator('type')
    @classmethod
    def validate_type(cls, v):
        """Validate element type is correct."""
        if v != 'Marker':
            raise ValueError("Type of a marker element must be 'Marker'.")
        return v
        
    def _check_element_specific_consistency(self) -> bool:
        """Marker-specific consistency checks (markers are always consistent).
        
        Returns:
            bool: Always True for markers
        """
        return True