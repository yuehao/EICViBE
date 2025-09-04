from .element import Element
from eicvibe.models.base import PhysicsBaseModel
from pydantic import Field, field_validator
from typing import Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

class Octupole(Element):
    """Octupole element with Pydantic validation.
    
    An octupole magnet provides higher-order nonlinear corrections for
    advanced beam dynamics control and aberration compensation.
    """
    type: str = Field(default='Octupole', description="Element type")
    plot_color: str = Field(default='C3', description="Color for plotting")
    plot_height: float = Field(default=0.6, ge=0.0, le=2.0, description="Height in beamline plot")
    plot_cross_section: float = Field(default=0.5, ge=0.0, le=5.0, description="Cross-section width")
    
    @field_validator('length')
    @classmethod
    def validate_length(cls, v):
        """Validate octupole length is non-negative."""
        if v < 0:
            raise ValueError("Length of an octupole element must be non-negative.")
        return v
    
    @field_validator('type')
    @classmethod
    def validate_type(cls, v):
        """Validate element type is correct."""
        if v != 'Octupole':
            raise ValueError("Type of an octupole element must be 'Octupole'.")
        return v
    
    def _check_element_specific_consistency(self) -> bool:
        """Octupole-specific consistency checks.
        
        Validates octupole parameters for physical reasonableness.
        Uses flexible validation to allow incremental parameter construction.
        
        Returns:
            bool: True if consistent, False otherwise
        """
        mm_group = self.get_parameter_group("MagneticMultipoleP")
        if mm_group is None:
            # Octupole can exist without parameters initially
            return True
        
        kn3 = mm_group.get_parameter("kn3")
        if kn3 is None:
            # Could warn here, but allow for incremental parameter addition
            return True
        
        # Additional validation for octupole-specific physics constraints
        try:
            kn3_val = float(kn3)
            # Reasonable octupole strength limits (m^-4)
            if abs(kn3_val) > 100000.0:  # Very strong octupole
                import warnings
                warnings.warn(f"Octupole strength kn3={kn3_val} m^-4 is very high")
        except (ValueError, TypeError):
            return False
            
        return True
        
    
    def plot_in_beamline(self, ax, s_start, normalized_strength=None):
        '''Plot the octupole element in the beamline, using a rectangle box.'''
        height = self.plot_height
        ax.add_patch(
            Rectangle((s_start, -height/2), self.length, height, angle=0.0, ec=self.plot_color,
                      fc=self.plot_color, alpha=0.8, lw=1)
        )
        return s_start + self.length
    
    def plot_in_floorplan(self, ax, entrance_coords, tangent_vector):
        '''Plot the octupole element in the floor plan, using a rectangle.'''
        if len(entrance_coords) != 2 or len(tangent_vector) != 2:
            raise ValueError("Entrance coordinates and tangent vector must be 2D.")
        exit_coords = (entrance_coords[0] + self.length * tangent_vector[0],
                       entrance_coords[1] + self.length * tangent_vector[1])
        angle = np.arctan2(tangent_vector[1], tangent_vector[0])
        half_width = self.plot_cross_section / 2.0
        rec_corner = (entrance_coords[0] + half_width * np.cos(angle-np.pi/2),
                      entrance_coords[1] + half_width * np.sin(angle-np.pi/2))
        ax.add_patch(
            Rectangle(rec_corner, self.length, self.plot_cross_section, angle=angle*180/np.pi, ec=self.plot_color,
                      fc=self.plot_color, alpha=0.8, lw=1)
        )
        return exit_coords, tangent_vector 