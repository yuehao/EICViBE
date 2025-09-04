from .element import Element
from eicvibe.models.base import PhysicsBaseModel
from pydantic import Field, field_validator
from typing import Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

class Sextupole(Element):
    """Sextupole element with Pydantic validation.
    
    A sextupole magnet provides nonlinear focusing forces for chromaticity correction
    and nonlinear dynamics control in beam optics.
    """
    type: str = Field(default='Sextupole', description="Element type")
    plot_color: str = Field(default='C2', description="Color for plotting")
    plot_height: float = Field(default=0.7, ge=0.0, le=2.0, description="Height in beamline plot")
    plot_cross_section: float = Field(default=0.6, ge=0.0, le=5.0, description="Cross-section width")
    
    @field_validator('length')
    @classmethod
    def validate_length(cls, v):
        """Validate sextupole length is non-negative."""
        if v < 0:
            raise ValueError("Length of a sextupole element must be non-negative.")
        return v
    
    @field_validator('type')
    @classmethod
    def validate_type(cls, v):
        """Validate element type is correct."""
        if v != 'Sextupole':
            raise ValueError("Type of a sextupole element must be 'Sextupole'.")
        return v
    
    def _check_element_specific_consistency(self) -> bool:
        """Sextupole-specific consistency checks.
        
        Validates sextupole parameters for physical reasonableness.
        Uses flexible validation to allow incremental parameter construction.
        
        Returns:
            bool: True if consistent, False otherwise
        """
        mm_group = self.get_parameter_group("MagneticMultipoleP")
        if mm_group is None:
            # Sextupole can exist without parameters initially
            return True
        
        kn2 = mm_group.get_parameter("kn2")
        if kn2 is None:
            # Could warn here, but allow for incremental parameter addition
            return True
        
        # Additional validation for sextupole-specific physics constraints
        try:
            kn2_val = float(kn2)
            # Reasonable sextupole strength limits (m^-3)
            if abs(kn2_val) > 10000.0:  # Very strong sextupole
                import warnings
                warnings.warn(f"Sextupole strength kn2={kn2_val} m^-3 is very high")
        except (ValueError, TypeError):
            return False
            
        return True

    def plot_in_beamline(self, ax, s_start, normalized_strength=None):
        '''Plot the sextupole element in the beamline, using a rectangle box.'''
        height = self.plot_height
        ax.add_patch(
            Rectangle((s_start, -height/2), self.length, height, angle=0.0, ec=self.plot_color,
                      fc=self.plot_color, alpha=0.8, lw=1)
        )
        return s_start + self.length
    
    def plot_in_floorplan(self, ax, entrance_coords, tangent_vector):
        '''Plot the sextupole element in the floor plan, using a rectangle.'''
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