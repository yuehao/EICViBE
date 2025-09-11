from .element import Element
from eicvibe.models.base import PhysicsBaseModel
from pydantic import Field, field_validator
from typing import Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

class Kicker(Element):
    """Kicker element with Pydantic validation.
    
    A fast kicker magnet used for beam injection, extraction, or orbit correction.
    Provides transverse deflection with precise timing control.
    """
    type: str = Field(default='Kicker', description="Element type")
    plot_color: str = Field(default='C0', description="Color for plotting")
    plot_height: float = Field(default=0.4, ge=0.0, le=2.0, description="Height in beamline plot")
    
    @field_validator('length')
    @classmethod
    def validate_length(cls, v):
        """Validate kicker length is non-negative."""
        if v < 0:
            raise ValueError("Length of a Kicker element must be non-negative.")
        return v
    
    @field_validator('type')
    @classmethod
    def validate_type(cls, v):
        """Validate element type is correct."""
        if v != 'Kicker':
            raise ValueError("Type of a Kicker element must be 'Kicker'.")
        return v
    
    def _check_element_specific_consistency(self) -> bool:
        """Kicker-specific consistency checks.
        
        Validates kicker parameters for physical reasonableness if present.
        
        Returns:
            bool: True if consistent, False otherwise
        """
        kicker_group = self.get_parameter_group("KickerP")
        if kicker_group is None:
            # Kicker can exist without parameters initially
            return True
            
        # Check kick angle parameter if present
        hkick = kicker_group.get_parameter("hkick")
        vkick = kicker_group.get_parameter("vkick")
        
        for kick_param, kick_name in [(hkick, "hkick"), (vkick, "vkick")]:
            if kick_param is not None:
                try:
                    kick_val = float(kick_param)
                    # Reasonable kick angle limits (radians)
                    if abs(kick_val) > 1.0:  # Very large kick
                        import warnings
                        warnings.warn(f"Kicker {kick_name}={kick_val} rad is very large")
                except (ValueError, TypeError):
                    return False
                    
        return True
    
    def plot_in_beamline(self, ax, s_start, normalized_strength=None):
        '''Plot the Kicker element in the beamline, using a rhombus spanning the full length.'''
        # Create rhombus points spanning the full length
        points = np.array([
            [s_start, 0],  # left center
            [s_start + self.length/2, self.plot_height/2],  # top center
            [s_start + self.length, 0],  # right center
            [s_start + self.length/2, -self.plot_height/2],  # bottom center
        ])
        rhombus = Polygon(points, edgecolor=self.plot_color, facecolor=self.plot_color, alpha=0.5, lw=1)
        ax.add_patch(rhombus)
        return s_start + self.length
    
    def plot_in_floorplan(self, ax, entrance_coords, tangent_vector):
        '''Plot the Kicker element in the floor plan, using a rhombus spanning the full length.'''
        if len(entrance_coords) != 2 or len(tangent_vector) != 2:
            raise ValueError("Entrance coordinates and tangent vector must be 2D.")
        exit_coords = (entrance_coords[0] + self.length * tangent_vector[0],
                       entrance_coords[1] + self.length * tangent_vector[1])
        angle = np.arctan2(tangent_vector[1], tangent_vector[0])
        
        # Create rhombus points spanning the full length, rotated by the angle
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        points = np.array([
            [entrance_coords[0], entrance_coords[1]],  # left center
            [entrance_coords[0] + self.length/2 * tangent_vector[0] + self.plot_height/2 * np.cos(angle-np.pi/2),
             entrance_coords[1] + self.length/2 * tangent_vector[1] + self.plot_height/2 * np.sin(angle-np.pi/2)],  # top center
            [exit_coords[0], exit_coords[1]],  # right center
            [entrance_coords[0] + self.length/2 * tangent_vector[0] - self.plot_height/2 * np.cos(angle-np.pi/2),
             entrance_coords[1] + self.length/2 * tangent_vector[1] - self.plot_height/2 * np.sin(angle-np.pi/2)],  # bottom center
        ])
        rhombus = Polygon(points, edgecolor=self.plot_color, facecolor=self.plot_color, alpha=0.5, lw=1)
        ax.add_patch(rhombus)
        
        return exit_coords, tangent_vector 