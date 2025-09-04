from .element import Element
from eicvibe.models.base import PhysicsBaseModel
from pydantic import Field, field_validator
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np

class Monitor(Element):
    """Monitor element with Pydantic validation.
    
    A beam position monitor (BPM) or other diagnostic element used to measure
    beam properties. Typically has very small but non-zero length.
    """
    type: str = Field(default='Monitor', description="Element type")
    plot_color: str = Field(default='k', description="Color for plotting")
    plot_height: float = Field(default=0.25, ge=0.0, le=2.0, description="Height in beamline plot")
    length: float = Field(default=0.0, ge=0.0, description="Monitor length (usually small)")
    
    @field_validator('type')
    @classmethod
    def validate_type(cls, v):
        """Validate element type is correct."""
        if v != 'Monitor':
            raise ValueError("Type of a Monitor element must be 'Monitor'.")
        return v
    
    def _check_element_specific_consistency(self) -> bool:
        """Monitor-specific consistency checks.
        
        Monitors are generally always consistent as they are passive diagnostic elements.
        
        Returns:
            bool: Always True for monitors
        """
        return True
    
    def plot_in_beamline(self, ax, s_start, normalized_strength=None):
        '''Plot the Monitor element as a short black vertical line.'''
        ax.plot([s_start + self.length/2.0, s_start + self.length/2.0], [-self.plot_height/2, self.plot_height/2], color=self.plot_color, lw=1)
        return s_start
    
    def plot_in_floorplan(self, ax, entrance_coords, tangent_vector):
        '''Plot the Monitor element as a short black line perpendicular to the beam.'''
        if len(entrance_coords) != 2 or len(tangent_vector) != 2:
            raise ValueError("Entrance coordinates and tangent vector must be 2D.")
        
        exit_coords = (entrance_coords[0] + self.length * tangent_vector[0],
                       entrance_coords[1] + self.length * tangent_vector[1])
        ax.plot([entrance_coords[0], exit_coords[0]], [entrance_coords[1], exit_coords[1]], 'k-', lw=1)

        mid_coords = ((entrance_coords[0] + exit_coords[0]) / 2, (entrance_coords[1] + exit_coords[1]) / 2)
        angle = np.arctan2(tangent_vector[1], tangent_vector[0])
        # Perpendicular direction
        perp_angle = angle + np.pi/2
        dx = (self.plot_height/2) * np.cos(perp_angle)
        dy = (self.plot_height/2) * np.sin(perp_angle)
        x0, y0 = mid_coords[0] - dx, mid_coords[1] - dy
        x1, y1 = mid_coords[0] + dx, mid_coords[1] + dy
        ax.plot([x0, x1], [y0, y1], color=self.plot_color, lw=1)
        return exit_coords, tangent_vector 