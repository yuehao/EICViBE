# Implementation of the quad module for the EICVibe machine portal.
from eicvibe.machine_portal.element import Element
from eicvibe.machine_portal.parameter_group import ParameterGroup
from eicvibe.models.base import PhysicsBaseModel
from pydantic import Field, field_validator
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

class Quadrupole(Element):
    """Quadrupole element with Pydantic validation.
    
    A quadrupole magnet provides focusing/defocusing forces in one transverse
    direction and opposite forces in the perpendicular direction.
    """
    type: str = Field(default='Quadrupole', description="Element type")
    plot_color: str = Field(default='C1', description="Color for plotting")
    plot_height: float = Field(default=0.6, ge=0.0, le=2.0, description="Height in beamline plot")
    plot_cross_section: float = Field(default=0.8, ge=0.0, le=5.0, description="Cross-section width")
    
    @field_validator('length')
    @classmethod
    def validate_length(cls, v):
        """Validate quadrupole length is positive."""
        if v < 0:
            raise ValueError("Length of a quadrupole element must be positive.")
        return v
    
    @field_validator('type')
    @classmethod
    def validate_type(cls, v):
        """Validate element type is correct."""
        if v != 'Quadrupole':
            raise ValueError("Type of a quadrupole element must be 'Quadrupole'.")
        return v
    
    def _check_element_specific_consistency(self) -> bool:
        """Quadrupole-specific consistency checks.
        
        Validates that the quadrupole has appropriate MagneticMultipoleP parameters.
        For flexible validation during construction, this uses warnings rather than
        strict enforcement.
        
        Returns:
            bool: True if consistent, False otherwise
        """
        mm_group = self.get_parameter_group("MagneticMultipoleP")
        if mm_group is None:
            # Flexible validation - quadrupole can exist without parameters initially
            return True
        
        kn1 = mm_group.get_parameter("kn1")
        if kn1 is None:
            # Could warn here, but allow for incremental parameter addition
            return True
        
        # Additional validation for quadrupole-specific physics constraints
        try:
            kn1_val = float(kn1)
            # Reasonable quadrupole strength limits (m^-2)
            if abs(kn1_val) > 1000.0:  # Very strong quadrupole
                import warnings
                warnings.warn(f"Quadrupole strength kn1={kn1_val} m^-2 is very high")
        except (ValueError, TypeError):
            return False
            
        return True
    
    def plot_in_beamline(self, ax, s_start, normalized_strength=None):
        """Plot the quadrupole element in the beamline.
        
        Uses a rectangular box to represent the quadrupole. Positive k1 (focusing in X)
        is plotted above the axis, negative k1 (defocusing in X) below the axis.
        
        Args:
            ax: Matplotlib axes object
            s_start: Starting s-coordinate
            normalized_strength: Optional normalization factor
            
        Returns:
            float: End s-coordinate
        """
        k1 = self.get_parameter("MagneticMultipoleP", "kn1")
        height = self.plot_height
        
        # Default to positive (focusing) if k1 not set or not a number
        try:
            k1_val = float(k1) if k1 is not None else 0.0
        except (ValueError, TypeError):
            k1_val = 0.0
            
        if k1_val >= 0:
            # Focusing quadrupole (above axis)
            ax.add_patch(
                Rectangle((s_start, 0), self.length, height, angle=0.0, 
                         ec=self.plot_color, fc=self.plot_color, alpha=0.8, lw=1)
            )
        else:
            # Defocusing quadrupole (below axis)
            ax.add_patch(
                Rectangle((s_start, -height), self.length, height, angle=0.0, 
                         ec=self.plot_color, fc=self.plot_color, alpha=0.8, lw=1)
            )
        return s_start + self.length
    
    def plot_in_floorplan(self, ax, entrance_coords, tangent_vector):
        """Plot the quadrupole element in the floor plan, using a rectangle to represent the quadrupole.
        Args:
              ax: Matplotlib Axes object to plot on.
              entrance_coords: (x, y) coordinates of the entrance point
              tangent_vector: (dx, dy) vector indicating the direction of the drift representing cosine and sine of the angle with the x-axis.
        Returns:
              exit_coords: (x, y) coordinates of the exit point
              tangent_vector: (dx, dy) vector indicating the direction of the exit point
        """
        # Draw a rectangle representing the quadrupole
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