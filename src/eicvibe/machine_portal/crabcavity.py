from .element import Element
from eicvibe.models.base import PhysicsBaseModel
from pydantic import Field, field_validator
from typing import Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

class CrabCavity(Element):
    """Crab Cavity element with Pydantic validation.
    
    A specialized RF cavity that provides transverse deflection with longitudinal
    dependence, used for bunch rotation and collision optimization.
    """
    type: str = Field(default='CrabCavity', description="Element type")
    plot_color: str = Field(default='C4', description="Color for plotting")
    plot_height: float = Field(default=0.4, ge=0.0, le=2.0, description="Height in beamline plot")
    plot_separation: float = Field(default=0.2, ge=0.0, le=1.0, description="Separation between cavity sections")
    
    @field_validator('length')
    @classmethod
    def validate_length(cls, v):
        """Validate crab cavity length is non-negative."""
        if v < 0:
            raise ValueError("Length of a CrabCavity element must be non-negative.")
        return v
    
    @field_validator('type')
    @classmethod
    def validate_type(cls, v):
        """Validate element type is correct."""
        if v != 'CrabCavity':
            raise ValueError("Type of a CrabCavity element must be 'CrabCavity'.")
        return v
    
    def _check_element_specific_consistency(self) -> bool:
        """CrabCavity-specific consistency checks.
        
        Validates crab cavity parameters for physical reasonableness if present.
        
        Returns:
            bool: True if consistent, False otherwise
        """
        crab_group = self.get_parameter_group("CrabP")
        if crab_group is None:
            # Crab cavity can exist without parameters initially
            return True
            
        # Check voltage parameter if present
        voltage = crab_group.get_parameter("voltage")
        if voltage is not None:
            try:
                voltage_val = float(voltage)
                # Reasonable voltage limits (MV)
                if abs(voltage_val) > 100.0:  # Very high voltage for crab cavity
                    import warnings
                    warnings.warn(f"Crab cavity voltage {voltage_val} MV is very high")
            except (ValueError, TypeError):
                return False
                
        # Check frequency parameter if present
        frequency = crab_group.get_parameter("frequency")
        if frequency is not None:
            try:
                freq_val = float(frequency)
                # Reasonable frequency limits (MHz)
                if freq_val <= 0 or freq_val > 5000.0:
                    import warnings
                    warnings.warn(f"Crab cavity frequency {freq_val} MHz may be unrealistic")
            except (ValueError, TypeError):
                return False
                
        return True
    
    def plot_in_beamline(self, ax, s_start, normalized_strength=None):
        '''Plot the CrabCavity element in the beamline, using two rectangles.'''
        # Upper rectangle
        upper_rect = Rectangle((s_start, self.plot_separation/2), self.length, self.plot_height, 
                               angle=0.0, ec=self.plot_color, fc=self.plot_color, alpha=0.8, lw=1)
        ax.add_patch(upper_rect)
        # Lower rectangle
        lower_rect = Rectangle((s_start, -self.plot_separation/2 - self.plot_height), self.length, self.plot_height, 
                               angle=0.0, ec=self.plot_color, fc=self.plot_color, alpha=0.8, lw=1)
        ax.add_patch(lower_rect)
        return s_start + self.length
    
    def plot_in_floorplan(self, ax, entrance_coords, tangent_vector):
        '''Plot the CrabCavity element in the floor plan, using two rectangles.'''
        if len(entrance_coords) != 2 or len(tangent_vector) != 2:
            raise ValueError("Entrance coordinates and tangent vector must be 2D.")
        exit_coords = (entrance_coords[0] + self.length * tangent_vector[0],
                       entrance_coords[1] + self.length * tangent_vector[1])
        angle = np.arctan2(tangent_vector[1], tangent_vector[0])
        half_separation = self.plot_separation / 2.0
        
        # Upper rectangle
        upper_corner = (entrance_coords[0] + half_separation * np.cos(angle+np.pi/2),
                        entrance_coords[1] + half_separation * np.sin(angle+np.pi/2))
        upper_rect = Rectangle(upper_corner, self.length, self.plot_height, angle=angle*180/np.pi, 
                               ec=self.plot_color, fc=self.plot_color, alpha=0.8, lw=1)
        ax.add_patch(upper_rect)
        
        # Lower rectangle
        lower_corner = (entrance_coords[0] + (half_separation+self.plot_height) * np.cos(angle-np.pi/2),
                        entrance_coords[1] + (half_separation+self.plot_height) * np.sin(angle-np.pi/2))
        lower_rect = Rectangle(lower_corner, self.length, self.plot_height, angle=angle*180/np.pi, 
                               ec=self.plot_color, fc=self.plot_color, alpha=0.8, lw=1)
        ax.add_patch(lower_rect)

        # center line
        ax.plot([entrance_coords[0], exit_coords[0]], [entrance_coords[1], exit_coords[1]], 'k-', lw=1)
        
        return exit_coords, tangent_vector 