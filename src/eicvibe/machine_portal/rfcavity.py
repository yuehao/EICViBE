from .element import Element
from eicvibe.models.base import PhysicsBaseModel
from pydantic import Field, field_validator
from typing import Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

class RFCavity(Element):
    """RF Cavity element with Pydantic validation.
    
    An RF cavity accelerates particles by applying time-varying electromagnetic fields.
    Used for acceleration and longitudinal focusing in both linacs and rings.
    """
    type: str = Field(default='RFCavity', description="Element type")
    plot_color: str = Field(default='C4', description="Color for plotting")
    plot_height: float = Field(default=0.8, ge=0.0, le=2.0, description="Height in beamline plot")
    
    @field_validator('length')
    @classmethod
    def validate_length(cls, v):
        """Validate RF cavity length is non-negative."""
        if v < 0:
            raise ValueError("Length of an RFCavity element must be non-negative.")
        return v
    
    @field_validator('type')
    @classmethod
    def validate_type(cls, v):
        """Validate element type is correct."""
        if v != 'RFCavity':
            raise ValueError("Type of an RFCavity element must be 'RFCavity'.")
        return v
    
    def _check_element_specific_consistency(self) -> bool:
        """RFCavity-specific consistency checks.
        
        Validates RF cavity parameters for physical reasonableness.
        Uses flexible validation to allow incremental parameter construction.
        
        Returns:
            bool: True if consistent, False otherwise
        """
        rf_group = self.get_parameter_group("RFP")
        if rf_group is None:
            # RF cavity can exist without parameters initially
            return True
            
        # Check voltage parameter if present
        voltage = rf_group.get_parameter("voltage")
        if voltage is not None:
            try:
                voltage_val = float(voltage)
                # Reasonable voltage limits (MV)
                if abs(voltage_val) > 1000.0:  # Very high voltage
                    import warnings
                    warnings.warn(f"RF cavity voltage {voltage_val} MV is very high")
            except (ValueError, TypeError):
                return False
                
        # Check frequency parameter if present
        frequency = rf_group.get_parameter("frequency")
        if frequency is not None:
            try:
                freq_val = float(frequency)
                # Reasonable frequency limits (MHz)
                if freq_val <= 0 or freq_val > 10000.0:
                    import warnings
                    warnings.warn(f"RF frequency {freq_val} MHz may be unrealistic")
            except (ValueError, TypeError):
                return False
                
        return True
    
    def plot_in_beamline(self, ax, s_start, normalized_strength=None):
        '''Plot the RFCavity element in the beamline, using an ellipse.'''
        center_x = s_start + self.length / 2
        ellipse = Ellipse((center_x, 0), width=self.length, height=self.plot_height, 
                          edgecolor=self.plot_color, facecolor=self.plot_color, alpha=0.8, lw=1)
        ax.add_patch(ellipse)
        return s_start + self.length
    
    def plot_in_floorplan(self, ax, entrance_coords, tangent_vector):
        '''Plot the RFCavity element in the floor plan, using an ellipse.'''
        if len(entrance_coords) != 2 or len(tangent_vector) != 2:
            raise ValueError("Entrance coordinates and tangent vector must be 2D.")
        center_x = entrance_coords[0] + self.length / 2 * tangent_vector[0]
        center_y = entrance_coords[1] + self.length / 2 * tangent_vector[1]
        angle = np.arctan2(tangent_vector[1], tangent_vector[0]) * 180 / np.pi
        ellipse = Ellipse((center_x, center_y), width=self.length, height=self.plot_height, 
                          angle=angle, edgecolor=self.plot_color, facecolor=self.plot_color, alpha=0.8, lw=1)
        ax.add_patch(ellipse)
        exit_coords = (entrance_coords[0] + self.length * tangent_vector[0],
                       entrance_coords[1] + self.length * tangent_vector[1])
        return exit_coords, tangent_vector 