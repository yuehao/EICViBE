from .element import Element
from eicvibe.models.base import PhysicsBaseModel
from pydantic import Field, field_validator
from typing import Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

class RBend(Element):
    """Rectangular Bend element with Pydantic validation.
    
    A rectangular bend magnet where the faces are perpendicular to the design orbit.
    Includes comprehensive bend geometry validation for length, angle, and chord_length.
    """
    type: str = Field(default='RBend', description="Element type")
    plot_color: str = Field(default='C0', description="Color for plotting")
    plot_height: float = Field(default=0.7, ge=0.0, le=2.0, description="Height in beamline plot")
    plot_cross_section: float = Field(default=0.5, ge=0.0, le=5.0, description="Cross-section width")
    
    @field_validator('length')
    @classmethod
    def validate_length(cls, v):
        """Validate RBend length is non-negative."""
        if v < 0:
            raise ValueError("Length of an RBend element must be non-negative.")
        return v
    
    @field_validator('type')
    @classmethod
    def validate_type(cls, v):
        """Validate element type is correct."""
        if v != 'RBend':
            raise ValueError("Type of an RBend element must be 'RBend'.")
        return v
    
    def _check_element_specific_consistency(self) -> bool:
        """RBend-specific consistency checks including geometric relationships.
        
        For RBend, validates bend geometry using the same comprehensive validation
        as regular Bend elements, but with flexible enforcement.
        
        Returns:
            bool: True if consistent, False otherwise
        """
        bend_group = self.get_parameter_group("BendP")
        if bend_group is None:
            # RBend can exist without BendP parameters initially
            return True
        
        # Use the same bend geometry validation as ParameterGroup
        try:
            bend_group.validate_bend_geometry_with_length(self.length)
            return True
        except ValueError:
            # Geometry validation failed
            return False

        
    
    def plot_in_beamline(self, ax, s_start, normalized_strength=None):
        """Plot the RBend element in the beamline.
        
        Uses a rectangular box to represent the bend. Height can be proportional
        to bend strength if normalization is provided.
        
        Args:
            ax: Matplotlib axes object
            s_start: Starting s-coordinate
            normalized_strength: Optional normalization factor for height scaling
            
        Returns:
            float: End s-coordinate
        """
        if normalized_strength is None:
            height = self.plot_height
        else:
            angle = self.get_parameter("BendP", "angle")
            if angle is not None:
                try:
                    angle_val = float(angle)
                    height = self.plot_height * abs(angle_val) / normalized_strength
                except (ValueError, TypeError):
                    height = self.plot_height
            else:
                height = self.plot_height
                
        ax.add_patch(
            Rectangle((s_start, -height/2), self.length, height, angle=0.0, 
                     ec=self.plot_color, fc=self.plot_color, alpha=0.8, lw=1)
        )
        return s_start + self.length
    
    def plot_in_floorplan(self, ax, entrance_coords, tangent_vector):
        """Plot the RBend element in the floor plan.
        
        Uses a rectangle and arc visualization to represent the bend geometry.
        Handles cases where bend parameters may not be fully defined.
        
        Args:
            ax: Matplotlib axes object
            entrance_coords: (x, y) entrance coordinates
            tangent_vector: (dx, dy) direction vector
            
        Returns:
            tuple: (exit_coords, new_tangent_vector)
        """
        if len(entrance_coords) != 2 or len(tangent_vector) != 2:
            raise ValueError("Entrance coordinates and tangent vector must be 2D.")

        angle = self.get_parameter("BendP", "angle")
        if angle is None:
            # No angle defined - treat as drift
            exit_coords = (entrance_coords[0] + self.length * tangent_vector[0],
                          entrance_coords[1] + self.length * tangent_vector[1])
            return exit_coords, tangent_vector
            
        try:
            angle_val = float(angle)
        except (ValueError, TypeError):
            # Invalid angle - treat as drift
            exit_coords = (entrance_coords[0] + self.length * tangent_vector[0],
                          entrance_coords[1] + self.length * tangent_vector[1])
            return exit_coords, tangent_vector
            
        if abs(angle_val) < 1e-12:
            # Zero angle - treat as drift
            exit_coords = (entrance_coords[0] + self.length * tangent_vector[0],
                          entrance_coords[1] + self.length * tangent_vector[1])
            return exit_coords, tangent_vector
            
        # Non-zero angle - proper bend visualization
        radius = self.length / abs(angle_val)
        entrance_angle = np.arctan2(tangent_vector[1], tangent_vector[0])
        exit_angle = entrance_angle + angle_val
        center_angle = entrance_angle + np.pi / 2 * np.sign(angle_val)
        center_x = entrance_coords[0] + radius * np.cos(center_angle)
        center_y = entrance_coords[1] + radius * np.sin(center_angle)
        new_tangent_vector = (tangent_vector[0] * np.cos(angle_val) - tangent_vector[1] * np.sin(angle_val),
                             tangent_vector[0] * np.sin(angle_val) + tangent_vector[1] * np.cos(angle_val))
        
        def draw_arc(ax, center, radius, start_angle, end_angle, color=self.plot_color, linestyle='-', alpha=1.0):
            """Draw an arc on the given axes."""
            theta = np.linspace(start_angle, end_angle, 30)
            x = center[0] + radius * np.cos(theta)
            y = center[1] + radius * np.sin(theta)
            ax.plot(x, y, color=color, ls=linestyle, alpha=alpha, lw=1)

        half_width = self.plot_cross_section / 2.0
        inv_center_angle = center_angle - np.pi
        draw_arc(ax, (center_x, center_y), radius, inv_center_angle, inv_center_angle + angle_val, 
                color='k', linestyle='--', alpha=0.2)
        rec_corner = (entrance_coords[0] + half_width * np.cos(entrance_angle + angle_val/2 - np.pi/2),
                     entrance_coords[1] + half_width * np.sin(entrance_angle + angle_val/2 - np.pi/2))
        chord_length = 2 * radius * np.sin(abs(angle_val)/2)
        ax.add_patch(
            Rectangle(rec_corner, chord_length, self.plot_cross_section, 
                     angle=(angle_val/2 + entrance_angle)*180/np.pi, ec=self.plot_color,
                     fc='none', alpha=1, lw=1)
        )
        exit_coords = (center_x + radius * np.cos(inv_center_angle + angle_val),
                      center_y + radius * np.sin(inv_center_angle + angle_val))

        return exit_coords, new_tangent_vector 