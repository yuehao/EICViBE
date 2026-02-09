# Implementation of the bend element for the EICVibe machine portal.
from eicvibe.machine_portal.element import Element
from eicvibe.machine_portal.parameter_group import ParameterGroup
from pydantic import Field
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class Bend(Element):
    """Bend element with enhanced validation."""
    
    type: str = Field(default='Bend', description="Element type (always 'Bend')")
    length: float = Field(ge=0.0, description="Bend length must be non-negative")
    plot_color: str = Field(default='C0', description="Color for plotting")
    plot_height: float = Field(default=0.7, description="Height of the bend element in the beamline")
    plot_cross_section: float = Field(default=0.5, description="Cross section for floor plan")
    
    def model_post_init(self, __context) -> None:
        """Post-initialization validation (replaces __post_init__)."""
        super().model_post_init(__context)
        # Additional Bend-specific initialization if needed
        pass
    
    def _check_element_specific_consistency(self):
        """Bend-specific consistency checks. The angle parameter of BendP parameter group must be set."""
        # Only check if we have parameters - during construction this might not be set yet
        bend_group = self.get_parameter_group("BendP")
        if bend_group is not None and bend_group.get_parameter("angle") is None:
            raise ValueError("Bend element with BendP group must have an angle parameter set.")
    
    def _validate_bend_geometry(self):
        """Validate bend geometry for enhanced Pydantic integration."""
        bend_group = self.get_parameter_group('BendP')
        if bend_group is not None:
            # Use the enhanced bend geometry validation from parameter group
            bend_group.validate_bend_geometry_with_length(self.length)
    
    def plot_in_beamline(self, ax, s_start, normalized_strength=None):
        '''Plot the bend element in the beamline, using an square box to represent the bend.'''
        if normalized_strength is None:
            height = self.plot_height
        else:
            angle = self.get_parameter("BendP", "angle")
            height = self.plot_height * angle / normalized_strength
        ax.add_patch(
                        Rectangle((s_start, -height/2), self.length, height, angle=0.0, ec=self.plot_color,
                                  fc=self.plot_color, alpha=0.8, lw=1)
                    )
        return s_start + self.length
        
    
    def plot_in_floorplan(self, ax, entrance_coords, tangent_vector, return_shape=False):
        """Plot the bend element in the floor plan, using an center arc and an annular section to represent a bend magnet.
        Args:
              ax: Matplotlib Axes object to plot on.
              entrance_coords: (x, y) coordinates of the entrance point
              tangent_vector: (dx, dy) vector indicating the direction of the drift representing cosine and sine of the angle with the x-axis.
              return_shape: If True, return shape polygon for hit detection
        Returns:
              exit_coords: (x, y) coordinates of the exit point
              tangent_vector: (dx, dy) vector indicating the direction of the exit point
              shape_polygon: (if return_shape=True) List of (x,y) tuples defining annular sector polygon
        """
        # Draw an arc representing the bend
        if len(entrance_coords) != 2 or len(tangent_vector) != 2:
            raise ValueError("Entrance coordinates and tangent vector must be 2D.")
        
        angle = self.get_parameter("BendP", "angle")
        if self.length == 0:  # If length is zero, we use wedge to represent a zero length bend, tangent vector need to be rotated by angle.
            exit_coords = entrance_coords
            extra_rotation = 0
            if angle < 0:
                extra_rotation = 180
            ax.plot(entrance_coords[0], entrance_coords[1], marker=(3, 0, np.arctan2(tangent_vector[1], tangent_vector[0])*180/np.pi + extra_rotation+ angle * 90/np.pi),
                    markersize=10, linestyle='None', color=self.plot_color, alpha=0.5)  # Mark the entrance point
            new_tangent_vector = (tangent_vector[0] * np.cos(angle) - tangent_vector[1] * np.sin(angle),
                                  tangent_vector[0] * np.sin(angle) + tangent_vector[1] * np.cos(angle))
            if return_shape:
                # Zero-length bend: no shape, return empty polygon
                return exit_coords, new_tangent_vector, []
            return exit_coords, new_tangent_vector
        
        radius = self.length / abs(angle)
        entrance_angle = np.arctan2(tangent_vector[1], tangent_vector[0])
        exit_angle = entrance_angle + angle
        center_angle=entrance_angle + np.pi / 2 * np.sign(angle)
        center_x = entrance_coords[0] + radius * np.cos(center_angle)
        center_y = entrance_coords[1] + radius * np.sin(center_angle)
        new_tangent_vector = (tangent_vector[0] * np.cos(angle) - tangent_vector[1] * np.sin(angle),
                                  tangent_vector[0] * np.sin(angle) + tangent_vector[1] * np.cos(angle))
        

        def draw_arc(ax, center, radius, start_angle, end_angle, color=self.plot_color, linestyle='-', alpha=1.0):
            """Draw an arc on the given axes."""
            theta = np.linspace(start_angle, end_angle, 30)
            x = center[0] + radius * np.cos(theta)
            y = center[1] + radius * np.sin(theta)
            ax.plot(x, y, color=color, ls=linestyle, alpha=alpha, lw=1)
        
        def draw_end(ax, center, radius1, radius2, end_angle, color=self.plot_color, alpha=1.0):
            """Draw the end of the bend element."""
            x1 = center[0] + radius1 * np.cos(end_angle)
            y1 = center[1] + radius1 * np.sin(end_angle)
            x2 = center[0] + radius2 * np.cos(end_angle)
            y2 = center[1] + radius2 * np.sin(end_angle)
            ax.plot([x1, x2], [y1, y2], color=color, ls='-', alpha=alpha, lw=1)

        # Draw the arc
        # width of the dipole is set to be 0.02 of the radius, this is a reasonable value for most cases. 
        dipole_width = self.plot_cross_section /2.0
        inv_center_angle = center_angle - np.pi
        draw_arc(ax, (center_x, center_y), radius, inv_center_angle, inv_center_angle+angle, color='k', linestyle='--', alpha=0.2)
        draw_arc(ax, (center_x, center_y), radius + dipole_width, inv_center_angle, inv_center_angle+angle, color=self.plot_color, linestyle='-', alpha=1.0)
        draw_arc(ax, (center_x, center_y), radius - dipole_width, inv_center_angle, inv_center_angle+angle, color=self.plot_color, linestyle='-', alpha=1.0)

        draw_end(ax, (center_x, center_y), radius + dipole_width, radius - dipole_width, inv_center_angle + angle, color=self.plot_color, alpha=1.0)
        draw_end(ax, (center_x, center_y), radius + dipole_width, radius - dipole_width, inv_center_angle, color=self.plot_color, alpha=1.0)

        exit_coords = (center_x + radius * np.cos(inv_center_angle+angle),
                       center_y + radius * np.sin(inv_center_angle+angle))
        
        # Create rectangle approximation with 4 corners for hit detection
        if return_shape:
            # Four corners of the annular sector (approximated as rectangle)
            # Corner 1: entrance, inner radius
            corner1_x = center_x + (radius - dipole_width) * np.cos(inv_center_angle)
            corner1_y = center_y + (radius - dipole_width) * np.sin(inv_center_angle)
            
            # Corner 2: entrance, outer radius
            corner2_x = center_x + (radius + dipole_width) * np.cos(inv_center_angle)
            corner2_y = center_y + (radius + dipole_width) * np.sin(inv_center_angle)
            
            # Corner 3: exit, outer radius
            corner3_x = center_x + (radius + dipole_width) * np.cos(inv_center_angle + angle)
            corner3_y = center_y + (radius + dipole_width) * np.sin(inv_center_angle + angle)
            
            # Corner 4: exit, inner radius
            corner4_x = center_x + (radius - dipole_width) * np.cos(inv_center_angle + angle)
            corner4_y = center_y + (radius - dipole_width) * np.sin(inv_center_angle + angle)
            
            shape_polygon = [
                (corner1_x, corner1_y),
                (corner2_x, corner2_y),
                (corner3_x, corner3_y),
                (corner4_x, corner4_y)
            ]
        
        if return_shape:
            return exit_coords, new_tangent_vector, shape_polygon
        return exit_coords, new_tangent_vector

        

            




        
    
    
    