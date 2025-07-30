from .element import Element
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

@dataclass
class RBend(Element):
    """Rectangular Bend element."""
    type: str = 'RBend'
    plot_color: str = 'C0'
    plot_height: float = 0.7
    plot_cross_section: float = 0.5
    
    def __post_init__(self):
        super().__post_init__()
        if self.length < 0:
            raise ValueError("Length of an RBend element must be non-negative.")
        if self.type != 'RBend':
            raise ValueError("Type of an RBend element must be 'RBend'.")
    
    def _check_element_specific_consistency(self):
        """RBend-specific consistency checks including geometric relationships."""
        import math
        
        # Get BendP parameter group
        bend_group = self.get_parameter_group("BendP")
        if bend_group is None:
            raise ValueError("RBend element must have a BendP parameter group.")
        
        # Get parameters
        angle = bend_group.get_parameter("angle")
        chord_length = bend_group.get_parameter("chord_length")
        arc_length = self.length
        
        if angle is None:
            raise ValueError("RBend element must have an angle parameter set.")
        
        # Convert angle to float
        angle = float(angle)
        
       
        
        # Handle geometric consistency between arc_length, angle, and chord_length
        # Relationships: 
        # - radius = arc_length / angle
        # - chord_length = 2 * radius * sin(angle/2)
        
        tolerance = 1e-12
        
        if chord_length is None:
            # Chord length not set - calculate it from arc_length and angle
            if abs(arc_length) < tolerance:
                raise ValueError("RBend must have either non-zero length or non-zero chord_length.")
            if abs(angle) < tolerance:
                # Zero angle: chord length equals arc length
                calculated_chord = arc_length
            else:
                # Non-zero angle: use geometric relationship
                radius = arc_length / angle
                calculated_chord = 2 * radius * math.sin(angle / 2)
            bend_group.add_parameter("chord_length", calculated_chord)
            
        else:
            # Chord length is set - need to handle different scenarios
            chord_length = float(chord_length)
            
            if abs(angle) < tolerance:
                # Zero angle case: arc_length should equal chord_length
                if abs(arc_length) < tolerance and abs(chord_length) < tolerance:
                    raise ValueError("RBend with zero angle must have non-zero length.")
                # Set arc_length to match chord_length for zero angle
                self.length = chord_length
            else:
                # Non-zero angle: calculate arc_length from chord_length and angle
                if abs(arc_length) < tolerance:
                    # Arc length is zero - calculate it from chord_length and angle
                    radius = chord_length / (2 * math.sin(angle / 2))
                    calculated_arc_length = radius * angle
                    self.length = calculated_arc_length
                else:
                    # Both arc_length and chord_length given - chord_length takes precedence
                    radius = chord_length / (2 * math.sin(angle / 2))
                    calculated_arc_length = radius * angle
                    self.length = calculated_arc_length

        
    
    def plot_in_beamline(self, ax, s_start, normalized_strength=None):
        '''Plot the RBend element in the beamline, using a rectangle box.'''
        if normalized_strength is None:
            height = self.plot_height
        else:
            angle = self.get_parameter("BendP", "angle")
            if angle is not None and isinstance(angle, (int, float)):
                height = self.plot_height * abs(float(angle)) / normalized_strength
            else:
                height = self.plot_height
        ax.add_patch(
            Rectangle((s_start, -height/2), self.length, height, angle=0.0, ec=self.plot_color,
                      fc=self.plot_color, alpha=0.8, lw=1)
        )
        return s_start + self.length
    
    def plot_in_floorplan(self, ax, entrance_coords, tangent_vector):
        """Plot the RBend element in the floor plan, using a rectangle."""
        if len(entrance_coords) != 2 or len(tangent_vector) != 2:
            raise ValueError("Entrance coordinates and tangent vector must be 2D.")

        angle = self.get_parameter("BendP", "angle")
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

        
        half_width = self.plot_cross_section /2.0
        inv_center_angle = center_angle - np.pi
        draw_arc(ax, (center_x, center_y), radius, inv_center_angle, inv_center_angle+angle, color='k', linestyle='--', alpha=0.2)
        rec_corner = (entrance_coords[0] + half_width * np.cos(entrance_angle + angle/2 -np.pi/2),
                      entrance_coords[1] + half_width * np.sin(entrance_angle + angle/2-np.pi/2))
        chord_length = 2 * radius * np.sin(angle/2)
        ax.add_patch(
            Rectangle(rec_corner, chord_length, self.plot_cross_section, angle=(angle/2+entrance_angle)*180/np.pi, ec=self.plot_color,
                      fc='none', alpha=1, lw=1)
        )
        exit_coords = (center_x + radius * np.cos(inv_center_angle+angle),
                       center_y + radius * np.sin(inv_center_angle+angle))

        return exit_coords, new_tangent_vector 