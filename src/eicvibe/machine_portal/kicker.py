from .element import Element
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

@dataclass
class Kicker(Element):
    """Kicker element."""
    type: str = 'Kicker'
    plot_color: str = 'C0'
    plot_height: float = 0.4
    
    def __post_init__(self):
        super().__post_init__()
        if self.length < 0:
            raise ValueError("Length of a Kicker element must be non-negative.")
        if self.type != 'Kicker':
            raise ValueError("Type of a Kicker element must be 'Kicker'.")
    
    def _check_element_specific_consistency(self):
        """Kicker-specific consistency checks (none required beyond parameter group validation)."""
        pass
    
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