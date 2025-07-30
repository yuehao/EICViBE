from .element import Element
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

@dataclass
class RFCavity(Element):
    """RF Cavity element."""
    type: str = 'RFCavity'
    plot_color: str = 'C4'
    plot_height: float = 0.8
    
    def __post_init__(self):
        super().__post_init__()
        if self.length < 0:
            raise ValueError("Length of an RFCavity element must be non-negative.")
        if self.type != 'RFCavity':
            raise ValueError("Type of an RFCavity element must be 'RFCavity'.")
    
    def _check_element_specific_consistency(self):
        """RFCavity-specific consistency checks (none required beyond parameter group validation)."""
        pass
    
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