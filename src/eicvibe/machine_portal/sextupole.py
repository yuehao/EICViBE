from .element import Element
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

@dataclass
class Sextupole(Element):
    """Sextupole element."""
    type: str = 'Sextupole'
    plot_color: str = 'C2'
    plot_height: float = 0.7
    plot_cross_section: float = 0.6
    
    def __post_init__(self):
        super().__post_init__()
        if self.length < 0:
            raise ValueError("Length of a sextupole element must be non-negative.")
        if self.type != 'Sextupole':
            raise ValueError("Type of a sextupole element must be 'Sextupole'.")
    
    def _check_element_specific_consistency(self):
        """Sextupole-specific consistency checks (none required beyond parameter group validation)."""
        mm_group = self.get_parameter_group("MagneticMultipoleP")
        if mm_group is None or mm_group.get_parameter("kn2") is None:
            raise ValueError("Sextupole element must have a MagneticMultipoleP group with a 'kn2' parameter set.")

    def plot_in_beamline(self, ax, s_start, normalized_strength=None):
        '''Plot the sextupole element in the beamline, using a rectangle box.'''
        height = self.plot_height
        ax.add_patch(
            Rectangle((s_start, -height/2), self.length, height, angle=0.0, ec=self.plot_color,
                      fc=self.plot_color, alpha=0.8, lw=1)
        )
        return s_start + self.length
    
    def plot_in_floorplan(self, ax, entrance_coords, tangent_vector):
        '''Plot the sextupole element in the floor plan, using a rectangle.'''
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