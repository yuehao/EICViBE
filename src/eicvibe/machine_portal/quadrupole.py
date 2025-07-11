# Implementation of the quad module for the EICVibe machine portal.
from eicvibe.machine_portal.element import Element
from eicvibe.machine_portal.parameter_group import ParameterGroup
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

@dataclass
class Quadrupole(Element):
    """Quadrupole element."""
    type: str = 'Quadrupole'
    plot_color: str = 'C1'
    plot_height: float = 0.9 # Height of the quadrupole element in the beamline
    plot_cross_section: float = 0.8
    def __post_init__(self):
        """Initialize the quadrupole element with a name, type, length and parameters."""
        super().__post_init__()
        if self.length < 0:
            raise ValueError("Length of a quadrupole element must be positive.")
        if self.type != 'Quadrupole':
            raise ValueError("Type of a quadrupole element must be 'Quadrupole'.")
    
    def check_consistency(self):
        """Check if the quadrupole element is consistent. The K1 parameter of MagneticMultipoleP parameter group must be set."""
        mm_group = self.get_parameter_group("MagneticMultipoleP")
        if mm_group is None or mm_group.get_parameter("K1") is None:
            raise ValueError("Quadrupole element must have a MagneticMultipoleP group with a K1 parameter set.")
        return True
    
    def plot_in_beamline(self, ax, s_start, normalized_strength=None):
        '''Plot the quadrupole element in the beamline, using an square box to represent the quadrupole.'''
    
        k1 = self.get_parameter("MagneticMultipoleP", "K1")
        height = self.plot_height
        if k1 > 0:
            ax.add_patch(
                        Rectangle((s_start, 0), self.length, height, angle=0.0, ec=self.plot_color,
                                  fc=self.plot_color, alpha=0.5, lw=2)
                    )
        else:
            ax.add_patch(
                        Rectangle((s_start, -height), self.length, height, angle=0.0, ec=self.plot_color,
                                  fc=self.plot_color, alpha=0.5, lw=2)
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
                      fc=self.plot_color, alpha=0.5, lw=2)
        )
        
        return exit_coords, tangent_vector