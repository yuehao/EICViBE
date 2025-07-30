from .element import Element
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

@dataclass
class CrabCavity(Element):
    """Crab Cavity element."""
    type: str = 'CrabCavity'
    plot_color: str = 'C4'
    plot_height: float = 0.4
    plot_separation: float = 0.2
    
    def __post_init__(self):
        super().__post_init__()
        if self.length < 0:
            raise ValueError("Length of a CrabCavity element must be non-negative.")
        if self.type != 'CrabCavity':
            raise ValueError("Type of a CrabCavity element must be 'CrabCavity'.")
    
    def _check_element_specific_consistency(self):
        """CrabCavity-specific consistency checks (none required beyond parameter group validation)."""
        pass
    
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