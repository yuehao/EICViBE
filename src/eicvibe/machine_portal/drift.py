# Implementation of the drift detection module for the EICVibe machine portal.
from eicvibe.machine_portal.element import Element
from eicvibe.machine_portal.parameter_group import ParameterGroup
from dataclasses import dataclass, field

@dataclass
class Drift(Element):
    """Drift space element."""
    type: str = 'Drift'
    
    def __post_init__(self):
        """Initialize the drift element with a name, type, length and parameters."""
        super().__post_init__()
        if self.length <= 0:
            raise ValueError("Length of a drift element must be positive.")
        if self.type != 'Drift':
            raise ValueError("Type of a drift element must be 'Drift'.")
    
    def check_consistency(self):
        return True  # Drift elements are always consistent as they have no parameters or subgroups.
    
    # Plotting functions for the drift element, nothing to be done of in-beamline 1-D here as it is just a space.
    # Need simple implement of plot_in_floorplan, limiting 2-D plotting:
    def plot_in_floorplan(self, ax, entrance_coords, tangent_vector):
        """Plot the drift element in the floor plan, using a line segment to represent the drift space.
        Args:
              ax: Matplotlib Axes object to plot on.
              entrance_coords: (x, y) coordinates of the entrance point
              tangent_vector: (dx, dy) vector indicating the direction of the drift representing cosine and sine of the angle with the x-axis.
        Returns:
              exit_coords: (x, y) coordinates of the exit point
              tangent_vector: (dx, dy) vector indicating the direction of the exit point
        """
        # Draw a line representing the drift space
        if len(entrance_coords) != 2 or len(tangent_vector) != 2:
            raise ValueError("Entrance coordinates and tangent vector must be 2D.")
        exit_coords = (entrance_coords[0] + self.length * tangent_vector[0],
                       entrance_coords[1] + self.length * tangent_vector[1])
        ax.plot([entrance_coords[0], exit_coords[0]], [entrance_coords[1], exit_coords[1]], 'k-', lw=1)
        return exit_coords, tangent_vector
    
    # Plotting function for the drift element in 3D, which is a simple line in space.
    def plot_in_3d(self, ax, entrance_coords, tangent_vector):
        """Plot the drift element in 3D."""
        # Draw a line representing the drift space in 3D
        if len(entrance_coords) != 3 or len(tangent_vector) != 3:
            raise ValueError("Entrance coordinates and tangent vector must be 3D.")
        exit_coords = (entrance_coords[0] + self.length * tangent_vector[0],
                       entrance_coords[1] + self.length * tangent_vector[1],
                       entrance_coords[2] + self.length * tangent_vector[2])
        ax.plot([entrance_coords[0], exit_coords[0]], 
                [entrance_coords[1], exit_coords[1]], 
                [entrance_coords[2], exit_coords[2]], 'k-')
        return exit_coords, tangent_vector  
    

