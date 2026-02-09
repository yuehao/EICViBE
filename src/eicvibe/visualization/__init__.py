"""
EICViBE Visualization Module

Provides GUI and notebook interfaces for lattice visualization.
"""

from .lattice_viewer import (
    BeamlinePlotter,
    TwissPlotter,
    BPMPlotter,
    FloorPlanPlotter
)

from .notebook_viewer import NotebookViewer, create_notebook_viewer

# Import GUI components only if PyQt5 is available
try:
    from .gui_app import LatticeViewerGUI, launch_gui
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    LatticeViewerGUI = None
    launch_gui = None

__all__ = [
    # Plotting components
    'BeamlinePlotter',
    'TwissPlotter',
    'BPMPlotter',
    'FloorPlanPlotter',
    
    # Notebook interface
    'NotebookViewer',
    'create_notebook_viewer',
    
    # GUI interface (if available)
    'LatticeViewerGUI',
    'launch_gui',
    'GUI_AVAILABLE',
]
