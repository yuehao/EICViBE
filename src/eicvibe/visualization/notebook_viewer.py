"""
Notebook-friendly visualization interface for EICViBE.

Provides matplotlib-based interactive widgets that work in Jupyter notebooks.
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, RadioButtons
import numpy as np
from typing import Optional, Dict, Any
import logging

from .lattice_viewer import (BeamlinePlotter, TwissPlotter, BPMPlotter, FloorPlanPlotter)

logger = logging.getLogger(__name__)


class NotebookViewer:
    """
    Interactive notebook viewer for EICViBE lattices.
    
    Engine-agnostic interface that works with EICViBE Lattice objects
    and generic data structures from any simulation engine.
    
    Uses matplotlib widgets for interaction within Jupyter notebooks.
    """
    
    def __init__(self, lattice=None, twiss=None, 
                 bpm_data=None, branch_name: str = "FODO"):
        """
        Initialize notebook viewer.
        
        Args:
            lattice: EICViBE Lattice object
            twiss: Twiss data (any object with s, betx, bety, dx attributes)
            bpm_data: Dictionary of BPM data {bpm_name: {'x': array, 'y': array, 's': float}}
            branch_name: Branch name to display
        """
        self.lattice = lattice
        self.twiss = twiss
        self.bpm_data = bpm_data
        self.branch_name = branch_name
        self.bpm_vs_turn = True
        
        # Create plotters
        self.beamline_plotter = BeamlinePlotter()
        self.floorplan_plotter = FloorPlanPlotter()
        self.twiss_plotter = TwissPlotter()
        self.bpm_plotter = BPMPlotter()
    
    def show_beamline(self):
        """Display beamline layout."""
        if self.lattice is None:
            print("No lattice data available")
            return
        
        fig, ax = self.beamline_plotter.create_figure()
        self.beamline_plotter.plot(self.lattice, self.branch_name)
        plt.show()
    
    def show_floorplan(self):
        """Display floor plan."""
        if self.lattice is None:
            print("No lattice data available")
            return
        
        fig, ax = self.floorplan_plotter.create_figure()
        self.floorplan_plotter.plot(self.lattice, self.branch_name)
        plt.show()
    
    def show_twiss(self):
        """Display Twiss parameters."""
        if self.twiss is None:
            print("No Twiss data available")
            return
        
        fig, axes = self.twiss_plotter.create_figure()
        self.twiss_plotter.plot(self.twiss)
        plt.show()
    
    def show_bpm(self, plot_vs_turn: bool = True):
        """
        Display BPM data.
        
        Args:
            plot_vs_turn: If True, plot vs turn number; if False, plot vs s position
        """
        if self.bpm_data is None or len(self.bpm_data) == 0:
            print("No BPM data available")
            return
        
        fig, axes = self.bpm_plotter.create_figure()
        self.bpm_plotter.plot(self.bpm_data, plot_vs_turn=plot_vs_turn)
        plt.show()
    
    def show_all(self):
        """Display all available plots in a grid."""
        # Determine layout based on available data
        n_plots = 0
        if self.lattice is not None:
            n_plots += 2  # beamline + floorplan
        if self.twiss is not None:
            n_plots += 1
        if self.bpm_data is not None and len(self.bpm_data) > 0:
            n_plots += 1
        
        if n_plots == 0:
            print("No data available to plot")
            return
        
        # Create figure with subplots
        if n_plots <= 2:
            fig = plt.figure(figsize=(14, 6))
            gs = fig.add_gridspec(1, n_plots)
        else:
            fig = plt.figure(figsize=(16, 12))
            gs = fig.add_gridspec(2, 2)
        
        plot_idx = 0
        
        # Beamline
        if self.lattice is not None:
            ax = fig.add_subplot(gs[plot_idx])
            self.beamline_plotter.ax = ax
            self.beamline_plotter.fig = fig
            self.beamline_plotter.plot(self.lattice, self.branch_name)
            plot_idx += 1
            
            # Floorplan
            ax = fig.add_subplot(gs[plot_idx])
            ax.set_aspect('equal')
            self.floorplan_plotter.ax = ax
            self.floorplan_plotter.fig = fig
            self.floorplan_plotter.plot(self.lattice, self.branch_name)
            plot_idx += 1
        
        # Twiss
        if self.twiss is not None:
            if n_plots == 3:
                ax1 = fig.add_subplot(gs[plot_idx, :])
            else:
                ax1 = fig.add_subplot(gs[plot_idx])
            ax2 = None  # Twiss plotter creates its own subplots
            
            # Create Twiss plot in current axes
            from matplotlib.gridspec import GridSpecFromSubplotSpec
            inner_gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[plot_idx], hspace=0.05)
            ax1 = fig.add_subplot(inner_gs[0])
            ax2 = fig.add_subplot(inner_gs[1], sharex=ax1)
            
            self.twiss_plotter.axes = [ax1, ax2]
            self.twiss_plotter.fig = fig
            self.twiss_plotter.plot(self.twiss)
            plot_idx += 1
        
        # BPM
        if self.bpm_data is not None and len(self.bpm_data) > 0:
            from matplotlib.gridspec import GridSpecFromSubplotSpec
            inner_gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[plot_idx], hspace=0.05)
            ax1 = fig.add_subplot(inner_gs[0])
            ax2 = fig.add_subplot(inner_gs[1], sharex=ax1)
            
            self.bpm_plotter.axes = [ax1, ax2]
            self.bpm_plotter.fig = fig
            self.bpm_plotter.plot(self.bpm_data, plot_vs_turn=True)
        
        plt.tight_layout()
        plt.show()
    
    def update_data(self, lattice=None, twiss=None, 
                    bpm_data=None, branch_name: Optional[str] = None):
        """
        Update data.
        
        Engine-agnostic interface - accepts data from any simulation engine.
        
        Args:
            lattice: EICViBE Lattice object
            twiss: Twiss data (any object with s, betx, bety, dx attributes)
            bpm_data: Dictionary of BPM data {bpm_name: {'x': array, 'y': array, 's': float}}
            branch_name: Branch name to display
        """
        if lattice is not None:
            self.lattice = lattice
        if twiss is not None:
            self.twiss = twiss
        if bpm_data is not None:
            self.bpm_data = bpm_data
        if branch_name is not None:
            self.branch_name = branch_name


def create_notebook_viewer(lattice=None, twiss=None,
                           bpm_data=None, branch_name: str = "FODO") -> NotebookViewer:
    """
    Create a notebook viewer instance.
    
    Engine-agnostic interface - works with data from any simulation engine.
    
    Args:
        lattice: EICViBE Lattice object
        twiss: Twiss data (any object with s, betx, bety, dx, etc. attributes)
        bpm_data: Dictionary of BPM data {bpm_name: {'x': array, 'y': array, 's': float}}
        branch_name: Branch name to display
    
    Returns:
        NotebookViewer instance
    
    Example:
        >>> viewer = create_notebook_viewer(lattice=fodo, twiss=twiss, bpm_data=bpm_data)
        >>> viewer.show_all()
    """
    return NotebookViewer(
        lattice=lattice,
        twiss=twiss,
        bpm_data=bpm_data,
        branch_name=branch_name
    )
