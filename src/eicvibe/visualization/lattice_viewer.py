"""
Lattice visualization components for EICViBE.

Modular plotting components that can be reused across different interfaces.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional, Dict, Any, List, Tuple
import logging

# Only import Qt components if PyQt5 is available (not needed for notebook viewer)
try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    QT_AVAILABLE = True
except ImportError:
    FigureCanvasQTAgg = None
    QT_AVAILABLE = False

logger = logging.getLogger(__name__)


class BeamlinePlotter:
    """Reusable component for plotting beamline layout."""
    
    def __init__(self, figsize: Tuple[float, float] = (14, 2)):
        """
        Initialize beamline plotter.
        
        Args:
            figsize: Figure size (width, height) in inches
        """
        self.figsize = figsize
        self.fig = None
        self.ax = None
        
    def create_figure(self) -> Tuple[Figure, Any]:
        """Create matplotlib figure and axis."""
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        return self.fig, self.ax
        
    def plot(self, lattice, branch_name: str, start_s: float = 0.0, 
             s_begin: Optional[float] = None, s_end: Optional[float] = None,
             normalized_strength: Optional[float] = None):
        """
        Plot beamline layout.
        
        Args:
            lattice: EICViBE Lattice object
            branch_name: Name of branch to plot
            start_s: Starting s position
            s_begin: Begin position for range selection
            s_end: End position for range selection
            normalized_strength: Normalization factor for element heights
        """
        if self.ax is None:
            self.create_figure()
            
        self.ax.clear()
        lattice.plot_branch_beamline(
            branch_name, 
            ax=self.ax, 
            start_s=start_s,
            s_begin=s_begin,
            s_end=s_end,
            normalized_strength=normalized_strength
        )
        
        if self.fig is not None:
            self.fig.tight_layout()


class TwissPlotter:
    """Reusable component for plotting Twiss parameters."""
    
    def __init__(self, figsize: Tuple[float, float] = (12, 8)):
        """
        Initialize Twiss plotter.
        
        Args:
            figsize: Figure size (width, height) in inches
        """
        self.figsize = figsize
        self.fig = None
        self.axes = None
        
    def create_figure(self) -> Tuple[Figure, Any]:
        """Create matplotlib figure with two subplots."""
        self.fig, self.axes = plt.subplots(2, 1, figsize=self.figsize, sharex=True)
        return self.fig, self.axes
        
    def plot(self, twiss, title: Optional[str] = None):
        """
        Plot Twiss parameters (beta functions and dispersion).
        
        Args:
            twiss: XSuite TwissTable object
            title: Optional title for the plot
        """
        if self.axes is None:
            self.create_figure()
            
        # Clear axes
        for ax in self.axes:
            ax.clear()
        
        ax1, ax2 = self.axes
        
        # Beta functions
        ax1.plot(twiss.s, twiss.betx, 'b-', label='βx', linewidth=2)
        ax1.plot(twiss.s, twiss.bety, 'r-', label='βy', linewidth=2)
        ax1.set_ylabel('Beta function [m]', fontsize=11)
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        if title is None:
            title = f'Optics Functions (Qx={twiss.qx:.4f}, Qy={twiss.qy:.4f})'
        ax1.set_title(title, fontsize=12, fontweight='bold')
        
        # Dispersion
        ax2.plot(twiss.s, twiss.dx, 'g-', label='Dx', linewidth=2)
        ax2.set_xlabel('S position [m]', fontsize=11)
        ax2.set_ylabel('Dispersion [m]', fontsize=11)
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        if self.fig is not None:
            self.fig.tight_layout()


class BPMPlotter:
    """Reusable component for plotting turn-by-turn BPM data."""
    
    def __init__(self, figsize: Tuple[float, float] = (14, 8)):
        """
        Initialize BPM plotter.
        
        Args:
            figsize: Figure size (width, height) in inches
        """
        self.figsize = figsize
        self.fig = None
        self.axes = None
        
    def create_figure(self) -> Tuple[Figure, Any]:
        """Create matplotlib figure with two subplots."""
        self.fig, self.axes = plt.subplots(2, 1, figsize=self.figsize, sharex=True)
        return self.fig, self.axes
        
    def plot(self, bpm_data: Dict[str, Dict[str, Any]], plot_vs_turn: bool = True):
        """
        Plot BPM readings.
        
        Args:
            bpm_data: Dictionary mapping BPM names to data dicts with 'x', 'y', 's' keys
            plot_vs_turn: If True, plot vs turn number; if False, plot vs s position
        """
        if self.axes is None:
            self.create_figure()
            
        # Clear axes
        for ax in self.axes:
            ax.clear()
        
        if len(bpm_data) == 0:
            self.axes[0].text(0.5, 0.5, 'No BPM data available', 
                            ha='center', va='center', transform=self.axes[0].transAxes)
            return
        
        ax_x, ax_y = self.axes
        
        if plot_vs_turn:
            # Plot vs turn number
            for bpm_name in sorted(bpm_data.keys()):
                data = bpm_data[bpm_name]
                turns = np.arange(len(data['x']))
                x_data = data['x'] if len(data['x'].shape) == 1 else np.mean(data['x'], axis=1)
                y_data = data['y'] if len(data['y'].shape) == 1 else np.mean(data['y'], axis=1)
                
                ax_x.plot(turns, x_data * 1e3, label=f"{bpm_name} (s={data['s']:.1f}m)", linewidth=1.5)
                ax_y.plot(turns, y_data * 1e3, label=f"{bpm_name} (s={data['s']:.1f}m)", linewidth=1.5)
            
            ax_x.set_ylabel('X position [mm]', fontsize=11)
            ax_y.set_xlabel('Turn number', fontsize=11)
            ax_y.set_ylabel('Y position [mm]', fontsize=11)
            ax_x.set_title('Turn-by-Turn BPM Readings', fontsize=12, fontweight='bold')
            
        else:
            # Plot vs s position (average over turns)
            s_positions = []
            x_means = []
            y_means = []
            x_stds = []
            y_stds = []
            labels = []
            
            for bpm_name in sorted(bpm_data.keys()):
                data = bpm_data[bpm_name]
                s_positions.append(data['s'])
                x_data = data['x'] if len(data['x'].shape) == 1 else np.mean(data['x'], axis=1)
                y_data = data['y'] if len(data['y'].shape) == 1 else np.mean(data['y'], axis=1)
                
                x_means.append(np.mean(x_data))
                y_means.append(np.mean(y_data))
                x_stds.append(np.std(x_data))
                y_stds.append(np.std(y_data))
                labels.append(bpm_name)
            
            # Plot with error bars
            ax_x.errorbar(s_positions, np.array(x_means) * 1e3, yerr=np.array(x_stds) * 1e3,
                         fmt='o-', capsize=5, linewidth=2, markersize=8)
            ax_y.errorbar(s_positions, np.array(y_means) * 1e3, yerr=np.array(y_stds) * 1e3,
                         fmt='o-', capsize=5, linewidth=2, markersize=8, color='red')
            
            # Add BPM labels
            for s, label in zip(s_positions, labels):
                ax_x.axvline(s, color='gray', linestyle='--', alpha=0.3)
                
            ax_x.set_ylabel('X position [mm]', fontsize=11)
            ax_y.set_xlabel('S position [m]', fontsize=11)
            ax_y.set_ylabel('Y position [mm]', fontsize=11)
            ax_x.set_title('BPM Readings vs S Position (averaged over turns)', fontsize=12, fontweight='bold')
        
        ax_x.legend(loc='best', fontsize=9)
        ax_y.legend(loc='best', fontsize=9)
        ax_x.grid(True, alpha=0.3)
        ax_y.grid(True, alpha=0.3)
        
        if self.fig is not None:
            self.fig.tight_layout()


class FloorPlanPlotter:
    """Reusable component for plotting lattice floor plan."""
    
    def __init__(self, figsize: Tuple[float, float] = (10, 10)):
        """
        Initialize floor plan plotter.
        
        Args:
            figsize: Figure size (width, height) in inches
        """
        self.figsize = figsize
        self.fig = None
        self.ax = None
        
    def create_figure(self) -> Tuple[Figure, Any]:
        """Create matplotlib figure and axis."""
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self.ax.set_aspect('equal')
        return self.fig, self.ax
        
    def plot(self, lattice, branch_name: str):
        """
        Plot lattice floor plan.
        
        Args:
            lattice: EICViBE Lattice object
            branch_name: Name of branch to plot
        """
        if self.ax is None:
            self.create_figure()
            
        self.ax.clear()
        self.ax.set_aspect('equal')
        lattice.plot_branch_floorplan(branch_name, ax=self.ax)
        
        if self.fig is not None:
            self.fig.tight_layout()
