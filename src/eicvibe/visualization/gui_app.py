"""
Main GUI application for EICViBE lattice visualization.

Provides interactive interface for viewing lattices, Twiss parameters, and BPM data.
Works in both terminal (PyQt5) and notebook (matplotlib interactive) modes.
"""

import sys
import logging
from typing import Optional, Dict, Any
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')  # Use Qt backend for standalone application

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QTabWidget, QLabel,
                             QComboBox, QCheckBox, QGroupBox, QSplitter,
                             QListWidget, QListWidgetItem, QRadioButton,
                             QButtonGroup, QScrollArea, QTextEdit)
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QFont

from .lattice_viewer import (BeamlinePlotter, TwissPlotter, BPMPlotter, FloorPlanPlotter)

logger = logging.getLogger(__name__)


class MplCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas widget for embedding in Qt."""
    
    def __init__(self, figure=None, parent=None, width=12, height=8, dpi=100):
        if figure is None:
            figure = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(figure)
        self.setParent(parent)


class LatticeViewerGUI(QMainWindow):
    """
    Main GUI application for EICViBE lattice visualization.
    
    Engine-agnostic interface that works with EICViBE Lattice objects
    and generic data structures from any simulation engine.
    
    Features:
    - Beamline layout view
    - Floor plan view
    - Twiss parameter plots
    - Turn-by-turn BPM data visualization
    - Toggle between different plot modes
    """
    
    def __init__(self, lattice=None, twiss=None, 
                 bpm_data=None, branch_name: str = "FODO"):
        super().__init__()
        
        self.lattice = lattice
        self.twiss = twiss
        self.bpm_data = bpm_data
        self.branch_name = branch_name
        
        self.setWindowTitle("EICViBE Lattice Viewer")
        self.setGeometry(100, 100, 1400, 900)
        
        self._init_ui()
        self._connect_signals()
        
        # Initial plot
        if self.lattice is not None:
            self.update_plots()
    
    def _init_ui(self):
        """Initialize UI components."""
        # Main widget with horizontal splitter
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Control panel
        control_panel = self._create_control_panel()
        main_layout.addWidget(control_panel)
        
        # Horizontal splitter: tabs on left, element list on right
        splitter = QSplitter(Qt.Horizontal)
        
        # Tab widget for different views (left side)
        self.tabs = QTabWidget()
        splitter.addWidget(self.tabs)
        
        # Element properties list (right side)
        self.element_list_widget = self._create_element_list()
        splitter.addWidget(self.element_list_widget)
        
        # Set initial splitter sizes (70% tabs, 30% list)
        splitter.setSizes([700, 300])
        
        main_layout.addWidget(splitter)
        
        # Create tabs
        self._create_lattice_tab()  # Combined beamline/floorplan with toggle
        self._create_twiss_tab()
        self._create_bpm_tab()
        
        # Status bar
        self.statusBar().showMessage("Ready")
    
    def _create_element_list(self) -> QWidget:
        """Create scrollable element properties list."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Title
        title = QLabel("Element Properties")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(10)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # List widget
        self.element_list = QListWidget()
        self.element_list.setWordWrap(True)
        layout.addWidget(self.element_list)
        
        # Populate list
        self._populate_element_list()
        
        return widget
    
    def _populate_element_list(self):
        """Populate element list with lattice elements (excluding Drifts)."""
        self.element_list.clear()
        
        if self.lattice is None or self.branch_name not in self.lattice.branches:
            return
        
        element_names = self.lattice.branches[self.branch_name]
        s_position = 0.0
        
        for elem_name in element_names:
            elem = self.lattice.get_element(elem_name)
            
            # Skip Drift elements in property display
            if elem.__class__.__name__ == 'Drift':
                s_position += elem.length
                continue
            
            # Format element info with type-specific key parameters
            elem_type = elem.__class__.__name__
            info_text = f"{elem.name}\n"
            info_text += f"  Type: {elem_type}\n"
            info_text += f"  Length: {elem.length:.3f} m\n"
            info_text += f"  S: {s_position:.3f} m\n"
            
            # Add key parameters based on element type
            if elem_type in ['Bend', 'RBend']:
                try:
                    angle = elem.get_parameter("BendP", "angle")
                    if angle is not None:
                        info_text += f"  Angle: {angle:.6g} rad ({np.degrees(angle):.3f}°)\n"
                except:
                    pass
            elif elem_type == 'Quadrupole':
                try:
                    k1 = elem.get_parameter("MagneticMultipoleP", "kn1")
                    if k1 is not None:
                        info_text += f"  k1: {k1:.6g} m⁻²\n"
                except:
                    pass
            elif elem_type == 'Sextupole':
                try:
                    k2 = elem.get_parameter("MagneticMultipoleP", "kn2")
                    if k2 is not None:
                        info_text += f"  k2: {k2:.6g} m⁻³\n"
                except:
                    pass
            elif elem_type == 'Octupole':
                try:
                    k3 = elem.get_parameter("MagneticMultipoleP", "kn3")
                    if k3 is not None:
                        info_text += f"  k3: {k3:.6g} m⁻⁴\n"
                except:
                    pass
            elif elem_type == 'RFCavity':
                try:
                    voltage = elem.get_parameter("RFP", "voltage")
                    frequency = elem.get_parameter("RFP", "frequency")
                    if voltage is not None:
                        info_text += f"  Voltage: {voltage:.3g} V\n"
                    if frequency is not None:
                        info_text += f"  Frequency: {frequency:.3g} Hz\n"
                except:
                    pass
            
            item = QListWidgetItem(info_text)
            item.setData(Qt.UserRole, {'name': elem.name, 's': s_position})
            self.element_list.addItem(item)
            
            s_position += elem.length
    
    def _create_control_panel(self) -> QWidget:
        """Create control panel with buttons and options."""
        panel = QGroupBox("Controls")
        layout = QHBoxLayout()
        
        # Refresh button
        self.btn_refresh = QPushButton("Refresh All")
        layout.addWidget(self.btn_refresh)
        
        # Zoom controls
        self.btn_zoom_out = QPushButton("Undo")
        self.btn_zoom_out.setEnabled(False)
        self.btn_zoom_out.setToolTip("Go back to previous zoom level")
        layout.addWidget(self.btn_zoom_out)
        
        self.btn_reset_zoom = QPushButton("Home")
        self.btn_reset_zoom.setEnabled(False)
        self.btn_reset_zoom.setToolTip("Reset to full lattice view")
        layout.addWidget(self.btn_reset_zoom)
        
        layout.addWidget(QLabel("|"))  # Separator
        
        # Branch selector
        layout.addWidget(QLabel("Branch:"))
        self.combo_branch = QComboBox()
        if self.lattice is not None:
            self.combo_branch.addItems(list(self.lattice.branches.keys()))
            self.combo_branch.setCurrentText(self.branch_name)
        layout.addWidget(self.combo_branch)
        
        # BPM plot mode toggle
        self.check_bpm_vs_s = QCheckBox("BPM vs S (instead of turn)")
        layout.addWidget(self.check_bpm_vs_s)
        
        layout.addWidget(QLabel("|"))  # Separator
        
        # Twiss plot controls
        layout.addWidget(QLabel("Twiss Plots:"))
        self.combo_twiss_num_plots = QComboBox()
        self.combo_twiss_num_plots.addItems(["1", "2"])
        self.combo_twiss_num_plots.setCurrentText("2")
        layout.addWidget(self.combo_twiss_num_plots)
        
        layout.addWidget(QLabel("Plot1:"))
        self.combo_twiss_plot1 = QComboBox()
        self.combo_twiss_plot1.addItems(["Beta", "Alpha", "Dispersion", "Phase Advance"])
        self.combo_twiss_plot1.setCurrentText("Beta")
        layout.addWidget(self.combo_twiss_plot1)
        
        layout.addWidget(QLabel("Plot2:"))
        self.combo_twiss_plot2 = QComboBox()
        self.combo_twiss_plot2.addItems(["Beta", "Alpha", "Dispersion", "Phase Advance"])
        self.combo_twiss_plot2.setCurrentText("Dispersion")
        layout.addWidget(self.combo_twiss_plot2)
        
        layout.addStretch()
        
        # Info label
        self.label_info = QLabel("")
        layout.addWidget(self.label_info)
        
        panel.setLayout(layout)
        return panel
    
    def _create_lattice_tab(self):
        """Create lattice tab with floor plan (top) and beamline (bottom) in 6:1 ratio."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Create both plotters
        self.beamline_plotter = BeamlinePlotter(figsize=(14, 2))
        self.floorplan_plotter = FloorPlanPlotter(figsize=(14, 12))
        
        # Create figure with two subplots (6:1 ratio)
        from matplotlib.gridspec import GridSpec
        fig = Figure(figsize=(14, 14))
        gs = GridSpec(2, 1, figure=fig, height_ratios=[6, 1], hspace=0.3)
        
        # Floor plan axes (top, 6 parts)
        self.floorplan_ax = fig.add_subplot(gs[0])
        self.floorplan_ax.set_aspect('equal')
        
        # Beamline axes (bottom, 1 part)
        self.beamline_ax = fig.add_subplot(gs[1])
        
        # Store axes in plotters
        self.floorplan_plotter.ax = self.floorplan_ax
        self.floorplan_plotter.fig = fig
        self.beamline_plotter.ax = self.beamline_ax
        self.beamline_plotter.fig = fig
        
        # Create canvas
        canvas = MplCanvas(figure=fig, parent=tab)
        canvas.setMouseTracking(True)  # Enable mouse tracking for hover
        
        # No navigation toolbar - zoom controlled by custom range selector
        layout.addWidget(canvas)
        
        self.tabs.addTab(tab, "Lattice")
        self.lattice_canvas = canvas
        
        # Connect hover event
        canvas.mpl_connect('motion_notify_event', self._on_lattice_hover)
        
        # Connect right-click event for undo zoom
        canvas.mpl_connect('button_press_event', self._on_mouse_click)
        
        # Add SpanSelector for interactive s-range selection on beamline axis
        from matplotlib.widgets import SpanSelector
        self.span_selector = SpanSelector(
            self.beamline_ax,
            self._on_span_select,
            'horizontal',
            useblit=True,
            props=dict(alpha=0.3, facecolor='blue'),
            interactive=False,  # Non-interactive to avoid drag conflicts
            drag_from_anywhere=False
        )
        
        # Zoom history stack
        self.zoom_history = []  # Stack of (s_start, s_end) tuples
        
        # Storage for beamline y-axis limits
        self.beamline_ylims = None
        
        # Storage for floor plan element positions (populated during plotting)
        self.floor_plan_elements = {}  # {element_name: {'s': float, 'x_center': float, 'y_center': float}}
        self.all_element_positions = {}  # Store entrance/tangent for ALL elements including drifts
        self.s_range = None  # Current s range (s_start, s_end) for zoomed view
    
    def _point_in_polygon(self, x, y, polygon):
        """Check if point (x, y) is inside polygon using ray casting algorithm.
        
        Args:
            x, y: Point coordinates
            polygon: List of (x, y) tuples defining polygon vertices
            
        Returns:
            True if point is inside polygon, False otherwise
        """
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
        
    def _calculate_floor_plan_shapes(self):
        """Calculate element shapes in floor plan by walking through elements.
        Does NOT call plotting functions - just calculates geometry.
        Stores entrance position and tangent for ALL elements.
        """
        if self.lattice is None or self.branch_name not in self.lattice.branches:
            return
        
        self.floor_plan_elements = {}
        self.all_element_positions = {}
        element_names = self.lattice.branches[self.branch_name]
        
        # Walk through elements and calculate positions/shapes
        current_s = 0.0
        entrance_coords = np.array([0.0, 0.0])
        tangent_vector = np.array([1.0, 0.0])
        
        captured_positions = []  # For debug output
        
        for elem_name in element_names:
            elem = self.lattice.get_element(elem_name)
            
            # Store entrance position and tangent for ALL elements (including drifts)
            self.all_element_positions[elem_name] = {
                's_start': current_s,
                's_end': current_s + elem.length,
                'entrance': entrance_coords.copy(),
                'tangent': tangent_vector.copy()
            }
            
            # Calculate exit coords based on element type
            elem_class = elem.__class__.__name__
            
            # Skip drifts and zero-length elements for hover
            if elem_class == 'Drift' or elem.length == 0:
                exit_coords = entrance_coords + elem.length * tangent_vector
                entrance_coords = exit_coords
                current_s += elem.length
                continue
            
            # Calculate shape corners based on element type
            shape = None
            
            if elem_class in ['Quadrupole', 'Sextupole', 'Octupole']:
                # Straight element - rectangle
                exit_coords = entrance_coords + elem.length * tangent_vector
                angle = np.arctan2(tangent_vector[1], tangent_vector[0])
                half_width = elem.plot_cross_section / 2.0
                
                # Perpendicular vector
                perp_vec = np.array([np.cos(angle - np.pi/2), np.sin(angle - np.pi/2)])
                
                # Four corners
                corner1 = entrance_coords + half_width * perp_vec
                corner2 = entrance_coords - half_width * perp_vec
                corner3 = exit_coords - half_width * perp_vec
                corner4 = exit_coords + half_width * perp_vec
                
                shape = [
                    (corner1[0], corner1[1]),
                    (corner2[0], corner2[1]),
                    (corner3[0], corner3[1]),
                    (corner4[0], corner4[1])
                ]
                new_tangent = tangent_vector
                
            elif elem_class in ['Bend', 'RBend']:
                # Bend element - annular sector approximated as rectangle
                try:
                    angle = elem.get_parameter("BendP", "angle")
                    if angle is not None and angle != 0:
                        radius = elem.length / abs(angle)
                        entrance_angle = np.arctan2(tangent_vector[1], tangent_vector[0])
                        center_angle = entrance_angle + np.pi / 2 * np.sign(angle)
                        center_x = entrance_coords[0] + radius * np.cos(center_angle)
                        center_y = entrance_coords[1] + radius * np.sin(center_angle)
                        
                        # Exit coords
                        inv_center_angle = center_angle - np.pi
                        exit_coords = np.array([
                            center_x + radius * np.cos(inv_center_angle + angle),
                            center_y + radius * np.sin(inv_center_angle + angle)
                        ])
                        
                        # Four corners
                        dipole_width = elem.plot_cross_section / 2.0
                        corner1 = np.array([
                            center_x + (radius - dipole_width) * np.cos(inv_center_angle),
                            center_y + (radius - dipole_width) * np.sin(inv_center_angle)
                        ])
                        corner2 = np.array([
                            center_x + (radius + dipole_width) * np.cos(inv_center_angle),
                            center_y + (radius + dipole_width) * np.sin(inv_center_angle)
                        ])
                        corner3 = np.array([
                            center_x + (radius + dipole_width) * np.cos(inv_center_angle + angle),
                            center_y + (radius + dipole_width) * np.sin(inv_center_angle + angle)
                        ])
                        corner4 = np.array([
                            center_x + (radius - dipole_width) * np.cos(inv_center_angle + angle),
                            center_y + (radius - dipole_width) * np.sin(inv_center_angle + angle)
                        ])
                        
                        shape = [
                            (corner1[0], corner1[1]),
                            (corner2[0], corner2[1]),
                            (corner3[0], corner3[1]),
                            (corner4[0], corner4[1])
                        ]
                        
                        # Update tangent
                        cos_a = np.cos(angle)
                        sin_a = np.sin(angle)
                        new_tangent = np.array([
                            tangent_vector[0] * cos_a - tangent_vector[1] * sin_a,
                            tangent_vector[0] * sin_a + tangent_vector[1] * cos_a
                        ])
                    else:
                        exit_coords = entrance_coords + elem.length * tangent_vector
                        new_tangent = tangent_vector
                except:
                    exit_coords = entrance_coords + elem.length * tangent_vector
                    new_tangent = tangent_vector
            else:
                # Other elements - treat as straight
                exit_coords = entrance_coords + elem.length * tangent_vector
                new_tangent = tangent_vector
            
            # Calculate center
            elem_center_x = (entrance_coords[0] + exit_coords[0]) / 2.0
            elem_center_y = (entrance_coords[1] + exit_coords[1]) / 2.0
            
            self.floor_plan_elements[elem_name] = {
                's': current_s + elem.length / 2,
                'x_center': elem_center_x,
                'y_center': elem_center_y,
                'entrance': entrance_coords.copy(),
                'exit': exit_coords.copy(),
                'length': elem.length,
                'tangent': new_tangent.copy(),
                'shape': shape
            }
            
            # Debug output
            captured_positions.append(
                f"{elem_name}: entrance=({entrance_coords[0]:.2f},{entrance_coords[1]:.2f}), "
                f"center=({elem_center_x:.2f},{elem_center_y:.2f}), "
                f"exit=({exit_coords[0]:.2f},{exit_coords[1]:.2f})"
            )
            
            # Update for next element
            entrance_coords = exit_coords
            tangent_vector = new_tangent
            current_s += elem.length
        
    def _on_lattice_hover(self, event):
        """Handle mouse hover over lattice plot with enhanced tooltip."""
        if event.inaxes is None or self.lattice is None:
            # Hide tooltip when not over axes
            QtWidgets.QToolTip.hideText()
            return
        
        # Check which axes the mouse is over
        if event.inaxes not in [self.beamline_ax, self.floorplan_ax]:
            QtWidgets.QToolTip.hideText()
            return
        
        # Get s position and element position - different logic for beamline vs floor plan
        element_plot_x = None
        element_plot_y = None
        
        if event.inaxes == self.beamline_ax:
            # Beamline: x coordinate is directly s position
            s_pos = event.xdata
            self.label_info.setText(f"Beamline hover at s={s_pos:.2f}")
        else:
            # Floor plan: need to find closest element based on captured positions
            mouse_x, mouse_z = event.xdata, event.ydata
            if mouse_x is None or mouse_z is None:
                return
            
            # Find closest element using captured floor plan positions
            min_dist = float('inf')
            closest_elem = None
            closest_s = None
            
            for elem_name, pos_info in self.floor_plan_elements.items():
                # Check if mouse is inside element shape
                shape = pos_info.get('shape')
                
                if shape and len(shape) >= 3:
                    # Use point-in-polygon test
                    if self._point_in_polygon(mouse_x, mouse_z, shape):
                        # Inside this element - select it immediately
                        closest_elem = elem_name
                        closest_s = pos_info['s']
                        element_plot_x = pos_info['x_center']
                        element_plot_y = pos_info['y_center']
                        min_dist = 0  # Inside shape
                        break  # Found it, no need to check others
                else:
                    # No shape info - fall back to distance to center
                    dx = mouse_x - pos_info['x_center']
                    dy = mouse_z - pos_info['y_center']
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_elem = elem_name
                        closest_s = pos_info['s']
                        element_plot_x = pos_info['x_center']
                        element_plot_y = pos_info['y_center']
            
            # Check if within threshold (only for fallback distance check)
            if closest_elem is None or (min_dist > 0 and min_dist > 0.5):
                QtWidgets.QToolTip.hideText()
                self.label_info.setText(f"Floor: mouse at ({mouse_x:.2f}, {mouse_z:.2f}) - no element")
                return
            
            s_pos = closest_s
            # Debug output
            self.label_info.setText(
                f"Floor: mouse=({mouse_x:.2f}, {mouse_z:.2f}), "
                f"{closest_elem} at ({element_plot_x:.2f}, {element_plot_y:.2f})"
            )
        
        if s_pos is None:
            return
        
        # Find element at this s position
        element_names = self.lattice.branches[self.branch_name]
        current_s = 0.0
        hovered_element = None
        hovered_index = None
        hovered_s_start = 0.0
        
        for idx, elem_name in enumerate(element_names):
            elem = self.lattice.get_element(elem_name)
            if current_s <= s_pos < current_s + elem.length:
                hovered_element = elem
                hovered_index = idx
                hovered_s_start = current_s
                break
            current_s += elem.length
        
        # Skip Drift elements - don't show tooltips for drifts
        if hovered_element and hovered_element.__class__.__name__ == 'Drift':
            QtWidgets.QToolTip.hideText()
            return
        
        if hovered_element:
            # Build comprehensive tooltip
            tooltip_lines = []
            tooltip_lines.append(f"<b>{hovered_element.name}</b>")
            tooltip_lines.append(f"Type: {hovered_element.__class__.__name__}")
            tooltip_lines.append(f"Length: {hovered_element.length:.4f} m")
            tooltip_lines.append(f"S position: {hovered_s_start:.4f} m")
            tooltip_lines.append("")  # Blank line
            
            # Add all parameters with full details
            if hasattr(hovered_element, 'get_all_parameters'):
                params = hovered_element.get_all_parameters()
                if params:
                    tooltip_lines.append("<b>Parameters:</b>")
                    for group, group_params in params.items():
                        tooltip_lines.append(f"  <i>{group}:</i>")
                        for param_name, param_value in group_params.items():
                            if isinstance(param_value, (int, float)):
                                tooltip_lines.append(f"    {param_name}: {param_value:.6g}")
                            elif isinstance(param_value, list):
                                tooltip_lines.append(f"    {param_name}: {param_value}")
                            else:
                                tooltip_lines.append(f"    {param_name}: {str(param_value)}")
            
            # Join with HTML line breaks
            tooltip_text = "<br>".join(tooltip_lines)
            
            # Calculate tooltip position using QCursor for reliable global coordinates
            from PyQt5.QtGui import QCursor
            
            # Get actual cursor position in global screen coordinates
            cursor_global = QCursor.pos()
            
            # Offset tooltip by 5 pixels right and 5 pixels down from cursor (close to cursor)
            tooltip_offset = QPoint(5, 5)
            tooltip_pos = cursor_global + tooltip_offset
            
            # Debug output
            if event.inaxes == self.floorplan_ax and element_plot_x is not None:
                self.label_info.setText(
                    f"Floor {hovered_element.name}: "
                    f"data=({element_plot_x:.2f},{element_plot_y:.2f}) | "
                    f"cursor=({cursor_global.x()},{cursor_global.y()}) | "
                    f"tooltip=({tooltip_pos.x()},{tooltip_pos.y()})"
                )
            elif event.inaxes == self.beamline_ax:
                elem_center_s = hovered_s_start + hovered_element.length / 2
                self.label_info.setText(
                    f"Beam {hovered_element.name}: "
                    f"s={elem_center_s:.2f}m | "
                    f"cursor=({cursor_global.x()},{cursor_global.y()}) | "
                    f"tooltip=({tooltip_pos.x()},{tooltip_pos.y()})"
                )
            else:
                self.label_info.setText(f"Hover: {hovered_element.name}")
            
            # Show rich text tooltip at calculated position
            QtWidgets.QToolTip.showText(tooltip_pos, tooltip_text, self.lattice_canvas)
            
            # Scroll to element in list
            if hovered_index is not None:
                self.element_list.setCurrentRow(hovered_index)
                self.element_list.scrollToItem(self.element_list.item(hovered_index))
    
    def _on_mouse_click(self, event):
        """Handle mouse click events - right-click to undo zoom."""
        # Check if right-click (button 3) in lattice plotting area
        if event.button == 3 and event.inaxes in [self.beamline_ax, self.floorplan_ax]:
            # Right-click in lattice area - undo zoom
            if self.zoom_history:  # Only if there's zoom history
                self._zoom_out_one_level()
    
    def _on_twiss_mouse_click(self, event):
        """Handle mouse click events in Twiss tab - right-click to undo zoom."""
        # Check if right-click (button 3) in twiss plotting area
        if event.button == 3 and event.inaxes in [self.twiss_plot1_ax, self.twiss_plot2_ax, self.twiss_lattice_ax]:
            # Right-click in twiss area - undo zoom
            if self.twiss_zoom_history:
                self._twiss_zoom_out_one_level()
    
    def _on_twiss_config_changed(self):
        """Handle change in Twiss plot configuration."""
        # Reconfigure layout and update Twiss plots
        if self.twiss is not None:
            self._reconfigure_twiss_layout()
            self.update_twiss_plot()
    
    def _reconfigure_twiss_layout(self):
        """Reconfigure Twiss layout based on number of plots selected."""
        num_plots = int(self.combo_twiss_num_plots.currentText())
        
        # Remove old axes
        self.twiss_plot1_ax.remove()
        self.twiss_plot2_ax.remove()
        self.twiss_lattice_ax.remove()
        
        # Clear old span selectors
        for selector in self.twiss_span_selectors:
            try:
                selector.set_active(False)
            except:
                pass
        self.twiss_span_selectors.clear()
        
        # Reconfigure GridSpec based on number of plots
        from matplotlib.gridspec import GridSpec
        if num_plots == 1:
            # 1 twiss plot + lattice with 18:1 ratio (full height for single plot)
            self.twiss_gs = GridSpec(2, 1, figure=self.twiss_fig, height_ratios=[18, 1], hspace=0.05)
            self.twiss_plot1_ax = self.twiss_fig.add_subplot(self.twiss_gs[0])
            self.twiss_lattice_ax = self.twiss_fig.add_subplot(self.twiss_gs[1], sharex=self.twiss_plot1_ax)
            
            # Create dummy plot2 ax (hidden)
            self.twiss_plot2_ax = self.twiss_fig.add_subplot(self.twiss_gs[0])
            self.twiss_plot2_ax.set_visible(False)
        else:
            # 2 twiss plots + lattice with 9:9:1 ratio
            self.twiss_gs = GridSpec(3, 1, figure=self.twiss_fig, height_ratios=[9, 9, 1], hspace=0.05)
            self.twiss_plot1_ax = self.twiss_fig.add_subplot(self.twiss_gs[0])
            self.twiss_plot2_ax = self.twiss_fig.add_subplot(self.twiss_gs[1], sharex=self.twiss_plot1_ax)
            self.twiss_lattice_ax = self.twiss_fig.add_subplot(self.twiss_gs[2], sharex=self.twiss_plot1_ax)
        
        # Hide x-axis labels on upper plots
        self.twiss_plot1_ax.tick_params(labelbottom=False)
        if num_plots == 2:
            self.twiss_plot2_ax.tick_params(labelbottom=False)
        
        # Create SpanSelectors for all visible plots
        from matplotlib.widgets import SpanSelector
        
        # Selector for plot1
        selector1 = SpanSelector(
            self.twiss_plot1_ax,
            self._on_twiss_span_select,
            'horizontal',
            useblit=True,
            props=dict(alpha=0.3, facecolor='red'),
            interactive=False,
            drag_from_anywhere=False
        )
        self.twiss_span_selectors.append(selector1)
        
        # Selector for plot2 (if visible)
        if num_plots == 2:
            selector2 = SpanSelector(
                self.twiss_plot2_ax,
                self._on_twiss_span_select,
                'horizontal',
                useblit=True,
                props=dict(alpha=0.3, facecolor='red'),
                interactive=False,
                drag_from_anywhere=False
            )
            self.twiss_span_selectors.append(selector2)
        
        # Selector for lattice
        selector3 = SpanSelector(
            self.twiss_lattice_ax,
            self._on_twiss_span_select,
            'horizontal',
            useblit=True,
            props=dict(alpha=0.3, facecolor='red'),
            interactive=False,
            drag_from_anywhere=False
        )
        self.twiss_span_selectors.append(selector3)
    
    def _on_tab_changed(self, index):
        """Update zoom button states based on active tab."""
        # Lattice tab (index 0)
        if index == 0:
            has_history = bool(self.zoom_history)
            self.btn_zoom_out.setEnabled(has_history)
            self.btn_reset_zoom.setEnabled(has_history)
        # Twiss tab (index 1)
        elif index == 1:
            has_history = bool(self.twiss_zoom_history)
            self.btn_zoom_out.setEnabled(has_history)
            self.btn_reset_zoom.setEnabled(has_history)
        else:
            # Other tabs - disable zoom buttons
            self.btn_zoom_out.setEnabled(False)
            self.btn_reset_zoom.setEnabled(False)
    
    def _on_twiss_span_select(self, xmin, xmax):
        """Handle span selection on Twiss plots - zoom to selected s range."""
        if self.twiss is None or abs(xmax - xmin) < 0.01:
            return
        
        # Store current y-limits before zooming
        num_plots = int(self.combo_twiss_num_plots.currentText())
        self.twiss_ylims = {
            'plot1': self.twiss_plot1_ax.get_ylim(),
        }
        if num_plots == 2:
            self.twiss_ylims['plot2'] = self.twiss_plot2_ax.get_ylim()
        
        # Save current range to history before zooming
        if self.twiss_s_range is not None:
            self.twiss_zoom_history.append(self.twiss_s_range)
        else:
            # Save the initial full range
            self.twiss_zoom_history.append((self.twiss.s[0], self.twiss.s[-1]))
        
        # Enable zoom buttons
        self.btn_zoom_out.setEnabled(True)
        self.btn_reset_zoom.setEnabled(True)
        
        # Ensure xmin < xmax
        s_start, s_end = min(xmin, xmax), max(xmin, xmax)
        
        # Store the range
        self.twiss_s_range = (s_start, s_end)
        
        # Update all twiss plots
        self._update_twiss_for_range(s_start, s_end)
        
        # Clear all span selector visuals
        for selector in self.twiss_span_selectors:
            try:
                selector.extents = (0, 0)
                selector.set_visible(False)
            except:
                pass
        self.twiss_canvas.draw_idle()
    
    def _twiss_zoom_out_one_level(self):
        """Zoom out Twiss plots to previous level in history."""
        if not self.twiss_zoom_history:
            return
        
        # Pop the last range from history
        prev_range = self.twiss_zoom_history.pop()
        
        # Update buttons
        if not self.twiss_zoom_history:
            self.btn_zoom_out.setEnabled(False)
            self.btn_reset_zoom.setEnabled(False)
            self.twiss_s_range = None
            # Show full twiss
            self.update_twiss_plot()
        else:
            self.twiss_s_range = prev_range
            self._update_twiss_for_range(prev_range[0], prev_range[1])
    
    def _twiss_reset_zoom(self):
        """Reset Twiss zoom to show full range."""
        self.twiss_zoom_history.clear()
        self.twiss_s_range = None
        self.btn_zoom_out.setEnabled(False)
        self.btn_reset_zoom.setEnabled(False)
        self.update_twiss_plot()
    
    def _update_twiss_for_range(self, s_start, s_end):
        """Update Twiss plots to show only the selected s range."""
        if self.twiss is None:
            return
        
        # Find indices for the s range
        mask = (self.twiss.s >= s_start) & (self.twiss.s <= s_end)
        
        # Get configuration
        num_plots = int(self.combo_twiss_num_plots.currentText())
        plot1_type = self.combo_twiss_plot1.currentText()
        plot2_type = self.combo_twiss_plot2.currentText()
        
        # Store current y-limits before clearing (if they exist)
        if self.twiss_plot1_ax.get_ylim() != (0.0, 1.0):  # Default limits
            self.twiss_ylims['plot1'] = self.twiss_plot1_ax.get_ylim()
        if num_plots == 2 and self.twiss_plot2_ax.get_ylim() != (0.0, 1.0):
            self.twiss_ylims['plot2'] = self.twiss_plot2_ax.get_ylim()
        
        # Clear axes
        self.twiss_plot1_ax.clear()
        self.twiss_plot2_ax.clear()
        self.twiss_lattice_ax.clear()
        
        # Plot first function
        self._plot_twiss_function(self.twiss_plot1_ax, plot1_type, mask, s_start, s_end)
        
        # Show/hide second plot based on configuration
        if num_plots == 2:
            self.twiss_plot2_ax.set_visible(True)
            self._plot_twiss_function(self.twiss_plot2_ax, plot2_type, mask, s_start, s_end, is_middle=True)
        else:
            self.twiss_plot2_ax.set_visible(False)
        
        # Restore y-limits if available
        if 'plot1' in self.twiss_ylims:
            self.twiss_plot1_ax.set_ylim(self.twiss_ylims['plot1'])
        if num_plots == 2 and 'plot2' in self.twiss_ylims:
            self.twiss_plot2_ax.set_ylim(self.twiss_ylims['plot2'])
        
        # Hide x-axis labels on upper plots
        self.twiss_plot1_ax.tick_params(labelbottom=False)
        if num_plots == 2:
            self.twiss_plot2_ax.tick_params(labelbottom=False)
        
        # Plot lattice beamline
        self._plot_twiss_lattice(self.twiss_lattice_ax, s_start, s_end)
        
        # Adjust layout to prevent overlap
        self.twiss_fig.tight_layout()
        
        self.twiss_canvas.draw()
    
    def _find_element_at_floor_position(self, x: float, z: float):
        """Find element s-position and plot coordinates closest to floor plan (x,z) coordinates.
        
        Note: Floor plan uses (x, y) internally but matplotlib displays y-axis as z in the plot.
        
        Returns:
            tuple: (s_position, element_x, element_y) or None if no element found
        """
        if self.lattice is None or self.branch_name not in self.lattice.branches:
            return None
            
        # Get element positions in floor plan - match the floor plan plotting logic
        element_names = self.lattice.branches[self.branch_name]
        
        # Calculate positions along the beamline using same logic as plot_branch_floorplan
        # The floor plan starts at (0, 0) pointing in +x direction
        positions = []
        current_s = 0.0
        current_x = 0.0
        current_y = 0.0  # This is y in floor plan coords, displayed as z in plot
        current_tangent = np.array([1.0, 0.0])  # Tangent vector (cos, sin of angle)
        
        for elem_name in element_names:
            elem = self.lattice.get_element(elem_name)
            
            # Skip drifts in floor plan detection
            if elem.__class__.__name__ == 'Drift':
                # Update position: move along tangent vector
                current_x += elem.length * current_tangent[0]
                current_y += elem.length * current_tangent[1]
                current_s += elem.length
                continue
            
            # Store element center position
            elem_center_s = current_s + elem.length / 2
            elem_center_x = current_x + (elem.length / 2) * current_tangent[0]
            elem_center_y = current_y + (elem.length / 2) * current_tangent[1]
            
            # Calculate distance to mouse position (z in plot is y in floor plan coords)
            distance = np.sqrt((elem_center_x - x)**2 + (elem_center_y - z)**2)
            positions.append((distance, elem_center_s, elem_center_x, elem_center_y))
            
            # Update position for next element
            current_x += elem.length * current_tangent[0]
            current_y += elem.length * current_tangent[1]
            current_s += elem.length
            
            # Update tangent vector if this is a bend
            if hasattr(elem, 'get_parameter'):
                try:
                    angle = elem.get_parameter('BendP', 'angle')
                    if angle is not None:
                        # Rotate tangent vector by bend angle using rotation matrix
                        cos_a = np.cos(angle)
                        sin_a = np.sin(angle)
                        new_tangent = np.array([
                            current_tangent[0] * cos_a - current_tangent[1] * sin_a,
                            current_tangent[0] * sin_a + current_tangent[1] * cos_a
                        ])
                        current_tangent = new_tangent
                except:
                    pass
        
        # Find closest element
        if positions:
            min_distance, s_pos, elem_x, elem_y = min(positions, key=lambda p: p[0])
            # Detection threshold - 0.5 meters for precise hovering
            if min_distance < 0.5:
                return (s_pos, elem_x, elem_y)
        
        return None
    
    def _create_twiss_tab(self):
        """Create Twiss parameters tab with lattice alignment and zoom."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Create figure with dynamic subplots (up to 2 twiss plots + lattice)
        from matplotlib.gridspec import GridSpec
        self.twiss_fig = Figure(figsize=(14, 12))
        # Initial: 2 twiss plots + lattice with 9:9:1 ratio
        self.twiss_gs = GridSpec(3, 1, figure=self.twiss_fig, height_ratios=[9, 9, 1], hspace=0.05)
        
        # Create all possible axes (we'll show/hide based on selection)
        self.twiss_plot1_ax = self.twiss_fig.add_subplot(self.twiss_gs[0])
        self.twiss_plot2_ax = self.twiss_fig.add_subplot(self.twiss_gs[1], sharex=self.twiss_plot1_ax)
        self.twiss_lattice_ax = self.twiss_fig.add_subplot(self.twiss_gs[2], sharex=self.twiss_plot1_ax)
        
        # Hide x-axis labels and ticks on upper plots
        self.twiss_plot1_ax.tick_params(labelbottom=False)
        self.twiss_plot2_ax.tick_params(labelbottom=False)
        
        # Create canvas
        canvas = MplCanvas(figure=self.twiss_fig, parent=tab)
        canvas.setMouseTracking(True)
        toolbar = NavigationToolbar2QT(canvas, tab)
        
        # Connect right-click event for undo zoom
        canvas.mpl_connect('button_press_event', self._on_twiss_mouse_click)
        
        # Twiss zoom history and y-axis limits storage
        self.twiss_zoom_history = []
        self.twiss_s_range = None
        self.twiss_ylims = {}  # Store y-limits for each plot type
        self.twiss_lattice_ylims = None  # Store lattice y-limits
        
        # SpanSelectors will be created/recreated when plots are configured
        self.twiss_span_selectors = []
        
        layout.addWidget(toolbar)
        layout.addWidget(canvas)
        
        self.tabs.addTab(tab, "Twiss Parameters")
        self.twiss_canvas = canvas
    
    def _create_bpm_tab(self):
        """Create BPM data tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Create plotter and canvas
        self.bpm_plotter = BPMPlotter(figsize=(14, 8))
        fig, _ = self.bpm_plotter.create_figure()
        
        canvas = MplCanvas(figure=fig, parent=tab)
        toolbar = NavigationToolbar2QT(canvas, tab)
        
        layout.addWidget(toolbar)
        layout.addWidget(canvas)
        
        self.tabs.addTab(tab, "BPM Data")
        self.bpm_canvas = canvas
    
    def _connect_signals(self):
        """Connect UI signals to slots."""
        self.btn_refresh.clicked.connect(self.update_plots)
        self.btn_zoom_out.clicked.connect(self._zoom_out_one_level)
        self.btn_reset_zoom.clicked.connect(self._reset_zoom)
        self.combo_branch.currentTextChanged.connect(self.on_branch_changed)
        self.check_bpm_vs_s.stateChanged.connect(self.update_bpm_plot)
        
        # Twiss plot selection handlers
        self.combo_twiss_num_plots.currentTextChanged.connect(self._on_twiss_config_changed)
        self.combo_twiss_plot1.currentTextChanged.connect(self._on_twiss_config_changed)
        self.combo_twiss_plot2.currentTextChanged.connect(self._on_twiss_config_changed)
        
        # Tab change handler to update zoom buttons based on active tab
        self.tabs.currentChanged.connect(self._on_tab_changed)
    
    def _on_span_select(self, xmin, xmax):
        """Handle span selection on beamline axis - zoom to selected s range."""
        if self.lattice is None or abs(xmax - xmin) < 0.01:
            return
        
        # Save current range to history before zooming
        if self.s_range is not None:
            self.zoom_history.append(self.s_range)
        else:
            # Save the initial full range
            element_names = self.lattice.branches[self.branch_name]
            total_length = sum(self.lattice.get_element(name).length for name in element_names)
            self.zoom_history.append((0, total_length))
        
        # Enable zoom out buttons
        self.btn_zoom_out.setEnabled(True)
        self.btn_reset_zoom.setEnabled(True)
        
        # Ensure xmin < xmax
        s_start, s_end = min(xmin, xmax), max(xmin, xmax)
        
        # Store the range
        self.s_range = (s_start, s_end)
        
        # Update both plots
        self._update_plots_for_range(s_start, s_end)
        
        # Clear the span selector visual to allow new selections
        # Set extents to None to clear the selection
        try:
            self.span_selector.extents = (0, 0)
            self.span_selector.set_visible(False)
            self.lattice_canvas.draw_idle()
        except:
            pass  # Graceful fallback if span selector methods unavailable
    
    def _zoom_out_one_level(self):
        """Zoom out to previous level in history."""
        # Check which tab is active
        current_tab = self.tabs.currentIndex()
        
        if current_tab == 0:  # Lattice tab
            if not self.zoom_history:
                return
            
            # Pop the last range from history
            prev_range = self.zoom_history.pop()
            
            # Update buttons
            if not self.zoom_history:
                self.btn_zoom_out.setEnabled(False)
                self.btn_reset_zoom.setEnabled(False)
                self.s_range = None
                # Show full lattice
                self.update_plots()
            else:
                self.s_range = prev_range
                self._update_plots_for_range(prev_range[0], prev_range[1])
        
        elif current_tab == 1:  # Twiss tab
            self._twiss_zoom_out_one_level()
    
    def _reset_zoom(self):
        """Reset zoom to show full view."""
        # Check which tab is active
        current_tab = self.tabs.currentIndex()
        
        if current_tab == 0:  # Lattice tab
            self.zoom_history.clear()
            self.s_range = None
            self.btn_zoom_out.setEnabled(False)
            self.btn_reset_zoom.setEnabled(False)
            self.update_plots()
        
        elif current_tab == 1:  # Twiss tab
            self._twiss_reset_zoom()
    
    def _on_beamline_zoom(self, ax):
        """Handle zoom/pan on beamline axis - update both plots to show selected s range."""
        if self.lattice is None:
            return
        
        # Get current x-axis limits (s range)
        s_start, s_end = ax.get_xlim()
        
        # Store the range
        self.s_range = (s_start, s_end)
        
        # Update both floor plan and beamline to show this range
        self._update_plots_for_range(s_start, s_end)
    
    def _get_elements_in_range(self, s_start, s_end):
        """Get elements and their positions in the given s range.
        
        Returns:
            filtered_elements: List of (elem_name, elem, s_elem_start, s_elem_end)
            entrance_coords: Starting coordinates for first element (from stored positions)
            entrance_tangent: Starting tangent for first element (from stored positions)
        """
        if self.lattice is None or self.branch_name not in self.lattice.branches:
            return [], np.array([0.0, 0.0]), np.array([1.0, 0.0])
        
        element_names = self.lattice.branches[self.branch_name]
        filtered = []
        first_elem_entrance = None
        first_elem_tangent = None
        
        for elem_name in element_names:
            if elem_name not in self.all_element_positions:
                continue
            
            elem = self.lattice.get_element(elem_name)
            pos_info = self.all_element_positions[elem_name]
            elem_start = pos_info['s_start']
            elem_end = pos_info['s_end']
            
            # Check if element overlaps with range
            if elem_end > s_start and elem_start < s_end:
                if first_elem_entrance is None:
                    # Use stored entrance position and tangent from full lattice
                    first_elem_entrance = pos_info['entrance'].copy()
                    first_elem_tangent = pos_info['tangent'].copy()
                
                filtered.append((elem_name, elem, elem_start, elem_end))
            
            # Stop if we're past the range
            if elem_start > s_end:
                break
        
        if first_elem_entrance is None:
            first_elem_entrance = np.array([0.0, 0.0])
            first_elem_tangent = np.array([1.0, 0.0])
        
        return filtered, first_elem_entrance, first_elem_tangent
    
    def _update_plots_for_range(self, s_start, s_end):
        """Update floor plan and beamline to show only elements in s range."""
        # Get filtered elements
        filtered_elements, entrance_coords, entrance_tangent = self._get_elements_in_range(s_start, s_end)
        
        if not filtered_elements:
            return
        
        # Clear and replot floor plan with filtered elements
        self.floorplan_ax.clear()
        self.floorplan_ax.set_aspect('equal')
        
        current_coords = entrance_coords.copy()
        current_vector = entrance_tangent.copy()
        
        for elem_name, elem, _, _ in filtered_elements:
            current_coords, current_vector = elem.plot_in_floorplan(self.floorplan_ax, current_coords, current_vector)
        
        self.floorplan_ax.set_xlabel('X coordinate (m)')
        self.floorplan_ax.set_ylabel('Y coordinate (m)')
        self.floorplan_ax.set_title(f'Floor plan view: {self.branch_name} (s={s_start:.1f} to {s_end:.1f}m)')
        self.floorplan_ax.grid(True, alpha=0.3)
        
        # Adjust aspect ratio to avoid super narrow plots
        self._adjust_floor_plan_aspect_ratio()
        
        # Update beamline axis limits to match the zoomed range
        self.beamline_ax.clear()
        # Replot beamline for the selected range
        for elem_name, elem, elem_start, _ in filtered_elements:
            elem.plot_in_beamline(self.beamline_ax, elem_start)
        
        # Add horizontal reference line at y=0
        self.beamline_ax.axhline(y=0, color='black', linewidth=0.5, linestyle='-', alpha=0.3)
        
        self.beamline_ax.set_xlim(s_start, s_end)
        self.beamline_ax.set_xlabel('S position (m)')
        self.beamline_ax.set_yticks([])
        self.beamline_ax.set_title(f'Beamline view: {self.branch_name} (s={s_start:.1f} to {s_end:.1f}m)')
        
        # Restore original y-axis limits to maintain vertical scale
        if self.beamline_ylims is not None:
            self.beamline_ax.set_ylim(self.beamline_ylims)
        
        # Recalculate shapes for hover using proper entrance coords/tangent
        self._calculate_floor_plan_shapes_for_range(filtered_elements, entrance_coords, entrance_tangent)
        
        # Redraw
        self.lattice_canvas.draw()
    
    def _calculate_floor_plan_shapes_for_range(self, filtered_elements, entrance_coords_start, entrance_tangent_start):
        """Calculate shapes for filtered elements."""
        self.floor_plan_elements = {}
        
        entrance_coords = entrance_coords_start.copy()
        tangent_vector = entrance_tangent_start.copy()
        
        for elem_name, elem, s_start, s_end in filtered_elements:
            elem_class = elem.__class__.__name__
            
            # Skip drifts and zero-length
            if elem_class == 'Drift' or elem.length == 0:
                exit_coords = entrance_coords + elem.length * tangent_vector
                entrance_coords = exit_coords
                continue
            
            # Calculate shape (same logic as _calculate_floor_plan_shapes)
            shape = None
            
            if elem_class in ['Quadrupole', 'Sextupole', 'Octupole']:
                exit_coords = entrance_coords + elem.length * tangent_vector
                angle = np.arctan2(tangent_vector[1], tangent_vector[0])
                half_width = elem.plot_cross_section / 2.0
                perp_vec = np.array([np.cos(angle - np.pi/2), np.sin(angle - np.pi/2)])
                
                corner1 = entrance_coords + half_width * perp_vec
                corner2 = entrance_coords - half_width * perp_vec
                corner3 = exit_coords - half_width * perp_vec
                corner4 = exit_coords + half_width * perp_vec
                
                shape = [(corner1[0], corner1[1]), (corner2[0], corner2[1]), 
                        (corner3[0], corner3[1]), (corner4[0], corner4[1])]
                new_tangent = tangent_vector
                
            elif elem_class in ['Bend', 'RBend']:
                try:
                    angle = elem.get_parameter("BendP", "angle")
                    if angle is not None and angle != 0:
                        radius = elem.length / abs(angle)
                        entrance_angle = np.arctan2(tangent_vector[1], tangent_vector[0])
                        center_angle = entrance_angle + np.pi / 2 * np.sign(angle)
                        center_x = entrance_coords[0] + radius * np.cos(center_angle)
                        center_y = entrance_coords[1] + radius * np.sin(center_angle)
                        
                        inv_center_angle = center_angle - np.pi
                        exit_coords = np.array([
                            center_x + radius * np.cos(inv_center_angle + angle),
                            center_y + radius * np.sin(inv_center_angle + angle)
                        ])
                        
                        dipole_width = elem.plot_cross_section / 2.0
                        corner1 = np.array([center_x + (radius - dipole_width) * np.cos(inv_center_angle),
                                           center_y + (radius - dipole_width) * np.sin(inv_center_angle)])
                        corner2 = np.array([center_x + (radius + dipole_width) * np.cos(inv_center_angle),
                                           center_y + (radius + dipole_width) * np.sin(inv_center_angle)])
                        corner3 = np.array([center_x + (radius + dipole_width) * np.cos(inv_center_angle + angle),
                                           center_y + (radius + dipole_width) * np.sin(inv_center_angle + angle)])
                        corner4 = np.array([center_x + (radius - dipole_width) * np.cos(inv_center_angle + angle),
                                           center_y + (radius - dipole_width) * np.sin(inv_center_angle + angle)])
                        
                        shape = [(corner1[0], corner1[1]), (corner2[0], corner2[1]),
                                (corner3[0], corner3[1]), (corner4[0], corner4[1])]
                        
                        cos_a = np.cos(angle)
                        sin_a = np.sin(angle)
                        new_tangent = np.array([tangent_vector[0] * cos_a - tangent_vector[1] * sin_a,
                                               tangent_vector[0] * sin_a + tangent_vector[1] * cos_a])
                    else:
                        exit_coords = entrance_coords + elem.length * tangent_vector
                        new_tangent = tangent_vector
                except:
                    exit_coords = entrance_coords + elem.length * tangent_vector
                    new_tangent = tangent_vector
            else:
                exit_coords = entrance_coords + elem.length * tangent_vector
                new_tangent = tangent_vector
            
            elem_center_x = (entrance_coords[0] + exit_coords[0]) / 2.0
            elem_center_y = (entrance_coords[1] + exit_coords[1]) / 2.0
            
            self.floor_plan_elements[elem_name] = {
                's': s_start + elem.length / 2,
                'x_center': elem_center_x,
                'y_center': elem_center_y,
                'entrance': entrance_coords.copy(),
                'exit': exit_coords.copy(),
                'length': elem.length,
                'tangent': new_tangent.copy(),
                'shape': shape
            }
            
            entrance_coords = exit_coords
            tangent_vector = new_tangent
    
    def _adjust_floor_plan_aspect_ratio(self):
        """Adjust floor plan axis limits to maintain reasonable aspect ratio (1:5 to 5:1)."""
        # Get current axis limits
        xlim = self.floorplan_ax.get_xlim()
        ylim = self.floorplan_ax.get_ylim()
        
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        
        # Avoid division by zero
        if x_range == 0 or y_range == 0:
            return
        
        # Calculate current aspect ratio (width:height)
        aspect_ratio = x_range / y_range
        
        # Define min and max acceptable ratios
        min_ratio = 0.25  # 1:4 (height is 5x width)
        max_ratio = 5.0  # 5:1 (width is 5x height)
        
        # If aspect ratio is acceptable, no adjustment needed
        if min_ratio <= aspect_ratio <= max_ratio:
            return
        
        # Calculate center points
        x_center = (xlim[0] + xlim[1]) / 2
        y_center = (ylim[0] + ylim[1]) / 2
        
        # Adjust limits to bring ratio within bounds
        if aspect_ratio < min_ratio:
            # Too tall (narrow), expand width
            new_x_range = y_range * min_ratio
            new_xlim = (x_center - new_x_range / 2, x_center + new_x_range / 2)
            self.floorplan_ax.set_xlim(new_xlim)
        elif aspect_ratio > max_ratio:
            # Too wide (flat), expand height
            new_y_range = x_range / max_ratio
            new_ylim = (y_center - new_y_range / 2, y_center + new_y_range / 2)
            self.floorplan_ax.set_ylim(new_ylim)
    
    def on_branch_changed(self, branch_name: str):
        """Handle branch selection change."""
        self.branch_name = branch_name
        self.update_plots()
    
    def update_plots(self):
        """Update all plots with current data."""
        try:
            # Reset s range to show full lattice
            self.s_range = None
            self.zoom_history.clear()
            self.btn_zoom_out.setEnabled(False)
            self.btn_reset_zoom.setEnabled(False)
            
            # Update lattice view (both floor plan and beamline)
            if self.lattice is not None:
                # Plot floor plan in top axes
                self.floorplan_plotter.plot(self.lattice, self.branch_name)
                
                # Plot beamline in bottom axes
                self.beamline_plotter.plot(self.lattice, self.branch_name)
                
                # Add horizontal reference line at y=0
                self.beamline_ax.axhline(y=0, color='black', linewidth=0.5, linestyle='-', alpha=0.3)
                
                # Store beamline y-axis limits for zoom operations
                self.beamline_ylims = self.beamline_ax.get_ylim()
                
                # Calculate and store element shapes for hover detection
                self._calculate_floor_plan_shapes()
                
                # Adjust floor plan aspect ratio to avoid super narrow plots
                self._adjust_floor_plan_aspect_ratio()
                
                # Redraw canvas
                self.lattice_canvas.draw()
                
                # Refresh element list
                self._populate_element_list()
            
            # Update Twiss
            self.update_twiss_plot()
            
            # Update BPM
            self.update_bpm_plot()
            
            # Update info
            if self.lattice is not None:
                # Get actual element objects from the branch
                element_names = self.lattice.branches[self.branch_name]
                elements = [self.lattice.get_element(name) for name in element_names]
                length = sum(e.length for e in elements)
                n_elements = len(element_names)
                self.label_info.setText(f"Length: {length:.2f}m | Elements: {n_elements}")
            
            self.statusBar().showMessage("Plots updated", 3000)
            
        except Exception as e:
            logger.error(f"Error updating plots: {e}")
            self.statusBar().showMessage(f"Error: {str(e)}", 5000)
    
    def _plot_twiss_function(self, ax, func_type, mask, s_start, s_end, is_middle=False):
        """Plot a specific Twiss function on given axes.
        
        Args:
            ax: Matplotlib axes
            func_type: Type of function ("Beta", "Alpha", "Dispersion", "Phase Advance")
            mask: Boolean mask for s range
            s_start: Start of s range
            s_end: End of s range
            is_middle: True if this is middle plot (different title)
        """
        if func_type == "Beta":
            ax.plot(self.twiss.s[mask], self.twiss.betx[mask], 'b-', label='βx', linewidth=2)
            ax.plot(self.twiss.s[mask], self.twiss.bety[mask], 'r-', label='βy', linewidth=2)
            ax.set_ylabel('Beta function [m]', fontsize=11)
            if not is_middle:
                ax.set_title(f'Optics Functions (Qx={self.twiss.qx:.4f}, Qy={self.twiss.qy:.4f}) - s={s_start:.1f} to {s_end:.1f}m', 
                           fontsize=12, fontweight='bold')
        
        elif func_type == "Alpha":
            ax.plot(self.twiss.s[mask], self.twiss.alfx[mask], 'b-', label='αx', linewidth=2)
            ax.plot(self.twiss.s[mask], self.twiss.alfy[mask], 'r-', label='αy', linewidth=2)
            ax.set_ylabel('Alpha function', fontsize=11)
            if not is_middle:
                ax.set_title(f'Alpha Functions - s={s_start:.1f} to {s_end:.1f}m', fontsize=12, fontweight='bold')
        
        elif func_type == "Dispersion":
            ax.plot(self.twiss.s[mask], self.twiss.dx[mask], 'g-', label='Dx', linewidth=2)
            if self.twiss.dy is not None:
                ax.plot(self.twiss.s[mask], self.twiss.dy[mask], 'm-', label='Dy', linewidth=2)
            ax.set_ylabel('Dispersion [m]', fontsize=11)
            if not is_middle:
                ax.set_title(f'Dispersion - s={s_start:.1f} to {s_end:.1f}m', fontsize=12, fontweight='bold')
        
        elif func_type == "Phase Advance":
            if self.twiss.mu_x is not None and self.twiss.mu_y is not None:
                # Convert to degrees and normalize to [0, 360]
                phase_x = np.degrees(self.twiss.mu_x[mask]) % 360
                phase_y = np.degrees(self.twiss.mu_y[mask]) % 360
                ax.plot(self.twiss.s[mask], phase_x, 'b-', label='μx', linewidth=2)
                ax.plot(self.twiss.s[mask], phase_y, 'r-', label='μy', linewidth=2)
                ax.set_ylabel('Phase advance [deg]', fontsize=11)
            else:
                ax.text(0.5, 0.5, 'Phase advance data not available', 
                       ha='center', va='center', transform=ax.transAxes)
            if not is_middle:
                ax.set_title(f'Phase Advance - s={s_start:.1f} to {s_end:.1f}m', fontsize=12, fontweight='bold')
        
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    def _plot_twiss_lattice(self, ax, s_start, s_end, store_ylims=False):
        """Plot lattice beamline in given s range.
        
        Args:
            ax: Axes to plot on
            s_start: Start s position
            s_end: End s position
            store_ylims: If True, store y-limits after plotting (for initial plot)
        """
        if self.lattice is not None:
            # Get elements in range
            element_names = self.lattice.branches[self.branch_name]
            current_s = 0.0
            for elem_name in element_names:
                elem = self.lattice.get_element(elem_name)
                elem_end = current_s + elem.length
                
                # Check if element overlaps with range
                if elem_end > s_start and current_s < s_end:
                    elem.plot_in_beamline(ax, current_s)
                
                current_s = elem_end
                if current_s > s_end:
                    break
        
        # Add horizontal reference line at y=0
        ax.axhline(y=0, color='black', linewidth=0.5, linestyle='-', alpha=0.3)
        
        ax.set_xlim(s_start, s_end)
        ax.set_xlabel('S position [m]', fontsize=11)
        ax.set_yticks([])
        
        # Store or restore y-limits
        if store_ylims:
            self.twiss_lattice_ylims = ax.get_ylim()
        elif self.twiss_lattice_ylims is not None:
            ax.set_ylim(self.twiss_lattice_ylims)
    
    def update_twiss_plot(self):
        """Update Twiss plot with lattice alignment."""
        if self.twiss is None:
            return
        
        try:
            # Get configuration
            num_plots = int(self.combo_twiss_num_plots.currentText())
            plot1_type = self.combo_twiss_plot1.currentText()
            plot2_type = self.combo_twiss_plot2.currentText()
            
            # Clear axes
            self.twiss_plot1_ax.clear()
            if num_plots == 2:
                self.twiss_plot2_ax.clear()
            self.twiss_lattice_ax.clear()
            
            # Full range mask
            mask = np.ones(len(self.twiss.s), dtype=bool)
            s_start, s_end = self.twiss.s[0], self.twiss.s[-1]
            
            # Plot first function
            self._plot_twiss_function(self.twiss_plot1_ax, plot1_type, mask, s_start, s_end)
            
            # Show/hide second plot based on configuration
            if num_plots == 2:
                self.twiss_plot2_ax.set_visible(True)
                self._plot_twiss_function(self.twiss_plot2_ax, plot2_type, mask, s_start, s_end, is_middle=True)
            else:
                self.twiss_plot2_ax.set_visible(False)
            
            # Store y-limits for later zoom operations
            self.twiss_ylims = {
                'plot1': self.twiss_plot1_ax.get_ylim(),
            }
            if num_plots == 2:
                self.twiss_ylims['plot2'] = self.twiss_plot2_ax.get_ylim()
            
            # Hide x-axis labels on upper plots
            self.twiss_plot1_ax.tick_params(labelbottom=False)
            if num_plots == 2:
                self.twiss_plot2_ax.tick_params(labelbottom=False)
            
            # Plot lattice beamline
            self._plot_twiss_lattice(self.twiss_lattice_ax, s_start, s_end, store_ylims=True)
            
            # Reset zoom state
            self.twiss_zoom_history.clear()
            self.twiss_s_range = None
            
            # Ensure SpanSelectors are initialized (needed for zoom to work)
            if not self.twiss_span_selectors:
                self._reconfigure_twiss_layout()
            
            # Adjust layout to prevent overlap
            self.twiss_fig.tight_layout()
            
            self.twiss_canvas.draw()
            
        except Exception as e:
            logger.error(f"Error updating Twiss plot: {e}")
    
    def update_bpm_plot(self):
        """Update BPM plot."""
        if self.bpm_data is not None and len(self.bpm_data) > 0:
            plot_vs_turn = not self.check_bpm_vs_s.isChecked()
            self.bpm_plotter.plot(self.bpm_data, plot_vs_turn=plot_vs_turn)
            self.bpm_canvas.draw()
    
    def set_data(self, lattice=None, twiss=None, 
                 bpm_data=None, branch_name: Optional[str] = None):
        """
        Update data and refresh plots.
        
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
            
        # Update branch selector
        if self.lattice is not None:
            self.combo_branch.clear()
            self.combo_branch.addItems(list(self.lattice.branches.keys()))
            if branch_name is not None:
                self.combo_branch.setCurrentText(branch_name)
        
        self.update_plots()


def launch_gui(lattice=None, twiss=None, bpm_data=None,
               branch_name: str = "FODO",
               standalone: bool = True) -> Optional[LatticeViewerGUI]:
    """
    Launch the EICViBE lattice viewer GUI.
    
    Engine-agnostic interface - works with data from any simulation engine.
    
    Args:
        lattice: EICViBE Lattice object
        twiss: Twiss data (any object with s, betx, bety, dx, etc. attributes)
        bpm_data: Dictionary of BPM data {bpm_name: {'x': array, 'y': array, 's': float}}
        branch_name: Branch name to display
        standalone: If True, start Qt event loop (for terminal launch)
                   If False, just return window (for notebook integration)
    
    Returns:
        LatticeViewerGUI window instance
    """
    # Create QApplication if needed
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    # Create and show window
    window = LatticeViewerGUI(
        lattice=lattice,
        twiss=twiss,
        bpm_data=bpm_data,
        branch_name=branch_name
    )
    window.show()
    
    if standalone:
        # Run event loop (blocks until window closed)
        sys.exit(app.exec_())
    else:
        # Return window for interactive use
        return window
