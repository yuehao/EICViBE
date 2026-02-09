# EICViBE Lattice Viewer GUI

This directory contains the interactive GUI components for visualizing EICViBE lattices.

## Features

- **Engine-Agnostic**: Works at EICViBE level with any simulation engine
  - Reads results from simulation engines (XSuite, future engines)
  - Not tied to any specific engine implementation
- **Modular Design**: Reusable plotting components that can be integrated anywhere
- **Multi-Platform**: Works in Jupyter notebooks and as standalone terminal application
- **Interactive Visualization**:
  - Beamline layout (1D element view)
  - Floor plan (2D physical layout)
  - Twiss parameters (beta functions, dispersion)
  - Turn-by-turn BPM readings
  - Toggle between different plot modes

## Components

### Core Plotting Modules

- **`lattice_viewer.py`**: Base plotting components
  - `BeamlinePlotter`: 1D beamline layout
  - `FloorPlanPlotter`: 2D floor plan
  - `TwissPlotter`: Twiss parameter visualization
  - `BPMPlotter`: Turn-by-turn BPM data

### Interfaces

- **`notebook_viewer.py`**: Matplotlib-based interface for Jupyter notebooks
  - Works inline in notebooks
  - No external dependencies beyond matplotlib
  
- **`gui_app.py`**: PyQt5-based standalone GUI
  - Full-featured interactive application
  - Tabs for different views
  - Control panel with options
  - Requires PyQt5

- **`launch_gui.py`**: Terminal launcher script

## Usage

### In Jupyter Notebooks

```python
from eicvibe.visualization import create_notebook_viewer

# Engine-agnostic interface: pass EICViBE Lattice and data from any engine
# Example with XSuite results:
xsuite_line = engine.convert_lattice(lattice, ...)
twiss = xsuite_line.twiss()
bpm_data = engine.get_bpm_data()  # or extract manually

# Create viewer with EICViBE lattice and engine results
viewer = create_notebook_viewer(
    lattice=lattice,      # EICViBE Lattice object
    twiss=twiss,          # Twiss data (any object with s, betx, bety, dx)
    bpm_data=bpm_data,    # BPM data dictionary
    branch_name="FODO"
)

# Display all plots
viewer.show_all()

# Or display individual plots
viewer.show_beamline()
viewer.show_twiss()
viewer.show_bpm(plot_vs_turn=True)
```

### Standalone GUI (PyQt5)

From notebook:
```python
from eicvibe.visualization import launch_gui, GUI_AVAILABLE

if GUI_AVAILABLE:
    window = launch_gui(
        lattice=lattice,
        twiss=twiss,
        bpm_data=bpm_data,
        standalone=False  # Non-blocking for notebooks
    )
```

From terminal:
```bash
# Launch GUI with lattice file
python -m eicvibe.visualization.launch_gui --lattice path/to/lattice.madx

# Or use as command (after installation)
eicvibe-viewer --lattice path/to/lattice.madx --branch FODO
```

### Using Individual Plotters

```python
from eicvibe.visualization import BeamlinePlotter, TwissPlotter

# Create custom plots
beamline_plotter = BeamlinePlotter(figsize=(14, 3))
fig, ax = beamline_plotter.create_figure()
beamline_plotter.plot(lattice, "FODO")
plt.show()
```

## Installation

Basic visualization (notebooks only):
```bash
# No additional dependencies needed
```

Full GUI support:
```bash
pip install PyQt5
```

## Architecture

The visualization system follows a modular, layered design:

1. **Plotting Components** (`lattice_viewer.py`):
   - Pure matplotlib plotting logic
   - Reusable across different interfaces
   - No Qt or notebook dependencies

2. **Notebook Interface** (`notebook_viewer.py`):
   - Wraps plotting components for notebook use
   - Matplotlib-only, no external GUI framework

3. **GUI Application** (`gui_app.py`):
   - PyQt5-based windowed application
   - Uses plotting components in Qt widgets
   - Full interactive controls

This design ensures:
- **Modularity**: Components can be reused independently
- **Flexibility**: Works in multiple environments
- **Maintainability**: Clear separation of concerns
- **Graceful degradation**: Notebook viewer works even without PyQt5

## Development

To extend the GUI:

1. Add new plotting component to `lattice_viewer.py`
2. Integrate into `NotebookViewer` for notebook support
3. Add tab/panel to `LatticeViewerGUI` for full GUI support

Example:
```python
# 1. Add to lattice_viewer.py
class PhaseSpacePlotter:
    def plot(self, particles):
        # Plotting logic
        pass

# 2. Add to NotebookViewer
class NotebookViewer:
    def show_phase_space(self):
        self.phase_space_plotter.plot(self.particles)

# 3. Add to LatticeViewerGUI
class LatticeViewerGUI:
    def _create_phase_space_tab(self):
        # Create tab with PhaseSpacePlotter
        pass
```

## Examples

See [examples/FODO/FODO_lattice_change.ipynb](../../examples/FODO/FODO_lattice_change.ipynb) for complete usage examples.
