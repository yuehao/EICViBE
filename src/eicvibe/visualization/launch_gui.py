#!/usr/bin/env python3
"""
Terminal launcher for EICViBE Lattice Viewer GUI.

Usage:
    python -m eicvibe.visualization.launch_gui [--lattice LATTICE_FILE]
    
Or from command line after installation:
    eicvibe-viewer [--lattice LATTICE_FILE]
"""

import sys
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for GUI launcher."""
    parser = argparse.ArgumentParser(
        description="EICViBE Lattice Viewer - Interactive GUI for accelerator lattice visualization"
    )
    parser.add_argument(
        '--lattice',
        type=str,
        help='Path to MAD-X lattice file to load'
    )
    parser.add_argument(
        '--branch',
        type=str,
        default='FODO',
        help='Branch name to display (default: FODO)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Import here to avoid loading Qt if just showing help
    try:
        from .gui_app import launch_gui
        from eicvibe.machine_portal.lattice import Lattice
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        logger.error("Make sure PyQt5 is installed: pip install PyQt5")
        sys.exit(1)
    
    # Load lattice if provided
    lattice = None
    twiss = None
    bpm_data = None
    
    if args.lattice:
        lattice_path = Path(args.lattice)
        if not lattice_path.exists():
            logger.error(f"Lattice file not found: {lattice_path}")
            sys.exit(1)
        
        logger.info(f"Loading lattice from {lattice_path}")
        try:
            # Import MAD-X lattice (EICViBE level)
            from eicvibe.utilities.madx_import import import_madx_lattice
            lattice = import_madx_lattice(str(lattice_path))
            logger.info(f"Successfully loaded lattice with {len(lattice.elements)} elements")
            
            # Optionally calculate Twiss using available simulation engine
            # This demonstrates engine-agnostic approach: GUI gets data, not engine
            try:
                from eicvibe.simulators.xsuite_interface import XSuiteSimulationEngine
                from eicvibe.simulators.types import SimulationMode
                
                engine = XSuiteSimulationEngine()
                if engine.initialize_engine():
                    xsuite_line = engine.convert_lattice(
                        lattice,
                        mode=SimulationMode.RING,
                        reference_energy=18e9,
                        reference_species="proton"
                    )
                    # Extract Twiss data (engine-agnostic result)
                    twiss = xsuite_line.twiss(method='4d')
                    logger.info("Successfully calculated Twiss parameters using XSuite")
                else:
                    logger.warning("Failed to initialize XSuite engine")
            except Exception as e:
                logger.warning(f"Could not calculate Twiss parameters: {e}")
                logger.info("GUI will display lattice without optics data")
                
        except Exception as e:
            logger.error(f"Failed to load lattice: {e}")
            sys.exit(1)
    
    # Launch GUI (engine-agnostic interface)
    logger.info("Launching GUI...")
    try:
        launch_gui(
            lattice=lattice,
            twiss=twiss,
            bpm_data=bpm_data,
            branch_name=args.branch,
            standalone=True
        )
    except Exception as e:
        logger.error(f"GUI failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
