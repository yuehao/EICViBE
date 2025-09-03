"""
XSuite simulation engine interface for EICViBE.

Provides integration with XSuite for particle tracking and beam dynamics simulations
with support for LINAC, RING, and RAMPING modes.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Callable
import logging
import time

# Import XSuite packages (handled gracefully if not available)
try:
    import xtrack as xt
    import xpart as xp
    import xobjects as xo
    XSUITE_AVAILABLE = True
except ImportError:
    xt = None
    xp = None  
    xo = None
    XSUITE_AVAILABLE = False

from .base import BaseSimulationEngine, SimulationMode
from eicvibe.machine_portal.lattice import Lattice

logger = logging.getLogger(__name__)


class XSuiteSimulationEngine(BaseSimulationEngine):
    """
    XSuite implementation of the base simulation engine.
    
    Provides XSuite-specific implementations for all three simulation modes:
    LINAC, RING, and RAMPING.
    """
    
    def __init__(self):
        super().__init__("XSuite")
        
        # XSuite specific objects
        self.xtrack_line = None
        self.xpart_particles = None
        self.context = None
        
        # Element mapping for parameter updates
        self.element_map = {}  # {eicvibe_name: xtrack_element}
        self.parameter_map = {}  # {(element, param_group, param_name): (xtrack_element, attribute)}
        self.name_mapping = {}  # {original_name: expanded_name} for parameter lookup
        
        # Tracking configuration
        self.num_particles = 1000
        self.reference_energy = 1e9  # eV
        self.reference_mass = 0.51099895e6  # electron mass in eV
        
        self.logger.info("XSuite simulation engine initialized")
    
    def initialize_engine(self) -> bool:
        """Initialize XSuite engine and check availability."""
        if not XSUITE_AVAILABLE:
            self.logger.error("XSuite packages not available. Please install xtrack, xpart, and xobjects.")
            return False
        
        try:
            # Initialize XSuite context (CPU by default)
            self.context = xo.ContextCpu()
            self.logger.info("XSuite engine initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize XSuite engine: {e}")
            return False
    
    def convert_lattice(self, eicvibe_lattice: Lattice, branch_name: str = "main") -> Any:
        """Convert EICViBE lattice to XTrack Line."""
        try:
            elements = eicvibe_lattice.expand_lattice(branch_name)
            self.logger.info(f"Converting {len(elements)} elements to XTrack format")
            
            # Build XTrack elements
            xtrack_elements = []
            element_names = []
            
            for elem in elements:
                xt_elem, name = self._convert_element(elem)
                if xt_elem is not None:
                    xtrack_elements.append(xt_elem)
                    element_names.append(name)
                    self.element_map[elem.name] = xt_elem
                    # Map original name to expanded name for parameter lookup
                    self.name_mapping[elem.name] = name
            
            # Create XTrack Line
            element_dict = dict(zip(element_names, xtrack_elements))
            self.xtrack_line = xt.Line(elements=element_dict)
            
            # Build the line and set up for tracking
            self.xtrack_line.build_tracker(_context=self.context)
            
            # Setup parameter mapping with both expanded and original elements
            original_elements = list(eicvibe_lattice.elements.values())
            self._setup_parameter_mapping(elements, original_elements)
            
            self.logger.info(f"Successfully converted lattice to XTrack Line with {len(xtrack_elements)} elements")
            return self.xtrack_line
            
        except Exception as e:
            self.logger.error(f"Failed to convert lattice: {e}")
            return None
    
    def create_particles(self, particle_params: Dict[str, Any]) -> Any:
        """Create XPart particle distribution."""
        try:
            # Extract parameters
            self.num_particles = particle_params.get('num_particles', 1000)
            self.reference_energy = particle_params.get('energy', 1e9)  # eV
            
            # Create initial particle distribution
            self.xpart_particles = xp.Particles(
                _context=self.context,
                p0c=self.reference_energy,
                x=particle_params.get('x', np.zeros(self.num_particles)),
                px=particle_params.get('px', np.zeros(self.num_particles)),
                y=particle_params.get('y', np.zeros(self.num_particles)),
                py=particle_params.get('py', np.zeros(self.num_particles)),
                zeta=particle_params.get('zeta', np.zeros(self.num_particles)),
                delta=particle_params.get('delta', np.zeros(self.num_particles))
            )
            
            self.logger.info(f"Created {self.num_particles} particles with energy {self.reference_energy/1e9:.3f} GeV")
            return self.xpart_particles
            
        except Exception as e:
            self.logger.error(f"Failed to create particles: {e}")
            return None
    
    def track_single_turn(self) -> bool:
        """Execute one turn of tracking with XTrack."""
        try:
            if self.xtrack_line is None or self.xpart_particles is None:
                self.logger.error("Lattice or particles not initialized")
                return False
            
            # Track particles for one turn
            self.xtrack_line.track(self.xpart_particles, num_turns=1)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Tracking failed: {e}")
            return False
    
    def get_particle_coordinates(self) -> Dict[str, np.ndarray]:
        """Get current particle coordinates from XPart."""
        try:
            if self.xpart_particles is None:
                return {}
            
            # Extract coordinates from XPart particles
            coords = {
                'x': np.array(self.xpart_particles.x),
                'px': np.array(self.xpart_particles.px),
                'y': np.array(self.xpart_particles.y),
                'py': np.array(self.xpart_particles.py),
                'zeta': np.array(self.xpart_particles.zeta),
                'delta': np.array(self.xpart_particles.delta),
                'state': np.array(self.xpart_particles.state)
            }
            
            return coords
            
        except Exception as e:
            self.logger.error(f"Failed to get particle coordinates: {e}")
            return {}
    
    def update_element_parameter(self, element_name: str, param_group: str, 
                                param_name: str, value: float) -> bool:
        """Update element parameter in XTrack lattice."""
        try:
            param_key = (element_name, param_group, param_name)
            
            if param_key in self.parameter_map:
                element, attr_info = self.parameter_map[param_key]
                
                # Handle array-based attributes (like knl[1] for quadrupoles) or direct attributes
                if isinstance(attr_info, tuple):
                    attr_name, index = attr_info
                    if index is not None:
                        # Array-based attribute (e.g., knl[1])
                        array = getattr(element, attr_name)
                        array[index] = value
                    else:
                        # Direct attribute (e.g., k1)
                        setattr(element, attr_name, value)
                else:
                    # Handle simple attributes
                    setattr(element, attr_info, value)
                    
                self.logger.debug(f"Updated {element_name}.{param_group}.{param_name} = {value}")
                return True
            else:
                self.logger.warning(f"Parameter {param_key} not found in parameter map")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to update parameter {element_name}.{param_group}.{param_name}: {e}")
            return False
    
    def get_element_parameter(self, element_name: str, param_group: str, 
                             param_name: str) -> float:
        """Get current element parameter value from XTrack lattice."""
        try:
            param_key = (element_name, param_group, param_name)
            
            if param_key in self.parameter_map:
                element, attr_info = self.parameter_map[param_key]
                
                # Handle array-based attributes (like knl[1] for quadrupoles) or direct attributes  
                if isinstance(attr_info, tuple):
                    attr_name, index = attr_info
                    if index is not None:
                        # Array-based attribute (e.g., knl[1])
                        array = getattr(element, attr_name)
                        value = array[index]
                    else:
                        # Direct attribute (e.g., k1)
                        value = getattr(element, attr_name)
                else:
                    # Handle simple attributes
                    value = getattr(element, attr_info)
                    
                return float(value)
            else:
                self.logger.warning(f"Parameter {param_key} not found in parameter map")
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Failed to get parameter {element_name}.{param_group}.{param_name}: {e}")
            return 0.0
    
    def cleanup_engine(self) -> None:
        """Clean up XSuite resources."""
        try:
            self.xtrack_line = None
            self.xpart_particles = None
            self.element_map.clear()
            self.parameter_map.clear()
            self.logger.info("XSuite engine cleaned up")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    # Private helper methods
    
    def _convert_element(self, eicvibe_element) -> Tuple[Any, str]:
        """Convert single EICViBE element to XTrack element."""
        try:
            elem_type = eicvibe_element.type.lower()
            name = eicvibe_element.name
            length = eicvibe_element.length
            
            if elem_type == 'drift':
                return xt.Drift(length=length), name
                
            elif elem_type == 'quadrupole':
                # Get k1 value from MagneticMultipoleP parameter group
                k1 = eicvibe_element.get_parameter("MagneticMultipoleP", "kn1") or 0.0
                return xt.Quadrupole(length=length, k1=k1), name
                
            elif elem_type == 'sextupole':
                k2 = eicvibe_element.parameters.get('multipole', {}).get('k2l', 0.0) / length if length > 0 else 0.0
                return xt.Sextupole(length=length, k2=k2), name
                
            elif elem_type == 'octupole':
                k3 = eicvibe_element.parameters.get('multipole', {}).get('k3l', 0.0) / length if length > 0 else 0.0
                return xt.Octupole(length=length, k3=k3), name
                
            elif elem_type in ['sbend', 'rbend', 'bend']:
                angle = eicvibe_element.parameters.get('bending', {}).get('angle', 0.0)
                k1 = eicvibe_element.parameters.get('multipole', {}).get('k1l', 0.0) / length if length > 0 else 0.0
                return xt.Bend(length=length, k0=angle/length if length > 0 else 0.0, k1=k1), name
                
            elif elem_type == 'rfcavity':
                voltage = eicvibe_element.parameters.get('rf', {}).get('voltage', 0.0)
                frequency = eicvibe_element.parameters.get('rf', {}).get('frequency', 500e6)
                lag = eicvibe_element.parameters.get('rf', {}).get('lag', 0.0)
                return xt.Cavity(voltage=voltage, frequency=frequency, lag=lag), name
                
            elif elem_type in ['monitor', 'bpm']:
                # Use BeamPositionMonitor for BPM functionality
                return xt.BeamPositionMonitor(
                    start_at_turn=0,
                    stop_at_turn=1000,  # Will be updated based on simulation
                    frev=1e6,  # Default, will be updated
                    sampling_frequency=1e6  # Default, will be updated
                ), name
                
            elif elem_type == 'marker':
                return xt.Marker(), name
                
            elif elem_type == 'kicker':
                hkick = eicvibe_element.parameters.get('kicker', {}).get('hkick', 0.0)
                vkick = eicvibe_element.parameters.get('kicker', {}).get('vkick', 0.0)
                return xt.XYShift(dx=hkick, dy=vkick), name
                
            else:
                # Unknown element type - treat as drift
                self.logger.warning(f"Unknown element type '{elem_type}' for {name}, treating as drift")
                return xt.Drift(length=length), name
                
        except Exception as e:
            self.logger.error(f"Failed to convert element {eicvibe_element.name}: {e}")
            return xt.Drift(length=0.0), eicvibe_element.name
    
    def _setup_parameter_mapping(self, eicvibe_elements, original_elements=None) -> None:
        """Setup mapping between EICViBE parameters and XTrack attributes."""
        self.parameter_map.clear()
        
        # If original_elements is not provided, use eicvibe_elements
        if original_elements is None:
            original_elements = []
        
        for eicvibe_elem in eicvibe_elements:
            if eicvibe_elem.name not in self.element_map:
                continue
                
            xt_elem = self.element_map[eicvibe_elem.name]
            elem_type = eicvibe_elem.type.lower()
            
            # Map both original name and expanded name for parameter lookup
            original_name = eicvibe_elem.name
            expanded_name = self.name_mapping.get(original_name, original_name)
            
            # Map common parameters based on element type
            if elem_type == 'quadrupole':
                # XSuite Quadrupole uses k1 parameter directly (not knl array)
                self.parameter_map[(original_name, 'MagneticMultipoleP', 'kn1')] = (xt_elem, ('k1', None))
                # Also map expanded name for compatibility
                if expanded_name != original_name:
                    self.parameter_map[(expanded_name, 'MagneticMultipoleP', 'kn1')] = (xt_elem, ('k1', None))
                
            elif elem_type == 'sextupole':
                # k2 is stored in knl[2] (index 2 for sextupole)
                self.parameter_map[(original_name, 'multipole', 'k2l')] = (xt_elem, ('knl', 2))
                if expanded_name != original_name:
                    self.parameter_map[(expanded_name, 'multipole', 'k2l')] = (xt_elem, ('knl', 2))
                
            elif elem_type == 'octupole':
                # k3 is stored in knl[3] (index 3 for octupole)
                self.parameter_map[(original_name, 'multipole', 'k3l')] = (xt_elem, ('knl', 3))
                if expanded_name != original_name:
                    self.parameter_map[(expanded_name, 'multipole', 'k3l')] = (xt_elem, ('knl', 3))
                
            elif elem_type in ['sbend', 'rbend', 'bend']:
                self.parameter_map[(original_name, 'bending', 'angle')] = (xt_elem, 'k0')
                if expanded_name != original_name:
                    self.parameter_map[(expanded_name, 'bending', 'angle')] = (xt_elem, 'k0')
                self.parameter_map[(original_name, 'multipole', 'k1l')] = (xt_elem, 'k1')
                if expanded_name != original_name:
                    self.parameter_map[(expanded_name, 'multipole', 'k1l')] = (xt_elem, 'k1')
                
            elif elem_type == 'rfcavity':
                self.parameter_map[(original_name, 'rf', 'voltage')] = (xt_elem, 'voltage')
                self.parameter_map[(original_name, 'rf', 'frequency')] = (xt_elem, 'frequency')
                self.parameter_map[(original_name, 'rf', 'lag')] = (xt_elem, 'lag')
                if expanded_name != original_name:
                    self.parameter_map[(expanded_name, 'rf', 'voltage')] = (xt_elem, 'voltage')
                    self.parameter_map[(expanded_name, 'rf', 'frequency')] = (xt_elem, 'frequency')
                    self.parameter_map[(expanded_name, 'rf', 'lag')] = (xt_elem, 'lag')
                
            elif elem_type == 'kicker':
                if hasattr(xt_elem, 'dx'):
                    self.parameter_map[(original_name, 'kicker', 'hkick')] = (xt_elem, 'dx')
                    if expanded_name != original_name:
                        self.parameter_map[(expanded_name, 'kicker', 'hkick')] = (xt_elem, 'dx')
                if hasattr(xt_elem, 'dy'):
                    self.parameter_map[(original_name, 'kicker', 'vkick')] = (xt_elem, 'dy')
                    if expanded_name != original_name:
                        self.parameter_map[(expanded_name, 'kicker', 'vkick')] = (xt_elem, 'dy')
        
        # After processing all elements, also add mappings for the original element names
        # This handles the case where original elements (QF, QD) aren't in element_map 
        # but their expanded versions (QF_1, QD_1) are
        additional_mappings = {}
        for (elem_name, param_group, param_name), mapping in self.parameter_map.items():
            # If this is an expanded name (ends with _1), also create mapping for original name
            if elem_name.endswith('_1'):
                original_name = elem_name[:-2]  # Remove '_1' suffix
                # Check if original element exists in EICViBE lattice but not in our mapping
                if (original_name, param_group, param_name) not in self.parameter_map:
                    # Find the original element in the original elements list to verify it exists
                    original_exists = any(elem.name == original_name for elem in original_elements)
                    if original_exists:
                        additional_mappings[(original_name, param_group, param_name)] = mapping
        
        # Add the additional mappings
        self.parameter_map.update(additional_mappings)
        
        self.logger.info(f"Setup parameter mapping for {len(self.parameter_map)} parameters")

    def get_parameter(self, element_name: str, param_group: str, param_name: str) -> float:
        """
        Get a parameter value from the simulation.
        
        Args:
            element_name: Name of the element
            param_group: Parameter group name
            param_name: Parameter name
            
        Returns:
            Current parameter value
        """
        key = (element_name, param_group, param_name)
        if key not in self.parameter_map:
            self.logger.warning(f"Parameter {key} not found in parameter map")
            return 0.0
            
        element, attr_path = self.parameter_map[key]
        attr_name, sub_attr = attr_path if isinstance(attr_path, tuple) else (attr_path, None)
        
        try:
            value = getattr(element, attr_name)
            if sub_attr is not None:
                value = value[sub_attr]
            return float(value)
        except Exception as e:
            self.logger.error(f"Error getting parameter {key}: {e}")
            return 0.0

    def set_parameter(self, element_name: str, param_group: str, param_name: str, value: float) -> bool:
        """
        Set a parameter value in the simulation.
        
        Args:
            element_name: Name of the element
            param_group: Parameter group name
            param_name: Parameter name
            value: New parameter value
            
        Returns:
            True if successful, False otherwise
        """
        key = (element_name, param_group, param_name)
        if key not in self.parameter_map:
            self.logger.warning(f"Parameter {key} not found in parameter map")
            return False
            
        element, attr_path = self.parameter_map[key]
        attr_name, sub_attr = attr_path if isinstance(attr_path, tuple) else (attr_path, None)
        
        try:
            if sub_attr is not None:
                # Handle array/sub-attribute access
                current_array = getattr(element, attr_name)
                current_array[sub_attr] = value
                setattr(element, attr_name, current_array)
            else:
                # Direct attribute access
                setattr(element, attr_name, value)
            return True
        except Exception as e:
            self.logger.error(f"Error setting parameter {key}: {e}")
            return False

    @property
    def line(self):
        """Access to the XTrack line object."""
        return self.xtrack_line


# Convenience function for creating XSuite simulation engine
def create_xsuite_engine() -> XSuiteSimulationEngine:
    """Create and return a new XSuite simulation engine instance."""
    return XSuiteSimulationEngine()


# Example usage and testing functions
def test_xsuite_engine():
    """Test function for XSuite engine functionality."""
    print("Testing XSuite Simulation Engine...")
    
    # Create engine
    engine = create_xsuite_engine()
    
    # Test initialization
    if not engine.initialize_engine():
        print("❌ Engine initialization failed")
        return False
    
    print("✅ Engine initialization successful")
    
    # Test with simple FODO lattice (would need actual Lattice object)
    # This is just a placeholder for testing
    
    return True


if __name__ == "__main__":
    test_xsuite_engine()
