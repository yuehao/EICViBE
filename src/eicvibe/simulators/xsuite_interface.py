"""
XSuite simulation engine interface for EICViBE.

This module provides the XSuite implementation of the BaseSimulationEngine interface,
enabling particle tracking simulations using the XSuite/XTrack framework.

References:
    - XSuite Documentation: https://xsuite.readthedocs.io/
    - XSuite API: https://xsuite.readthedocs.io/en/latest/apireference.html
"""

import logging
import yaml
import numpy as np
import multiprocessing as mp
from typing import Any, Dict, Optional, Tuple, List
from pathlib import Path

from ..machine_portal.lattice import Lattice
from ..machine_portal.element import Element
from ..models.parameter_groups import (
    MagneticMultipoleP,
    BendP,
    RFP,
    ApertureP
)
from .base import BaseSimulationEngine
from .types import (
    SimulationMode,
    ParticleDistribution,
    TrackingParameters,
    TrackingResults,
    TwissData,
    BeamPositionData,
    LatticeChangeAction,
    LatticeConversionError,
    TrackingError
)

logger = logging.getLogger(__name__)

# Try to import XSuite components
try:
    import xtrack as xt
    import xpart as xp
    import xobjects as xo
    XSUITE_AVAILABLE = True
    logger.info("XSuite is available")
except ImportError:
    XSUITE_AVAILABLE = False
    logger.warning("XSuite is not available. Install with: pip install xsuite")
    xt = None
    xp = None
    xo = None


class XSuiteSimulationEngine(BaseSimulationEngine):
    """
    XSuite-based simulation engine for EICViBE.
    
    This engine converts EICViBE lattices to XTrack Line objects and performs
    particle tracking simulations using the XSuite framework.
    
    Attributes:
        interface_config: Configuration mapping EICViBE to XSuite elements
        _context: XSuite computation context (CPU/CUDA/OpenCL)
        _line: Current XTrack Line object
        _particles: Current XPart Particles object
    """
    
    def __init__(self, name: str = "xsuite", config: Optional[Dict[str, Any]] = None):
        """
        Initialize XSuite simulation engine.
        
        Args:
            name: Engine identifier
            config: Optional configuration dictionary
            
        Raises:
            ImportError: If XSuite is not available
        """
        if not XSUITE_AVAILABLE:
            raise ImportError(
                "XSuite is required for this engine. "
                "Install with: pip install xsuite"
            )
        
        # Create engine configuration
        from .base import EngineConfiguration
        engine_config = EngineConfiguration(
            name=name,
            **(config or {})
        )
        super().__init__(engine_config)
        
        # Load interface configuration
        interface_path = Path(__file__).parent / "interface.yaml"
        with open(interface_path, 'r') as f:
            self.interface_config = yaml.safe_load(f)
        
        # Initialize XSuite components
        self._context: Optional[Any] = None
        self._line: Optional[Any] = None
        self._particles: Optional[Any] = None
        self._bpm_monitors: Dict[str, Any] = {}  # Track BPM monitor objects by name
        
        logger.info(f"XSuiteSimulationEngine '{name}' initialized")
    
    def initialize_engine(self) -> bool:
        """
        Initialize the XSuite computation context.
        
        Returns:
            True if initialization successful
        """
        try:
            # Create CPU context by default
            # Can be upgraded to CUDA/OpenCL based on parameters
            self._context = xo.ContextCpu()
            logger.info("XSuite context initialized (CPU)")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize XSuite context: {e}")
            return False
    
    def cleanup_engine(self):
        """Clean up XSuite resources."""
        self._line = None
        self._particles = None
        self._context = None
        self._bpm_monitors = {}
        logger.info("XSuite engine resources cleaned up")
    
    def convert_lattice(
        self, 
        lattice: Lattice, 
        mode: SimulationMode,
        num_slices: int = 10,
        bpm_num_turns: Optional[int] = None,
        bpm_frev: float = 1e6,
        bpm_sampling_frequency: Optional[float] = None,
        reference_energy: Optional[float] = None,
        reference_species: str = "proton"
    ) -> Any:
        """
        Convert EICViBE Lattice to XTrack Line.
        
        Args:
            lattice: EICViBE Lattice object
            mode: Simulation mode for optimization
            num_slices: Number of uniform slices for thick elements (default: 100)
            bpm_num_turns: Number of turns for BPM recording (optional, can be set later)
            bpm_frev: Revolution frequency for BPMs in Hz (default: 1 MHz)
            bpm_sampling_frequency: BPM sampling frequency in Hz (default: same as frev)
            reference_energy: Reference beam energy in eV (optional, sets line.particle_ref)
            reference_species: Particle species - "proton", "electron", or "ion" (default: "proton")
            
        Returns:
            XTrack Line object with optional reference particle set
            
        Raises:
            LatticeConversionError: If conversion fails
        """
        try:
            # Store BPM configuration for use during element conversion
            self._bpm_config = {
                'num_turns': bpm_num_turns,
                'frev': bpm_frev,
                'sampling_frequency': bpm_sampling_frequency if bpm_sampling_frequency is not None else bpm_frev
            }
            
            # Expand lattice to get flat element list
            elements_list = lattice.expand_lattice(force_expand=True)
            
            if not elements_list:
                raise LatticeConversionError("Lattice expansion returned empty list")
            
            # Create XTrack elements
            elements = []
            element_names = []
            
            for element in elements_list:
                xsuite_element = self._convert_element(element)
                if xsuite_element is not None:
                    elements.append(xsuite_element)
                    element_names.append(element.name)
                    
                    # Special handling: If Monitor has non-zero length, add a drift
                    # because XSuite BeamPositionMonitor must be zero-length
                    if element.type == 'Monitor' and element.length > 0:
                        drift = xt.Drift(length=element.length)
                        elements.append(drift)
                        element_names.append(f"{element.name}_drift")
                        logger.debug(f"Added drift of {element.length:.6f} m after Monitor {element.name}")
                    
                    # Special handling: If RFCavity has non-zero length, add a drift
                    # because XSuite Cavity must be zero-length (thin element)
                    if element.type == 'RFCavity' and element.length > 0:
                        drift = xt.Drift(length=element.length)
                        elements.append(drift)
                        element_names.append(f"{element.name}_drift")
                        logger.debug(f"Added drift of {element.length:.6f} m after RFCavity {element.name}")
            
            # Create XTrack line
            line = xt.Line(elements=elements, element_names=element_names)
            
            # Slice thick elements into thin elements for tracking
            slicing_strategies = [
                xt.Strategy(slicing=xt.Teapot(num_slices)),  # For bends
                xt.Strategy(slicing=xt.Uniform(num_slices)),
            ]
            line.slice_thick_elements(slicing_strategies)
            
            # Set reference particle if energy provided
            if reference_energy is not None:
                # Determine particle mass based on species
                if reference_species.lower() == "proton":
                    mass0 = xp.PROTON_MASS_EV
                    q0 = 1
                elif reference_species.lower() == "electron":
                    mass0 = xp.ELECTRON_MASS_EV
                    q0 = -1
                elif reference_species.lower() == "ion":
                    # Default to proton mass for generic ion (user can customize)
                    mass0 = xp.PROTON_MASS_EV
                    q0 = 1
                    logger.warning("Using proton mass for 'ion' species. Override if needed.")
                else:
                    raise ValueError(f"Unknown particle species: {reference_species}")
                
                # Create reference particle
                line.particle_ref = xp.Particles(
                    p0c=reference_energy,
                    q0=q0,
                    mass0=mass0
                )
                logger.info(f"Set reference particle: {reference_species} at {reference_energy/1e9:.2f} GeV")
            elif hasattr(lattice, 'reference_particle'):
                # Fallback: use lattice-defined reference particle
                # TODO: Implement reference particle conversion from lattice
                pass
            
            logger.info(f"Converted lattice to XTrack Line with {len(elements)} elements (sliced)")
            self._line = line
            return line
            
        except Exception as e:
            raise LatticeConversionError(f"Failed to convert lattice: {e}")
    
    def _convert_element(self, element: Element) -> Optional[Any]:
        """
        Convert a single EICViBE Element to XSuite element.
        
        Args:
            element: EICViBE Element object
            
        Returns:
            XSuite element or None if not supported
        """
        element_type = element.type
        
        # Get XSuite configuration for this element type
        if element_type not in self.interface_config:
            logger.warning(f"Element type '{element_type}' not in interface config")
            return None
        
        xsuite_config = self.interface_config[element_type].get('XSuite')
        if not xsuite_config:
            logger.warning(f"No XSuite mapping for '{element_type}'")
            return None
        
        # Get XSuite element class
        xsuite_type = xsuite_config['type']
        
        # Special handling for BeamPositionMonitor - create with configuration
        if xsuite_type == 'BeamPositionMonitor':
            default_values = xsuite_config.get('default_values', {})
            params = {
                'start_at_turn': default_values.get('start_at_turn', 0),
            }
            
            # Apply BPM configuration if provided during convert_lattice()
            bpm_config = getattr(self, '_bpm_config', {})
            if bpm_config.get('num_turns') is not None:
                params['stop_at_turn'] = bpm_config['num_turns']
                params['frev'] = bpm_config.get('frev', 1e6)
                params['sampling_frequency'] = bpm_config.get('sampling_frequency', params['frev'])
                logger.debug(f"Creating BPM {element.name} with stop_at_turn={params['stop_at_turn']}, frev={params['frev']:.2e}")
            
            # TODO: Allow override via MonitorP parameter group if needed
            bpm_monitor = xt.BeamPositionMonitor(**params)
            self._bpm_monitors[element.name] = bpm_monitor
            logger.debug(f"Created BeamPositionMonitor for {element.name}")
            return bpm_monitor
        
        xsuite_class = getattr(xt, xsuite_type, None)
        if not xsuite_class:
            logger.error(f"XSuite class '{xsuite_type}' not found")
            return None
        
        # Build parameters dictionary
        params = {}
        param_mappings = xsuite_config.get('parameters', {})
        
        for eicvibe_param, xsuite_param in param_mappings.items():
            # Skip knl/ksl for Multipole - they're handled specially below
            if xsuite_type == 'Multipole' and xsuite_param in ('knl', 'ksl'):
                continue
            
            # Handle parameter group notation (e.g., "MagneticMultipoleP.kn1")
            if '.' in eicvibe_param:
                group_name, param_name = eicvibe_param.split('.', 1)
                value = element.get_parameter(group_name, param_name)
            else:
                # Direct attribute access
                value = getattr(element, eicvibe_param, None)
            
            if value is not None:
                params[xsuite_param] = value
        
        # Special handling for Bend elements: set both h and k0 if k0 was not explicitly provided
        # Standard XSuite pattern for pure dipole: h = k0 = angle/length
        # if xsuite_type == 'Bend' and 'k0' not in params and 'angle' in params and 'length' in params:
        #     bend_length = params['length']
        #     bend_angle = params['angle']
        #     if bend_length > 0:
        #         # h will be automatically calculated from angle and length by XSuite
        #         # Set k0 to match h for a pure dipole field
        #         params['h'] = bend_angle / bend_length
        #         params['k0'] = params['h']
        #         logger.debug(f"Set h={params['h']:.6f} and k0={params['k0']:.6f} for bend {element.name}")
        
        # Special handling for Multipole: build knl/ksl arrays
        if xsuite_type == 'Multipole':
            # XSuite Multipole expects knl and ksl as arrays
            knl = []
            ksl = []
            
            # Check if this is a Kicker (uses KickerP) or generic Multipole (uses MagneticMultipoleP)
            if element.type == 'Kicker':
                # Kicker: hkick → knl[0], vkick → ksl[0]
                hkick = element.get_parameter('KickerP', 'hkick')
                vkick = element.get_parameter('KickerP', 'vkick')
                if hkick is not None and hkick != 0:
                    knl = [hkick]
                if vkick is not None and vkick != 0:
                    ksl = [vkick]
            else:
                # Generic Multipole: collect all kn0, kn1, kn2, ... and build arrays
                for i in range(10):  # Support up to decapole
                    kn_val = element.get_parameter('MagneticMultipoleP', f'kn{i}')
                    ks_val = element.get_parameter('MagneticMultipoleP', f'ks{i}')
                    if kn_val is not None:
                        # Extend array to include this order
                        while len(knl) <= i:
                            knl.append(0.0)
                        knl[i] = kn_val
                    if ks_val is not None:
                        while len(ksl) <= i:
                            ksl.append(0.0)
                        ksl[i] = ks_val
            
            # Set arrays - at least knl must be present for XSuite Multipole
            if knl:
                params['knl'] = knl
            if ksl:
                params['ksl'] = ksl
            
            # If both empty, create zero-strength multipole with knl=[0.0]
            if not knl and not ksl:
                params['knl'] = [0.0]
                logger.debug(f"Multipole {element.name} has no magnetic strengths - creating zero-strength element")
            elif not knl:
                # If only ksl has values, still need knl for XSuite
                params['knl'] = [0.0]
            
            # Ensure length is set (it should already be in params from param_mappings)
            if 'length' not in params:
                params['length'] = element.length
            
            # CRITICAL: Set isthick=True for Multipoles with non-zero length
            # Without this, XSuite's get_s_position() treats them as thin elements
            # even though they have a length attribute, causing incorrect line length calculations
            if params.get('length', 0) > 0:
                params['isthick'] = True
            
            logger.debug(f"Created Multipole {element.name} with length={params.get('length', 0):.3f}, isthick={params.get('isthick', False)}, knl={params.get('knl', [])}, ksl={params.get('ksl', [])}")
        
        # Debug output for RFCavity/Cavity conversion
        if xsuite_type == 'Cavity':
            logger.info(f"Converting RFCavity {element.name}:")
            logger.info(f"  Parameters to be used: {params}")
            if not params:
                logger.warning(f"  No parameters found for RFCavity {element.name}!")
                logger.warning(f"  Element has parameter groups: {element.get_all_parameters() if hasattr(element, 'get_all_parameters') else 'N/A'}")
        
        # Create XSuite element
        logger.debug(f"Attempting to create {xsuite_type} for {element.name} with params: {params}")
        try:
            xsuite_element = xsuite_class(**params)
            logger.debug(f"✓ Successfully converted {element.name} ({element_type}) to {xsuite_type}")
            return xsuite_element
        except Exception as e:
            logger.error(f"✗ Failed to create {xsuite_type} for {element.name}: {e}")
            logger.error(f"  Element type: {element_type}")
            logger.error(f"  Parameters were: {params}")
            import traceback
            logger.error(f"  Traceback: {traceback.format_exc()}")
            return None
    
    def create_particles(
        self, 
        distribution: ParticleDistribution
    ) -> Any:
        """
        Create XPart Particles object from distribution.
        
        Args:
            distribution: Particle distribution configuration
            
        Returns:
            XPart Particles object
        """
        try:
            # Create particles with reference parameters
            particles = xp.Particles(
                p0c=distribution.energy,  # Reference momentum in eV
                q0=1,  # Charge (proton = 1)
                mass0=xp.PROTON_MASS_EV,  # Proton mass
                x=np.zeros(distribution.num_particles),
                px=np.zeros(distribution.num_particles),
                y=np.zeros(distribution.num_particles),
                py=np.zeros(distribution.num_particles),
                zeta=np.zeros(distribution.num_particles),
                delta=np.zeros(distribution.num_particles)
            )
            
            # Apply distribution based on type
            dtype = distribution.distribution_type
            dtype_str = dtype.value if hasattr(dtype, 'value') else str(dtype).lower()
            
            if dtype_str == "gaussian":
                # Check if using explicit RMS sizes or emittance-based
                if distribution.x_std is not None and distribution.px_std is not None:
                    # Use explicit RMS sizes (new method)
                    sigma_x = distribution.x_std
                    sigma_px = distribution.px_std
                    sigma_y = distribution.y_std if distribution.y_std is not None else 0.0
                    sigma_py = distribution.py_std if distribution.py_std is not None else 0.0
                    
                    # Generate Gaussian distribution
                    particles.x = np.random.randn(distribution.num_particles) * sigma_x
                    if distribution.x_mean is not None:
                        particles.x += distribution.x_mean
                    particles.px = np.random.randn(distribution.num_particles) * sigma_px
                    if distribution.px_mean is not None:
                        particles.px += distribution.px_mean
                    particles.y = np.random.randn(distribution.num_particles) * sigma_y
                    if distribution.y_mean is not None:
                        particles.y += distribution.y_mean
                    particles.py = np.random.randn(distribution.num_particles) * sigma_py
                    if distribution.py_mean is not None:
                        particles.py += distribution.py_mean
                    
                elif distribution.emittance_x is not None and distribution.beta_x is not None:
                    # Calculate beam sizes from emittance and beta functions (old method)
                    # σ_x = sqrt(ε_x * β_x), σ_px = sqrt(ε_x / β_x)
                    sigma_x = np.sqrt(distribution.emittance_x * distribution.beta_x)
                    sigma_px = np.sqrt(distribution.emittance_x / distribution.beta_x)
                    sigma_y = np.sqrt(distribution.emittance_y * distribution.beta_y)
                    sigma_py = np.sqrt(distribution.emittance_y / distribution.beta_y)
                    
                    # Generate Gaussian distribution
                    particles.x = np.random.randn(distribution.num_particles) * sigma_x
                    particles.px = np.random.randn(distribution.num_particles) * sigma_px
                    particles.y = np.random.randn(distribution.num_particles) * sigma_y
                    particles.py = np.random.randn(distribution.num_particles) * sigma_py
                    
                    # Apply alpha function correlations (betatron coupling)
                    if distribution.alpha_x is not None:
                        particles.px -= distribution.alpha_x * particles.x / distribution.beta_x
                    if distribution.alpha_y is not None:
                        particles.py -= distribution.alpha_y * particles.y / distribution.beta_y
                    
                    # Apply mean offsets if specified
                    if distribution.x_mean is not None:
                        particles.x += distribution.x_mean
                    if distribution.px_mean is not None:
                        particles.px += distribution.px_mean
                    if distribution.y_mean is not None:
                        particles.y += distribution.y_mean
                    if distribution.py_mean is not None:
                        particles.py += distribution.py_mean
                else:
                    raise ValueError("GAUSSIAN distribution requires either (x_std, px_std, y_std, py_std) or (emittance_x, emittance_y, beta_x, beta_y)")
                
                # Longitudinal distribution
                if distribution.delta_std is not None and distribution.delta_std > 0:
                    particles.delta = np.random.randn(distribution.num_particles) * distribution.delta_std
                    if distribution.delta_mean is not None:
                        particles.delta += distribution.delta_mean
                
                # Apply dispersion effects (momentum-dependent position offset)
                # x = x_betatron + D_x * δ, px = px_betatron + D'_x * δ
                if distribution.dx is not None:
                    particles.x += distribution.dx * particles.delta
                if distribution.dpx is not None:
                    particles.px += distribution.dpx * particles.delta
                if distribution.dy is not None:
                    particles.y += distribution.dy * particles.delta
                if distribution.dpy is not None:
                    particles.py += distribution.dpy * particles.delta
                
                if distribution.zeta_std is not None and distribution.zeta_std > 0:
                    particles.zeta = np.random.randn(distribution.num_particles) * distribution.zeta_std
                    if distribution.zeta_mean is not None:
                        particles.zeta += distribution.zeta_mean
            
            logger.info(f"Created {distribution.num_particles} particles with p0c={distribution.energy:.3e} eV")
            logger.info(f"  σ_x={np.std(particles.x):.3e} m, σ_y={np.std(particles.y):.3e} m")
            self._particles = particles
            return particles
            
        except Exception as e:
            raise TrackingError(f"Failed to create particles: {e}")
    
    def create_matched_particles(
        self,
        distribution: ParticleDistribution,
        element_name: Optional[str] = None,
        nemitt_x: Optional[float] = None,
        nemitt_y: Optional[float] = None,
        use_lattice_twiss: bool = True
    ) -> Any:
        """
        Create particles matched to the lattice optics using XSuite's build_particles.
        
        This method uses XSuite's built-in particle generation that:
        - Uses the 1-turn transfer matrix (R-matrix) for proper coupling
        - Accounts for dispersion (D_x, D_y)
        - Centers around the closed orbit
        - Matches to actual lattice beta functions at the creation point
        
        Args:
            distribution: Particle distribution configuration
            element_name: Name of element where particles are created (default: start of line)
            nemitt_x: Normalized emittance in x (m·rad). If None, uses distribution.emittance_x
            nemitt_y: Normalized emittance in y (m·rad). If None, uses distribution.emittance_y
            use_lattice_twiss: If True, ignore distribution.beta_x/y and use lattice Twiss
            
        Returns:
            XPart Particles object matched to lattice
            
        Raises:
            TrackingError: If line is not initialized or matching fails
            
        Note:
            For RING mode, this requires a stable lattice with closed orbit.
            For LINAC mode, uses initial conditions instead of closed orbit.
        """
        if self._line is None:
            raise TrackingError("Line must be initialized before creating matched particles")
        
        try:
            # Convert geometric emittances to normalized emittances
            if nemitt_x is None:
                # ε_norm = ε_geom × β_rel × γ_rel
                gamma_rel = distribution.energy / (xp.PROTON_MASS_EV)
                beta_rel = np.sqrt(1 - 1/gamma_rel**2)
                nemitt_x = distribution.emittance_x * beta_rel * gamma_rel
                nemitt_y = distribution.emittance_y * beta_rel * gamma_rel
                logger.info(f"Converted geometric emittances to normalized:")
                logger.info(f"  ε_x: {distribution.emittance_x:.3e} m → {nemitt_x:.3e} m")
                logger.info(f"  ε_y: {distribution.emittance_y:.3e} m → {nemitt_y:.3e} m")
            
            # Determine creation point
            at_element = element_name if element_name else 0
            
            # Build matched particles using XSuite's built-in function
            # This automatically uses the lattice Twiss parameters
            particles = xp.generate_matched_gaussian_bunch(
                num_particles=distribution.num_particles,
                total_intensity_particles=distribution.num_particles,
                nemitt_x=nemitt_x,
                nemitt_y=nemitt_y,
                sigma_z=distribution.zeta_std if distribution.zeta_std and distribution.zeta_std > 0 else 1e-3,
                particle_ref=self._line.particle_ref,
                line=self._line,
                at_element=at_element
            )
            
            # Apply energy spread if specified
            if distribution.delta_std and distribution.delta_std > 0:
                particles.delta += np.random.randn(distribution.num_particles) * distribution.delta_std
            
            # Get Twiss at creation point for logging
            if use_lattice_twiss:
                twiss = self._line.twiss(at_elements=[at_element] if element_name else [0])
                beta_x_lattice = twiss.betx[0] if hasattr(twiss.betx, '__getitem__') else twiss.betx
                beta_y_lattice = twiss.bety[0] if hasattr(twiss.bety, '__getitem__') else twiss.bety
                logger.info(f"Matched particles to lattice Twiss at {element_name or 'start'}:")
                logger.info(f"  β_x = {beta_x_lattice:.3f} m, β_y = {beta_y_lattice:.3f} m")
            
            logger.info(f"Created {distribution.num_particles} matched particles")
            logger.info(f"  σ_x = {np.std(particles.x):.3e} m, σ_y = {np.std(particles.y):.3e} m")
            logger.info(f"  <x·px> = {np.mean(particles.x * particles.px):.3e} (coupling indicator)")
            
            self._particles = particles
            return particles
            
        except Exception as e:
            logger.error(f"Failed to create matched particles: {e}")
            logger.info("Falling back to unmatched particle creation")
            # Fall back to the standard create_particles method
            return self.create_particles(distribution)
    
    def track_single_turn(self) -> bool:
        """
        Execute one turn of tracking.
        
        Returns:
            True if successful
        """
        if self._line is None or self._particles is None:
            logger.error("Line or particles not initialized")
            return False
        
        try:
            self._line.track(self._particles, num_turns=1)
            return True
        except Exception as e:
            logger.error(f"Single turn tracking failed: {e}")
            return False
    
    def get_particle_coordinates(self) -> Dict[str, np.ndarray]:
        """
        Get current particle coordinates.
        
        Returns:
            Dictionary of coordinate arrays
        """
        if self._particles is None:
            return {}
        
        return {
            'x': np.copy(self._particles.x),
            'px': np.copy(self._particles.px),
            'y': np.copy(self._particles.y),
            'py': np.copy(self._particles.py),
            'zeta': np.copy(self._particles.zeta),
            'delta': np.copy(self._particles.delta),
            'state': np.copy(self._particles.state)
        }
    
    def get_twiss(
        self, 
        converted_lattice: Any, 
        energy: float,
        mode: SimulationMode,
        **kwargs
    ) -> TwissData:
        """
        Calculate Twiss parameters using XTrack.
        
        Args:
            converted_lattice: XTrack Line object
            energy: Beam energy in eV
            mode: Simulation mode
            **kwargs: Additional twiss options
            
        Returns:
            TwissData object with optical functions
        """
        try:
            line = converted_lattice
            
            # Set reference particle energy
            line.particle_ref = xp.Particles(p0c=energy)
            
            # Build tracker if not already built
            if not hasattr(line, 'tracker') or line.tracker is None:
                line.build_tracker(_context=self._context)
            
            # Compute Twiss based on mode
            if mode == SimulationMode.RING:
                # Periodic solution (uses all slice points automatically)
                twiss = line.twiss(method='4d')
            elif mode == SimulationMode.LINAC:
                # Open line - need initial conditions
                twiss = line.twiss(method='4d')
            else:  # RAMPING
                # Use periodic for now
                twiss = line.twiss(method='4d')
            
            # Convert to TwissData (schema-aligned)
            twiss_data = TwissData(
                s=twiss.s,
                beta_x=twiss.betx,
                beta_y=twiss.bety,
                alpha_x=twiss.alfx,
                alpha_y=twiss.alfy,
                dx=getattr(twiss, 'dx', None),
                dy=getattr(twiss, 'dy', None),
                dpx=getattr(twiss, 'dpx', None),
                dpy=getattr(twiss, 'dpy', None),
                mu_x=getattr(twiss, 'mux', None),
                mu_y=getattr(twiss, 'muy', None),
                simulation_mode=mode,
                reference_energy=float(energy),
                tune_x=float(twiss.qx),
                tune_y=float(twiss.qy)
            )
            
            logger.info(f"Twiss calculated: Qx={twiss.qx:.4f}, Qy={twiss.qy:.4f}")
            return twiss_data
            
        except Exception as e:
            logger.error(f"Twiss calculation failed: {e}")
            raise
    
    def update_element_parameter(
        self, 
        element_name: str, 
        parameter_group: str,
        parameter_name: str, 
        value: float
    ) -> bool:
        """
        Update element parameter in XTrack Line.
        
        Args:
            element_name: Name of element
            parameter_group: Parameter group name
            parameter_name: Parameter name within group
            value: New value
            
        Returns:
            True if successful
        """
        if self._line is None:
            return False
        
        try:
            # Get element from line
            element = self._line[element_name]
            
            # Map parameter to XSuite attribute
            # TODO: Use interface.yaml mapping
            xsuite_attr = parameter_name  # Simplified for now
            
            # Update parameter
            if hasattr(element, xsuite_attr):
                setattr(element, xsuite_attr, value)
                logger.debug(f"Updated {element_name}.{xsuite_attr} = {value}")
                return True
            else:
                logger.warning(f"Element {element_name} has no attribute {xsuite_attr}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update parameter: {e}")
            return False
    
    def get_element_parameter(
        self, 
        element_name: str,
        parameter_group: str, 
        parameter_name: str
    ) -> Optional[float]:
        """
        Get element parameter from XTrack Line.
        
        Args:
            element_name: Name of element
            parameter_group: Parameter group name
            parameter_name: Parameter name within group
            
        Returns:
            Parameter value or None
        """
        if self._line is None:
            return None
        
        try:
            element = self._line[element_name]
            xsuite_attr = parameter_name  # Simplified for now
            
            if hasattr(element, xsuite_attr):
                return getattr(element, xsuite_attr)
            else:
                return None
                
        except Exception as e:
            logger.error(f"Failed to get parameter: {e}")
            return None
    
    # === Abstract Properties Implementation ===
    
    @property
    def available(self) -> bool:
        """Check if XSuite is available."""
        return XSUITE_AVAILABLE
    
    @property
    def supported_modes(self) -> List[SimulationMode]:
        """List of supported simulation modes."""
        return [SimulationMode.RING, SimulationMode.LINAC, SimulationMode.RAMPING]
    
    # === Abstract Methods Implementation ===
    
    def track(self, particles: Any, parameters: TrackingParameters) -> TrackingResults:
        """
        Perform particle tracking simulation.
        
        Args:
            particles: XPart particles object
            parameters: Tracking configuration
            
        Returns:
            Tracking results
        """
        # Store particles for later use
        self._particles = particles
        
        # Simple tracking implementation (will be extended with process isolation later)
        try:
            if parameters.num_turns > 0:
                self._line.track(
                    particles,
                    num_turns=parameters.num_turns,
                    turn_by_turn_monitor=True if parameters.num_turns > 1 else False
                )
            
            # Build results
            results = TrackingResults(
                success=True,
                num_turns=parameters.num_turns,
                num_particles=len(particles.x),
                survived_particles=np.sum(particles.state > 0),
                final_coordinates={
                    'x': particles.x.copy() if hasattr(particles.x, 'copy') else np.array(particles.x),
                    'px': particles.px.copy() if hasattr(particles.px, 'copy') else np.array(particles.px),
                    'y': particles.y.copy() if hasattr(particles.y, 'copy') else np.array(particles.y),
                    'py': particles.py.copy() if hasattr(particles.py, 'copy') else np.array(particles.py),
                    'zeta': particles.zeta.copy() if hasattr(particles.zeta, 'copy') else np.array(particles.zeta),
                    'delta': particles.delta.copy() if hasattr(particles.delta, 'copy') else np.array(particles.delta),
                    'state': particles.state.copy() if hasattr(particles.state, 'copy') else np.array(particles.state)
                }
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Tracking failed: {e}")
            raise TrackingError(f"XSuite tracking failed: {e}")
    
    def get_closed_orbit(self, converted_lattice: Any, energy: float) -> Dict[str, float]:
        """
        Find closed orbit for ring lattices.
        
        Args:
            converted_lattice: XTrack Line object
            energy: Reference energy in eV
            
        Returns:
            Closed orbit parameters
        """
        try:
            # Use XTrack's Twiss to get closed orbit
            twiss = converted_lattice.twiss(method='4d')
            
            return {
                'x': float(twiss['x'][0]),
                'px': float(twiss['px'][0]),
                'y': float(twiss['y'][0]),
                'py': float(twiss['py'][0]),
                'delta': 0.0,  # On-momentum
                's': 0.0  # At start
            }
        except Exception as e:
            logger.error(f"Failed to compute closed orbit: {e}")
            return {'x': 0.0, 'px': 0.0, 'y': 0.0, 'py': 0.0, 'delta': 0.0, 's': 0.0}
    
    def get_bpm_readings(self, monitor_elements: List[Element]) -> Dict[str, BeamPositionData]:
        """
        Get BPM readings from tracked particles.
        
        Args:
            monitor_elements: List of monitor elements
            
        Returns:
            Dictionary of BPM readings
        """
        if self._particles is None:
            raise TrackingError("No particles loaded - run tracking first")
        
        readings = {}
        
        for monitor in monitor_elements:
            # Find monitor position in lattice
            try:
                s_position = self._line.get_s_position(monitor.name)
            except:
                logger.warning(f"Monitor {monitor.name} not found in lattice")
                continue
            
            # Calculate mean position from particles
            # Note: BPMs only measure beam position (centroid), not beam size
            # In a real implementation, we'd track particles to this position
            # For now, use current particle distribution
            x_mean = float(np.mean(self._particles.x))
            y_mean = float(np.mean(self._particles.y))
            
            readings[monitor.name] = BeamPositionData(
                name=monitor.name,
                s_position=float(s_position),
                x_mean=x_mean,
                y_mean=y_mean,
                intensity=float(np.sum(self._particles.state > 0)),
                timestamp=None
            )
        
        return readings
    
    def configure_bpm_tracking(self, num_turns: int, frev: float = 1e6, sampling_frequency: Optional[float] = None):
        """
        Configure BPM monitors for turn-by-turn tracking.
        
        This method must be called before tracking to set up the stop_at_turn and frequency parameters.
        
        Args:
            num_turns: Number of turns to track
            frev: Revolution frequency in Hz (default: 1 MHz)
            sampling_frequency: Sampling frequency in Hz (default: same as frev)
        """
        if self._line is None:
            logger.warning("No line available - call convert_lattice() first")
            return
        
        if sampling_frequency is None:
            sampling_frequency = frev
        
        # Find BPM monitors in the line (not from stored dict, which may have stale references)
        # After slicing, the line is rebuilt so we need to get fresh references
        bpm_count = 0
        for name, elem in zip(self._line.element_names, self._line.elements):
            if isinstance(elem, xt.BeamPositionMonitor):
                elem.stop_at_turn = num_turns
                elem.frev = frev
                elem.sampling_frequency = sampling_frequency
                bpm_count += 1
                logger.debug(f"Configured {name}: num_turns={num_turns}, frev={frev:.2e} Hz, sampling={sampling_frequency:.2e} Hz")
        
        logger.info(f"Configured {bpm_count} BPM monitors for {num_turns} turns")
    
    def get_bpm_data(self) -> Dict[str, Any]:
        """
        Get BPM data from all monitors after tracking.
        
        Returns:
            Dictionary mapping BPM names to their monitor objects (with x_mean, y_mean arrays)
        """
        if self._line is None:
            logger.warning("No line available")
            return {}
        
        # Get BPM data from the line (not from stored dict)
        # After slicing and tracking, the actual data is in the line's BPM elements
        bpm_data = {}
        for name, elem in zip(self._line.element_names, self._line.elements):
            if isinstance(elem, xt.BeamPositionMonitor):
                if hasattr(elem, 'x_mean') and len(elem.x_mean) > 0:
                    bpm_data[name] = elem
                    logger.debug(f"{name}: {len(elem.x_mean)} samples")
                else:
                    logger.warning(f"{name}: No data recorded")
        
        return bpm_data
    
    def apply_lattice_change(
        self,
        converted_lattice: Any,
        change: LatticeChangeAction,
        current_value: Optional[float] = None
    ) -> bool:
        """
        Apply lattice parameter change.
        
        Args:
            converted_lattice: XTrack Line
            change: Change specification
            current_value: Current value (for ramping)
            
        Returns:
            True if successful
        """
        try:
            element = converted_lattice[change.element_name]
            
            # Map parameter group.parameter to XSuite attribute
            # Use interface.yaml mapping (simplified for now)
            param_parts = change.parameter_name.split('.')
            if len(param_parts) == 2:
                group_name, param_name = param_parts
                # Simplified mapping - in production use interface.yaml
                if hasattr(element, param_name):
                    setattr(element, param_name, change.new_value)
                    return True
            else:
                if hasattr(element, change.parameter_name):
                    setattr(element, change.parameter_name, change.new_value)
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to apply lattice change: {e}")
            return False
    
    @property
    def available(self) -> bool:
        """Check if XSuite is available."""
        return XSUITE_AVAILABLE
    
    @property
    def supported_modes(self) -> List[SimulationMode]:
        """Return list of supported simulation modes."""
        return [SimulationMode.RING, SimulationMode.LINAC, SimulationMode.RAMPING]
    
    def get_bpm_readings(
        self,
        session_id: str,
        bpm_names: Optional[List[str]] = None,
        turn_range: Optional[Tuple[int, int]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Get BPM readings from tracking session.
        
        Args:
            session_id: Tracking session identifier
            bpm_names: Optional list of BPM names to retrieve
            turn_range: Optional tuple (start_turn, end_turn)
            
        Returns:
            Dictionary mapping BPM names to reading arrays
        """
        # For now, return empty dict - will be implemented with turn-by-turn monitor
        logger.warning("get_bpm_readings not yet fully implemented for XSuite")
        return {}
    
    def _tracking_process_main(
        self,
        session_id: str,
        particles: Any,
        parameters: TrackingParameters,
        command_queue: mp.Queue,
        response_queue: mp.Queue,
        data_file: Path
    ):
        """
        Main tracking process function (for future process isolation).
        
        This is a placeholder for future implementation of process-isolated tracking.
        Currently, tracking runs in the main process via the track() method.
        """
        # Future implementation will handle:
        # - RING mode with turn-by-turn tracking
        # - LINAC mode with single-pass tracking  
        # - RAMPING mode with time-dependent parameters
        # - Command queue processing for real-time parameter changes
        # - Response queue for status updates
        # - HDF5 data file I/O
        pass
