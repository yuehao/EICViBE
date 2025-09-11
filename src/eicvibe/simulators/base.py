"""
Base simulation engine abstract class for EICViBE.

This module defines the abstract base class that all simulation engines
must implement to integrate with the EICViBE framework. It provides a
standardized interface for lattice conversion, particle tracking, and
beam analysis.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Callable
import logging
import time
import numpy as np

from ..machine_portal.lattice import Lattice
from ..machine_portal.element import Element
from .types import (
    SimulationMode, 
    TrackingParameters, 
    ParticleDistribution, 
    TrackingResults,
    BeamPositionData,
    TwissData,
    EngineConfiguration,
    SimulationError,
    TrackingError
)

logger = logging.getLogger(__name__)


class SimulationMonitor(ABC):
    """
    Abstract base class for simulation monitoring and callbacks.
    
    Monitors can be attached to simulation engines to receive real-time
    updates about simulation progress and results.
    """
    
    @abstractmethod
    def on_simulation_start(self, parameters: TrackingParameters):
        """Called when simulation starts."""
        pass
    
    @abstractmethod
    def on_turn_complete(self, turn: int, data: Dict[str, Any]):
        """Called after each tracking turn/step."""
        pass
    
    @abstractmethod
    def on_simulation_complete(self, results: TrackingResults):
        """Called when simulation finishes."""
        pass
    
    @abstractmethod
    def on_error(self, error: Exception):
        """Called when an error occurs during simulation."""
        pass


class ProgressMonitor(SimulationMonitor):
    """Simple progress monitoring implementation."""
    
    def __init__(self, show_progress: bool = True):
        self.show_progress = show_progress
        self.start_time = None
        self.total_turns = 0
        
    def on_simulation_start(self, parameters: TrackingParameters):
        self.start_time = time.time()
        self.total_turns = parameters.num_turns if parameters.mode == SimulationMode.RING else parameters.num_steps
        if self.show_progress:
            logger.info(f"Starting {parameters.mode} simulation with {parameters.num_particles} particles")
            
    def on_turn_complete(self, turn: int, data: Dict[str, Any]):
        if self.show_progress and turn % 100 == 0:
            elapsed = time.time() - self.start_time if self.start_time else 0
            progress = (turn / self.total_turns) * 100
            logger.info(f"Progress: {progress:.1f}% (turn {turn}/{self.total_turns}, {elapsed:.1f}s elapsed)")
            
    def on_simulation_complete(self, results: TrackingResults):
        if self.show_progress:
            logger.info(f"Simulation completed in {results.execution_time:.2f}s")
            logger.info(f"Transmission: {results.transmission:.3f} ({results.particles_final}/{results.particles_initial})")
            
    def on_error(self, error: Exception):
        logger.error(f"Simulation error: {error}")


class BaseSimulationEngine(ABC):
    """
    Abstract base class for all simulation engines.
    
    This class defines the interface that all simulation engines must implement
    to integrate with the EICViBE framework. It provides standardized methods
    for lattice conversion, particle tracking, and beam analysis.
    
    Key features:
    - Abstract methods for core simulation functionality
    - Standardized parameter handling with Pydantic validation
    - Monitor/callback system for real-time updates
    - Error handling and logging
    - Performance monitoring
    """
    
    def __init__(self, config: Optional[EngineConfiguration] = None):
        self.config = config or EngineConfiguration(name="base")
        self.name = self.config.name
        
        # Initialize internal state
        self._particles = None
        self._converted_lattice = None
        self._lattice_hash = None
        self.monitors: List[SimulationMonitor] = []
        
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
        self.logger.setLevel(getattr(logging, self.config.log_level))
        
        # Add default progress monitor
        if not hasattr(self, '_no_default_monitor'):
            self.add_monitor(ProgressMonitor())
    
    # === Abstract Properties ===
    
    @property
    @abstractmethod
    def available(self) -> bool:
        """
        Check if the simulation engine is available.
        
        Returns:
            True if all dependencies are installed and the engine can be used
        """
        pass
        
    @property
    @abstractmethod
    def supported_modes(self) -> List[SimulationMode]:
        """
        List of simulation modes supported by this engine.
        
        Returns:
            List of supported SimulationMode values
        """
        pass
    
    # === Abstract Methods ===
    
    @abstractmethod
    def convert_lattice(self, lattice: Lattice, mode: SimulationMode) -> Any:
        """
        Convert EICViBE lattice to engine-specific format.
        
        Args:
            lattice: EICViBE Lattice object to convert
            mode: Simulation mode for optimization
            
        Returns:
            Engine-specific lattice representation
            
        Raises:
            LatticeConversionError: If conversion fails
        """
        pass
        
    @abstractmethod
    def create_particles(self, distribution: ParticleDistribution) -> Any:
        """
        Create particle distribution for tracking.
        
        Args:
            distribution: Particle distribution configuration
            
        Returns:
            Engine-specific particle representation
        """
        pass
        
    @abstractmethod
    def track(self, particles: Any, parameters: TrackingParameters) -> TrackingResults:
        """
        Perform particle tracking simulation.
        
        Args:
            particles: Particle distribution from create_particles()
            parameters: Tracking configuration parameters
            
        Returns:
            Complete tracking results including statistics and monitor data
            
        Raises:
            TrackingError: If tracking simulation fails
        """
        pass
        
    @abstractmethod
    def get_twiss(self, converted_lattice: Any, energy: float, mode: SimulationMode, **kwargs) -> TwissData:
        """
        Calculate Twiss parameters for the lattice in the specified simulation mode.
        
        This method computes optical functions (beta, alpha, dispersion, etc.) 
        appropriate for the simulation mode:
        
        - LINAC mode: Linear optics through transport line with initial conditions
        - RING mode: Periodic optics with closed orbit and tunes
        - RAMPING mode: Time-dependent optics during parameter evolution
        
        Args:
            converted_lattice: Engine-specific lattice from convert_lattice()
            energy: Reference energy in eV
            mode: Simulation mode for Twiss calculation
            **kwargs: Mode-specific parameters:
                - initial_conditions: For LINAC mode (beta_x, alpha_x, etc.)
                - time_points: For RAMPING mode time evolution
                - delta: Energy deviation for off-momentum calculations
                
        Returns:
            TwissData object with comprehensive optical functions
            
        Raises:
            ValueError: If mode is not supported or parameters are invalid
            SimulationError: If Twiss calculation fails
        """
        pass
        
    @abstractmethod
    def get_closed_orbit(self, converted_lattice: Any, energy: float) -> Dict[str, float]:
        """
        Find closed orbit for ring lattices.
        
        Args:
            converted_lattice: Engine-specific lattice from convert_lattice() (must be ring topology)
            energy: Reference energy in eV
            
        Returns:
            Dictionary with closed orbit parameters (x, px, y, py, delta, s)
        """
        pass
    
    @abstractmethod
    def get_bpm_readings(self, particles: Any, monitor_elements: List[Element]) -> Dict[str, BeamPositionData]:
        """
        Simulate beam position monitor readings.
        
        Args:
            particles: Current particle distribution
            monitor_elements: List of BPM/Monitor elements
            
        Returns:
            Dictionary mapping monitor names to BeamPositionData
        """
        pass
    
    # === Concrete Methods ===
    
    def get_converted_lattice(self, lattice: Lattice, mode: SimulationMode) -> Any:
        """
        Get converted lattice with caching to avoid redundant conversions.
        
        Args:
            lattice: EICViBE lattice to convert
            mode: Simulation mode for conversion
            
        Returns:
            Engine-specific lattice representation (cached if already converted)
        """
        # Create a simple hash of the lattice for caching
        lattice_hash = hash((id(lattice), mode.value if hasattr(mode, 'value') else str(mode)))
        
        # Return cached version if available
        if self._converted_lattice is not None and self._lattice_hash == lattice_hash:
            self.logger.debug("Using cached converted lattice")
            return self._converted_lattice
        
        # Convert and cache the lattice
        self.logger.debug(f"Converting lattice for {mode} mode")
        self._converted_lattice = self.convert_lattice(lattice, mode)
        self._lattice_hash = lattice_hash
        
        return self._converted_lattice
    
    def clear_converted_lattice(self):
        """Clear the cached converted lattice."""
        self._converted_lattice = None
        self._lattice_hash = None
        self.logger.debug("Cleared cached converted lattice")
    
    def add_monitor(self, monitor: SimulationMonitor):
        """Add a monitoring callback."""
        self.monitors.append(monitor)
        self.logger.debug(f"Added monitor: {type(monitor).__name__}")
    
    def remove_monitor(self, monitor: SimulationMonitor):
        """Remove a monitoring callback."""
        if monitor in self.monitors:
            self.monitors.remove(monitor)
            self.logger.debug(f"Removed monitor: {type(monitor).__name__}")
    
    def _notify_monitors(self, event: str, *args, **kwargs):
        """Notify all registered monitors of an event."""
        for monitor in self.monitors:
            try:
                if hasattr(monitor, f'on_{event}'):
                    getattr(monitor, f'on_{event}')(*args, **kwargs)
            except Exception as e:
                self.logger.warning(f"Monitor {type(monitor).__name__} error in on_{event}: {e}")
    
    def get_engine_info(self) -> Dict[str, Any]:
        """
        Get comprehensive engine information.
        
        Returns:
            Dictionary containing engine metadata
        """
        return {
            'name': self.name,
            'version': getattr(self, 'version', 'unknown'),
            'available': self.available,
            'supported_modes': [mode.value for mode in self.supported_modes],
            'config': self.config.model_dump() if self.config else None,
        }

    def validate_lattice(self, lattice: Lattice, mode: SimulationMode) -> bool:
        """
        Validate that lattice is compatible with simulation mode.
        
        Args:
            lattice: Lattice to validate
            mode: Intended simulation mode
            
        Returns:
            True if lattice is valid for the mode
            
        Raises:
            ValueError: If lattice is incompatible with mode
        """
        if mode == SimulationMode.RING:
            # Ring mode requires at least one ring branch
            has_ring_branch = False
            for branch_name in lattice.branches.keys():
                if lattice.get_branch_type(branch_name) == "ring":
                    has_ring_branch = True
                    break
            
            if not has_ring_branch:
                raise ValueError("RING mode requires at least one branch with 'ring' topology")
        
        # Check for required elements based on mode
        if not lattice.elements:
            raise ValueError("Lattice contains no elements")
            
        self.logger.debug(f"Lattice validation passed for {mode} mode ({len(lattice.elements)} elements)")
        return True
    
    def validate_parameters(self, parameters: TrackingParameters):
        """
        Validate tracking parameters for this engine.
        
        Args:
            parameters: Parameters to validate
            
        Raises:
            ValueError: If parameters are invalid
        """
        # Check if mode is supported
        if parameters.mode not in self.supported_modes:
            raise ValueError(f"Mode {parameters.mode} not supported by {self.name} engine")
        
        # Validate mode-specific parameters
        parameters.validate_mode_parameters()
        
        self.logger.debug(f"Parameters validated for {parameters.mode} mode")
    
    def calculate_optics(
        self, 
        lattice: Lattice, 
        energy: float, 
        mode: SimulationMode,
        **kwargs
    ) -> TwissData:
        """
        High-level method to calculate optical functions for any simulation mode.
        
        This method provides a unified interface for Twiss parameter calculation
        across all simulation modes with appropriate validation and error handling.
        
        Args:
            lattice: EICViBE lattice definition
            energy: Reference energy in eV
            mode: Simulation mode for calculation
            **kwargs: Mode-specific parameters
                
        Returns:
            Complete TwissData with optical functions
            
        Raises:
            ValueError: If parameters are invalid for the mode
            SimulationError: If calculation fails
        """
        # Validate inputs
        self.validate_lattice(lattice, mode)
        
        if energy <= 0:
            raise ValueError("Energy must be positive")
        
        if mode not in self.supported_modes:
            raise ValueError(f"Mode {mode} not supported by {self.name} engine")
        
        # Mode-specific validation and defaults
        if mode == SimulationMode.LINAC:
            # LINAC mode requires initial conditions
            if 'initial_conditions' not in kwargs:
                # Provide reasonable defaults
                kwargs['initial_conditions'] = {
                    'beta_x': 10.0, 'alpha_x': 0.0,
                    'beta_y': 10.0, 'alpha_y': 0.0,
                    'dx': 0.0, 'dpx': 0.0
                }
                self.logger.warning("Using default initial conditions for LINAC mode")
                
        elif mode == SimulationMode.RING:
            # Ring mode - closed orbit calculation included
            pass
            
        elif mode == SimulationMode.RAMPING:
            # Ramping mode requires time points
            if 'time_points' not in kwargs:
                # Default to single time point
                kwargs['time_points'] = np.array([0.0])
                self.logger.warning("Using single time point for RAMPING mode")
        
        self.logger.info(f"Calculating {mode} optics at E={energy/1e9:.3f} GeV")
        
        try:
            # Get converted lattice (with caching)
            converted_lattice = self.get_converted_lattice(lattice, mode)
            
            # Calculate Twiss parameters using converted lattice
            twiss_data = self.get_twiss(converted_lattice, energy, mode, **kwargs)
            
            # Validate results
            if not isinstance(twiss_data, TwissData):
                raise SimulationError("get_twiss must return TwissData object")
            
            if len(twiss_data.s) == 0:
                raise SimulationError("Empty Twiss calculation result")
            
            self.logger.info(f"Optics calculation successful: {len(twiss_data.s)} points")
            return twiss_data
            
        except Exception as e:
            self.logger.error(f"Optics calculation failed: {e}")
            raise SimulationError(f"Twiss calculation failed: {e}") from e
        """
        Get information about this engine.
        
        Returns:
            Dictionary with engine metadata and capabilities
        """
        return {
            'name': self.name,
            'available': self.available,
            'supported_modes': [mode.value for mode in self.supported_modes],
            'config': self.config.model_dump(),
            'version': getattr(self, 'version', 'unknown')
        }
    
    def run_simulation(
        self, 
        lattice: Lattice, 
        distribution: ParticleDistribution, 
        parameters: TrackingParameters
    ) -> TrackingResults:
        """
        High-level method to run a complete simulation.
        
        This method orchestrates the full simulation workflow:
        1. Validate inputs
        2. Convert lattice
        3. Create particles
        4. Run tracking
        5. Return results
        
        Args:
            lattice: EICViBE lattice definition
            distribution: Initial particle distribution
            parameters: Tracking parameters
            
        Returns:
            Complete simulation results
        """
        start_time = time.time()
        
        try:
            # Validate inputs
            self.validate_lattice(lattice, parameters.mode)
            self.validate_parameters(parameters)
            
            self._notify_monitors('simulation_start', parameters)
            
            # Get converted lattice (with caching)
            self.logger.info(f"Converting lattice for {parameters.mode} simulation")
            converted_lattice = self.get_converted_lattice(lattice, parameters.mode)
            
            # Create particle distribution
            self.logger.info(f"Creating {distribution.num_particles} particles")
            particles = self.create_particles(distribution)
            
            # Run tracking
            self.logger.info("Starting particle tracking")
            results = self.track(particles, parameters)
            
            # Update execution time
            results.execution_time = time.time() - start_time
            
            self._notify_monitors('simulation_complete', results)
            
            return results
            
        except Exception as e:
            self._notify_monitors('error', e)
            raise TrackingError(f"Simulation failed: {e}") from e

    def __str__(self) -> str:
        """String representation of the engine."""
        status = "available" if self.available else "unavailable"
        modes = ", ".join([mode.value for mode in self.supported_modes])
        return f"{self.name} engine ({status}, modes: {modes})"
    
    def __repr__(self) -> str:
        """Detailed representation of the engine."""
        return f"{self.__class__.__name__}(name='{self.name}', available={self.available})"
