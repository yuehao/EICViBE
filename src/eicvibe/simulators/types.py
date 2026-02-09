"""
Type definitions and enums for EICViBE simulation engines.

This module provides common type definitions, enums, and data structures
used across all simulation engines in the EICViBE framework.
"""

from enum import Enum
from typing import Dict, Any, Optional, List, Callable, Union
from collections import deque
from pydantic import BaseModel, Field, field_validator
import numpy as np
import math
from ..models.base import PhysicsBaseModel

# Import package-level standards
from ..optics import SimulationMode, TwissData as _TwissData

# Re-export for backward compatibility
TwissData = _TwissData


class DistributionType(str, Enum):
    """Types of initial particle distributions."""
    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"
    MATCHED = "matched"
    CUSTOM = "custom"


class TrackingStatus(str, Enum):
    """Status of tracking simulations."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"
    STOPPED = "stopped"


class LatticeChangeAction(PhysicsBaseModel):
    """
    Represents a scheduled lattice parameter change during simulation.
    
    This model defines when and how to modify lattice parameters
    during a running simulation, particularly for RING mode.
    """
    element_name: str = Field(description="Name of element to modify")
    parameter_group: str = Field(description="Parameter group name (e.g., 'MagneticMultipoleP')")
    parameter_name: str = Field(description="Parameter name within the group")
    old_value: Optional[float] = Field(default=None, description="Old parameter value (auto-detected if None)")
    new_value: float = Field(description="New parameter value")
    activation_turn: int = Field(ge=0, description="Turn number when change begins")
    ramp_turns: Optional[int] = Field(default=None, ge=1, description="Number of turns to ramp to new value")
    ramp_function: str = Field(default="linear", description="Ramping function: 'linear', 'exponential', 'cosine'")
    change_rate: Optional[float] = Field(default=None, gt=0, description="Rate of change per turn (alternative to ramp_turns)")
    description: Optional[str] = Field(default=None, description="Optional description of the change")
    
    @field_validator('ramp_function')
    @classmethod
    def validate_ramp_function(cls, v):
        valid_functions = ['linear', 'exponential', 'cosine', 'sigmoid']
        if v not in valid_functions:
            raise ValueError(f"ramp_function must be one of {valid_functions}")
        return v
    
    def calculate_parameter_value(self, current_turn: int) -> Optional[float]:
        """
        Calculate the parameter value at a given turn during ramping.
        
        Args:
            current_turn: Current turn number
            
        Returns:
            Parameter value at this turn, or None if change hasn't started
        """
        if current_turn < self.activation_turn:
            return None  # Change hasn't started yet
        
        if self.ramp_turns is None or self.ramp_turns <= 1:
            # Instantaneous change
            return self.new_value
        
        # Calculate ramping progress
        turns_since_activation = current_turn - self.activation_turn
        
        if turns_since_activation >= self.ramp_turns:
            # Ramping is complete
            return self.new_value
        
        # Calculate ramping fraction (0.0 to 1.0)
        fraction = turns_since_activation / self.ramp_turns
        
        # Apply ramping function
        if self.ramp_function == "linear":
            ramp_factor = fraction
        elif self.ramp_function == "exponential":
            ramp_factor = 1.0 - math.exp(-3.0 * fraction)  # 95% complete at fraction=1.0
        elif self.ramp_function == "cosine":
            ramp_factor = 0.5 * (1.0 - math.cos(math.pi * fraction))  # Smooth S-curve
        elif self.ramp_function == "sigmoid":
            # Sigmoid function for smooth transitions
            x = 12.0 * fraction - 6.0  # Map [0,1] to [-6,6]
            ramp_factor = 1.0 / (1.0 + math.exp(-x))
        else:
            ramp_factor = fraction  # Fallback to linear
        
        # Interpolate between old and new values
        if self.old_value is not None:
            return self.old_value + (self.new_value - self.old_value) * ramp_factor
        else:
            # If old_value is unknown, assume starting from 0
            return self.new_value * ramp_factor


class MonitorDataRequest(PhysicsBaseModel):
    """Configuration for requesting monitor data from running simulations."""
    start_turn: int = Field(ge=0, description="Starting turn number for data collection")
    end_turn: Optional[int] = Field(default=None, description="Ending turn number (None for current)")
    monitor_names: Optional[List[str]] = Field(default=None, description="Specific monitors (None for all)")
    data_types: List[str] = Field(
        default_factory=lambda: ["position", "size"],
        description="Types of data to collect"
    )
    turn_step: int = Field(default=1, ge=1, description="Turn step size for data collection")


class TrackingParameters(PhysicsBaseModel):
    """
    Base configuration parameters for particle tracking simulations.
    
    This is a BASE CLASS that engines can extend with their own specific parameters.
    Common parameters are defined here, and each engine can subclass this to add
    engine-specific options.
    
    Examples:
        XSuite can add GPU settings, context configuration, backtracker options
        MAD-X can add observation points, matching constraints
        Custom engines can add their own specific parameters
    
    The model allows extra fields via model_config for maximum flexibility.
    """
    # Allow engines to add extra fields
    model_config = {"extra": "allow"}
    
    # ========== Core Parameters (Common to All Engines) ==========
    mode: SimulationMode = Field(description="Simulation mode")
    num_particles: int = Field(gt=0, description="Number of particles to track")
    num_turns: int = Field(default=100000000, ge=1, description="Number of turns (RING mode)")
    num_steps: int = Field(default=1000, ge=1, description="Number of tracking steps (LINAC/RAMPING)")
    monitor_frequency: int = Field(default=1, ge=1, description="Monitor every N turns/steps")
    save_particles: bool = Field(default=False, description="Save particle coordinates")
    
    # ========== Collective Effects (Support Varies by Engine) ==========
    enable_beam_beam: bool = Field(default=False, description="Enable beam-beam interactions (if supported)")
    enable_wakefields: bool = Field(default=False, description="Enable wakefield effects (if supported)")
    enable_space_charge: bool = Field(default=False, description="Enable space charge effects (if supported)")
    enable_synchrotron_radiation: bool = Field(default=False, description="Enable synchrotron radiation (if supported)")
    
    # ========== Data Collection ==========
    use_bpm_only: bool = Field(default=True, description="Retrieve BPM data only for tracking")
    
    def validate_mode_parameters(self):
        """Validate parameters are appropriate for the simulation mode."""
        if self.mode == SimulationMode.RING and self.num_turns < 1:
            raise ValueError("RING mode requires num_turns >= 1")
        if self.mode in {SimulationMode.LINAC, SimulationMode.RAMPING} and self.num_steps < 1:
            raise ValueError(f"{self.mode.value.upper()} mode requires num_steps >= 1")
    
    @classmethod
    def get_supported_features(cls) -> Dict[str, bool]:
        """
        Get dictionary of supported features for this parameter class.
        
        Override in engine-specific subclasses to indicate which features
        the engine supports.
        
        Returns:
            Dictionary mapping feature names to support status
        """
        return {
            "beam_beam": False,
            "wakefields": False,
            "space_charge": False,
            "synchrotron_radiation": False,
        }


# ========================================================================
# Engine-Specific Tracking Parameters (Examples)
# ========================================================================
# Engines should define their own TrackingParameters subclasses in their
# implementation modules. These are examples showing the pattern.

class XSuiteTrackingParameters(TrackingParameters):
    """
    XSuite-specific tracking parameters.
    
    Extends base TrackingParameters with XSuite-specific options like
    GPU acceleration, context configuration, and backtracker settings.
    
    Aligns with XSuite Line.track() and Line.build_tracker() API parameters.
    See: https://xsuite.readthedocs.io/en/latest/apireference.html
    
    Note: This is an example. Actual implementation should be in
    xsuite_interface.py module.
    """
    # ========== GPU and Performance ==========
    use_gpu: bool = Field(default=False, description="Use GPU acceleration via cupy")
    context: Optional[str] = Field(default=None, description="XSuite context: 'cpu', 'cuda', 'opencl'")
    num_threads: Optional[int] = Field(default=None, description="Number of CPU threads")
    compile_tracker: bool = Field(default=True, description="Compile tracker on build (line.build_tracker)")
    use_prebuilt_kernels: bool = Field(default=True, description="Use prebuilt XSuite kernels")
    
    # ========== Tracking Control (from line.track API) ==========
    enable_backtracker: bool = Field(default=False, description="Enable particle backtracking (backtrack parameter)")
    enable_aperture_check: bool = Field(default=True, description="Check aperture losses")
    freeze_longitudinal: bool = Field(default=False, description="Freeze longitudinal coordinates during tracking")
    with_progress: bool = Field(default=False, description="Display progress bar during tracking")
    turn_by_turn_monitor: Optional[str] = Field(
        default=None, 
        description="Turn-by-turn monitoring: None, 'ONE_TURN_EBE', or monitor name"
    )
    
    # ========== Time-dependent Parameters ==========
    time_step: Optional[float] = Field(default=None, description="Time step for RAMPING mode in seconds")
    
    # ========== Collective Effects Configuration ==========
    # Note: enable_* flags are in base class, these are XSuite-specific configs
    radiation_model: Optional[str] = Field(
        default=None, 
        description="Radiation model: 'mean' or 'quantum' (if enable_synchrotron_radiation=True)"
    )
    beamstrahlung_model: Optional[str] = Field(
        default=None,
        description="Beamstrahlung model for beam-beam: 'mean' or 'quantum'"
    )
    
    @classmethod
    def get_supported_features(cls) -> Dict[str, bool]:
        """
        XSuite supports most collective effects and advanced features.
        
        Based on XSuite API documentation:
        - Beam-beam: BeamBeamBiGaussian2D/3D elements
        - Wakefields: Collective elements support
        - Space charge: SpaceChargeBiGaussian, SpaceCharge3D
        - Radiation: radiation_flag in magnets, configure_radiation()
        - GPU: Cupy context for CUDA, PyOpenCL for OpenCL
        - Backtracker: backtrack parameter in line.track()
        - Progress bar: with_progress parameter
        - Frozen longitudinal: freeze_longitudinal parameter
        """
        return {
            "beam_beam": True,
            "wakefields": True,
            "space_charge": True,
            "synchrotron_radiation": True,
            "gpu_acceleration": True,
            "backtracker": True,
            "progress_monitoring": True,
            "freeze_longitudinal": True,
            "turn_by_turn_monitor": True,
        }


class MADXTrackingParameters(TrackingParameters):
    """
    MAD-X-specific tracking parameters.
    
    Extends base TrackingParameters with MAD-X-specific options like
    observation points and matching constraints.
    
    Note: This is an example. Actual implementation should be in
    madx_interface.py module.
    """
    # MAD-X specific
    observation_points: Optional[List[str]] = Field(
        default=None, 
        description="Element names for detailed observation"
    )
    dump_frequency: int = Field(default=100, ge=1, description="Frequency to dump particle data")
    
    # Matching and correction
    enable_correction: bool = Field(default=False, description="Enable orbit correction")
    correction_elements: Optional[List[str]] = Field(
        default=None,
        description="Elements used for correction"
    )
    
    @classmethod
    def get_supported_features(cls) -> Dict[str, bool]:
        """MAD-X has limited collective effects support."""
        return {
            "beam_beam": True,
            "wakefields": False,
            "space_charge": False,
            "synchrotron_radiation": True,
            "matching": True,
            "correction": True,
        }


class ParticleDistribution(PhysicsBaseModel):
    """
    Configuration for initial particle distribution.
    
    Defines the initial beam parameters and distribution type for
    particle tracking simulations. Different distribution types use
    different sets of parameters:
    
    - GAUSSIAN: Use x_mean, x_std, px_mean, px_std, etc. for explicit RMS sizes
    - MATCHED: Use emittance_x/y, beta_x/y, alpha_x/y for matched distributions
    - UNIFORM: Use x_min, x_max, etc. for uniform ranges
    - CUSTOM: Provide custom particle coordinates
    """
    distribution_type: DistributionType = Field(description="Type of distribution")
    num_particles: int = Field(gt=0, description="Number of particles")
    energy: float = Field(gt=0, description="Reference energy in eV")
    
    # ===== Longitudinal Parameters =====
    zeta_mean: Optional[float] = Field(default=None, description="Mean longitudinal position in m")
    zeta_std: Optional[float] = Field(default=None, ge=0, description="RMS bunch length in m")
    delta_mean: Optional[float] = Field(default=None, description="Mean relative energy deviation")
    delta_std: Optional[float] = Field(default=None, ge=0, description="RMS relative energy spread (ΔP/P)")
    
    # ===== GAUSSIAN Distribution Parameters (explicit RMS sizes) =====
    x_mean: Optional[float] = Field(default=None, description="Mean horizontal position in m")
    x_std: Optional[float] = Field(default=None, ge=0, description="RMS horizontal position in m")
    px_mean: Optional[float] = Field(default=None, description="Mean horizontal momentum")
    px_std: Optional[float] = Field(default=None, ge=0, description="RMS horizontal momentum")
    y_mean: Optional[float] = Field(default=None, description="Mean vertical position in m")
    y_std: Optional[float] = Field(default=None, ge=0, description="RMS vertical position in m")
    py_mean: Optional[float] = Field(default=None, description="Mean vertical momentum")
    py_std: Optional[float] = Field(default=None, ge=0, description="RMS vertical momentum")
    
    # ===== MATCHED Distribution Parameters (emittance-based) =====
    emittance_x: Optional[float] = Field(default=None, ge=0, description="Horizontal emittance in m⋅rad")
    emittance_y: Optional[float] = Field(default=None, ge=0, description="Vertical emittance in m⋅rad")
    beta_x: Optional[float] = Field(default=None, gt=0, description="Horizontal beta function in m")
    beta_y: Optional[float] = Field(default=None, gt=0, description="Vertical beta function in m")
    alpha_x: Optional[float] = Field(default=None, description="Horizontal alpha function")
    alpha_y: Optional[float] = Field(default=None, description="Vertical alpha function")
    dx: Optional[float] = Field(default=None, description="Horizontal dispersion function in m")
    dy: Optional[float] = Field(default=None, description="Vertical dispersion function in m")
    dpx: Optional[float] = Field(default=None, description="Horizontal dispersion derivative")
    dpy: Optional[float] = Field(default=None, description="Vertical dispersion derivative")
    

    # ===== UNIFORM Distribution Parameters (ranges) =====
    x_min: Optional[float] = Field(default=None, description="Minimum horizontal position in m")
    x_max: Optional[float] = Field(default=None, description="Maximum horizontal position in m")
    px_min: Optional[float] = Field(default=None, description="Minimum horizontal momentum")
    px_max: Optional[float] = Field(default=None, description="Maximum horizontal momentum")
    y_min: Optional[float] = Field(default=None, description="Minimum vertical position in m")
    y_max: Optional[float] = Field(default=None, description="Maximum vertical position in m")
    py_min: Optional[float] = Field(default=None, description="Minimum vertical momentum")
    py_max: Optional[float] = Field(default=None, description="Maximum vertical momentum")
    zeta_min: Optional[float] = Field(default=None, description="Minimum longitudinal position in m")
    zeta_max: Optional[float] = Field(default=None, description="Maximum longitudinal position in m")
    delta_min: Optional[float] = Field(default=None, description="Minimum relative energy deviation")
    delta_max: Optional[float] = Field(default=None, description="Maximum relative energy deviation")
    
    # ===== Common Offset Parameters =====
    x_offset: Optional[float] = Field(default=None, description="Horizontal position offset in m (deprecated, use x_mean)")
    y_offset: Optional[float] = Field(default=None, description="Vertical position offset in m (deprecated, use y_mean)")
    px_offset: Optional[float] = Field(default=None, description="Horizontal momentum offset (deprecated, use px_mean)")
    py_offset: Optional[float] = Field(default=None, description="Vertical momentum offset (deprecated, use py_mean)")

    # ===== Advanced Options =====
    correlations: Optional[np.ndarray] = Field(default=None, description="6-D correlations matrix")
    custom_particles: Optional[np.ndarray] = Field(default=None, description="Custom particle coordinates [N, 6] for CUSTOM type")
    
    def validate_parameters(self) -> None:
        """
        Validate that appropriate parameters are provided for the distribution type.
        
        Raises:
            ValueError: If required parameters are missing for the selected distribution type
        """
        if self.distribution_type == DistributionType.GAUSSIAN:
            # For Gaussian, need at least some transverse parameters
            if self.x_std is None or self.y_std is None:
                raise ValueError("GAUSSIAN distribution requires x_std and y_std")
            if self.px_std is None or self.py_std is None:
                raise ValueError("GAUSSIAN distribution requires px_std and py_std")
                
        elif self.distribution_type == DistributionType.MATCHED:
            # For matched, need emittance and Twiss parameters
            if self.emittance_x is None or self.emittance_y is None:
                raise ValueError("MATCHED distribution requires emittance_x and emittance_y")
            if self.beta_x is None or self.beta_y is None:
                raise ValueError("MATCHED distribution requires beta_x and beta_y")
                
        elif self.distribution_type == DistributionType.UNIFORM:
            # For uniform, need ranges
            if self.x_min is None or self.x_max is None:
                raise ValueError("UNIFORM distribution requires x_min and x_max")
            if self.y_min is None or self.y_max is None:
                raise ValueError("UNIFORM distribution requires y_min and y_max")
                
        elif self.distribution_type == DistributionType.CUSTOM:
            # For custom, need particle array
            if self.custom_particles is None:
                raise ValueError("CUSTOM distribution requires custom_particles array")
            if self.custom_particles.shape != (self.num_particles, 6):
                raise ValueError(f"custom_particles must have shape ({self.num_particles}, 6)")
    
    @classmethod
    def from_twiss(
        cls,
        twiss: TwissData,
        distribution_type: DistributionType = DistributionType.GAUSSIAN,
        num_particles: int = 10000,
        emittance_x: float = 1e-9,
        emittance_y: float = 1e-9,
        zeta_std: float = 1e-2,
        delta_std: float = 1e-3,
        s_position: float = 0.0,
        x_mean: float = 0.0,
        y_mean: float = 0.0,
    ) -> "ParticleDistribution":
        """
        Create a matched particle distribution from TwissData.
        
        This is a simplified factory method that automatically extracts
        Twiss parameters at a given s-position and creates a matched
        distribution without requiring manual parameter passing.
        
        Args:
            twiss: TwissData object containing optics functions
            distribution_type: Type of distribution (default: GAUSSIAN for matched beam)
            num_particles: Number of particles to generate
            emittance_x: Horizontal geometric emittance in m⋅rad
            emittance_y: Vertical geometric emittance in m⋅rad
            zeta_std: RMS bunch length in m
            delta_std: RMS relative energy spread (ΔP/P)
            s_position: Longitudinal position to extract Twiss parameters (default: 0.0)
            x_mean: Horizontal beam center offset in m
            y_mean: Vertical beam center offset in m
            
        Returns:
            ParticleDistribution configured with Twiss parameters at s_position
            
        Example:
            >>> # Simple usage with TwissData
            >>> dist = ParticleDistribution.from_twiss(
            ...     twiss=twiss,
            ...     num_particles=10000,
            ...     emittance_x=1e-9,
            ...     emittance_y=1e-9
            ... )
        """
        # Find index closest to requested s-position
        idx = int(np.argmin(np.abs(twiss.s - s_position)))
        
        # Extract Twiss parameters at this position
        beta_x = twiss.beta_x[idx]
        beta_y = twiss.beta_y[idx]
        alpha_x = twiss.alpha_x[idx]
        alpha_y = twiss.alpha_y[idx]
        
        # Extract dispersion if available
        dx = twiss.dx[idx] if twiss.dx is not None else 0.0
        dy = twiss.dy[idx] if twiss.dy is not None else 0.0
        dpx = twiss.dpx[idx] if twiss.dpx is not None else 0.0
        dpy = twiss.dpy[idx] if twiss.dpy is not None else 0.0
        
        return cls(
            distribution_type=distribution_type,
            num_particles=num_particles,
            energy=twiss.reference_energy,
            emittance_x=emittance_x,
            emittance_y=emittance_y,
            beta_x=beta_x,
            beta_y=beta_y,
            alpha_x=alpha_x,
            alpha_y=alpha_y,
            dx=dx,
            dy=dy,
            dpx=dpx,
            dpy=dpy,
            x_mean=x_mean,
            y_mean=y_mean,
            zeta_std=zeta_std,
            delta_std=delta_std,
        )
    
    def get_transverse_rms_sizes(self) -> Dict[str, float]:
        """
        Get transverse RMS sizes regardless of distribution type.
        
        Returns:
            Dictionary with x_rms, y_rms, px_rms, py_rms
        """
        if self.distribution_type == DistributionType.GAUSSIAN:
            return {
                'x_rms': self.x_std if self.x_std is not None else 0.0,
                'y_rms': self.y_std if self.y_std is not None else 0.0,
                'px_rms': self.px_std if self.px_std is not None else 0.0,
                'py_rms': self.py_std if self.py_std is not None else 0.0,
            }
        elif self.distribution_type == DistributionType.MATCHED:
            # Calculate RMS from emittance and beta
            import math
            return {
                'x_rms': math.sqrt(self.beta_x * self.emittance_x) if self.beta_x and self.emittance_x else 0.0,
                'y_rms': math.sqrt(self.beta_y * self.emittance_y) if self.beta_y and self.emittance_y else 0.0,
                'px_rms': math.sqrt(self.emittance_x / self.beta_x) if self.beta_x and self.emittance_x else 0.0,
                'py_rms': math.sqrt(self.emittance_y / self.beta_y) if self.beta_y and self.emittance_y else 0.0,
            }
        else:
            return {'x_rms': 0.0, 'y_rms': 0.0, 'px_rms': 0.0, 'py_rms': 0.0}


class BeamPositionData(PhysicsBaseModel):
    """Data structure for single-turn beam position monitor readings.
    
    Note: BPMs measure beam positions (centroids), not beam sizes.
    For beam size information, use wire scanners or other diagnostic devices.
    """
    name: str = Field(description="Monitor name")
    s_position: float = Field(description="Longitudinal position in m")
    x_mean: float = Field(description="Mean horizontal position in m")
    y_mean: float = Field(description="Mean vertical position in m")
    intensity: float = Field(ge=0, description="Beam intensity (relative)")
    timestamp: Optional[float] = Field(default=None, description="Measurement timestamp")


class TurnByTurnBPMData(PhysicsBaseModel):
    """
    Multi-turn beam position monitor data storage.
    
    This class stores turn-by-turn data from a single BPM, supporting:
    - Full history for LINAC/RAMPING modes
    - Circular buffer for RING mode (memory-efficient)
    - Time-averaged data for RAMPING mode
    
    Note: BPMs measure beam positions (centroids) only, not beam sizes.
    """
    name: str = Field(description="Monitor name")
    s_position: float = Field(description="Longitudinal position in m")
    
    # Turn-by-turn data arrays
    turn_numbers: np.ndarray = Field(description="Turn indices")
    x_mean: np.ndarray = Field(description="Mean horizontal position per turn in m")
    y_mean: np.ndarray = Field(description="Mean vertical position per turn in m")
    intensity: np.ndarray = Field(description="Beam intensity per turn (relative)")
    
    # Optional time-series data (for RAMPING mode)
    timestamps: Optional[np.ndarray] = Field(default=None, description="Timestamp for each turn in seconds")
    
    # Statistics
    num_turns: int = Field(description="Number of turns recorded")
    
    @field_validator('intensity')
    @classmethod
    def validate_non_negative(cls, v):
        """Ensure physical quantities are non-negative."""
        if np.any(v < 0):
            raise ValueError("Intensity must be non-negative")
        return v
    
    @field_validator('turn_numbers', 'x_mean', 'y_mean', 'intensity')
    @classmethod
    def validate_array_lengths(cls, v, info):
        """Validate that all arrays have consistent length."""
        if 'turn_numbers' in info.data:
            expected_len = len(info.data['turn_numbers'])
            if len(v) != expected_len:
                raise ValueError(f"All data arrays must have same length as turn_numbers (got {len(v)}, expected {expected_len})")
        return v
    
    def get_turn_data(self, turn: int) -> Optional[BeamPositionData]:
        """
        Get BPM data for a specific turn.
        
        Args:
            turn: Turn number to retrieve
            
        Returns:
            BeamPositionData for the specified turn, or None if not found
        """
        # Find index of the turn
        indices = np.where(self.turn_numbers == turn)[0]
        if len(indices) == 0:
            return None
        
        idx = indices[0]
        return BeamPositionData(
            name=self.name,
            s_position=self.s_position,
            x_mean=float(self.x_mean[idx]),
            y_mean=float(self.y_mean[idx]),
            intensity=float(self.intensity[idx]),
            timestamp=float(self.timestamps[idx]) if self.timestamps is not None else None
        )
    
    def get_turn_range(self, start_turn: int, end_turn: int) -> 'TurnByTurnBPMData':
        """
        Extract data for a range of turns.
        
        Args:
            start_turn: Starting turn (inclusive)
            end_turn: Ending turn (inclusive)
            
        Returns:
            New TurnByTurnBPMData with filtered data
        """
        mask = (self.turn_numbers >= start_turn) & (self.turn_numbers <= end_turn)
        
        return TurnByTurnBPMData(
            name=self.name,
            s_position=self.s_position,
            turn_numbers=self.turn_numbers[mask],
            x_mean=self.x_mean[mask],
            y_mean=self.y_mean[mask],
            intensity=self.intensity[mask],
            timestamps=self.timestamps[mask] if self.timestamps is not None else None,
            num_turns=int(np.sum(mask))
        )
    
    def compute_statistics(self) -> Dict[str, Any]:
        """
        Compute statistical summary of the data.
        
        Returns:
            Dictionary with mean, std, min, max for each quantity
        """
        return {
            'x_mean': {'mean': float(np.mean(self.x_mean)), 'std': float(np.std(self.x_mean)),
                      'min': float(np.min(self.x_mean)), 'max': float(np.max(self.x_mean))},
            'y_mean': {'mean': float(np.mean(self.y_mean)), 'std': float(np.std(self.y_mean)),
                      'min': float(np.min(self.y_mean)), 'max': float(np.max(self.y_mean))},
            'intensity': {'mean': float(np.mean(self.intensity)), 'std': float(np.std(self.intensity)),
                         'min': float(np.min(self.intensity)), 'max': float(np.max(self.intensity))},
            'num_turns': self.num_turns
        }


class BPMBuffer:
    """
    Circular buffer for BPM data in RING mode simulations.
    
    This class provides memory-efficient storage of recent BPM readings
    using a circular buffer. Older data is automatically discarded when
    the buffer is full.
    
    Note: BPMs measure beam positions (centroids) only, not beam sizes.
    
    NOT a Pydantic model because it uses deque and has mutable state.
    """
    
    def __init__(self, name: str, s_position: float, buffer_size: int = 10000):
        """
        Initialize BPM buffer.
        
        Args:
            name: Monitor name
            s_position: Longitudinal position in m
            buffer_size: Maximum number of turns to store
        """
        self.name = name
        self.s_position = s_position
        self.buffer_size = buffer_size
        
        # Circular buffers for each quantity
        self._turn_numbers = deque(maxlen=buffer_size)
        self._x_mean = deque(maxlen=buffer_size)
        self._y_mean = deque(maxlen=buffer_size)
        self._intensity = deque(maxlen=buffer_size)
        self._timestamps = deque(maxlen=buffer_size)
    
    def append(self, turn: int, x_mean: float, y_mean: float, 
               intensity: float, timestamp: Optional[float] = None):
        """
        Add new BPM reading to buffer.
        
        Args:
            turn: Turn number
            x_mean: Mean horizontal position in m
            y_mean: Mean vertical position in m
            intensity: Beam intensity (relative)
            timestamp: Optional timestamp in seconds
        """
        self._turn_numbers.append(turn)
        self._x_mean.append(x_mean)
        self._y_mean.append(y_mean)
        self._intensity.append(intensity)
        self._timestamps.append(timestamp)
    
    def append_from_data(self, turn: int, data: BeamPositionData):
        """
        Add BPM reading from BeamPositionData object.
        
        Args:
            turn: Turn number
            data: BeamPositionData object
        """
        self.append(
            turn=turn,
            x_mean=data.x_mean,
            y_mean=data.y_mean,
            intensity=data.intensity,
            timestamp=data.timestamp
        )
    
    def get_recent(self, num_turns: int) -> TurnByTurnBPMData:
        """
        Get most recent N turns of data.
        
        Args:
            num_turns: Number of recent turns to retrieve
            
        Returns:
            TurnByTurnBPMData with recent data
        """
        n = min(num_turns, len(self._turn_numbers))
        
        return TurnByTurnBPMData(
            name=self.name,
            s_position=self.s_position,
            turn_numbers=np.array(list(self._turn_numbers)[-n:]),
            x_mean=np.array(list(self._x_mean)[-n:]),
            y_mean=np.array(list(self._y_mean)[-n:]),
            intensity=np.array(list(self._intensity)[-n:]),
            timestamps=np.array(list(self._timestamps)[-n:]) if any(t is not None for t in list(self._timestamps)[-n:]) else None,
            num_turns=n
        )
    
    def get_all(self) -> TurnByTurnBPMData:
        """
        Get all data in buffer.
        
        Returns:
            TurnByTurnBPMData with all buffered data
        """
        return TurnByTurnBPMData(
            name=self.name,
            s_position=self.s_position,
            turn_numbers=np.array(list(self._turn_numbers)),
            x_mean=np.array(list(self._x_mean)),
            y_mean=np.array(list(self._y_mean)),
            intensity=np.array(list(self._intensity)),
            timestamps=np.array(list(self._timestamps)) if any(t is not None for t in self._timestamps) else None,
            num_turns=len(self._turn_numbers)
        )
    
    def get_turn_range(self, start_turn: int, end_turn: int) -> TurnByTurnBPMData:
        """
        Get data for a specific turn range.
        
        Args:
            start_turn: Starting turn (inclusive)
            end_turn: Ending turn (inclusive)
            
        Returns:
            TurnByTurnBPMData with filtered data
        """
        # Convert deques to arrays for easier filtering
        turn_array = np.array(list(self._turn_numbers))
        mask = (turn_array >= start_turn) & (turn_array <= end_turn)
        
        indices = np.where(mask)[0]
        
        return TurnByTurnBPMData(
            name=self.name,
            s_position=self.s_position,
            turn_numbers=turn_array[mask],
            x_mean=np.array([list(self._x_mean)[i] for i in indices]),
            y_mean=np.array([list(self._y_mean)[i] for i in indices]),
            intensity=np.array([list(self._intensity)[i] for i in indices]),
            timestamps=np.array([list(self._timestamps)[i] for i in indices]) if any(t is not None for t in self._timestamps) else None,
            num_turns=int(np.sum(mask))
        )
    
    def clear(self):
        """Clear all data from buffer."""
        self._turn_numbers.clear()
        self._x_mean.clear()
        self._y_mean.clear()
        self._intensity.clear()
        self._timestamps.clear()
    
    @property
    def size(self) -> int:
        """Current number of turns stored in buffer."""
        return len(self._turn_numbers)
    
    @property
    def is_full(self) -> bool:
        """Check if buffer is at maximum capacity."""
        return len(self._turn_numbers) == self.buffer_size


class RampingBPMData(PhysicsBaseModel):
    """
    Time-averaged BPM data for RAMPING mode simulations.
    
    During ramping, BPM readings are averaged over configurable time windows
    to reduce data volume while capturing parameter evolution.
    
    Note: BPMs measure beam positions (centroids) only, not beam sizes.
    """
    name: str = Field(description="Monitor name")
    s_position: float = Field(description="Longitudinal position in m")
    
    # Time-series data (averaged over time windows)
    time_points: np.ndarray = Field(description="Time points in seconds")
    x_mean: np.ndarray = Field(description="Mean horizontal position in m")
    y_mean: np.ndarray = Field(description="Mean vertical position in m")
    intensity: np.ndarray = Field(description="Beam intensity (relative)")
    
    # Averaging metadata
    averaging_time: float = Field(gt=0, description="Time window for averaging in seconds")
    turns_per_point: np.ndarray = Field(description="Number of turns averaged per time point")
    
    # Statistics per time point
    x_mean_std: Optional[np.ndarray] = Field(default=None, description="Standard deviation of x_mean within each time window")
    y_mean_std: Optional[np.ndarray] = Field(default=None, description="Standard deviation of y_mean within each time window")
    
    num_points: int = Field(description="Number of time points")
    
    @field_validator('intensity')
    @classmethod
    def validate_non_negative(cls, v):
        """Ensure physical quantities are non-negative."""
        if np.any(v < 0):
            raise ValueError("Intensity must be non-negative")
        return v
    
    @field_validator('time_points', 'x_mean', 'y_mean', 'intensity', 'turns_per_point')
    @classmethod
    def validate_array_lengths(cls, v, info):
        """Validate that all arrays have consistent length."""
        if 'time_points' in info.data:
            expected_len = len(info.data['time_points'])
            if len(v) != expected_len:
                raise ValueError(f"All data arrays must have same length as time_points (got {len(v)}, expected {expected_len})")
        return v
    
    def get_time_point(self, time: float) -> Optional[BeamPositionData]:
        """
        Get BPM data at a specific time (nearest time point).
        
        Args:
            time: Time in seconds
            
        Returns:
            BeamPositionData for the nearest time point
        """
        idx = int(np.argmin(np.abs(self.time_points - time)))
        
        return BeamPositionData(
            name=self.name,
            s_position=self.s_position,
            x_mean=float(self.x_mean[idx]),
            y_mean=float(self.y_mean[idx]),
            intensity=float(self.intensity[idx]),
            timestamp=float(self.time_points[idx])
        )
    
    def get_time_range(self, start_time: float, end_time: float) -> 'RampingBPMData':
        """
        Extract data for a time range.
        
        Args:
            start_time: Starting time in seconds (inclusive)
            end_time: Ending time in seconds (inclusive)
            
        Returns:
            New RampingBPMData with filtered data
        """
        mask = (self.time_points >= start_time) & (self.time_points <= end_time)
        
        return RampingBPMData(
            name=self.name,
            s_position=self.s_position,
            time_points=self.time_points[mask],
            x_mean=self.x_mean[mask],
            y_mean=self.y_mean[mask],
            intensity=self.intensity[mask],
            averaging_time=self.averaging_time,
            turns_per_point=self.turns_per_point[mask],
            x_mean_std=self.x_mean_std[mask] if self.x_mean_std is not None else None,
            y_mean_std=self.y_mean_std[mask] if self.y_mean_std is not None else None,
            num_points=int(np.sum(mask))
        )
    
    def interpolate_at_time(self, time: float, method: str = 'linear') -> BeamPositionData:
        """
        Interpolate BPM data at arbitrary time point.
        
        Args:
            time: Time in seconds
            method: Interpolation method ('linear', 'nearest', 'cubic')
            
        Returns:
            Interpolated BeamPositionData
        """
        from scipy import interpolate
        
        if method == 'nearest':
            return self.get_time_point(time)
        
        # Linear or cubic interpolation
        kind = 'linear' if method == 'linear' else 'cubic'
        
        x_mean_interp = interpolate.interp1d(self.time_points, self.x_mean, kind=kind, fill_value='extrapolate')
        y_mean_interp = interpolate.interp1d(self.time_points, self.y_mean, kind=kind, fill_value='extrapolate')
        intensity_interp = interpolate.interp1d(self.time_points, self.intensity, kind=kind, fill_value='extrapolate')
        
        return BeamPositionData(
            name=self.name,
            s_position=self.s_position,
            x_mean=float(x_mean_interp(time)),
            y_mean=float(y_mean_interp(time)),
            intensity=float(intensity_interp(time)),
            timestamp=time
        )


# TwissData is now defined in eicvibe.optics
# Kept here as alias for backward compatibility
# Original class definition moved to package level

if False:  # Type checking only - actual class imported above
    class _TwissDataDocOnly(PhysicsBaseModel):
        """
        [DEPRECATED: Use eicvibe.optics.TwissData instead]
        
        Comprehensive Twiss parameter data structure.
        
        This class has been moved to eicvibe.optics.TwissData as it is
        a package-level standard that any simulation engine can populate.
        
        Contains all optical functions calculated along the lattice,
        with support for different simulation modes and coordinate systems.
        Compatible with XSuite's comprehensive Twiss table output.
    """
    # ========== Core Longitudinal Positions ==========
    s: np.ndarray = Field(description="Longitudinal positions along lattice in m")
    
    # ========== Beta Functions (Core Optical Functions) ==========
    beta_x: np.ndarray = Field(description="Horizontal beta function in m")
    beta_y: np.ndarray = Field(description="Vertical beta function in m")
    
    # ========== Alpha Functions ==========
    alpha_x: np.ndarray = Field(description="Horizontal alpha function")
    alpha_y: np.ndarray = Field(description="Vertical alpha function")
    
    # ========== Gamma Functions ==========
    gamma_x: Optional[np.ndarray] = Field(default=None, description="Horizontal gamma function in m^-1")
    gamma_y: Optional[np.ndarray] = Field(default=None, description="Vertical gamma function in m^-1")
    
    # ========== Phase Advance ==========
    mu_x: Optional[np.ndarray] = Field(default=None, description="Horizontal phase advance in radians")
    mu_y: Optional[np.ndarray] = Field(default=None, description="Vertical phase advance in radians")
    
    # ========== Dispersion Functions ==========
    dx: Optional[np.ndarray] = Field(default=None, description="Horizontal dispersion in m")
    dy: Optional[np.ndarray] = Field(default=None, description="Vertical dispersion in m")
    dpx: Optional[np.ndarray] = Field(default=None, description="Horizontal dispersion derivative")
    dpy: Optional[np.ndarray] = Field(default=None, description="Vertical dispersion derivative")
    
    # ========== Higher-Order Dispersion (XSuite) ==========
    ddx: Optional[np.ndarray] = Field(default=None, description="Second-order horizontal dispersion in m")
    ddy: Optional[np.ndarray] = Field(default=None, description="Second-order vertical dispersion in m")
    ddpx: Optional[np.ndarray] = Field(default=None, description="Second-order horizontal dispersion derivative")
    ddpy: Optional[np.ndarray] = Field(default=None, description="Second-order vertical dispersion derivative")
    
    # ========== Closed Orbit (RING Mode) ==========
    x: Optional[np.ndarray] = Field(default=None, description="Horizontal closed orbit in m")
    px: Optional[np.ndarray] = Field(default=None, description="Horizontal momentum in closed orbit")
    y: Optional[np.ndarray] = Field(default=None, description="Vertical closed orbit in m")
    py: Optional[np.ndarray] = Field(default=None, description="Vertical momentum in closed orbit")
    zeta: Optional[np.ndarray] = Field(default=None, description="Longitudinal closed orbit position in m")
    delta: Optional[np.ndarray] = Field(default=None, description="Relative energy deviation in closed orbit")
    
    # ========== Energy Parameters ==========
    energy: Optional[np.ndarray] = Field(default=None, description="Beam energy along lattice in eV")
    ptau: Optional[np.ndarray] = Field(default=None, description="Longitudinal momentum deviation")
    
    # ========== Amplitude-Dependent Chromaticity (XSuite) ==========
    ax_chrom: Optional[np.ndarray] = Field(default=None, description="Horizontal amplitude-dependent chromaticity")
    ay_chrom: Optional[np.ndarray] = Field(default=None, description="Vertical amplitude-dependent chromaticity")
    bx_chrom: Optional[np.ndarray] = Field(default=None, description="Horizontal second-order amplitude chromaticity")
    by_chrom: Optional[np.ndarray] = Field(default=None, description="Vertical second-order amplitude chromaticity")
    
    # ========== W Functions (XSuite Chromaticity) ==========
    wx: Optional[np.ndarray] = Field(default=None, description="Horizontal W function for chromaticity")
    wy: Optional[np.ndarray] = Field(default=None, description="Vertical W function for chromaticity")
    wx_chrom: Optional[np.ndarray] = Field(default=None, description="Horizontal W-function chromaticity")
    wy_chrom: Optional[np.ndarray] = Field(default=None, description="Vertical W-function chromaticity")
    
    # ========== Coupling Parameters (XSuite 4D Coupling) ==========
    c_minus: Optional[np.ndarray] = Field(default=None, description="Coupling coefficient C-")
    c_plus: Optional[np.ndarray] = Field(default=None, description="Coupling coefficient C+")
    
    # ========== Transfer Matrix Elements (R-matrix, XSuite) ==========
    # First-order (2×2 blocks for x and y)
    r11: Optional[np.ndarray] = Field(default=None, description="R11 transfer matrix element (x|x)")
    r12: Optional[np.ndarray] = Field(default=None, description="R12 transfer matrix element (x|px)")
    r21: Optional[np.ndarray] = Field(default=None, description="R21 transfer matrix element (px|x)")
    r22: Optional[np.ndarray] = Field(default=None, description="R22 transfer matrix element (px|px)")
    r33: Optional[np.ndarray] = Field(default=None, description="R33 transfer matrix element (y|y)")
    r34: Optional[np.ndarray] = Field(default=None, description="R34 transfer matrix element (y|py)")
    r43: Optional[np.ndarray] = Field(default=None, description="R43 transfer matrix element (py|y)")
    r44: Optional[np.ndarray] = Field(default=None, description="R44 transfer matrix element (py|py)")
    
    # ========== Longitudinal Dynamics (RING Mode) ==========
    momentum_compaction: Optional[float] = Field(default=None, description="Momentum compaction factor (alpha_c)")
    slip_factor: Optional[float] = Field(default=None, description="Phase slip factor (eta)")
    T_rev: Optional[float] = Field(default=None, description="Revolution period in seconds")
    f_rev: Optional[float] = Field(default=None, description="Revolution frequency in Hz")
    
    # ========== Mode-Specific Metadata ==========
    simulation_mode: SimulationMode = Field(description="Simulation mode used for calculation")
    reference_energy: float = Field(description="Reference energy in eV")
    reference_momentum: Optional[float] = Field(default=None, description="Reference momentum p0c in eV")
    particle_mass: Optional[float] = Field(default=None, description="Particle rest mass in eV")
    particle_charge: Optional[int] = Field(default=None, description="Particle charge in elementary charges")
    
    # ========== Tune Information (RING Mode) ==========
    tune_x: Optional[float] = Field(default=None, description="Horizontal tune (Qx)")
    tune_y: Optional[float] = Field(default=None, description="Vertical tune (Qy)")
    tune_z: Optional[float] = Field(default=None, description="Synchrotron tune (Qs)")
    
    # ========== Chromaticity (RING Mode) ==========
    chromaticity_x: Optional[float] = Field(default=None, description="Horizontal chromaticity (dQx/dp)")
    chromaticity_y: Optional[float] = Field(default=None, description="Vertical chromaticity (dQy/dp)")
    
    # Second-order chromaticity (XSuite)
    chromaticity_x2: Optional[float] = Field(default=None, description="Second-order horizontal chromaticity")
    chromaticity_y2: Optional[float] = Field(default=None, description="Second-order vertical chromaticity")
    
    # ========== Synchrotron Radiation (XSuite) ==========
    damping_time_x: Optional[float] = Field(default=None, description="Horizontal damping time in seconds")
    damping_time_y: Optional[float] = Field(default=None, description="Vertical damping time in seconds")
    damping_time_z: Optional[float] = Field(default=None, description="Longitudinal damping time in seconds")
    emittance_x: Optional[float] = Field(default=None, description="Equilibrium horizontal emittance in m·rad")
    emittance_y: Optional[float] = Field(default=None, description="Equilibrium vertical emittance in m·rad")
    emittance_z: Optional[float] = Field(default=None, description="Equilibrium longitudinal emittance in eV·s")
    energy_loss_per_turn: Optional[float] = Field(default=None, description="Energy loss per turn in eV")
    
    # ========== Time-Dependent Parameters (RAMPING Mode) ==========
    time_points: Optional[np.ndarray] = Field(default=None, description="Time points for ramping in s")
    
    # ========== Element Reference Information ==========
    element_names: Optional[List[str]] = Field(default=None, description="Element names at each position")
    element_types: Optional[List[str]] = Field(default=None, description="Element types at each position")
    element_lengths: Optional[np.ndarray] = Field(default=None, description="Element lengths in m")
    
    # ========== Beam-Beam and Collective Effects ==========
    beam_beam_strength: Optional[np.ndarray] = Field(default=None, description="Beam-beam parameter at interaction points")
    space_charge_strength: Optional[np.ndarray] = Field(default=None, description="Space charge tune shift along lattice")
    
    @field_validator('beta_x', 'beta_y')
    @classmethod
    def validate_beta_functions(cls, v):
        """Validate that beta functions are positive."""
        if np.any(v <= 0):
            raise ValueError("Beta functions must be positive")
        return v
    
    @field_validator('s', 'beta_x', 'beta_y', 'alpha_x', 'alpha_y')
    @classmethod
    def validate_array_lengths(cls, v, info):
        """Validate that all arrays have the same length as s."""
        if info.data.get('s') is not None:
            s_len = len(info.data['s'])
            if len(v) != s_len:
                raise ValueError(f"All arrays must have the same length as s (got {len(v)}, expected {s_len})")
        return v

    def get_envelope_x(self, emittance_x: float) -> np.ndarray:
        """Calculate horizontal beam envelope from beta function and emittance."""
        return np.sqrt(self.beta_x * emittance_x)
    
    def get_envelope_y(self, emittance_y: float) -> np.ndarray:
        """Calculate vertical beam envelope from beta function and emittance."""
        return np.sqrt(self.beta_y * emittance_y)
    
    def find_position_index(self, s_target: float) -> int:
        """Find the closest index for a given longitudinal position."""
        return int(np.argmin(np.abs(self.s - s_target)))
    
    def get_at_position(self, s_target: float) -> Dict[str, Any]:
        """Get all Twiss parameters at a specific longitudinal position."""
        idx = self.find_position_index(s_target)
        result = {
            's': self.s[idx],
            'beta_x': self.beta_x[idx],
            'beta_y': self.beta_y[idx],
            'alpha_x': self.alpha_x[idx],
            'alpha_y': self.alpha_y[idx],
        }
        
        # Add optional array parameters if available
        array_params = [
            'gamma_x', 'gamma_y', 'mu_x', 'mu_y', 
            'dx', 'dy', 'dpx', 'dpy', 'ddx', 'ddy', 'ddpx', 'ddpy',
            'x', 'px', 'y', 'py', 'zeta', 'delta', 'energy', 'ptau',
            'ax_chrom', 'ay_chrom', 'bx_chrom', 'by_chrom',
            'wx', 'wy', 'wx_chrom', 'wy_chrom',
            'c_minus', 'c_plus',
            'r11', 'r12', 'r21', 'r22', 'r33', 'r34', 'r43', 'r44',
            'element_lengths', 'beam_beam_strength', 'space_charge_strength'
        ]
        
        for param in array_params:
            value = getattr(self, param, None)
            if value is not None and len(value) > idx:
                result[param] = value[idx]
        
        return result
    
    @property
    def length(self) -> float:
        """Total length of the lattice."""
        return float(self.s[-1] - self.s[0]) if len(self.s) > 0 else 0.0
    
    @property
    def gamma_x_computed(self) -> np.ndarray:
        """Calculate gamma_x = (1 + alpha_x²) / beta_x."""
        return (1 + self.alpha_x**2) / self.beta_x
    
    @property
    def gamma_y_computed(self) -> np.ndarray:
        """Calculate gamma_y = (1 + alpha_y²) / beta_y."""
        return (1 + self.alpha_y**2) / self.beta_y
    
    def get_natural_chromaticity(self) -> Dict[str, float]:
        """
        Calculate natural chromaticity from dispersion and beta functions.
        
        This is an approximate calculation using the sextupole-free natural chromaticity formula.
        For accurate results including sextupoles, use the engine's native Twiss calculation.
        
        Returns:
            Dictionary with 'xi_x' and 'xi_y' natural chromaticities
        """
        if self.dx is None or self.dpx is None:
            raise ValueError("Dispersion data required for chromaticity calculation")
        
        # Natural chromaticity: ξ = -1/(4π) ∮ (D/β²) ds (approximate)
        # This is a simplified formula; accurate calculation needs integration
        ds = np.diff(self.s, prepend=0)
        
        xi_x = -np.sum((self.dx / self.beta_x**2) * ds) / (4 * np.pi)
        xi_y = np.sum((self.dy / self.beta_y**2) * ds) / (4 * np.pi) if self.dy is not None else 0.0
        
        return {'xi_x': float(xi_x), 'xi_y': float(xi_y)}
    
    def get_momentum_compaction_approx(self) -> float:
        """
        Calculate approximate momentum compaction factor from dispersion.
        
        Returns:
            Momentum compaction factor alpha_c
        """
        if self.dx is None:
            raise ValueError("Dispersion data required for momentum compaction calculation")
        
        if self.momentum_compaction is not None:
            return self.momentum_compaction
        
        # α_c = (1/C) ∮ (D/ρ) ds ≈ (1/C) ∮ D·h ds for separated function lattices
        # This is approximate; accurate calculation needs bending angles
        ds = np.diff(self.s, prepend=0)
        circumference = self.s[-1] if len(self.s) > 0 else 1.0
        
        alpha_c = np.sum(self.dx * ds) / circumference
        return float(alpha_c)
    
    def get_damping_partition_numbers(self) -> Dict[str, float]:
        """
        Get Robinson's damping partition numbers if available.
        
        Returns:
            Dictionary with Jx, Jy, Jz partition numbers
        """
        if self.damping_time_x is None:
            raise ValueError("Damping time data not available")
        
        # Robinson's theorem: Jx + Jy + Jz = 4
        # From damping times: τᵢ = τ₀/Jᵢ
        if self.damping_time_x and self.damping_time_y and self.damping_time_z:
            tau_0 = (self.damping_time_x + self.damping_time_y + self.damping_time_z) / 4.0
            Jx = tau_0 / self.damping_time_x
            Jy = tau_0 / self.damping_time_y
            Jz = tau_0 / self.damping_time_z
            return {'Jx': float(Jx), 'Jy': float(Jy), 'Jz': float(Jz), 'tau_0': float(tau_0)}
        else:
            return {}
    
    def export_to_dict(self, include_arrays: bool = True) -> Dict[str, Any]:
        """
        Export Twiss data to dictionary format (XSuite-compatible).
        
        Args:
            include_arrays: If True, include array data; if False, only scalars
            
        Returns:
            Dictionary with all Twiss parameters
        """
        result = {
            'simulation_mode': self.simulation_mode.value,
            'reference_energy': self.reference_energy,
        }
        
        # Add scalar parameters
        scalars = [
            'tune_x', 'tune_y', 'tune_z',
            'chromaticity_x', 'chromaticity_y', 'chromaticity_x2', 'chromaticity_y2',
            'momentum_compaction', 'slip_factor', 'T_rev', 'f_rev',
            'damping_time_x', 'damping_time_y', 'damping_time_z',
            'emittance_x', 'emittance_y', 'emittance_z',
            'energy_loss_per_turn',
            'reference_momentum', 'particle_mass', 'particle_charge'
        ]
        
        for param in scalars:
            value = getattr(self, param, None)
            if value is not None:
                result[param] = value
        
        # Add array parameters if requested
        if include_arrays:
            array_params = [
                's', 'beta_x', 'beta_y', 'alpha_x', 'alpha_y', 'gamma_x', 'gamma_y',
                'mu_x', 'mu_y', 'dx', 'dy', 'dpx', 'dpy', 'ddx', 'ddy', 'ddpx', 'ddpy',
                'x', 'px', 'y', 'py', 'zeta', 'delta', 'energy', 'ptau',
                'ax_chrom', 'ay_chrom', 'bx_chrom', 'by_chrom',
                'wx', 'wy', 'wx_chrom', 'wy_chrom', 'c_minus', 'c_plus',
                'r11', 'r12', 'r21', 'r22', 'r33', 'r34', 'r43', 'r44',
                'element_lengths', 'beam_beam_strength', 'space_charge_strength'
            ]
            
            for param in array_params:
                value = getattr(self, param, None)
                if value is not None:
                    result[param] = value
            
            # Add list parameters
            if self.element_names is not None:
                result['element_names'] = self.element_names
            if self.element_types is not None:
                result['element_types'] = self.element_types
        
        return result


class TrackingResults(PhysicsBaseModel):
    """
    Results from particle tracking simulation.
    
    Contains all output data from a completed simulation including
    particle statistics, monitor data, Twiss parameters, and performance metrics.
    """
    simulation_mode: SimulationMode = Field(description="Simulation mode used")
    particles_initial: int = Field(description="Initial particle count")
    particles_final: int = Field(description="Final particle count")
    transmission: float = Field(ge=0, le=1, description="Transmission efficiency")
    turns_completed: int = Field(ge=0, description="Turns successfully completed")
    execution_time: float = Field(ge=0, description="Execution time in seconds")
    
    # Optional detailed data
    particle_coordinates: Optional[np.ndarray] = Field(default=None, description="Final particle coordinates [N, 6]")
    monitor_data: Optional[Dict[str, BeamPositionData]] = Field(default=None, description="BPM readings")
    twiss_data: Optional[TwissData] = Field(default=None, description="Twiss parameters and optical functions")
    
    # Performance and diagnostic information
    memory_usage_mb: Optional[float] = Field(default=None, description="Peak memory usage in MB")
    particles_lost: Optional[List[Dict[str, Any]]] = Field(default=None, description="Lost particle information")


class EngineConfiguration(PhysicsBaseModel):
    """Base configuration for simulation engines."""
    name: str = Field(description="Engine name")
    context: str = Field(default="cpu", description="Computation context (cpu, cuda, opencl)")
    precision: str = Field(default="double", description="Numerical precision (single, double)")
    enable_gpu: bool = Field(default=False, description="Enable GPU acceleration if available")
    max_memory_mb: Optional[int] = Field(default=None, description="Maximum memory usage in MB")
    log_level: str = Field(default="INFO", description="Logging level")
    
    # Engine-specific configuration can be stored in a flexible dict
    engine_specific: Dict[str, Any] = Field(default_factory=dict, description="Engine-specific settings")


class SimulationError(Exception):
    """Base exception class for simulation errors."""
    pass


class EngineNotFoundError(SimulationError):
    """Raised when a requested simulation engine is not available."""
    pass


class LatticeConversionError(SimulationError):
    """Raised when lattice conversion fails."""
    pass


class TrackingError(SimulationError):
    """Raised when particle tracking fails."""
    pass


class ConfigurationError(SimulationError):
    """Raised when simulation configuration is invalid."""
    pass
