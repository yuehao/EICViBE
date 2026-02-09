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
import threading
import multiprocessing as mp
import queue
import uuid
from pathlib import Path
import pickle
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
    TrackingError,
    TrackingStatus,
    LatticeChangeAction,
    MonitorDataRequest
)

logger = logging.getLogger(__name__)


class TrackingMessage:
    """Base class for inter-process communication messages."""
    def __init__(self, session_id: str, **kwargs):
        self.session_id = session_id
        self.timestamp = time.time()
        # Store additional data as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)


class StatusQuery(TrackingMessage):
    """Request tracking status."""
    pass  # Inherits all functionality from base


class DataRequest(TrackingMessage):
    """Request monitor data."""
    def __init__(self, session_id: str, request: MonitorDataRequest):
        super().__init__(session_id, request=request)


class LatticeChangeRequest(TrackingMessage):
    """Request lattice parameter change."""
    def __init__(self, session_id: str, change: LatticeChangeAction):
        super().__init__(session_id, change=change)


class ControlCommand(TrackingMessage):
    """Control commands for tracking process."""
    def __init__(self, session_id: str, command: str):
        super().__init__(session_id, command=command)


class TrackingResponse:
    """Base class for tracking process responses."""
    def __init__(self, session_id: str, success: bool = True, error: Optional[str] = None, **kwargs):
        self.session_id = session_id
        self.success = success
        self.error = error
        self.timestamp = time.time()
        # Store additional data as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)


class StatusResponse(TrackingResponse):
    """Status response from tracking process."""
    def __init__(self, session_id: str, status: TrackingStatus, current_turn: int = 0, 
                 particles_alive: int = 0, **kwargs):
        super().__init__(session_id, status=status, current_turn=current_turn, 
                        particles_alive=particles_alive, **kwargs)


class DataResponse(TrackingResponse):
    """Monitor data response from tracking process."""
    def __init__(self, session_id: str, data: Dict[str, Any], **kwargs):
        super().__init__(session_id, data=data, **kwargs)


class TrackingSession:
    """Manages a multi-process tracking session."""
    def __init__(self, session_id: str, mode: SimulationMode, process: Optional[mp.Process] = None):
        self.session_id = session_id
        self.mode = mode
        self.process = process
        self.status = TrackingStatus.IDLE
        self.created_time = time.time()
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        
        # Communication queues
        self.command_queue: Optional[mp.Queue] = None
        self.response_queue: Optional[mp.Queue] = None
        
        # Data storage
        self.data_file: Optional[Path] = None
        self.monitor_buffer: Dict[str, List] = {}
        
        # Change tracking
        self.scheduled_changes: List[LatticeChangeAction] = []
        self.applied_changes: List[LatticeChangeAction] = []


class SimulationCallback(ABC):
    """
    Abstract base class for simulation event callbacks.
    
    Callbacks can be attached to simulation engines to receive real-time
    updates about simulation progress and results. These are NOT beam position
    monitors - those are physical elements in the lattice file.
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


class ProgressCallback(SimulationCallback):
    """Simple progress reporting callback implementation."""
    
    def __init__(self, show_progress: bool = True):
        self.show_progress = show_progress
        self.start_time = None
        self.total_turns = 0
        
    def on_simulation_start(self, parameters: TrackingParameters):
        self.start_time = time.time()
        total_steps = getattr(parameters, 'num_steps', None)
        self.total_turns = parameters.num_turns if parameters.mode == SimulationMode.RING else (total_steps or 1)
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
    - Event callback system for real-time updates
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
        self.callbacks: List[SimulationCallback] = []
        
        # Multi-process tracking state
        self._sessions: Dict[str, TrackingSession] = {}
        self._session_lock = threading.Lock()
        self._data_dir: Optional[Path] = None
        
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
        self.logger.setLevel(getattr(logging, self.config.log_level))
        
        # Add default progress callback
        if not hasattr(self, '_no_default_callback'):
            self.add_callback(ProgressCallback())
    
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
        Perform particle tracking simulation in a separate process.
        
        All tracking operations (RING, LINAC, RAMPING) should run in separate 
        processes for proper isolation. Implementations should delegate to 
        background tracking processes and manage the session lifecycle.
        
        IMPORTANT: Implementations MUST ensure that particles are stored as 
        self._particles for use by other methods (get_bmp_readings, etc.)
        
        PARAMETER VALIDATION: Implementations should call validate_tracking_parameters()
        before starting tracking to ensure parameters are compatible with the engine.
        
        Args:
            particles: Particle distribution from create_particles()
            parameters: Tracking configuration (may be engine-specific subclass)
            
        Returns:
            Complete tracking results including statistics and monitor data
            
        Raises:
            TrackingError: If tracking simulation fails
            ValueError: If parameters are invalid or incompatible
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
    def get_bpm_readings(self, monitor_elements: List[Element]) -> Dict[str, BeamPositionData]:
        """
        Simulate beam position monitor readings from current simulation state.
        
        This method calculates BPM readings from the internal particle distribution
        (self._particles) that is being tracked by the simulation engine.
        
        Args:
            monitor_elements: List of BPM/Monitor elements
            
        Returns:
            Dictionary mapping monitor names to BeamPositionData
            
        Raises:
            SimulationError: If no particles are loaded or simulation not initialized
        """
        pass
    
    @abstractmethod
    def apply_lattice_change(
        self, 
        converted_lattice: Any,
        change: LatticeChangeAction,
        current_value: Optional[float] = None
    ) -> bool:
        """
        Apply a lattice parameter change to the engine-specific lattice.
        
        This method implements the engine-specific logic for modifying lattice
        parameters during simulation. Different engines have different internal
        representations and APIs for parameter updates.
        
        Args:
            converted_lattice: Engine-specific lattice object
            change: Lattice change specification with element, parameter, and new value
            current_value: Current parameter value (if known), used for ramping calculations
            
        Returns:
            True if change was successfully applied
            
        Raises:
            ValueError: If element or parameter not found
            SimulationError: If parameter update fails
            
        Example:
            For XSuite:
                element = converted_lattice[change.element_name]
                element.knl[0] = change.new_value  # For dipole strength
                
            For MAD-X:
                madx.input(f"{change.element_name}->{change.parameter_name} = {change.new_value};")
        """
        pass
    
    # === Advanced Simulation Control with Multi-Process Support ===
    
    def setup_data_directory(self, data_dir: Optional[Path] = None) -> Path:
        """
        Setup data directory for multi-process communication.
        
        Args:
            data_dir: Optional custom data directory path
            
        Returns:
            Path to the data directory
        """
        if data_dir is None:
            data_dir = Path.cwd() / "eicvibe_tracking_data" / self.name
        
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Tracking data directory: {self._data_dir}")
        return self._data_dir

    def start_background_tracking(self, particles: Any, parameters: TrackingParameters) -> str:
        """
        Start background tracking in a separate process for all simulation modes.
        
        All tracking operations (RING, LINAC, RAMPING) run in separate processes
        to ensure proper isolation and allow concurrent operations. Communication
        happens via queues and shared data files.
        
        PARAMETER VALIDATION: This method calls validate_tracking_parameters()
        before starting the tracking process.
        
        Args:
            particles: Particle distribution from create_particles()
            parameters: Tracking configuration (may be engine-specific subclass)
            
        Returns:
            Session identifier for managing the background simulation
            
        Raises:
            TrackingError: If background tracking cannot be started
            ValueError: If parameters are invalid or incompatible with engine
        """
        # Validate tracking parameters
        self.validate_tracking_parameters(parameters)
        
        # Validate engine is available
        if not self.available:
            raise TrackingError(f"Engine {self.name} is not available")
        
        # Setup data directory if needed
        if self._data_dir is None:
            self.setup_data_directory()
        
        # Create session
        session_id = str(uuid.uuid4())
        
        with self._session_lock:
            # Create communication queues
            command_queue = mp.Queue()
            response_queue = mp.Queue()
            
            # Create session
            session = TrackingSession(
                session_id=session_id,
                mode=parameters.mode,
                process=None  # Will be set after process creation
            )
            session.command_queue = command_queue
            session.response_queue = response_queue
            session.data_file = self._data_dir / f"session_{session_id}.pkl"
            
            # Create and start tracking process
            process = mp.Process(
                target=self._tracking_process_main,
                args=(session_id, particles, parameters, command_queue, response_queue, session.data_file),
                daemon=False
            )
            
            session.process = process
            session.status = TrackingStatus.RUNNING
            session.start_time = time.time()
            
            self._sessions[session_id] = session
        
        # Start the process
        process.start()
        
        self.logger.info(f"Started background tracking session: {session_id}")
        return session_id

    def get_tracking_status(self, session_id: str) -> Dict[str, Any]:
        """
        Get current status of background tracking simulation.
        
        Args:
            session_id: Session identifier from start_background_tracking()
            
        Returns:
            Dictionary with detailed status information
        """
        with self._session_lock:
            if session_id not in self._sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session = self._sessions[session_id]
        
        # Send status query to tracking process
        try:
            session.command_queue.put(StatusQuery(session_id), timeout=1.0)
            response = session.response_queue.get(timeout=5.0)
            
            if isinstance(response, StatusResponse):
                return {
                    'session_id': session_id,
                    'status': response.status,
                    'current_turn': response.current_turn,
                    'particles_alive': response.particles_alive,
                    'process_alive': session.process.is_alive() if session.process else False,
                    'start_time': session.start_time,
                    'metadata': response.metadata
                }
            else:
                return {
                    'session_id': session_id,
                    'status': TrackingStatus.ERROR,
                    'error': getattr(response, 'error', 'Unknown error'),
                    'process_alive': session.process.is_alive() if session.process else False
                }
                
        except queue.Empty:
            # Process may be unresponsive
            return {
                'session_id': session_id,
                'status': TrackingStatus.ERROR,
                'error': 'Process unresponsive',
                'process_alive': session.process.is_alive() if session.process else False
            }

    def get_monitor_data(self, session_id: str, request: MonitorDataRequest) -> Dict[str, Any]:
        """
        Retrieve monitor data from running or completed simulation.
        
        For RING mode: Returns turn-by-turn data from specified range
        For LINAC/RAMPING: Returns data from completed simulation runs
        
        Args:
            session_id: Session identifier 
            request: Data request specification
            
        Returns:
            Dictionary containing requested monitor data with structure:
            {
                'turn_numbers': array of turn indices,
                'monitor_data': {
                    'monitor_name': {
                        'x_mean': array, 'y_mean': array, 
                        'x_rms': array, 'y_rms': array, ...
                    }
                },
                'metadata': {'start_turn': int, 'end_turn': int, 'step': int}
            }
        """
        with self._session_lock:
            if session_id not in self._sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session = self._sessions[session_id]
        
        try:
            # Send data request to tracking process
            session.command_queue.put(DataRequest(session_id, request), timeout=1.0)
            response = session.response_queue.get(timeout=30.0)  # Longer timeout for data transfer
            
            if isinstance(response, DataResponse):
                return response.data
            else:
                raise TrackingError(f"Data request failed: {getattr(response, 'error', 'Unknown error')}")
                
        except queue.Empty:
            raise TrackingError("Data request timed out - process may be unresponsive")

    def schedule_lattice_change(self, session_id: str, change: LatticeChangeAction) -> bool:
        """
        Schedule a lattice parameter change during tracking.
        
        For RING mode: Change takes effect at specified turn number with optional ramping
        For LINAC/RAMPING: Change applies to next simulation run
        
        Args:
            session_id: Session identifier
            change: Lattice change specification with turn-based activation and ramping
            
        Returns:
            True if change was successfully scheduled
            
        Raises:
            ValueError: If change parameters are invalid
            TrackingError: If change cannot be scheduled
        """
        with self._session_lock:
            if session_id not in self._sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session = self._sessions[session_id]
        
        # Validate the change request
        self.validate_lattice_change(change, session_id)
        
        try:
            # Send change request to tracking process
            session.command_queue.put(LatticeChangeRequest(session_id, change), timeout=1.0)
            response = session.response_queue.get(timeout=5.0)
            
            if response.success:
                # Track the scheduled change
                session.scheduled_changes.append(change)
                self.logger.info(f"Scheduled lattice change: {change.element_name}.{change.parameter_group}.{change.parameter_name} = {change.new_value} at turn {change.activation_turn}")
                return True
            else:
                raise TrackingError(f"Failed to schedule change: {response.error}")
                
        except queue.Empty:
            raise TrackingError("Change request timed out - process may be unresponsive")

    def stop_tracking(self, session_id: str) -> TrackingResults:
        """
        Stop background tracking and return final results.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Final tracking results including all collected data
        """
        with self._session_lock:
            if session_id not in self._sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session = self._sessions[session_id]
        
        try:
            # Send stop command
            session.command_queue.put(ControlCommand(session_id, 'stop'), timeout=1.0)
            response = session.response_queue.get(timeout=30.0)  # Long timeout for final results
            
            # Wait for process to finish
            if session.process and session.process.is_alive():
                session.process.join(timeout=10.0)
                if session.process.is_alive():
                    self.logger.warning(f"Force terminating tracking process for session {session_id}")
                    session.process.terminate()
                    session.process.join(timeout=5.0)
            
            session.status = TrackingStatus.COMPLETED
            session.end_time = time.time()
            
            # Load final results
            if isinstance(response, TrackingResults):
                return response
            elif hasattr(response, 'data') and 'results' in response.data:
                return response.data['results']
            else:
                raise TrackingError(f"Invalid final results: {response}")
                
        except queue.Empty:
            raise TrackingError("Stop request timed out")
        finally:
            # Clean up session
            self._cleanup_session(session_id)

    def pause_tracking(self, session_id: str) -> bool:
        """
        Pause background tracking simulation.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if successfully paused
        """
        return self._send_control_command(session_id, 'pause')

    def resume_tracking(self, session_id: str) -> bool:
        """
        Resume paused tracking simulation.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if successfully resumed
        """
        return self._send_control_command(session_id, 'resume')

    def _send_control_command(self, session_id: str, command: str) -> bool:
        """Helper method to send control commands."""
        with self._session_lock:
            if session_id not in self._sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session = self._sessions[session_id]
        
        try:
            session.command_queue.put(ControlCommand(session_id, command), timeout=1.0)
            response = session.response_queue.get(timeout=5.0)
            
            if response.success:
                if command == 'pause':
                    session.status = TrackingStatus.PAUSED
                elif command == 'resume':
                    session.status = TrackingStatus.RUNNING
                
                self.logger.info(f"Successfully {command}ed tracking session: {session_id}")
                return True
            else:
                self.logger.error(f"Failed to {command} session {session_id}: {response.error}")
                return False
                
        except queue.Empty:
            self.logger.error(f"Control command '{command}' timed out for session {session_id}")
            return False

    def _cleanup_session(self, session_id: str):
        """Clean up session resources."""
        with self._session_lock:
            if session_id in self._sessions:
                session = self._sessions[session_id]
                
                # Clean up queues
                if session.command_queue:
                    try:
                        while not session.command_queue.empty():
                            session.command_queue.get_nowait()
                    except:
                        pass
                
                if session.response_queue:
                    try:
                        while not session.response_queue.empty():
                            session.response_queue.get_nowait()
                    except:
                        pass
                
                # Remove session
                del self._sessions[session_id]
                
                self.logger.info(f"Cleaned up session: {session_id}")
    
    def _process_lattice_change_request(
        self,
        request: LatticeChangeRequest,
        converted_lattice: Any,
        current_turn: int,
        active_changes: Optional[List[LatticeChangeAction]] = None
    ) -> TrackingResponse:
        """
        Process a lattice change request in the tracking process.
        
        This is a concrete helper method that can be used by engine implementations
        in their _tracking_process_main() method to handle lattice change requests
        in a standardized way.
        
        Args:
            request: LatticeChangeRequest message from command queue
            converted_lattice: Engine-specific lattice object
            current_turn: Current turn/step number in simulation
            active_changes: List to track active ramping changes (modified in-place)
            
        Returns:
            TrackingResponse with success status
            
        Example usage in engine's _tracking_process_main():
            if isinstance(message, LatticeChangeRequest):
                response = self._process_lattice_change_request(
                    message, converted_lattice, current_turn, active_changes
                )
                response_queue.put(response)
        """
        change = request.change
        
        try:
            # Validate the change
            self.validate_lattice_change(change, request.session_id)
            
            # Check if this is an immediate change or needs scheduling
            if change.activation_turn <= current_turn:
                # Apply change immediately
                success = self.apply_lattice_change(converted_lattice, change)
                
                if success:
                    self.logger.info(
                        f"Applied lattice change: {change.element_name}."
                        f"{change.parameter_group}.{change.parameter_name} = {change.new_value}"
                    )
                    return TrackingResponse(
                        session_id=request.session_id,
                        success=True,
                        message=f"Change applied immediately at turn {current_turn}"
                    )
                else:
                    return TrackingResponse(
                        session_id=request.session_id,
                        success=False,
                        error="Failed to apply lattice change"
                    )
            else:
                # Schedule for future turn
                if active_changes is not None:
                    active_changes.append(change)
                
                self.logger.info(
                    f"Scheduled lattice change: {change.element_name}."
                    f"{change.parameter_group}.{change.parameter_name} = {change.new_value} "
                    f"at turn {change.activation_turn}"
                )
                return TrackingResponse(
                    session_id=request.session_id,
                    success=True,
                    message=f"Change scheduled for turn {change.activation_turn}"
                )
                
        except Exception as e:
            self.logger.error(f"Error processing lattice change: {e}")
            return TrackingResponse(
                session_id=request.session_id,
                success=False,
                error=str(e)
            )
    
    def _apply_active_changes(
        self,
        converted_lattice: Any,
        current_turn: int,
        active_changes: List[LatticeChangeAction]
    ) -> int:
        """
        Apply all active ramping changes for the current turn.
        
        This helper method should be called each turn in the tracking loop
        to update parameters that are currently ramping.
        
        Args:
            converted_lattice: Engine-specific lattice object
            current_turn: Current turn/step number
            active_changes: List of active ramping changes
            
        Returns:
            Number of changes applied this turn
            
        Example usage in tracking loop:
            for turn in range(num_turns):
                # Apply any active ramping changes
                self._apply_active_changes(converted_lattice, turn, active_changes)
                
                # Track particles
                line.track(particles)
        """
        changes_applied = 0
        completed_changes = []
        
        for change in active_changes:
            # Calculate parameter value for this turn
            new_value = change.calculate_parameter_value(current_turn)
            
            if new_value is not None:
                # Create a temporary change object with the ramped value
                ramped_change = LatticeChangeAction(
                    element_name=change.element_name,
                    parameter_group=change.parameter_group,
                    parameter_name=change.parameter_name,
                    new_value=new_value,
                    activation_turn=current_turn,
                    old_value=change.old_value
                )
                
                try:
                    self.apply_lattice_change(converted_lattice, ramped_change)
                    changes_applied += 1
                    
                    # Check if ramping is complete
                    if (change.ramp_turns is not None and 
                        current_turn >= change.activation_turn + change.ramp_turns):
                        completed_changes.append(change)
                        self.logger.debug(
                            f"Completed ramping: {change.element_name}."
                            f"{change.parameter_group}.{change.parameter_name}"
                        )
                        
                except Exception as e:
                    self.logger.error(
                        f"Error applying ramping change at turn {current_turn}: {e}"
                    )
        
        # Remove completed changes
        for change in completed_changes:
            active_changes.remove(change)
        
        return changes_applied

    @abstractmethod
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
        Main function for the tracking process - runs all simulation modes in separate process.
        
        This method runs in a separate process and handles all simulation modes 
        (RING, LINAC, RAMPING) with appropriate behavior:
        
        - RING: Continuous tracking with periodic monitor data collection
        - LINAC: Start-to-end tracking with final results collection  
        - RAMPING: Time-dependent tracking with parameter evolution
        
        Common tasks for all modes:
        - Particle tracking simulation
        - Monitor data collection and buffering
        - Command processing (status, data requests, lattice changes)
        - Real-time parameter changes and ramping schedules
        
        Helper methods available for common tasks:
        - _process_lattice_change_request(): Handle LatticeChangeRequest messages
        - _apply_active_changes(): Apply ramping changes each turn
        - apply_lattice_change(): Engine-specific parameter update (abstract)
        
        Args:
            session_id: Session identifier
            particles: Initial particle distribution
            parameters: Tracking configuration (any mode)
            command_queue: Queue for receiving commands from main process
            response_queue: Queue for sending responses to main process
            data_file: File path for persistent data storage
            
        Example message processing:
            while tracking:
                if not command_queue.empty():
                    message = command_queue.get_nowait()
                    
                    if isinstance(message, LatticeChangeRequest):
                        response = self._process_lattice_change_request(
                            message, converted_lattice, current_turn, active_changes
                        )
                        response_queue.put(response)
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
    
    def add_callback(self, callback: SimulationCallback):
        """Add a simulation event callback."""
        self.callbacks.append(callback)
        self.logger.debug(f"Added callback: {type(callback).__name__}")
    
    def remove_callback(self, callback: SimulationCallback):
        """Remove a simulation event callback."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
            self.logger.debug(f"Removed callback: {type(callback).__name__}")
    
    def _notify_callbacks(self, event: str, *args, **kwargs):
        """Notify all registered callbacks of an event."""
        for callback in self.callbacks:
            try:
                if hasattr(callback, f'on_{event}'):
                    getattr(callback, f'on_{event}')(*args, **kwargs)
            except Exception as e:
                self.logger.warning(f"Callback {type(callback).__name__} error in on_{event}: {e}")
    
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

    # === Helper Methods for Advanced Simulation Control ===
    
    def validate_simulation_mode(self, mode: SimulationMode, operation: str = "general") -> bool:
        """
        Validate if a simulation mode supports a specific operation.
        
        Args:
            mode: Simulation mode to validate
            operation: Type of operation ('background_tracking', 'lattice_change', 'general')
            
        Returns:
            True if mode supports the operation
            
        Raises:
            ValueError: If mode doesn't support the operation
        """
        if operation == "background_tracking":
            # All simulation modes now support background tracking in separate processes
            pass  # No restriction needed
        
        if operation == "lattice_change" and mode not in [SimulationMode.RING, SimulationMode.RAMPING]:
            raise ValueError(f"Dynamic lattice changes supported for RING/RAMPING modes, got {mode}")
            
        if mode not in self.supported_modes:
            raise ValueError(f"Mode {mode} not supported by {self.name} engine")
        
        return True
    
    def validate_lattice_change(self, change: LatticeChangeAction, session_id: str) -> bool:
        """
        Validate a lattice change request.
        
        Args:
            change: Lattice change specification
            session_id: Session identifier
            
        Returns:
            True if change is valid
            
        Raises:
            ValueError: If change parameters are invalid
        """
        if change.activation_turn < 0:
            raise ValueError("Activation turn must be non-negative")
            
        if change.ramp_turns is not None and change.ramp_turns < 1:
            raise ValueError("Ramp turns must be positive if specified")
            
        # Additional engine-specific validation
        return self.validate_parameter_update(change)
    
    def validate_parameter_update(self, change: LatticeChangeAction) -> bool:
        """
        Validate engine-specific parameter update constraints.
        
        Override this method in subclasses to add engine-specific validation
        for parameter updates (e.g., checking parameter ranges, compatibility).
        
        Args:
            change: Lattice change specification
            
        Returns:
            True if parameter update is valid
            
        Raises:
            ValueError: If parameter update violates engine constraints
            
        Example:
            For XSuite, validate that magnetic field strengths are within
            hardware limits, or that RF frequencies are compatible with
            harmonic numbers.
        """
        # Base implementation allows all changes
        # Subclasses can override for specific constraints
        return True
    
    def validate_tracking_parameters(self, parameters: TrackingParameters) -> bool:
        """
        Validate tracking parameters are compatible with this engine.
        
        Override this method to validate engine-specific tracking parameters
        and check that requested features are supported by the engine.
        
        Args:
            parameters: Tracking parameters (may be engine-specific subclass)
            
        Returns:
            True if parameters are valid for this engine
            
        Raises:
            ValueError: If parameters request unsupported features
            TypeError: If parameter type is incompatible with engine
            
        Example:
            def validate_tracking_parameters(self, parameters):
                # Check if requesting GPU but engine doesn't support it
                if hasattr(parameters, 'use_gpu') and parameters.use_gpu:
                    if not self._gpu_available:
                        raise ValueError("GPU requested but not available")
                
                # Check collective effects support
                if parameters.enable_space_charge:
                    if 'space_charge' not in self.get_supported_features():
                        raise ValueError("Space charge not supported by this engine")
                
                return super().validate_tracking_parameters(parameters)
        """
        # Base validation: check mode compatibility
        if parameters.mode not in self.supported_modes:
            raise ValueError(
                f"Simulation mode {parameters.mode} not supported by {self.name}. "
                f"Supported modes: {[m.value for m in self.supported_modes]}"
            )
        
        # Validate mode-specific parameters
        parameters.validate_mode_parameters()
        
        # Check if collective effects are enabled but not supported
        supported_features = self.get_supported_features()
        
        if parameters.enable_beam_beam and not supported_features.get('beam_beam', False):
            self.logger.warning(f"Beam-beam interactions requested but not supported by {self.name}")
        
        if parameters.enable_wakefields and not supported_features.get('wakefields', False):
            self.logger.warning(f"Wakefields requested but not supported by {self.name}")
        
        if parameters.enable_space_charge and not supported_features.get('space_charge', False):
            self.logger.warning(f"Space charge requested but not supported by {self.name}")
        
        if parameters.enable_synchrotron_radiation and not supported_features.get('synchrotron_radiation', False):
            self.logger.warning(f"Synchrotron radiation requested but not supported by {self.name}")
        
        return True
    
    def get_supported_features(self) -> Dict[str, bool]:
        """
        Get dictionary of features supported by this engine.
        
        Override in engine implementations to specify which physics features
        and capabilities are supported.
        
        Returns:
            Dictionary mapping feature names to support status
            
        Example:
            def get_supported_features(self):
                return {
                    'beam_beam': True,
                    'wakefields': True,
                    'space_charge': True,
                    'synchrotron_radiation': True,
                    'gpu_acceleration': self._gpu_available,
                    'backtracker': True,
                }
        """
        # Base implementation: no collective effects
        return {
            'beam_beam': False,
            'wakefields': False,
            'space_charge': False,
            'synchrotron_radiation': False,
        }
    
    def create_monitor_data_request(
        self, 
        start_turn: int = 0, 
        end_turn: Optional[int] = None,
        monitor_names: Optional[List[str]] = None,
        data_types: Optional[List[str]] = None,
        turn_step: int = 1
    ) -> MonitorDataRequest:
        """
        Helper method to create monitor data requests.
        
        Args:
            start_turn: Starting turn for data collection
            end_turn: Ending turn (None for current)
            monitor_names: Specific monitors to collect from
            data_types: Types of data to collect
            turn_step: Step size for data collection
            
        Returns:
            Validated MonitorDataRequest object
        """
        if data_types is None:
            data_types = ["position", "size"]
            
        return MonitorDataRequest(
            start_turn=start_turn,
            end_turn=end_turn,
            monitor_names=monitor_names,
            data_types=data_types,
            turn_step=turn_step
        )
    
    def create_lattice_change(
        self, 
        element_name: str, 
        parameter_group: str, 
        parameter_name: str,
        new_value: float, 
        activation_turn: int,
        ramp_turns: Optional[int] = None,
        description: Optional[str] = None
    ) -> LatticeChangeAction:
        """
        Helper method to create lattice change actions.
        
        Args:
            element_name: Name of element to modify
            parameter_group: Parameter group name
            parameter_name: Parameter name within group
            new_value: New parameter value
            activation_turn: Turn when change takes effect
            ramp_turns: Number of turns to ramp (optional)
            description: Optional description
            
        Returns:
            Validated LatticeChangeAction object
        """
        return LatticeChangeAction(
            element_name=element_name,
            parameter_group=parameter_group,
            parameter_name=parameter_name,
            new_value=new_value,
            activation_turn=activation_turn,
            ramp_turns=ramp_turns,
            description=description
        )

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
            
            self._notify_callbacks('simulation_start', parameters)
            
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
            
            self._notify_callbacks('simulation_complete', results)
            
            return results
            
        except Exception as e:
            self._notify_callbacks('error', e)
            raise TrackingError(f"Simulation failed: {e}") from e
    
    def run_ring_simulation(
        self, 
        lattice: Lattice, 
        distribution: ParticleDistribution, 
        parameters: TrackingParameters
    ) -> str:
        """
        Start a ring simulation with background tracking support.
        
        This method is designed for RING mode simulations that need:
        - Long-running background tracking
        - Real-time monitor data access
        - Dynamic lattice parameter changes
        
        Args:
            lattice: EICViBE lattice definition
            distribution: Initial particle distribution
            parameters: Tracking parameters (must be RING mode)
            
        Returns:
            Session identifier for managing the simulation
            
        Raises:
            ValueError: If mode is not RING or parameters are invalid
            TrackingError: If simulation cannot be started
        """
        # Validate inputs for background tracking
        self.validate_simulation_mode(parameters.mode, "background_tracking")
        self.validate_lattice(lattice, parameters.mode)
        self.validate_parameters(parameters)
        
        # Get converted lattice and create particles
        converted_lattice = self.get_converted_lattice(lattice, parameters.mode)
        particles = self.create_particles(distribution)
        
        # Start background tracking
        session_id = self.start_background_tracking(particles, parameters)
        self.logger.info(f"Started ring simulation session: {session_id}")
        
        return session_id
    
    def run_repeated_simulations(
        self, 
        lattice: Lattice, 
        distribution: ParticleDistribution, 
        parameters: TrackingParameters,
        lattice_updates: Optional[List[Callable[[Lattice], Lattice]]] = None,
        num_repetitions: int = 1
    ) -> List[TrackingResults]:
        """
        Run repeated simulations with optional lattice updates between runs.
        
        This method is designed for LINAC and RAMPING modes where multiple
        simulation runs are needed with lattice modifications between runs.
        
        Args:
            lattice: Initial EICViBE lattice definition
            distribution: Particle distribution (same for all runs)
            parameters: Tracking parameters (LINAC or RAMPING mode)
            lattice_updates: Optional list of lattice modification functions
            num_repetitions: Number of simulation runs
            
        Returns:
            List of TrackingResults from each simulation run
            
        Raises:
            ValueError: If mode is RING (use run_ring_simulation instead)
            TrackingError: If any simulation fails
        """
        if parameters.mode == SimulationMode.RING:
            raise ValueError("Use run_ring_simulation() for RING mode simulations")
        
        # Validate inputs
        self.validate_lattice(lattice, parameters.mode)
        self.validate_parameters(parameters)
        
        results = []
        current_lattice = lattice
        
        for i in range(num_repetitions):
            self.logger.info(f"Starting simulation run {i+1}/{num_repetitions}")
            
            # Apply lattice update if provided
            if lattice_updates and i < len(lattice_updates):
                current_lattice = lattice_updates[i](current_lattice)
                # Clear cached converted lattice after modification
                self.clear_converted_lattice()
            
            # Run single simulation with full validation and monitoring
            result = self.run_simulation(current_lattice, distribution, parameters)
            results.append(result)
            
            self.logger.info(f"Completed run {i+1}: transmission={result.transmission:.3f}")
        
        self.logger.info(f"Completed all {num_repetitions} simulation runs")
        return results

    def __str__(self) -> str:
        """String representation of the engine."""
        status = "available" if self.available else "unavailable"
        modes = ", ".join([mode.value for mode in self.supported_modes])
        return f"{self.name} engine ({status}, modes: {modes})"
    
    def __repr__(self) -> str:
        """Detailed representation of the engine."""
        return f"{self.__class__.__name__}(name='{self.name}', available={self.available})"
