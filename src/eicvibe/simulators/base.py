"""
Base classes for physics simulation engines in EICViBE.

This module provides the foundational architecture for all physics simulation
engines (XSuite, JuTrack, etc.) with support for three operation modes:
- LINAC: Single-pass simulation with automatic re-run capability
- RING: Continuous multi-turn simulation with real-time data access
- RAMPING: Time-dependent parameter evolution during simulation
"""

import logging
import threading
import time
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from collections import deque
from dataclasses import dataclass
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)


class SimulationMode(Enum):
    """Simulation operation modes."""
    LINAC = "linac"
    RING = "ring" 
    RAMPING = "ramping"


class SimulationState(Enum):
    """Simulation execution states."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class BeamStatistics:
    """Container for beam statistics at monitoring points."""
    turn: int
    timestamp: float
    x_mean: float
    y_mean: float
    x_rms: float
    y_rms: float
    x_emittance: float
    y_emittance: float
    particles_alive: int
    survival_rate: float
    energy_mean: float
    energy_spread: float


@dataclass
class MonitorData:
    """Data structure for beam position monitor readings."""
    monitor_name: str
    location_s: float  # longitudinal position in lattice
    beam_stats: BeamStatistics
    raw_data: Optional[Dict[str, np.ndarray]] = None  # particle coordinates if available


@dataclass
class RampingPlan:
    """Parameter ramping plan for time-dependent simulations."""
    element_name: str
    parameter_group: str
    parameter_name: str
    time_points: List[float]  # time values
    parameter_values: List[float]  # corresponding parameter values
    interpolation_type: str = "linear"  # linear, cubic, step


class CircularBuffer:
    """Thread-safe circular buffer for real-time data storage."""
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.RLock()
    
    def append(self, item: Any) -> None:
        """Add item to buffer."""
        with self.lock:
            self.buffer.append(item)
    
    def get_latest(self, n: int = None) -> List[Any]:
        """Get latest n items (or all if n is None)."""
        with self.lock:
            if n is None:
                return list(self.buffer)
            return list(self.buffer)[-n:]
    
    def get_all(self) -> List[Any]:
        """Get all items in chronological order."""
        with self.lock:
            return list(self.buffer)
    
    def clear(self) -> None:
        """Clear all items."""
        with self.lock:
            self.buffer.clear()
    
    def size(self) -> int:
        """Get current buffer size."""
        with self.lock:
            return len(self.buffer)


class BaseSimulationEngine(ABC):
    """
    Abstract base class for all physics simulation engines.
    
    Provides common infrastructure for LINAC, RING, and RAMPING modes
    with real-time monitoring and parameter control capabilities.
    """
    
    def __init__(self, engine_name: str):
        self.engine_name = engine_name
        self.logger = logging.getLogger(f"{__name__}.{engine_name}")
        
        # Simulation state
        self.mode = SimulationMode.RING  # Default mode, will be set in setup_simulation
        self.state = SimulationState.IDLE
        self.current_turn = 0
        self.current_time = 0.0
        self.start_time = 0.0
        
        # Threading for background execution
        self.simulation_thread = None
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        
        # Lattice and simulation objects
        self.lattice = None
        self.physics_lattice = None  # engine-specific lattice object
        self.particles = None
        self.simulation_params = {}
        
        # Monitoring infrastructure
        self.monitors = {}  # {monitor_name: monitor_info}
        self.monitor_buffers = {}  # {monitor_name: CircularBuffer}
        self.buffer_size = 1024  # default buffer size
        
        # Ramping infrastructure
        self.ramping_plans = {}  # {element_name: {param_group: {param_name: RampingPlan}}}
        
        # LINAC mode specific
        self.linac_auto_rerun = True
        self.linac_lattice_changed = False
        self.linac_results = []
        
        # Callbacks
        self.turn_callback = None
        self.monitor_callback = None
        self.state_change_callback = None
        
        # Performance monitoring
        self.performance_stats = {
            'turns_per_second': 0.0,
            'monitor_access_time': 0.0,
            'parameter_update_time': 0.0
        }
    
    @abstractmethod
    def initialize_engine(self) -> bool:
        """Initialize the specific physics engine (XSuite, JuTrack, etc.)."""
        pass
    
    @abstractmethod
    def convert_lattice(self, eicvibe_lattice, branch_name: str = "main"):
        """Convert EICViBE lattice to engine-specific format."""
        pass
    
    @abstractmethod
    def create_particles(self, particle_params: Dict[str, Any]) -> Any:
        """Create initial particle distribution."""
        pass
    
    @abstractmethod
    def track_single_turn(self) -> bool:
        """Execute one turn of tracking. Returns True if successful."""
        pass
    
    @abstractmethod
    def get_particle_coordinates(self) -> Dict[str, np.ndarray]:
        """Get current particle coordinates (x, px, y, py, s, delta)."""
        pass
    
    @abstractmethod
    def update_element_parameter(self, element_name: str, param_group: str, 
                                param_name: str, value: float) -> bool:
        """Update element parameter in the physics lattice."""
        pass
    
    @abstractmethod
    def get_element_parameter(self, element_name: str, param_group: str, 
                             param_name: str) -> float:
        """Get current element parameter value."""
        pass
    
    @abstractmethod
    def cleanup_engine(self) -> None:
        """Clean up engine resources."""
        pass
    
    # Public API methods
    
    def setup_simulation(self, eicvibe_lattice, simulation_params: Dict[str, Any], 
                        branch_name: str = "main") -> bool:
        """
        Setup simulation with EICViBE lattice and parameters.
        
        Args:
            eicvibe_lattice: EICViBE Lattice object
            simulation_params: Simulation configuration
            branch_name: Branch name to use from lattice
            
        Returns:
            True if setup successful
        """
        try:
            self.logger.info(f"Setting up {self.engine_name} simulation")
            
            # Initialize engine
            if not self.initialize_engine():
                return False
            
            # Store configuration
            self.lattice = eicvibe_lattice
            self.simulation_params = simulation_params.copy()
            
            # Convert lattice
            self.physics_lattice = self.convert_lattice(eicvibe_lattice, branch_name)
            if self.physics_lattice is None:
                return False
            
            # Create particles
            particle_params = simulation_params.get('particle_params', {})
            self.particles = self.create_particles(particle_params)
            if self.particles is None:
                return False
            
            # Setup monitoring
            self._setup_monitors()
            
            # Setup ramping if specified
            if 'ramping_plans' in simulation_params:
                self._setup_ramping(simulation_params['ramping_plans'])
            
            # Configure mode-specific settings
            self.mode = SimulationMode(simulation_params.get('mode', 'ring'))
            self.buffer_size = simulation_params.get('buffer_size', 1024)
            self.linac_auto_rerun = simulation_params.get('linac_auto_rerun', True)
            
            self.state = SimulationState.IDLE
            self.logger.info(f"Simulation setup complete for {self.mode.value} mode")
            return True
            
        except Exception as e:
            self.logger.error(f"Simulation setup failed: {e}")
            self.state = SimulationState.ERROR
            return False
    
    def start_simulation(self, **kwargs) -> bool:
        """Start simulation in background thread."""
        if self.state != SimulationState.IDLE:
            self.logger.warning(f"Cannot start simulation in state: {self.state}")
            return False
        
        try:
            # Reset control events
            self.stop_event.clear()
            self.pause_event.clear()
            
            # Store start time
            self.start_time = time.time()
            self.current_time = 0.0
            self.current_turn = 0
            
            # Store callbacks
            self.turn_callback = kwargs.get('turn_callback')
            self.monitor_callback = kwargs.get('monitor_callback')
            self.state_change_callback = kwargs.get('state_change_callback')
            
            # Start simulation thread
            self.simulation_thread = threading.Thread(
                target=self._simulation_loop,
                args=(kwargs,),
                daemon=True
            )
            self.simulation_thread.start()
            
            self.state = SimulationState.RUNNING
            self._notify_state_change()
            self.logger.info(f"Started {self.mode.value} simulation")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start simulation: {e}")
            self.state = SimulationState.ERROR
            return False
    
    def stop_simulation(self) -> bool:
        """Stop running simulation."""
        if self.state not in [SimulationState.RUNNING, SimulationState.PAUSED]:
            return True
        
        self.stop_event.set()
        if self.simulation_thread and self.simulation_thread.is_alive():
            self.simulation_thread.join(timeout=5.0)
        
        self.state = SimulationState.STOPPED
        self._notify_state_change()
        self.logger.info("Simulation stopped")
        return True
    
    def pause_simulation(self) -> bool:
        """Pause running simulation."""
        if self.state != SimulationState.RUNNING:
            return False
        
        self.pause_event.set()
        self.state = SimulationState.PAUSED
        self._notify_state_change()
        return True
    
    def resume_simulation(self) -> bool:
        """Resume paused simulation."""
        if self.state != SimulationState.PAUSED:
            return False
        
        self.pause_event.clear()
        self.state = SimulationState.RUNNING
        self._notify_state_change()
        return True
    
    def get_monitor_data(self, monitor_name: str = None, last_n_turns: int = None) -> Dict[str, List[MonitorData]]:
        """
        Get monitor data from buffers.
        
        Args:
            monitor_name: Specific monitor name (None for all)
            last_n_turns: Number of recent turns to retrieve (None for all)
            
        Returns:
            Dictionary of monitor data
        """
        if monitor_name:
            monitors_to_get = [monitor_name] if monitor_name in self.monitor_buffers else []
        else:
            monitors_to_get = list(self.monitor_buffers.keys())
        
        result = {}
        for mon_name in monitors_to_get:
            buffer = self.monitor_buffers[mon_name]
            data = buffer.get_latest(last_n_turns)
            result[mon_name] = data
        
        return result
    
    def get_latest_monitor_reading(self, monitor_name: str = None) -> Dict[str, MonitorData]:
        """Get latest reading from monitors."""
        data = self.get_monitor_data(monitor_name, last_n_turns=1)
        result = {}
        for mon_name, readings in data.items():
            if readings:
                result[mon_name] = readings[-1]
        return result
    
    def update_parameter_during_simulation(self, element_name: str, param_group: str,
                                         param_name: str, new_value: float,
                                         transition_turns: int = 0) -> bool:
        """
        Update element parameter during simulation.
        
        Args:
            element_name: Name of element to update
            param_group: Parameter group
            param_name: Parameter name
            new_value: New parameter value
            transition_turns: Number of turns for gradual transition (0 for immediate)
            
        Returns:
            True if update successful
        """
        try:
            if transition_turns <= 0:
                # Immediate update
                return self.update_element_parameter(element_name, param_group, param_name, new_value)
            else:
                # Gradual transition - create temporary ramping plan
                current_value = self.get_element_parameter(element_name, param_group, param_name)
                transition_time = transition_turns  # use turns as time units for simplicity
                
                ramp_plan = RampingPlan(
                    element_name=element_name,
                    parameter_group=param_group,
                    parameter_name=param_name,
                    time_points=[self.current_time, self.current_time + transition_time],
                    parameter_values=[current_value, new_value],
                    interpolation_type="linear"
                )
                
                # Add to ramping plans
                if element_name not in self.ramping_plans:
                    self.ramping_plans[element_name] = {}
                if param_group not in self.ramping_plans[element_name]:
                    self.ramping_plans[element_name][param_group] = {}
                
                self.ramping_plans[element_name][param_group][param_name] = ramp_plan
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to update parameter: {e}")
            return False
    
    def signal_lattice_change(self) -> None:
        """Signal that lattice has changed (for LINAC mode)."""
        if self.mode == SimulationMode.LINAC:
            self.linac_lattice_changed = True
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get simulation performance statistics."""
        return self.performance_stats.copy()
    
    @property 
    def simulation_status(self) -> Dict[str, Any]:
        """Get comprehensive simulation status."""
        return {
            'engine': self.engine_name,
            'mode': self.mode.value,
            'state': self.state.value,
            'current_turn': self.current_turn,
            'current_time': self.current_time,
            'elapsed_time': time.time() - self.start_time if self.start_time > 0 else 0,
            'monitors_active': len(self.monitor_buffers),
            'ramping_plans_active': len(self.ramping_plans),
            'performance': self.performance_stats
        }
    
    # Internal methods
    
    def _setup_monitors(self) -> None:
        """Setup monitoring infrastructure based on lattice."""
        # Find all monitors in lattice
        elements = self.lattice.expand_lattice()
        monitor_elements = [elem for elem in elements if elem.type.lower() in ['monitor', 'bpm']]
        
        self.monitors.clear()
        self.monitor_buffers.clear()
        
        s_position = 0.0
        for elem in elements:
            if elem.type.lower() in ['monitor', 'bpm']:
                self.monitors[elem.name] = {
                    'element': elem,
                    'location_s': s_position,
                    'active': True
                }
                self.monitor_buffers[elem.name] = CircularBuffer(self.buffer_size)
            s_position += elem.length
        
        self.logger.info(f"Setup {len(self.monitors)} monitors")
    
    def _setup_ramping(self, ramping_plans: List[Dict[str, Any]]) -> None:
        """Setup parameter ramping plans."""
        self.ramping_plans.clear()
        
        for plan_dict in ramping_plans:
            plan = RampingPlan(**plan_dict)
            
            if plan.element_name not in self.ramping_plans:
                self.ramping_plans[plan.element_name] = {}
            if plan.parameter_group not in self.ramping_plans[plan.element_name]:
                self.ramping_plans[plan.element_name][plan.parameter_group] = {}
            
            self.ramping_plans[plan.element_name][plan.parameter_group][plan.parameter_name] = plan
        
        self.logger.info(f"Setup {len(ramping_plans)} ramping plans")
    
    def _simulation_loop(self, kwargs: Dict[str, Any]) -> None:
        """Main simulation loop (runs in background thread)."""
        try:
            if self.mode == SimulationMode.LINAC:
                self._run_linac_mode(kwargs)
            elif self.mode == SimulationMode.RING:
                self._run_ring_mode(kwargs)
            elif self.mode == SimulationMode.RAMPING:
                self._run_ramping_mode(kwargs)
            else:
                raise ValueError(f"Unknown simulation mode: {self.mode}")
                
        except Exception as e:
            self.logger.error(f"Simulation loop error: {e}")
            self.state = SimulationState.ERROR
        finally:
            self._notify_state_change()
    
    def _run_linac_mode(self, kwargs: Dict[str, Any]) -> None:
        """Execute LINAC mode simulation."""
        self.logger.info("Starting LINAC mode simulation")
        
        while not self.stop_event.is_set():
            # Wait if paused
            while self.pause_event.is_set() and not self.stop_event.is_set():
                time.sleep(0.1)
            
            if self.stop_event.is_set():
                break
            
            # Single pass through linac
            turn_start_time = time.time()
            
            # Track through entire lattice
            success = self.track_single_turn()
            if not success:
                self.logger.error("Tracking failed in LINAC mode")
                self.state = SimulationState.ERROR
                break
            
            # Collect monitor data
            monitor_data = self._collect_monitor_data()
            
            # Store results
            self.linac_results.append({
                'turn': self.current_turn,
                'time': self.current_time,
                'monitor_data': monitor_data,
                'success': True
            })
            
            # Update performance stats
            turn_time = time.time() - turn_start_time
            self.performance_stats['turns_per_second'] = 1.0 / max(turn_time, 1e-6)
            
            # Call turn callback
            if self.turn_callback:
                self.turn_callback(self.current_turn, monitor_data)
            
            self.current_turn += 1
            self.current_time = time.time() - self.start_time
            
            # Check if auto-rerun is enabled and lattice hasn't changed
            if not (self.linac_auto_rerun and not self.linac_lattice_changed):
                break
            
            # Small delay before next run
            time.sleep(0.01)
        
        self.state = SimulationState.COMPLETED
        self.logger.info(f"LINAC simulation completed after {self.current_turn} runs")
    
    def _run_ring_mode(self, kwargs: Dict[str, Any]) -> None:
        """Execute RING mode simulation."""
        max_turns = kwargs.get('max_turns', float('inf'))
        self.logger.info(f"Starting RING mode simulation for {max_turns} turns")
        
        while self.current_turn < max_turns and not self.stop_event.is_set():
            # Wait if paused
            while self.pause_event.is_set() and not self.stop_event.is_set():
                time.sleep(0.1)
            
            if self.stop_event.is_set():
                break
            
            turn_start_time = time.time()
            
            # Update ramping parameters if needed
            self._update_ramping_parameters()
            
            # Track one turn
            success = self.track_single_turn()
            if not success:
                self.logger.error(f"Tracking failed at turn {self.current_turn}")
                self.state = SimulationState.ERROR
                break
            
            # Collect and store monitor data
            monitor_data = self._collect_monitor_data()
            for monitor_name, data in monitor_data.items():
                if monitor_name in self.monitor_buffers:
                    self.monitor_buffers[monitor_name].append(data)
            
            # Update performance stats
            turn_time = time.time() - turn_start_time
            self.performance_stats['turns_per_second'] = 1.0 / max(turn_time, 1e-6)
            
            # Call callbacks
            if self.turn_callback:
                self.turn_callback(self.current_turn, monitor_data)
            if self.monitor_callback:
                self.monitor_callback(monitor_data)
            
            self.current_turn += 1
            self.current_time = time.time() - self.start_time
        
        if self.current_turn >= max_turns:
            self.state = SimulationState.COMPLETED
        else:
            self.state = SimulationState.STOPPED
        
        self.logger.info(f"RING simulation finished at turn {self.current_turn}")
    
    def _run_ramping_mode(self, kwargs: Dict[str, Any]) -> None:
        """Execute RAMPING mode simulation."""
        max_turns = kwargs.get('max_turns', float('inf'))
        self.logger.info(f"Starting RAMPING mode simulation for {max_turns} turns")
        
        # Similar to RING mode but with mandatory parameter updates
        while self.current_turn < max_turns and not self.stop_event.is_set():
            # Wait if paused
            while self.pause_event.is_set() and not self.stop_event.is_set():
                time.sleep(0.1)
            
            if self.stop_event.is_set():
                break
            
            turn_start_time = time.time()
            
            # Update ramping parameters (mandatory in ramping mode)
            param_update_start = time.time()
            self._update_ramping_parameters()
            self.performance_stats['parameter_update_time'] = time.time() - param_update_start
            
            # Track one turn
            success = self.track_single_turn()
            if not success:
                self.logger.error(f"Tracking failed at turn {self.current_turn}")
                self.state = SimulationState.ERROR
                break
            
            # Collect and store monitor data
            monitor_data = self._collect_monitor_data()
            for monitor_name, data in monitor_data.items():
                if monitor_name in self.monitor_buffers:
                    self.monitor_buffers[monitor_name].append(data)
            
            # Update performance stats
            turn_time = time.time() - turn_start_time
            self.performance_stats['turns_per_second'] = 1.0 / max(turn_time, 1e-6)
            
            # Call callbacks
            if self.turn_callback:
                self.turn_callback(self.current_turn, monitor_data)
            if self.monitor_callback:
                self.monitor_callback(monitor_data)
            
            self.current_turn += 1
            self.current_time = time.time() - self.start_time
        
        if self.current_turn >= max_turns:
            self.state = SimulationState.COMPLETED
        else:
            self.state = SimulationState.STOPPED
        
        self.logger.info(f"RAMPING simulation finished at turn {self.current_turn}")
    
    def _collect_monitor_data(self) -> Dict[str, MonitorData]:
        """Collect data from all active monitors."""
        monitor_access_start = time.time()
        
        # Get particle coordinates
        coords = self.get_particle_coordinates()
        
        monitor_data = {}
        for monitor_name, monitor_info in self.monitors.items():
            if not monitor_info['active']:
                continue
            
            # Calculate beam statistics
            beam_stats = self._calculate_beam_statistics(coords)
            
            # Create monitor data
            monitor_data[monitor_name] = MonitorData(
                monitor_name=monitor_name,
                location_s=monitor_info['location_s'],
                beam_stats=beam_stats,
                raw_data=coords if self.simulation_params.get('store_raw_data', False) else None
            )
        
        self.performance_stats['monitor_access_time'] = time.time() - monitor_access_start
        return monitor_data
    
    def _calculate_beam_statistics(self, coords: Dict[str, np.ndarray]) -> BeamStatistics:
        """Calculate beam statistics from particle coordinates."""
        # Filter living particles
        if 'state' in coords:
            alive_mask = coords['state'] > 0
        else:
            # Assume all particles are alive if no state info
            alive_mask = np.ones(len(coords['x']), dtype=bool)
        
        n_alive = np.sum(alive_mask)
        survival_rate = n_alive / len(alive_mask) if len(alive_mask) > 0 else 0.0
        
        if n_alive == 0:
            # All particles lost
            return BeamStatistics(
                turn=self.current_turn,
                timestamp=time.time(),
                x_mean=0.0, y_mean=0.0,
                x_rms=0.0, y_rms=0.0,
                x_emittance=0.0, y_emittance=0.0,
                particles_alive=0,
                survival_rate=0.0,
                energy_mean=0.0,
                energy_spread=0.0
            )
        
        # Calculate statistics for living particles
        x = coords['x'][alive_mask]
        px = coords['px'][alive_mask] if 'px' in coords else np.zeros_like(x)
        y = coords['y'][alive_mask]
        py = coords['py'][alive_mask] if 'py' in coords else np.zeros_like(y)
        delta = coords['delta'][alive_mask] if 'delta' in coords else np.zeros_like(x)
        
        # Basic statistics
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        x_rms = np.std(x)
        y_rms = np.std(y)
        energy_mean = np.mean(delta)
        energy_spread = np.std(delta)
        
        # Emittance calculation (with safety check for numerical stability)
        if len(x) > 1:
            # Calculate the discriminant first
            x_discriminant = np.var(x) * np.var(px) - np.cov(x, px)[0,1]**2
            x_emittance = np.sqrt(max(0.0, x_discriminant))  # Ensure non-negative
        else:
            x_emittance = 0.0
            
        if len(y) > 1:
            # Calculate the discriminant first  
            y_discriminant = np.var(y) * np.var(py) - np.cov(y, py)[0,1]**2
            y_emittance = np.sqrt(max(0.0, y_discriminant))  # Ensure non-negative
        else:
            y_emittance = 0.0
        
        return BeamStatistics(
            turn=self.current_turn,
            timestamp=time.time(),
            x_mean=x_mean,
            y_mean=y_mean,
            x_rms=x_rms,
            y_rms=y_rms,
            x_emittance=x_emittance,
            y_emittance=y_emittance,
            particles_alive=n_alive,
            survival_rate=survival_rate,
            energy_mean=energy_mean,
            energy_spread=energy_spread
        )
    
    def _update_ramping_parameters(self) -> None:
        """Update element parameters according to ramping plans."""
        for element_name, param_groups in self.ramping_plans.items():
            for param_group, params in param_groups.items():
                for param_name, ramp_plan in params.items():
                    
                    # Interpolate value at current time
                    new_value = self._interpolate_ramp_value(ramp_plan, self.current_time)
                    
                    # Update parameter
                    self.update_element_parameter(element_name, param_group, param_name, new_value)
    
    def _interpolate_ramp_value(self, ramp_plan: RampingPlan, current_time: float) -> float:
        """Interpolate parameter value from ramping plan."""
        time_points = np.array(ramp_plan.time_points)
        values = np.array(ramp_plan.parameter_values)
        
        if current_time <= time_points[0]:
            return values[0]
        elif current_time >= time_points[-1]:
            return values[-1]
        else:
            if ramp_plan.interpolation_type == "linear":
                return np.interp(current_time, time_points, values)
            elif ramp_plan.interpolation_type == "step":
                idx = np.searchsorted(time_points, current_time, side='right') - 1
                return values[idx]
            else:
                # Default to linear
                return np.interp(current_time, time_points, values)
    
    def _notify_state_change(self) -> None:
        """Notify about state changes."""
        if self.state_change_callback:
            try:
                self.state_change_callback(self.state, self.simulation_status)
            except Exception as e:
                self.logger.error(f"State change callback error: {e}")
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.stop_simulation()
            self.cleanup_engine()
        except:
            pass
