"""
EICViBE Simulation Engine Interfaces.

This module provides continuous simulation services that integrate with external
simulation codes like XSuite and JuTrack.
"""

from .base import (
    BaseSimulationEngine,
    SimulationMode,
    SimulationState,
    BeamStatistics,
    MonitorData,
    RampingPlan,
    CircularBuffer
)

from .xsuite_interface import XSuiteSimulationEngine, create_xsuite_engine

# Compatibility aliases for existing tests
XSuiteSimulator = XSuiteSimulationEngine  # Alias for backward compatibility
SimulatorManager = XSuiteSimulationEngine  # Alias for backward compatibility

try:
    # JuTrack interface not updated yet for new architecture
    # from .jutrack_interface import JuTrackSimulator, create_jutrack_simulator
    JUTRACK_AVAILABLE = False
except ImportError:
    # JuTrack dependencies not available
    JuTrackSimulator = None
    create_jutrack_simulator = None
    JUTRACK_AVAILABLE = False


__all__ = [
    'BaseSimulationEngine',
    'SimulationMode',
    'SimulationState',
    'BeamStatistics',
    'MonitorData',
    'RampingPlan',
    'CircularBuffer',
    'XSuiteSimulationEngine',
    'create_xsuite_engine',
    'XSuiteSimulator',  # Compatibility alias
    'SimulatorManager'  # Compatibility alias
]

# Add JuTrack exports if available
if JUTRACK_AVAILABLE:
    __all__.extend(['JuTrackSimulator', 'create_jutrack_simulator'])
