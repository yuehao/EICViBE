"""
EICViBE Simulation Engines Package.

This package provides a comprehensive framework for accelerator physics
simulation with support for multiple simulation engines and standardized
interfaces for lattice conversion, particle tracking, and beam analysis.

Key Components:
- BaseSimulationEngine: Abstract base class for all engines
- SimulationEngineRegistry: Central registry for engine management
- Type definitions and validation models for simulation parameters
- Monitoring and callback systems for real-time simulation updates

Supported Engines:
- XSuite: High-performance tracking with GPU support
- (Future: MAD-X, Elegant, BMAD, PyORBIT)

Example Usage:
    from eicvibe.simulators import SimulationEngineRegistry, TrackingParameters
    
    # List available engines
    engines = SimulationEngineRegistry.available_engines()
    
    # Create an engine
    engine = SimulationEngineRegistry.create_engine('xsuite')
    
    # Run simulation
    results = engine.run_simulation(lattice, distribution, parameters)
"""

from .types import (
    # Enums
    SimulationMode,
    DistributionType,
    
    # Data models
    TrackingParameters,
    ParticleDistribution,
    BeamPositionData,
    TwissData,
    TrackingResults,
    EngineConfiguration,
    
    # Exceptions
    SimulationError,
    EngineNotFoundError,
    LatticeConversionError,
    TrackingError,
    ConfigurationError
)

from .base import (
    BaseSimulationEngine,
    SimulationMonitor,
    ProgressMonitor
)

from .registry import (
    SimulationEngineRegistry,
    EngineFactory
)

# Version info
__version__ = "0.1.0"
__author__ = "EICViBE Development Team"

# Convenience aliases
Registry = SimulationEngineRegistry
Factory = EngineFactory

# Public API
__all__ = [
    # Core classes
    "BaseSimulationEngine",
    "SimulationEngineRegistry", 
    "EngineFactory",
    "Registry",  # Alias
    "Factory",   # Alias
    
    # Monitoring
    "SimulationMonitor",
    "ProgressMonitor",
    
    # Types and enums
    "SimulationMode",
    "DistributionType",
    
    # Data models
    "TrackingParameters",
    "ParticleDistribution", 
    "BeamPositionData",
    "TwissData",
    "TrackingResults",
    "EngineConfiguration",
    
    # Exceptions
    "SimulationError",
    "EngineNotFoundError",
    "LatticeConversionError",
    "TrackingError",
    "ConfigurationError",
    
    # Version
    "__version__"
]

# Initialize logging for the package
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())