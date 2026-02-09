"""
Simulation engine registry and factory for EICViBE.

This module provides a centralized registry for managing simulation engines,
including automatic discovery, registration, and factory methods for creating
engine instances with proper configuration validation.
"""

from typing import Dict, Type, List, Optional, Any
import logging
import importlib
from pathlib import Path

from .base import BaseSimulationEngine
from .types import (
    EngineConfiguration, 
    EngineNotFoundError, 
    ConfigurationError,
    SimulationMode
)

logger = logging.getLogger(__name__)


class SimulationEngineRegistry:
    """
    Registry for managing available simulation engines.
    
    This class maintains a central registry of simulation engines and provides
    factory methods for creating engine instances. It supports automatic
    discovery of engine implementations and validation of their capabilities.
    """
    
    _engines: Dict[str, Type[BaseSimulationEngine]] = {}
    _engine_metadata: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register(
        cls, 
        name: str, 
        engine_class: Type[BaseSimulationEngine], 
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Register a simulation engine implementation.
        
        Args:
            name: Unique name for the engine
            engine_class: Engine class that inherits from BaseSimulationEngine
            metadata: Optional metadata about the engine (version, description, etc.)
            
        Raises:
            ValueError: If engine name is already registered or class is invalid
        """
        # Validate engine class
        if not issubclass(engine_class, BaseSimulationEngine):
            raise ValueError(f"Engine class must inherit from BaseSimulationEngine")
        
        if name in cls._engines:
            logger.warning(f"Engine '{name}' is already registered, overwriting")
        
        cls._engines[name] = engine_class
        cls._engine_metadata[name] = metadata or {}
        
        logger.info(f"Registered engine: {name}")
        logger.debug(f"Engine class: {engine_class.__name__}")
        if metadata:
            logger.debug(f"Engine metadata: {metadata}")
    
    @classmethod
    def unregister(cls, name: str):
        """
        Unregister a simulation engine.
        
        Args:
            name: Name of the engine to unregister
        """
        if name in cls._engines:
            del cls._engines[name]
            if name in cls._engine_metadata:
                del cls._engine_metadata[name]
            logger.info(f"Unregistered engine: {name}")
        else:
            logger.warning(f"Engine '{name}' not found in registry")
    
    @classmethod
    def get_engine_class(cls, name: str) -> Type[BaseSimulationEngine]:
        """
        Get engine class by name.
        
        Args:
            name: Name of the engine
            
        Returns:
            Engine class
            
        Raises:
            EngineNotFoundError: If engine is not registered
        """
        if name not in cls._engines:
            available = list(cls._engines.keys())
            raise EngineNotFoundError(
                f"Engine '{name}' not registered. Available engines: {available}"
            )
        return cls._engines[name]
    
    @classmethod
    def create_engine(
        cls, 
        name: str, 
        config: Optional[EngineConfiguration] = None
    ) -> BaseSimulationEngine:
        """
        Create engine instance with configuration validation.
        
        Args:
            name: Name of the engine to create
            config: Optional configuration for the engine
            
        Returns:
            Configured engine instance
            
        Raises:
            EngineNotFoundError: If engine is not registered
            ConfigurationError: If configuration is invalid
        """
        engine_class = cls.get_engine_class(name)
        
        # Use default configuration if none provided
        if config is None:
            config = EngineConfiguration(name=name)
        elif config.name != name:
            config.name = name  # Ensure consistency
        
        try:
            engine = engine_class(config)
            logger.debug(f"Created {name} engine instance")
            return engine
        except Exception as e:
            raise ConfigurationError(f"Failed to create {name} engine: {e}") from e
    
    @classmethod
    def list_engines(cls) -> List[str]:
        """
        List all registered engine names.
        
        Returns:
            List of registered engine names
        """
        return list(cls._engines.keys())
    
    @classmethod
    def available_engines(cls) -> List[str]:
        """
        List engines that are currently available (dependencies installed).
        
        Returns:
            List of available engine names
        """
        available = []
        for name, engine_class in cls._engines.items():
            try:
                # Create temporary instance to check availability
                temp_config = EngineConfiguration(name=name)
                temp_engine = engine_class(temp_config)
                if temp_engine.available:
                    available.append(name)
            except Exception as e:
                logger.debug(f"Engine {name} not available: {e}")
                continue
        return available
    
    @classmethod
    def get_engine_info(cls, name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about registered engines.
        
        Args:
            name: Specific engine name, or None for all engines
            
        Returns:
            Dictionary with engine information
        """
        if name is not None:
            if name not in cls._engines:
                raise EngineNotFoundError(f"Engine '{name}' not registered")
            
            engine_class = cls._engines[name]
            metadata = cls._engine_metadata[name]
            
            # Try to get availability
            try:
                temp_config = EngineConfiguration(name=name)
                temp_engine = engine_class(temp_config)
                available = temp_engine.available
                supported_modes = [mode.value for mode in temp_engine.supported_modes]
            except Exception:
                available = False
                supported_modes = []
            
            return {
                'name': name,
                'class': engine_class.__name__,
                'available': available,
                'supported_modes': supported_modes,
                'metadata': metadata
            }
        else:
            # Return info for all engines
            all_info = {}
            for engine_name in cls._engines.keys():
                all_info[engine_name] = cls.get_engine_info(engine_name)
            return all_info
    
    @classmethod
    def engines_for_mode(cls, mode: SimulationMode) -> List[str]:
        """
        Get list of engines that support a specific simulation mode.
        
        Args:
            mode: Simulation mode to check
            
        Returns:
            List of engine names that support the mode
        """
        supporting_engines = []
        for name in cls.available_engines():
            try:
                engine_info = cls.get_engine_info(name)
                if mode.value in engine_info['supported_modes']:
                    supporting_engines.append(name)
            except Exception:
                continue
        return supporting_engines
    
    @classmethod
    def auto_discover_engines(cls, package_path: Optional[str] = None):
        """
        Automatically discover and register engine implementations.
        
        This method scans for engine implementations in the simulators package
        and attempts to register them automatically.
        
        Args:
            package_path: Optional path to search for engines
        """
        if package_path is None:
            # Use current package
            package_path = str(Path(__file__).parent)
        
        logger.info(f"Auto-discovering engines in {package_path}")
        
        # Look for Python files that might contain engines
        search_path = Path(package_path)
        for py_file in search_path.glob("*_interface.py"):
            module_name = py_file.stem
            try:
                # Import the module
                full_module_name = f"eicvibe.simulators.{module_name}"
                module = importlib.import_module(full_module_name)
                
                # Look for engine classes
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and 
                        issubclass(attr, BaseSimulationEngine) and 
                        attr != BaseSimulationEngine):
                        
                        # Auto-register with default name
                        engine_name = attr_name.lower().replace('engine', '').replace('simulation', '')
                        if not engine_name:
                            engine_name = module_name.replace('_interface', '')
                        
                        cls.register(
                            engine_name, 
                            attr, 
                            {'auto_discovered': True, 'module': full_module_name}
                        )
                        
            except Exception as e:
                logger.debug(f"Could not import {module_name}: {e}")
                continue
    
    @classmethod
    def clear_registry(cls):
        """Clear all registered engines (mainly for testing)."""
        cls._engines.clear()
        cls._engine_metadata.clear()
        logger.info("Cleared engine registry")


class EngineFactory:
    """
    Factory class for creating simulation engines with smart defaults.
    
    This factory provides convenience methods for creating engines with
    common configurations and automatic fallback options.
    """
    
    @staticmethod
    def create_best_engine(
        mode: SimulationMode,
        prefer_gpu: bool = False,
        config: Optional[EngineConfiguration] = None
    ) -> BaseSimulationEngine:
        """
        Create the best available engine for a given simulation mode.
        
        Args:
            mode: Simulation mode required
            prefer_gpu: Whether to prefer GPU-capable engines
            config: Optional base configuration
            
        Returns:
            Best available engine instance
            
        Raises:
            EngineNotFoundError: If no suitable engine is available
        """
        available_engines = SimulationEngineRegistry.engines_for_mode(mode)
        
        if not available_engines:
            raise EngineNotFoundError(f"No engines available for {mode} mode")
        
        # Simple preference order (can be made more sophisticated)
        engine_preferences = ['xsuite', 'madx', 'elegant']  # XSuite first
        
        selected_engine = None
        for preferred in engine_preferences:
            if preferred in available_engines:
                selected_engine = preferred
                break
        
        # If no preferred engine found, use first available
        if selected_engine is None:
            selected_engine = available_engines[0]
        
        # Create configuration if needed
        if config is None:
            config = EngineConfiguration(
                name=selected_engine,
                enable_gpu=prefer_gpu,
                context="cuda" if prefer_gpu else "cpu"
            )
        
        logger.info(f"Selected {selected_engine} engine for {mode} mode")
        return SimulationEngineRegistry.create_engine(selected_engine, config)
    
    @staticmethod
    def create_engine_with_fallback(
        preferred_engine: str,
        fallback_engines: List[str],
        config: Optional[EngineConfiguration] = None
    ) -> BaseSimulationEngine:
        """
        Create engine with fallback options if preferred engine is unavailable.
        
        Args:
            preferred_engine: First choice engine name
            fallback_engines: List of fallback engines to try
            config: Optional configuration
            
        Returns:
            First available engine instance
            
        Raises:
            EngineNotFoundError: If no engines in the list are available
        """
        engines_to_try = [preferred_engine] + fallback_engines
        available = SimulationEngineRegistry.available_engines()
        
        for engine_name in engines_to_try:
            if engine_name in available:
                if config is None:
                    config = EngineConfiguration(name=engine_name)
                else:
                    config.name = engine_name
                
                logger.info(f"Using {engine_name} engine")
                return SimulationEngineRegistry.create_engine(engine_name, config)
        
        raise EngineNotFoundError(
            f"None of the requested engines are available: {engines_to_try}. "
            f"Available engines: {available}"
        )


# Auto-register engines when module is imported
def _auto_register_default_engines():
    """Automatically register default engines."""
    try:
        SimulationEngineRegistry.auto_discover_engines()
    except Exception as e:
        logger.debug(f"Auto-registration failed: {e}")


# Initialize registry with discovered engines
_auto_register_default_engines()
