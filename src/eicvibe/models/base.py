"""
Base Pydantic models for EICViBE accelerator physics framework.

This module provides foundational Pydantic model classes with physics-specific
configurations and utilities for type validation and serialization.
"""

from pydantic import BaseModel, ConfigDict, Field, validator
from typing import Optional, Union, List, Dict, Any
import numpy as np


class PhysicsBaseModel(BaseModel):
    """
    Base Pydantic model for all physics-related data structures in EICViBE.
    
    This model provides:
    - Strict validation with assignment checking
    - Physics-specific configurations
    - Numpy array serialization support
    - Backward compatibility utilities
    
    Example:
        >>> class BeamParameters(PhysicsBaseModel):
        ...     energy: float = Field(gt=0, description="Beam energy in eV")
        ...     particles: int = Field(gt=0, description="Number of particles")
        
        >>> params = BeamParameters(energy=1e9, particles=1000)
        >>> params.energy
        1000000000.0
    """
    
    model_config = ConfigDict(
        # Validation settings
        validate_assignment=True,        # Validate on attribute assignment
        extra="forbid",                  # Reject unknown fields for safety
        use_enum_values=True,           # Use enum values in serialization
        
        # Type handling
        arbitrary_types_allowed=True,    # Allow numpy arrays and custom types
        
        # Serialization
        json_encoders={
            np.ndarray: lambda v: v.tolist(),  # Convert numpy arrays to lists
            np.float64: float,                  # Convert numpy floats to Python floats
            np.int64: int,                      # Convert numpy ints to Python ints
        }
    )
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """
        Create instance from dictionary for backward compatibility.
        
        Args:
            data: Dictionary with model field values
            
        Returns:
            Instance of the model
            
        Example:
            >>> data = {"energy": 1e9, "particles": 1000}
            >>> params = BeamParameters.from_dict(data)
        """
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert model to dictionary for backward compatibility.
        
        Returns:
            Dictionary representation of the model
            
        Example:
            >>> params = BeamParameters(energy=1e9, particles=1000)
            >>> params.to_dict()
            {'energy': 1000000000.0, 'particles': 1000}
        """
        return self.model_dump()
    
    def to_yaml_dict(self) -> Dict[str, Any]:
        """
        Convert to YAML-compatible dictionary.
        
        This method ensures all values are serializable to YAML format,
        which is important for EICViBE's YAML-based configuration system.
        
        Returns:
            Dictionary suitable for YAML serialization
        """
        data = self.model_dump()
        
        # Convert any remaining numpy types
        def convert_numpy_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        return convert_numpy_types(data)


class ElementConfig(PhysicsBaseModel):
    """
    Base configuration model for accelerator elements.
    
    This provides common validation and fields shared by all accelerator
    elements in the EICViBE framework.
    """
    
    name: str = Field(..., min_length=1, description="Element name")
    length: float = Field(ge=0.0, description="Element length in meters")
    inherit: Optional[str] = Field(None, description="Prototype element name")
    
    @validator('name')
    def validate_name_format(cls, v):
        """Validate element name follows reasonable conventions."""
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError("Element name must contain only alphanumeric characters, underscores, and hyphens")
        return v
    
    @validator('length')
    def validate_physical_length(cls, v):
        """Validate length is physically reasonable."""
        if v > 10000:  # 10 km seems like a reasonable upper limit for accelerator elements
            raise ValueError(f"Element length {v} m seems unreasonably large")
        return v


class PhysicsParameterGroup(PhysicsBaseModel):
    """
    Base model for physics parameter groups with validation.
    
    This provides the foundation for specialized parameter groups like
    magnetic multipoles, RF parameters, etc.
    """
    
    name: str = Field(..., min_length=1, description="Parameter group name")
    type: str = Field(..., min_length=1, description="Parameter group type")
    
    @validator('type')
    def validate_group_type_format(cls, v):
        """Ensure parameter group type follows naming conventions."""
        if not v.endswith('P'):
            raise ValueError("Parameter group type must end with 'P' (e.g., 'MagneticMultipoleP')")
        return v