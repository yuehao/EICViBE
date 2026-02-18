"""
Base Pydantic models for EICViBE accelerator physics framework.

This module provides foundational Pydantic model classes with physics-specific
configurations and utilities for type validation and serialization.
"""

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_serializer
from typing import Optional, Union, List, Dict, Any, Callable
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PhysicsBaseModel(BaseModel):
    """Base Pydantic model for all physics-related data structures in EICViBE."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        use_enum_values=True,
        arbitrary_types_allowed=True,
    )

    @model_serializer(mode="wrap")
    def serialize_with_numpy(self, handler: Callable) -> Dict[str, Any]:
        data = handler(self)

        def convert_numpy_types(obj: Any) -> Any:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif type(obj) in (np.float64, np.float32):
                return float(obj)
            elif type(obj) in (np.int64, np.int32):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj

        return convert_numpy_types(data)

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

    @field_validator("name")
    @classmethod
    def validate_name_format(cls, v):
        """Validate element name follows reasonable conventions."""
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                "Element name must contain only alphanumeric characters, underscores, and hyphens"
            )
        return v

    @field_validator("length")
    @classmethod
    def validate_physical_length(cls, v):
        """Validate length is physically reasonable."""
        if (
            v > 10000
        ):  # 10 km seems like a reasonable upper limit for accelerator elements
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

    @field_validator("type")
    @classmethod
    def validate_group_type_format(cls, v):
        """Ensure parameter group type follows naming conventions."""
        if not v.endswith("P"):
            raise ValueError(
                "Parameter group type must end with 'P' (e.g., 'MagneticMultipoleP')"
            )
        return v
