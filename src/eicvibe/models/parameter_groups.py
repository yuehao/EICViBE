"""
Specialized Pydantic models for accelerator physics parameter groups.

This module provides validated parameter group models that replace the generic
parameter dictionary approach with type-safe, physics-aware validation.
"""

from pydantic import Field, field_validator, model_validator
from typing import Optional, List, Tuple
from .base import PhysicsBaseModel
from .validators import (
    validate_magnetic_strength, validate_rf_frequency, validate_bending_angle,
    validate_survival_rate
)
import math


class MagneticMultipoleP(PhysicsBaseModel):
    """
    Magnetic multipole parameters with physics validation.
    
    This model validates magnetic multipole strengths and ensures they are
    within reasonable ranges for accelerator physics applications.
    """
    
    # Normal multipole components (Tesla/m^n)
    kn0: Optional[float] = Field(None, description="Dipole component (T/m)")
    kn1: Optional[float] = Field(None, description="Quadrupole component (T/m²)")  
    kn2: Optional[float] = Field(None, description="Sextupole component (T/m³)")
    kn3: Optional[float] = Field(None, description="Octupole component (T/m⁴)")
    kn4: Optional[float] = Field(None, description="Decapole component (T/m⁵)")
    
    # Skew multipole components (Tesla/m^n)
    ks0: Optional[float] = Field(None, description="Skew dipole component (T/m)")
    ks1: Optional[float] = Field(None, description="Skew quadrupole component (T/m²)")
    ks2: Optional[float] = Field(None, description="Skew sextupole component (T/m³)")
    ks3: Optional[float] = Field(None, description="Skew octupole component (T/m⁴)")
    ks4: Optional[float] = Field(None, description="Skew decapole component (T/m⁵)")
    
    # Rotation angle
    tilt: Optional[float] = Field(None, description="Element rotation angle (radians)")
    
    @field_validator('kn1', 'ks1')
    @classmethod
    def validate_quadrupole_strength(cls, v):
        """Validate quadrupole strength is within reasonable limits."""
        return validate_magnetic_strength(v, max_strength=1000.0)  # 1000 T/m²
    
    @field_validator('kn2', 'ks2')
    @classmethod
    def validate_sextupole_strength(cls, v):
        """Validate sextupole strength is within reasonable limits."""
        return validate_magnetic_strength(v, max_strength=10000.0)  # 10000 T/m³
    
    @field_validator('kn3', 'ks3')
    @classmethod
    def validate_octupole_strength(cls, v):
        """Validate octupole strength is within reasonable limits."""
        return validate_magnetic_strength(v, max_strength=100000.0)  # 100000 T/m⁴
    
    @field_validator('tilt')
    @classmethod
    def validate_tilt_angle(cls, v):
        """Validate tilt angle is reasonable."""
        if v is not None and abs(v) > 2 * math.pi:
            raise ValueError(f"Tilt angle {v} rad seems unreasonably large (>2π)")
        return v


class BendP(PhysicsBaseModel):
    """
    Bending magnet parameters with geometric validation.
    
    This model ensures geometric consistency between bending angles,
    edge angles, and other bend-specific parameters.
    """
    
    angle: Optional[float] = Field(None, description="Bending angle (radians)")
    E1: Optional[float] = Field(0.0, description="Entry edge angle (radians)")
    E2: Optional[float] = Field(0.0, description="Exit edge angle (radians)")
    tilt: Optional[float] = Field(0.0, description="Element rotation angle (radians)")
    edge_int1: Optional[float] = Field(0.0, description="Entry edge integral")
    edge_int2: Optional[float] = Field(0.0, description="Exit edge integral")
    hgap: Optional[float] = Field(0.0, ge=0.0, description="Half gap height (m)")
    h1: Optional[float] = Field(0.0, description="Entry pole face curvature (1/m)")
    h2: Optional[float] = Field(0.0, description="Exit pole face curvature (1/m)")
    chord_length: Optional[float] = Field(None, ge=0.0, description="Chord length (m)")
    
    @field_validator('angle')
    @classmethod
    def validate_bending_angle(cls, v):
        """Validate bending angle is reasonable."""
        if v is not None:
            return validate_bending_angle(v)
        return v
    
    @field_validator('E1', 'E2')
    @classmethod
    def validate_edge_angles(cls, v):
        """Validate edge angles are reasonable."""
        if v is not None and abs(v) > math.pi:
            raise ValueError(f"Edge angle {v} rad seems unreasonably large (>π)")
        return v
    
    @model_validator(mode='after')
    def validate_geometric_consistency(self):
        """Validate geometric relationships between length, angle, and chord_length.
        
        For bend magnets, these parameters are related by:
        - radius = length / angle (for arc length)
        - chord_length = 2 * radius * sin(angle/2)
        
        If all three are provided, they must be consistent.
        If only two are provided, the third can be calculated.
        At least two parameters are required for a valid bend.
        """
        import math
        
        # Get length from the containing element if we have access to it
        # Note: This validation works with the BendP parameters only
        # The length parameter is stored in the Element class
        
        # Only validate if angle is set
        if self.angle is not None:
            # Check for reasonable combinations of edge angles and bending angle
            if self.E1 is not None and self.E2 is not None:
                total_edge = abs(self.E1) + abs(self.E2)
                if total_edge > abs(self.angle) * 2:
                    raise ValueError(f"Sum of edge angles ({total_edge:.3f}) exceeds reasonable fraction of bending angle ({self.angle:.3f})")
            
            # Geometric consistency validation for angle and chord_length
            # Note: We can't access the element's length parameter directly from here
            # This validation will be enhanced when we have access to the full element context
            if self.chord_length is not None:
                # Basic sanity check: chord length should be reasonable relative to angle
                # For small angles: chord_length ≈ length
                # For large angles: chord_length < length
                if abs(self.angle) > 1e-6:  # Avoid division by zero
                    # Estimate minimum possible length for this angle and chord
                    # Using: length = angle * chord_length / (2 * sin(angle/2))
                    estimated_min_length = abs(self.angle * self.chord_length / (2 * math.sin(abs(self.angle) / 2)))
                    
                    # Sanity check: chord length shouldn't be unreasonably large
                    if self.chord_length > 1000.0:  # 1km seems unreasonable
                        raise ValueError(f"Chord length {self.chord_length} m seems unreasonably large (>1000m)")
                        
                    # For very small angles, chord ≈ length, so they should be similar
                    if abs(self.angle) < 0.01 and self.chord_length > 0:  # Small angle approximation
                        max_reasonable_length = self.chord_length * 1.1  # 10% tolerance
                        if estimated_min_length > max_reasonable_length:
                            raise ValueError(f"For small angle {self.angle:.6f} rad, chord length {self.chord_length} m seems inconsistent with estimated length {estimated_min_length:.3f} m")
        
        return self


class RFP(PhysicsBaseModel):
    """
    RF cavity parameters with frequency and phase validation.
    
    This model validates RF parameters for typical accelerator cavity applications.
    """
    
    voltage: Optional[float] = Field(None, ge=0, description="RF voltage (V)")
    freq: Optional[float] = Field(None, gt=0, description="RF frequency (Hz)")
    phase: Optional[float] = Field(0.0, description="RF phase (radians)")
    harmonic: Optional[float] = Field(0.0, ge=0, description="Harmonic number")
    
    @field_validator('freq')
    @classmethod
    def validate_rf_frequency(cls, v):
        """Validate RF frequency is in reasonable range."""
        if v is not None:
            return validate_rf_frequency(v)
        return v
    
    @field_validator('phase')
    @classmethod
    def validate_rf_phase(cls, v):
        """Validate RF phase is within reasonable range."""
        if v is not None and abs(v) > 4 * math.pi:
            raise ValueError(f"RF phase {v} rad seems unreasonably large (>4π)")
        return v
    
    @field_validator('voltage')
    @classmethod
    def validate_rf_voltage(cls, v):
        """Validate RF voltage is reasonable."""
        if v > 1e9:  # 1 GV seems like a reasonable upper limit
            raise ValueError(f"RF voltage {v} V seems unreasonably large (>1 GV)")
        return v


class SolenoidP(PhysicsBaseModel):
    """Solenoid magnet parameters."""
    
    ks: Optional[float] = Field(None, description="Solenoid strength (1/m)")
    
    @field_validator('ks')
    @classmethod
    def validate_solenoid_strength(cls, v):
        """Validate solenoid strength is reasonable."""
        return validate_magnetic_strength(v, max_strength=100.0)  # 100 1/m


class ApertureP(PhysicsBaseModel):
    """
    Aperture parameters defining beam pipe dimensions.
    
    This model validates aperture dimensions and ensures physical consistency.
    """
    
    X: Optional[Tuple[float, float]] = Field(None, description="[min, max] aperture in X (m)")
    Y: Optional[Tuple[float, float]] = Field(None, description="[min, max] aperture in Y (m)")
    
    @field_validator('X', 'Y')
    @classmethod
    def validate_aperture_range(cls, v):
        """Validate aperture range is physically reasonable."""
        if v is not None:
            min_val, max_val = v
            if min_val >= max_val:
                raise ValueError(f"Aperture minimum ({min_val}) must be less than maximum ({max_val})")
            if abs(min_val) > 1.0 or abs(max_val) > 1.0:
                raise ValueError(f"Aperture dimensions ({min_val}, {max_val}) seem unreasonably large (>1m)")
        return v


class ControlP(PhysicsBaseModel):
    """Control system parameters."""
    
    on: bool = Field(True, description="Element is active")
    scale: float = Field(1.0, gt=0, description="Scaling factor")
    
    @field_validator('scale')
    @classmethod
    def validate_scale_factor(cls, v):
        """Validate scaling factor is reasonable."""
        if v <= 0 or v > 100:
            raise ValueError(f"Scale factor {v} should be between 0 and 100")
        return v


class MetaP(PhysicsBaseModel):
    """Metadata parameters for documentation and tracking."""
    
    comment: Optional[str] = Field("", description="User comment")
    author: Optional[str] = Field("", description="Author name")
    date: Optional[str] = Field("", description="Creation/modification date")
    
    @field_validator('comment', 'author')
    @classmethod
    def validate_string_length(cls, v):
        """Validate string fields have reasonable length."""
        if v is not None and len(v) > 1000:
            raise ValueError(f"String field too long (>{1000} characters)")
        return v


class BodyShiftP(PhysicsBaseModel):
    """Body shift parameters for element positioning."""
    
    dx: Optional[float] = Field(0.0, description="X displacement (m)")
    dy: Optional[float] = Field(0.0, description="Y displacement (m)")  
    dz: Optional[float] = Field(0.0, description="Z displacement (m)")
    
    @field_validator('dx', 'dy', 'dz')
    @classmethod
    def validate_displacement(cls, v):
        """Validate displacement is reasonable."""
        if v is not None and abs(v) > 10.0:
            raise ValueError(f"Displacement {v} m seems unreasonably large (>10m)")
        return v


class MarkerP(PhysicsBaseModel):
    """Marker element parameters (typically empty)."""
    pass


class MonitorP(PhysicsBaseModel):
    """Monitor element parameters (typically empty)."""
    pass


class KickerP(PhysicsBaseModel):
    """Kicker element parameters."""
    
    hkick: Optional[float] = Field(0.0, description="Horizontal kick angle (radians)")
    vkick: Optional[float] = Field(0.0, description="Vertical kick angle (radians)")
    tilt: Optional[float] = Field(0.0, description="Element rotation angle (radians)")
    
    @field_validator('hkick', 'vkick')
    @classmethod
    def validate_kick_angle(cls, v):
        """Validate kick angle is reasonable."""
        if v is not None and abs(v) > 0.1:  # 0.1 rad ≈ 5.7 degrees
            raise ValueError(f"Kick angle {v} rad seems unreasonably large (>0.1 rad)")
        return v


class BeamBeamP(PhysicsBaseModel):
    """Beam-beam interaction parameters."""
    
    sigx: Optional[float] = Field(0.0, ge=0, description="Horizontal beam size (m)")
    sigy: Optional[float] = Field(0.0, ge=0, description="Vertical beam size (m)")
    xma: Optional[float] = Field(0.0, description="Horizontal beam separation (m)")
    yma: Optional[float] = Field(0.0, description="Vertical beam separation (m)")
    charge: Optional[float] = Field(0.0, description="Opposing beam charge")
    
    @field_validator('sigx', 'sigy')
    @classmethod
    def validate_beam_size(cls, v):
        """Validate beam size is reasonable."""
        if v is not None and (v < 0 or v > 0.01):  # 10mm max beam size
            raise ValueError(f"Beam size {v} m outside reasonable range (0-10mm)")
        return v


# Parameter group type mapping for factory creation
PARAMETER_GROUP_MODELS = {
    'MagneticMultipoleP': MagneticMultipoleP,
    'BendP': BendP,
    'RFP': RFP,
    'SolenoidP': SolenoidP,
    'ApertureP': ApertureP,
    'ControlP': ControlP,
    'MetaP': MetaP,
    'BodyShiftP': BodyShiftP,
    'MarkerP': MarkerP,
    'MonitorP': MonitorP,
    'KickerP': KickerP,
    'BeamBeamP': BeamBeamP,
}


def create_parameter_group_model(group_type: str, **kwargs) -> PhysicsBaseModel:
    """
    Factory function to create the appropriate parameter group model.
    
    Args:
        group_type: Parameter group type string (e.g., 'MagneticMultipoleP')
        **kwargs: Parameter values for the group
        
    Returns:
        Specialized parameter group model instance
        
    Raises:
        ValueError: If group_type is not recognized
    """
    model_class = PARAMETER_GROUP_MODELS.get(group_type)
    if model_class is None:
        raise ValueError(f"Unknown parameter group type: {group_type}")
    
    return model_class(**kwargs)