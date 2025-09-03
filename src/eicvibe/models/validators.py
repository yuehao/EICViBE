"""
Custom validators for physics-specific constraints in EICViBE.

This module provides specialized validation functions for accelerator physics
parameters, ensuring physical correctness and reasonable value ranges.
"""

from pydantic import validator
from typing import Optional, List, Union
import math


def validate_magnetic_strength(value: Optional[float], max_strength: float = 1000.0) -> Optional[float]:
    """
    Validate magnetic field strength parameters.
    
    Args:
        value: Magnetic strength value (can be None)
        max_strength: Maximum allowed strength
        
    Returns:
        Validated value
        
    Raises:
        ValueError: If strength exceeds reasonable limits
    """
    if value is not None and abs(value) > max_strength:
        raise ValueError(f"Magnetic strength {value} exceeds reasonable limit (±{max_strength})")
    return value


def validate_rf_frequency(frequency: float) -> float:
    """
    Validate RF frequency is in reasonable range for accelerators.
    
    Args:
        frequency: RF frequency in Hz
        
    Returns:
        Validated frequency
        
    Raises:
        ValueError: If frequency is outside reasonable range
    """
    if not (1e6 <= frequency <= 1e12):  # 1 MHz to 1 THz
        raise ValueError(f"RF frequency {frequency} Hz outside reasonable range (1 MHz - 1 THz)")
    return frequency


def validate_bending_angle(angle: float) -> float:
    """
    Validate bending magnet angle.
    
    Args:
        angle: Bending angle in radians
        
    Returns:
        Validated angle
        
    Raises:
        ValueError: If angle seems unreasonably large
    """
    if abs(angle) > 4 * math.pi:  # More than 4π radians (2 full circles)
        raise ValueError(f"Bending angle {angle} rad seems unreasonably large (>{4*math.pi:.2f} rad)")
    return angle


def validate_energy_range(energy: float, min_energy: float = 1e3, max_energy: float = 1e15) -> float:
    """
    Validate particle energy is in reasonable range.
    
    Args:
        energy: Particle energy in eV
        min_energy: Minimum allowed energy (default: 1 keV)
        max_energy: Maximum allowed energy (default: 1 PeV)
        
    Returns:
        Validated energy
        
    Raises:
        ValueError: If energy is outside reasonable range
    """
    if not (min_energy <= energy <= max_energy):
        raise ValueError(f"Energy {energy} eV outside reasonable range ({min_energy} - {max_energy} eV)")
    return energy


def validate_survival_rate(rate: float) -> float:
    """
    Validate particle survival rate.
    
    Args:
        rate: Survival rate (should be between 0 and 1)
        
    Returns:
        Validated rate
        
    Raises:
        ValueError: If rate is not between 0 and 1
    """
    if not (0.0 <= rate <= 1.0):
        raise ValueError(f"Survival rate {rate} must be between 0.0 and 1.0")
    return rate


def validate_time_sequence(time_points: List[float]) -> List[float]:
    """
    Validate time sequence is monotonically increasing.
    
    Args:
        time_points: List of time values
        
    Returns:
        Validated time points
        
    Raises:
        ValueError: If time points are not in ascending order
    """
    if len(time_points) > 1:
        for i in range(len(time_points) - 1):
            if time_points[i] > time_points[i + 1]:
                raise ValueError(f"Time points must be in ascending order: {time_points[i]} > {time_points[i+1]}")
    return time_points


def validate_element_name(name: str) -> str:
    """
    Validate element name follows accelerator naming conventions.
    
    Args:
        name: Element name
        
    Returns:
        Validated name
        
    Raises:
        ValueError: If name doesn't follow conventions
    """
    if not name:
        raise ValueError("Element name cannot be empty")
    
    # Check for reasonable length
    if len(name) > 50:
        raise ValueError(f"Element name '{name}' is too long (max 50 characters)")
    
    # Check for valid characters (alphanumeric, underscore, hyphen, dot)
    valid_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.')
    if not all(c in valid_chars for c in name):
        raise ValueError(f"Element name '{name}' contains invalid characters")
    
    return name


def validate_lattice_branch_type(branch_type: str) -> str:
    """
    Validate lattice branch type.
    
    Args:
        branch_type: Branch type string
        
    Returns:
        Validated branch type
        
    Raises:
        ValueError: If branch type is not recognized
    """
    valid_types = {'ring', 'linac'}
    if branch_type not in valid_types:
        raise ValueError(f"Branch type '{branch_type}' must be one of {valid_types}")
    return branch_type