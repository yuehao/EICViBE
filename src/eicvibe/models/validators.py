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


def validate_bend_geometry(length: Optional[float], angle: Optional[float], chord_length: Optional[float], 
                          tolerance: float = 1e-6) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Validate and compute consistent bend geometry parameters.
    
    For bend magnets, the three parameters are related by:
    - radius = length / angle (for arc length)
    - chord_length = 2 * radius * sin(angle/2)
    
    Args:
        length: Arc length of the bend (m)
        angle: Bending angle (radians)
        chord_length: Straight-line chord length (m)
        tolerance: Numerical tolerance for consistency checks
        
    Returns:
        Tuple of (length, angle, chord_length) with computed missing values
        
    Raises:
        ValueError: If parameters are inconsistent or insufficient
    """
    # Count how many parameters are provided
    provided = sum(x is not None for x in [length, angle, chord_length])
    
    if provided < 2:
        raise ValueError("At least two of (length, angle, chord_length) must be provided for bend geometry")
    
    # Handle special case: zero angle
    if angle is not None and abs(angle) < tolerance:
        if length is not None and chord_length is not None:
            if abs(length - chord_length) > tolerance:
                raise ValueError(f"For zero bending angle, length ({length}) and chord_length ({chord_length}) must be equal")
        elif length is not None:
            chord_length = length
        elif chord_length is not None:
            length = chord_length
        return length, angle, chord_length
    
    # Normal case: non-zero angle
    if provided == 2:
        # Calculate the missing parameter
        if angle is None:
            # Given length and chord_length, solve for angle
            if length <= 0 or chord_length <= 0:
                raise ValueError("Length and chord_length must be positive for angle calculation")
            
            # Solve: chord_length = 2 * (length/angle) * sin(angle/2)
            # This requires numerical solution, but we can use approximation for small angles
            # or exact solution for specific cases
            
            # For small angles: chord ≈ length, so angle ≈ 0
            # For exact solution: need to solve transcendental equation
            # We'll use Newton's method for general case
            
            def f(a):
                if abs(a) < tolerance:
                    return length - chord_length
                return chord_length - 2 * (length/a) * math.sin(a/2)
            
            def df_da(a):
                if abs(a) < tolerance:
                    return 0
                radius = length / a
                return -2 * radius * math.cos(a/2) + chord_length / a
            
            # Initial guess
            if abs(length - chord_length) < tolerance:
                angle = 0.0
            else:
                # For reasonable initial guess, use small angle approximation
                angle = 2 * (length - chord_length) / length
            
            # Newton's method
            for _ in range(10):
                if abs(angle) < tolerance:
                    break
                fa = f(angle)
                if abs(fa) < tolerance:
                    break
                dfa = df_da(angle)
                if abs(dfa) < tolerance:
                    break
                angle_new = angle - fa / dfa
                if abs(angle_new - angle) < tolerance:
                    angle = angle_new
                    break
                angle = angle_new
            
            # Validate the solution
            if abs(f(angle)) > tolerance:
                raise ValueError(f"Could not find consistent angle for length={length}, chord_length={chord_length}")
                
        elif length is None:
            # Given angle and chord_length, calculate length
            if abs(angle) < tolerance:
                length = chord_length
            else:
                radius = chord_length / (2 * math.sin(abs(angle) / 2))
                length = abs(angle) * radius
                
        elif chord_length is None:
            # Given length and angle, calculate chord_length
            if abs(angle) < tolerance:
                chord_length = length
            else:
                radius = length / abs(angle)
                chord_length = 2 * radius * math.sin(abs(angle) / 2)
    
    else:  # provided == 3
        # All three parameters provided - check consistency
        if abs(angle) < tolerance:
            # Zero angle case
            if abs(length - chord_length) > tolerance:
                raise ValueError(f"For zero angle, length ({length}) and chord_length ({chord_length}) must be equal")
        else:
            # Non-zero angle case
            radius = length / abs(angle)
            expected_chord = 2 * radius * math.sin(abs(angle) / 2)
            
            if abs(chord_length - expected_chord) > tolerance:
                raise ValueError(
                    f"Bend geometry parameters are inconsistent: "
                    f"length={length}, angle={angle}, chord_length={chord_length}. "
                    f"Expected chord_length={expected_chord:.6f} for given length and angle"
                )
    
    # Final validation
    if length is not None and length < 0:
        raise ValueError(f"Bend length must be non-negative, got {length}")
    if chord_length is not None and chord_length < 0:
        raise ValueError(f"Bend chord_length must be non-negative, got {chord_length}")
    
    return length, angle, chord_length