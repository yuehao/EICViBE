#!/usr/bin/env python3
"""
Demonstration of bend geometry validation in EICViBE.

This example shows how the Pydantic-enhanced parameter validation ensures
consistency between bend length, angle, and chord_length parameters.
"""

import math
from eicvibe.machine_portal.lattice import create_element_by_type
from eicvibe.models.validators import validate_bend_geometry

def main():
    print("=== EICViBE Bend Geometry Validation Demo ===\n")
    
    # Example 1: Create a bend with length and angle, calculate chord_length
    print("1. Creating bend with length=2.0m and angle=0.1 rad...")
    bend1 = create_element_by_type("Bend", "B1", length=2.0)
    bend1.add_parameter("BendP", "angle", 0.1)
    
    # Validate and calculate chord_length
    bend1_group = bend1.get_parameter_group("BendP")
    bend1_group.validate_bend_geometry_with_length(bend1.length)
    
    angle = bend1_group.get_parameter("angle")
    chord_length = bend1_group.get_parameter("chord_length")
    
    print(f"   Length: {bend1.length} m")
    print(f"   Angle: {angle} rad ({math.degrees(angle):.2f}°)")
    print(f"   Calculated chord_length: {chord_length:.6f} m")
    
    # Verify the relationship manually
    radius = bend1.length / angle
    expected_chord = 2 * radius * math.sin(angle / 2)
    print(f"   Manual verification: chord = 2R*sin(θ/2) = {expected_chord:.6f} m ✓")
    
    print()
    
    # Example 2: Demonstrate direct geometry validation
    print("2. Direct geometry validation examples...")
    
    # Case 2a: Given length and angle, calculate chord
    length, angle, chord = validate_bend_geometry(length=1.5, angle=0.05, chord_length=None)
    print(f"   Input: length=1.5m, angle=0.05rad → chord_length={chord:.6f}m")
    
    # Case 2b: Given angle and chord, calculate length
    length, angle, chord = validate_bend_geometry(length=None, angle=0.1, chord_length=1.99)
    print(f"   Input: angle=0.1rad, chord=1.99m → length={length:.6f}m")
    
    # Case 2c: Zero angle case
    length, angle, chord = validate_bend_geometry(length=2.0, angle=0.0, chord_length=None)
    print(f"   Input: length=2.0m, angle=0.0rad → chord_length={chord}m (straight)")
    
    print()
    
    # Example 3: Consistency checking
    print("3. Consistency validation...")
    
    # Valid consistent parameters
    try:
        test_length = 2.0
        test_angle = 0.1
        test_radius = test_length / test_angle
        test_chord = 2 * test_radius * math.sin(test_angle / 2)
        
        result = validate_bend_geometry(test_length, test_angle, test_chord)
        print(f"   ✓ Consistent parameters validated: length={test_length}m, angle={test_angle}rad, chord={test_chord:.6f}m")
    except ValueError as e:
        print(f"   ✗ Validation failed: {e}")
    
    # Invalid inconsistent parameters
    try:
        validate_bend_geometry(length=2.0, angle=0.1, chord_length=5.0)
        print("   ✗ Should have failed - inconsistent parameters")
    except ValueError as e:
        print(f"   ✓ Correctly caught inconsistent parameters: {e}")
    
    print()
    
    # Example 4: Integration with element validation
    print("4. Element-level validation...")
    
    try:
        # Create bend with inconsistent geometry
        bend2 = create_element_by_type("Bend", "B2", length=1.0)
        bend2.add_parameter("BendP", "angle", 0.2)
        bend2.add_parameter("BendP", "chord_length", 10.0)  # Clearly wrong
        
        bend2_group = bend2.get_parameter_group("BendP")
        bend2_group.validate_bend_geometry_with_length(bend2.length)
        print("   ✗ Should have failed - inconsistent element geometry")
    except ValueError as e:
        print(f"   ✓ Element validation correctly caught error: {e}")
    
    print()
    print("=== Demo Complete ===")
    print("The bend geometry validation ensures physical consistency between")
    print("length, angle, and chord_length parameters in bend magnets.")

if __name__ == "__main__":
    main()