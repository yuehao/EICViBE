"""
Test suite for Pydantic-enhanced Element classes in EICViBE.

This module tests the enhanced element validation functionality
added in Phase 3 of the Pydantic integration.
"""

import pytest
import warnings
from typing import List

from eicvibe.machine_portal.element import Element
from eicvibe.machine_portal.drift import Drift
from eicvibe.machine_portal.bend import Bend
from eicvibe.machine_portal.parameter_group import ParameterGroup
from pydantic import ValidationError


class TestPydanticElement:
    """Test the enhanced Element base class with Pydantic validation."""
    
    def test_basic_element_creation(self):
        """Test basic element creation with Pydantic."""
        element = Element(name="test_element", type="Drift", length=1.0)
        assert element.name == "test_element"
        assert element.type == "Drift"
        assert element.length == 1.0
        assert element.inherit is None
        assert element.parameters == []
    
    def test_element_validation_errors(self):
        """Test element validation catches various error conditions."""
        # Empty name should fail
        with pytest.raises(ValidationError):
            Element(name="", type="Drift", length=1.0)
        
        # Empty type should fail  
        with pytest.raises(ValidationError):
            Element(name="test", type="", length=1.0)
        
        # Negative length should fail
        with pytest.raises(ValidationError):
            Element(name="test", type="Drift", length=-1.0)
        
        # Very large length should fail
        with pytest.raises(ValidationError):
            Element(name="test", type="Drift", length=20000.0)  # > 10km
    
    def test_element_parameter_group_validation(self):
        """Test parameter group validation against elements.yaml."""
        element = Element(name="test", type="Drift", length=1.0)
        
        # ApertureP should be allowed for Drift (from "All" groups)
        group = ParameterGroup(name="ApertureP", type="ApertureP")
        group.add_parameter("X", [-0.01, 0.01])
        element.add_parameter_group(group)
        assert len(element.parameters) == 1
        
        # BendP should not be allowed for Drift
        with pytest.raises(ValueError, match="not allowed"):
            bend_group = ParameterGroup(name="BendP", type="BendP")
            element.add_parameter_group(bend_group)
    
    def test_element_inheritance_and_plotting(self):
        """Test element inheritance and plotting attributes."""
        element = Element(
            name="test", 
            type="Drift", 
            length=1.0, 
            inherit="prototype",
            plot_color="red",
            plot_height=0.5
        )
        assert element.inherit == "prototype"
        assert element.plot_color == "red"
        assert element.plot_height == 0.5
    
    def test_element_methods(self):
        """Test element getter/setter methods."""
        element = Element(name="test", type="Drift", length=1.0)
        
        # Test getters
        assert element.get_name() == "test"
        assert element.get_type() == "Drift"
        assert element.get_length() == 1.0
        assert element.get_inherit() is None
        
        # Test setters
        element.set_name("new_name")
        assert element.get_name() == "new_name"
        
        element.set_length(2.0)
        assert element.get_length() == 2.0
        
        element.set_inherit("prototype")
        assert element.get_inherit() == "prototype"


class TestPydanticDrift:
    """Test the enhanced Drift class with Pydantic validation."""
    
    def test_drift_creation(self):
        """Test Drift element creation."""
        drift = Drift(name="D1", length=2.5)
        assert drift.name == "D1"
        assert drift.type == "Drift"
        assert drift.length == 2.5
        assert isinstance(drift, Element)  # Inheritance check
    
    def test_drift_validation(self):
        """Test Drift-specific validation."""
        # Positive length should work
        drift = Drift(name="D1", length=1.0)
        assert drift.length == 1.0
        
        # Zero length should fail
        with pytest.raises(ValidationError):
            Drift(name="D1", length=0.0)
        
        # Negative length should fail
        with pytest.raises(ValidationError):
            Drift(name="D1", length=-1.0)
    
    def test_drift_parameter_groups(self):
        """Test parameter groups on Drift elements."""
        drift = Drift(name="D1", length=2.0)
        
        # Add allowed parameter group
        group = ParameterGroup(name="ApertureP", type="ApertureP")
        group.add_parameter("X", [-0.02, 0.02])
        drift.add_parameter_group(group)
        
        # Verify parameter was added
        retrieved_group = drift.get_parameter_group("ApertureP")
        assert retrieved_group is not None
        assert retrieved_group.get_parameter("X") == [-0.02, 0.02]
    
    def test_drift_string_representation(self):
        """Test Drift string representation."""
        drift = Drift(name="D1", length=2.0, inherit="prototype")
        drift_str = str(drift)
        assert "Drift" in drift_str
        assert "D1" in drift_str
        assert "2.0" in drift_str
        assert "prototype" in drift_str


class TestPydanticBend:
    """Test the enhanced Bend class with Pydantic validation and geometry."""
    
    def test_bend_creation(self):
        """Test Bend element creation."""
        bend = Bend(name="B1", length=2.0)
        assert bend.name == "B1"
        assert bend.type == "Bend"
        assert bend.length == 2.0
        assert bend.plot_color == "C0"  # Default value
        assert isinstance(bend, Element)  # Inheritance check
    
    def test_bend_validation(self):
        """Test Bend-specific validation."""
        # Non-negative length should work
        bend = Bend(name="B1", length=0.0)  # Zero length allowed for bends
        assert bend.length == 0.0
        
        # Positive length should work
        bend = Bend(name="B1", length=2.0)
        assert bend.length == 2.0
        
        # Negative length should fail
        with pytest.raises(ValidationError):
            Bend(name="B1", length=-1.0)
    
    def test_bend_geometry_validation(self):
        """Test bend geometry validation integration."""
        bend = Bend(name="B1", length=2.0)
        
        # Add angle parameter
        bend.add_parameter("BendP", "angle", 0.1)
        
        # Get the bend group and verify geometry validation
        bend_group = bend.get_parameter_group("BendP")
        assert bend_group is not None
        
        # Trigger geometry validation
        bend_group.validate_bend_geometry_with_length(bend.length)
        
        # Check if chord_length was calculated
        chord_length = bend_group.get_parameter("chord_length")
        assert chord_length is not None
        assert isinstance(chord_length, float)
        
        # Verify the geometric relationship
        import math
        radius = bend.length / 0.1
        expected_chord = 2 * radius * math.sin(0.1 / 2)
        assert abs(chord_length - expected_chord) < 1e-6
    
    def test_bend_geometry_validation_failure(self):
        """Test bend geometry validation catches inconsistent parameters."""
        bend = Bend(name="B1", length=1.0)
        bend.add_parameter("BendP", "angle", 0.1)
        bend.add_parameter("BendP", "chord_length", 10.0)  # Clearly wrong
        
        bend_group = bend.get_parameter_group("BendP")
        with pytest.raises(ValueError, match="inconsistent"):
            bend_group.validate_bend_geometry_with_length(bend.length)
    
    def test_bend_plotting_attributes(self):
        """Test bend plotting attributes."""
        bend = Bend(
            name="B1", 
            length=2.0, 
            plot_color="red", 
            plot_height=0.8,
            plot_cross_section=0.6
        )
        assert bend.plot_color == "red"
        assert bend.plot_height == 0.8
        assert bend.plot_cross_section == 0.6


class TestElementIntegration:
    """Test integration between different element types."""
    
    def test_element_factory_pattern(self):
        """Test that elements can be created through factory pattern."""
        # Test direct creation
        drift = Drift(name="D1", length=1.0)
        bend = Bend(name="B1", length=2.0)
        
        assert isinstance(drift, Drift)
        assert isinstance(bend, Bend)
        assert isinstance(drift, Element)
        assert isinstance(bend, Element)
    
    def test_parameter_group_compatibility(self):
        """Test parameter group compatibility across element types."""
        drift = Drift(name="D1", length=1.0)
        bend = Bend(name="B1", length=2.0)
        
        # Both should accept ApertureP (from "All" groups)
        aperture_group = ParameterGroup(name="ApertureP", type="ApertureP")
        aperture_group.add_parameter("X", [-0.01, 0.01])
        
        drift.add_parameter_group(aperture_group)
        
        # Create a new group for bend (can't reuse the same instance)
        bend_aperture_group = ParameterGroup(name="ApertureP", type="ApertureP")
        bend_aperture_group.add_parameter("Y", [-0.005, 0.005])
        bend.add_parameter_group(bend_aperture_group)
        
        assert len(drift.parameters) == 1
        assert len(bend.parameters) == 1
    
    def test_consistency_checking(self):
        """Test element consistency checking."""
        drift = Drift(name="D1", length=1.0)
        bend = Bend(name="B1", length=2.0)
        
        # Drift should always be consistent
        assert drift.check_consistency() is True
        
        # Bend without BendP group should be consistent (flexible validation)
        assert bend.check_consistency() is True
        
        # Bend with BendP group should be consistent
        bend.add_parameter("BendP", "angle", 0.1)
        assert bend.check_consistency() is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])