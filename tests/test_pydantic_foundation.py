"""
Test suite for Pydantic foundation in EICViBE.

This module tests the basic Pydantic models and validation functionality
added in Phase 1 of the Pydantic integration.
"""

import pytest
import numpy as np
from typing import List

from eicvibe.models.base import PhysicsBaseModel, ElementConfig, PhysicsParameterGroup
from eicvibe.models.validators import (
    validate_magnetic_strength, validate_rf_frequency, validate_bending_angle,
    validate_energy_range, validate_survival_rate, validate_element_name
)
from pydantic import Field, ValidationError


class TestPhysicsBaseModel:
    """Test the base PhysicsBaseModel functionality."""
    
    def test_basic_model_creation(self):
        """Test basic model creation and validation."""
        
        class TestModel(PhysicsBaseModel):
            energy: float = Field(gt=0, description="Test energy")
            particles: int = Field(gt=0, description="Number of particles")
        
        # Valid model creation
        model = TestModel(energy=1e9, particles=1000)
        assert model.energy == 1e9
        assert model.particles == 1000
    
    def test_validation_errors(self):
        """Test that validation errors are properly caught."""
        
        class TestModel(PhysicsBaseModel):
            energy: float = Field(gt=0, description="Test energy")
            particles: int = Field(gt=0, description="Number of particles")
        
        # Test negative energy
        with pytest.raises(ValidationError) as exc_info:
            TestModel(energy=-1000, particles=100)
        assert "greater than 0" in str(exc_info.value)
        
        # Test negative particles
        with pytest.raises(ValidationError) as exc_info:
            TestModel(energy=1000, particles=-10)
        assert "greater than 0" in str(exc_info.value)
    
    def test_backward_compatibility_methods(self):
        """Test backward compatibility methods."""
        
        class TestModel(PhysicsBaseModel):
            energy: float = Field(gt=0)
            particles: int = Field(gt=0)
        
        # Test from_dict
        data = {"energy": 1e9, "particles": 1000}
        model = TestModel.from_dict(data)
        assert model.energy == 1e9
        assert model.particles == 1000
        
        # Test to_dict
        result_dict = model.to_dict()
        assert result_dict == data
        
        # Test to_yaml_dict
        yaml_dict = model.to_yaml_dict()
        assert yaml_dict == data
    
    def test_numpy_array_support(self):
        """Test that numpy arrays are properly handled."""
        
        class BeamData(PhysicsBaseModel):
            positions: List[float] = Field(description="Particle positions")
            count: int = Field(gt=0)
        
        # Create with numpy array
        np_positions = np.random.normal(0, 1e-3, 100)
        beam = BeamData(positions=np_positions.tolist(), count=100)
        
        assert len(beam.positions) == 100
        assert beam.count == 100
        
        # Test YAML serialization
        yaml_data = beam.to_yaml_dict()
        assert isinstance(yaml_data['positions'], list)
        assert isinstance(yaml_data['positions'][0], float)


class TestElementConfig:
    """Test the ElementConfig base class."""
    
    def test_valid_element_creation(self):
        """Test valid element configuration creation."""
        element = ElementConfig(name="test_quad", length=0.5)
        assert element.name == "test_quad"
        assert element.length == 0.5
        assert element.inherit is None
    
    def test_element_with_inheritance(self):
        """Test element with inheritance."""
        element = ElementConfig(name="quad_instance", length=0.3, inherit="quad_prototype")
        assert element.inherit == "quad_prototype"
    
    def test_invalid_element_name(self):
        """Test validation of invalid element names."""
        # Empty name
        with pytest.raises(ValidationError):
            ElementConfig(name="", length=0.5)
        
        # Name validation is done by custom validator
        element = ElementConfig(name="valid_name-123", length=0.5)
        assert element.name == "valid_name-123"
    
    def test_invalid_element_length(self):
        """Test validation of invalid element lengths."""
        # Negative length
        with pytest.raises(ValidationError):
            ElementConfig(name="test", length=-1.0)
        
        # Zero length is valid
        element = ElementConfig(name="marker", length=0.0)
        assert element.length == 0.0


class TestPhysicsParameterGroup:
    """Test the PhysicsParameterGroup base class."""
    
    def test_valid_parameter_group(self):
        """Test valid parameter group creation."""
        group = PhysicsParameterGroup(name="test_group", type="MagneticMultipoleP")
        assert group.name == "test_group"
        assert group.type == "MagneticMultipoleP"
    
    def test_parameter_group_type_validation(self):
        """Test parameter group type validation."""
        # Valid type (ends with P)
        group = PhysicsParameterGroup(name="test", type="TestP")
        assert group.type == "TestP"
        
        # Invalid type (doesn't end with P)
        with pytest.raises(ValidationError) as exc_info:
            PhysicsParameterGroup(name="test", type="Invalid")
        assert "must end with 'P'" in str(exc_info.value)


class TestPhysicsValidators:
    """Test the physics-specific validators."""
    
    def test_magnetic_strength_validation(self):
        """Test magnetic strength validation."""
        # Valid strength
        assert validate_magnetic_strength(10.0) == 10.0
        assert validate_magnetic_strength(None) is None
        
        # Invalid strength
        with pytest.raises(ValueError):
            validate_magnetic_strength(2000.0)  # Exceeds default limit
    
    def test_rf_frequency_validation(self):
        """Test RF frequency validation."""
        # Valid frequencies
        assert validate_rf_frequency(1e6) == 1e6  # 1 MHz
        assert validate_rf_frequency(1e9) == 1e9  # 1 GHz
        
        # Invalid frequencies
        with pytest.raises(ValueError):
            validate_rf_frequency(100)  # Too low
        with pytest.raises(ValueError):
            validate_rf_frequency(1e15)  # Too high
    
    def test_bending_angle_validation(self):
        """Test bending angle validation."""
        # Valid angles
        assert validate_bending_angle(0.1) == 0.1
        assert validate_bending_angle(3.14159) == 3.14159
        
        # Invalid angle
        with pytest.raises(ValueError):
            validate_bending_angle(20.0)  # Too large
    
    def test_energy_range_validation(self):
        """Test energy range validation."""
        # Valid energies
        assert validate_energy_range(1e6) == 1e6  # 1 MeV
        assert validate_energy_range(1e9) == 1e9  # 1 GeV
        
        # Invalid energies
        with pytest.raises(ValueError):
            validate_energy_range(100)  # Too low
        with pytest.raises(ValueError):
            validate_energy_range(1e20)  # Too high
    
    def test_survival_rate_validation(self):
        """Test survival rate validation."""
        # Valid rates
        assert validate_survival_rate(0.0) == 0.0
        assert validate_survival_rate(0.95) == 0.95
        assert validate_survival_rate(1.0) == 1.0
        
        # Invalid rates
        with pytest.raises(ValueError):
            validate_survival_rate(-0.1)
        with pytest.raises(ValueError):
            validate_survival_rate(1.1)
    
    def test_element_name_validation(self):
        """Test element name validation."""
        # Valid names
        assert validate_element_name("quad_1") == "quad_1"
        assert validate_element_name("BPM-1.2") == "BPM-1.2"
        
        # Invalid names
        with pytest.raises(ValueError):
            validate_element_name("")  # Empty
        with pytest.raises(ValueError):
            validate_element_name("a" * 60)  # Too long


class TestIntegrationWithExistingCode:
    """Test integration with existing EICViBE code."""
    
    def test_lattice_compatibility(self):
        """Test that Pydantic models work with existing lattice code."""
        from eicvibe.machine_portal.lattice import Lattice, create_element_by_type
        
        # Create lattice using existing API
        lattice = Lattice(name="test_lattice")
        quad = create_element_by_type("Quadrupole", "Q1", length=0.5)
        
        # Add parameter using existing API
        quad.add_parameter("MagneticMultipoleP", "kn1", 2.0)
        lattice.add_element(quad)
        
        # Verify it still works
        assert lattice.name == "test_lattice"
        assert "Q1" in lattice.elements
        assert lattice.elements["Q1"].length == 0.5
    
    def test_parameter_group_compatibility(self):
        """Test parameter group compatibility."""
        from eicvibe.machine_portal.parameter_group import ParameterGroup
        
        # Create parameter group using existing API
        group = ParameterGroup(name="test", type="MagneticMultipoleP")
        group.add_parameter("kn1", 1.5)
        
        # Verify it works
        assert group.name == "test"
        assert group.type == "MagneticMultipoleP"
        assert group.get_parameter("kn1") == 1.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])