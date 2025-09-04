"""
Test suite for Pydantic-enhanced parameter groups in EICViBE.

This module tests the enhanced parameter group validation functionality
added in Phase 2 of the Pydantic integration.
"""

import pytest
import warnings
from typing import List

from eicvibe.machine_portal.parameter_group import ParameterGroup
from eicvibe.models.parameter_groups import (
    MagneticMultipoleP, BendP, RFP, SolenoidP, ApertureP, ControlP,
    MetaP, KickerP, create_parameter_group_model
)
from pydantic import ValidationError


class TestPydanticParameterGroup:
    """Test the enhanced ParameterGroup with Pydantic validation."""
    
    def test_basic_parameter_group_creation(self):
        """Test basic parameter group creation with Pydantic."""
        group = ParameterGroup(name="test_group", type="MagneticMultipoleP")
        assert group.name == "test_group"
        assert group.type == "MagneticMultipoleP"
        assert group.parameters == {}
        assert group.subgroups == []
    
    def test_parameter_validation_success(self):
        """Test successful parameter validation."""
        group = ParameterGroup(name="quad", type="MagneticMultipoleP")
        
        # Add valid parameters
        group.add_parameter("kn1", 2.0)
        group.add_parameter("ks1", 0.5)
        
        assert group.get_parameter("kn1") == 2.0
        assert group.get_parameter("ks1") == 0.5
    
    def test_parameter_validation_failure(self):
        """Test parameter validation catches invalid parameters."""
        group = ParameterGroup(name="quad", type="MagneticMultipoleP")
        
        # Try to add invalid parameter
        with pytest.raises(ValueError) as exc_info:
            group.add_parameter("invalid_param", 1.0)
        assert "not allowed" in str(exc_info.value)
    
    def test_physics_validation_quadrupole(self):
        """Test physics validation for quadrupole strength."""
        group = ParameterGroup(name="quad", type="MagneticMultipoleP")
        
        # Valid quadrupole strength
        group.add_parameter("kn1", 10.0)
        assert group.get_parameter("kn1") == 10.0
        
        # Invalid quadrupole strength (too large)
        with pytest.raises(ValueError) as exc_info:
            group.add_parameter("kn1", 5000.0)  # Exceeds 1000 T/m²
        assert "exceeds reasonable limit" in str(exc_info.value)
    
    def test_physics_validation_bend(self):
        """Test physics validation for bend parameters."""
        group = ParameterGroup(name="bend", type="BendP")
        
        # Valid bend angle
        group.add_parameter("angle", 0.1)
        assert group.get_parameter("angle") == 0.1
        
        # Invalid bend angle (too large)
        with pytest.raises(ValueError) as exc_info:
            group.add_parameter("angle", 50.0)  # Exceeds 4π
        assert "unreasonably large" in str(exc_info.value)
    
    def test_physics_validation_rf(self):
        """Test physics validation for RF parameters."""
        group = ParameterGroup(name="cavity", type="RFP")
        
        # Valid RF parameters
        group.add_parameter("voltage", 1e6)  # 1 MV
        group.add_parameter("freq", 1e9)     # 1 GHz
        group.add_parameter("phase", 0.5)    # 0.5 rad
        
        assert group.get_parameter("voltage") == 1e6
        assert group.get_parameter("freq") == 1e9
        
        # Invalid RF frequency
        with pytest.raises(ValueError) as exc_info:
            group.add_parameter("freq", 100)  # Too low
        assert "outside reasonable range" in str(exc_info.value)
    
    def test_backward_compatibility(self):
        """Test backward compatibility with existing API."""
        # Test creation with unknown parameter group type
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            group = ParameterGroup(name="test", type="UnknownGroupType")
            assert len(w) == 1
            assert "Unknown parameter group type" in str(w[0].message)
        
        # Should still allow adding parameters (backward compatibility)
        group.add_parameter("any_param", 1.0)
        assert group.get_parameter("any_param") == 1.0
    
    def test_subgroup_functionality(self):
        """Test subgroup addition and validation."""
        parent = ParameterGroup(name="parent", type="MetaP")
        child = ParameterGroup(name="child", type="MagneticMultipoleP")
        
        # Add valid subgroup
        parent.add_subgroup(child)
        assert len(parent.subgroups) == 1
        assert parent.get_subgroup_by_name("child") == child
        assert parent.get_subgroup_by_type("MagneticMultipoleP") == child
        
        # Test invalid subgroup type
        with pytest.raises(TypeError):
            parent.add_subgroup("not_a_parameter_group")
    
    def test_parameter_removal(self):
        """Test parameter and subgroup removal."""
        group = ParameterGroup(name="test", type="MagneticMultipoleP")
        group.add_parameter("kn1", 1.0)
        
        # Remove parameter
        group.remove_parameter("kn1")
        assert group.get_parameter("kn1") is None
        
        # Remove subgroup
        subgroup = ParameterGroup(name="sub", type="MetaP")
        group.add_subgroup(subgroup)
        group.remove_subgroup("sub")
        assert group.get_subgroup_by_name("sub") is None
    
    def test_serialization_methods(self):
        """Test serialization to dict and YAML formats."""
        group = ParameterGroup(name="test", type="MagneticMultipoleP")
        group.add_parameter("kn1", 2.0)
        group.add_parameter("kn2", 0.5)
        
        # Test to_dict
        data = group.to_dict()
        assert data["name"] == "test"
        assert data["type"] == "MagneticMultipoleP"
        assert data["parameters"]["kn1"] == 2.0
        assert data["parameters"]["kn2"] == 0.5
        
        # Test to_yaml_dict
        yaml_data = group.to_yaml_dict()
        assert yaml_data["kn1"] == 2.0
        assert yaml_data["kn2"] == 0.5
        
        # Test from_dict
        reconstructed = ParameterGroup.from_dict(data)
        assert reconstructed.name == group.name
        assert reconstructed.type == group.type
        assert reconstructed.parameters == group.parameters
    
    def test_allowed_parameters_methods(self):
        """Test methods for querying allowed parameters."""
        group = ParameterGroup(name="test", type="MagneticMultipoleP")
        
        # Test instance method
        allowed = group.get_allowed_parameters()
        assert isinstance(allowed, list)
        assert "kn1" in allowed
        assert "ks1" in allowed
        
        # Test static methods
        static_allowed = ParameterGroup.get_allowed_parameters_for_type("MagneticMultipoleP")
        assert static_allowed == allowed
        
        all_groups = ParameterGroup.get_all_parameter_groups()
        assert isinstance(all_groups, list)
        assert "MagneticMultipoleP" in all_groups


class TestSpecializedParameterModels:
    """Test the specialized Pydantic parameter group models."""
    
    def test_magnetic_multipole_model(self):
        """Test MagneticMultipoleP model validation."""
        # Valid magnetic multipole
        mm = MagneticMultipoleP(kn1=2.0, ks1=0.5, tilt=0.1)
        assert mm.kn1 == 2.0
        assert mm.ks1 == 0.5
        assert mm.tilt == 0.1
        
        # Invalid quadrupole strength
        with pytest.raises(ValidationError):
            MagneticMultipoleP(kn1=5000.0)  # Too large
        
        # Invalid tilt angle
        with pytest.raises(ValidationError):
            MagneticMultipoleP(tilt=10.0)  # Too large
    
    def test_bend_model(self):
        """Test BendP model validation."""
        # Valid bend
        bend = BendP(angle=0.1, E1=0.01, E2=0.01)
        assert bend.angle == 0.1
        assert bend.E1 == 0.01
        assert bend.E2 == 0.01
        
        # Invalid angle
        with pytest.raises(ValidationError):
            BendP(angle=50.0)  # Too large
        
        # Inconsistent edge angles (too large relative to bend angle)
        with pytest.raises(ValidationError):
            BendP(angle=0.1, E1=1.0, E2=1.0)  # Edge angles too large
    
    def test_bend_geometry_validation(self):
        """Test bend geometry consistency validation."""
        from eicvibe.models.validators import validate_bend_geometry
        import math
        
        # Test case 1: Given length and angle, calculate chord_length
        length, angle, chord_length = validate_bend_geometry(length=2.0, angle=0.1, chord_length=None)
        assert length == 2.0
        assert angle == 0.1
        assert chord_length is not None
        
        # Verify the geometric relationship
        radius = length / angle
        expected_chord = 2 * radius * math.sin(angle / 2)
        assert abs(chord_length - expected_chord) < 1e-6
        
        # Test case 2: Given angle and chord_length, calculate length
        length2, angle2, chord_length2 = validate_bend_geometry(length=None, angle=0.1, chord_length=1.99)
        assert angle2 == 0.1
        assert chord_length2 == 1.99
        assert length2 is not None
        
        # Test case 3: Zero angle case
        length3, angle3, chord_length3 = validate_bend_geometry(length=2.0, angle=0.0, chord_length=None)
        assert length3 == 2.0
        assert angle3 == 0.0
        assert chord_length3 == 2.0  # Should equal length for zero angle
        
        # Test case 4: All three parameters consistent
        length = 2.0
        angle = 0.1
        radius = length / angle
        chord = 2 * radius * math.sin(angle / 2)
        
        # Should not raise an error
        result = validate_bend_geometry(length=length, angle=angle, chord_length=chord)
        assert result == (length, angle, chord)
        
        # Test case 5: Inconsistent parameters should raise error
        with pytest.raises(ValueError, match="inconsistent"):
            validate_bend_geometry(length=2.0, angle=0.1, chord_length=5.0)  # Clearly wrong
        
        # Test case 6: Insufficient parameters
        with pytest.raises(ValueError, match="At least two"):
            validate_bend_geometry(length=2.0, angle=None, chord_length=None)
        
        # Test case 7: Zero angle with inconsistent length and chord
        with pytest.raises(ValueError, match="must be equal"):
            validate_bend_geometry(length=2.0, angle=0.0, chord_length=3.0)
    
    def test_rf_model(self):
        """Test RFP model validation."""
        # Valid RF cavity
        rf = RFP(voltage=1e6, freq=1e9, phase=0.5)
        assert rf.voltage == 1e6
        assert rf.freq == 1e9
        assert rf.phase == 0.5
        
        # Invalid voltage (negative)
        with pytest.raises(ValidationError):
            RFP(voltage=-1000, freq=1e9)
        
        # Invalid frequency (too low)
        with pytest.raises(ValidationError):
            RFP(voltage=1000, freq=100)
    
    def test_aperture_model(self):
        """Test ApertureP model validation."""
        # Valid aperture
        aperture = ApertureP(X=(-0.01, 0.01), Y=(-0.005, 0.005))
        assert aperture.X == (-0.01, 0.01)
        assert aperture.Y == (-0.005, 0.005)
        
        # Invalid aperture (min >= max)
        with pytest.raises(ValidationError):
            ApertureP(X=(0.01, -0.01))  # Min > Max
        
        # Invalid aperture (too large)
        with pytest.raises(ValidationError):
            ApertureP(X=(-2.0, 2.0))  # Unreasonably large
    
    def test_control_model(self):
        """Test ControlP model validation."""
        # Valid control parameters
        control = ControlP(on=True, scale=1.5)
        assert control.on is True
        assert control.scale == 1.5
        
        # Invalid scale factor (negative)
        with pytest.raises(ValidationError):
            ControlP(scale=-1.0)
        
        # Invalid scale factor (too large)
        with pytest.raises(ValidationError):
            ControlP(scale=200.0)
    
    def test_parameter_group_factory(self):
        """Test the parameter group factory function."""
        # Valid creation
        mm = create_parameter_group_model("MagneticMultipoleP", kn1=2.0)
        assert isinstance(mm, MagneticMultipoleP)
        assert mm.kn1 == 2.0
        
        bend = create_parameter_group_model("BendP", angle=0.1)
        assert isinstance(bend, BendP)
        assert bend.angle == 0.1
        
        # Invalid group type
        with pytest.raises(ValueError):
            create_parameter_group_model("UnknownType", param=1.0)


class TestIntegrationWithExistingElements:
    """Test integration with existing element classes."""
    
    def test_quadrupole_with_enhanced_validation(self):
        """Test quadrupole element with enhanced parameter validation."""
        from eicvibe.machine_portal.lattice import create_element_by_type
        
        # Create quadrupole
        quad = create_element_by_type("Quadrupole", "Q1", length=0.5)
        
        # Add parameter with enhanced validation
        quad.add_parameter("MagneticMultipoleP", "kn1", 2.0)
        
        # Verify parameter was added correctly
        group = quad.get_parameter_group("MagneticMultipoleP")
        assert group is not None
        assert group.get_parameter("kn1") == 2.0
        
        # Test that invalid parameters are rejected
        with pytest.raises(ValueError):
            quad.add_parameter("MagneticMultipoleP", "kn1", 5000.0)  # Too large
    
    def test_rf_cavity_with_enhanced_validation(self):
        """Test RF cavity with enhanced parameter validation."""
        from eicvibe.machine_portal.lattice import create_element_by_type
        
        # Create RF cavity
        cavity = create_element_by_type("RFCavity", "CAV1", length=1.0)
        
        # Add valid RF parameters
        cavity.add_parameter("RFP", "voltage", 1e6)
        cavity.add_parameter("RFP", "freq", 1e9)
        
        # Verify parameters
        rf_group = cavity.get_parameter_group("RFP")
        assert rf_group.get_parameter("voltage") == 1e6
        assert rf_group.get_parameter("freq") == 1e9
        
        # Test invalid RF frequency
        with pytest.raises(ValueError):
            cavity.add_parameter("RFP", "freq", 100)  # Too low
    
    def test_lattice_with_enhanced_parameters(self):
        """Test complete lattice with enhanced parameter validation."""
        from eicvibe.machine_portal.lattice import Lattice, create_element_by_type
        
        # Create lattice
        lattice = Lattice(name="test_lattice")
        
        # Create elements with validated parameters
        quad = create_element_by_type("Quadrupole", "Q1", length=0.5)
        quad.add_parameter("MagneticMultipoleP", "kn1", 2.0)
        
        bend = create_element_by_type("Bend", "B1", length=1.0)
        bend.add_parameter("BendP", "angle", 0.1)
        
        # Add to lattice
        lattice.add_element(quad)
        lattice.add_element(bend)
        
        # Create branch
        lattice.add_branch("main", branch_type="linac")
        lattice.add_element_to_branch("main", "Q1")
        lattice.add_element_to_branch("main", "B1")
        
        # Verify lattice consistency
        # The lattice creates copies when elements are added to branches
        # So we expect 4 elements: original Q1, B1 and their branch copies Q1_1, B1_1
        assert len(lattice.elements) == 4
        assert lattice.get_total_path_length("main") == 1.5
    
    def test_bend_geometry_integration(self):
        """Test bend geometry validation integration with actual elements."""
        from eicvibe.machine_portal.lattice import create_element_by_type
        from eicvibe.models.validators import validate_bend_geometry
        import math
        
        # Create a bend element
        bend = create_element_by_type("Bend", "B1", length=2.0)
        bend.add_parameter("BendP", "angle", 0.1)
        
        # Test the geometry validation with element
        bend_group = bend.get_parameter_group("BendP")
        
        # This should calculate and add chord_length
        bend_group.validate_bend_geometry_with_length(bend.length)
        
        # Verify chord_length was calculated correctly
        chord_length = bend_group.get_parameter("chord_length")
        assert chord_length is not None
        
        # Verify the geometric relationship
        angle = bend_group.get_parameter("angle")
        radius = bend.length / angle
        expected_chord = 2 * radius * math.sin(angle / 2)
        assert abs(chord_length - expected_chord) < 1e-6
        
        # Test inconsistent geometry
        bend2 = create_element_by_type("Bend", "B2", length=2.0)
        bend2.add_parameter("BendP", "angle", 0.1)
        bend2.add_parameter("BendP", "chord_length", 10.0)  # Clearly wrong
        
        bend2_group = bend2.get_parameter_group("BendP")
        with pytest.raises(ValueError, match="inconsistent"):
            bend2_group.validate_bend_geometry_with_length(bend2.length)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])