"""
Test suite for Pydantic-enhanced Lattice and Branch classes in EICViBE.

This module tests the enhanced lattice topology validation functionality
added in Phase 4 of the Pydantic integration.
"""

import pytest
import warnings
from typing import List

from eicvibe.machine_portal.lattice import Branch, Lattice
from eicvibe.machine_portal.drift import Drift
from eicvibe.machine_portal.bend import Bend
from eicvibe.machine_portal.element import Element
from eicvibe.machine_portal.parameter_group import ParameterGroup
from pydantic import ValidationError


class TestPydanticBranch:
    """Test the enhanced Branch class with Pydantic validation."""
    
    def test_branch_creation(self):
        """Test Branch creation with Pydantic."""
        branch = Branch(name="test_branch")
        assert branch.name == "test_branch"
        assert branch.elements == []
        assert isinstance(branch, Branch)
    
    def test_branch_validation_errors(self):
        """Test branch validation catches error conditions."""
        # Empty name should fail
        with pytest.raises(ValidationError):
            Branch(name="")
    
    def test_branch_element_operations(self):
        """Test adding and managing elements in branches."""
        branch = Branch(name="main")
        
        # Create and add elements
        drift = Drift(name="D1", length=1.0)
        bend = Bend(name="B1", length=2.0)
        
        branch.add_element(drift)
        branch.add_element(bend)
        
        # Test counts and properties
        assert branch.get_element_count() == 2
        assert branch.get_total_length() == 3.0  # 1.0 + 2.0
        
        # Test filtering by type
        drifts = branch.get_elements_by_type("Drift")
        bends = branch.get_elements_by_type("Bend")
        
        assert len(drifts) == 1
        assert len(bends) == 1
        assert drifts[0].name == "D1"
        assert bends[0].name == "B1"
    
    def test_branch_element_validation(self):
        """Test branch validates element types."""
        branch = Branch(name="main")
        
        # Valid element should work
        drift = Drift(name="D1", length=1.0)
        branch.add_element(drift)
        assert len(branch.elements) == 1
        
        # Invalid element should fail
        with pytest.raises(TypeError):
            branch.add_element("not_an_element")


class TestPydanticLattice:
    """Test the enhanced Lattice class with Pydantic validation."""
    
    def test_lattice_creation(self):
        """Test Lattice creation with Pydantic."""
        lattice = Lattice(name="test_lattice")
        assert lattice.name == "test_lattice"
        assert lattice.branches == {}
        assert lattice.elements == {}
        assert lattice.root_branch_name == ""
        assert isinstance(lattice, Lattice)
    
    def test_lattice_validation_errors(self):
        """Test lattice validation catches error conditions."""
        # Empty name should fail
        with pytest.raises(ValidationError):
            Lattice(name="")
    
    def test_lattice_element_operations(self):
        """Test adding and managing elements in lattice."""
        lattice = Lattice(name="test_lattice")
        
        # Create elements
        drift = Drift(name="D1", length=1.0)
        bend = Bend(name="B1", length=2.0)
        
        # Add elements to lattice
        lattice.add_element(drift)
        lattice.add_element(bend)
        
        assert len(lattice.elements) == 2
        assert "D1" in lattice.elements
        assert "B1" in lattice.elements
        
        # Test getting elements
        retrieved_drift = lattice.get_element("D1")
        assert retrieved_drift.name == "D1"
        assert retrieved_drift.length == 1.0
        
        # Test duplicate element names should fail
        duplicate_drift = Drift(name="D1", length=2.0)
        with pytest.raises(ValueError, match="already exists"):
            lattice.add_element(duplicate_drift)
    
    def test_lattice_branch_operations(self):
        """Test adding and managing branches in lattice."""
        lattice = Lattice(name="test_lattice")
        
        # Add elements first
        drift = Drift(name="D1", length=1.0)
        bend = Bend(name="B1", length=2.0)
        lattice.add_element(drift)
        lattice.add_element(bend)
        
        # Add branches
        lattice.add_branch("main", branch_type="linac")
        lattice.add_branch("bypass", branch_type="linac")
        
        assert len(lattice.branches) == 2
        assert "main" in lattice.branches
        assert "bypass" in lattice.branches
        assert lattice.root_branch_name == "main"  # First branch becomes root
        
        # Test branch type validation
        with pytest.raises(ValueError):
            lattice.add_branch("invalid", branch_type="invalid_type")
        
        # Test duplicate branch names should fail
        with pytest.raises(ValueError, match="already exists"):
            lattice.add_branch("main", branch_type="ring")
    
    def test_lattice_branch_element_management(self):
        """Test adding elements to branches with automatic copying."""
        lattice = Lattice(name="test_lattice")
        
        # Add prototype elements
        drift = Drift(name="D1", length=1.0)
        bend = Bend(name="B1", length=2.0)
        lattice.add_element(drift)
        lattice.add_element(bend)
        
        # Add branch
        lattice.add_branch("main", branch_type="linac")
        
        # Add elements to branch (should create copies)
        lattice.add_element_to_branch("main", "D1")
        lattice.add_element_to_branch("main", "B1")
        lattice.add_element_to_branch("main", "D1")  # Second occurrence
        
        # Check that copies were created
        assert len(lattice.elements) == 5  # 2 prototypes + 3 copies
        assert "D1_1" in lattice.elements
        assert "B1_1" in lattice.elements
        assert "D1_2" in lattice.elements
        
        # Check branch contents
        assert len(lattice.branches["main"]) == 3
        assert lattice.branches["main"] == ["D1_1", "B1_1", "D1_2"]
        
        # Check inheritance
        assert lattice.elements["D1_1"].inherit == "D1"
        assert lattice.elements["B1_1"].inherit == "B1"
    
    def test_lattice_topology_validation(self):
        """Test lattice topology validation."""
        lattice = Lattice(name="test_lattice")
        
        # Add elements
        drift = Drift(name="D1", length=1.0)
        lattice.add_element(drift)
        
        # Add branch and elements
        lattice.add_branch("main", branch_type="linac")
        lattice.add_element_to_branch("main", "D1")
        
        # Set root branch
        lattice.set_root_branch("main")
        assert lattice.root_branch_name == "main"
        
        # Test invalid root branch should fail
        with pytest.raises(ValueError):
            lattice.set_root_branch("nonexistent")
        
        # Test path length calculation
        total_length = lattice.get_total_path_length("main")
        assert total_length == 1.0
        
        # Test default to root branch
        total_length_default = lattice.get_total_path_length()
        assert total_length_default == 1.0
    
    def test_lattice_consistency_validation(self):
        """Test lattice consistency validation catches topology errors."""
        # Test creating lattice with invalid root branch should fail validation
        with pytest.raises(ValidationError, match="Root branch"):
            Lattice(
                name="test_lattice",
                branches={"main": ["D1"]},
                root_branch_name="nonexistent",
                elements={}
            )
    
    def test_lattice_branch_specs_validation(self):
        """Test branch specifications validation."""
        lattice = Lattice(name="test_lattice")
        
        # Add branch with valid type
        lattice.add_branch("ring_branch", branch_type="ring")
        lattice.add_branch("linac_branch", branch_type="linac")
        
        assert lattice.branch_specs["ring_branch"] == "ring"
        assert lattice.branch_specs["linac_branch"] == "linac"
        
        # Invalid branch specs in creation should fail
        with pytest.raises(ValidationError):
            Lattice(
                name="test_lattice",
                branch_specs={"invalid": "bad_type"}
            )


class TestLatticeIntegration:
    """Test integration between Lattice, Branch, and Element classes."""
    
    def test_complete_lattice_workflow(self):
        """Test complete lattice creation and manipulation workflow."""
        lattice = Lattice(name="EIC_lattice")
        
        # Create prototype elements with parameters
        drift = Drift(name="D1", length=2.0)
        bend = Bend(name="B1", length=1.5)
        bend.add_parameter("BendP", "angle", 0.1)
        
        # Add elements to lattice
        lattice.add_element(drift)
        lattice.add_element(bend)
        
        # Create main branch
        lattice.add_branch("main", branch_type="linac")
        
        # Build lattice sequence
        lattice.add_element_to_branch("main", "D1")
        lattice.add_element_to_branch("main", "B1")
        lattice.add_element_to_branch("main", "D1")
        
        # Verify lattice structure
        assert len(lattice.branches["main"]) == 3
        assert lattice.get_total_path_length("main") == 5.5  # 2.0 + 1.5 + 2.0
        
        # Verify bend geometry validation was applied
        bend_copy = lattice.get_element("B1_1")
        bend_group = bend_copy.get_parameter_group("BendP")
        assert bend_group is not None
        assert bend_group.get_parameter("angle") == 0.1
        
        # Verify chord_length was calculated
        chord_length = bend_group.get_parameter("chord_length")
        assert chord_length is not None
        assert isinstance(chord_length, float)
    
    def test_lattice_parameter_overrides(self):
        """Test parameter overrides when adding elements to branches."""
        lattice = Lattice(name="test_lattice")
        
        # Create prototype bend
        bend = Bend(name="B1", length=2.0)
        bend.add_parameter("BendP", "angle", 0.1)
        lattice.add_element(bend)
        
        # Add branch
        lattice.add_branch("main", branch_type="linac")
        
        # Add element with parameter override
        lattice.add_element_to_branch(
            "main", 
            "B1", 
            BendP={"angle": 0.2}  # Override the angle
        )
        
        # Verify override was applied
        bend_copy = lattice.get_element("B1_1")
        bend_group = bend_copy.get_parameter_group("BendP")
        assert bend_group.get_parameter("angle") == 0.2  # Overridden value
        
        # Trigger geometry validation with the new angle
        bend_group.validate_bend_geometry_with_length(bend_copy.length)
        
        # Verify geometry validation with new angle
        chord_length = bend_group.get_parameter("chord_length")
        assert chord_length is not None
        
        # Calculate expected chord_length for angle=0.2
        import math
        radius = 2.0 / 0.2
        expected_chord = 2 * radius * math.sin(0.2 / 2)
        assert abs(chord_length - expected_chord) < 1e-6
    
    def test_lattice_error_handling(self):
        """Test comprehensive error handling in lattice operations."""
        lattice = Lattice(name="test_lattice")
        
        # Test adding element to nonexistent branch
        drift = Drift(name="D1", length=1.0)
        lattice.add_element(drift)
        
        with pytest.raises(ValueError, match="does not exist"):
            lattice.add_element_to_branch("nonexistent", "D1")
        
        # Test adding nonexistent element to branch
        lattice.add_branch("main", branch_type="linac")
        with pytest.raises(ValueError, match="not found"):
            lattice.add_element_to_branch("main", "nonexistent")
        
        # Test removing root branch
        with pytest.raises(ValueError, match="Cannot remove"):
            lattice.remove_branch("main")
        
        # Test getting nonexistent element
        with pytest.raises(ValueError, match="not found"):
            lattice.get_element("nonexistent")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])