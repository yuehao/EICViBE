"""
Test XSuite interface with FODO cell lattice.

This test demonstrates:
1. Building a FODO cell using EICViBE elements
2. Converting to XTrack Line
3. Computing Twiss parameters
4. Tracking particles in RING mode
"""

import pytest
import numpy as np
from pathlib import Path

# Import EICViBE components
from eicvibe.machine_portal.lattice import Lattice
from eicvibe.machine_portal.drift import Drift
from eicvibe.machine_portal.quadrupole import Quadrupole
from eicvibe.machine_portal.marker import Marker
from eicvibe.machine_portal.monitor import Monitor
from eicvibe.simulators.types import (
    SimulationMode,
    ParticleDistribution,
    DistributionType,
    TrackingParameters
)

# Try to import XSuite interface
try:
    from eicvibe.simulators.xsuite_interface import XSuiteSimulationEngine, XSUITE_AVAILABLE
except ImportError:
    XSUITE_AVAILABLE = False
    XSuiteSimulationEngine = None


@pytest.mark.skipif(not XSUITE_AVAILABLE, reason="XSuite not available")
class TestXSuiteFODOCell:
    """Test XSuite interface with FODO cell lattice."""
    
    @pytest.fixture
    def fodo_lattice(self):
        """Build a FODO cell lattice for testing."""
        # FODO cell parameters
        # Cell length = 10 m
        # Drift length = 2 m
        # Quadrupole length = 0.5 m
        # Focusing strength k1 = 0.5 m^-2
        
        lattice = Lattice(name="fodo_cell")
        
        # Create a branch for the FODO cell
        lattice.add_branch(
            branch_name="ring",
            branch_type="ring",
            elements=None
        )
        
        # Define FODO cell elements
        # QF: Focusing quadrupole
        qf = Quadrupole(name="qf", length=0.5)
        qf.add_parameter("MagneticMultipoleP", "kn1", 0.5)  # Focusing
        lattice.add_element(qf)
        
        # QD: Defocusing quadrupole  
        qd = Quadrupole(name="qd", length=0.5)
        qd.add_parameter("MagneticMultipoleP", "kn1", -0.5)  # Defocusing
        lattice.add_element(qd)
        
        # Drifts
        d1 = Drift(name="d1", length=2.0)
        lattice.add_element(d1)
        d2 = Drift(name="d2", length=2.0)
        lattice.add_element(d2)
        d3 = Drift(name="d3", length=2.0)
        lattice.add_element(d3)
        
        # BPMs (markers for diagnostics - zero length)
        bpm1 = Marker(name="bpm1", length=0.0)
        lattice.add_element(bpm1)
        bpm2 = Marker(name="bpm2", length=0.0)
        lattice.add_element(bpm2)
        
        # Build FODO cell: D1 - QF - BPM1 - D2 - QD - BPM2 - D3
        lattice.add_element_to_branch("ring", "d1")
        lattice.add_element_to_branch("ring", "qf")
        lattice.add_element_to_branch("ring", "bpm1")
        lattice.add_element_to_branch("ring", "d2")
        lattice.add_element_to_branch("ring", "qd")
        lattice.add_element_to_branch("ring", "bpm2")
        lattice.add_element_to_branch("ring", "d3")
        
        # Set root branch
        lattice.set_root_branch("ring")
        
        return lattice
    
    def test_build_fodo_lattice(self, fodo_lattice):
        """Test building a FODO cell lattice."""
        # Verify lattice
        assert fodo_lattice.root_branch_name == "ring"
        assert len(fodo_lattice.branches["ring"]) == 7
        
        # Verify quadrupole parameters
        qf = fodo_lattice.get_element("qf")
        qd = fodo_lattice.get_element("qd")
        assert qf.get_parameter("MagneticMultipoleP", "kn1") == 0.5
        assert qd.get_parameter("MagneticMultipoleP", "kn1") == -0.5
    
    def test_xsuite_engine_initialization(self):
        """Test XSuite engine initialization."""
        engine = XSuiteSimulationEngine(name="xsuite_test")
        
        assert engine.name == "xsuite_test"
        assert engine.interface_config is not None
        
        # Initialize engine
        success = engine.initialize_engine()
        assert success is True
        
        # Cleanup
        engine.cleanup_engine()
    
    def test_lattice_conversion(self, fodo_lattice):
        """Test converting FODO lattice to XTrack Line."""
        # Create engine
        engine = XSuiteSimulationEngine()
        engine.initialize_engine()
        
        # Convert lattice
        line = engine.convert_lattice(fodo_lattice, SimulationMode.RING)
        
        assert line is not None
        assert len(line.element_names) > 0

        # Verify elements were converted (copies have suffix _1)
        assert any(name.startswith('d1') for name in line.element_names)
        assert any(name.startswith('qf') for name in line.element_names)
        assert any(name.startswith('qd') for name in line.element_names)

        print(f"\nConverted lattice:")
        print(f"  Number of elements: {len(line.element_names)}")
        print(f"  Element names: {line.element_names}")

        engine.cleanup_engine()
    
    def test_twiss_calculation(self, fodo_lattice):
        """Test Twiss parameter calculation for FODO cell."""
        # Create engine
        engine = XSuiteSimulationEngine()
        engine.initialize_engine()
        
        # Convert lattice
        line = engine.convert_lattice(fodo_lattice, SimulationMode.RING)
        
        # Calculate Twiss
        energy = 3e9  # 3 GeV
        twiss = engine.get_twiss(line, energy, SimulationMode.RING)

        assert twiss is not None
        # Validate TwissData fields (schema)
        assert hasattr(twiss, 'tune_x')
        assert hasattr(twiss, 'tune_y')
        assert hasattr(twiss, 'beta_x')
        assert hasattr(twiss, 'beta_y')

        print(f"\nTwiss parameters:")
        print(f"  Qx = {twiss.tune_x:.6f}")
        print(f"  Qy = {twiss.tune_y:.6f}")
        print(f"  Max betx = {np.max(twiss.beta_x):.3f} m")
        print(f"  Max bety = {np.max(twiss.beta_y):.3f} m")

        # Verify reasonable values for FODO cell
        assert 0 < twiss.tune_x < 1, "Horizontal tune should be fractional"
        assert 0 < twiss.tune_y < 1, "Vertical tune should be fractional"
        assert np.all(twiss.beta_x > 0), "Beta functions should be positive"
        assert np.all(twiss.beta_y > 0), "Beta functions should be positive"

        engine.cleanup_engine()
    
    def test_particle_creation_and_tracking(self, fodo_lattice):
        """Test particle creation and single-turn tracking."""
        # Create engine
        engine = XSuiteSimulationEngine()
        engine.initialize_engine()
        
        # Convert lattice
        line = engine.convert_lattice(fodo_lattice, SimulationMode.RING)
        
        # Build tracker
        line.build_tracker()
        
        # Create particle distribution
        distribution = ParticleDistribution(
            distribution_type=DistributionType.GAUSSIAN,
            num_particles=100,
            energy=3e9,               # 3 GeV
            delta_std=1e-3,           # relative energy spread
            zeta_std=0.01,            # m bunch length
            emittance_x=1e-6,         # m rad
            emittance_y=1e-6,         # m rad
            beta_x=10.0,              # m
            beta_y=10.0               # m
        )
        
        particles = engine.create_particles(distribution)
        
        assert particles is not None
        assert len(particles.x) == 100
        
        # Get initial coordinates
        coords_initial = engine.get_particle_coordinates()
        assert 'x' in coords_initial
        assert len(coords_initial['x']) == 100
        
        # Track one turn
        success = engine.track_single_turn()
        assert success is True
        
        # Get final coordinates
        coords_final = engine.get_particle_coordinates()
        
        # Verify particles moved
        x_change = np.abs(coords_final['x'] - coords_initial['x'])
        assert np.any(x_change > 0), "Particles should have moved"
        
        # Check particle survival
        alive = np.sum(coords_final['state'] > 0)
        print(f"\nTracking results:")
        print(f"  Initial particles: {100}")
        print(f"  Surviving particles: {alive}")
        print(f"  Mean x displacement: {np.mean(x_change):.6e} m")
        
        assert alive > 50, "Most particles should survive one turn in FODO cell"
        
        engine.cleanup_engine()
    
    def test_multi_turn_tracking(self, fodo_lattice):
        """Test multi-turn tracking in FODO cell."""
        # Create engine
        engine = XSuiteSimulationEngine()
        engine.initialize_engine()
        
        # Convert lattice
        line = engine.convert_lattice(fodo_lattice, SimulationMode.RING)
        line.build_tracker()
        
        # Create particles
        distribution = ParticleDistribution(
            distribution_type=DistributionType.GAUSSIAN,
            num_particles=50,
            energy=3e9,
            delta_std=1e-3,
            zeta_std=0.01,
            emittance_x=1e-6,
            emittance_y=1e-6,
            beta_x=10.0,
            beta_y=10.0
        )
        particles = engine.create_particles(distribution)
        
        # Track multiple turns
        num_turns = 10
        survival_rate = []
        
        for turn in range(num_turns):
            success = engine.track_single_turn()
            assert success is True
            
            coords = engine.get_particle_coordinates()
            alive = np.sum(coords['state'] > 0)
            survival_rate.append(alive / 50)
        
        print(f"\nMulti-turn tracking ({num_turns} turns):")
        print(f"  Initial particles: 50")
        print(f"  Final survival rate: {survival_rate[-1]*100:.1f}%")
        print(f"  Survival history: {[f'{r*100:.0f}%' for r in survival_rate]}")
        
        # Most particles should survive in a stable FODO cell
        assert survival_rate[-1] > 0.8, "Good survival rate expected in FODO cell"
        
        engine.cleanup_engine()


def test_xsuite_availability():
    """Test XSuite availability detection."""
    if XSUITE_AVAILABLE:
        print("\n✓ XSuite is available and ready for testing")
    else:
        print("\n✗ XSuite is not available - install with: pip install xsuite")
        pytest.skip("XSuite not available")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
