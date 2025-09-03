#!/usr/bin/env python3
"""
Test script for the new simulation modes in EICViBE.

This script tests LINAC, RING, and RAMPING simulation modes.
"""

import sys
from eicvibe.machine_portal.lattice import Lattice, create_element_by_type
from eicvibe.simulators import (
    SimulatorManager, XSuiteSimulator, 
    SimulationMode, RampingPlan
)


def create_simple_lattice():
    """Create a simple test lattice."""
    lattice = Lattice(name="TestLattice")
    
    # Create elements
    drift1 = create_element_by_type("Drift", "D1", length=1.0)
    lattice.add_element(drift1)
    
    quad = create_element_by_type("Quadrupole", "Q1", length=0.5)
    quad.add_parameter("MagneticMultipoleP", "kn1", 1.0)
    lattice.add_element(quad)
    
    drift2 = create_element_by_type("Drift", "D2", length=1.0)
    lattice.add_element(drift2)
    
    # Build lattice
    lattice.add_branch("main", branch_type="linac")
    lattice.add_element_to_branch("main", "D1")
    lattice.add_element_to_branch("main", "Q1")
    lattice.add_element_to_branch("main", "D2")
    
    return lattice


def create_ring_lattice():
    """Create a simple ring lattice."""
    lattice = Lattice(name="RingLattice")
    
    # Create FODO cell elements
    drift = create_element_by_type("Drift", "D1", length=2.0)
    lattice.add_element(drift)
    
    quad_f = create_element_by_type("Quadrupole", "QF", length=0.5)
    quad_f.add_parameter("MagneticMultipoleP", "kn1", 1.0)  # Focusing
    lattice.add_element(quad_f)
    
    quad_d = create_element_by_type("Quadrupole", "QD", length=0.5)
    quad_d.add_parameter("MagneticMultipoleP", "kn1", -1.0)  # Defocusing
    lattice.add_element(quad_d)
    
    # Build ring
    lattice.add_branch("main", branch_type="ring")
    lattice.add_element_to_branch("main", "D1")
    lattice.add_element_to_branch("main", "QF")
    lattice.add_element_to_branch("main", "D1")
    lattice.add_element_to_branch("main", "QD")
    
    return lattice


def test_linac_mode():
    """Test LINAC simulation mode."""
    print("\n=== LINAC Mode Test ===")
    
    try:
        simulator = XSuiteSimulator()
        simulator.start_service()
        
        lattice = create_simple_lattice()
        
        # Submit LINAC simulation
        request_id = simulator.submit_linac_simulation(
            lattice=lattice,
            generation_rate=500.0,  # particles/second
            continuous_duration=2.0,  # seconds
            parameters={
                'batch_size': 100,
                'save_particles': False
            }
        )
        
        print(f"✅ LINAC simulation submitted: {request_id}")
        
        # Wait for result
        import time
        time.sleep(3)
        
        result = simulator.get_latest_result()
        if result and result.success:
            data = result.data
            print(f"✅ LINAC simulation completed:")
            print(f"   - Generated: {data['total_particles_generated']} particles")
            print(f"   - Survival rate: {data['overall_survival_rate']:.1%}")
            print(f"   - Batches: {data['batches_processed']}")
            print(f"   - Computation time: {result.computation_time:.3f}s")
        else:
            print(f"❌ LINAC simulation failed: {result.error_message if result else 'No result'}")
            return False
        
        simulator.stop_service()
        return True
        
    except Exception as e:
        print(f"❌ LINAC mode test failed: {e}")
        return False


def test_ring_mode():
    """Test RING simulation mode."""
    print("\n=== RING Mode Test ===")
    
    try:
        simulator = XSuiteSimulator()
        simulator.start_service()
        
        lattice = create_ring_lattice()
        
        # Submit RING simulation
        request_id = simulator.submit_ring_simulation(
            lattice=lattice,
            num_particles=200,
            continuous_duration=0.01,  # Short duration for test
            parameters={
                'save_particles': False,
                'max_turns': 100  # Limit turns for testing
            }
        )
        
        print(f"✅ RING simulation submitted: {request_id}")
        
        # Wait for result
        import time
        time.sleep(3)
        
        result = simulator.get_latest_result()
        if result and result.success:
            data = result.data
            print(f"✅ RING simulation completed:")
            print(f"   - Initial particles: {data['initial_particles']}")
            print(f"   - Final survival: {data['final_survival_rate']:.1%}")
            print(f"   - Turns completed: {data['num_turns']}")
            print(f"   - Revolution time: {data['revolution_time']:.6f}s")
            print(f"   - Computation time: {result.computation_time:.3f}s")
        else:
            print(f"❌ RING simulation failed: {result.error_message if result else 'No result'}")
            return False
        
        simulator.stop_service()
        return True
        
    except Exception as e:
        print(f"❌ RING mode test failed: {e}")
        return False


def test_ramping_mode():
    """Test RAMPING simulation mode."""
    print("\n=== RAMPING Mode Test ===")
    
    try:
        simulator = XSuiteSimulator()
        simulator.start_service()
        
        lattice = create_simple_lattice()
        
        # Create a simple ramping plan
        ramping_plan = RampingPlan(
            name="test_ramp",
            duration=1.0,  # 1 second ramp
            parameters={"Q1.k1": 0.5}  # Placeholder parameter
        )
        
        # Submit RAMPING simulation
        request_id = simulator.submit_ramping_simulation(
            lattice=lattice,
            ramping_plan=ramping_plan,
            parameters={
                'time_steps': 20,
                'num_particles': 100,
                'save_particles': False
            }
        )
        
        print(f"✅ RAMPING simulation submitted: {request_id}")
        
        # Wait for result
        import time
        time.sleep(3)
        
        result = simulator.get_latest_result()
        if result and result.success:
            data = result.data
            print(f"✅ RAMPING simulation completed:")
            print(f"   - Ramping plan: {data['ramping_plan_name']}")
            print(f"   - Duration: {data['ramping_duration']}s")
            print(f"   - Time steps: {data['time_steps']}")
            print(f"   - Final survival: {data['final_survival_rate']:.1%}")
            print(f"   - Computation time: {result.computation_time:.3f}s")
        else:
            print(f"❌ RAMPING simulation failed: {result.error_message if result else 'No result'}")
            return False
        
        simulator.stop_service()
        return True
        
    except Exception as e:
        print(f"❌ RAMPING mode test failed: {e}")
        return False


def test_mode_switching():
    """Test switching between different simulation modes."""
    print("\n=== Mode Switching Test ===")
    
    try:
        manager = SimulatorManager()
        simulator = XSuiteSimulator()
        manager.register_simulator(simulator)
        manager.start_all_services()
        
        lattice = create_simple_lattice()
        
        # Test updating lattice with different modes
        print("Testing mode switching...")
        
        # LINAC mode
        request_ids = manager.update_lattice_for_all(
            lattice, simulation_mode=SimulationMode.LINAC,
            simulation_parameters={'generation_rate': 100, 'continuous_duration': 0.5}
        )
        print(f"✅ LINAC mode update: {request_ids}")
        
        import time
        time.sleep(2)
        
        # RING mode  
        ring_lattice = create_ring_lattice()
        request_ids = manager.update_lattice_for_all(
            ring_lattice, simulation_mode=SimulationMode.RING,
            simulation_parameters={'num_particles': 50, 'continuous_duration': 0.001}
        )
        print(f"✅ RING mode update: {request_ids}")
        
        time.sleep(2)
        
        # Get final results
        results = manager.get_all_latest_results()
        if results.get("XSuite") and results["XSuite"].success:
            data = results["XSuite"].data
            mode = data.get('simulation_mode', 'unknown')
            print(f"✅ Final simulation mode: {mode}")
        
        manager.stop_all_services()
        return True
        
    except Exception as e:
        print(f"❌ Mode switching test failed: {e}")
        return False


def main():
    """Main test function."""
    print("EICViBE Simulation Modes Test")
    print("=============================")
    
    tests = [
        ("LINAC Mode", test_linac_mode),
        ("RING Mode", test_ring_mode), 
        ("RAMPING Mode", test_ramping_mode),
        ("Mode Switching", test_mode_switching)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {100*passed/total:.1f}%")
    
    if passed == total:
        print("🎉 All simulation mode tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
