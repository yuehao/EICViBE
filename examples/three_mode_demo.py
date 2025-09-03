"""
Demonstration script for the redesigned EICViBE physics engine interface.

Shows how to use the three simulation modes: LINAC, RING, and RAMPING.
"""

import sys
import os
import numpy as np
import time
import logging

# Add the src directory to Python path
sys.path.insert(0, '/Users/haoyue/src/EICViBE/src')

from eicvibe.simulators.xsuite_interface import XSuiteSimulationEngine
from eicvibe.simulators.base import SimulationMode, RampingPlan
from eicvibe.machine_portal.lattice import Lattice
from eicvibe.machine_portal.quadrupole import Quadrupole
from eicvibe.machine_portal.drift import Drift
from eicvibe.machine_portal.monitor import Monitor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_simple_fodo_lattice():
    """Create a simple FODO lattice for testing."""
    lattice = Lattice(name="test_fodo")
    
    # Create elements with proper parameter groups and unique names
    qf = Quadrupole(name="QF", length=0.3)
    qf.add_parameter("MagneticMultipoleP", "kn1", 2.0)
    
    qd = Quadrupole(name="QD", length=0.3)
    qd.add_parameter("MagneticMultipoleP", "kn1", -2.0)
    
    drift1 = Drift(name="DRIFT1", length=1.0)
    drift2 = Drift(name="DRIFT2", length=1.0)
    drift3 = Drift(name="DRIFT3", length=1.0)
    drift4 = Drift(name="DRIFT4", length=1.0)
    bpm1 = Monitor(name="BPM1")
    bpm2 = Monitor(name="BPM2")
    
    # Define sequence with unique names
    sequence = [qf, drift1, bpm1, drift2, qd, drift3, bpm2, drift4]
    
    # Add elements to lattice and create main branch
    lattice.add_branch("main", sequence)
    
    return lattice


def demonstrate_linac_mode():
    """Demonstrate LINAC mode simulation."""
    print("\n🚀 LINAC Mode Demonstration")
    print("=" * 50)
    
    # Create engine
    engine = XSuiteSimulationEngine()
    
    # Create simple lattice
    lattice = create_simple_fodo_lattice()
    
    # Simulation parameters for LINAC mode
    sim_params = {
        'mode': 'linac',
        'particle_params': {
            'num_particles': 100,
            'energy': 1e9,  # 1 GeV
            'x': np.random.normal(0, 1e-3, 100),  # 1mm RMS
            'y': np.random.normal(0, 1e-3, 100),
        },
        'linac_auto_rerun': False,  # Single shot
        'buffer_size': 10
    }
    
    # Setup simulation
    if not engine.setup_simulation(lattice, sim_params):
        print("❌ LINAC setup failed")
        return False
    
    # Start simulation
    def turn_callback(turn, monitor_data):
        print(f"  Turn {turn}: {len(monitor_data)} monitors read")
        for name, data in monitor_data.items():
            stats = data.beam_stats
            print(f"    {name}: {stats.particles_alive}/{100} particles alive, "
                  f"x_rms={stats.x_rms*1e3:.2f}mm")
    
    success = engine.start_simulation(turn_callback=turn_callback)
    if success:
        # Wait for completion
        time.sleep(2)
        status = engine.simulation_status
        print(f"  Final state: {status['state']}")
        print(f"  Total turns: {status['current_turn']}")
        print("✅ LINAC mode completed")
        
        # Get results
        print(f"  LINAC results stored: {len(engine.linac_results)} entries")
    else:
        print("❌ LINAC mode failed")
    
    engine.stop_simulation()
    engine.cleanup_engine()
    return success


def demonstrate_ring_mode():
    """Demonstrate RING mode simulation."""
    print("\n🔄 RING Mode Demonstration")
    print("=" * 50)
    
    # Create engine
    engine = XSuiteSimulationEngine()
    
    # Create simple lattice
    lattice = create_simple_fodo_lattice()
    
    # Simulation parameters for RING mode
    sim_params = {
        'mode': 'ring',
        'particle_params': {
            'num_particles': 100,
            'energy': 1e9,  # 1 GeV
            'x': np.random.normal(0, 1e-3, 100),
            'y': np.random.normal(0, 1e-3, 100),
        },
        'buffer_size': 50
    }
    
    # Setup simulation
    if not engine.setup_simulation(lattice, sim_params):
        print("❌ RING setup failed")
        return False
    
    # Start simulation
    turn_count = 0
    def turn_callback(turn, monitor_data):
        nonlocal turn_count
        turn_count = turn
        if turn % 5 == 0:  # Print every 5 turns
            print(f"  Turn {turn}: simulation running...")
    
    success = engine.start_simulation(max_turns=20, turn_callback=turn_callback)
    if success:
        # Let it run for a bit
        time.sleep(3)
        
        # Get real-time monitor data
        monitor_data = engine.get_monitor_data(last_n_turns=5)
        print(f"  Monitor data from last 5 turns:")
        for monitor_name, readings in monitor_data.items():
            print(f"    {monitor_name}: {len(readings)} readings")
            if readings:
                latest = readings[-1]
                stats = latest.beam_stats
                print(f"      Latest: {stats.particles_alive} alive, "
                      f"x_rms={stats.x_rms*1e3:.2f}mm")
        
        # Test parameter update during simulation
        print("  Testing parameter update during simulation...")
        engine.update_parameter_during_simulation("QF", "MagneticMultipoleP", "kn1", 2.5, transition_turns=5)
        
        # Wait for completion
        time.sleep(2)
        status = engine.simulation_status
        print(f"  Final state: {status['state']}")
        print(f"  Performance: {status['performance']['turns_per_second']:.1f} turns/s")
        print("✅ RING mode completed")
    else:
        print("❌ RING mode failed")
    
    engine.stop_simulation()
    engine.cleanup_engine()
    return success


def demonstrate_ramping_mode():
    """Demonstrate RAMPING mode simulation."""
    print("\n📈 RAMPING Mode Demonstration")
    print("=" * 50)
    
    # Create engine
    engine = XSuiteSimulationEngine()
    
    # Create simple lattice
    lattice = create_simple_fodo_lattice()
    
    # Create ramping plan
    ramping_plans = [
        {
            'element_name': 'QF',
            'parameter_group': 'MagneticMultipoleP',
            'parameter_name': 'kn1',
            'time_points': [0.0, 1.0, 2.0],
            'parameter_values': [2.0, 3.0, 1.5],
            'interpolation_type': 'linear'
        },
        {
            'element_name': 'QD',
            'parameter_group': 'MagneticMultipoleP',
            'parameter_name': 'kn1',
            'time_points': [0.0, 1.5, 3.0],
            'parameter_values': [-2.0, -1.0, -2.5],
            'interpolation_type': 'linear'
        }
    ]
    
    # Simulation parameters for RAMPING mode
    sim_params = {
        'mode': 'ramping',
        'particle_params': {
            'num_particles': 100,
            'energy': 1e9,  # 1 GeV
            'x': np.random.normal(0, 1e-3, 100),
            'y': np.random.normal(0, 1e-3, 100),
        },
        'ramping_plans': ramping_plans,
        'buffer_size': 50
    }
    
    # Setup simulation
    if not engine.setup_simulation(lattice, sim_params):
        print("❌ RAMPING setup failed")
        return False
    
    # Start simulation
    def turn_callback(turn, monitor_data):
        if turn % 3 == 0:  # Print every 3 turns
            # Get current parameter values
            qf_k1l = engine.get_element_parameter("QF", "MagneticMultipoleP", "kn1")
            qd_k1l = engine.get_element_parameter("QD", "MagneticMultipoleP", "kn1")
            print(f"  Turn {turn}: QF.kn1={qf_k1l:.2f}, QD.kn1={qd_k1l:.2f}")
    
    success = engine.start_simulation(max_turns=15, turn_callback=turn_callback)
    if success:
        # Let it run
        time.sleep(4)
        
        status = engine.simulation_status
        print(f"  Final state: {status['state']}")
        print(f"  Ramping plans active: {status['ramping_plans_active']}")
        print(f"  Parameter update time: {status['performance']['parameter_update_time']*1000:.2f}ms")
        print("✅ RAMPING mode completed")
    else:
        print("❌ RAMPING mode failed")
    
    engine.stop_simulation()
    engine.cleanup_engine()
    return success


def main():
    """Main demonstration function."""
    print("🎯 EICViBE Physics Engine Interface Demonstration")
    print("Three Mode Support: LINAC, RING, and RAMPING")
    print("=" * 60)
    
    try:
        # Test all three modes
        linac_success = demonstrate_linac_mode()
        ring_success = demonstrate_ring_mode()
        ramping_success = demonstrate_ramping_mode()
        
        # Summary
        print("\n📊 Summary")
        print("=" * 50)
        print(f"LINAC mode:   {'✅ Success' if linac_success else '❌ Failed'}")
        print(f"RING mode:    {'✅ Success' if ring_success else '❌ Failed'}")
        print(f"RAMPING mode: {'✅ Success' if ramping_success else '❌ Failed'}")
        
        if all([linac_success, ring_success, ramping_success]):
            print("\n🎉 All simulation modes working correctly!")
        else:
            print("\n⚠️  Some modes had issues - check logs above")
            
    except Exception as e:
        print(f"\n❌ Demonstration failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
