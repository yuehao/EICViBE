"""
Demo: Lattice Change Management with Integrated API

This script demonstrates the new integrated API where users interact
directly with the Lattice object without needing to create a cache manually.
"""

# === Setup (same as before) ===
from eicvibe.machine_portal.lattice import Lattice
from eicvibe.machine_portal.quadrupole import Quadrupole
from eicvibe.machine_portal.drift import Drift
from eicvibe.simulators.xsuite_interface import XSuiteSimulationEngine
from eicvibe.simulators.types import SimulationMode, ParticleDistribution, DistributionType
import numpy as np

# Create simple lattice
lattice = Lattice(name="FODO")
qf = Quadrupole(name="Quad1", length=0.6)
qd = Quadrupole(name="Quad2", length=0.6)
d1 = Drift(name="Drift1", length=1.0)

qf.add_parameter("MagneticMultipoleP", "kn1", -0.5)
qd.add_parameter("MagneticMultipoleP", "kn1", 0.5)

lattice.add_element(qf)
lattice.add_element(qd)
lattice.add_element(d1)
lattice.add_branch("FODO", [qf, d1, qd, d1])

# Initialize engine and convert lattice
engine = XSuiteSimulationEngine()
engine.initialize_engine()
xsuite_line = engine.convert_lattice(lattice, mode=SimulationMode.RING,
                                     bpm_num_turns=500, bpm_frev=1e6)

# === NEW INTEGRATED API ===

# Step 1: Attach simulation to enable change management
lattice.attach_simulation(engine, xsuite_line)

# Step 2: Select elements and propose changes (supports method chaining!)
lattice.select_elements_by_type('Quadrupole') \
    .propose_change(
        parameter_group='MagneticMultipoleP',
        parameter_name='kn1',
        new_value=-0.55,
        description='Increase focusing by 10%',
        ramp_steps=8
    )

# Step 3: Review changes before applying
print("\n" + "="*80)
print("PROPOSED CHANGES - Review before activation")
print("="*80)
review_df = lattice.review_changes()
print(review_df)

# Step 4: Activate changes with ramping
particle_dist = ParticleDistribution(
    distribution_type=DistributionType.GAUSSIAN,
    num_particles=1000,
    energy=3e9,
    emittance_x=1e-9,
    emittance_y=1e-9
)
particles = engine.create_particles(particle_dist)

bpm_data = lattice.activate_changes(
    particles,
    num_turns_per_step=20,
    track_during_ramp=True
)

# Step 5: Review completed changes
print("\n" + "="*80)
print("COMPLETED CHANGES")
print("="*80)
from eicvibe.machine_portal.lattice_change import ChangeStatus
review_df = lattice.review_changes(status=ChangeStatus.COMPLETED)
print(review_df)

print("\n✓ Demo complete!")
print("  Users only interact with the Lattice object")
print("  No need to create or manage a cache manually")
