# Creating Matched Particle Beams

## Overview

A **matched beam** is a particle distribution whose statistical properties (RMS sizes, correlations) are consistent with the lattice optics (Twiss parameters) at the creation point. When properly matched, the beam envelope remains stable as particles circulate through the lattice.

## Why Use Matched Beams?

### Benefits

1. **Beam Stability**: Matched beams maintain constant RMS sizes over many turns
2. **Realistic Simulations**: Real accelerators inject matched beams to minimize emittance growth
3. **Proper Coupling**: Accounts for x-y coupling from lattice elements
4. **Closed Orbit**: Automatically centers distribution around the closed orbit
5. **Dispersion**: Includes dispersive effects in the particle distribution

### When to Use

- ✅ Ring simulations requiring long-term stability
- ✅ Injection studies where beam quality matters
- ✅ Collective effects simulations (space charge, impedance)
- ✅ Comparing simulations with experimental beam measurements

### When NOT to Use

- ❌ Intentionally mismatched injection studies
- ❌ Testing lattice response to large oscillations
- ❌ LINAC simulations where matching concept doesn't apply

## The Physics

### Beam Size Formula

For a matched Gaussian beam in the absence of dispersion:

$$\sigma_x = \sqrt{\epsilon_x \cdot \beta_x}$$

$$\sigma_{p_x} = \sqrt{\frac{\epsilon_x}{\beta_x}}$$

where:
- $\epsilon_x$ is the geometric emittance (m·rad)
- $\beta_x$ is the lattice beta function at the creation point (m)
- Similar equations hold for the y-plane

### Emittance Definitions

**Geometric emittance** (used in EICViBE):
$$\epsilon_{\text{geom}} = \text{area in phase space} / \pi$$

**Normalized emittance** (used by XSuite):
$$\epsilon_{\text{norm}} = \epsilon_{\text{geom}} \times \beta_{\text{rel}} \times \gamma_{\text{rel}}$$

where $\beta_{\text{rel}}$ and $\gamma_{\text{rel}}$ are relativistic factors.

### Coupling and Correlations

A properly matched beam has:
- Zero correlation in uncoupled planes: $\langle x \cdot p_x \rangle = 0$
- Proper correlation if coupling exists (handled automatically by XSuite)
- Centered around closed orbit (not necessarily zero)

## Usage

### Basic Usage

```python
from eicvibe.simulators.xsuite_interface import XSuiteSimulationEngine
from eicvibe.simulators.types import ParticleDistribution, DistributionType

# Initialize engine and convert lattice
engine = XSuiteSimulationEngine()
engine.initialize_engine()
engine.convert_lattice(lattice, "main")

# Define distribution parameters
distribution = ParticleDistribution(
    num_particles=10000,
    energy=10e9,  # 10 GeV
    distribution_type=DistributionType.GAUSSIAN,
    emittance_x=5e-6,  # m·rad (geometric emittance)
    emittance_y=2e-6,  # m·rad (geometric emittance)
    beta_x=10.0,  # Ignored when use_lattice_twiss=True
    beta_y=5.0,   # Ignored when use_lattice_twiss=True
    bunch_length=1e-3,
    energy_spread=1e-4
)

# Create matched particles
particles = engine.create_matched_particles(
    distribution,
    element_name=None,  # Create at start of lattice
    use_lattice_twiss=True
)
```

### Advanced Options

#### Create at Specific Element

```python
# Create matched beam at a specific element (e.g., injection point)
particles = engine.create_matched_particles(
    distribution,
    element_name="injection_point",
    use_lattice_twiss=True
)
```

#### Specify Normalized Emittances Directly

```python
# If you already have normalized emittances
gamma_rel = distribution.energy / 938.27208816e6
beta_rel = np.sqrt(1 - 1/gamma_rel**2)
nemitt_x = distribution.emittance_x * beta_rel * gamma_rel

particles = engine.create_matched_particles(
    distribution,
    nemitt_x=1e-6,  # m·rad (normalized)
    nemitt_y=1e-6,  # m·rad (normalized)
    use_lattice_twiss=True
)
```

## Verification

### Check Beam Sizes

```python
import numpy as np
from math import sqrt

# Get Twiss parameters at creation point
twiss = engine.calculate_twiss()
beta_x_lattice = twiss.beta_x[0]
beta_y_lattice = twiss.beta_y[0]

# Calculate expected beam sizes
gamma_rel = distribution.energy / 938.27208816e6
beta_rel = sqrt(1 - 1/gamma_rel**2)
nemitt_x = distribution.emittance_x * beta_rel * gamma_rel

sigma_x_expected = sqrt(nemitt_x * beta_x_lattice / (beta_rel * gamma_rel))

# Compare with actual
sigma_x_actual = np.std(particles.x)
print(f"Expected σ_x: {sigma_x_expected*1e3:.4f} mm")
print(f"Actual σ_x: {sigma_x_actual*1e3:.4f} mm")
```

### Verify Matching Quality

```python
# Track for multiple turns
rms_x = [np.std(particles.x)]
for turn in range(10):
    engine.track_single_turn()
    coords = engine.get_particle_coordinates()
    rms_x.append(np.std(coords['x']))

# Calculate variation
variation = (np.max(rms_x) - np.min(rms_x)) / np.mean(rms_x) * 100

if variation < 1.0:
    print(f"✅ Well matched! Variation: {variation:.3f}%")
else:
    print(f"⚠️ Possible mismatch. Variation: {variation:.3f}%")
```

### Check Correlations

```python
# For matched beam without coupling: <x·px> should be ≈ 0
correlation_x = np.mean(particles.x * particles.px)
correlation_y = np.mean(particles.y * particles.py)

print(f"<x·px> = {correlation_x:.2e} (should be ≈ 0)")
print(f"<y·py> = {correlation_y:.2e} (should be ≈ 0)")
```

## Comparison: Matched vs Unmatched

| Property | Unmatched (create_particles) | Matched (create_matched_particles) |
|----------|------------------------------|-------------------------------------|
| Beta functions | User-specified | Lattice Twiss parameters |
| Coupling | Uncoupled Gaussians | Uses 1-turn transfer matrix |
| Dispersion | Not included | Automatically included |
| Closed orbit | Centered at zero | Centered around CO |
| Stability | May oscillate | Stable RMS envelope |
| Use case | Quick tests, mismatched injection | Realistic simulations |

## Example: Full Workflow

See `examples/matched_beam_demo.py` for a complete working example that:

1. Creates a FODO lattice
2. Initializes XSuite engine
3. Creates matched particles
4. Tracks for 10 turns
5. Verifies beam size stability
6. Creates phase space visualizations

Run with:
```bash
uv run python examples/matched_beam_demo.py
```

## Implementation Details

The `create_matched_particles()` method uses XSuite's `generate_matched_gaussian_bunch()` function, which:

1. Calls `line.twiss()` to get lattice optics at creation point
2. Finds closed orbit (for RING mode)
3. Computes 1-turn transfer matrix (R-matrix)
4. Generates particles in normalized coordinates
5. Transforms to physical coordinates using R-matrix
6. Applies dispersion corrections
7. Centers around closed orbit

### Fallback Behavior

If matching fails (e.g., unstable lattice, no closed orbit), the function automatically falls back to the standard `create_particles()` method and logs a warning.

## Troubleshooting

### "Closed orbit search failed"

**Problem**: Your lattice doesn't have a stable closed orbit.

**Solutions**:
- Verify lattice is actually a ring (bending angles sum to 2π)
- Check quadrupole strengths for stability
- Use LINAC mode instead of RING mode
- Fall back to unmatched particles for testing

### "Beam sizes don't match expected values"

**Problem**: Mismatch between actual and expected beam sizes.

**Check**:
1. Units: Are you using geometric emittance (m·rad)?
2. Relativistic factors: Correct γ and β for beam energy?
3. Twiss calculation: Did `calculate_twiss()` succeed?
4. Element location: Is `element_name` correct?

### "Beam size grows over turns"

**Problem**: Beam is not actually matched or lattice is unstable.

**Investigate**:
1. Check lattice stability: `twiss.qx`, `twiss.qy` should be real
2. Verify no strong nonlinearities (high-order multipoles)
3. Check for space charge (not included in linear matching)
4. Confirm sufficient numerical precision (slicing)

## See Also

- [XSuite Particle Generation Documentation](https://xsuite.readthedocs.io/en/latest/particlesmanip.html)
- [Simulation Engines Guide](simulation_engines.md)
- [Lattice Design Guide](lattice_design.md)
- Example: `examples/matched_beam_demo.py`
- Test notebook: `tests/test_fodo.ipynb` (cells on matched particles)
