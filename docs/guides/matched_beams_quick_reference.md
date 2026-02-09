# Matched Beam Quick Reference

## When to Use

**Use `create_matched_particles()` when:**
- Simulating realistic accelerator operation
- Need stable beam envelope over many turns
- Studying collective effects (space charge, impedance)
- Comparing with experimental measurements

**Use `create_particles()` when:**
- Quick testing with simple distributions
- Intentionally mismatched injection studies
- Testing lattice response to large oscillations

## Basic Usage

```python
from eicvibe.simulators.xsuite_interface import XSuiteSimulationEngine
from eicvibe.simulators.types import ParticleDistribution, DistributionType

# 1. Setup engine
engine = XSuiteSimulationEngine()
engine.initialize_engine()
converted = engine.convert_lattice(lattice, "main")

# 2. Define distribution
distribution = ParticleDistribution(
    num_particles=10000,
    energy=10e9,  # 10 GeV
    distribution_type=DistributionType.GAUSSIAN,
    emittance_x=5e-6,  # m·rad (geometric)
    emittance_y=2e-6,  # m·rad (geometric)
    beta_x=10.0,  # IGNORED when use_lattice_twiss=True
    beta_y=5.0,   # IGNORED when use_lattice_twiss=True
    bunch_length=1e-3,
    energy_spread=1e-4
)

# 3. Create matched particles
particles = engine.create_matched_particles(
    distribution,
    use_lattice_twiss=True  # Use lattice beta, not distribution beta
)
```

## Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `distribution` | ParticleDistribution | Required | Beam parameters (energy, emittance, etc.) |
| `element_name` | str or None | None | Creation point (None = start of lattice) |
| `nemitt_x` | float or None | None | Normalized emittance X (auto-calculated if None) |
| `nemitt_y` | float or None | None | Normalized emittance Y (auto-calculated if None) |
| `use_lattice_twiss` | bool | True | Use lattice beta (True) or distribution beta (False) |

## Verification Checklist

```python
import numpy as np

# ✅ 1. Check correlations (should be ≈ 0)
print(f"<x·px> = {np.mean(particles.x * particles.px):.2e}")
print(f"<y·py> = {np.mean(particles.y * particles.py):.2e}")
# Good: ~10⁻⁸ or smaller

# ✅ 2. Track and check stability (exclude first 2 turns)
rms_x = []
for turn in range(10):
    engine.track_single_turn()
    coords = engine.get_particle_coordinates()
    rms_x.append(np.std(coords['x']))

# Calculate variation after turn 2
variation = (max(rms_x[2:]) - min(rms_x[2:])) / np.mean(rms_x[2:]) * 100
print(f"Beam size variation (turn 2-10): {variation:.2f}%")
# Good: <1% variation

# ✅ 3. Compare with expected beam size
from math import sqrt
gamma_rel = distribution.energy / 938.27208816e6
beta_rel = sqrt(1 - 1/gamma_rel**2)
twiss = engine.get_twiss(converted, distribution.energy, SimulationMode.RING)
nemitt_x = distribution.emittance_x * beta_rel * gamma_rel
sigma_x_expected = sqrt(nemitt_x * twiss.beta_x[0] / (beta_rel * gamma_rel))
sigma_x_actual = np.std(particles.x)
print(f"Expected σ_x: {sigma_x_expected*1e3:.3f} mm")
print(f"Actual σ_x: {sigma_x_actual*1e3:.3f} mm")
# Should be within factor of ~2
```

## Common Issues & Solutions

### Issue: "Line must be initialized"
**Solution**: Call `convert_lattice()` before `create_matched_particles()`

### Issue: "Closed orbit search failed"  
**Solution**: 
- Check lattice stability (bending angles sum to 2π for rings)
- Verify quadrupole strengths
- Use LINAC mode if lattice is not a ring

### Issue: Large beam size variation
**Check**:
- Are you excluding first 1-2 turns from stability calculation?
- Is lattice actually stable? (check Twiss tunes)
- Are there strong nonlinear elements?

### Issue: Beam sizes don't match formula
**Note**: XSuite includes dispersion and closed orbit offsets that simple formulas don't account for. As long as:
1. Correlations <x·px> ≈ 0 ✅
2. Beam stable after turn 2 ✅  
The beam IS matched!

## Expected Behavior

**Initial filamentation** (turns 0-2):
```
Turn 0: σ_x = 7.05 mm, σ_y = 3.15 mm  ← Initial
Turn 1: σ_x = 2.91 mm, σ_y = 8.36 mm  ← Oscillates
Turn 2: σ_x = 10.84 mm, σ_y = 5.45 mm ← Stabilizes
```

**Stable phase** (turns 2+):
```
Turn 3: σ_x = 10.84 mm, σ_y = 5.45 mm ← No change
Turn 4: σ_x = 10.84 mm, σ_y = 5.45 mm ← No change
Turn 5: σ_x = 10.84 mm, σ_y = 5.45 mm ← No change
...
```

This is **correct** - matched beams filamentation during first betatron period, then stabilize.

## Physics Formulas

**Beam size** (matched beam):
```
σ_x = √(ε_x * β_x)
σ_px = √(ε_x / β_x)
```

**Emittance conversion**:
```
ε_norm = ε_geom × β_rel × γ_rel
where:
  γ_rel = E_beam / m_proton c²
  β_rel = √(1 - 1/γ²)
```

**Relativistic factors** (10 GeV proton):
```
γ_rel = 10×10⁹ / 938.27×10⁶ = 10.657
β_rel = 0.9956
```

## Examples

**See**:
- Full demo: `examples/matched_beam_demo.py`
- Notebook: `tests/test_fodo.ipynb` (matched beam cells)
- Documentation: `docs/guides/matched_beams.md`

**Run demo**:
```bash
uv run python examples/matched_beam_demo.py
```

## Comparison: Matched vs Unmatched

| Property | create_particles() | create_matched_particles() |
|----------|-------------------|----------------------------|
| Beta source | User input | Lattice Twiss |
| Phase space | Uncoupled | Coupled via R-matrix |
| Dispersion | No | Yes |
| Closed orbit | Zero | Actual CO |
| RMS stability | May oscillate | Stable after ~2 turns |
| Use case | Testing | Production |

## Function Signature

```python
def create_matched_particles(
    self,
    distribution: ParticleDistribution,
    element_name: Optional[str] = None,
    nemitt_x: Optional[float] = None,
    nemitt_y: Optional[float] = None,
    use_lattice_twiss: bool = True
) -> xpart.Particles
```

**Returns**: `xpart.Particles` object matched to lattice optics

**Raises**: `TrackingError` if line not initialized

**Fallback**: Automatically uses `create_particles()` if matching fails
