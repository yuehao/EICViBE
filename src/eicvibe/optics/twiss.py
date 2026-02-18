"""
Twiss parameter data structures for beam optics.

Package-level standard data structures that any simulation engine can populate.
"""

from enum import Enum
from typing import Optional
from pydantic import Field
import numpy as np

from ..models.base import PhysicsBaseModel


class SimulationMode(str, Enum):
    """
    Supported simulation modes for accelerator physics simulations.

    Each mode represents a different approach to particle tracking and
    beam dynamics analysis suitable for different accelerator types.
    """

    LINAC = "linac"  # Linear accelerator single-pass tracking
    RING = "ring"  # Circular multi-turn tracking with periodic boundary
    RAMPING = "ramping"  # Time-dependent parameter evolution


class TwissData(PhysicsBaseModel):
    """
    Package-level standard for Twiss parameter data.

    This is an EICViBE-level data structure that any simulation engine
    can populate. It provides a standardized interface for optical functions
    independent of the specific simulation engine used.

    Design Philosophy:
    - Engine-agnostic: Any engine (XSuite, MAD-X, Elegant, etc.) can fill this
    - Comprehensive: Supports all common optical functions
    - Flexible: Optional fields for engine-specific advanced features
    - Validated: Pydantic ensures data consistency

    Contains all optical functions calculated along the lattice,
    with support for different simulation modes and coordinate systems.
    """

    # ========== Core Longitudinal Positions ==========
    s: np.ndarray = Field(description="Longitudinal positions along lattice in m")

    # ========== Beta Functions (Core Optical Functions) ==========
    beta_x: np.ndarray = Field(description="Horizontal beta function in m")
    beta_y: np.ndarray = Field(description="Vertical beta function in m")

    # ========== Alpha Functions ==========
    alpha_x: np.ndarray = Field(description="Horizontal alpha function")
    alpha_y: np.ndarray = Field(description="Vertical alpha function")

    # ========== Gamma Functions ==========
    gamma_x: Optional[np.ndarray] = Field(
        default=None, description="Horizontal gamma function in m^-1"
    )
    gamma_y: Optional[np.ndarray] = Field(
        default=None, description="Vertical gamma function in m^-1"
    )

    # ========== Phase Advance ==========
    mu_x: Optional[np.ndarray] = Field(
        default=None, description="Horizontal phase advance in radians"
    )
    mu_y: Optional[np.ndarray] = Field(
        default=None, description="Vertical phase advance in radians"
    )

    # ========== Dispersion Functions ==========
    dx: Optional[np.ndarray] = Field(
        default=None, description="Horizontal dispersion in m"
    )
    dy: Optional[np.ndarray] = Field(
        default=None, description="Vertical dispersion in m"
    )
    dpx: Optional[np.ndarray] = Field(
        default=None, description="Horizontal dispersion derivative"
    )
    dpy: Optional[np.ndarray] = Field(
        default=None, description="Vertical dispersion derivative"
    )

    # ========== Higher-Order Dispersion ==========
    ddx: Optional[np.ndarray] = Field(
        default=None, description="Second-order horizontal dispersion in m"
    )
    ddy: Optional[np.ndarray] = Field(
        default=None, description="Second-order vertical dispersion in m"
    )
    ddpx: Optional[np.ndarray] = Field(
        default=None, description="Second-order horizontal dispersion derivative"
    )
    ddpy: Optional[np.ndarray] = Field(
        default=None, description="Second-order vertical dispersion derivative"
    )

    # ========== Closed Orbit (RING Mode) ==========
    x: Optional[np.ndarray] = Field(
        default=None, description="Horizontal closed orbit in m"
    )
    px: Optional[np.ndarray] = Field(
        default=None, description="Horizontal momentum in closed orbit"
    )
    y: Optional[np.ndarray] = Field(
        default=None, description="Vertical closed orbit in m"
    )
    py: Optional[np.ndarray] = Field(
        default=None, description="Vertical momentum in closed orbit"
    )
    zeta: Optional[np.ndarray] = Field(
        default=None, description="Longitudinal closed orbit position in m"
    )
    delta: Optional[np.ndarray] = Field(
        default=None, description="Relative energy deviation in closed orbit"
    )

    # ========== Energy Parameters ==========
    energy: Optional[np.ndarray] = Field(
        default=None, description="Beam energy along lattice in eV"
    )
    ptau: Optional[np.ndarray] = Field(
        default=None, description="Longitudinal momentum deviation"
    )

    # ========== Amplitude-Dependent Chromaticity ==========
    ax_chrom: Optional[np.ndarray] = Field(
        default=None, description="Horizontal amplitude-dependent chromaticity"
    )
    ay_chrom: Optional[np.ndarray] = Field(
        default=None, description="Vertical amplitude-dependent chromaticity"
    )
    bx_chrom: Optional[np.ndarray] = Field(
        default=None, description="Horizontal second-order amplitude chromaticity"
    )
    by_chrom: Optional[np.ndarray] = Field(
        default=None, description="Vertical second-order amplitude chromaticity"
    )

    # ========== W Functions (Chromaticity) ==========
    wx: Optional[np.ndarray] = Field(
        default=None, description="Horizontal W function for chromaticity"
    )
    wy: Optional[np.ndarray] = Field(
        default=None, description="Vertical W function for chromaticity"
    )
    wx_chrom: Optional[np.ndarray] = Field(
        default=None, description="Horizontal W-function chromaticity"
    )
    wy_chrom: Optional[np.ndarray] = Field(
        default=None, description="Vertical W-function chromaticity"
    )

    # ========== Coupling Parameters (4D Coupling) ==========
    c_minus: Optional[np.ndarray] = Field(
        default=None, description="Coupling coefficient C-"
    )
    c_plus: Optional[np.ndarray] = Field(
        default=None, description="Coupling coefficient C+"
    )

    # ========== Transfer Matrix Elements (R-matrix) ==========
    # First-order (2×2 blocks for x and y)
    r11: Optional[np.ndarray] = Field(
        default=None, description="R11 transfer matrix element (x|x)"
    )
    r12: Optional[np.ndarray] = Field(
        default=None, description="R12 transfer matrix element (x|px)"
    )
    r21: Optional[np.ndarray] = Field(
        default=None, description="R21 transfer matrix element (px|x)"
    )
    r22: Optional[np.ndarray] = Field(
        default=None, description="R22 transfer matrix element (px|px)"
    )
    r33: Optional[np.ndarray] = Field(
        default=None, description="R33 transfer matrix element (y|y)"
    )
    r34: Optional[np.ndarray] = Field(
        default=None, description="R34 transfer matrix element (y|py)"
    )
    r43: Optional[np.ndarray] = Field(
        default=None, description="R43 transfer matrix element (py|y)"
    )
    r44: Optional[np.ndarray] = Field(
        default=None, description="R44 transfer matrix element (py|py)"
    )

    # ========== Longitudinal Dynamics (RING Mode) ==========
    momentum_compaction: Optional[float] = Field(
        default=None, description="Momentum compaction factor (alpha_c)"
    )
    slip_factor: Optional[float] = Field(
        default=None, description="Phase slip factor (eta)"
    )
    T_rev: Optional[float] = Field(
        default=None, description="Revolution period in seconds"
    )
    f_rev: Optional[float] = Field(
        default=None, description="Revolution frequency in Hz"
    )

    # ========== Mode-Specific Metadata ==========
    simulation_mode: SimulationMode = Field(
        description="Simulation mode used for calculation"
    )
    reference_energy: float = Field(description="Reference energy in eV")
    reference_momentum: Optional[float] = Field(
        default=None, description="Reference momentum p0c in eV"
    )
    particle_mass: Optional[float] = Field(
        default=None, description="Particle rest mass in eV"
    )
    particle_charge: Optional[int] = Field(
        default=None, description="Particle charge in elementary charges"
    )

    # ========== Tune Information (RING Mode) ==========
    tune_x: Optional[float] = Field(default=None, description="Horizontal tune (Qx)")
    tune_y: Optional[float] = Field(default=None, description="Vertical tune (Qy)")
    tune_z: Optional[float] = Field(default=None, description="Synchrotron tune (Qs)")

    # XSuite-compatible aliases
    @property
    def qx(self) -> Optional[float]:
        """Alias for tune_x (XSuite compatibility)."""
        return self.tune_x

    @property
    def qy(self) -> Optional[float]:
        """Alias for tune_y (XSuite compatibility)."""
        return self.tune_y

    @property
    def qs(self) -> Optional[float]:
        """Alias for tune_z (XSuite compatibility)."""
        return self.tune_z

    @property
    def betx(self) -> np.ndarray:
        """Alias for beta_x (XSuite compatibility)."""
        return self.beta_x

    @property
    def bety(self) -> np.ndarray:
        """Alias for beta_y (XSuite compatibility)."""
        return self.beta_y

    @property
    def alfx(self) -> np.ndarray:
        """Alias for alpha_x (XSuite compatibility)."""
        return self.alpha_x

    @property
    def alfy(self) -> np.ndarray:
        """Alias for alpha_y (XSuite compatibility)."""
        return self.alpha_y

    @property
    def momentum_compaction_factor(self) -> Optional[float]:
        """Alias for momentum_compaction (XSuite compatibility)."""
        return self.momentum_compaction

    # ========== Chromaticity (RING Mode) ==========
    chromaticity_x: Optional[float] = Field(
        default=None, description="Horizontal chromaticity (dQx/dp)"
    )
    chromaticity_y: Optional[float] = Field(
        default=None, description="Vertical chromaticity (dQy/dp)"
    )

    # Second-order chromaticity
    chromaticity_x2: Optional[float] = Field(
        default=None, description="Second-order horizontal chromaticity"
    )
    chromaticity_y2: Optional[float] = Field(
        default=None, description="Second-order vertical chromaticity"
    )

    # ========== Additional Lattice Information ==========
    element_names: Optional[list] = Field(
        default=None, description="Names of elements at each s position"
    )
    element_types: Optional[list] = Field(
        default=None, description="Types of elements at each s position"
    )

    # ========== Computation Metadata ==========
    engine_name: Optional[str] = Field(
        default=None, description="Name of engine that computed this data"
    )
    computation_method: Optional[str] = Field(
        default=None, description="Method used (e.g., '4d', '6d')"
    )
    timestamp: Optional[str] = Field(
        default=None, description="When this data was computed"
    )

    def summary(self) -> str:
        """Generate a human-readable summary of the Twiss data."""
        lines = [
            "Twiss Parameter Summary",
            "=" * 50,
            f"Simulation Mode: {self.simulation_mode.value}",
            f"Reference Energy: {self.reference_energy / 1e9:.3f} GeV",
            f"Lattice Length: {self.s[-1]:.2f} m",
            f"Number of Points: {len(self.s)}",
        ]

        if self.tune_x is not None:
            lines.append(f"\nTunes:")
            lines.append(f"  Qx = {self.tune_x:.6f}")
            lines.append(f"  Qy = {self.tune_y:.6f}")
            if self.tune_z is not None:
                lines.append(f"  Qs = {self.tune_z:.6f}")

        lines.append(f"\nBeta Functions:")
        lines.append(
            f"  βx: min = {np.min(self.beta_x):.2f} m, max = {np.max(self.beta_x):.2f} m"
        )
        lines.append(
            f"  βy: min = {np.min(self.beta_y):.2f} m, max = {np.max(self.beta_y):.2f} m"
        )

        if self.dx is not None:
            lines.append(f"\nDispersion:")
            lines.append(
                f"  Dx: min = {np.min(self.dx):.3f} m, max = {np.max(self.dx):.3f} m"
            )

        if self.chromaticity_x is not None:
            lines.append(f"\nChromaticity:")
            lines.append(f"  ξx = {self.chromaticity_x:.3f}")
            lines.append(f"  ξy = {self.chromaticity_y:.3f}")

        if self.momentum_compaction is not None:
            lines.append(f"\nLongitudinal:")
            lines.append(f"  αc = {self.momentum_compaction:.6f}")
            if self.f_rev is not None:
                lines.append(f"  frev = {self.f_rev / 1e6:.3f} MHz")

        if self.engine_name:
            lines.append(f"\nComputed by: {self.engine_name}")
            if self.computation_method:
                lines.append(f"Method: {self.computation_method}")

        return "\n".join(lines)
