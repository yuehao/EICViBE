"""
EICViBE - Electron Ion Collider Virtual Beamline Emulator

A modular framework for accelerator physics simulations.
"""

# Import package-level standards for optics
from .optics import TwissData, SimulationMode

__version__ = "0.1.0"

__all__ = [
    'TwissData',
    'SimulationMode',
]


def main() -> None:
    print("Hello from eicvibe!")
