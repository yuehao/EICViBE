"""
Optics data structures for EICViBE.

Package-level standard data structures for beam optics and dynamics.
Any simulation engine can populate these structures.
"""

from .twiss import SimulationMode, TwissData

__all__ = [
    'SimulationMode',
    'TwissData',
]
