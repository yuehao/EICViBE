"""
Lattice Change Management System for EICViBE

This module provides a comprehensive system for managing lattice parameter changes:
- Cache proposed changes before activation
- Review changes to ensure correctness
- Select elements by name, type, or s-position range
- Activate changes with controlled ramping from old to new values

Typical workflow:
    1. Create cache: cache = LatticeChangeCache(lattice, engine, xsuite_line)
    2. Select elements: cache.select_by_type('Quadrupole')
    3. Propose changes: cache.propose_change('MagneticMultipoleP', 'kn1', new_value)
    4. Review: df = cache.review_changes()
    5. Activate: bpm_data = cache.activate_changes(particles)
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional, Any
import numpy as np
import pandas as pd


class ChangeStatus(Enum):
    """Status of a lattice change in the workflow"""
    PENDING = "pending"      # Proposed but not yet activated
    RAMPING = "ramping"      # Currently being ramped from old to new value
    COMPLETED = "completed"  # Successfully applied and ramping finished
    CANCELLED = "cancelled"  # User cancelled before activation


@dataclass
class LatticeChange:
    """
    Represents a single parameter change for a lattice element.
    
    Attributes:
        element_name: Name of the element to modify
        parameter_group: EICViBE parameter group (e.g., 'MagneticMultipoleP', 'BendP')
        parameter_name: Parameter name within the group (e.g., 'kn1', 'angle')
        old_value: Current parameter value before change
        new_value: Target parameter value after change
        status: Current status of the change
        description: Human-readable description of the change purpose
        ramp_steps: Number of steps for ramping (more steps = smoother transition)
    """
    element_name: str
    parameter_group: str
    parameter_name: str
    old_value: float
    new_value: float
    status: ChangeStatus = ChangeStatus.PENDING
    description: str = ""
    ramp_steps: int = 10
    
    @property
    def change_magnitude(self) -> float:
        """Calculate percentage change from old to new value"""
        if abs(self.old_value) < 1e-12:
            return float('inf') if self.new_value != 0 else 0.0
        return ((self.new_value - self.old_value) / self.old_value) * 100
    
    @property
    def absolute_change(self) -> float:
        """Calculate absolute change magnitude"""
        return self.new_value - self.old_value
    
    def __repr__(self) -> str:
        return (f"LatticeChange({self.element_name}, "
                f"{self.parameter_group}.{self.parameter_name}: "
                f"{self.old_value:.4f} → {self.new_value:.4f}, "
                f"Δ={self.change_magnitude:+.1f}%, "
                f"status={self.status.value})")


class LatticeChangeCache:
    """
    Manages a cache of proposed lattice changes with review and activation capabilities.
    
    This class implements the full workflow for safe lattice parameter modification:
    1. Element selection (by name, type, or s-position range)
    2. Change proposal (stored in cache)
    3. Change review (inspect before applying)
    4. Change activation (apply with ramping to simulation)
    
    The cache stores all proposed changes and their status, allowing users to:
    - Inspect what will be changed before applying
    - Modify or cancel changes before activation
    - Track the history of applied changes
    - Apply changes gradually with controlled ramping
    
    Args:
        lattice: EICViBE Lattice object containing element definitions
        engine: Simulation engine (e.g., XSuiteSimulationEngine)
        xsuite_line: XSuite Line object (converted lattice for tracking)
    
    Example:
        >>> cache = LatticeChangeCache(fodo, engine, xsuite_line)
        >>> cache.select_by_type('Quadrupole')
        >>> cache.propose_change('MagneticMultipoleP', 'kn1', -0.6, 'Increase focusing')
        >>> df = cache.review_changes()  # Inspect before applying
        >>> bpm_data = cache.activate_changes(particles, num_turns_per_step=20)
    """
    
    # Parameter mapping from EICViBE parameter groups to XSuite attributes
    PARAMETER_MAP = {
        'MagneticMultipoleP': {
            'kn1': 'k1',  # Quadrupole normalized gradient
            'kn2': 'k2',  # Sextupole normalized gradient
            'ks1': 'k1s', # Skew quadrupole
            'ks2': 'k2s', # Skew sextupole
        },
        'BendP': {
            'angle': 'h',  # Bending angle (XSuite uses h for curvature)
        },
        'RFP': {
            'voltage': 'voltage',    # RF cavity voltage
            'frequency': 'frequency', # RF cavity frequency
            'lag': 'lag',            # RF phase lag
        },
        'ApertureP': {
            'X': 'max_x',  # Horizontal aperture limit
            'Y': 'max_y',  # Vertical aperture limit
        }
    }
    
    def __init__(self, lattice, engine, xsuite_line):
        """
        Initialize the lattice change cache.
        
        Args:
            lattice: EICViBE Lattice object
            engine: Simulation engine instance
            xsuite_line: XSuite Line object (converted lattice)
        """
        self.lattice = lattice
        self.engine = engine
        self.xsuite_line = xsuite_line
        self.changes: List[LatticeChange] = []
        self._selected_elements: List[str] = []
    
    # ========================================================================
    # ELEMENT SELECTION METHODS
    # ========================================================================
    
    def select_element(self, element_name: str) -> 'LatticeChangeCache':
        """
        Select a single element by its exact name.
        
        Args:
            element_name: Exact name of the element in the lattice
            
        Returns:
            Self for method chaining
            
        Example:
            >>> cache.select_element('Quad1_1')
        """
        if element_name in self.xsuite_line.element_names:
            self._selected_elements = [element_name]
            print(f"Selected element: {element_name}")
        elif element_name in self.xsuite_line.element_dict:
            # Handle sliced elements - element might be in dict but not in names list
            self._selected_elements = [element_name]
            print(f"Selected element: {element_name}")
        else:
            print(f"⚠️ Element '{element_name}' not found in lattice")
            print(f"Available elements: {list(self.xsuite_line.element_dict.keys())[:10]}...")
        return self
    
    def select_by_type(self, element_type: str) -> 'LatticeChangeCache':
        """
        Select all elements of a specific type.
        
        Args:
            element_type: Element type name ('Quadrupole', 'Bend', 'RFCavity', etc.)
            
        Returns:
            Self for method chaining
            
        Example:
            >>> cache.select_by_type('Quadrupole')  # Select all quadrupoles
            >>> cache.select_by_type('Bend')         # Select all bends
        """
        try:
            import xtrack as xt
        except ImportError:
            print("⚠️ XTrack not available - cannot determine element types")
            return self
        
        # Map type names to XTrack classes
        type_map = {
            'Quadrupole': xt.Quadrupole,
            'Bend': xt.Bend,
            'Drift': xt.Drift,
            'RFCavity': xt.Cavity,
            'Monitor': xt.BeamPositionMonitor,
            'Marker': xt.Marker,
            'Sextupole': xt.Sextupole,
            'Octupole': xt.Octupole,
            'Multipole': xt.Multipole,
        }
        
        if element_type not in type_map:
            print(f"⚠️ Unknown element type '{element_type}'")
            print(f"Available types: {list(type_map.keys())}")
            return self
        
        target_class = type_map[element_type]
        
        # Find all elements of this type
        selected = []
        for name, elem in zip(self.xsuite_line.element_names, self.xsuite_line.elements):
            if isinstance(elem, target_class):
                selected.append(name)
        
        # Remove sliced elements (keep only base names)
        # After slicing, "Quad1" becomes "Quad1..0", "Quad1..1", etc.
        base_elements = set()
        for name in selected:
            if '..' in name:
                base_name = name.split('..')[0]
                # Only add if it's the type we want (not drift slices)
                if element_type.lower() in base_name.lower() or 'drift' not in base_name.lower():
                    base_elements.add(base_name)
            else:
                base_elements.add(name)
        
        self._selected_elements = sorted(list(base_elements))
        print(f"Selected {len(self._selected_elements)} {element_type} elements: {self._selected_elements}")
        return self
    
    def select_range(self, s_start: float, s_end: float, 
                    element_type: Optional[str] = None) -> 'LatticeChangeCache':
        """
        Select elements within a range of s positions.
        
        Args:
            s_start: Start position in meters
            s_end: End position in meters
            element_type: Optional filter by element type
            
        Returns:
            Self for method chaining
            
        Example:
            >>> cache.select_range(0, 10.5)  # All elements in first 10.5m
            >>> cache.select_range(5, 15, element_type='Quadrupole')  # Quads in range
        """
        try:
            import xtrack as xt
        except ImportError:
            print("⚠️ XTrack not available")
            return self
        
        selected = []
        for name in self.xsuite_line.element_names:
            s_pos = self.xsuite_line.get_s_position(name)
            if s_start <= s_pos <= s_end:
                # Filter by type if specified
                if element_type:
                    elem = self.xsuite_line.element_dict[name]
                    type_map = {
                        'Quadrupole': xt.Quadrupole,
                        'Bend': xt.Bend,
                        'Sextupole': xt.Sextupole,
                        'RFCavity': xt.Cavity,
                    }
                    if element_type in type_map and isinstance(elem, type_map[element_type]):
                        selected.append(name)
                else:
                    selected.append(name)
        
        # Remove duplicates from sliced elements
        base_elements = set()
        for name in selected:
            if '..' in name:
                base_name = name.split('..')[0]
                base_elements.add(base_name)
            else:
                base_elements.add(name)
        
        self._selected_elements = sorted(list(base_elements))
        print(f"Selected {len(self._selected_elements)} elements in range [{s_start:.2f}, {s_end:.2f}] m:")
        for elem in self._selected_elements[:10]:  # Show first 10
            s_pos = self.xsuite_line.get_s_position(elem)
            print(f"  {elem} at s = {s_pos:.3f} m")
        if len(self._selected_elements) > 10:
            print(f"  ... and {len(self._selected_elements) - 10} more")
        return self
    
    def get_selected_elements(self) -> List[str]:
        """Return the list of currently selected elements"""
        return self._selected_elements.copy()
    
    # ========================================================================
    # CHANGE PROPOSAL METHODS
    # ========================================================================
    
    def propose_change(self, parameter_group: str, parameter_name: str, 
                      new_value: float, description: str = "", 
                      ramp_steps: int = 10) -> 'LatticeChangeCache':
        """
        Propose a parameter change for all currently selected elements.
        
        Changes are stored in the cache but not applied until activate_changes() is called.
        
        Args:
            parameter_group: EICViBE parameter group name (e.g., 'MagneticMultipoleP')
            parameter_name: Parameter name within the group (e.g., 'kn1')
            new_value: Target value for the parameter
            description: Human-readable description of the change purpose
            ramp_steps: Number of ramping steps (default 10)
            
        Returns:
            Self for method chaining
            
        Example:
            >>> cache.select_by_type('Quadrupole')
            >>> cache.propose_change('MagneticMultipoleP', 'kn1', -0.55, 
            ...                      'Increase focusing by 10%', ramp_steps=8)
        """
        if not self._selected_elements:
            print("⚠️ No elements selected. Use select_element(), select_by_type(), or select_range() first.")
            return self
        
        for element_name in self._selected_elements:
            # Get current value
            elem = self.xsuite_line.element_dict[element_name]
            
            # Map parameter group to XSuite attribute
            if parameter_group in self.PARAMETER_MAP:
                param_map = self.PARAMETER_MAP[parameter_group]
                if parameter_name in param_map:
                    xsuite_attr = param_map[parameter_name]
                    if hasattr(elem, xsuite_attr):
                        old_value = getattr(elem, xsuite_attr)
                        
                        # Create change record
                        change = LatticeChange(
                            element_name=element_name,
                            parameter_group=parameter_group,
                            parameter_name=parameter_name,
                            old_value=old_value,
                            new_value=new_value,
                            description=description,
                            ramp_steps=ramp_steps
                        )
                        self.changes.append(change)
                    else:
                        print(f"⚠️ Element {element_name} does not have parameter {xsuite_attr}")
                else:
                    print(f"⚠️ Unknown parameter {parameter_name} in group {parameter_group}")
            else:
                print(f"⚠️ Unknown parameter group {parameter_group}")
        
        print(f"✓ Proposed {len(self._selected_elements)} changes")
        return self
    
    # ========================================================================
    # REVIEW METHODS
    # ========================================================================
    
    def review_changes(self, status: Optional[ChangeStatus] = None) -> pd.DataFrame:
        """
        Review all cached changes as a pandas DataFrame.
        
        Args:
            status: Optional filter by change status (PENDING, COMPLETED, etc.)
            
        Returns:
            DataFrame with columns: ID, Element, Parameter, Old Value, New Value, 
            Change (%), Status, Description
            
        Example:
            >>> df = cache.review_changes()  # All changes
            >>> df = cache.review_changes(status=ChangeStatus.PENDING)  # Only pending
        """
        if not self.changes:
            print("No changes in cache")
            return pd.DataFrame()
        
        # Filter by status if specified
        changes_to_show = self.changes
        if status:
            changes_to_show = [c for c in self.changes if c.status == status]
        
        # Create DataFrame
        data = []
        for i, change in enumerate(changes_to_show):
            data.append({
                'ID': i,
                'Element': change.element_name,
                'Parameter': f"{change.parameter_group}.{change.parameter_name}",
                'Old Value': f"{change.old_value:.6f}",
                'New Value': f"{change.new_value:.6f}",
                'Change (%)': f"{change.change_magnitude:+.2f}",
                'Ramp Steps': change.ramp_steps,
                'Status': change.status.value,
                'Description': change.description
            })
        
        df = pd.DataFrame(data)
        print(f"\n{'='*80}")
        print(f"Change Summary: {len(changes_to_show)} changes")
        print(f"{'='*80}")
        return df
    
    def get_changes(self, status: Optional[ChangeStatus] = None) -> List[LatticeChange]:
        """
        Get list of LatticeChange objects (optionally filtered by status).
        
        Args:
            status: Optional filter by change status
            
        Returns:
            List of LatticeChange objects
        """
        if status:
            return [c for c in self.changes if c.status == status]
        return self.changes.copy()
    
    def clear_changes(self, status: Optional[ChangeStatus] = None):
        """
        Clear changes from cache.
        
        Args:
            status: Optional filter - only clear changes with this status
            
        Example:
            >>> cache.clear_changes()  # Clear all
            >>> cache.clear_changes(status=ChangeStatus.PENDING)  # Clear only pending
        """
        if status:
            self.changes = [c for c in self.changes if c.status != status]
            print(f"✓ Cleared changes with status: {status.value}")
        else:
            self.changes.clear()
            print("✓ Cleared all changes from cache")
    
    # ========================================================================
    # ACTIVATION METHODS
    # ========================================================================
    
    def activate_changes(self, particles, num_turns_per_step: int = 10, 
                        track_during_ramp: bool = True) -> Dict[str, Any]:
        """
        Activate all pending changes with controlled ramping.
        
        This method applies all pending changes by:
        1. Grouping changes by element for coordinated ramping
        2. Ramping parameters linearly from old to new values over multiple steps
        3. Tracking particles at each ramp step to observe beam response
        4. Collecting BPM data during the ramping process
        5. Marking changes as COMPLETED when done
        
        Args:
            particles: Particle beam to track during ramping
            num_turns_per_step: Number of turns to track at each ramp step
            track_during_ramp: Whether to track particles during ramp (default True)
            
        Returns:
            Dictionary containing BPM data during ramping with keys:
            - 'turns': Turn numbers
            - 'x_mean': Horizontal centroid [mm]
            - 'y_mean': Vertical centroid [mm]
            - 'step': Ramp step number
            
        Example:
            >>> bpm_data = cache.activate_changes(particles, num_turns_per_step=20)
            >>> # Plot bpm_data to see beam response during ramping
        """
        pending_changes = [c for c in self.changes if c.status == ChangeStatus.PENDING]
        
        if not pending_changes:
            print("No pending changes to activate")
            return {}
        
        print(f"\n{'='*80}")
        print(f"Activating {len(pending_changes)} changes with ramping")
        print(f"{'='*80}")
        
        # Group changes by element for coordinated ramping
        changes_by_element: Dict[str, List[LatticeChange]] = {}
        for change in pending_changes:
            if change.element_name not in changes_by_element:
                changes_by_element[change.element_name] = []
            changes_by_element[change.element_name].append(change)
        
        # Determine maximum ramp steps needed
        max_ramp_steps = max(c.ramp_steps for c in pending_changes)
        
        # BPM data collection
        bpm_ramp_data = {
            'turns': [], 
            'x_mean': [], 
            'y_mean': [], 
            'step': [],
            'element_states': []  # Track parameter values at each step
        }
        current_turn = 0
        
        # Perform ramping
        for step in range(max_ramp_steps + 1):
            ramp_fraction = step / max_ramp_steps
            
            print(f"\nRamp step {step}/{max_ramp_steps} ({ramp_fraction*100:.0f}%)")
            
            # Update all element parameters for this ramp step
            for element_name, element_changes in changes_by_element.items():
                elem = self.xsuite_line.element_dict[element_name]
                
                for change in element_changes:
                    if step <= change.ramp_steps:
                        # Linear interpolation between old and new value
                        current_value = (change.old_value + 
                                       ramp_fraction * (change.new_value - change.old_value))
                        
                        # Apply to element
                        if change.parameter_group in self.PARAMETER_MAP:
                            param_map = self.PARAMETER_MAP[change.parameter_group]
                            if change.parameter_name in param_map:
                                xsuite_attr = param_map[change.parameter_name]
                                setattr(elem, xsuite_attr, current_value)
                        
                        change.status = ChangeStatus.RAMPING
                        
                        print(f"  {element_name}.{change.parameter_name} = {current_value:.6f}")
            
            # Track particles at this ramp step
            if track_during_ramp and step < max_ramp_steps:
                self.xsuite_line.track(particles, num_turns=num_turns_per_step)
                
                # Collect BPM data
                bpm_data = self.engine.get_bpm_data()
                if bpm_data:
                    first_bpm = list(bpm_data.keys())[0]
                    if first_bpm in bpm_data:
                        bpm = bpm_data[first_bpm]
                        for i in range(num_turns_per_step):
                            turn_idx = current_turn + i
                            if turn_idx < len(bpm.x_mean):
                                bpm_ramp_data['turns'].append(turn_idx)
                                bpm_ramp_data['x_mean'].append(bpm.x_mean[turn_idx] * 1e3)  # Convert to mm
                                bpm_ramp_data['y_mean'].append(bpm.y_mean[turn_idx] * 1e3)
                                bpm_ramp_data['step'].append(step)
                
                current_turn += num_turns_per_step
        
        # Mark all changes as completed
        for change in pending_changes:
            change.status = ChangeStatus.COMPLETED
        
        print(f"\n{'='*80}")
        print(f"✓ Ramping completed: All changes now active")
        print(f"✓ Tracked {current_turn} turns during {max_ramp_steps} ramp steps")
        surviving = np.sum(particles.state > 0) if hasattr(particles, 'state') else len(particles.x)
        total = len(particles.x)
        print(f"✓ Surviving particles: {surviving}/{total} ({surviving/total*100:.1f}%)")
        print(f"{'='*80}")
        
        return bpm_ramp_data
