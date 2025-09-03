"""
Utility to import a MAD-X lattice file using cpymad and convert it to an EICViBE Lattice object.
"""
from cpymad.madx import Madx
from eicvibe.machine_portal.lattice import Lattice, create_element_by_type
from eicvibe.machine_portal.parameter_group import ParameterGroup
import yaml
import os


def load_element_types_yaml():
    """Load the element type mapping from element_types.yaml in utilities."""
    yaml_path = os.path.join(os.path.dirname(__file__), 'element_types.yaml')
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"element_types.yaml not found at {yaml_path}")
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data


def build_madx_to_eicvibe_mapping(element_types_yaml: dict) -> dict:
    """Build a mapping from MAD-X type to EICViBE type and parameter mappings using element_types.yaml."""
    madx_mapping = {}
    for eicvibe_type, info in element_types_yaml.items():
        madx_info = info.get('madx', {})
        if isinstance(madx_info, dict) and 'type' in madx_info:
            madx_type = madx_info['type'].lower()
            madx_mapping[madx_type] = {
                'eicvibe_type': eicvibe_type,
                'parameters': madx_info.get('parameters', {})
            }
    return madx_mapping


def madx_to_eicvibe_type(madx_type: str, madx_mapping: dict) -> str:
    """Map MAD-X element type to EICViBE type using the mapping from element_types.yaml."""
    madx_type = madx_type.lower()
    mapping_info = madx_mapping.get(madx_type, {})
    return mapping_info.get('eicvibe_type', madx_type.capitalize())


def get_madx_element_type(madx_element):
    """Get MAD-X element type from element.parent.name only."""
    # Only use parent.name for type detection, ignore other attributes
    if hasattr(madx_element, 'parent') and madx_element.parent is not None:
        parent_name = madx_element.parent.name
        if parent_name:
            return parent_name.lower()
    
    # Fallback: assume drift for elements with length, marker otherwise
    length = getattr(madx_element, 'l', 0)
    return "drift" if length > 0 else "marker"


def map_madx_parameters(madx_element, eicvibe_element, madx_mapping: dict):
    """Map MAD-X element parameters to EICViBE parameter groups based on the mapping.
    Avoids known MAD-X default values that weren't explicitly set."""
    madx_type = get_madx_element_type(madx_element)
    mapping_info = madx_mapping.get(madx_type, {})
    parameter_mappings = mapping_info.get('parameters', {})
    
    # Known MAD-X default values to skip (these appear even when not explicitly set)
    MADX_DEFAULTS_TO_SKIP = {
        'fintx': -1.0,  # Default fringe field integral for exit
        'fint': -1.0,   # Default fringe field integral for entrance  
        'hgap': 0.0,    # Default half gap
        'h1': 0.0,      # Default entrance pole face curvature
        'h2': 0.0,      # Default exit pole face curvature
    }
    
    for madx_param, eicvibe_path in parameter_mappings.items():
        # Get the value from MAD-X element
        madx_value = getattr(madx_element, madx_param.lower(), None)
        
        # Skip if None, zero, or matches known default values
        if (madx_value is not None and 
            madx_value != 0 and 
            madx_value != MADX_DEFAULTS_TO_SKIP.get(madx_param.lower())):
            
            if eicvibe_path == 'length':
                # Special case: length is a direct attribute, not in a parameter group
                eicvibe_element.set_length(madx_value)
            else:
                # Parse the parameter group and parameter name
                if '.' in eicvibe_path:
                    group_type, param_name = eicvibe_path.split('.', 1)
                    eicvibe_element.add_parameter(group_type, param_name, madx_value)
                else:
                    # If no dot, assume it's a top-level parameter in a default group
                    eicvibe_element.add_parameter('MetaP', eicvibe_path, madx_value)


def extract_base_name(element_name: str) -> str:
    """Extract base name from MAD-X element name by removing _N suffix pattern."""
    # Handle common MAD-X naming patterns like: name, name_1, name_2, etc.
    if '_' in element_name:
        parts = element_name.split('_')
        if len(parts) >= 2 and parts[-1].isdigit():
            return '_'.join(parts[:-1])
    return element_name


def consolidate_drifts(element_pool: dict, branches: dict, tolerance: float = 1e-10) -> tuple[dict, dict]:
    """
    Consolidate drift elements by combining continuous drifts and identifying repeating patterns.
    Uses pattern-based consolidation to properly group identical drift sequences.
    
    Returns:
        tuple: (consolidated_element_pool, updated_branches)
    """
    print("Starting drift consolidation...")
    
    # Step 1: Group individual drift elements by base name and length
    drift_groups_by_name = {}
    drift_groups_by_length = {}
    
    for name, element in element_pool.items():
        if type(element).__name__ == 'Drift':
            base_name = extract_base_name(name)
            length = element.length
            
            # Group by base name first (priority)
            if base_name not in drift_groups_by_name:
                drift_groups_by_name[base_name] = []
            drift_groups_by_name[base_name].append((name, element))
            
            # Also group by length for fallback
            length_key = round(length / tolerance) * tolerance
            if length_key not in drift_groups_by_length:
                drift_groups_by_length[length_key] = []
            drift_groups_by_length[length_key].append((name, element))
    
    print(f"Found {len(drift_groups_by_name)} unique drift base names")
    print(f"Found {len(drift_groups_by_length)} unique drift lengths")
    
    # Step 2: Create consolidated drift definitions from individual drifts
    consolidated_drifts = {}
    drift_mapping = {}  # Maps original name -> consolidated name
    
    # Process name-based groups first
    for base_name, drift_list in drift_groups_by_name.items():
        if len(drift_list) > 1:
            # Multiple drifts with same base name - consolidate
            consolidated_name = f"DRIFT_{base_name.upper()}"
            # Use the first drift as template, they should have same length
            template_element = drift_list[0][1]
            consolidated_drifts[consolidated_name] = template_element
            
            # Map all variants to consolidated name
            for orig_name, _ in drift_list:
                drift_mapping[orig_name] = consolidated_name
        else:
            # Single drift with unique base name, keep as is but still map it
            orig_name = drift_list[0][0]
            drift_mapping[orig_name] = orig_name
    
    # Step 3: Process remaining drifts by length (fallback for those not grouped by name)
    ungrouped_drifts = []
    for name, element in element_pool.items():
        if type(element).__name__ == 'Drift' and name not in drift_mapping:
            ungrouped_drifts.append((name, element))
    
    if ungrouped_drifts:
        print(f"Processing {len(ungrouped_drifts)} ungrouped drifts by length...")
        length_groups = {}
        for name, element in ungrouped_drifts:
            length_key = round(element.length / tolerance) * tolerance
            if length_key not in length_groups:
                length_groups[length_key] = []
            length_groups[length_key].append((name, element))
        
        for length_key, drift_list in length_groups.items():
            if len(drift_list) > 1:
                consolidated_name = f"DRIFT_MERGED_L{length_key:.6f}".replace('.', '_')
                template_element = drift_list[0][1]
                consolidated_drifts[consolidated_name] = template_element
                
                for orig_name, _ in drift_list:
                    drift_mapping[orig_name] = consolidated_name
            else:
                orig_name = drift_list[0][0]
                drift_mapping[orig_name] = orig_name
    
    # Step 4: Process branches to merge consecutive drifts using pattern-based naming
    updated_branches = {}
    merged_drift_patterns = {}  # Maps (pattern_signature, total_length) -> merged_drift_name
    pattern_counter = 0
    
    for branch_name, element_names in branches.items():
        new_element_names = []
        i = 0
        
        while i < len(element_names):
            current_name = element_names[i]
            current_element = element_pool[current_name]
            
            if type(current_element).__name__ == 'Drift':
                # Start of drift sequence - collect consecutive drifts
                drift_sequence = [current_name]
                j = i + 1
                
                while j < len(element_names):
                    next_name = element_names[j]
                    next_element = element_pool[next_name]
                    
                    if type(next_element).__name__ == 'Drift':
                        drift_sequence.append(next_name)
                        j += 1
                    else:
                        break
                
                if len(drift_sequence) > 1:
                    # Multiple consecutive drifts - merge them using pattern-based approach
                    total_length = sum(element_pool[name].length for name in drift_sequence)
                    
                    # Create pattern signature based on drift names and surrounding elements
                    mapped_sequence = [drift_mapping.get(name, name) for name in drift_sequence]
                    element_before = "START" if i == 0 else element_names[i-1]
                    element_after = "END" if (i + len(drift_sequence)) >= len(element_names) else element_names[i + len(drift_sequence)]
                    
                    # Create pattern signature: (before_element, drift_pattern, after_element, total_length)
                    pattern_signature = (element_before, tuple(mapped_sequence), element_after, round(total_length / tolerance) * tolerance)
                    
                    # Check if this exact pattern has been seen before
                    if pattern_signature in merged_drift_patterns:
                        # Reuse existing merged drift
                        merged_name = merged_drift_patterns[pattern_signature]
                    else:
                        # Create new merged drift with pattern-based name
                        pattern_counter += 1
                        merged_name = f"DRIFT_MERGED_P{pattern_counter:03d}_L{total_length:.3f}".replace('.', '_')
                        
                        # Create merged drift element
                        from eicvibe.machine_portal.lattice import create_element_by_type
                        merged_element = create_element_by_type(
                            element_type='Drift',
                            name=merged_name,
                            length=total_length
                        )
                        consolidated_drifts[merged_name] = merged_element
                        merged_drift_patterns[pattern_signature] = merged_name
                    
                    new_element_names.append(merged_name)
                else:
                    # Single drift - use existing mapping
                    mapped_name = drift_mapping[current_name]
                    new_element_names.append(mapped_name)
                
                i = j  # Skip processed drifts
            else:
                # Non-drift element - keep as is
                new_element_names.append(current_name)
                i += 1
        
        updated_branches[branch_name] = new_element_names
    
    # Step 5: Build consolidated element pool
    consolidated_pool = {}
    
    # Add all non-drift elements (preserve originals)
    for name, element in element_pool.items():
        if type(element).__name__ != 'Drift':
            consolidated_pool[name] = element
    
    # Add consolidated drift elements (both individual and merged)
    consolidated_pool.update(consolidated_drifts)
    
    # Add individual drifts that weren't consolidated (mapped to themselves)
    for orig_name, mapped_name in drift_mapping.items():
        if mapped_name == orig_name:  # Not consolidated, keep original
            consolidated_pool[orig_name] = element_pool[orig_name]
    
    # Calculate and display statistics
    original_drift_count = sum(1 for elem in element_pool.values() if type(elem).__name__ == 'Drift')
    consolidated_drift_count = sum(1 for elem in consolidated_pool.values() if type(elem).__name__ == 'Drift')
    individual_consolidated = len([name for name, mapped in drift_mapping.items() if name != mapped])
    merged_patterns = len(merged_drift_patterns)
    
    print(f"Drift consolidation complete:")
    print(f"  Original drifts: {original_drift_count}")
    print(f"  Individual drifts consolidated: {individual_consolidated}")
    print(f"  Unique merged drift patterns: {merged_patterns}")
    print(f"  Final drift count: {consolidated_drift_count}")
    print(f"  Total elements: {len(element_pool)} -> {len(consolidated_pool)}")
    print(f"  Reduction: {len(element_pool) - len(consolidated_pool)} elements ({100 * (len(element_pool) - len(consolidated_pool)) / len(element_pool):.1f}%)")
    
    return consolidated_pool, updated_branches


def lattice_from_madx_file(madx_file: str, lattice_name: str = "ImportedLattice", consolidate_drifts_option: bool = True) -> Lattice:
    """
    Import a MAD-X lattice file using cpymad and convert it to an EICViBE Lattice object.
    
    Args:
        madx_file: Path to MAD-X lattice file
        lattice_name: Name for the imported lattice
        consolidate_drifts_option: Whether to consolidate drift elements (default: True)
    """
    madx = Madx()
    madx.call(madx_file)
    element_types_yaml = load_element_types_yaml()
    madx_mapping = build_madx_to_eicvibe_mapping(element_types_yaml)

    # Build element pool from sequences
    element_pool = {}
    branches = {}

    for seq_name, seq_obj in madx.sequence.items():
        # Get all elements from this sequence (including markers)
        all_elements = seq_obj.elements
        # Convert to list and ignore first and last elements (special markers)
        element_list = list(all_elements)
        if len(element_list) > 2:
            real_elements = element_list[1:-1]
        else:
            real_elements = element_list

        # Build element pool from real elements
        for element in real_elements:
            if element.name not in element_pool:
                # Create EICViBE element
                madx_type = get_madx_element_type(element)
                eicvibe_type = madx_to_eicvibe_type(madx_type, madx_mapping)
                length = getattr(element, 'l', 0.0)  # length attribute

                element_obj = create_element_by_type(
                    element_type=eicvibe_type,
                    name=element.name,
                    length=length
                )

                # Map MAD-X parameters to EICViBE parameter groups using the mapping
                # (avoiding known default values that weren't explicitly set)
                map_madx_parameters(element, element_obj, madx_mapping)

                element_pool[element.name] = element_obj

        # Store element names for this branch
        element_names = [e.name for e in real_elements]
        branches[seq_name] = element_names

    # Apply drift consolidation if requested
    if consolidate_drifts_option:
        element_pool, branches = consolidate_drifts(element_pool, branches)

    # Create Lattice
    lattice = Lattice(name=lattice_name)
    for element in element_pool.values():
        lattice.add_element(element)
    for branch_name, element_names in branches.items():
        lattice.add_branch(branch_name, element_names)
    lattice.set_root_branch(next(iter(branches)))
    return lattice 