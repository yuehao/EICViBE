# User Guides

## Getting Started

### [Lattice Design Guide](lattice_design.md)
Learn how to design accelerator lattices from scratch using EICViBE's element system and inheritance model. Covers basic patterns like FODO cells, interaction regions, and complex multi-branch accelerators.

### [Element Selection Guide](element_selection.md)
Master EICViBE's powerful element selection system including advanced features like relative positioning and ring wrap-around behavior.

### [MAD-X Integration Guide](madx_integration.md)
Import existing MAD-X lattice files and leverage EICViBE's analysis capabilities. Includes drift consolidation and parameter mapping.

## Advanced Topics

### Ring vs Linac Topology
Understanding how to properly specify and work with different accelerator topologies:

- **Linac topology** - Linear accelerators with defined start/end points
- **Ring topology** - Circular machines with wrap-around behavior for element selection

### Element Inheritance System
How to use EICViBE's inheritance model effectively:

- Define **prototype elements** with base parameters
- Create **instance elements** that inherit and can override parameters
- Automatic naming and occurrence tracking

### Parameter Management
Working with element parameters through the ParameterGroup system:

- **MagneticMultipoleP** - For magnetic elements (kn1, kn2, etc.)
- **GeometryP** - For geometric parameters (angles, positions)
- **RFP** - For RF cavity parameters (voltage, frequency, phase)

## Best Practices

1. **Design with inheritance** - Use prototypes for common element types
2. **Use relative positioning** - More maintainable than absolute positions
3. **Set topology correctly** - Enable ring features when needed
4. **Validate designs** - Use consistency checking tools
5. **Document your lattice** - Clear naming and parameter documentation

## Common Workflows

### Creating a New Lattice
1. Create lattice object
2. Define prototype elements
3. Add branches with appropriate topology
4. Build lattice structure using instances
5. Validate and analyze

### Importing from MAD-X
1. Import using `madx_import.import_madx_file()`
2. Set branch topology if needed
3. Use EICViBE selection tools for analysis
4. Export or modify as needed

### Analysis and Modification
1. Select elements by criteria
2. Analyze parameters and positions
3. Modify element parameters
4. Validate changes
5. Export results

For detailed examples and code samples, see the individual guide pages.
