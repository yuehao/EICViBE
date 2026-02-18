# define elements of accelerator
# its parameter should be in a ParameterGroup object.
# The only exceptions are the name, type and the length of the element.
# Current the type contains:
## ACKicker : Time varying kicker element, Allowed parameter groups: 'MagneticMultipoleP', 'ApertureP', 'RFP', 'MetaP'
## BeamBeam : Colliding beam interaction element, Allowed parameter groups: 'BeamBeamP', 'ApertureP', 'MetaP'
## Bend : Dipole bending magnet, Allowed parameter groups: 'BendP', 'ApertureP', 'MetaP', 'MagneticMultipoleP'
## CrabCavity : RF crab cavity element, Allowed parameter groups: 'RFP', 'ApertureP', 'MagneticMultipoleP', 'MetaP'
## Drift : Drift space element, Allowed parameter groups: 'ApertureP', 'MetaP'
## EGun : Electron gun element, Allowed parameter groups: 'EGunP', 'ApertureP', 'MetaP'
## Instrument : Diagnostic element, Allowed parameter groups: 'InstrumentP', 'ApertureP', 'MetaP'
## Kicker : Static kicker element, Allowed parameter groups: 'KickerP', 'ApertureP', 'MetaP'
## Marker : Marker element, Allowed parameter groups: 'ApertureP', 'MetaP'
## Match : Orbit/Twiss/Dispersion matching element, Allowed parameter groups: 'MatchP', 'ApertureP', 'MetaP'
## Multipole : Multipole element, Allowed parameter groups: 'MagneticMultipoleP', 'ApertureP', 'MetaP'
## Octupole : Octupole element, Allowed parameter groups: 'MagneticMultipoleP', 'ApertureP', 'MetaP'
## Quadrupole : Quadrupole element, Allowed parameter groups: 'MagneticMultipoleP', 'ApertureP', 'MetaP'
## RFCavity : RF cavity element, Allowed parameter groups: 'RFP', 'ApertureP', 'MagneticMultipoleP', 'MetaP'
## Sextupole : Sextupole element, Allowed parameter groups: 'MagneticMultipoleP', 'ApertureP', 'MetaP'
## Solenoid : Solenoid element, Allowed parameter groups: 'SolenoidP', 'ApertureP', 'MetaP'
## Taylor : Taylor map element, Allowed parameter groups: 'TaylorP', 'ApertureP', 'MetaP'
## Undulator : Undulator element, Allowed parameter groups: 'UndulatorP', 'ApertureP', 'MetaP'

# Define the Element class for general accelerator elements and
# each element type that inherits from it.


from .parameter_group import ParameterGroup
from ..models.base import PhysicsBaseModel
from pydantic import Field, field_validator, model_validator
from typing import List, Optional, Union
import os
import yaml
import logging

logger = logging.getLogger(__name__)


def _load_allowed_groups_from_yaml():
    yaml_path = os.path.join(os.path.dirname(__file__), "elements.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"elements.yaml not found at {yaml_path}")
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("elements.yaml is malformed: root should be a mapping")
    all_groups = set(data.get("All", {}).get("group", []))
    allowed = {}
    for elem_type, v in data.items():
        if elem_type == "All":
            continue
        groups = set(v.get("group", [])) | all_groups
        allowed[elem_type] = list(groups)
    return allowed


try:
    element_type_allowed_groups = _load_allowed_groups_from_yaml()
except Exception as e:
    raise RuntimeError(
        f"Failed to load allowed parameter groups from elements.yaml: {e}"
    )


class Element(PhysicsBaseModel):
    """Base Pydantic model for accelerator elements with enhanced validation."""

    name: str = Field(..., min_length=1, description="Element name")
    type: str = Field(..., min_length=1, description="Element type")
    length: float = Field(ge=0.0, description="Element length in meters")
    inherit: Optional[str] = Field(None, description="Prototype element name")
    parameters: List[ParameterGroup] = Field(
        default_factory=list, description="Parameter groups for this element"
    )

    # Plotting attributes (not validated but available for subclasses)
    plot_color: Optional[str] = Field(default=None, description="Color for plotting")
    plot_height: Optional[float] = Field(
        default=None, description="Height for beamline plots"
    )
    plot_cross_section: Optional[float] = Field(
        default=None, description="Cross section for floor plan plots"
    )

    @field_validator("name")
    @classmethod
    def validate_element_name(cls, v):
        """Validate element name follows conventions."""
        from ..models.validators import validate_element_name

        return validate_element_name(v)

    @field_validator("length")
    @classmethod
    def validate_element_length(cls, v):
        """Validate element length is physically reasonable."""
        if v > 10000:  # 10 km seems like a reasonable upper limit
            raise ValueError(f"Element length {v} m seems unreasonably large (>10km)")
        return v

    @field_validator("parameters")
    @classmethod
    def validate_parameter_groups_allowed(cls, v, info):
        """Validate parameter groups are allowed for element type."""
        if info.data and "type" in info.data:
            element_type = info.data["type"]
            allowed_groups = element_type_allowed_groups.get(element_type, [])

            for group in v:
                if group.type not in allowed_groups:
                    raise ValueError(
                        f"Parameter group '{group.type}' not allowed for element type '{element_type}'. "
                        f"Allowed groups: {allowed_groups}"
                    )
        return v

    @model_validator(mode="after")
    def validate_element_consistency(self):
        """Validate element consistency including parameter groups and element-specific checks."""
        # Validate parameter groups and run element-specific consistency checks
        self._validate_parameter_groups()
        self._check_element_specific_consistency()

        # Special handling for bend geometry validation
        if self.type in ["Bend", "RBend"] and hasattr(self, "_validate_bend_geometry"):
            self._validate_bend_geometry()

        return self

    def model_post_init(self, __context) -> None:
        """Post-initialization hook (replaces __post_init__)."""
        # This replaces the dataclass __post_init__ method
        # Subclasses can override this for custom initialization
        pass

    def add_parameter_group(self, group: ParameterGroup):
        """Add a parameter group to the element."""
        # More robust type checking for Pydantic compatibility
        if not hasattr(group, "type") or not hasattr(group, "parameters"):
            raise TypeError("Parameter group must be an instance of ParameterGroup.")

        # Check if parameter group is allowed for this element type
        allowed_groups = element_type_allowed_groups.get(self.type, [])
        if group.type not in allowed_groups:
            raise ValueError(
                f"Parameter group '{group.type}' is not allowed for element type '{self.type}'. "
                f"Allowed groups: {allowed_groups}"
            )

        # Add the parameter group
        self.parameters.append(group)

        # Trigger re-validation if this is a critical change
        if self.type in ["Bend", "RBend"] and group.type == "BendP":
            self._validate_bend_geometry_if_possible()

    def _validate_bend_geometry_if_possible(self):
        """Validate bend geometry if we have sufficient parameters."""
        try:
            bend_group = self.get_parameter_group("BendP")
            if bend_group is not None:
                bend_group.validate_bend_geometry_with_length(self.length)
        except Exception as e:
            logger.warning(f"Bend geometry validation skipped for {self.name}: {e}")

    def get_parameter_group(self, group_type: str) -> Optional[ParameterGroup]:
        """Get a parameter group by type."""
        for group in self.parameters:
            if group.type == group_type:
                return group
        return None

    def add_parameter(
        self,
        group_type: str,
        parameter_name: str,
        value: Union[str, float, int, List[float], List[int]],
    ):
        """Add a parameter to a specific group."""
        group = self.get_parameter_group(group_type)
        if group is None:
            group = ParameterGroup(name=group_type, type=group_type)
            self.add_parameter_group(group)
        group.add_parameter(parameter_name, value)

    def get_parameter(
        self, group_type: str, parameter_name: str
    ) -> Union[str, float, int, List[float], List[int], None]:
        """Get a parameter value by group type and parameter name."""
        group = self.get_parameter_group(group_type)
        if group is not None:
            return group.get_parameter(parameter_name)
        return None

    def remove_parameter(self, group_type: str, name: str):
        """Remove a parameter from a specific group."""
        group = self.get_parameter_group(group_type)
        if group is not None:
            group.remove_parameter(name)

    def get_length(self) -> float:
        """Get the length of the element."""
        return self.length

    def set_length(self, length: float):
        """Set the length of the element."""
        if length < 0:
            raise ValueError("Length must be non-negative.")
        self.length = length
        # Re-validate bend geometry if applicable
        if self.type in ["Bend", "RBend"]:
            self._validate_bend_geometry_if_possible()

    def get_type(self) -> str:
        """Get the type of the element."""
        return self.type

    def get_name(self) -> str:
        """Get the name of the element."""
        return self.name

    def set_name(self, name: str):
        """Set the name of the element."""
        if not name:
            raise ValueError("Name cannot be empty.")
        self.name = name

    def get_inherit(self) -> Optional[str]:
        """Get the name of the prototype this element inherits from."""
        return self.inherit

    def set_inherit(self, prototype_name: Optional[str]):
        """Set the prototype this element inherits from."""
        self.inherit = prototype_name

    def __str__(self) -> str:
        """String representation of the element."""
        inherit_str = f", inherit={self.inherit}" if self.inherit is not None else ""
        return f"{self.__class__.__name__}(name={self.name}, type={self.type}, length={self.length}{inherit_str}, parameters={self.parameters})"

    def __repr__(self) -> str:
        """String representation of the element."""
        return self.__str__()

    # Convert element to yaml dictionary format.
    def to_yaml_dict(self) -> dict:
        """Convert the element to a dictionary in the format:
        element_type:
              name: element_name
              length: number
              inherit: prototype_name (optional)
              parameter_group_1:
                      parameter_name: parameter_value
        """
        result = {}
        element_dict = {}
        element_dict["name"] = self.name
        element_dict["length"] = self.length

        # Add inherit field if not None
        if self.inherit is not None:
            element_dict["inherit"] = self.inherit

        # Add parameter groups using the ParameterGroup.to_yaml_dict() method
        for group in self.parameters:
            element_dict[group.type] = group.to_yaml_dict()

        result[self.type] = element_dict
        return result

    def check_consistency(self):
        """Check if the element is consistent. Validates parameter groups and calls element-specific checks."""
        # This method is maintained for backward compatibility
        # The actual validation happens in the Pydantic model validators
        self._validate_parameter_groups()
        self._check_element_specific_consistency()
        return True

    def _validate_parameter_groups(self):
        """Validate that all parameter groups are allowed for this element type."""
        allowed_groups = element_type_allowed_groups.get(self.type, [])
        for group in self.parameters:
            if group.type not in allowed_groups:
                raise ValueError(
                    f"Parameter group '{group.type}' not allowed for element type '{self.type}'. "
                    f"Allowed groups: {allowed_groups}"
                )

    def _check_element_specific_consistency(self):
        """Element-specific consistency checks. Override in subclasses as needed."""
        pass

    def _validate_bend_geometry(self):
        """Validate bend geometry for Bend and RBend elements."""
        if self.type in ["Bend", "RBend"]:
            bend_group = self.get_parameter_group("BendP")
            if bend_group is not None:
                # Use the enhanced bend geometry validation
                bend_group.validate_bend_geometry_with_length(self.length)

    # Define a plot method to visualize the element in a beamline floor view. The input should be matplotlib Axes object, the entrance coordinates and the tangent vector.
    # The output should be the coordinates of the exit point and the tangent vector, Each element type should implement its own plot method.
    def plot_in_floorplan(self, ax, entrance_coords, tangent_vector):
        return (
            entrance_coords,
            tangent_vector,
        )  # Default implementation does nothing, to be overridden by subclasses

    # Define a plot method to visualize the element in a 1-D beamline view, represented by boxes.  The input should be matplotlib Axes object, the initial s coordinate, and the optional normailzed strength.
    # If the normalized strength is not provided, all boxes will be full heigth; otherwise the height will be scaled by the normalizd strength.
    # The output should be the existing s coordinate
    def plot_in_beamline(self, ax, s_start, normalized_strength=None):
        """Plot the element in a 1-D beamline view."""
        return (
            s_start + self.length
        )  # Default implementation does nothing, to be overridden by subclasses

    # Expremental plotting function for 3-D visualization of the element, to be implemented by subclasses.
    def plot_in_3d(self, ax, entrance_coords, tangent_vector):
        return (
            entrance_coords,
            tangent_vector,
        )  # Default implementation does nothing, to be overridden by subclasses
