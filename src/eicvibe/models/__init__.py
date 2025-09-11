"""
EICViBE Pydantic Models Package

This package contains Pydantic models for enhanced validation and type safety
in the EICViBE accelerator physics simulation framework.
"""

from .base import PhysicsBaseModel, ElementConfig, PhysicsParameterGroup
from .parameter_groups import (
    MagneticMultipoleP, BendP, RFP, SolenoidP, ApertureP, ControlP,
    MetaP, BodyShiftP, MarkerP, MonitorP, KickerP, BeamBeamP,
    PARAMETER_GROUP_MODELS, create_parameter_group_model
)

__all__ = [
    'PhysicsBaseModel', 
    'ElementConfig', 
    'PhysicsParameterGroup',
    'MagneticMultipoleP', 
    'BendP', 
    'RFP', 
    'SolenoidP', 
    'ApertureP', 
    'ControlP',
    'MetaP', 
    'BodyShiftP', 
    'MarkerP', 
    'MonitorP', 
    'KickerP', 
    'BeamBeamP',
    'PARAMETER_GROUP_MODELS', 
    'create_parameter_group_model'
]