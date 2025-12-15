# -*- coding: utf-8 -*-
"""
Data Models for Geotechnical Analysis
Clean, validated data structures using dataclasses
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import math
import numpy as np


@dataclass
class SoilProperties:
    """Soil material properties with validation."""
    
    cohesion_c: float
    friction_angle_phi: float
    unit_weight_dry: float
    unit_weight_sat: float
    poisson_ratio: float
    consolidation_cv: float = 1.0
    ocr: float = 1.0
    description: str = "Custom Soil"
    
    def __post_init__(self):
        if not 0 <= self.friction_angle_phi <= 45:
            raise ValueError(f"Friction angle must be 0-45")
        if not 0 <= self.poisson_ratio <= 0.5:
            raise ValueError(f"Poisson ratio must be 0-0.5")
        if self.cohesion_c < 0:
            raise ValueError(f"Cohesion must be >= 0")
    
    @property
    def K0(self) -> float:
        sin_phi = math.sin(math.radians(self.friction_angle_phi))
        k0_nc = 1 - sin_phi
        if self.ocr == 1.0:
            return k0_nc
        return k0_nc * (self.ocr ** sin_phi)
    
    @classmethod
    def from_preset(cls, soil_type: str):
        presets = {
            "Soft Clay": cls(10.0, 22.0, 16.0, 18.0, 0.35, 1.0, 1.0, "Soft Marine Clay"),
            "Stiff Clay": cls(25.0, 28.0, 18.0, 20.0, 0.30, 3.0, 2.0, "Stiff Clay"),
            "Dense Sand": cls(0.0, 35.0, 17.0, 20.0, 0.25, 100.0, 1.0, "Dense Sand"),
            "Loose Sand": cls(0.0, 28.0, 15.0, 18.0, 0.30, 50.0, 1.0, "Loose Sand"),
            "Silty Clay": cls(15.0, 25.0, 17.0, 19.0, 0.32, 2.0, 1.5, "Silty Clay"),
        }
        if soil_type not in presets:
            raise ValueError(f"Unknown: {soil_type}")
        return presets[soil_type]


@dataclass
class FoundationGeometry:
    width_B: float
    length_L: float
    depth_Df: float
    applied_stress_q: float
    
    def __post_init__(self):
        if self.width_B <= 0 or self.length_L <= 0:
            raise ValueError("Dimensions must be positive")


@dataclass
class AnalysisParameters:
    max_depth: float = 10.0
    water_table_depth: float = 2.0
    depth_increment: float = 0.5
    time_stages: List[float] = field(default_factory=lambda: [0, 1, 7, 30, 365])
    layer_thickness: float = 4.0
    
    def get_depth_array(self):
        return np.arange(0, self.max_depth + self.depth_increment/2, self.depth_increment)
    
    def get_time_array_seconds(self):
        return [t * 86400 for t in self.time_stages]


@dataclass
class StressState:
    depth: float
    time_days: float
    sigma_v_total: float
    sigma_h_total: float
    pore_pressure: float
    
    @property
    def sigma_v_eff(self): return self.sigma_v_total - self.pore_pressure
    @property
    def sigma_h_eff(self): return self.sigma_h_total - self.pore_pressure
    @property
    def sigma_1(self): return max(self.sigma_v_eff, self.sigma_h_eff)
    @property
    def sigma_3(self): return min(self.sigma_v_eff, self.sigma_h_eff)


@dataclass
class MohrCircle:
    sigma_1: float
    sigma_3: float
    depth: float = 0.0
    time_days: float = 0.0
    
    def __post_init__(self):
        if self.sigma_1 < self.sigma_3:
            self.sigma_1, self.sigma_3 = self.sigma_3, self.sigma_1
    
    @property
    def center(self): return (self.sigma_1 + self.sigma_3) / 2
    @property
    def radius(self): return (self.sigma_1 - self.sigma_3) / 2
    @property
    def max_shear(self): return self.radius
    
    def get_semicircle_points(self, n_points=100):
        theta = np.linspace(0, np.pi, n_points)
        sigma = self.center + self.radius * np.cos(theta)
        tau = self.radius * np.sin(theta)
        return sigma, tau
    
    @classmethod
    def from_stress_state(cls, stress_state):
        return cls(stress_state.sigma_1, stress_state.sigma_3, 
                   stress_state.depth, stress_state.time_days)


@dataclass
class FailureEnvelope:
    cohesion_c: float
    friction_angle_phi: float
    
    @property
    def phi_rad(self): return math.radians(self.friction_angle_phi)
    @property
    def tan_phi(self): return math.tan(self.phi_rad)
    @property
    def sin_phi(self): return math.sin(self.phi_rad)
    @property
    def cos_phi(self): return math.cos(self.phi_rad)
    
    def get_shear_strength(self, sigma):
        return self.cohesion_c + sigma * self.tan_phi
    
    def get_envelope_points(self, sigma_max, n_points=100):
        sigma = np.linspace(0, sigma_max, n_points)
        tau = self.cohesion_c + sigma * self.tan_phi
        return sigma, tau
    
    def get_failure_radius(self, sigma_center):
        return self.cohesion_c * self.cos_phi + sigma_center * self.sin_phi
    
    def get_tangent_point(self, mohr_circle):
        tangent_angle = math.radians(90 + self.friction_angle_phi)
        sigma_t = mohr_circle.center - mohr_circle.radius * math.cos(tangent_angle)
        tau_t = mohr_circle.radius * math.sin(tangent_angle)
        return sigma_t, tau_t
    
    def calculate_safety_factor(self, mohr_circle):
        if mohr_circle.radius < 1e-6:
            return float("inf")
        R_failure = self.get_failure_radius(mohr_circle.center)
        if R_failure <= 0:
            return 0.0
        return R_failure / mohr_circle.radius


@dataclass
class AnalysisResult:
    soil: SoilProperties
    foundation: FoundationGeometry
    parameters: AnalysisParameters
    stress_states: Dict = field(default_factory=dict)
    mohr_circles: Dict = field(default_factory=dict)
    safety_factors: Dict = field(default_factory=dict)
    
    def find_critical_condition(self):
        if not self.safety_factors:
            return {"min_fs": None, "depth": None, "time": None}
        min_key = min(self.safety_factors.keys(), key=lambda k: self.safety_factors[k])
        min_fs = self.safety_factors[min_key]
        return {"min_fs": min_fs, "depth": min_key[0], "time_days": min_key[1],
                "status": "SAFE" if min_fs > 1.0 else "FAILED"}
