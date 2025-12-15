# -*- coding: utf-8 -*-
"""
Geotechnical Simulation Engine

Implements:
- Terzaghi consolidation theory
- Boussinesq stress distribution
- Complete stress state calculation
"""

import math
import numpy as np
from typing import Dict, Tuple, List, Any
from dataclasses import dataclass

from .models import (
    SoilProperties, FoundationGeometry, AnalysisParameters,
    StressState, MohrCircle, FailureEnvelope, AnalysisResult
)

# Physical constants
GAMMA_WATER = 9.81  # kN/m3


def calculate_geostatic_stress(
    depth: float,
    water_table_depth: float,
    unit_weight_dry: float,
    unit_weight_sat: float
) -> Tuple[float, float, float]:
    """
    Calculate initial geostatic stresses before external loading.
    
    Returns:
        (sigma_v_total, pore_pressure, sigma_v_eff)
    """
    if depth <= 0:
        return 0.0, 0.0, 0.0
    
    if depth <= water_table_depth:
        sigma_v_total = depth * unit_weight_dry
        pore_pressure = 0.0
    else:
        sigma_above_wt = water_table_depth * unit_weight_dry
        sigma_below_wt = (depth - water_table_depth) * unit_weight_sat
        sigma_v_total = sigma_above_wt + sigma_below_wt
        pore_pressure = (depth - water_table_depth) * GAMMA_WATER
    
    sigma_v_eff = sigma_v_total - pore_pressure
    return sigma_v_total, pore_pressure, sigma_v_eff


def calculate_boussinesq_stress(
    q: float,
    width_B: float,
    length_L: float,
    depth: float
) -> float:
    """
    Calculate induced vertical stress using Boussinesq solution.
    
    Uses the Steinbrenner influence factor for rectangular loaded area.
    
    Args:
        q: Applied surface stress (kPa)
        width_B: Foundation width (m)
        length_L: Foundation length (m)
        depth: Depth below foundation (m)
        
    Returns:
        Induced vertical stress (kPa)
    """
    if depth <= 0.01:
        return q
    
    # Using influence factor approach
    m = width_B / (2 * depth)
    n = length_L / (2 * depth)
    
    m2, n2 = m**2, n**2
    
    try:
        # Steinbrenner formula for corner
        term1 = (2 * m * n * math.sqrt(m2 + n2 + 1)) / (m2 + n2 + m2*n2 + 1)
        term2 = (m2 + n2 + 2) / (m2 + n2 + 1)
        
        denom = m2 + n2 + 1 - m2*n2
        if abs(denom) < 1e-10:
            denom = 1e-10
        
        term3 = math.atan((2 * m * n * math.sqrt(m2 + n2 + 1)) / denom)
        
        if denom < 0:
            term3 += math.pi
        
        I_corner = (1 / (4 * math.pi)) * (term1 * term2 + term3)
        
        # Center of rectangle = 4 x corner value
        delta_sigma = 4 * I_corner * q
        
    except (ValueError, ZeroDivisionError):
        # Fallback to simplified formula
        influence = 0.5 / (1 + (2 * depth / width_B)**2)
        delta_sigma = influence * q
    
    return max(0, delta_sigma)


def calculate_consolidation_degree(time_days: float, cv: float, Hd: float) -> float:
    """
    Calculate average degree of consolidation using Terzaghi theory.
    
    Args:
        time_days: Time since loading (days)
        cv: Coefficient of consolidation (m2/year)
        Hd: Drainage path length (m)
        
    Returns:
        Consolidation degree U (0 to 1)
    """
    if time_days <= 0:
        return 0.0
    
    # Convert time to years
    t_years = time_days / 365.0
    
    # Time factor
    Tv = (cv * t_years) / (Hd ** 2)
    
    # Calculate U based on Tv range
    if Tv < 0.00001:
        U = 0.0
    elif Tv >= 4.0:
        U = 1.0
    elif Tv < 0.286:
        # Approximate formula for low Tv
        U = math.sqrt(4 * Tv / math.pi)
    else:
        # Approximate formula for high Tv
        exponent = (1.781 - Tv) / 0.933
        U_percent = 100 - (10 ** exponent)
        U = U_percent / 100.0
    
    return max(0.0, min(1.0, U))


def calculate_horizontal_stress(
    sigma_v_eff: float,
    K0: float,
    poisson_ratio: float,
    delta_sigma_v: float = 0.0
) -> float:
    """
    Calculate effective horizontal stress.
    
    For geostatic: sigma_h = K0 * sigma_v
    Additional from loading in plane strain
    
    Args:
        sigma_v_eff: Effective vertical stress (kPa)
        K0: At-rest earth pressure coefficient
        poisson_ratio: Poisson's ratio
        delta_sigma_v: Additional vertical stress from loading (kPa)
        
    Returns:
        Effective horizontal stress (kPa)
    """
    # Initial horizontal stress from geostatic conditions
    sigma_h_initial = K0 * sigma_v_eff
    
    # Additional horizontal stress from plane strain loading
    if delta_sigma_v > 0:
        delta_sigma_h = (poisson_ratio / (1 - poisson_ratio)) * delta_sigma_v
        return sigma_h_initial + delta_sigma_h
    
    return sigma_h_initial


def run_consolidation_analysis(
    soil: SoilProperties,
    foundation: FoundationGeometry,
    params: AnalysisParameters
) -> AnalysisResult:
    """
    Run complete consolidation analysis.
    
    Args:
        soil: Soil properties
        foundation: Foundation geometry and loading
        params: Analysis parameters
        
    Returns:
        Complete AnalysisResult with stress states, Mohr circles, and safety factors
    """
    result = AnalysisResult(
        soil=soil,
        foundation=foundation,
        parameters=params
    )
    
    envelope = FailureEnvelope(
        cohesion_c=soil.cohesion_c,
        friction_angle_phi=soil.friction_angle_phi
    )
    
    depths = params.get_depth_array()
    Hd = params.layer_thickness / 2  # Double drainage
    
    for t_days in params.time_stages:
        # Calculate consolidation degree at this time
        U = calculate_consolidation_degree(t_days, soil.consolidation_cv, Hd)
        
        for z in depths:
            if z < foundation.depth_Df:
                continue  # Skip above foundation level
            
            z_below_foundation = z - foundation.depth_Df
            
            # Initial geostatic stress
            sigma_v0, u0, sigma_v0_eff = calculate_geostatic_stress(
                z, params.water_table_depth,
                soil.unit_weight_dry, soil.unit_weight_sat
            )
            
            # Induced stress from foundation
            delta_sigma_v = calculate_boussinesq_stress(
                foundation.applied_stress_q,
                foundation.width_B,
                foundation.length_L,
                z_below_foundation
            )
            
            # Initial excess pore pressure (undrained)
            u_excess_initial = delta_sigma_v
            
            # Pore pressure at current time
            u_excess_current = u_excess_initial * (1 - U)
            u_total = u0 + u_excess_current
            
            # Total stresses
            sigma_v_total = sigma_v0 + delta_sigma_v
            
            # Effective vertical stress
            sigma_v_eff = sigma_v_total - u_total
            
            # Horizontal effective stress
            delta_sigma_eff = delta_sigma_v * U  # Effective stress increase
            sigma_h_eff = calculate_horizontal_stress(
                sigma_v0_eff, soil.K0, soil.poisson_ratio, delta_sigma_eff
            )
            
            # Total horizontal stress
            sigma_h_total = sigma_h_eff + u_total
            
            # Create stress state
            stress_state = StressState(
                depth=float(z),
                time_days=float(t_days),
                sigma_v_total=float(sigma_v_total),
                sigma_h_total=float(sigma_h_total),
                pore_pressure=float(u_total)
            )
            
            # Create Mohr circle
            mohr = MohrCircle.from_stress_state(stress_state)
            
            # Calculate safety factor
            fs = envelope.calculate_safety_factor(mohr)
            
            # Store results
            key = (float(z), float(t_days))
            result.stress_states[key] = stress_state
            result.mohr_circles[key] = mohr
            result.safety_factors[key] = float(fs)
    
    return result


def get_results_as_dict(result: AnalysisResult) -> Dict[float, Dict[float, Dict[str, Any]]]:
    """
    Convert AnalysisResult to nested dictionary for plotting.
    
    Returns:
        Dict[time, Dict[depth, data_dict]]
    """
    output = {}
    
    for (depth, time), stress in result.stress_states.items():
        if time not in output:
            output[time] = {}
        
        fs = result.safety_factors.get((depth, time), 1.0)
        
        output[time][depth] = {
            "sigma_v_total": stress.sigma_v_total,
            "sigma_h_total": stress.sigma_h_total,
            "sigma_v_eff": stress.sigma_v_eff,
            "sigma_h_eff": stress.sigma_h_eff,
            "pore_pressure": stress.pore_pressure,
            "sigma_1": stress.sigma_1,
            "sigma_3": stress.sigma_3,
            "fs": fs
        }
    
    return output
