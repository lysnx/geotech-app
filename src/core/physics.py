import math
import numpy as np
from typing import Dict, List, Tuple, Optional
from .models import SoilProperties, FoundationGeometry, SoilColumn, StressState, MohrCircle, FailureAnalysis

def calculate_geostatic_stress_profile(soil_column: SoilColumn, depth_increment: float = 0.5) -> Dict[float, Tuple[float, float, float]]:
    """
    Calculates geostatic stress profile (sigma_z, u, sigma_z_eff) vs depth.
    Returns: {depth: (sigma_z_total, u, sigma_z_eff)}
    """
    profile = {}
    z_values = np.arange(0, soil_column.total_depth_H + depth_increment/2, depth_increment)
    
    for z in z_values:
        # Total stress
        if z <= soil_column.water_table_depth:
            sigma_z = z * soil_column.soil_properties.unit_weight_dry # Assuming dry above WT
        else:
            sigma_z_above = soil_column.water_table_depth * soil_column.soil_properties.unit_weight_dry
            sigma_z_below = (z - soil_column.water_table_depth) * soil_column.soil_properties.unit_weight_sat
            sigma_z = sigma_z_above + sigma_z_below
            
        # Pore pressure
        if z <= soil_column.water_table_depth:
            u = 0.0
        else:
            u = (z - soil_column.water_table_depth) * 9.81 # gamma_w = 9.81 kN/m3
            
        # Effective stress
        sigma_z_eff = sigma_z - u
        profile[float(z)] = (sigma_z, u, sigma_z_eff)
        
    return profile

def calculate_induced_stress_boussinesq(q: float, width_B: float, length_L: float, z: float) -> float:
    """
    Calculates induced vertical stress using Boussinesq method (center of rectangular area).
    """
    if z <= 0:
        return q # At surface
        
    m = width_B / (2 * z) # Using B/2 and L/2 for corner formula x 4
    n = length_L / (2 * z)
    
    # Influence factor I_z for corner of rectangle B/2 x L/2
    
    m2 = m**2
    n2 = n**2
    term1 = (2 * m * n * math.sqrt(m2 + n2 + 1)) / (m2 + n2 + m2*n2 + 1)
    term2 = (m2 + n2 + 2) / (m2 + n2 + 1)
    term3 = math.atan((2 * m * n * math.sqrt(m2 + n2 + 1)) / (m2 + n2 + 1 - m2*n2))
    
    if (m2 + n2 + 1 - m2*n2) < 0:
        term3 += math.pi
        
    I_z_corner = (1 / (4 * math.pi)) * (term1 * term2 + term3)
    
    return 4 * I_z_corner * q

def calculate_consolidation_parameters(t_days: float, c_v: float, H_d: float, delta_sigma_z: float) -> Tuple[float, float, float]:
    """
    Calculates consolidation degree U, pore pressure u, and effective stress change.
    Returns: (U_avg, u_at_time, delta_sigma_eff)
    """
    if t_days <= 0:
        return 0.0, delta_sigma_z, 0.0
        
    # Convert time to years for c_v (m2/year)
    t_years = t_days / 365.0
    
    T_c = (c_v * t_years) / (H_d ** 2)
    
    # U_avg calculation
    if T_c < 0.00001:
        U_avg = 0.0
    elif T_c >= 4.0: # Fully consolidated
        U_avg = 1.0
    else:
        # Standard approximation
        if T_c < 0.286:
            U_avg = math.sqrt(4 * T_c / math.pi)
        else:
            exponent = (1.781 - T_c) / 0.933
            U_percent = 100 - (10 ** exponent)
            U_avg = U_percent / 100.0
                 
    U_avg = max(0.0, min(1.0, U_avg))
    
    u_at_time = delta_sigma_z * (1 - U_avg)
    delta_sigma_eff = delta_sigma_z - u_at_time
    
    return U_avg, u_at_time, delta_sigma_eff

def calculate_horizontal_stress_plane_strain(sigma_z_eff_initial: float, K0: float, nu: float, delta_sigma_z: float, delta_u: float) -> float:
    """
    Calculates effective horizontal stress under plane strain conditions.
    sigma'_h = K0 * sigma'_z0 + (nu / (1 - nu)) * (delta_sigma_z - delta_u)
    """
    delta_sigma_z_eff = delta_sigma_z - delta_u
    sigma_h_eff_initial = K0 * sigma_z_eff_initial
    
    delta_sigma_h_eff = (nu / (1 - nu)) * delta_sigma_z_eff
    
    return sigma_h_eff_initial + delta_sigma_h_eff

def construct_failure_envelope(c: float, phi: float, max_sigma: float) -> List[Tuple[float, float]]:
    """
    Generates points for the Mohr-Coulomb failure envelope.
    tau = c + sigma * tan(phi)
    """
    points = []
    phi_rad = math.radians(phi)
    tan_phi = math.tan(phi_rad)
    
    # Generate points
    sigma_values = np.linspace(0, max_sigma, 50)
    for sigma in sigma_values:
        tau = c + sigma * tan_phi
        points.append((float(sigma), float(tau)))
        
    return points

def calculate_safety_factor(mohr_circle: MohrCircle, c: float, phi: float) -> float:
    """
    Calculates Safety Factor FS = R_failure / R_current.
    R_failure is the radius of a circle with the same center that is tangent to the envelope.
    R_f = c * cos(phi) + center * sin(phi)
    """
    phi_rad = math.radians(phi)
    sin_phi = math.sin(phi_rad)
    cos_phi = math.cos(phi_rad)
    
    R_failure = c * cos_phi + mohr_circle.center * sin_phi
    
    if mohr_circle.radius < 1e-6:
        return float('inf') # No shear stress
        
    if R_failure <= 0:
        return 0.0 # Already failed
        
    return R_failure / mohr_circle.radius
