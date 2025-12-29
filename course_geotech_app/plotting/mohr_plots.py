# -*- coding: utf-8 -*-
"""
Mohr Circle and Envelope Plotting
STANDARD CONVENTION: sigma (normal stress) on X-axis, tau (shear stress) on Y-axis
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from typing import List, Tuple, Optional, Dict
import math
import io
import base64

COLORS = [
    '#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c',
    '#e91e63', '#00bcd4', '#8bc34a', '#ff9800', '#673ab7', '#009688'
]

def plot_mohr_circles(
    circles: List[Tuple[float, float]],
    envelope_c: float = None,
    envelope_phi_rad: float = None,
    labels: List[str] = None,
    title: str = "Mohr Circle Analysis",
    show_envelope: bool = True,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot Mohr circles with optional failure envelope.
    STANDARD: X-axis = sigma (normal stress), Y-axis = tau (shear stress)
    
    Args:
        circles: List of (sigma_1, sigma_3) tuples (principal stresses)
        envelope_c: Cohesion for envelope (kPa)
        envelope_phi_rad: Friction angle for envelope (radians)
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('#fafafa')
    
    if not circles:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=16)
        return fig
    
    all_sigma1 = [c[0] for c in circles]
    all_sigma3 = [c[1] for c in circles]
    
    max_sigma = max(all_sigma1) * 1.3
    min_sigma = min(min(all_sigma3), 0) - 10
    max_radius = max((s1 - s3) / 2 for s1, s3 in circles)
    
    if envelope_c is not None and envelope_phi_rad is not None and show_envelope:
        envelope_tau = envelope_c + max_sigma * math.tan(envelope_phi_rad)
        max_tau = max(max_radius * 1.4, envelope_tau * 1.1)
    else:
        max_tau = max_radius * 1.5
    
    # X = sigma (normal stress), Y = tau (shear stress)
    ax.set_xlim(min_sigma, max_sigma)
    ax.set_ylim(-max_tau * 0.1, max_tau)
    
    # Plot failure envelope: tau = c + sigma * tan(phi)
    if envelope_c is not None and envelope_phi_rad is not None and show_envelope:
        sigma_env = np.linspace(0, max_sigma * 1.1, 100)
        tau_env = envelope_c + sigma_env * np.tan(envelope_phi_rad)
        
        verts = [(0, envelope_c)] + list(zip(sigma_env, tau_env))
        verts += [(max_sigma * 1.1, max_tau * 1.2), (0, max_tau * 1.2)]
        ax.add_patch(Polygon(verts, facecolor='#fadbd8', edgecolor='none', alpha=0.4, zorder=1))
        
        phi_deg = math.degrees(envelope_phi_rad)
        ax.plot(sigma_env, tau_env, color='#e74c3c', linewidth=3,
               label=f"Envelope: c={envelope_c:.1f} kPa, phi={phi_deg:.1f} deg", zorder=10)
        ax.scatter([0], [envelope_c], color='#e74c3c', s=100, zorder=15)
    
    # Plot Mohr circles
    for i, (sigma_1, sigma_3) in enumerate(circles):
        color = COLORS[i % len(COLORS)]
        label = labels[i] if labels and i < len(labels) else f'Test {i+1}'
        
        center = (sigma_1 + sigma_3) / 2
        radius = (sigma_1 - sigma_3) / 2
        
        # Semicircle: sigma = center + R*cos(theta), tau = R*sin(theta)
        theta = np.linspace(0, np.pi, 100)
        sigma = center + radius * np.cos(theta)  # X-axis: normal stress
        tau = radius * np.sin(theta)             # Y-axis: shear stress
        
        ax.fill(sigma, tau, color=color, alpha=0.15, zorder=2)
        ax.plot(sigma, tau, color=color, linewidth=2.5, label=label, zorder=5)
        
        # Line on sigma axis at tau=0
        ax.plot([sigma_3, sigma_1], [0, 0], color=color, linewidth=2.5, zorder=4)
        ax.scatter([sigma_3, sigma_1], [0, 0], color=color, s=60, zorder=15, edgecolors='white', linewidths=1.5)
        
        ax.annotate(f'sigma3={sigma_3:.0f}', (sigma_3, 0), textcoords="offset points",
                   xytext=(0, -15), ha='center', fontsize=8, color=color)
        ax.annotate(f'sigma1={sigma_1:.0f}', (sigma_1, 0), textcoords="offset points",
                   xytext=(0, -15), ha='center', fontsize=8, color=color)
    
    ax.axhline(y=0, color='#2c3e50', linewidth=1.2, zorder=2)
    ax.axvline(x=0, color='#2c3e50', linewidth=1.2, zorder=2)
    ax.grid(True, linestyle=':', alpha=0.5, color='#bdc3c7', zorder=0)
    
    ax.set_xlabel('Normal Stress sigma (kPa)', fontsize=12, fontweight='medium')
    ax.set_ylabel('Shear Stress tau (kPa)', fontsize=12, fontweight='medium')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    ax.legend(loc='upper left', fontsize=9, framealpha=0.95)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    return fig


def plot_direct_shear_data(
    sigma_n_values: List[float],
    tau_failure_values: List[float],
    fitted_c: float = None,
    fitted_phi_rad: float = None,
    title: str = "Direct Shear Test Results",
    figsize: Tuple[int, int] = (10, 7)
) -> plt.Figure:
    """Plot direct shear test data. X = sigma_n (normal), Y = tau_f (shear)."""
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('#fafafa')
    
    ax.scatter(sigma_n_values, tau_failure_values, s=100, c='#3498db', 
              edgecolors='white', linewidths=2, zorder=10, label='Test Data')
    
    if fitted_c is not None and fitted_phi_rad is not None:
        max_sigma = max(sigma_n_values) * 1.2
        sigma_line = np.linspace(0, max_sigma, 100)
        tau_line = fitted_c + sigma_line * np.tan(fitted_phi_rad)
        
        phi_deg = math.degrees(fitted_phi_rad)
        ax.plot(sigma_line, tau_line, 'r-', linewidth=2.5, 
               label=f'Envelope: c={fitted_c:.2f} kPa, phi={phi_deg:.1f} deg', zorder=5)
        ax.scatter([0], [fitted_c], color='red', s=80, zorder=15, marker='s')
    
    ax.axhline(y=0, color='#2c3e50', linewidth=1)
    ax.axvline(x=0, color='#2c3e50', linewidth=1)
    ax.grid(True, linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Normal Stress sigma_n (kPa)', fontsize=12)
    ax.set_ylabel('Shear Stress tau_f (kPa)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    return fig


def fit_envelope_to_circles(circles: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Fit failure envelope. Returns (c, phi_rad)."""
    if len(circles) < 2:
        return 0.0, math.radians(30)
    
    centers = [(s1 + s3) / 2 for s1, s3 in circles]
    radii = [(s1 - s3) / 2 for s1, s3 in circles]
    
    p_values = np.array(centers)
    q_values = np.array(radii)
    
    n = len(p_values)
    sum_p = np.sum(p_values)
    sum_q = np.sum(q_values)
    sum_pq = np.sum(p_values * q_values)
    sum_p2 = np.sum(p_values ** 2)
    
    denom = n * sum_p2 - sum_p ** 2
    if abs(denom) < 1e-10:
        return 0.0, math.radians(30)
    
    sin_phi = (n * sum_pq - sum_p * sum_q) / denom
    sin_phi = min(max(sin_phi, 0), 0.99)
    
    a = (sum_q - sin_phi * sum_p) / n
    phi_rad = math.asin(sin_phi)
    cos_phi = math.cos(phi_rad)
    c = a / cos_phi if cos_phi > 0.01 else 0.0
    c = max(0, c)
    
    return c, phi_rad


def fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def save_figure(fig: plt.Figure, filepath: str, format: str = 'png'):
    fig.savefig(filepath, format=format, dpi=150, bbox_inches='tight', facecolor='white')
