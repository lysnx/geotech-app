# -*- coding: utf-8 -*-
"""Enhanced plotting for triaxial tests and multiple Mohr circles."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from typing import List, Tuple, Optional
import math

COLORS = [
    '#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c',
    '#e91e63', '#00bcd4', '#8bc34a', '#ff9800', '#673ab7', '#009688',
    '#ff5722', '#795548', '#607d8b', '#3f51b5', '#cddc39', '#ffc107',
    '#4caf50', '#2196f3', '#ff4081', '#00e676', '#651fff', '#18ffff'
]

def plot_multiple_mohr_circles(circles, envelope=None, labels=None, title="Mohr Circle Analysis", figsize=(14, 10)):
    """Plot multiple Mohr circles with failure envelope."""
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('#fafafa')
    
    if not circles:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=16)
        return fig
    
    all_sigma1 = [c.sigma_1 for c in circles]
    all_sigma3 = [c.sigma_3 for c in circles]
    
    max_sigma = max(all_sigma1) * 1.3
    min_sigma = min(min(all_sigma3), 0) - 10
    max_radius = max(c.radius for c in circles)
    
    if envelope:
        envelope_tau = envelope.get_shear_strength(max_sigma)
        max_tau = max(max_radius * 1.4, envelope_tau * 1.1)
    else:
        max_tau = max_radius * 1.5
    
    ax.set_xlim(min_sigma, max_sigma)
    ax.set_ylim(-max_tau * 0.1, max_tau)
    
    # Draw failure envelope
    if envelope:
        sigma_env, tau_env = envelope.get_envelope_points(max_sigma * 1.1)
        verts = [(0, envelope.cohesion_c)] + list(zip(sigma_env, tau_env))
        verts += [(max_sigma * 1.1, max_tau * 1.2), (0, max_tau * 1.2)]
        ax.add_patch(Polygon(verts, facecolor='#fadbd8', edgecolor='none', alpha=0.4, zorder=1))
        ax.plot(sigma_env, tau_env, color='#e74c3c', linewidth=3,
               label=f"Envelope: c={envelope.cohesion_c:.1f}kPa, phi={envelope.friction_angle_phi:.1f}deg", zorder=10)
        ax.scatter([0], [envelope.cohesion_c], color='#e74c3c', s=100, zorder=15)
    
    # Draw circles
    for i, circle in enumerate(circles):
        color = COLORS[i % len(COLORS)]
        label = labels[i] if labels and i < len(labels) else f'Test {i+1}'
        
        sigma, tau = circle.get_semicircle_points()
        ax.fill(sigma, tau, color=color, alpha=0.15, zorder=2)
        ax.plot(sigma, tau, color=color, linewidth=2.5, label=label, zorder=5)
        ax.plot([circle.sigma_3, circle.sigma_1], [0, 0], color=color, linewidth=2.5, zorder=4)
        ax.scatter([circle.sigma_3, circle.sigma_1], [0, 0], color=color, s=60, zorder=15, edgecolors='white', linewidths=1.5)
    
    ax.axhline(y=0, color='#2c3e50', linewidth=1.2, zorder=2)
    ax.axvline(x=0, color='#2c3e50', linewidth=1.2, zorder=2)
    ax.grid(True, linestyle=':', alpha=0.5, color='#bdc3c7', zorder=0)
    
    ax.set_xlabel('Normal Stress sigma (kPa)', fontsize=12, fontweight='medium')
    ax.set_ylabel('Shear Stress tau (kPa)', fontsize=12, fontweight='medium')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    ax.legend(loc='upper right', fontsize=9, framealpha=0.95)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    return fig


def plot_triaxial_test_results(series, title=None):
    """Plot triaxial test results with fitted envelope."""
    from src.core.models import FailureEnvelope
    
    circles = series.get_all_mohr_circles(use_effective=(series.test_type in ["CU", "CD"]))
    c, phi = series.calculate_failure_envelope(use_effective=(series.test_type in ["CU", "CD"]))
    envelope = FailureEnvelope(cohesion_c=c, friction_angle_phi=phi)
    
    labels = [f'{s.sample_id} (s3={s.confining_pressure_sigma3:.0f}kPa)' for s in series.samples]
    title = title or f'{series.test_type} Triaxial Test Results'
    
    return plot_multiple_mohr_circles(circles, envelope, labels, title)
