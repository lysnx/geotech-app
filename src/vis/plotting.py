# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.models import MohrCircle, FailureEnvelope

COLORS = {'safe': '#27ae60', 'marginal': '#f39c12', 'failed': '#c0392b',
          'envelope': '#e74c3c', 'envelope_fill': '#fadbd8',
          'axis': '#2c3e50', 'grid': '#bdc3c7', 'text': '#2c3e50', 'tangent': '#8e44ad'}
MULTI_COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

def get_safety_color(fs):
    if fs < 1.0: return COLORS['failed']
    elif fs < 1.5: return COLORS['marginal']
    return COLORS['safe']

def get_safety_status(fs):
    if fs < 1.0: return 'FAILED'
    elif fs < 1.5: return 'MARGINAL'
    return 'SAFE'

def plot_mohr_diagram(mohr_circles, envelope, time_labels=None, title='Mohr-Coulomb Analysis',
                      show_tangent=True, show_annotations=True, show_failure_zone=True, figsize=(14, 10)):
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('white')
    if mohr_circles:
        max_sigma = max(c.sigma_1 for c in mohr_circles) * 1.3
        max_radius = max(c.radius for c in mohr_circles)
        min_sigma = min(c.sigma_3 for c in mohr_circles)
    else:
        max_sigma, max_radius, min_sigma = 200, 50, 0
    envelope_tau = envelope.get_shear_strength(max_sigma)
    max_tau = max(max_radius * 1.5, envelope_tau * 1.1)
    ax.set_xlim(min(-max_sigma * 0.05, min_sigma - 10), max_sigma)
    ax.set_ylim(-max_tau * 0.15, max_tau)
    if show_failure_zone:
        sigma_range = np.linspace(0, max_sigma * 1.1, 100)
        tau_env = envelope.cohesion_c + sigma_range * envelope.tan_phi
        verts = [(0, envelope.cohesion_c)] + list(zip(sigma_range, tau_env))
        verts += [(max_sigma * 1.1, max_tau * 1.2), (0, max_tau * 1.2)]
        ax.add_patch(Polygon(verts, facecolor=COLORS['envelope_fill'], edgecolor='none', alpha=0.6, zorder=1))
    sigma_env = np.linspace(0, max_sigma * 1.1, 100)
    tau_env = envelope.cohesion_c + sigma_env * envelope.tan_phi
    ax.plot(sigma_env, tau_env, color=COLORS['envelope'], linewidth=3, label='Failure Envelope', zorder=10)
    ax.scatter([0], [envelope.cohesion_c], color=COLORS['envelope'], s=120, zorder=15, marker='o', edgecolors='white', linewidths=2)
    for i, circle in enumerate(mohr_circles):
        color = MULTI_COLORS[i % len(MULTI_COLORS)] if len(mohr_circles) > 1 else get_safety_color(envelope.calculate_safety_factor(circle))
        label = time_labels[i] if time_labels and i < len(time_labels) else f'State {i+1}'
        fs = envelope.calculate_safety_factor(circle)
        theta = np.linspace(0, np.pi, 150)
        sigma = circle.center + circle.radius * np.cos(theta)
        tau = circle.radius * np.sin(theta)
        ax.fill(sigma, tau, color=color, alpha=0.15, zorder=3)
        ax.plot(sigma, tau, color=color, linewidth=2.5, label=f'{label}: FS={fs:.2f} ({get_safety_status(fs)})', zorder=5)
        ax.plot([circle.sigma_3, circle.sigma_1], [0, 0], color=color, linewidth=2.5, zorder=4)
        ax.scatter([circle.sigma_3, circle.sigma_1], [0, 0], color=color, s=80, zorder=15, marker='o', edgecolors='white', linewidths=1.5)
        if show_tangent:
            R_f = envelope.get_failure_radius(circle.center)
            sigma_t = circle.center - R_f * np.sin(envelope.phi_rad)
            tau_t = R_f * np.cos(envelope.phi_rad)
            if 0 < sigma_t < max_sigma and 0 < tau_t < max_tau:
                ax.scatter([sigma_t], [tau_t], color=COLORS['tangent'], s=150, marker='*', zorder=20, edgecolors='white')
                ax.plot([circle.center, sigma_t], [0, tau_t], color=COLORS['tangent'], linewidth=1.5, linestyle='--', alpha=0.7, zorder=8)
    ax.axhline(y=0, color=COLORS['axis'], linewidth=1.2, zorder=2)
    ax.axvline(x=0, color=COLORS['axis'], linewidth=1.2, zorder=2)
    ax.grid(True, linestyle=':', alpha=0.5, color=COLORS['grid'], zorder=0)
    ax.set_xlabel('Normal Effective Stress (kPa)', fontsize=13, fontweight='medium')
    ax.set_ylabel('Shear Stress (kPa)', fontsize=13, fontweight='medium')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.text(0.02, 0.98, f"c'={envelope.cohesion_c:.1f}kPa, phi'={envelope.friction_angle_phi:.1f}deg",
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=COLORS['grid'], alpha=0.95))
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    return fig

def plot_single_mohr_diagram(sigma_1, sigma_3, cohesion_c, friction_angle_phi, depth=0.0, time_days=0.0, title=None):
    circle = MohrCircle(sigma_1=sigma_1, sigma_3=sigma_3, depth=depth, time_days=time_days)
    envelope = FailureEnvelope(cohesion_c=cohesion_c, friction_angle_phi=friction_angle_phi)
    fs = envelope.calculate_safety_factor(circle)
    if title is None:
        title = f'Mohr Analysis at z={depth:.1f}m, t={time_days:.0f}d (FS={fs:.2f})'
    return plot_mohr_diagram([circle], envelope, [f'z={depth:.1f}m'], title)

def plot_stress_profiles(results_dict, time_stages):
    fig, axes = plt.subplots(1, 4, figsize=(18, 10), sharey=True)
    fig.patch.set_facecolor('white')
    times = sorted([t for t in results_dict.keys() if results_dict[t]])
    if not times: return fig
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(times)))
    for i, t in enumerate(times):
        depths = sorted(results_dict[t].keys())
        if not depths: continue
        sv_tot = [results_dict[t][z].get('sigma_v_total', 0) for z in depths]
        sv_eff = [results_dict[t][z].get('sigma_v_eff', 0) for z in depths]
        u_vals = [results_dict[t][z].get('pore_pressure', 0) for z in depths]
        fs_vals = [results_dict[t][z].get('fs', 1.0) for z in depths]
        lw = 2.5 if i == len(times) - 1 else 2
        axes[0].plot(sv_tot, depths, label=f't={t:.0f}d', color=colors[i], linewidth=lw)
        axes[1].plot(u_vals, depths, color=colors[i], linewidth=lw)
        axes[2].plot(sv_eff, depths, color=colors[i], linewidth=lw)
        axes[3].plot(fs_vals, depths, color=colors[i], linewidth=lw)
    titles = ['Total Vertical Stress', 'Pore Pressure', 'Effective Vertical Stress', 'Safety Factor']
    xlabels = ['sv (kPa)', 'u (kPa)', 'sv_eff (kPa)', 'FS']
    for i, ax in enumerate(axes):
        ax.set_xlabel(xlabels[i], fontsize=12)
        ax.set_title(titles[i], fontsize=13, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.invert_yaxis()
    axes[0].set_ylabel('Depth (m)', fontsize=12)
    axes[0].legend(loc='lower left', fontsize=9)
    axes[3].axvline(x=1.0, color=COLORS['failed'], linestyle='--', linewidth=2.5)
    axes[3].axvline(x=1.5, color=COLORS['marginal'], linestyle='--', linewidth=2)
    plt.suptitle('Stress and Safety Factor Profiles', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig

def plot_safety_factor_evolution(results_dict, critical_depth):
    fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
    times = sorted(results_dict.keys())
    fs_vals = []
    for t in times:
        if critical_depth in results_dict[t]:
            fs_vals.append(results_dict[t][critical_depth].get('fs', float('nan')))
        else:
            depths = list(results_dict[t].keys())
            closest = min(depths, key=lambda d: abs(d - critical_depth)) if depths else 0
            fs_vals.append(results_dict[t].get(closest, {}).get('fs', float('nan')))
    ax.axhspan(0, 1.0, alpha=0.2, color=COLORS['failed'])
    ax.axhspan(1.0, 1.5, alpha=0.15, color=COLORS['marginal'])
    ax.axhspan(1.5, 10, alpha=0.1, color=COLORS['safe'])
    ax.axhline(y=1.0, color=COLORS['failed'], linestyle='--', linewidth=2.5)
    ax.axhline(y=1.5, color=COLORS['marginal'], linestyle='--', linewidth=2)
    ax.plot(times, fs_vals, color='#2980b9', linewidth=3, marker='o', markersize=10,
            markerfacecolor='white', markeredgewidth=2, label=f'FS at z={critical_depth:.1f}m', zorder=10)
    ax.set_xlabel('Time (days)', fontsize=13)
    ax.set_ylabel('Safety Factor', fontsize=13)
    ax.set_title(f'Safety Factor Evolution (z={critical_depth:.1f}m)', fontsize=15, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', fontsize=10)
    max_fs = max(fs_vals) if fs_vals else 2.0
    ax.set_ylim(0, max(max_fs * 1.2, 2.5))
    plt.tight_layout()
    return fig