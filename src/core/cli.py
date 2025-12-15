"""
CLI mode for non-interactive usage.
Maintains backwards compatibility with original command-line interface.
"""

import json
import pickle
import os
from pathlib import Path


def run_cli_mode(args):
    """Run analysis in CLI mode (non-interactive)."""
    from src.core.models import SoilProperties, FoundationGeometry, SoilColumn
    from src.core.simulation import run_consolidation_simulation
    from src.vis.plotting import plot_mohr_diagram, plot_stress_profiles, plot_safety_factor_evolution
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        # Default parameters
        config = {
            "soil": {
                "cohesion_c": 25.0,
                "friction_angle_phi": 30.0,
                "unit_weight_sat": 19.5,
                "unit_weight_dry": 15.0,
                "poissons_ratio": 0.35,
                "consolidation_coeff": 0.01,
                "permeability_k": 1e-9,
                "ocr": 1.0
            },
            "foundation": {
                "width_B": 2.0,
                "length_L": 2.0,
                "depth_D": 1.0,
                "applied_stress_q": 150.0
            },
            "soil_column": {
                "total_depth_H": 10.0,
                "water_table_depth": 2.0
            },
            "analysis": {
                "duration_days": 365,
                "num_stages": 5,
                "depth_increment": 0.5
            }
        }
    
    # Initialize objects
    soil_props = SoilProperties(**config['soil'])
    foundation = FoundationGeometry(**config['foundation'])
    soil_column = SoilColumn(
        total_depth_H=config['soil_column']['total_depth_H'],
        water_table_depth=config['soil_column']['water_table_depth'],
        soil_properties=soil_props
    )
    
    # Time stages
    duration = config['analysis']['duration_days']
    time_stages = [0, 1, 7, 30, duration]
    
    print("Running simulation...")
    results = run_consolidation_simulation(
        soil_column,
        foundation,
        time_stages,
        depth_increment=config['analysis']['depth_increment']
    )
    
    # Find critical condition
    min_fs = float('inf')
    critical_time = 0
    critical_depth = 0
    
    for t in results:
        for z in results[t]:
            fs = results[t][z]['fs']
            if fs < min_fs:
                min_fs = fs
                critical_time = t
                critical_depth = z
    
    print(f"Critical Condition: FS={min_fs:.2f} at t={critical_time}d, z={critical_depth}m")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Generate plots
    print("Generating plots...")
    
    mohr_circles = []
    labels = []
    for t in time_stages:
        if critical_depth in results[t]:
            mohr_circles.append(results[t][critical_depth]['mohr_circle'])
            labels.append(f"{t}d")
    
    fig_mohr = plot_mohr_diagram(mohr_circles, soil_props, labels, title=f"Mohr Circles at z={critical_depth}m")
    fig_mohr.savefig(output_dir / 'mohr_circles.png')
    
    fig_stress = plot_stress_profiles(results, time_stages)
    fig_stress.savefig(output_dir / 'stress_profiles.png')
    
    fig_fs = plot_safety_factor_evolution(results, critical_depth)
    fig_fs.savefig(output_dir / 'fs_evolution.png')
    
    # Save results
    with open(output_dir / 'results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Analysis complete. Results saved to {output_dir}")
