"""
Main Interactive Application for Geotechnical Analysis.
Rich-based Terminal UI with menu-driven simulation.
"""

import os
import sys
import json
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.table import Table
from rich import box
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ui.themes import GEOTECH_THEME, LOGO, MENU_ICONS, format_fs_display
from src.ui.components import (
    ParameterInput, 
    SoilPresetSelector, 
    ProgressDisplay,
    ResultsTable,
    PARAM_BOUNDS
)

# Initialize console with theme
console = Console(theme=GEOTECH_THEME)


class GeotechApp:
    """Interactive Geotechnical Analysis Application."""
    
    def __init__(self):
        self.soil_params: Dict[str, float] = {}
        self.foundation_params: Dict[str, float] = {}
        self.analysis_params: Dict[str, Any] = {}
        self.results: Optional[Dict] = None
        self.output_dir = Path("output")
        
    def run(self):
        """Main application entry point."""
        console.clear()
        self._show_welcome()
        
        while True:
            choice = self._main_menu()
            
            if choice == 1:
                self._new_analysis()
            elif choice == 2:
                self._load_previous()
            elif choice == 3:
                self._settings()
            elif choice == 4:
                self._show_help()
            elif choice == 5:
                if Confirm.ask("[warning]Exit application?[/warning]"):
                    console.print("[info]Thank you for using GeoTech Analysis. Goodbye![/info]")
                    break
    
    def _show_welcome(self):
        """Display welcome screen with logo."""
        console.print(LOGO, style="primary")
        console.print()
    
    def _main_menu(self) -> int:
        """Display main menu and get user choice."""
        menu_panel = Panel(
            f"""
{MENU_ICONS['new']}  [menu_item][1][/menu_item] New Analysis
{MENU_ICONS['load']}  [menu_item][2][/menu_item] Load Previous Analysis
{MENU_ICONS['settings']}  [menu_item][3][/menu_item] Settings & Preferences
{MENU_ICONS['help']}  [menu_item][4][/menu_item] Help & Documentation
{MENU_ICONS['exit']}  [menu_item][5][/menu_item] Exit
            """,
            title="[title]Main Menu[/title]",
            border_style="border",
            box=box.DOUBLE
        )
        console.print(menu_panel)
        
        return IntPrompt.ask("[cyan]Select option[/cyan]", choices=["1", "2", "3", "4", "5"], default="1")
    
    def _new_analysis(self):
        """Run a new analysis workflow."""
        console.clear()
        console.print(Panel("[title]New Analysis Wizard[/title]", border_style="cyan"))
        
        # Step 1: Soil Properties
        if not self._input_soil_properties():
            return
        
        # Step 2: Foundation Geometry
        if not self._input_foundation_geometry():
            return
        
        # Step 3: Analysis Parameters
        if not self._input_analysis_parameters():
            return
        
        # Step 4: Review and Confirm
        if not self._confirm_parameters():
            return
        
        # Step 5: Run Simulation
        self._run_simulation()
        
        # Step 6: Show Results
        self._show_results()
    
    def _input_soil_properties(self) -> bool:
        """Input soil properties with preset option."""
        console.print()
        console.print(f"{MENU_ICONS['soil']} [subtitle]Step 1/3: Soil Properties[/subtitle]")
        console.print()
        
        use_preset = Confirm.ask("[cyan]Use a preset soil type?[/cyan]", default=True)
        
        if use_preset:
            preset = SoilPresetSelector.select()
            if preset:
                self.soil_params = preset
                return True
        
        console.print()
        console.print("[info]Enter soil properties manually:[/info]")
        
        self.soil_params = {
            "cohesion_c": ParameterInput.get_float("cohesion_c"),
            "friction_angle_phi": ParameterInput.get_float("friction_angle_phi"),
            "unit_weight_sat": ParameterInput.get_float("unit_weight_sat"),
            "unit_weight_dry": ParameterInput.get_float("unit_weight_dry"),
            "poissons_ratio": ParameterInput.get_float("poissons_ratio"),
            "consolidation_coeff": ParameterInput.get_float("consolidation_coeff"),
            "permeability_k": ParameterInput.get_float("permeability_k"),
            "ocr": ParameterInput.get_float("ocr"),
        }
        
        return True
    
    def _input_foundation_geometry(self) -> bool:
        """Input foundation geometry parameters."""
        console.print()
        console.print(f"{MENU_ICONS['foundation']} [subtitle]Step 2/3: Foundation Geometry[/subtitle]")
        console.print()
        
        self.foundation_params = {
            "width_B": ParameterInput.get_float("width_B"),
            "length_L": ParameterInput.get_float("length_L"),
            "depth_D": ParameterInput.get_float("depth_D"),
            "applied_stress_q": ParameterInput.get_float("applied_stress_q"),
        }
        
        console.print()
        console.print("[info]Enter soil column parameters:[/info]")
        
        self.analysis_params["total_depth_H"] = ParameterInput.get_float("total_depth_H")
        self.analysis_params["water_table_depth"] = ParameterInput.get_float("water_table_depth")
        
        return True
    
    def _input_analysis_parameters(self) -> bool:
        """Input analysis configuration."""
        console.print()
        console.print(f"{MENU_ICONS['analysis']} [subtitle]Step 3/3: Analysis Parameters[/subtitle]")
        console.print()
        
        self.analysis_params["duration_days"] = ParameterInput.get_int("duration_days")
        self.analysis_params["depth_increment"] = ParameterInput.get_float("depth_increment")
        
        console.print()
        console.print("[info]Define time stages (comma-separated days, e.g., 0,1,7,30,365):[/info]")
        stages_str = Prompt.ask("[cyan]Time stages[/cyan]", default="0,1,7,30,365")
        
        try:
            self.analysis_params["time_stages"] = [int(x.strip()) for x in stages_str.split(",")]
        except ValueError:
            console.print("[warning]Invalid format. Using default stages.[/warning]")
            self.analysis_params["time_stages"] = [0, 1, 7, 30, 365]
        
        return True
    
    def _confirm_parameters(self) -> bool:
        """Display parameters for confirmation."""
        console.print()
        
        soil_table = Table(title="Soil Properties", box=box.ROUNDED, header_style="bold cyan")
        soil_table.add_column("Parameter", style="label")
        soil_table.add_column("Value", style="value", justify="right")
        soil_table.add_column("Unit", style="unit")
        
        param_display = {
            "cohesion_c": ("Cohesion (c')", "kPa"),
            "friction_angle_phi": ("Friction Angle (phi')", "deg"),
            "unit_weight_sat": ("Sat. Unit Weight", "kN/m3"),
            "unit_weight_dry": ("Dry Unit Weight", "kN/m3"),
            "poissons_ratio": ("Poisson's Ratio", "-"),
            "consolidation_coeff": ("Consolidation Coeff", "m2/year"),
            "permeability_k": ("Permeability", "m/s"),
            "ocr": ("OCR", "-"),
        }
        
        for key, (name, unit) in param_display.items():
            if key in self.soil_params:
                if key == "permeability_k":
                    soil_table.add_row(name, f"{self.soil_params[key]:.2e}", unit)
                else:
                    soil_table.add_row(name, f"{self.soil_params[key]:.2f}", unit)
        
        console.print(soil_table)
        console.print()
        
        found_table = Table(title="Foundation & Soil Column", box=box.ROUNDED, header_style="bold cyan")
        found_table.add_column("Parameter", style="label")
        found_table.add_column("Value", style="value", justify="right")
        found_table.add_column("Unit", style="unit")
        
        found_table.add_row("Width (B)", f"{self.foundation_params['width_B']:.2f}", "m")
        found_table.add_row("Length (L)", f"{self.foundation_params['length_L']:.2f}", "m")
        found_table.add_row("Depth (D)", f"{self.foundation_params['depth_D']:.2f}", "m")
        found_table.add_row("Applied Stress (q)", f"{self.foundation_params['applied_stress_q']:.1f}", "kPa")
        found_table.add_row("Total Depth (H)", f"{self.analysis_params['total_depth_H']:.2f}", "m")
        found_table.add_row("Water Table", f"{self.analysis_params['water_table_depth']:.2f}", "m")
        
        console.print(found_table)
        console.print()
        console.print(f"[info]Time stages: {self.analysis_params['time_stages']} days[/info]")
        console.print()
        
        return Confirm.ask("[cyan]Proceed with these parameters?[/cyan]", default=True)
    
    def _run_simulation(self):
        """Execute the consolidation simulation."""
        console.print()
        console.print(Panel("[title]Running Simulation...[/title]", border_style="cyan"))
        
        from src.core.models import SoilProperties, FoundationGeometry, SoilColumn
        from src.core.simulation import run_consolidation_simulation
        
        soil_props = SoilProperties(**self.soil_params)
        foundation = FoundationGeometry(**self.foundation_params)
        soil_column = SoilColumn(
            total_depth_H=self.analysis_params["total_depth_H"],
            water_table_depth=self.analysis_params["water_table_depth"],
            soil_properties=soil_props
        )
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Analyzing consolidation", total=100)
            
            self.results = run_consolidation_simulation(
                soil_column,
                foundation,
                self.analysis_params["time_stages"],
                depth_increment=self.analysis_params["depth_increment"]
            )
            
            progress.update(task, advance=100)
        
        console.print("[success]Simulation complete![/success]")
    
    def _show_results(self):
        """Display and explore simulation results."""
        if self.results is None:
            console.print("[error]No results available.[/error]")
            return
        
        min_fs = float('inf')
        critical_time = 0
        critical_depth = 0
        
        for t in self.results:
            for z in self.results[t]:
                fs = self.results[t][z].get('fs', float('inf'))
                if fs < min_fs:
                    min_fs = fs
                    critical_time = t
                    critical_depth = z
        
        ResultsTable.display_summary(self.results, critical_time, critical_depth, min_fs)
        
        while True:
            console.print()
            console.print("[subtitle]Explore Results:[/subtitle]")
            console.print("  [1] View stress profiles at specific time")
            console.print("  [2] Generate Mohr circle diagram")
            console.print("  [3] Generate all plots")
            console.print("  [4] Export results")
            console.print("  [5] Return to main menu")
            console.print()
            
            choice = IntPrompt.ask("[cyan]Select option[/cyan]", choices=["1", "2", "3", "4", "5"], default="3")
            
            if choice == 1:
                self._view_stress_profiles()
            elif choice == 2:
                self._generate_mohr_diagram(critical_depth)
            elif choice == 3:
                self._generate_all_plots(critical_depth)
            elif choice == 4:
                self._export_results()
            elif choice == 5:
                break
    
    def _view_stress_profiles(self):
        """View stress values at a specific time."""
        times = sorted(self.results.keys())
        console.print(f"[info]Available times: {times} days[/info]")
        
        time_choice = IntPrompt.ask("[cyan]Select time (days)[/cyan]", default=times[-1])
        
        if time_choice in self.results:
            ResultsTable.display_stress_table(self.results, time_choice)
        else:
            console.print("[warning]Invalid time selected.[/warning]")
    
    def _generate_mohr_diagram(self, critical_depth: float):
        """Generate Mohr circle diagram."""
        from src.core.models import SoilProperties
        from src.vis.plotting import plot_mohr_diagram
        
        console.print("[info]Generating Mohr circle diagram...[/info]")
        
        mohr_circles = []
        labels = []
        
        for t in self.analysis_params["time_stages"]:
            if t in self.results and critical_depth in self.results[t]:
                mohr_circles.append(self.results[t][critical_depth]['mohr_circle'])
                labels.append(f"{t}d")
        
        soil_props = SoilProperties(**self.soil_params)
        
        self.output_dir.mkdir(exist_ok=True)
        fig = plot_mohr_diagram(mohr_circles, soil_props, labels, title=f"Mohr Circles at z={critical_depth}m")
        output_path = self.output_dir / "mohr_circles.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        
        console.print(f"[success]Saved to {output_path}[/success]")
    
    def _generate_all_plots(self, critical_depth: float):
        """Generate all visualization plots."""
        from src.core.models import SoilProperties
        from src.vis.plotting import plot_mohr_diagram, plot_stress_profiles, plot_safety_factor_evolution
        
        self.output_dir.mkdir(exist_ok=True)
        
        console.print("[info]Generating all plots...[/info]")
        
        mohr_circles = []
        labels = []
        for t in self.analysis_params["time_stages"]:
            if t in self.results and critical_depth in self.results[t]:
                mohr_circles.append(self.results[t][critical_depth]['mohr_circle'])
                labels.append(f"{t}d")
        
        soil_props = SoilProperties(**self.soil_params)
        
        fig_mohr = plot_mohr_diagram(mohr_circles, soil_props, labels, title=f"Mohr Circles at z={critical_depth}m")
        fig_mohr.savefig(self.output_dir / "mohr_circles.png", dpi=150, bbox_inches='tight')
        console.print("  [success]mohr_circles.png[/success]")
        
        fig_stress = plot_stress_profiles(self.results, self.analysis_params["time_stages"])
        fig_stress.savefig(self.output_dir / "stress_profiles.png", dpi=150, bbox_inches='tight')
        console.print("  [success]stress_profiles.png[/success]")
        
        fig_fs = plot_safety_factor_evolution(self.results, critical_depth)
        fig_fs.savefig(self.output_dir / "fs_evolution.png", dpi=150, bbox_inches='tight')
        console.print("  [success]fs_evolution.png[/success]")
        
        console.print(f"[success]All plots saved to {self.output_dir}/[/success]")
    
    def _export_results(self):
        """Export results to various formats."""
        console.print()
        console.print("[subtitle]Export Options:[/subtitle]")
        console.print("  [1] Pickle (Python object)")
        console.print("  [2] JSON (data only)")
        console.print("  [3] CSV (tabular data)")
        console.print()
        
        choice = IntPrompt.ask("[cyan]Select format[/cyan]", choices=["1", "2", "3"], default="1")
        
        self.output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if choice == 1:
            output_path = self.output_dir / f"results_{timestamp}.pkl"
            with open(output_path, 'wb') as f:
                pickle.dump({
                    'results': self.results,
                    'soil_params': self.soil_params,
                    'foundation_params': self.foundation_params,
                    'analysis_params': self.analysis_params
                }, f)
            console.print(f"[success]Saved to {output_path}[/success]")
        
        elif choice == 2:
            output_path = self.output_dir / f"results_{timestamp}.json"
            json_results = {}
            for t in self.results:
                json_results[str(t)] = {}
                for z in self.results[t]:
                    json_results[str(t)][str(z)] = {
                        k: float(v) if isinstance(v, (int, float)) else str(v)
                        for k, v in self.results[t][z].items()
                        if k != 'mohr_circle'
                    }
            
            with open(output_path, 'w') as f:
                json.dump({
                    'results': json_results,
                    'soil_params': self.soil_params,
                    'foundation_params': self.foundation_params,
                }, f, indent=2)
            console.print(f"[success]Saved to {output_path}[/success]")
        
        elif choice == 3:
            output_path = self.output_dir / f"results_{timestamp}.csv"
            with open(output_path, 'w') as f:
                f.write("time_days,depth_m,sigma_v,sigma_v_eff,sigma_h_eff,pore_pressure,fs\n")
                for t in sorted(self.results.keys()):
                    for z in sorted(self.results[t].keys()):
                        d = self.results[t][z]
                        f.write(f"{t},{z},{d.get('sigma_v',0):.2f},{d.get('sigma_v_eff',0):.2f},")
                        f.write(f"{d.get('sigma_h_eff',0):.2f},{d.get('pore_pressure',0):.2f},{d.get('fs',0):.4f}\n")
            console.print(f"[success]Saved to {output_path}[/success]")
    
    def _load_previous(self):
        """Load a previous analysis from file."""
        console.print()
        console.print("[subtitle]Load Previous Analysis[/subtitle]")
        
        if not self.output_dir.exists():
            console.print("[warning]No previous analyses found.[/warning]")
            return
        
        pkl_files = list(self.output_dir.glob("*.pkl"))
        
        if not pkl_files:
            console.print("[warning]No saved analyses found.[/warning]")
            return
        
        console.print("[info]Available analyses:[/info]")
        for i, f in enumerate(pkl_files, 1):
            console.print(f"  [{i}] {f.name}")
        
        choice = IntPrompt.ask("[cyan]Select file[/cyan]", default=1)
        
        if 1 <= choice <= len(pkl_files):
            with open(pkl_files[choice - 1], 'rb') as f:
                data = pickle.load(f)
            
            self.results = data.get('results')
            self.soil_params = data.get('soil_params', {})
            self.foundation_params = data.get('foundation_params', {})
            self.analysis_params = data.get('analysis_params', {})
            
            console.print("[success]Analysis loaded successfully![/success]")
            self._show_results()
        else:
            console.print("[warning]Invalid selection.[/warning]")
    
    def _settings(self):
        """Application settings."""
        console.print()
        console.print(Panel("[subtitle]Settings[/subtitle]", border_style="cyan"))
        console.print()
        
        new_output = Prompt.ask("[cyan]Output directory[/cyan]", default=str(self.output_dir))
        self.output_dir = Path(new_output)
        
        console.print("[success]Settings updated![/success]")
    
    def _show_help(self):
        """Display help and documentation."""
        help_text = """
[bold]Geotechnical Analysis System - Help[/bold]

[subtitle]Overview[/subtitle]
This application performs plane strain consolidation analysis with 
Mohr-Coulomb failure criterion evaluation.

[subtitle]Analysis Steps[/subtitle]
1. [bold]Soil Properties[/bold]: Define material parameters or select a preset
2. [bold]Foundation[/bold]: Specify geometry and applied load
3. [bold]Analysis[/bold]: Configure time stages and depth resolution
4. [bold]Simulation[/bold]: Run the consolidation analysis
5. [bold]Results[/bold]: Explore stress profiles, Mohr circles, safety factors

[subtitle]Key Concepts[/subtitle]
- [bold]Safety Factor (FS)[/bold]: Ratio of shear strength to mobilized stress
  FS > 1.5: Safe (green) | 1.0 < FS <= 1.5: Marginal (yellow) | FS <= 1.0: Critical (red)

- [bold]Consolidation[/bold]: Time-dependent pore pressure dissipation
- [bold]Mohr Circle[/bold]: Graphical stress state representation
- [bold]Failure Envelope[/bold]: tau = c' + sigma' tan(phi')

[info]Press Enter to return to main menu.[/info]
        """
        
        console.print(Panel(help_text, title="[title]Help & Documentation[/title]", border_style="cyan"))
        Prompt.ask("")


def main():
    """Application entry point."""
    app = GeotechApp()
    app.run()


if __name__ == "__main__":
    main()
