"""
Reusable UI components for the Geotechnical Analysis TUI.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Any
from rich.console import Console
from rich.prompt import Prompt, FloatPrompt, IntPrompt, Confirm
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.live import Live
from rich import box

console = Console()


# ============================================================================
# Soil Presets - Common geotechnical soil types
# ============================================================================

SOIL_PRESETS: Dict[str, Dict[str, float]] = {
    "Soft Marine Clay": {
        "cohesion_c": 10.0,
        "friction_angle_phi": 22.0,
        "unit_weight_sat": 16.5,
        "unit_weight_dry": 13.0,
        "poissons_ratio": 0.40,
        "consolidation_coeff": 0.005,
        "permeability_k": 1e-10,
        "ocr": 1.0,
    },
    "Medium Stiff Clay": {
        "cohesion_c": 25.0,
        "friction_angle_phi": 28.0,
        "unit_weight_sat": 19.0,
        "unit_weight_dry": 15.5,
        "poissons_ratio": 0.35,
        "consolidation_coeff": 0.01,
        "permeability_k": 1e-9,
        "ocr": 1.5,
    },
    "Stiff Overconsolidated Clay": {
        "cohesion_c": 50.0,
        "friction_angle_phi": 32.0,
        "unit_weight_sat": 20.5,
        "unit_weight_dry": 17.0,
        "poissons_ratio": 0.30,
        "consolidation_coeff": 0.02,
        "permeability_k": 5e-9,
        "ocr": 4.0,
    },
    "Loose Sand": {
        "cohesion_c": 0.0,
        "friction_angle_phi": 28.0,
        "unit_weight_sat": 18.5,
        "unit_weight_dry": 15.0,
        "poissons_ratio": 0.30,
        "consolidation_coeff": 100.0,
        "permeability_k": 1e-4,
        "ocr": 1.0,
    },
    "Dense Sand": {
        "cohesion_c": 0.0,
        "friction_angle_phi": 38.0,
        "unit_weight_sat": 20.0,
        "unit_weight_dry": 17.5,
        "poissons_ratio": 0.25,
        "consolidation_coeff": 200.0,
        "permeability_k": 5e-4,
        "ocr": 1.2,
    },
    "Silty Clay": {
        "cohesion_c": 15.0,
        "friction_angle_phi": 25.0,
        "unit_weight_sat": 18.0,
        "unit_weight_dry": 14.5,
        "poissons_ratio": 0.38,
        "consolidation_coeff": 0.008,
        "permeability_k": 5e-9,
        "ocr": 1.2,
    },
    "Custom": None,
}


@dataclass
class ParameterBounds:
    """Defines valid bounds for a parameter."""
    min_val: float
    max_val: float
    default: float
    unit: str
    description: str


# Parameter validation bounds
PARAM_BOUNDS = {
    "cohesion_c": ParameterBounds(0.0, 500.0, 25.0, "kPa", "Effective cohesion"),
    "friction_angle_phi": ParameterBounds(0.0, 45.0, 30.0, "degrees", "Effective friction angle"),
    "unit_weight_sat": ParameterBounds(14.0, 25.0, 19.5, "kN/m", "Saturated unit weight"),
    "unit_weight_dry": ParameterBounds(12.0, 22.0, 15.0, "kN/m", "Dry unit weight"),
    "poissons_ratio": ParameterBounds(0.1, 0.5, 0.35, "-", "Poisson's ratio"),
    "consolidation_coeff": ParameterBounds(0.0001, 500.0, 0.01, "m²/year", "Coefficient of consolidation"),
    "permeability_k": ParameterBounds(1e-12, 1e-2, 1e-9, "m/s", "Permeability"),
    "ocr": ParameterBounds(1.0, 20.0, 1.0, "-", "Overconsolidation ratio"),
    "width_B": ParameterBounds(0.5, 50.0, 2.0, "m", "Foundation width"),
    "length_L": ParameterBounds(0.5, 100.0, 2.0, "m", "Foundation length"),
    "depth_D": ParameterBounds(0.0, 10.0, 1.0, "m", "Foundation depth"),
    "applied_stress_q": ParameterBounds(10.0, 1000.0, 150.0, "kPa", "Applied stress"),
    "total_depth_H": ParameterBounds(2.0, 100.0, 10.0, "m", "Total soil depth"),
    "water_table_depth": ParameterBounds(0.0, 50.0, 2.0, "m", "Water table depth"),
    "duration_days": ParameterBounds(1, 36500, 365, "days", "Analysis duration"),
    "depth_increment": ParameterBounds(0.1, 5.0, 0.5, "m", "Depth increment"),
}


class ParameterInput:
    """Interactive parameter input with validation."""
    
    @staticmethod
    def get_float(name: str, prompt_text: str = None) -> float:
        """Get a float parameter with bounds validation."""
        bounds = PARAM_BOUNDS.get(name)
        if bounds is None:
            return FloatPrompt.ask(prompt_text or name)
        
        display_prompt = prompt_text or f"{bounds.description} [{bounds.unit}]"
        
        while True:
            try:
                value = FloatPrompt.ask(
                    f"[cyan]{display_prompt}[/cyan]",
                    default=bounds.default
                )
                if bounds.min_val <= value <= bounds.max_val:
                    return value
                else:
                    console.print(
                        f"[warning] Value must be between {bounds.min_val} and {bounds.max_val}[/warning]"
                    )
            except Exception as e:
                console.print(f"[error]Invalid input: {e}[/error]")
    
    @staticmethod
    def get_int(name: str, prompt_text: str = None) -> int:
        """Get an integer parameter with bounds validation."""
        bounds = PARAM_BOUNDS.get(name)
        if bounds is None:
            return IntPrompt.ask(prompt_text or name)
        
        display_prompt = prompt_text or f"{bounds.description} [{bounds.unit}]"
        
        while True:
            try:
                value = IntPrompt.ask(
                    f"[cyan]{display_prompt}[/cyan]",
                    default=int(bounds.default)
                )
                if bounds.min_val <= value <= bounds.max_val:
                    return value
                else:
                    console.print(
                        f"[warning] Value must be between {int(bounds.min_val)} and {int(bounds.max_val)}[/warning]"
                    )
            except Exception as e:
                console.print(f"[error]Invalid input: {e}[/error]")


class SoilPresetSelector:
    """Interactive soil preset selection."""
    
    @staticmethod
    def display_presets() -> None:
        """Display available soil presets in a table."""
        table = Table(
            title="Available Soil Presets",
            box=box.ROUNDED,
            header_style="bold cyan"
        )
        
        table.add_column("ID", style="dim", width=4)
        table.add_column("Soil Type", style="cyan")
        table.add_column("c' (kPa)", justify="right")
        table.add_column("φ' (°)", justify="right")
        table.add_column("γsat (kN/m)", justify="right")
        table.add_column("OCR", justify="right")
        
        for i, (name, props) in enumerate(SOIL_PRESETS.items(), 1):
            if props is None:
                table.add_row(str(i), name, "-", "-", "-", "-")
            else:
                table.add_row(
                    str(i),
                    name,
                    f"{props['cohesion_c']:.1f}",
                    f"{props['friction_angle_phi']:.1f}",
                    f"{props['unit_weight_sat']:.1f}",
                    f"{props['ocr']:.1f}"
                )
        
        console.print(table)
    
    @staticmethod
    def select() -> Optional[Dict[str, float]]:
        """Let user select a soil preset."""
        SoilPresetSelector.display_presets()
        
        preset_names = list(SOIL_PRESETS.keys())
        
        while True:
            choice = IntPrompt.ask(
                "[cyan]Select soil type[/cyan]",
                default=2  # Medium Stiff Clay
            )
            
            if 1 <= choice <= len(preset_names):
                selected = preset_names[choice - 1]
                if SOIL_PRESETS[selected] is None:
                    console.print("[info]Custom parameters selected. You will enter values manually.[/info]")
                    return None
                else:
                    console.print(f"[success] Selected: {selected}[/success]")
                    return SOIL_PRESETS[selected].copy()
            else:
                console.print(f"[warning]Please enter a number between 1 and {len(preset_names)}[/warning]")


class ProgressDisplay:
    """Display simulation progress with live updates."""
    
    def __init__(self, description: str = "Running simulation"):
        self.description = description
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        )
    
    def run_with_progress(self, total_steps: int, step_callback: Callable[[int], Any]) -> List[Any]:
        """Run a process with progress display."""
        results = []
        
        with self.progress:
            task = self.progress.add_task(self.description, total=total_steps)
            
            for i in range(total_steps):
                result = step_callback(i)
                results.append(result)
                self.progress.update(task, advance=1)
        
        return results


class ResultsTable:
    """Display analysis results in formatted tables."""
    
    @staticmethod
    def display_summary(results: Dict, critical_time: float, critical_depth: float, min_fs: float) -> None:
        """Display analysis summary."""
        from .themes import format_fs_display
        
        panel = Panel(
            f"""
[bold]Critical Condition Found[/bold]

Time:  [value]{critical_time}[/value] [unit]days[/unit]
Depth: [value]{critical_depth}[/value] [unit]m[/unit]
Safety Factor: {format_fs_display(min_fs)}
            """,
            title="[title]Analysis Results[/title]",
            border_style="cyan",
            box=box.DOUBLE
        )
        console.print(panel)
    
    @staticmethod
    def display_stress_table(results: Dict, time: float) -> None:
        """Display stress values at a specific time."""
        table = Table(
            title=f"Stress Profile at t = {time} days",
            box=box.ROUNDED,
            header_style="bold cyan"
        )
        
        table.add_column("Depth (m)", justify="right")
        table.add_column("σ_v (kPa)", justify="right")
        table.add_column("σ'_v (kPa)", justify="right")
        table.add_column("σ_h (kPa)", justify="right")
        table.add_column("u (kPa)", justify="right")
        table.add_column("FS", justify="right")
        
        if time in results:
            for depth in sorted(results[time].keys()):
                data = results[time][depth]
                fs = data['fs']
                
                # Color code FS
                if fs > 1.5:
                    fs_str = f"[green]{fs:.3f}[/green]"
                elif fs > 1.0:
                    fs_str = f"[yellow]{fs:.3f}[/yellow]"
                else:
                    fs_str = f"[red]{fs:.3f}[/red]"
                
                table.add_row(
                    f"{depth:.2f}",
                    f"{data.get('sigma_v', 0):.1f}",
                    f"{data.get('sigma_v_eff', 0):.1f}",
                    f"{data.get('sigma_h_eff', 0):.1f}",
                    f"{data.get('pore_pressure', 0):.1f}",
                    fs_str
                )
        
        console.print(table)
