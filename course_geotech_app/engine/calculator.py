# -*- coding: utf-8 -*-
"""
Calculation Engine with Full Traceability
"""
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import math

from catalog.models import FormulaCatalog, FormulaEntry, CalculationStep
from catalog.parser import FormulaParser, StrictModeError


@dataclass
class CalculationResult:
    """Result of a calculation with traceability."""
    success: bool
    value: Optional[float] = None
    unit: str = ""
    steps: List[CalculationStep] = field(default_factory=list)
    error_message: str = ""
    missing_variables: List[str] = field(default_factory=list)
    missing_formulas: List[str] = field(default_factory=list)


class CalculationEngine:
    """Engine that performs calculations using ONLY catalog formulas."""
    
    def __init__(self, catalog: FormulaCatalog, strict_mode: bool = True):
        self.catalog = catalog
        self.strict_mode = strict_mode
        self.parser = FormulaParser(catalog.helper_functions)
        self.calculation_log: List[CalculationStep] = []
    
    def clear_log(self):
        self.calculation_log = []
    
    def evaluate_formula(self, formula_id: str, inputs: Dict[str, float]) -> CalculationResult:
        """Evaluate a specific formula from the catalog."""
        formula = self.catalog.get_formula(formula_id)
        
        if formula is None:
            return CalculationResult(
                success=False,
                error_message=f"Formula '{formula_id}' not found in catalog",
                missing_formulas=[formula_id]
            )
        
        if self.strict_mode:
            if formula.requires_confirmation:
                return CalculationResult(
                    success=False,
                    error_message=f"Formula '{formula_id}' requires confirmation from course notes. Reference: {formula.reference_label}",
                    missing_formulas=[formula_id]
                )
            if not formula.enabled:
                return CalculationResult(
                    success=False,
                    error_message=f"Formula '{formula_id}' is disabled. Enable it in the Formula Catalog.",
                    missing_formulas=[formula_id]
                )
        
        if formula.is_rule_only:
            return CalculationResult(
                success=True,
                value=None,
                error_message="This is a rule/procedure, not a numeric formula"
            )
        
        result, sub_str, missing = self.parser.evaluate(formula, inputs, self.strict_mode)
        
        if missing:
            return CalculationResult(
                success=False,
                error_message=f"Missing input variables: {', '.join(missing)}",
                missing_variables=missing
            )
        
        if result is None:
            return CalculationResult(
                success=False,
                error_message=sub_str
            )
        
        output_var, _, _ = self.parser.parse_equation(formula.equation)
        result_unit = formula.units.get(output_var, "") if output_var else ""
        
        step = CalculationStep(
            formula_id=formula.id,
            formula_name=formula.name,
            reference_label=formula.reference_label,
            equation_symbolic=formula.equation,
            equation_substituted=sub_str,
            result=result,
            result_unit=result_unit,
            notes=formula.notes
        )
        self.calculation_log.append(step)
        
        return CalculationResult(
            success=True,
            value=result,
            unit=result_unit,
            steps=[step]
        )
    
    def calculate_effective_stress(self, sigma_total: float = None, u: float = None, sigma_eff: float = None) -> CalculationResult:
        """Calculate effective stress relationship. Any two inputs -> compute third."""
        formula = self.catalog.get_formula("EFFECTIVE_STRESS")
        if formula is None or (self.strict_mode and not formula.enabled):
            return CalculationResult(
                success=False,
                error_message="EFFECTIVE_STRESS formula not available in catalog",
                missing_formulas=["EFFECTIVE_STRESS"]
            )
        
        inputs_provided = sum([sigma_total is not None, u is not None, sigma_eff is not None])
        
        if inputs_provided < 2:
            return CalculationResult(
                success=False,
                error_message="Need at least 2 of: sigma_total, u, sigma_eff"
            )
        
        if sigma_eff is None:
            return self.evaluate_formula("EFFECTIVE_STRESS", {"sigma_total": sigma_total, "u": u})
        elif u is None:
            u_calc = sigma_total - sigma_eff
            step = CalculationStep(
                formula_id="EFFECTIVE_STRESS",
                formula_name="Effective Stress (rearranged for u)",
                reference_label=formula.reference_label,
                equation_symbolic="u = sigma_total - sigma_eff",
                equation_substituted=f"u = {sigma_total:.4g} - {sigma_eff:.4g} = {u_calc:.4g} kPa",
                result=u_calc,
                result_unit="kPa"
            )
            self.calculation_log.append(step)
            return CalculationResult(success=True, value=u_calc, unit="kPa", steps=[step])
        else:
            sigma_total_calc = sigma_eff + u
            step = CalculationStep(
                formula_id="EFFECTIVE_STRESS",
                formula_name="Effective Stress (rearranged for sigma_total)",
                reference_label=formula.reference_label,
                equation_symbolic="sigma_total = sigma_eff + u",
                equation_substituted=f"sigma_total = {sigma_eff:.4g} + {u:.4g} = {sigma_total_calc:.4g} kPa",
                result=sigma_total_calc,
                result_unit="kPa"
            )
            self.calculation_log.append(step)
            return CalculationResult(success=True, value=sigma_total_calc, unit="kPa", steps=[step])
    
    def calculate_mohr_coulomb_failure(self, c: float, sigma_n: float, phi_rad: float) -> CalculationResult:
        """Calculate shear stress at failure using Mohr-Coulomb criterion."""
        return self.evaluate_formula("MC_FAILURE", {
            "c": c,
            "sigma_n_failure": sigma_n,
            "phi": phi_rad
        })
    
    def calculate_mohr_circle_params(self, sigma_1: float, sigma_3: float) -> Tuple[CalculationResult, CalculationResult]:
        """Calculate Mohr circle center and radius."""
        center_result = self.evaluate_formula("MOHR_CIRCLE_CENTER", {"sigma_1": sigma_1, "sigma_3": sigma_3})
        radius_result = self.evaluate_formula("MOHR_CIRCLE_RADIUS", {"sigma_1": sigma_1, "sigma_3": sigma_3})
        return center_result, radius_result
    
    def get_calculation_report(self) -> str:
        """Generate a text report of all calculations."""
        if not self.calculation_log:
            return "No calculations performed."
        
        lines = ["=" * 60, "CALCULATION REPORT WITH FORMULA TRACEABILITY", "=" * 60, ""]
        
        for i, step in enumerate(self.calculation_log, 1):
            lines.append(f"Step {i}: {step.formula_name}")
            lines.append(f"  Formula ID: {step.formula_id}")
            lines.append(f"  Reference: {step.reference_label}")
            lines.append(f"  Equation: {step.equation_symbolic}")
            lines.append(f"  Calculation: {step.equation_substituted}")
            lines.append(f"  Result: {step.result:.4g} {step.result_unit}")
            if step.notes:
                lines.append(f"  Notes: {step.notes}")
            lines.append("")
        
        return "\n".join(lines)
    
    def get_html_report(self) -> str:
        """Generate HTML report of calculations."""
        if not self.calculation_log:
            return "<p>No calculations performed.</p>"
        
        html = ['<div class="calculation-report">']
        html.append('<h2>Calculation Report with Formula Traceability</h2>')
        
        for i, step in enumerate(self.calculation_log, 1):
            html.append(f'<div class="calc-step">')
            html.append(f'<h3>Step {i}: {step.formula_name}</h3>')
            html.append(f'<table>')
            html.append(f'<tr><td><b>Formula ID:</b></td><td><code>{step.formula_id}</code></td></tr>')
            html.append(f'<tr><td><b>Reference:</b></td><td>{step.reference_label}</td></tr>')
            html.append(f'<tr><td><b>Equation:</b></td><td>{step.equation_symbolic}</td></tr>')
            html.append(f'<tr><td><b>Calculation:</b></td><td>{step.equation_substituted}</td></tr>')
            html.append(f'<tr><td><b>Result:</b></td><td><b>{step.result:.4g} {step.result_unit}</b></td></tr>')
            html.append(f'</table>')
            if step.notes:
                html.append(f'<p class="notes"><i>Notes: {step.notes}</i></p>')
            html.append('</div>')
        
        html.append('</div>')
        return "\n".join(html)
