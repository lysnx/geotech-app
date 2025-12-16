# -*- coding: utf-8 -*-
"""
Formula Parser and Evaluator
Supports course notation like tg() for tan()
"""
import re
import math
from typing import Tuple, List, Dict, Optional, Any
from .models import FormulaEntry, CalculationStep

class FormulaParser:
    """Parse and evaluate formulas with custom notation support."""
    
    def __init__(self, helper_functions: Dict[str, str] = None):
        self.helpers = helper_functions or {"tg": "tan", "ctg": "1/tan", "arctg": "atan"}
        self.safe_functions = {
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "asin": math.asin, "acos": math.acos, "atan": math.atan,
            "sqrt": math.sqrt, "abs": abs, "pow": pow,
            "log": math.log, "log10": math.log10, "exp": math.exp,
            "pi": math.pi, "e": math.e,
            "radians": math.radians, "degrees": math.degrees
        }
    
    def preprocess_equation(self, equation: str) -> str:
        """Replace course notation with Python notation."""
        result = equation
        for course_func, python_func in self.helpers.items():
            pattern = rf'\b{course_func}\s*\('
            result = re.sub(pattern, f'{python_func}(', result)
        return result
    
    def parse_equation(self, equation: str) -> Tuple[str, str, List[str]]:
        """
        Parse equation of form 'output = expression'.
        Returns (output_var, expression, input_vars)
        """
        if "=" not in equation or equation.strip().startswith("RULE") or equation.strip().startswith("PROCEDURE"):
            return None, equation, []
        
        parts = equation.split("=", 1)
        if len(parts) != 2:
            return None, equation, []
        
        output_var = parts[0].strip()
        expression = parts[1].strip()
        
        processed_expr = self.preprocess_equation(expression)
        
        word_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
        all_words = set(re.findall(word_pattern, processed_expr))
        
        reserved = set(self.safe_functions.keys()) | {"tan", "sin", "cos", "sqrt", "abs", "pow", "log", "exp", "pi", "e"}
        input_vars = [w for w in all_words if w not in reserved]
        
        return output_var, processed_expr, input_vars
    
    def evaluate(self, formula: FormulaEntry, inputs: Dict[str, float], strict_mode: bool = True) -> Tuple[Optional[float], str, List[str]]:
        """
        Evaluate formula with given inputs.
        Returns (result, substitution_string, missing_vars)
        """
        if formula.is_rule_only:
            return None, "Rule only - no numeric evaluation", []
        
        if formula.requires_confirmation and strict_mode:
            return None, f"Formula requires confirmation: {formula.reference_label}", []
        
        if not formula.enabled and strict_mode:
            return None, f"Formula is disabled: {formula.id}", []
        
        output_var, expression, required_vars = self.parse_equation(formula.equation)
        
        if output_var is None:
            return None, "Cannot parse equation", []
        
        missing = [v for v in required_vars if v not in inputs]
        if missing:
            return None, f"Missing variables: {missing}", missing
        
        try:
            safe_dict = {**self.safe_functions, **inputs}
            result = eval(expression, {"__builtins__": {}}, safe_dict)
            
            sub_str = self._build_substitution_string(formula.equation, expression, inputs, result, formula.units)
            
            return float(result), sub_str, []
        except Exception as e:
            return None, f"Evaluation error: {str(e)}", []
    
    def _build_substitution_string(self, original_eq: str, processed_expr: str, inputs: dict, result: float, units: dict) -> str:
        """Build a string showing the substitution and result."""
        parts = original_eq.split("=", 1)
        output_var = parts[0].strip() if len(parts) == 2 else "result"
        
        sub_expr = processed_expr
        for var, val in inputs.items():
            if isinstance(val, float):
                sub_expr = re.sub(rf'\b{var}\b', f'{val:.4g}', sub_expr)
        
        result_unit = units.get(output_var, "")
        return f"{output_var} = {sub_expr} = {result:.4g} {result_unit}".strip()
    
    def solve_for_variable(self, formula: FormulaEntry, target_var: str, known_values: Dict[str, float], strict_mode: bool = True) -> Tuple[Optional[float], str, List[str]]:
        """
        Try to solve for a specific variable given other values.
        Limited to simple algebraic rearrangements.
        """
        if formula.is_rule_only:
            return None, "Rule only", []
        
        if formula.requires_confirmation and strict_mode:
            return None, f"Formula requires confirmation", []
        
        output_var, expression, required_vars = self.parse_equation(formula.equation)
        
        if output_var == target_var:
            return self.evaluate(formula, known_values, strict_mode)
        
        if target_var in required_vars:
            all_vars = required_vars + [output_var] if output_var else required_vars
            other_vars = [v for v in all_vars if v != target_var]
            missing = [v for v in other_vars if v not in known_values]
            
            if missing:
                return None, f"Cannot solve: missing {missing}", missing
            
            if output_var and output_var in known_values:
                pass
            else:
                return None, f"Cannot solve for {target_var}: need {output_var} value", [output_var]
        
        return None, f"Cannot algebraically solve for {target_var}", []
    
    def validate_formula(self, formula: FormulaEntry) -> List[str]:
        """Validate a formula entry and return list of issues."""
        issues = []
        
        if not formula.id:
            issues.append("Missing formula ID")
        if not formula.equation:
            issues.append("Missing equation")
        
        if not formula.is_rule_only:
            output_var, expr, input_vars = self.parse_equation(formula.equation)
            if output_var is None and "RULE" not in formula.equation and "PROCEDURE" not in formula.equation:
                issues.append("Equation must be in form 'output = expression'")
        
        return issues


class StrictModeError(Exception):
    """Raised when strict mode blocks a calculation."""
    def __init__(self, message: str, formula_id: str = None, missing_formulas: List[str] = None):
        super().__init__(message)
        self.formula_id = formula_id
        self.missing_formulas = missing_formulas or []
