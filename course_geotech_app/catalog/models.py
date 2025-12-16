# -*- coding: utf-8 -*-
"""
Data Models for Formula Catalog
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import json

@dataclass
class FormulaEntry:
    """Single formula entry in the catalog."""
    id: str
    name: str
    equation: str
    variables: List[str] = field(default_factory=list)
    assumptions: str = ""
    units: Dict[str, str] = field(default_factory=dict)
    reference_label: str = ""
    notes: str = ""
    is_rule_only: bool = False
    requires_confirmation: bool = False
    enabled: bool = True
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "equation": self.equation,
            "variables": self.variables,
            "assumptions": self.assumptions,
            "units": self.units,
            "reference_label": self.reference_label,
            "notes": self.notes,
            "is_rule_only": self.is_rule_only,
            "requires_confirmation": self.requires_confirmation,
            "enabled": self.enabled
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "FormulaEntry":
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            equation=data.get("equation", ""),
            variables=data.get("variables", []),
            assumptions=data.get("assumptions", ""),
            units=data.get("units", {}),
            reference_label=data.get("reference_label", ""),
            notes=data.get("notes", ""),
            is_rule_only=data.get("is_rule_only", False),
            requires_confirmation=data.get("requires_confirmation", False),
            enabled=data.get("enabled", True)
        )

@dataclass
class CalculationStep:
    """Record of a single calculation step for traceability."""
    formula_id: str
    formula_name: str
    reference_label: str
    equation_symbolic: str
    equation_substituted: str
    result: float
    result_unit: str
    notes: str = ""
    
    def to_dict(self) -> dict:
        return {
            "formula_id": self.formula_id,
            "formula_name": self.formula_name,
            "reference_label": self.reference_label,
            "equation_symbolic": self.equation_symbolic,
            "equation_substituted": self.equation_substituted,
            "result": self.result,
            "result_unit": self.result_unit,
            "notes": self.notes
        }

@dataclass
class FormulaCatalog:
    """Complete formula catalog with all entries and helpers."""
    formulas: Dict[str, FormulaEntry] = field(default_factory=dict)
    helper_functions: Dict[str, str] = field(default_factory=dict)
    
    def get_formula(self, formula_id: str) -> Optional[FormulaEntry]:
        return self.formulas.get(formula_id)
    
    def add_formula(self, formula: FormulaEntry):
        self.formulas[formula.id] = formula
    
    def remove_formula(self, formula_id: str):
        if formula_id in self.formulas:
            del self.formulas[formula_id]
    
    def get_enabled_formulas(self) -> Dict[str, FormulaEntry]:
        return {k: v for k, v in self.formulas.items() if v.enabled and not v.requires_confirmation}
    
    def get_pending_formulas(self) -> Dict[str, FormulaEntry]:
        return {k: v for k, v in self.formulas.items() if v.requires_confirmation and not v.enabled}
    
    def confirm_formula(self, formula_id: str):
        if formula_id in self.formulas:
            self.formulas[formula_id].requires_confirmation = False
            self.formulas[formula_id].enabled = True
    
    def to_dict(self) -> dict:
        return {
            "helper_functions": self.helper_functions,
            "formulas": [f.to_dict() for f in self.formulas.values()]
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: dict) -> "FormulaCatalog":
        catalog = cls()
        catalog.helper_functions = data.get("helper_functions", {})
        for f_data in data.get("formulas", []):
            formula = FormulaEntry.from_dict(f_data)
            catalog.formulas[formula.id] = formula
        return catalog
    
    @classmethod
    def from_json(cls, json_str: str) -> "FormulaCatalog":
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    @classmethod
    def load_default(cls) -> "FormulaCatalog":
        import os
        catalog_path = os.path.join(os.path.dirname(__file__), "default_catalog.json")
        with open(catalog_path, "r", encoding="utf-8-sig") as f:
            return cls.from_json(f.read())
