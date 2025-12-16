# -*- coding: utf-8 -*-
"""
Course-Strict Geotechnical Analysis Application
All calculations are driven by an editable Formula Catalog.
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from catalog.models import FormulaCatalog, FormulaEntry, CalculationStep
from catalog.parser import FormulaParser
from engine.calculator import CalculationEngine, CalculationResult
from plotting.mohr_plots import plot_mohr_circles, plot_direct_shear_data, fit_envelope_to_circles

st.set_page_config(
    page_title="Course-Strict Geotech",
    page_icon="graduation_cap",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {font-size: 2rem; font-weight: bold; color: #2c3e50; text-align: center; 
                  padding: 0.5rem; border-bottom: 3px solid #3498db; margin-bottom: 1rem;}
    .section-header {font-size: 1.3rem; font-weight: bold; color: #34495e; 
                     border-left: 4px solid #3498db; padding-left: 0.8rem; margin: 1rem 0;}
    .formula-box {background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; 
                  padding: 1rem; margin: 0.5rem 0;}
    .strict-badge {background: #27ae60; color: white; padding: 0.3rem 0.8rem; 
                   border-radius: 15px; font-weight: bold; font-size: 0.9rem;}
    .warning-badge {background: #f39c12; color: white; padding: 0.3rem 0.8rem; 
                    border-radius: 15px; font-weight: bold; font-size: 0.9rem;}
    .step-box {background: #e8f4fd; border-left: 4px solid #3498db; padding: 0.8rem; margin: 0.5rem 0;}
    .ref-label {background: #eee; padding: 0.2rem 0.5rem; border-radius: 4px; font-family: monospace;}
</style>
""", unsafe_allow_html=True)


def init_session():
    """Initialize session state."""
    if 'catalog' not in st.session_state:
        st.session_state.catalog = FormulaCatalog.load_default()
    if 'strict_mode' not in st.session_state:
        st.session_state.strict_mode = True
    if 'calculation_log' not in st.session_state:
        st.session_state.calculation_log = []
    if 'direct_shear_data' not in st.session_state:
        st.session_state.direct_shear_data = []
    if 'triaxial_data' not in st.session_state:
        st.session_state.triaxial_data = []


def get_engine() -> CalculationEngine:
    """Get calculation engine with current catalog."""
    return CalculationEngine(st.session_state.catalog, st.session_state.strict_mode)


def main():
    init_session()
    
    st.sidebar.markdown("## Navigation")
    
    strict_mode = st.sidebar.toggle("Course Mode: STRICT", value=st.session_state.strict_mode)
    st.session_state.strict_mode = strict_mode
    
    if strict_mode:
        st.sidebar.markdown('<span class="strict-badge">STRICT MODE ON</span>', unsafe_allow_html=True)
        st.sidebar.caption("Only catalog formulas allowed")
    else:
        st.sidebar.markdown('<span class="warning-badge">STRICT MODE OFF</span>', unsafe_allow_html=True)
        st.sidebar.caption("Warnings shown but not blocked")
    
    st.sidebar.markdown("---")
    
    tab = st.sidebar.radio("Select Module", [
        "Formula Catalog",
        "Stress & Effective Stress",
        "Direct Shear Test",
        "Triaxial Tests",
        "Mohr Circles & Envelope",
        "Reports & Export"
    ])
    
    catalog_stats(st.sidebar)
    
    if tab == "Formula Catalog":
        tab_formula_catalog()
    elif tab == "Stress & Effective Stress":
        tab_stress()
    elif tab == "Direct Shear Test":
        tab_direct_shear()
    elif tab == "Triaxial Tests":
        tab_triaxial()
    elif tab == "Mohr Circles & Envelope":
        tab_mohr()
    elif tab == "Reports & Export":
        tab_reports()


def catalog_stats(container):
    """Show catalog statistics in sidebar."""
    catalog = st.session_state.catalog
    total = len(catalog.formulas)
    enabled = len([f for f in catalog.formulas.values() if f.enabled and not f.requires_confirmation])
    pending = len([f for f in catalog.formulas.values() if f.requires_confirmation])
    
    container.markdown("### Catalog Status")
    container.metric("Enabled Formulas", enabled)
    if pending > 0:
        container.warning(f"{pending} formulas need confirmation")


def tab_formula_catalog():
    """Formula Catalog tab - view and edit all formulas."""
    st.markdown('<div class="main-header">Formula Catalog (Course Only)</div>', unsafe_allow_html=True)
    
    st.info("This catalog controls ALL calculations. Edit formulas to match your course exactly.")
    
    catalog = st.session_state.catalog
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Export Catalog JSON"):
            st.download_button(
                "Download JSON",
                catalog.to_json(),
                "course_catalog.json",
                "application/json"
            )
    with col2:
        uploaded = st.file_uploader("Import Catalog", type=['json'], key='catalog_upload')
        if uploaded:
            try:
                new_catalog = FormulaCatalog.from_json(uploaded.read().decode('utf-8'))
                st.session_state.catalog = new_catalog
                st.success("Catalog imported successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Import error: {e}")
    with col3:
        if st.button("Reset to Default"):
            st.session_state.catalog = FormulaCatalog.load_default()
            st.success("Reset to default catalog")
            st.rerun()
    
    st.markdown("---")
    st.markdown('<div class="section-header">Formula Entries</div>', unsafe_allow_html=True)
    
    for formula_id, formula in catalog.formulas.items():
        with st.expander(f"{'checkmark' if formula.enabled and not formula.requires_confirmation else 'warning' if formula.requires_confirmation else 'x'} {formula.id} - {formula.name}", expanded=False):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**Reference:** `{formula.reference_label}`")
                st.code(formula.equation, language=None)
                st.caption(f"Variables: {', '.join(formula.variables) if formula.variables else 'None'}")
                if formula.notes:
                    st.info(formula.notes)
            
            with col2:
                if formula.requires_confirmation:
                    st.warning("Needs confirmation")
                    if st.button(f"Confirm from course", key=f"confirm_{formula_id}"):
                        catalog.confirm_formula(formula_id)
                        st.success("Confirmed!")
                        st.rerun()
                else:
                    new_enabled = st.checkbox("Enabled", value=formula.enabled, key=f"enable_{formula_id}")
                    if new_enabled != formula.enabled:
                        formula.enabled = new_enabled
                        st.rerun()
            
            with st.form(key=f"edit_{formula_id}"):
                st.markdown("**Edit Formula**")
                new_eq = st.text_input("Equation", value=formula.equation, key=f"eq_{formula_id}")
                new_ref = st.text_input("Reference Label", value=formula.reference_label, key=f"ref_{formula_id}")
                new_notes = st.text_area("Notes", value=formula.notes, key=f"notes_{formula_id}")
                
                if st.form_submit_button("Save Changes"):
                    formula.equation = new_eq
                    formula.reference_label = new_ref
                    formula.notes = new_notes
                    st.success("Formula updated!")
                    st.rerun()
    
    st.markdown("---")
    st.markdown('<div class="section-header">Add New Formula</div>', unsafe_allow_html=True)
    
    with st.form("add_formula"):
        new_id = st.text_input("Formula ID (e.g., MY_FORMULA)")
        new_name = st.text_input("Name")
        new_equation = st.text_input("Equation (e.g., y = a + b * x)")
        new_ref = st.text_input("Reference Label (e.g., Chap6 p5)")
        new_notes = st.text_area("Notes")
        is_rule = st.checkbox("This is a rule/procedure (non-numeric)")
        
        if st.form_submit_button("Add Formula"):
            if new_id and new_equation:
                new_formula = FormulaEntry(
                    id=new_id,
                    name=new_name,
                    equation=new_equation,
                    reference_label=new_ref,
                    notes=new_notes,
                    is_rule_only=is_rule,
                    enabled=True
                )
                catalog.add_formula(new_formula)
                st.success(f"Added formula: {new_id}")
                st.rerun()
            else:
                st.error("ID and Equation are required")


def tab_stress():
    """Stress & Effective Stress calculations."""
    st.markdown('<div class="main-header">Stress & Effective Stress</div>', unsafe_allow_html=True)
    
    engine = get_engine()
    
    eff_formula = st.session_state.catalog.get_formula("EFFECTIVE_STRESS")
    if eff_formula and eff_formula.enabled:
        st.success(f"Using: {eff_formula.equation} (Ref: {eff_formula.reference_label})")
    else:
        st.error("EFFECTIVE_STRESS formula not enabled in catalog!")
        return
    
    st.markdown('<div class="section-header">Calculate Effective Stress Relationship</div>', unsafe_allow_html=True)
    st.latex(r"\sigma' = \sigma_{total} - u")
    
    st.info("Enter any TWO values to compute the third.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        use_sigma_total = st.checkbox("sigma_total known", value=True)
        sigma_total = st.number_input("sigma_total (kPa)", value=100.0, disabled=not use_sigma_total)
    
    with col2:
        use_u = st.checkbox("u known", value=True)
        u = st.number_input("Pore pressure u (kPa)", value=30.0, disabled=not use_u)
    
    with col3:
        use_sigma_eff = st.checkbox("sigma' known", value=False)
        sigma_eff = st.number_input("sigma' (kPa)", value=70.0, disabled=not use_sigma_eff)
    
    if st.button("Calculate", type="primary"):
        engine.clear_log()
        
        result = engine.calculate_effective_stress(
            sigma_total=sigma_total if use_sigma_total else None,
            u=u if use_u else None,
            sigma_eff=sigma_eff if use_sigma_eff else None
        )
        
        if result.success:
            st.success(f"**Result: {result.value:.2f} {result.unit}**")
            
            st.markdown("### Calculation Steps")
            for step in result.steps:
                st.markdown(f"""
                <div class="step-box">
                <b>Formula:</b> {step.formula_name}<br>
                <b>ID:</b> <code>{step.formula_id}</code> | 
                <b>Reference:</b> <span class="ref-label">{step.reference_label}</span><br>
                <b>Equation:</b> {step.equation_symbolic}<br>
                <b>Calculation:</b> {step.equation_substituted}
                </div>
                """, unsafe_allow_html=True)
            
            st.session_state.calculation_log.extend(result.steps)
        else:
            st.error(result.error_message)


def tab_direct_shear():
    """Direct Shear Test tab."""
    st.markdown('<div class="main-header">Direct Shear Test</div>', unsafe_allow_html=True)
    
    mc_formula = st.session_state.catalog.get_formula("MC_FAILURE")
    if mc_formula and mc_formula.enabled:
        st.success(f"Using: {mc_formula.equation} (Ref: {mc_formula.reference_label})")
    else:
        st.error("MC_FAILURE formula not enabled!")
        return
    
    st.markdown('<div class="section-header">Enter Test Data</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        num_tests = st.number_input("Number of Tests", min_value=2, max_value=10, value=3)
        
        test_data = []
        for i in range(int(num_tests)):
            st.markdown(f"**Test {i+1}**")
            sigma_n = st.number_input(f"sigma_n (kPa)", value=50.0 + i*50, key=f"ds_sigma_{i}")
            tau_f = st.number_input(f"tau_failure (kPa)", value=40.0 + i*30, key=f"ds_tau_{i}")
            test_data.append((sigma_n, tau_f))
    
    with col2:
        if st.button("Analyze Direct Shear", type="primary"):
            sigma_n_vals = [t[0] for t in test_data]
            tau_f_vals = [t[1] for t in test_data]
            
            n = len(sigma_n_vals)
            sum_x = sum(sigma_n_vals)
            sum_y = sum(tau_f_vals)
            sum_xy = sum(x*y for x, y in zip(sigma_n_vals, tau_f_vals))
            sum_x2 = sum(x**2 for x in sigma_n_vals)
            
            denom = n * sum_x2 - sum_x**2
            if abs(denom) > 1e-10:
                tan_phi = (n * sum_xy - sum_x * sum_y) / denom
                c = (sum_y - tan_phi * sum_x) / n
                phi_rad = math.atan(tan_phi)
                phi_deg = math.degrees(phi_rad)
                c = max(0, c)
                
                st.success(f"**Results:** c = {c:.2f} kPa, phi = {phi_deg:.1f} deg")
                
                st.markdown(f"""
                <div class="step-box">
                <b>Formula:</b> {mc_formula.name}<br>
                <b>ID:</b> <code>{mc_formula.id}</code> | 
                <b>Reference:</b> <span class="ref-label">{mc_formula.reference_label}</span><br>
                <b>Method:</b> Linear regression on tau_f vs sigma_n data<br>
                <b>Equation:</b> {mc_formula.equation}<br>
                <b>Fitted:</b> c = {c:.2f} kPa, tan(phi) = {tan_phi:.4f}, phi = {phi_deg:.1f} deg
                </div>
                """, unsafe_allow_html=True)
                
                fig = plot_direct_shear_data(sigma_n_vals, tau_f_vals, c, phi_rad)
                st.pyplot(fig)
                plt.close(fig)
                
                st.session_state.direct_shear_data = test_data
            else:
                st.error("Cannot fit line - insufficient data variation")


def tab_triaxial():
    """Triaxial Tests tab (CD/CU/UU)."""
    st.markdown('<div class="main-header">Triaxial Tests (CD/CU/UU)</div>', unsafe_allow_html=True)
    
    rules_formula = st.session_state.catalog.get_formula("TRIAXIAL_TEST_TYPES")
    if rules_formula:
        with st.expander("Triaxial Test Types (from catalog)"):
            st.code(rules_formula.equation)
            st.caption(f"Reference: {rules_formula.reference_label}")
            st.info(rules_formula.notes)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        test_type = st.selectbox("Test Type", ["CD", "CU", "UU"])
        
        if test_type == "CD":
            st.info("CD: Consolidated Drained - excess pore pressure negligible")
        elif test_type == "CU":
            st.info("CU: Consolidated Undrained - measure pore pressure")
        else:
            st.info("UU: Unconsolidated Undrained - closed drainage")
        
        num_samples = st.number_input("Number of Samples", min_value=2, max_value=10, value=3)
        
        samples = []
        for i in range(int(num_samples)):
            with st.expander(f"Sample {i+1}", expanded=(i < 3)):
                sigma3 = st.number_input("sigma_3 (kPa)", value=50.0 + i*50, key=f"tri_s3_{i}")
                deviator = st.number_input("Deviator stress (kPa)", value=100.0 + i*40, key=f"tri_dev_{i}")
                
                if test_type == "CU":
                    u = st.number_input("Pore pressure u (kPa)", value=20.0 + i*10, key=f"tri_u_{i}")
                else:
                    u = 0.0
                
                samples.append({"sigma3": sigma3, "deviator": deviator, "u": u})
    
    with col2:
        if st.button("Analyze Triaxial Tests", type="primary"):
            engine = get_engine()
            
            s1_formula = st.session_state.catalog.get_formula("SIGMA1_FROM_DEVIATOR")
            if not s1_formula or not s1_formula.enabled:
                st.warning("SIGMA1_FROM_DEVIATOR formula needs confirmation. Using: sigma_1 = sigma_3 + deviator")
            
            circles = []
            results_data = []
            
            for i, s in enumerate(samples):
                sigma_1 = s["sigma3"] + s["deviator"]
                sigma_3 = s["sigma3"]
                
                if test_type in ["CU", "CD"]:
                    sigma_1_eff = sigma_1 - s["u"]
                    sigma_3_eff = sigma_3 - s["u"]
                    circles.append((sigma_1_eff, sigma_3_eff))
                else:
                    circles.append((sigma_1, sigma_3))
                
                results_data.append({
                    "Sample": i + 1,
                    "sigma_3": sigma_3,
                    "Deviator": s["deviator"],
                    "sigma_1": sigma_1,
                    "u": s["u"],
                    "sigma_1_eff": sigma_1 - s["u"],
                    "sigma_3_eff": sigma_3 - s["u"]
                })
            
            st.dataframe(pd.DataFrame(results_data))
            
            c, phi_rad = fit_envelope_to_circles(circles)
            phi_deg = math.degrees(phi_rad)
            
            st.success(f"**Fitted Envelope:** c' = {c:.2f} kPa, phi' = {phi_deg:.1f} deg")
            
            envelope_ref = st.session_state.catalog.get_formula("ENVELOPE_FROM_MOHR_CIRCLES")
            if envelope_ref:
                st.markdown(f"""
                <div class="step-box">
                <b>Method:</b> {envelope_ref.name}<br>
                <b>Reference:</b> <span class="ref-label">{envelope_ref.reference_label}</span><br>
                <b>Notes:</b> {envelope_ref.notes}
                </div>
                """, unsafe_allow_html=True)
            
            labels = [f"Sample {i+1}" for i in range(len(circles))]
            fig = plot_mohr_circles(circles, c, phi_rad, labels, f"{test_type} Triaxial Test Results")
            st.pyplot(fig)
            plt.close(fig)
            
            st.session_state.triaxial_data = circles


def tab_mohr():
    """Mohr Circles & Failure Envelope tab."""
    st.markdown('<div class="main-header">Mohr Circles & Failure Envelope</div>', unsafe_allow_html=True)
    
    mc_formula = st.session_state.catalog.get_formula("MC_FAILURE")
    if mc_formula:
        st.success(f"Envelope form: {mc_formula.equation} (Ref: {mc_formula.reference_label})")
    
    st.markdown('<div class="section-header">Enter Test Data</div>', unsafe_allow_html=True)
    
    input_method = st.radio("Input Method", ["Enter sigma_1, sigma_3", "Use previous triaxial data"])
    
    circles = []
    
    if input_method == "Enter sigma_1, sigma_3":
        num_circles = st.number_input("Number of Circles", min_value=1, max_value=10, value=3)
        
        for i in range(int(num_circles)):
            col1, col2 = st.columns(2)
            with col1:
                s1 = st.number_input(f"Circle {i+1}: sigma_1 (kPa)", value=150.0 + i*50, key=f"mohr_s1_{i}")
            with col2:
                s3 = st.number_input(f"Circle {i+1}: sigma_3 (kPa)", value=50.0 + i*50, key=f"mohr_s3_{i}")
            circles.append((s1, s3))
    else:
        if st.session_state.triaxial_data:
            circles = st.session_state.triaxial_data
            st.info(f"Using {len(circles)} circles from triaxial analysis")
        else:
            st.warning("No previous triaxial data. Run triaxial analysis first.")
            return
    
    st.markdown("---")
    st.markdown('<div class="section-header">Failure Envelope</div>', unsafe_allow_html=True)
    
    envelope_method = st.radio("Envelope Method", ["Manual: Adjust c and phi", "Assisted: Fit tangent line"])
    
    if envelope_method == "Manual: Adjust c and phi":
        col1, col2 = st.columns(2)
        with col1:
            c = st.slider("Cohesion c (kPa)", 0.0, 100.0, 20.0, 1.0)
        with col2:
            phi_deg = st.slider("Friction angle phi (deg)", 0.0, 45.0, 30.0, 0.5)
        phi_rad = math.radians(phi_deg)
    else:
        c, phi_rad = fit_envelope_to_circles(circles)
        phi_deg = math.degrees(phi_rad)
        st.success(f"**Fitted:** c = {c:.2f} kPa, phi = {phi_deg:.1f} deg")
        
        envelope_ref = st.session_state.catalog.get_formula("ENVELOPE_FROM_MOHR_CIRCLES")
        if envelope_ref:
            st.caption(f"Method: {envelope_ref.notes} (Ref: {envelope_ref.reference_label})")
    
    labels = [f"Test {i+1}" for i in range(len(circles))]
    fig = plot_mohr_circles(circles, c, phi_rad, labels, "Mohr Circles with Failure Envelope")
    st.pyplot(fig)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Export as PNG"):
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            st.download_button("Download PNG", buf.getvalue(), "mohr_circles.png", "image/png")
    with col2:
        if st.button("Export as SVG"):
            buf = io.BytesIO()
            fig.savefig(buf, format='svg', bbox_inches='tight')
            st.download_button("Download SVG", buf.getvalue(), "mohr_circles.svg", "image/svg+xml")
    
    plt.close(fig)


def tab_reports():
    """Reports & Export tab."""
    st.markdown('<div class="main-header">Reports & Export</div>', unsafe_allow_html=True)
    
    engine = get_engine()
    
    st.markdown('<div class="section-header">Calculation Log</div>', unsafe_allow_html=True)
    
    if st.session_state.calculation_log:
        for i, step in enumerate(st.session_state.calculation_log, 1):
            st.markdown(f"""
            <div class="step-box">
            <b>Step {i}: {step.formula_name}</b><br>
            <b>ID:</b> <code>{step.formula_id}</code> | 
            <b>Reference:</b> <span class="ref-label">{step.reference_label}</span><br>
            <b>Equation:</b> {step.equation_symbolic}<br>
            <b>Calculation:</b> {step.equation_substituted}<br>
            <b>Result:</b> {step.result:.4g} {step.result_unit}
            </div>
            """, unsafe_allow_html=True)
        
        if st.button("Clear Log"):
            st.session_state.calculation_log = []
            st.rerun()
    else:
        st.info("No calculations logged yet. Perform calculations in other tabs.")
    
    st.markdown("---")
    st.markdown('<div class="section-header">Export Options</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export Catalog as JSON"):
            st.download_button(
                "Download Catalog",
                st.session_state.catalog.to_json(),
                "formula_catalog.json",
                "application/json"
            )
    
    with col2:
        if st.button("Generate HTML Report"):
            html = generate_html_report()
            st.download_button(
                "Download HTML Report",
                html,
                "geotech_report.html",
                "text/html"
            )


def generate_html_report() -> str:
    """Generate complete HTML report."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Geotechnical Analysis Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 2rem; }
            h1 { color: #2c3e50; border-bottom: 2px solid #3498db; }
            .step { background: #f8f9fa; padding: 1rem; margin: 0.5rem 0; border-left: 4px solid #3498db; }
            .ref { background: #eee; padding: 0.2rem 0.5rem; font-family: monospace; }
            table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
            th, td { border: 1px solid #ddd; padding: 0.5rem; text-align: left; }
            th { background: #3498db; color: white; }
        </style>
    </head>
    <body>
        <h1>Geotechnical Analysis Report</h1>
        <p><b>Generated by:</b> Course-Strict Geotechnical Analysis App</p>
        <p><b>Course Mode:</b> """ + ("STRICT" if st.session_state.strict_mode else "Normal") + """</p>
        
        <h2>Calculation Steps</h2>
    """
    
    if st.session_state.calculation_log:
        for i, step in enumerate(st.session_state.calculation_log, 1):
            html += f"""
            <div class="step">
                <h3>Step {i}: {step.formula_name}</h3>
                <p><b>Formula ID:</b> <code>{step.formula_id}</code></p>
                <p><b>Reference:</b> <span class="ref">{step.reference_label}</span></p>
                <p><b>Equation:</b> {step.equation_symbolic}</p>
                <p><b>Calculation:</b> {step.equation_substituted}</p>
                <p><b>Result:</b> <strong>{step.result:.4g} {step.result_unit}</strong></p>
            </div>
            """
    else:
        html += "<p>No calculations recorded.</p>"
    
    html += """
        <h2>Formula Catalog Used</h2>
        <table>
            <tr><th>ID</th><th>Name</th><th>Equation</th><th>Reference</th><th>Status</th></tr>
    """
    
    for f in st.session_state.catalog.formulas.values():
        status = "Enabled" if f.enabled and not f.requires_confirmation else "Pending" if f.requires_confirmation else "Disabled"
        html += f"<tr><td>{f.id}</td><td>{f.name}</td><td>{f.equation}</td><td>{f.reference_label}</td><td>{status}</td></tr>"
    
    html += """
        </table>
    </body>
    </html>
    """
    return html


import io

if __name__ == "__main__":
    main()
