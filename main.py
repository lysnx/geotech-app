# -*- coding: utf-8 -*-
"""
Geotechnical Consolidation Analysis - Streamlit GUI

A professional application for:
- Plane strain consolidation analysis
- Mohr-Coulomb failure criterion evaluation
- Interactive visualization

Run with: streamlit run main.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

# Add src to path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.models import (
    SoilProperties, FoundationGeometry, AnalysisParameters,
    MohrCircle, FailureEnvelope, AnalysisResult
)
from src.core.simulation import run_consolidation_analysis, get_results_as_dict
from src.vis.plotting import (
    plot_mohr_diagram, plot_single_mohr_diagram,
    plot_stress_profiles, plot_safety_factor_evolution
)

# Page configuration
st.set_page_config(
    page_title="Geotechnical Analysis",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #7f8c8d;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
    }
    .safe-status {
        color: #27ae60;
        font-weight: bold;
    }
    .warning-status {
        color: #f39c12;
        font-weight: bold;
    }
    .failed-status {
        color: #e74c3c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main application entry point."""
    
    # Sidebar navigation
    st.sidebar.title(" Navigation")
    page = st.sidebar.radio(
        "Select Page",
        [" Home", " New Analysis", " Results Explorer", " Help"]
    )
    
    if page == " Home":
        show_home_page()
    elif page == " New Analysis":
        show_analysis_page()
    elif page == " Results Explorer":
        show_results_explorer()
    elif page == " Help":
        show_help_page()


def show_home_page():
    """Display the home/landing page."""
    st.markdown('<p class="main-header"> Geotechnical Consolidation Analysis</p>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Interactive Mohr-Coulomb Failure Analysis with Terzaghi Consolidation</p>', 
                unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("###  Features")
        st.markdown("""
        - **Terzaghi Consolidation Theory** - Pore pressure dissipation over time
        - **Boussinesq Stress Distribution** - Foundation-induced stresses
        - **Mohr-Coulomb Analysis** - Failure criterion evaluation
        - **Interactive Visualization** - Publication-quality diagrams
        - **Safety Factor Tracking** - Critical condition detection
        """)
    
    with col2:
        st.markdown("###  Quick Start")
        st.markdown("""
        1. Go to ** New Analysis** page
        2. Select soil type or enter custom properties
        3. Define foundation geometry and loading
        4. Click **Run Analysis**
        5. Explore results in ** Results Explorer**
        """)
    
    st.divider()
    
    # Quick demo
    st.markdown("###  Sample Mohr Circle")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        demo_sigma_1 = st.slider("s' (kPa)", 50, 300, 150)
        demo_sigma_3 = st.slider("s' (kPa)", 10, 150, 50)
        demo_c = st.slider("c' (kPa)", 0, 50, 20)
        demo_phi = st.slider("f' (degrees)", 0, 45, 30)
    
    with col2:
        circle = MohrCircle(demo_sigma_1, demo_sigma_3)
        envelope = FailureEnvelope(demo_c, demo_phi)
        fs = envelope.calculate_safety_factor(circle)
        
        fig = plot_mohr_diagram([circle], envelope, ["Demo"], 
                               f"Interactive Mohr Circle (FS = {fs:.2f})")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    
    with col3:
        st.metric("Safety Factor", f"{fs:.2f}")
        if fs < 1.0:
            st.error(" FAILED")
        elif fs < 1.5:
            st.warning(" MARGINAL")
        else:
            st.success(" SAFE")


def show_analysis_page():
    """Display the analysis input and execution page."""
    st.markdown("##  New Consolidation Analysis")
    
    # Initialize session state
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    
    tab1, tab2, tab3 = st.tabs([" Soil Properties", " Foundation", " Analysis"])
    
    with tab1:
        st.markdown("### Soil Properties")
        
        preset = st.selectbox(
            "Select Soil Preset",
            ["Custom", "Soft Clay", "Stiff Clay", "Dense Sand", "Loose Sand", "Silty Clay"]
        )
        
        if preset != "Custom":
            soil = SoilProperties.from_preset(preset)
            cohesion_c = st.number_input("Cohesion c' (kPa)", value=soil.cohesion_c, disabled=True)
            friction_phi = st.number_input("Friction Angle f' (°)", value=soil.friction_angle_phi, disabled=True)
            gamma_dry = st.number_input("Dry Unit Weight (kN/m)", value=soil.unit_weight_dry, disabled=True)
            gamma_sat = st.number_input("Saturated Unit Weight (kN/m)", value=soil.unit_weight_sat, disabled=True)
            poisson = st.number_input("Poisson Ratio", value=soil.poisson_ratio, disabled=True)
            cv = st.number_input("Consolidation Coefficient (m²/year)", value=soil.consolidation_cv, disabled=True)
        else:
            cohesion_c = st.number_input("Cohesion c' (kPa)", value=15.0, min_value=0.0, max_value=200.0)
            friction_phi = st.number_input("Friction Angle f' (°)", value=25.0, min_value=0.0, max_value=45.0)
            gamma_dry = st.number_input("Dry Unit Weight (kN/m)", value=17.0, min_value=10.0, max_value=25.0)
            gamma_sat = st.number_input("Saturated Unit Weight (kN/m)", value=19.0, min_value=10.0, max_value=25.0)
            poisson = st.number_input("Poisson Ratio", value=0.3, min_value=0.1, max_value=0.5)
            cv = st.number_input("Consolidation Coefficient (m²/year)", value=2.0, min_value=0.01, max_value=100.0)
            
            soil = SoilProperties(
                cohesion_c=cohesion_c,
                friction_angle_phi=friction_phi,
                unit_weight_dry=gamma_dry,
                unit_weight_sat=gamma_sat,
                poisson_ratio=poisson,
                consolidation_cv=cv,
                description="Custom"
            )
    
    with tab2:
        st.markdown("### Foundation Geometry")
        
        col1, col2 = st.columns(2)
        with col1:
            width_B = st.number_input("Width B (m)", value=5.0, min_value=0.5, max_value=50.0)
            length_L = st.number_input("Length L (m)", value=5.0, min_value=0.5, max_value=50.0)
        with col2:
            depth_Df = st.number_input("Embedment Depth (m)", value=2.0, min_value=0.0, max_value=20.0)
            applied_q = st.number_input("Applied Stress q (kPa)", value=150.0, min_value=0.0, max_value=1000.0)
        
        foundation = FoundationGeometry(
            width_B=width_B,
            length_L=length_L,
            depth_Df=depth_Df,
            applied_stress_q=applied_q
        )
    
    with tab3:
        st.markdown("### Analysis Parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            max_depth = st.number_input("Max Depth (m)", value=10.0, min_value=5.0, max_value=50.0)
            water_table = st.number_input("Water Table Depth (m)", value=2.0, min_value=0.0, max_value=20.0)
        with col2:
            layer_thickness = st.number_input("Compressible Layer (m)", value=4.0, min_value=1.0, max_value=20.0)
            depth_increment = st.number_input("Depth Increment (m)", value=0.5, min_value=0.1, max_value=2.0)
        
        time_stages_str = st.text_input("Time Stages (days, comma-separated)", "0, 1, 7, 30, 365")
        time_stages = [float(t.strip()) for t in time_stages_str.split(",")]
        
        params = AnalysisParameters(
            max_depth=max_depth,
            water_table_depth=water_table,
            depth_increment=depth_increment,
            time_stages=time_stages,
            layer_thickness=layer_thickness
        )
    
    st.divider()
    
    # Run Analysis Button
    if st.button(" Run Analysis", type="primary", use_container_width=True):
        with st.spinner("Running consolidation analysis..."):
            result = run_consolidation_analysis(soil, foundation, params)
            st.session_state.analysis_result = result
            st.success(" Analysis completed!")
            
            # Show summary
            critical = result.find_critical_condition()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Min Safety Factor", f"{critical['min_fs']:.2f}" if critical['min_fs'] else "N/A")
            with col2:
                st.metric("Critical Depth", f"{critical['depth']:.1f} m" if critical['depth'] else "N/A")
            with col3:
                st.metric("Critical Time", f"{critical['time_days']:.0f} days" if critical['time_days'] else "N/A")
            with col4:
                status = critical.get('status', 'N/A')
                if status == 'SAFE':
                    st.success(f"Status: {status}")
                elif status == 'FAILED':
                    st.error(f"Status: {status}")
                else:
                    st.warning(f"Status: {status}")


def show_results_explorer():
    """Display the interactive results explorer."""
    st.markdown("##  Results Explorer")
    
    if st.session_state.get("analysis_result") is None:
        st.warning(" No analysis results available. Please run an analysis first.")
        st.page_link("main.py", label="Go to New Analysis", icon="")
        return
    
    result = st.session_state.analysis_result
    results_dict = get_results_as_dict(result)
    
    tab1, tab2, tab3, tab4 = st.tabs([" Mohr Diagram", " Stress Profiles", " Time Evolution", " Data Table"])
    
    with tab1:
        st.markdown("### Interactive Mohr Circle Diagram")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            available_times = sorted(set(t for _, t in result.stress_states.keys()))
            available_depths = sorted(set(d for d, _ in result.stress_states.keys()))
            
            selected_time = st.select_slider("Time (days)", options=available_times, value=available_times[0])
            selected_depth = st.select_slider("Depth (m)", options=available_depths, value=available_depths[len(available_depths)//2])
            
            show_all_times = st.checkbox("Show all time stages", value=True)
        
        with col2:
            envelope = FailureEnvelope(
                result.soil.cohesion_c,
                result.soil.friction_angle_phi
            )
            
            if show_all_times:
                circles = []
                labels = []
                for t in available_times:
                    key = (selected_depth, t)
                    if key in result.mohr_circles:
                        circles.append(result.mohr_circles[key])
                        labels.append(f"t={t:.0f}d")
                
                fig = plot_mohr_diagram(circles, envelope, labels,
                    f"Mohr Circles at z={selected_depth:.1f}m - Consolidation Progress")
            else:
                key = (selected_depth, selected_time)
                if key in result.mohr_circles:
                    circle = result.mohr_circles[key]
                    fs = envelope.calculate_safety_factor(circle)
                    fig = plot_mohr_diagram([circle], envelope, [f"t={selected_time:.0f}d"],
                        f"Mohr Circle at z={selected_depth:.1f}m, t={selected_time:.0f}d (FS={fs:.2f})")
                else:
                    st.error("No data for selected depth/time")
                    return
            
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
    
    with tab2:
        st.markdown("### Stress Profiles vs Depth")
        
        fig = plot_stress_profiles(results_dict, result.parameters.time_stages)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    
    with tab3:
        st.markdown("### Safety Factor Evolution")
        
        available_depths = sorted(set(d for d, _ in result.stress_states.keys()))
        critical_depth = st.select_slider("Select depth for time evolution", 
                                          options=available_depths,
                                          value=available_depths[len(available_depths)//2])
        
        fig = plot_safety_factor_evolution(results_dict, critical_depth)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    
    with tab4:
        st.markdown("### Results Data Table")
        
        import pandas as pd
        
        data = []
        for (depth, time), stress in result.stress_states.items():
            fs = result.safety_factors.get((depth, time), 0)
            data.append({
                "Depth (m)": depth,
                "Time (days)": time,
                "s_v Total (kPa)": f"{stress.sigma_v_total:.1f}",
                "Pore Pressure (kPa)": f"{stress.pore_pressure:.1f}",
                "s'_v Eff (kPa)": f"{stress.sigma_v_eff:.1f}",
                "s'_1 (kPa)": f"{stress.sigma_1:.1f}",
                "s'_3 (kPa)": f"{stress.sigma_3:.1f}",
                "Safety Factor": f"{fs:.2f}"
            })
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label=" Download CSV",
            data=csv,
            file_name="geotechnical_analysis.csv",
            mime="text/csv"
        )


def show_help_page():
    """Display help and documentation."""
    st.markdown("##  Help & Documentation")
    
    with st.expander(" Mohr-Coulomb Failure Criterion", expanded=True):
        st.markdown(r"""
        The **Mohr-Coulomb failure criterion** defines the shear strength of soil:
        
        $$\tau_f = c' + \sigma' \cdot \tan(\phi')$$
        
        Where:
        - $\tau_f$ = Shear strength at failure (kPa)
        - $c'$ = Effective cohesion (kPa)
        - $\sigma'$ = Effective normal stress (kPa)
        - $\phi'$ = Effective friction angle (degrees)
        
        **Safety Factor:**
        $$FS = \frac{R_{failure}}{R_{circle}} = \frac{c' \cos\phi' + \sigma_{center} \sin\phi'}{(\sigma'_1 - \sigma'_3)/2}$$
        """)
    
    with st.expander(" Terzaghi Consolidation Theory"):
        st.markdown(r"""
        Consolidation describes how pore pressure dissipates over time:
        
        **Time Factor:**
        $$T_v = \frac{c_v \cdot t}{H_d^2}$$
        
        **Consolidation Degree:**
        - For $U < 60\%$: $U \approx \sqrt{\frac{4 T_v}{\pi}}$
        - For $U \geq 60\%$: Use series solution
        
        **Pore Pressure:**
        $$u(t) = u_0 \cdot (1 - U)$$
        
        **Effective Stress (Terzaghi's Principle):**
        $$\sigma' = \sigma_{total} - u$$
        """)
    
    with st.expander(" References"):
        st.markdown("""
        - [Mohr-Coulomb Failure Criteria](https://www.geological-digressions.com/mohr-coulomb-failure-criteria/)
        - Terzaghi, K. (1943). Theoretical Soil Mechanics
        - Das, B.M. (2010). Principles of Geotechnical Engineering
        """)


if __name__ == "__main__":
    main()
