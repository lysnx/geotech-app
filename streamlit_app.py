# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.core.models import SoilProperties, FoundationGeometry, AnalysisParameters, MohrCircle, FailureEnvelope
from src.core.simulation import run_consolidation_analysis, get_results_as_dict
from src.vis.plotting import plot_mohr_diagram, plot_stress_profiles, plot_safety_factor_evolution

st.set_page_config(page_title='Geotechnical Analysis', page_icon='', layout='wide', initial_sidebar_state='expanded')

st.markdown('''<style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #2c3e50; text-align: center; padding: 1rem 0; border-bottom: 3px solid #3498db; margin-bottom: 2rem;}
    .section-header {font-size: 1.5rem; font-weight: bold; color: #34495e; padding: 0.5rem 0; border-left: 4px solid #3498db; padding-left: 1rem; margin: 1rem 0;}
</style>''', unsafe_allow_html=True)

SOIL_PRESETS = {
    'Soft Clay': {'cohesion_c': 10.0, 'friction_angle_phi': 22.0, 'unit_weight_dry': 16.0, 'unit_weight_sat': 18.0, 'poisson_ratio': 0.35, 'consolidation_cv': 1.0, 'ocr': 1.0, 'desc': 'Soft Marine Clay'},
    'Stiff Clay': {'cohesion_c': 25.0, 'friction_angle_phi': 28.0, 'unit_weight_dry': 18.0, 'unit_weight_sat': 20.0, 'poisson_ratio': 0.30, 'consolidation_cv': 3.0, 'ocr': 2.0, 'desc': 'Overconsolidated stiff clay'},
    'Dense Sand': {'cohesion_c': 0.0, 'friction_angle_phi': 35.0, 'unit_weight_dry': 17.0, 'unit_weight_sat': 20.0, 'poisson_ratio': 0.25, 'consolidation_cv': 100.0, 'ocr': 1.0, 'desc': 'Dense granular sand'},
    'Loose Sand': {'cohesion_c': 0.0, 'friction_angle_phi': 28.0, 'unit_weight_dry': 15.0, 'unit_weight_sat': 18.0, 'poisson_ratio': 0.30, 'consolidation_cv': 50.0, 'ocr': 1.0, 'desc': 'Loose sand'},
    'Silty Clay': {'cohesion_c': 15.0, 'friction_angle_phi': 25.0, 'unit_weight_dry': 17.0, 'unit_weight_sat': 19.0, 'poisson_ratio': 0.32, 'consolidation_cv': 2.0, 'ocr': 1.5, 'desc': 'Mixed silty clay'},
}

def init_session():
    for key in ['analysis_complete', 'results', 'results_dict', 'soil', 'foundation', 'params']:
        if key not in st.session_state:
            st.session_state[key] = None if key != 'analysis_complete' else False

def main():
    init_session()
    st.sidebar.markdown('## Navigation')
    page = st.sidebar.radio('Select Page', ['Home', 'Input Parameters', 'Analysis Results', 'Mohr Explorer', 'Help'])
    st.sidebar.markdown('---')
    st.sidebar.markdown('### Status')
    if st.session_state.analysis_complete and st.session_state.results:
        crit = st.session_state.results.find_critical_condition()
        if crit['min_fs']:
            fs = crit['min_fs']
            if fs < 1.0: st.sidebar.error(f'Min FS: {fs:.2f} (FAILED)')
            elif fs < 1.5: st.sidebar.warning(f'Min FS: {fs:.2f} (MARGINAL)')
            else: st.sidebar.success(f'Min FS: {fs:.2f} (SAFE)')
    else:
        st.sidebar.info('No analysis yet')
    
    if page == 'Home': show_home()
    elif page == 'Input Parameters': show_input()
    elif page == 'Analysis Results': show_results()
    elif page == 'Mohr Explorer': show_explorer()
    elif page == 'Help': show_help()

def show_home():
    st.markdown('<div class="main-header">Geotechnical Analysis System</div>', unsafe_allow_html=True)
    st.markdown('''Welcome to the **Geotechnical Soil Analysis Application**. This tool performs:
    - Plane strain consolidation analysis using Terzaghi theory
    - Mohr-Coulomb failure criterion evaluation
    - Publication-quality visualizations of stress states
    - Safety factor calculations at any depth and time''')

def show_input():
    st.markdown('<div class="main-header">Input Parameters</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Soil Properties</div>', unsafe_allow_html=True)
        preset = st.selectbox('Select Soil Preset', list(SOIL_PRESETS.keys()))
        p = SOIL_PRESETS[preset]
        st.info(p['desc'])
        cohesion = st.number_input('Cohesion c (kPa)', 0.0, 200.0, p['cohesion_c'])
        friction = st.number_input('Friction Angle (deg)', 0.0, 45.0, p['friction_angle_phi'])
        gamma_dry = st.number_input('Dry Unit Weight (kN/m3)', 10.0, 25.0, p['unit_weight_dry'])
        gamma_sat = st.number_input('Sat Unit Weight (kN/m3)', 10.0, 25.0, p['unit_weight_sat'])
        poisson = st.number_input('Poisson Ratio', 0.0, 0.5, p['poisson_ratio'])
        cv = st.number_input('Consolidation Cv (m2/yr)', 0.1, 500.0, p['consolidation_cv'])
    with col2:
        st.markdown('<div class="section-header">Foundation Geometry</div>', unsafe_allow_html=True)
        width = st.number_input('Width B (m)', 0.5, 50.0, 5.0)
        length = st.number_input('Length L (m)', 0.5, 50.0, 5.0)
        depth_f = st.number_input('Foundation Depth Df (m)', 0.0, 10.0, 2.0)
        applied_q = st.number_input('Applied Stress q (kPa)', 0.0, 1000.0, 150.0)
        st.markdown('<div class="section-header">Analysis Parameters</div>', unsafe_allow_html=True)
        max_depth = st.number_input('Max Analysis Depth (m)', 5.0, 50.0, 10.0)
        water_table = st.number_input('Water Table Depth (m)', 0.0, max_depth, 2.0)
        depth_incr = st.number_input('Depth Increment (m)', 0.1, 2.0, 0.5)
        time_stages = st.text_input('Time Stages (days, comma-separated)', '0, 1, 7, 30, 365')
    if st.button('Run Analysis', type='primary', use_container_width=True):
        try:
            times = [float(t.strip()) for t in time_stages.split(',')]
            soil = SoilProperties(cohesion_c=cohesion, friction_angle_phi=friction, unit_weight_dry=gamma_dry, unit_weight_sat=gamma_sat, poisson_ratio=poisson, consolidation_cv=cv)
            foundation = FoundationGeometry(width_B=width, length_L=length, depth_Df=depth_f, applied_stress_q=applied_q)
            params = AnalysisParameters(max_depth=max_depth, water_table_depth=water_table, depth_increment=depth_incr, time_stages=times)
            with st.spinner('Running consolidation analysis...'):
                results = run_consolidation_analysis(soil, foundation, params)
                results_dict = get_results_as_dict(results)
            st.session_state.results = results
            st.session_state.results_dict = results_dict
            st.session_state.soil = soil
            st.session_state.foundation = foundation
            st.session_state.params = params
            st.session_state.analysis_complete = True
            st.success('Analysis complete! Go to Analysis Results to view.')
            crit = results.find_critical_condition()
            if crit['min_fs']:
                c1, c2, c3 = st.columns(3)
                c1.metric('Minimum FS', f"{crit['min_fs']:.2f}")
                c2.metric('Critical Depth', f"{crit['depth']:.1f} m")
                c3.metric('Critical Time', f"{crit['time_days']:.0f} days")
        except Exception as e:
            st.error(f'Error: {str(e)}')

def show_results():
    st.markdown('<div class="main-header">Analysis Results</div>', unsafe_allow_html=True)
    if not st.session_state.analysis_complete:
        st.warning('No analysis results. Please run an analysis first.')
        return
    results = st.session_state.results
    results_dict = st.session_state.results_dict
    soil = st.session_state.soil
    tab1, tab2, tab3 = st.tabs(['Critical Condition', 'Stress Profiles', 'FS Evolution'])
    with tab1:
        crit = results.find_critical_condition()
        if crit['min_fs']:
            c1, c2, c3, c4 = st.columns(4)
            fs = crit['min_fs']
            c1.metric('Safety Factor', f'{fs:.2f}', delta='FAILED' if fs < 1 else ('MARGINAL' if fs < 1.5 else 'SAFE'))
            c2.metric('Critical Depth', f"{crit['depth']:.1f} m")
            c3.metric('Critical Time', f"{crit['time_days']:.0f} days")
            c4.metric('Status', crit['status'])
            st.markdown('---')
            key = (crit['depth'], crit['time_days'])
            if key in results.mohr_circles:
                mohr = results.mohr_circles[key]
                envelope = FailureEnvelope(cohesion_c=soil.cohesion_c, friction_angle_phi=soil.friction_angle_phi)
                fig = plot_mohr_diagram([mohr], envelope, [f"z={crit['depth']:.1f}m"], title=f'Critical Mohr Circle (FS={fs:.2f})')
                st.pyplot(fig)
                plt.close(fig)
    with tab2:
        if results_dict:
            fig = plot_stress_profiles(results_dict, st.session_state.params.time_stages)
            st.pyplot(fig)
            plt.close(fig)
    with tab3:
        depths = sorted(set(k[0] for k in results.safety_factors.keys()))
        selected_depth = st.select_slider('Select Depth for FS Evolution', options=depths, value=depths[len(depths)//2] if depths else 0)
        if results_dict:
            fig = plot_safety_factor_evolution(results_dict, selected_depth)
            st.pyplot(fig)
            plt.close(fig)

def show_explorer():
    st.markdown('<div class="main-header">Mohr Circle Explorer</div>', unsafe_allow_html=True)
    if not st.session_state.analysis_complete:
        st.warning('No analysis results. Please run an analysis first.')
        return
    results = st.session_state.results
    soil = st.session_state.soil
    depths = sorted(set(k[0] for k in results.mohr_circles.keys()))
    times = sorted(set(k[1] for k in results.mohr_circles.keys()))
    c1, c2 = st.columns(2)
    with c1:
        selected_depth = st.select_slider('Depth (m)', options=depths, value=depths[len(depths)//2] if depths else 0)
    with c2:
        selected_time = st.select_slider('Time (days)', options=times, value=times[-1] if times else 0)
    key = (selected_depth, selected_time)
    if key in results.mohr_circles:
        mohr = results.mohr_circles[key]
        fs = results.safety_factors.get(key, 0)
        envelope = FailureEnvelope(cohesion_c=soil.cohesion_c, friction_angle_phi=soil.friction_angle_phi)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric('sigma1 (kPa)', f'{mohr.sigma_1:.1f}')
        c2.metric('sigma3 (kPa)', f'{mohr.sigma_3:.1f}')
        c3.metric('Center (kPa)', f'{mohr.center:.1f}')
        c4.metric('Radius (kPa)', f'{mohr.radius:.1f}')
        c5.metric('Safety Factor', f'{fs:.2f}')
        st.markdown('---')
        fig = plot_mohr_diagram([mohr], envelope, [f'z={selected_depth:.1f}m, t={selected_time:.0f}d'], title=f'Mohr Circle at z={selected_depth:.1f}m, t={selected_time:.0f}d')
        st.pyplot(fig)
        plt.close(fig)

def show_help():
    st.markdown('<div class="main-header">Help and Documentation</div>', unsafe_allow_html=True)
    with st.expander('Physics Overview', expanded=True):
        st.markdown('''### Terzaghi Effective Stress Principle
sigma_eff = sigma_total - u

### Mohr-Coulomb Failure Criterion
tau_f = c + sigma_eff * tan(phi)''')
    with st.expander('Safety Factor'):
        st.markdown('''FS = R_failure / R_circle

- **FS > 1.5**: Safe
- **1.0 < FS < 1.5**: Marginal
- **FS < 1.0**: Failed''')

if __name__ == '__main__':
    main()
