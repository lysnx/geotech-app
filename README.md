# Geotechnical Analysis Application

A professional Python application for geotechnical engineering analysis with **Course-Strict Mode** - all calculations trace back to your course formulas.

## Features

### Course-Strict App (`course_geotech_app/`)
- **Formula Catalog**: Editable JSON catalog - add/edit/remove formulas to match YOUR course
- **Strict Mode Toggle**: Block any calculation not backed by catalog formulas
- **Full Traceability**: Every result shows formula ID and course reference
- **Triaxial Tests (CD/CU/UU)**: Multiple samples with automatic envelope fitting
- **Direct Shear Test**: Plot tau vs sigma, compute c and phi
- **Mohr Circles & Envelope**: Standard convention (sigma on X, tau on Y)
- **Reports & Export**: HTML/PDF with formula traceability

### Standard App (`streamlit_app.py`)
- Consolidation analysis with Terzaghi theory
- Mohr-Coulomb failure criterion evaluation
- Safety factor calculation

## Quick Start

```bash
# Clone the repository
git clone https://github.com/lysnx/geotech-app.git
cd geotech-app

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run Course-Strict App (RECOMMENDED)
streamlit run course_geotech_app/app.py

# Or run Standard App
streamlit run streamlit_app.py
```

## Project Structure

```
geotech-app/
 course_geotech_app/     # Course-strict application
    app.py              # Main Streamlit app
    catalog/            # Formula catalog (JSON + parser)
    engine/             # Calculation engine with traceability
    plotting/           # Mohr circle visualization
 src/                    # Core modules
    core/               # Models, physics, simulation
    vis/                # Plotting functions
 streamlit_app.py        # Standard Streamlit app
 requirements.txt
 README.md
```

## Default Formula Catalog

The app comes preloaded with Chapter 6 formulas:

| ID | Name | Reference |
|----|------|-----------|
| MC_FAILURE | Mohr-Coulomb Failure | Chap6 p4 |
| EFFECTIVE_STRESS | Effective Stress Principle | Chap6 p6 |
| TRIAXIAL_TEST_TYPES | CD/CU/UU Rules | Chap6 p13-14 |
| ENVELOPE_FROM_MOHR_CIRCLES | Graphical Envelope | Chap6 p17 |
| UU_STRENGTH | Undrained Strength | Chap6 p20 |

## Requirements

- Python 3.8+
- Streamlit, NumPy, Matplotlib, Pandas

## License

MIT License

## Contributors

- Amir Wechtati
- Mohamed Amine Sliti
- Mohamed Chandoul
- Badis Zammouri
