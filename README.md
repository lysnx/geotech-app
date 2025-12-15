# Geotechnical Analysis Application

A professional Python application for geotechnical engineering analysis, featuring Mohr Circle visualization and failure envelope calculations.

## Features

- **Mohr Circle Analysis**: Visualize stress states with publication-quality Mohr circle diagrams
- **Failure Envelope Calculation**: Compute cohesion and friction angle from test data
- **Interactive GUI**: Streamlit-based web interface for easy data input and visualization
- **Plane Strain Consolidation**: Analysis with Mohr-Coulomb failure criterion evaluation

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/geotech-app.git
   cd geotech-app
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # or
   source .venv/bin/activate  # Linux/Mac
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Streamlit Web Application
```bash
streamlit run streamlit_app.py
```

### Command Line
```bash
python main.py
```

## Project Structure

```
geotech-app/
 src/                    # Source code modules
 streamlit_app.py        # Streamlit web application
 main.py                 # Main application entry point
 requirements.txt        # Python dependencies
 README.md
```

## Requirements

- Python 3.8+
- See requirements.txt for full dependency list

## License

MIT License
