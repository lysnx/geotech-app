"""
Geotechnical Analysis System - Entry Point
Interactive plane strain consolidation analysis with Mohr-Coulomb failure criterion.
"""

import sys
import os
import argparse

# Add the parent directory to path for imports when running directly
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description="Geotechnical Soil Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/main.py              Launch interactive mode
  python src/main.py --cli        Run with command-line arguments
  python src/main.py --config config.json  Use configuration file
        """
    )
    parser.add_argument('--cli', action='store_true', help='Use non-interactive CLI mode')
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    
    args = parser.parse_args()
    
    if args.cli or args.config:
        from src.core.cli import run_cli_mode
        run_cli_mode(args)
    else:
        try:
            from src.ui.app import GeotechApp
            app = GeotechApp()
            app.run()
        except ImportError as e:
            print(f"Error: Could not import UI module. {e}")
            print("Try installing dependencies: pip install rich")
            sys.exit(1)
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)


if __name__ == "__main__":
    main()
