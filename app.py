"""
Main entry point for Deep Researcher Agent Streamlit app.
"""

import sys
from pathlib import Path

# Add the current directory to the path
sys.path.append(str(Path(__file__).parent))

from deep_researcher.app import main

if __name__ == "__main__":
    main()
