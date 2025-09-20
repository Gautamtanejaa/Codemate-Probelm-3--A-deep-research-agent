"""
Main entry point for Deep Researcher Agent CLI.
"""

import sys
from pathlib import Path

# Add the current directory to the path
sys.path.append(str(Path(__file__).parent))

from deep_researcher.cli import cli

if __name__ == "__main__":
    cli()
