"""
This file has been moved to multimind/cli/
Please use the CLI module instead.
"""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import from multimind
sys.path.append(str(Path(__file__).parent.parent.parent))

from multimind.cli import cli

if __name__ == "__main__":
    cli()