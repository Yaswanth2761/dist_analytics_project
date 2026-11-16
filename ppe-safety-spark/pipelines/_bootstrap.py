"""Makes the repo root importable when a stage is run as `python pipelines/pXX_*.py`."""
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
