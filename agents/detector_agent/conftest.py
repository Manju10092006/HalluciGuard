"""Make the detector_agent package importable from its own test files.

The detector tests use `from detector_agent import ...` imports, which require
the parent ``agents/`` directory on ``sys.path`` (the repo-level pytest.ini only
adds the project root and agents/verifier_agent).
"""

import os
import sys

_AGENTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _AGENTS_DIR not in sys.path:
    sys.path.insert(0, _AGENTS_DIR)