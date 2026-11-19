"""Make `app.*` importable from the backend package for every test module."""

import os
import sys

BACKEND = os.path.join(os.path.dirname(__file__), "..", "backend")
if BACKEND not in sys.path:
    sys.path.insert(0, os.path.abspath(BACKEND))
