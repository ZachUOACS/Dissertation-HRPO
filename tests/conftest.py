import os
import sys

# Make the repo root importable so tests can `import hrpo_ext` regardless of pytest's cwd.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
