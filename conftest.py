"""Root conftest.py – makes the project root a pytest rootdir so that
`src` and `app` are importable as top-level packages without installation."""
import sys
from pathlib import Path

# Insert the project root at the front of sys.path so that
#   from src.mp301.cleaning import ...
#   from app.main import app
# both resolve correctly.
ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
