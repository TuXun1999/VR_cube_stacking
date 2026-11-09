"""Import tools/fake_table_publisher.py from tests (tools/ is not a package)."""
from pathlib import Path
import sys

TOOLS = str(Path(__file__).resolve().parents[1] / "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import fake_table_publisher as publisher  # noqa: E402,F401
