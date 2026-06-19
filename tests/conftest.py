import os
import sys
from pathlib import Path

# Deterministic CLI help output across environments. Some CI runners (e.g.
# GitHub Actions) export FORCE_COLOR, which makes Rich style the Typer help so
# option flags like "--model" are broken up by ANSI escape codes and plain
# substring assertions fail. Force plain, uncolored output everywhere; Rich
# precedence requires FORCE_COLOR to be removed, not just NO_COLOR set.
os.environ.pop("FORCE_COLOR", None)
os.environ["NO_COLOR"] = "1"
os.environ["TERM"] = "dumb"

# Ensure local baml_client is imported, not the one from baml_spam_detector
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
