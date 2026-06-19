import sys
from pathlib import Path

# Ensure local baml_client is imported, not the one from baml_spam_detector
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
