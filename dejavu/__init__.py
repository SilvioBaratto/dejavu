"""dejavu — prompt caching cost simulator for Anthropic."""

# Ensure dejavu's bundled baml_client wins over any same-named top-level
# package leaked onto sys.path by another project's editable install (e.g.
# baml_spam_detector's .pth). The console-script entry point has no cwd on
# sys.path, so without this the wrong baml_client (missing types.Message) is
# imported. Mirror tests/conftest.py: put the dir holding our baml_client first.
import sys as _sys
from pathlib import Path as _Path

_pkg_root = _Path(__file__).resolve().parent.parent
if (_pkg_root / "baml_client").is_dir() and str(_pkg_root) not in _sys.path[:1]:
    _sys.path.insert(0, str(_pkg_root))
del _sys, _Path, _pkg_root

from dejavu.contract import load_contract
from dejavu.conversation import load_questions
from dejavu.metrics import TurnMetrics, turn_metrics
from dejavu.pricing import (
    Model,
    Price,
    price_cached_turn,
    price_uncached_turn,
)
from dejavu.runner import (
    RunConfig,
    Side,
    SideResult,
    TokenEvent,
    TurnResult,
    build_registry,
    run_session,
)

__version__ = "0.1.0"
__all__: list[str] = [
    "load_contract",
    "load_questions",
    "TurnMetrics",
    "turn_metrics",
    "Model",
    "Price",
    "price_uncached_turn",
    "price_cached_turn",
    "Side",
    "SideResult",
    "TurnResult",
    "TokenEvent",
    "RunConfig",
    "build_registry",
    "run_session",
]
