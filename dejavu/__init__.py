"""dejavu — prompt caching cost simulator for Anthropic."""

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
