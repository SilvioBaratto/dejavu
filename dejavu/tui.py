"""Two-panel TUI: state layer, Rich renderer, and async driver.

State   : PanelState, CostStripState, RenderState
Renderer: TuiRenderer
Driver  : run_tui
Helpers : _fmt_usd, _fmt_int
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from dejavu.runner import Side
from dejavu.resilience import resilient_session

if TYPE_CHECKING:
    from dejavu.runner import SideResult, TokenEvent, TurnResult

# Fixed column widths keep numbers right-aligned on a phone-recorded terminal.
_USD_NUM_WIDTH = 9  # numeric part: up to "9999.9999"
_INT_WIDTH = 8  # integer part: up to "9,999,999"


def _fmt_usd(value: float) -> str:
    """Fixed-width right-aligned USD string: $ + 4 decimal places."""
    return f"${value:{_USD_NUM_WIDTH}.4f}"


def _fmt_int(value: int) -> str:
    """Fixed-width right-aligned comma-grouped integer string."""
    return f"{value:>{_INT_WIDTH},}"


@dataclass
class Entry:
    """One transcript line: a role ('user'/'assistant') and its text."""

    role: str
    text: str


@dataclass
class PanelState:
    """Mutable display state for one session panel (left = uncached, right = cached)."""

    title: str
    running_cost: float = 0.0
    turn_number: int = 0
    this_turn_cache_read: int = 0
    last_turn_delta: float = 0.0
    transcript: list[Entry] = field(default_factory=list)
    live_partial: str = ""


@dataclass
class CostStripState:
    """Cumulative cost totals and the 'Nx cheaper / pricier' strip metrics."""

    uncached_total: float = 0.0
    cached_total: float = 0.0

    def multiple(self) -> float:
        """Raw ratio uncached / cached.  Returns 1.0 when cached_total == 0."""
        if self.cached_total == 0.0:
            return 1.0
        return self.uncached_total / self.cached_total

    def saved(self) -> float:
        """Raw dollar savings (uncached − cached); can be negative on turn 1."""
        return self.uncached_total - self.cached_total

    def multiple_text(self) -> str:
        """Formatted strip label; shows 'Nx pricier' during the turn-1 write premium."""
        if self.cached_total == 0.0:
            return "—"
        if self.cached_total > self.uncached_total:
            inv = (
                self.cached_total / self.uncached_total
                if self.uncached_total > 0.0
                else 1.0
            )
            return f"{inv:.1f}x pricier"
        return f"{self.multiple():.1f}x cheaper"


@dataclass
class RenderState:
    """Orchestrates state updates across both panels and the cost strip."""

    uncached: PanelState = field(default_factory=lambda: PanelState(title="Uncached"))
    cached: PanelState = field(default_factory=lambda: PanelState(title="Cached"))
    cost_strip: CostStripState = field(default_factory=CostStripState)
    note: str | None = None

    def apply_token(self, event: TokenEvent) -> None:
        """Set the matching panel's live_partial to the cumulative stream partial."""
        panel = self.uncached if event.side == Side.UNCACHED else self.cached
        panel.live_partial = event.partial

    def set_note(self, note: str | None) -> None:
        """Set or clear the transient retry/status note (wired by the runner layer)."""
        self.note = note

    def apply_turn(self, result: TurnResult) -> None:
        """Update all state from a completed TurnResult; clears partials and note."""
        self._populate(self.uncached, result.question, result.uncached, result.index)
        self._populate(self.cached, result.question, result.cached, result.index)
        self.cost_strip.uncached_total += result.uncached.cost_usd
        self.cost_strip.cached_total += result.cached.cost_usd
        self.note = None

    def _populate(
        self, panel: PanelState, question: str, side: SideResult, index: int
    ) -> None:
        panel.transcript.extend(
            [Entry("user", question), Entry("assistant", side.text)]
        )
        panel.running_cost += side.cost_usd
        panel.last_turn_delta = side.cost_usd
        panel.this_turn_cache_read = side.cached_input_tokens or 0
        panel.turn_number = index
        panel.live_partial = ""


# ---------------------------------------------------------------------------
# Rich rendering helpers (module-level to keep TuiRenderer thin)
# ---------------------------------------------------------------------------


def _body_text(panel: PanelState) -> Text:
    """Transcript entries + live streaming partial for one panel."""
    t = Text(overflow="fold")
    for entry in panel.transcript:
        style = "bold cyan" if entry.role == "user" else "white"
        t.append(f"{entry.role}: ", style=style)
        t.append(entry.text + "\n")
    if panel.live_partial:
        t.append(panel.live_partial, style="dim italic")
    return t


def _header_text(panel: PanelState) -> Text:
    """Cost / turn / cache-reads / delta header row."""
    t = Text(justify="left")
    t.append(_fmt_usd(panel.running_cost), style="bold bright_white")
    t.append(f"  Turn {panel.turn_number}", style="bold yellow")
    t.append(f"  Cache {_fmt_int(panel.this_turn_cache_read)}", style="cyan")
    t.append(f"  Δ {_fmt_usd(panel.last_turn_delta)}", style="green")
    return t


def _panel_renderable(panel: PanelState) -> Panel:
    """One session panel with a cost header and scrolling body."""
    return Panel(
        _body_text(panel),
        title=_header_text(panel),
        subtitle=f"[bold]{panel.title}[/bold]",
        border_style="bright_blue",
    )


def _footer_renderable(state: RenderState) -> Panel:
    """Center/footer strip: Nx cheaper + total saved + optional retry note."""
    t = Text(justify="center")
    t.append(state.cost_strip.multiple_text(), style="bold bright_green")
    t.append(f"   Saved: {_fmt_usd(state.cost_strip.saved())}", style="bold white")
    if state.note:
        t.append(f"   ↻ {state.note}", style="bold yellow")
    return Panel(t, border_style="bright_green")


# ---------------------------------------------------------------------------
# TuiRenderer
# ---------------------------------------------------------------------------


class TuiRenderer:
    """Builds the Rich two-column + footer Layout from a RenderState."""

    def render(self, state: RenderState) -> Layout:
        """Return a fresh Layout renderable reflecting the current state."""
        layout = Layout()
        layout.split_column(Layout(name="panels"), Layout(name="footer", size=5))
        layout["panels"].split_row(Layout(name="left"), Layout(name="right"))
        layout["left"].update(_panel_renderable(state.uncached))
        layout["right"].update(_panel_renderable(state.cached))
        layout["footer"].update(_footer_renderable(state))
        return layout


# ---------------------------------------------------------------------------
# Async TUI driver
# ---------------------------------------------------------------------------


async def run_tui(
    cfg: Any,
    *,
    console: Console | None = None,
    session_factory: Callable = resilient_session,
) -> RenderState:
    """Open Rich Live, drive session_factory, and update the display each turn."""
    state = RenderState()
    renderer = TuiRenderer()
    _console = console or Console()
    with Live(renderer.render(state), console=_console, auto_refresh=False) as live:

        def _on_token(event: TokenEvent) -> None:
            state.apply_token(event)
            live.update(renderer.render(state))

        async for turn in session_factory(
            cfg, on_token=_on_token, on_note=state.set_note
        ):
            state.apply_turn(turn)
            live.update(renderer.render(state))
    return state
