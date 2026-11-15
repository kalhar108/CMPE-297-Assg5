"""Configuration and execution mode management for RSI (Recursive Self-Improvement) Suite.

Supports automatic sensing of live API keys (Google Gemini) with seamless,
deterministic offline mock fallback when keys are absent or when --mock is requested.
"""

from __future__ import annotations

import os
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field

# Try loading python-dotenv if present to automatically pick up .env from repo root
try:
    from dotenv import load_dotenv

    # Search for .env up to 3 directory levels above this file
    _current_dir = Path(__file__).resolve().parent
    for p in [_current_dir, _current_dir.parent, _current_dir.parent.parent]:
        env_file = p / ".env"
        if env_file.exists():
            load_dotenv(env_file, override=False)
            break
except ImportError:
    pass


class ExecutionMode(str, Enum):
    """Execution modes for RSI components."""
    LIVE = "live"
    MOCK = "mock"
    AUTO = "auto"


class RSIExecutionConfig(BaseModel):
    """Configuration model for RSI execution engine."""

    mode: Literal["mock", "live", "auto"] = Field(
        default="auto",
        description="Execution mode: 'auto' (senses API keys), 'mock' (offline), 'live' (frontier LLM)",
    )
    model_name: str = Field(
        default="gemini-3.8-flash",
        description="Model name to use in live mode (defaults to gemini-3.8-flash or env HARNESS_MODEL)",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="Optional explicit API key; if None, detects GOOGLE_API_KEY / GEMINI_API_KEY from environment",
    )
    temperature: float = Field(
        default=0.2,
        description="Sampling temperature for live model calls",
    )
    max_generations: int = Field(
        default=3,
        description="Maximum recursive self-improvement generations to execute",
    )
    seed: int = Field(
        default=42,
        description="Random seed for reproducible offline simulation",
    )

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        # Check for model name in environment if default
        if self.model_name == "gemini-3.8-flash" and os.environ.get("HARNESS_MODEL"):
            self.model_name = os.environ["HARNESS_MODEL"]

    def get_api_key(self) -> Optional[str]:
        """Resolves the active API key from instance or environment."""
        if self.api_key:
            return self.api_key
        return (
            os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("API_KEY")
        )

    def resolved_mode(self) -> Literal["live", "mock"]:
        """Resolves the operational mode to either 'live' or 'mock'."""
        if self.mode == "mock":
            return "mock"
        if self.mode == "live":
            return "live"

        # Check environment flag override
        env_mode = os.environ.get("RSI_MODE", "").lower().strip()
        if not env_mode:
            env_mode = os.environ.get("RSI_EXECUTION_MODE", "").lower().strip()
        if env_mode in ("mock", "offline"):
            return "mock"
        if env_mode == "live":
            return "live"

        # Auto-sense available API keys
        key = self.get_api_key()
        if key and len(key.strip()) > 10:
            return "live"
        return "mock"

    def print_banner(self, console: Any = None) -> None:
        """Prints a rich banner communicating active execution mode and provider status."""
        active_mode = self.resolved_mode()
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.text import Text

            c = console or Console()
            if active_mode == "mock":
                text = Text()
                text.append("[RSI Alpha Engine] Running in ", style="bold cyan")
                text.append("DETERMINISTIC OFFLINE MOCK", style="bold yellow")
                text.append(" mode\n", style="bold cyan")
                text.append("  Reason: ", style="bold white")
                text.append("No API key found or --mock requested\n", style="dim")
                text.append("  Network: ", style="bold white")
                text.append("DISABLED (0 network traffic, 100% deterministic schema output)", style="green")
                panel = Panel(
                    text,
                    title="[bold yellow]Offline Execution Engine[/bold yellow]",
                    border_style="yellow",
                )
                c.print(panel)
            else:
                text = Text()
                text.append("[RSI Alpha Engine] Running in ", style="bold cyan")
                text.append("FRONTIER LIVE GEMINI", style="bold green")
                text.append(" mode\n", style="bold cyan")
                text.append("  Model: ", style="bold white")
                text.append(f"{self.model_name}\n", style="cyan")
                text.append("  Key Source: ", style="bold white")
                text.append("GOOGLE_API_KEY environment\n", style="dim")
                text.append("  Provider: ", style="bold white")
                text.append("google-genai SDK (v1.59.0)", style="bold magenta")
                panel = Panel(
                    text,
                    title="[bold green]Live Gemini Engine[/bold green]",
                    border_style="green",
                )
                c.print(panel)
        except ImportError:
            # Fallback if rich is not available
            print(f"[RSI Alpha Engine] Mode: {active_mode.upper()} | Model: {self.model_name}")
