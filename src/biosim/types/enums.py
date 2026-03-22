from __future__ import annotations

from enum import Enum


class Department(Enum):
    """Business departments mapped to colors for petri dish rendering."""

    RED = ("Finance", "#E74C3C")
    BLUE = ("R&D", "#3498DB")
    GREEN = ("Distribution", "#2ECC71")
    YELLOW = ("Production", "#F1C40F")
    ORANGE = ("Sales", "#E67E22")
    PURPLE = ("Marketing", "#9B59B6")
    TEAL = ("HR", "#1ABC9C")
    NAVY = ("Executive", "#2C3E50")
    PINK = ("Customer Service", "#E91E63")
    BROWN = ("Legal", "#795548")
    SLATE = ("IT", "#607D8B")
    LIME = ("Procurement", "#8BC34A")

    def __init__(self, label: str, hex_color: str) -> None:
        self.label = label
        self.hex_color = hex_color


class CompanyStage(Enum):
    STARTUP = "startup"
    GROWTH = "growth"
    MATURE = "mature"
    DECLINE = "decline"


class StructureType(Enum):
    FLAT = "flat"
    MATRIX = "matrix"
    HIERARCHICAL = "hierarchical"
    DIVISIONAL = "divisional"


class Outlook(Enum):
    BOOM = "boom"
    GROWTH = "growth"
    STABLE = "stable"
    RECESSION = "recession"
    CRISIS = "crisis"


class TickPhase(Enum):
    SENSE = "sense"
    SOLVE_ODE = "solve_ode"
    AGENT_DECISIONS = "agent_decisions"
    INTERACTIONS = "interactions"
    GROWTH_DIVISION = "growth_division"
    ENV_UPDATE = "env_update"
    SELECTION = "selection"
    EMIT_STATE = "emit_state"
