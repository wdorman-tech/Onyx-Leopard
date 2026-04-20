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


class DecisionTier(Enum):
    ODE = 0          # Pure math, zero cost
    HEURISTIC = 1    # Rule-based, zero cost
    HAIKU = 2        # Claude Haiku via CAMEL-AI, low cost
    SONNET = 3       # Claude Sonnet via CAMEL-AI, standard cost


class AgentRole(Enum):
    """Maps department index to agent decision domain and LLM call frequency."""

    FINANCE = (0, "Budget reallocation, debt decisions, dividend policy", 4)
    RD = (1, "Innovation direction, patent strategy, tech adoption", 8)
    DISTRIBUTION = (2, "Route optimization, warehouse placement", 8)
    PRODUCTION = (3, "Capacity planning, quality targets, make-vs-buy", 4)
    SALES = (4, "Pricing, deal strategy, channel mix", 2)
    MARKETING = (5, "Campaign strategy, brand positioning", 4)
    HR = (6, "Hiring plan, retention strategy, culture", 4)
    EXECUTIVE = (7, "M&A, market entry/exit, major pivots", 12)
    CUSTOMER_SERVICE = (8, "Service level targets, feedback response", 4)
    LEGAL = (9, "Compliance posture, IP strategy", 12)
    IT = (10, "Infrastructure investment, tooling decisions", 8)
    PROCUREMENT = (11, "Vendor selection, contract negotiation", 4)

    def __init__(self, dept_index: int, domain: str, llm_frequency: int) -> None:
        self.dept_index = dept_index
        self.domain = domain
        self.llm_frequency = llm_frequency
