from enum import Enum


class Department(Enum):
    """12 department types with color and label."""

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


class TickPhase(Enum):
    SENSE = "sense"
    SOLVE_ODE = "solve_ode"
    INTERACT = "interact"
    GROW_DIE = "grow_die"
    EMIT = "emit"
