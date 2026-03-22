import re

from biosim.types.enums import (
    CompanyStage,
    Department,
    Outlook,
    StructureType,
    TickPhase,
)


class TestDepartment:
    def test_has_12_members(self) -> None:
        assert len(Department) == 12

    def test_each_has_label(self) -> None:
        for dept in Department:
            assert isinstance(dept.label, str)
            assert len(dept.label) > 0

    def test_each_has_valid_hex_color(self) -> None:
        hex_pattern = re.compile(r"^#[0-9A-Fa-f]{6}$")
        for dept in Department:
            assert hex_pattern.match(dept.hex_color), (
                f"{dept.name} has invalid hex: {dept.hex_color}"
            )

    def test_labels_are_unique(self) -> None:
        labels = [d.label for d in Department]
        assert len(labels) == len(set(labels))

    def test_specific_mappings(self) -> None:
        assert Department.RED.label == "Finance"
        assert Department.BLUE.label == "R&D"
        assert Department.NAVY.hex_color == "#2C3E50"


class TestCompanyStage:
    def test_members(self) -> None:
        names = {s.name for s in CompanyStage}
        assert names == {"STARTUP", "GROWTH", "MATURE", "DECLINE"}


class TestOutlook:
    def test_members(self) -> None:
        names = {o.name for o in Outlook}
        assert names == {"BOOM", "GROWTH", "STABLE", "RECESSION", "CRISIS"}


class TestStructureType:
    def test_members(self) -> None:
        names = {s.name for s in StructureType}
        assert names == {"FLAT", "MATRIX", "HIERARCHICAL", "DIVISIONAL"}


class TestTickPhase:
    def test_has_8_phases(self) -> None:
        assert len(TickPhase) == 8
