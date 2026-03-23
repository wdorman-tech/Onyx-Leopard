from __future__ import annotations


class TestDecisionLogWidget:
    def test_empty_state_no_crash(self, qtbot):
        from biosim.gui.dashboard.decision_log import DecisionLogWidget

        widget = DecisionLogWidget()
        qtbot.addWidget(widget)
        widget.update_state({})
        assert widget._table.rowCount() == 0

    def test_add_decision_row(self, qtbot):
        from biosim.gui.dashboard.decision_log import DecisionLogWidget

        widget = DecisionLogWidget()
        qtbot.addWidget(widget)
        widget.update_state(
            {
                "decisions": [
                    {
                        "tick": 5,
                        "company_name": "Alpha Corp",
                        "dept": 0,
                        "tier": 1,
                        "action": "adjust_budget",
                        "confidence": 0.85,
                    }
                ]
            }
        )
        assert widget._table.rowCount() == 1

    def test_max_rows_trimmed(self, qtbot):
        from biosim.gui.dashboard.decision_log import DecisionLogWidget

        widget = DecisionLogWidget()
        qtbot.addWidget(widget)
        for i in range(150):
            widget.update_state(
                {
                    "decisions": [
                        {
                            "tick": i,
                            "company_name": "X",
                            "dept": 0,
                            "tier": 0,
                            "action": "x",
                            "confidence": 0.5,
                        }
                    ]
                }
            )
        assert widget._table.rowCount() <= 100

    def test_clear_data(self, qtbot):
        from biosim.gui.dashboard.decision_log import DecisionLogWidget

        widget = DecisionLogWidget()
        qtbot.addWidget(widget)
        widget.update_state(
            {
                "decisions": [
                    {
                        "tick": 1,
                        "company_name": "A",
                        "dept": 0,
                        "tier": 0,
                        "action": "x",
                        "confidence": 0.5,
                    }
                ]
            }
        )
        assert widget._table.rowCount() > 0
        widget.clear_data()
        assert widget._table.rowCount() == 0

    def test_department_color_applied(self, qtbot):
        from PyQt6.QtGui import QColor

        from biosim.gui.dashboard.decision_log import DEPARTMENT_COLORS, DecisionLogWidget

        widget = DecisionLogWidget()
        qtbot.addWidget(widget)
        widget.update_state(
            {
                "decisions": [
                    {
                        "tick": 1,
                        "company_name": "A",
                        "dept": 3,
                        "tier": 0,
                        "action": "x",
                        "confidence": 0.5,
                    }
                ]
            }
        )
        dept_item = widget._table.item(0, 2)
        assert dept_item.foreground().color() == QColor(DEPARTMENT_COLORS[3])

    def test_tier_labels(self, qtbot):
        from biosim.gui.dashboard.decision_log import DecisionLogWidget

        widget = DecisionLogWidget()
        qtbot.addWidget(widget)
        for tier in range(4):
            widget.update_state(
                {
                    "decisions": [
                        {
                            "tick": 1,
                            "company_name": "A",
                            "dept": 0,
                            "tier": tier,
                            "action": "x",
                            "confidence": 0.5,
                        }
                    ]
                }
            )
        labels = [widget._table.item(i, 3).text() for i in range(4)]
        assert labels == ["ODE", "Heuristic", "Haiku", "Sonnet"]
