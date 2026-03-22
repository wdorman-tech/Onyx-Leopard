def test_kpi_panel_creation(qtbot):
    from biosim.gui.dashboard.kpi import KpiPanel

    panel = KpiPanel()
    qtbot.addWidget(panel)
    assert panel._cards == {}


def test_kpi_panel_add_companies(qtbot):
    from biosim.gui.dashboard.kpi import KpiPanel

    panel = KpiPanel()
    qtbot.addWidget(panel)
    state = {
        "company_names": ["Alpha", "Beta"],
        "company_colors": ["#FF6B6B", "#4ECDC4"],
        "revenue": [100000, 200000],
        "cash": [500000, 1000000],
        "market_share": [0.4, 0.6],
        "dept_headcount": [[5] * 12, [10] * 12],
        "health_score": [0.8, 0.9],
    }
    panel.update_state(state)
    assert len(panel._cards) == 2
    assert "Alpha" in panel._cards


def test_kpi_panel_remove_dead_company(qtbot):
    from biosim.gui.dashboard.kpi import KpiPanel

    panel = KpiPanel()
    qtbot.addWidget(panel)

    state_full = {
        "company_names": ["Alpha", "Beta"],
        "company_colors": ["#FF6B6B", "#4ECDC4"],
        "revenue": [100000, 200000],
        "cash": [500000, 1000000],
        "market_share": [0.4, 0.6],
        "dept_headcount": [[5] * 12, [10] * 12],
        "health_score": [0.8, 0.9],
    }
    panel.update_state(state_full)
    assert len(panel._cards) == 2

    state_reduced = {
        "company_names": ["Alpha"],
        "company_colors": ["#FF6B6B"],
        "revenue": [150000],
        "cash": [600000],
        "market_share": [1.0],
        "dept_headcount": [[5] * 12],
        "health_score": [0.85],
    }
    panel.update_state(state_reduced)
    assert len(panel._cards) == 1
    assert "Beta" not in panel._cards


def test_kpi_card_metrics_display(qtbot):
    from biosim.gui.dashboard.kpi import KpiCard

    card = KpiCard("TestCo", "#FF6B6B")
    qtbot.addWidget(card)
    card.update_metrics(
        revenue=123456,
        cash=789000,
        market_share=0.35,
        headcount=42,
        health=0.92,
    )
    assert "123,456" in card._revenue_label.text()
    assert "789,000" in card._cash_label.text()
    assert "35.0%" in card._share_label.text()
    assert "42" in card._headcount_label.text()
