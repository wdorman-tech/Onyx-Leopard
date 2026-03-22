def test_revenue_chart_creation(qtbot):
    from biosim.gui.dashboard.charts import RevenueChart

    chart = RevenueChart()
    qtbot.addWidget(chart)
    assert chart.plot_widget is not None


def test_revenue_chart_update(qtbot):
    from biosim.gui.dashboard.charts import RevenueChart

    chart = RevenueChart()
    qtbot.addWidget(chart)
    state = {
        "company_names": ["Alpha", "Beta"],
        "company_colors": ["#FF6B6B", "#4ECDC4"],
        "revenue": [100000, 200000],
        "market_share": [0.4, 0.6],
    }
    chart.update_state(state)
    assert "Alpha" in chart._data
    assert len(chart._data["Alpha"]["values"]) == 1


def test_market_share_chart(qtbot):
    from biosim.gui.dashboard.charts import MarketShareChart

    chart = MarketShareChart()
    qtbot.addWidget(chart)
    state = {
        "company_names": ["A", "B", "C"],
        "company_colors": ["#FF6B6B", "#4ECDC4", "#45B7D1"],
        "market_share": [0.5, 0.3, 0.2],
    }
    chart.update_state(state)
    assert chart._tick == 1


def test_revenue_chart_history_limit(qtbot):
    from biosim.gui.dashboard.charts import RevenueChart

    chart = RevenueChart()
    qtbot.addWidget(chart)
    for i in range(250):
        chart.update_state(
            {
                "company_names": ["A"],
                "company_colors": ["#FF6B6B"],
                "revenue": [float(i * 1000)],
            }
        )
    assert len(chart._data["A"]["ticks"]) == chart.MAX_HISTORY


def test_cash_chart_creation(qtbot):
    from biosim.gui.dashboard.charts import CashChart

    chart = CashChart()
    qtbot.addWidget(chart)
    assert chart.plot_widget is not None
    assert chart._zero_line is not None


def test_cash_chart_update(qtbot):
    from biosim.gui.dashboard.charts import CashChart

    chart = CashChart()
    qtbot.addWidget(chart)
    state = {
        "company_names": ["Alpha"],
        "company_colors": ["#FF6B6B"],
        "cash": [500000],
    }
    chart.update_state(state)
    assert "Alpha" in chart._data
    assert chart._data["Alpha"]["values"] == [500000]


def test_cash_chart_clear(qtbot):
    from biosim.gui.dashboard.charts import CashChart

    chart = CashChart()
    qtbot.addWidget(chart)
    chart.update_state(
        {
            "company_names": ["A"],
            "company_colors": ["#FF6B6B"],
            "cash": [100000],
        }
    )
    chart.clear_data()
    assert chart._data == {}
    assert chart._curves == {}
    assert chart._tick == 0


def test_revenue_chart_clear(qtbot):
    from biosim.gui.dashboard.charts import RevenueChart

    chart = RevenueChart()
    qtbot.addWidget(chart)
    chart.update_state(
        {
            "company_names": ["A"],
            "company_colors": ["#FF6B6B"],
            "revenue": [50000],
        }
    )
    chart.clear_data()
    assert chart._data == {}
    assert chart._tick == 0
