import numpy as np

from biosim.math.growth import growth_rhs, step_growth


class TestGrowthRHS:
    def test_steady_state_at_carrying_capacity(self):
        """When N = K, logistic dN/dt should be ~0."""
        n_agents = 3
        k = np.array([100.0, 200.0, 150.0])
        y0 = np.column_stack([
            k,  # firm_size = K
            np.array([1e6, 2e6, 1.5e6]),  # cash
            np.array([0.05, 0.03, 0.04]),  # growth_rate
        ]).ravel()

        dy = growth_rhs(
            0.0, y0, n_agents, k,
            revenue=np.array([1e5, 2e5, 1.5e5]),
            fixed_costs=np.array([5e4, 1e5, 7.5e4]),
            variable_cost_rate=np.array([500.0, 500.0, 500.0]),
        )
        dy_reshaped = dy.reshape(n_agents, 3)
        # firm_size derivative should be ~0 at carrying capacity
        np.testing.assert_allclose(dy_reshaped[:, 0], 0.0, atol=1e-10)

    def test_exponential_growth_when_small(self):
        """When N << K, growth is approximately r * N (exponential)."""
        n_agents = 1
        firm_size = np.array([1.0])
        k = np.array([1e6])
        r = np.array([0.1])
        y0 = np.column_stack([firm_size, np.array([1e6]), r]).ravel()

        dy = growth_rhs(
            0.0, y0, n_agents, k,
            revenue=np.array([1e5]),
            fixed_costs=np.array([5e4]),
            variable_cost_rate=np.array([100.0]),
        )
        d_firm = dy.reshape(n_agents, 3)[:, 0]
        # Should be close to r * N = 0.1 * 1.0 = 0.1
        np.testing.assert_allclose(d_firm, 0.1, rtol=1e-4)

    def test_cash_increases_when_revenue_exceeds_costs(self):
        n_agents = 1
        firm_size = np.array([10.0])
        revenue = np.array([1e5])
        fixed_costs = np.array([2e4])
        vcr = np.array([100.0])
        # costs = 2e4 + 100*10 = 2.1e4, revenue = 1e5 => positive cash flow
        y0 = np.column_stack([firm_size, np.array([1e6]), np.array([0.05])]).ravel()

        dy = growth_rhs(0.0, y0, n_agents, np.array([100.0]), revenue, fixed_costs, vcr)
        d_cash = dy.reshape(n_agents, 3)[:, 1]
        assert d_cash[0] > 0

    def test_cash_decreases_when_costs_exceed_revenue(self):
        n_agents = 1
        firm_size = np.array([10.0])
        revenue = np.array([1e3])
        fixed_costs = np.array([5e4])
        vcr = np.array([1000.0])
        y0 = np.column_stack([firm_size, np.array([1e6]), np.array([0.05])]).ravel()

        dy = growth_rhs(0.0, y0, n_agents, np.array([100.0]), revenue, fixed_costs, vcr)
        d_cash = dy.reshape(n_agents, 3)[:, 1]
        assert d_cash[0] < 0


class TestStepGrowth:
    def test_vectorization_matches_individual(self, default_params):
        """Batch result should match running each agent individually."""
        p = default_params
        new_fs, new_c, new_gr = step_growth(
            p["firm_size"], p["cash"], p["growth_rate"],
            p["carrying_capacity"], p["revenue"],
            p["fixed_costs"], p["variable_cost_rate"],
        )

        for i in range(len(p["firm_size"])):
            fs_i, c_i, gr_i = step_growth(
                p["firm_size"][i:i+1], p["cash"][i:i+1], p["growth_rate"][i:i+1],
                p["carrying_capacity"][i:i+1], p["revenue"][i:i+1],
                p["fixed_costs"][i:i+1], p["variable_cost_rate"][i:i+1],
            )
            np.testing.assert_allclose(new_fs[i], fs_i[0], rtol=1e-5)
            np.testing.assert_allclose(new_c[i], c_i[0], rtol=1e-5)
            np.testing.assert_allclose(new_gr[i], gr_i[0], rtol=1e-5)

    def test_numerical_stability_extreme_values(self):
        """Very large and very small values should not produce NaN/Inf."""
        new_fs, new_c, new_gr = step_growth(
            firm_size=np.array([1e-15, 1e10, 1.0]),
            cash=np.array([1e-15, 1e15, 1.0]),
            growth_rate=np.array([1e-10, 0.5, 0.01]),
            carrying_capacity=np.array([1e-10, 1e12, 100.0]),
            revenue=np.array([1e-10, 1e12, 1e5]),
            fixed_costs=np.array([0.0, 1e11, 5e4]),
            variable_cost_rate=np.array([0.0, 1.0, 1000.0]),
        )
        for arr in [new_fs, new_c, new_gr]:
            assert not np.any(np.isnan(arr)), f"NaN detected: {arr}"
            assert not np.any(np.isinf(arr)), f"Inf detected: {arr}"

    def test_euler_fallback_reasonable(self):
        """Euler fallback should still produce directionally correct results."""
        from biosim.math.growth import growth_rhs
        from biosim.math.solver import euler_step

        n_agents = 2
        firm_size = np.array([10.0, 20.0])
        cash = np.array([1e6, 2e6])
        growth_rate = np.array([0.05, 0.03])
        k = np.array([100.0, 200.0])
        rev = np.array([1e5, 2e5])
        fc = np.array([5e4, 1e5])
        vcr = np.array([1000.0, 800.0])

        y0 = np.column_stack([firm_size, cash, growth_rate]).ravel()
        y1 = euler_step(growth_rhs, y0, 1.0, n_agents, k, rev, fc, vcr)
        result = y1.reshape(n_agents, 3)

        assert np.all(result[:, 0] > firm_size)
        assert not np.allclose(result[:, 1], cash)
