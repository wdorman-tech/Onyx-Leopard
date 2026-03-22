import numpy as np

from biosim.math.growth import step_growth
from biosim.math.solver import euler_step, solve_tick


class TestSolveTick:
    def test_matches_step_growth(self, default_params):
        """solve_tick should produce same results as step_growth for one tick."""
        p = default_params
        new_fs, new_c, new_gr = step_growth(
            p["firm_size"], p["cash"], p["growth_rate"],
            p["carrying_capacity"], p["revenue"],
            p["fixed_costs"], p["variable_cost_rate"],
        )
        result = solve_tick(
            p["firm_size"], p["cash"], p["growth_rate"],
            p["carrying_capacity"], p["revenue"],
            p["fixed_costs"], p["variable_cost_rate"],
        )
        np.testing.assert_allclose(result["firm_size"], new_fs, rtol=1e-6)
        np.testing.assert_allclose(result["cash"], new_c, rtol=1e-6)
        np.testing.assert_allclose(result["growth_rate"], new_gr, rtol=1e-6)

    def test_multi_agent_matches_individual(self, default_params):
        """Batch solve for N agents matches solving each agent individually."""
        p = default_params
        batch = solve_tick(
            p["firm_size"], p["cash"], p["growth_rate"],
            p["carrying_capacity"], p["revenue"],
            p["fixed_costs"], p["variable_cost_rate"],
        )

        for i in range(len(p["firm_size"])):
            single = solve_tick(
                p["firm_size"][i:i+1], p["cash"][i:i+1], p["growth_rate"][i:i+1],
                p["carrying_capacity"][i:i+1], p["revenue"][i:i+1],
                p["fixed_costs"][i:i+1], p["variable_cost_rate"][i:i+1],
            )
            np.testing.assert_allclose(batch["firm_size"][i], single["firm_size"][0], rtol=1e-5)
            np.testing.assert_allclose(batch["cash"][i], single["cash"][0], rtol=1e-5)
            np.testing.assert_allclose(batch["growth_rate"][i], single["growth_rate"][0], rtol=1e-5)

    def test_extreme_params_trigger_fallback(self):
        """Extreme parameters should still produce finite results via fallback chain."""
        result = solve_tick(
            firm_size=np.array([1e-300, 1e300]),
            cash=np.array([1e-300, 1e300]),
            growth_rate=np.array([1e-15, 100.0]),
            carrying_capacity=np.array([1e-300, 1e300]),
            revenue=np.array([1e-300, 1e300]),
            fixed_costs=np.array([0.0, 1e299]),
            variable_cost_rate=np.array([0.0, 1.0]),
        )
        for key in ["firm_size", "cash", "growth_rate"]:
            assert not np.any(np.isnan(result[key])), f"NaN in {key}"


def _constant_rhs(_t: float, _y: np.ndarray) -> np.ndarray:
    return np.array([1.0])


def _identity_rhs(_t: float, y: np.ndarray) -> np.ndarray:
    return y


class TestEulerStep:
    def test_simple_linear_ode(self):
        """dy/dt = 1 with y0=0, dt=1 => y=1"""
        y1 = euler_step(_constant_rhs, np.array([0.0]), 1.0)
        np.testing.assert_allclose(y1, [1.0])

    def test_exponential_growth_euler(self):
        """dy/dt = y with y0=1, dt=0.01 => y ~ e^0.01 ~ 1.01"""
        y1 = euler_step(_identity_rhs, np.array([1.0]), 0.01)
        np.testing.assert_allclose(y1, [1.01], rtol=1e-10)
