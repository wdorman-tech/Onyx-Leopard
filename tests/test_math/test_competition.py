import numpy as np

from biosim.math.competition import (
    build_competition_matrix,
    coexistence_check,
    lotka_volterra_rhs,
    step_competition,
)


class TestLotkaVolterraRHS:
    def test_single_species_is_logistic(self):
        """With one species, LV reduces to standard logistic growth."""
        pop = np.array([10.0])
        r = np.array([0.1])
        k = np.array([100.0])
        alpha = np.array([[1.0]])

        dy = lotka_volterra_rhs(0.0, pop, r, k, alpha)
        expected = r[0] * pop[0] * (1 - pop[0] / k[0])
        np.testing.assert_allclose(dy[0], expected, rtol=1e-10)

    def test_steady_state_at_k(self):
        """At carrying capacity, single-species growth is zero."""
        pop = np.array([100.0])
        r = np.array([0.1])
        k = np.array([100.0])
        alpha = np.array([[1.0]])

        dy = lotka_volterra_rhs(0.0, pop, r, k, alpha)
        np.testing.assert_allclose(dy[0], 0.0, atol=1e-10)


class TestStepCompetition:
    def test_two_equal_species_converge_to_half_k(self):
        """Two identical species with alpha_ij=1 should each converge to K/2."""
        pop = np.array([10.0, 10.0])
        r = np.array([0.1, 0.1])
        k = np.array([100.0, 100.0])
        alpha = np.array([[1.0, 1.0], [1.0, 1.0]])

        # Run many steps to approach equilibrium
        for _ in range(500):
            pop = step_competition(pop, r, k, alpha, dt=1.0)

        np.testing.assert_allclose(pop, [50.0, 50.0], rtol=0.01)

    def test_competitive_exclusion(self):
        """Stronger species (higher r, higher K) drives weaker toward zero."""
        pop = np.array([50.0, 50.0])
        r = np.array([0.2, 0.05])
        k = np.array([200.0, 50.0])
        # Strong cross-competition
        alpha = np.array([[1.0, 0.8], [1.2, 1.0]])

        for _ in range(2000):
            pop = step_competition(pop, r, k, alpha, dt=1.0)

        # Weaker species should be near zero
        assert pop[1] < 1.0
        # Stronger species should dominate
        assert pop[0] > 100.0

    def test_populations_stay_nonnegative(self):
        """Populations should never go negative."""
        pop = np.array([1.0, 1.0, 1.0])
        r = np.array([0.1, 0.1, 0.1])
        k = np.array([100.0, 100.0, 100.0])
        alpha = np.array([[1.0, 2.0, 2.0], [2.0, 1.0, 2.0], [2.0, 2.0, 1.0]])

        for _ in range(200):
            pop = step_competition(pop, r, k, alpha, dt=1.0)
            assert np.all(pop >= 0.0)


class TestBuildCompetitionMatrix:
    def test_diagonal_is_one(self):
        alpha = build_competition_matrix(5, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(np.diag(alpha), np.ones(5))

    def test_off_diagonal_near_base(self):
        base = 0.5
        alpha = build_competition_matrix(4, base_competition=base, rng=np.random.default_rng(0))
        mask = ~np.eye(4, dtype=bool)
        off_diag = alpha[mask]
        # Should be within +-10% of base
        assert np.all(off_diag >= 0.0)
        assert np.all(off_diag < 1.0)
        np.testing.assert_allclose(np.mean(off_diag), base, atol=0.1)

    def test_shape(self):
        alpha = build_competition_matrix(7)
        assert alpha.shape == (7, 7)


class TestCoexistenceCheck:
    def test_two_species_coexistence(self):
        """Two species with weak cross-competition can coexist."""
        r = np.array([0.1, 0.1])
        k = np.array([100.0, 100.0])
        alpha = np.array([[1.0, 0.3], [0.3, 1.0]])

        result = coexistence_check(r, k, alpha)
        assert np.all(result)

    def test_exclusion_detected(self):
        """Asymmetric competition where species 2 dominates excludes species 1."""
        r = np.array([0.1, 0.1])
        k = np.array([50.0, 200.0])
        # Species 2 strongly suppresses species 1: N* = solve(alpha, K) -> [-25, 250]
        alpha = np.array([[1.0, 0.3], [2.0, 1.0]])

        result = coexistence_check(r, k, alpha)
        assert not np.all(result)
