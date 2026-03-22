import numpy as np

from biosim.math.production import (
    cobb_douglas,
    marginal_product_capital,
    marginal_product_labor,
    optimal_labor,
)


class TestCobbDouglas:
    def test_constant_returns_to_scale(self):
        """With alpha + beta = 1, doubling inputs doubles output."""
        tfp = np.array([1.0, 2.0])
        k = np.array([100.0, 200.0])
        lab = np.array([50.0, 100.0])
        alpha = np.array([0.3, 0.4])
        beta = 1.0 - alpha

        y1 = cobb_douglas(tfp, k, lab, alpha, beta)
        y2 = cobb_douglas(tfp, 2 * k, 2 * lab, alpha, beta)
        np.testing.assert_allclose(y2, 2 * y1, rtol=1e-10)

    def test_increasing_returns_to_scale(self):
        """With alpha + beta > 1, doubling inputs more than doubles output."""
        tfp = np.array([1.0])
        k = np.array([100.0])
        lab = np.array([50.0])
        alpha = np.array([0.6])
        beta = np.array([0.6])

        y1 = cobb_douglas(tfp, k, lab, alpha, beta)
        y2 = cobb_douglas(tfp, 2 * k, 2 * lab, alpha, beta)
        assert y2[0] > 2 * y1[0]

    def test_zero_inputs_guarded(self):
        """Zero capital or labor should produce near-zero output, not NaN/error."""
        tfp = np.array([1.0, 1.0])
        k = np.array([0.0, 100.0])
        lab = np.array([50.0, 0.0])
        alpha = np.array([0.3, 0.3])
        beta = np.array([0.7, 0.7])

        y = cobb_douglas(tfp, k, lab, alpha, beta)
        assert not np.any(np.isnan(y))
        # Output should be very small when an input is zero (guarded to eps)
        assert y[0] < 1.0
        assert y[1] < 1.0

    def test_vectorization(self, default_params):
        """Batch matches individual computations."""
        p = default_params
        y_batch = cobb_douglas(p["tfp"], p["capital"], p["labor"], p["alpha"], p["beta"])

        for i in range(len(p["tfp"])):
            y_i = cobb_douglas(
                p["tfp"][i:i+1], p["capital"][i:i+1], p["labor"][i:i+1],
                p["alpha"][i:i+1], p["beta"][i:i+1],
            )
            np.testing.assert_allclose(y_batch[i], y_i[0], rtol=1e-10)


class TestMarginalProducts:
    def test_mpk_positive(self, default_params):
        p = default_params
        mpk = marginal_product_capital(p["tfp"], p["capital"], p["labor"], p["alpha"], p["beta"])
        assert np.all(mpk > 0)

    def test_mpl_positive(self, default_params):
        p = default_params
        mpl = marginal_product_labor(p["tfp"], p["capital"], p["labor"], p["alpha"], p["beta"])
        assert np.all(mpl > 0)

    def test_mpk_diminishing(self):
        """Marginal product of capital decreases as capital increases (alpha < 1)."""
        tfp = np.array([1.0, 1.0, 1.0])
        k = np.array([100.0, 200.0, 400.0])
        lab = np.array([50.0, 50.0, 50.0])
        alpha = np.array([0.3, 0.3, 0.3])
        beta = np.array([0.7, 0.7, 0.7])

        mpk = marginal_product_capital(tfp, k, lab, alpha, beta)
        assert mpk[0] > mpk[1] > mpk[2]

    def test_mpl_diminishing(self):
        """Marginal product of labor decreases as labor increases (beta < 1)."""
        tfp = np.array([1.0, 1.0, 1.0])
        k = np.array([100.0, 100.0, 100.0])
        lab = np.array([50.0, 100.0, 200.0])
        alpha = np.array([0.3, 0.3, 0.3])
        beta = np.array([0.7, 0.7, 0.7])

        mpl = marginal_product_labor(tfp, k, lab, alpha, beta)
        assert mpl[0] > mpl[1] > mpl[2]


class TestOptimalLabor:
    def test_roundtrip(self, default_params):
        """Computing optimal labor from a known output should reproduce the original labor."""
        p = default_params
        y = cobb_douglas(p["tfp"], p["capital"], p["labor"], p["alpha"], p["beta"])
        l_opt = optimal_labor(y, p["capital"], p["tfp"], p["alpha"], p["beta"], wage=np.ones(5))
        np.testing.assert_allclose(l_opt, p["labor"], rtol=1e-5)
