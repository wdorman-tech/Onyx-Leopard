from __future__ import annotations

import numpy as np
import pytest

from biosim.gui.petri.cluster import ClusterPhysics


@pytest.fixture()
def physics() -> ClusterPhysics:
    return ClusterPhysics()


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


class TestComputeSectorTargets:
    def test_empty_headcount(self, physics: ClusterPhysics) -> None:
        result = physics.compute_sector_targets([0.0] * 12, radius=50.0)
        assert result == {}

    def test_single_department(self, physics: ClusterPhysics) -> None:
        hc = [0.0] * 12
        hc[3] = 10.0
        result = physics.compute_sector_targets(hc, radius=50.0)
        assert len(result) == 1
        assert 3 in result
        # Single dept gets the full circle, centroid at angle=pi (midpoint of 0..2pi)
        x, y = result[3]
        dist = np.sqrt(x**2 + y**2)
        assert dist == pytest.approx(50.0 * 0.55, abs=0.1)

    def test_two_equal_departments(self, physics: ClusterPhysics) -> None:
        hc = [0.0] * 12
        hc[0] = 10.0
        hc[1] = 10.0
        result = physics.compute_sector_targets(hc, radius=100.0)
        assert len(result) == 2
        assert 0 in result
        assert 1 in result
        # Both should be at 55% radius
        for target in result.values():
            dist = np.sqrt(target[0] ** 2 + target[1] ** 2)
            assert dist == pytest.approx(100.0 * 0.55, abs=0.1)
        # Targets should be roughly opposite sides (180 degrees apart)
        dx = result[0][0] - result[1][0]
        dy = result[0][1] - result[1][1]
        separation = np.sqrt(dx**2 + dy**2)
        assert separation > 100.0 * 0.55  # should be well separated

    def test_proportional_wedge_sizes(self, physics: ClusterPhysics) -> None:
        hc = [0.0] * 12
        hc[0] = 30.0  # 75%
        hc[1] = 10.0  # 25%
        result = physics.compute_sector_targets(hc, radius=80.0)
        # Dept 0 centroid at midpoint of [0, 0.75*2pi], dept 1 at midpoint of [0.75*2pi, 2pi]
        angle_0 = np.arctan2(result[0][1], result[0][0])
        angle_1 = np.arctan2(result[1][1], result[1][0])
        # Dept 0 should be in the first 270 degrees, dept 1 in the last 90
        assert -np.pi < angle_0 < np.pi
        assert -np.pi < angle_1 < np.pi

class TestStep:
    def test_empty_positions(self, physics: ClusterPhysics, rng: np.random.Generator) -> None:
        positions = np.empty((0, 2))
        dept_indices = np.array([], dtype=int)
        result = physics.step(positions, dept_indices, {}, 50.0, rng)
        assert result.shape == (0, 2)

    def test_single_cell_stays_near_center(
        self, physics: ClusterPhysics, rng: np.random.Generator
    ) -> None:
        positions = np.array([[0.0, 0.0]])
        dept_indices = np.array([0])
        targets = {0: (10.0, 0.0)}
        # Run several steps — cell should drift toward target
        for _ in range(50):
            positions = physics.step(positions, dept_indices, targets, 50.0, rng)
        assert positions[0][0] > 2.0  # moved toward target at (10, 0)

    def test_boundary_containment(
        self, physics: ClusterPhysics, rng: np.random.Generator
    ) -> None:
        """Cells far outside boundary get pushed back."""
        radius = 50.0
        # Place cell well outside the 70% boundary
        positions = np.array([[45.0, 0.0]])
        dept_indices = np.array([0])
        targets = {0: (45.0, 0.0)}
        new_pos = physics.step(positions, dept_indices, targets, radius, rng)
        # Should be pushed inward (x should decrease)
        assert new_pos[0][0] < 45.0

    def test_different_depts_repel(
        self, physics: ClusterPhysics, rng: np.random.Generator
    ) -> None:
        """Two cells of different departments close together should repel."""
        # Use deterministic physics (no noise) for cleaner test
        quiet_physics = ClusterPhysics(k_attract=0.0, k_boundary=0.0, noise_sigma=0.0)
        positions = np.array([[5.0, 0.0], [-5.0, 0.0]])
        dept_indices = np.array([0, 1])
        targets = {0: (5.0, 0.0), 1: (-5.0, 0.0)}
        new_pos = quiet_physics.step(positions, dept_indices, targets, 100.0, rng)
        # Cells should move apart
        initial_dist = 10.0
        new_dist = np.sqrt((new_pos[0][0] - new_pos[1][0]) ** 2
                           + (new_pos[0][1] - new_pos[1][1]) ** 2)
        assert new_dist > initial_dist

    def test_same_dept_no_repulsion(
        self, physics: ClusterPhysics, rng: np.random.Generator
    ) -> None:
        """Two cells of the same department should NOT repel each other."""
        quiet_physics = ClusterPhysics(k_attract=0.0, k_boundary=0.0, noise_sigma=0.0)
        positions = np.array([[5.0, 0.0], [-5.0, 0.0]])
        dept_indices = np.array([0, 0])
        targets = {0: (0.0, 0.0)}
        new_pos = quiet_physics.step(positions, dept_indices, targets, 100.0, rng)
        # Without attraction or repulsion or noise, positions should not change
        np.testing.assert_allclose(new_pos, positions, atol=1e-10)

    def test_cohesion_pulls_toward_target(self, rng: np.random.Generator) -> None:
        """Intra-department attraction should pull cells toward sector target."""
        physics = ClusterPhysics(k_attract=0.15, k_repel=0.0, k_boundary=0.0, noise_sigma=0.0)
        positions = np.array([[0.0, 0.0], [2.0, 0.0]])
        dept_indices = np.array([0, 0])
        targets = {0: (20.0, 0.0)}
        new_pos = physics.step(positions, dept_indices, targets, 100.0, rng)
        # Both cells should move toward x=20 (rightward)
        assert new_pos[0][0] > 0.0
        assert new_pos[1][0] > 2.0

    def test_output_shape_matches_input(
        self, physics: ClusterPhysics, rng: np.random.Generator
    ) -> None:
        n = 20
        positions = rng.uniform(-30, 30, size=(n, 2))
        dept_indices = rng.integers(0, 12, size=n)
        targets = physics.compute_sector_targets([5.0] * 12, radius=50.0)
        result = physics.step(positions, dept_indices, targets, 50.0, rng)
        assert result.shape == (n, 2)

    def test_clustering_over_many_steps(self, rng: np.random.Generator) -> None:
        """After many steps, cells of the same department should be closer together
        than cells of different departments."""
        physics = ClusterPhysics()
        # 10 cells each in 2 departments, start randomly
        n = 20
        positions = rng.uniform(-10, 10, size=(n, 2))
        dept_indices = np.array([0] * 10 + [1] * 10)
        targets = physics.compute_sector_targets([10.0, 10.0] + [0.0] * 10, radius=80.0)

        for _ in range(200):
            positions = physics.step(positions, dept_indices, targets, 80.0, rng)

        # Measure intra-dept vs inter-dept distances
        dept0 = positions[:10]
        dept1 = positions[10:]
        intra0 = np.mean(np.linalg.norm(dept0 - dept0.mean(axis=0), axis=1))
        intra1 = np.mean(np.linalg.norm(dept1 - dept1.mean(axis=0), axis=1))
        inter = np.linalg.norm(dept0.mean(axis=0) - dept1.mean(axis=0))

        avg_intra = (intra0 + intra1) / 2
        # Departments should be farther apart than the spread within each cluster
        assert inter > avg_intra


class TestClusterPhysicsInit:
    def test_custom_params(self) -> None:
        p = ClusterPhysics(k_attract=0.5, k_repel=1.0, k_boundary=0.3,
                           noise_sigma=0.1, repel_radius=30.0)
        assert p._k_attract == 0.5
        assert p._k_repel == 1.0
        assert p._k_boundary == 0.3
        assert p._noise_sigma == 0.1
        assert p._repel_radius == 30.0

    def test_defaults(self) -> None:
        p = ClusterPhysics()
        assert p._k_attract == 0.15
        assert p._k_repel == 0.8
        assert p._k_boundary == 0.5
        assert p._noise_sigma == 0.3
        assert p._repel_radius == 20.0
