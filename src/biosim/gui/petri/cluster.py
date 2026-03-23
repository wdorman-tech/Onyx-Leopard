from __future__ import annotations

import numpy as np


class ClusterPhysics:
    """Vectorized force-based clustering that self-organizes cells by department.

    Each cell experiences 4 forces per step:
    1. Intra-department cohesion — linear spring toward department centroid
    2. Inter-department repulsion — inverse-square push from cells of other departments
    3. Boundary containment — soft wall at 70% of organism radius
    4. Brownian noise — small random jitter for organic liveness
    """

    __slots__ = ("_k_attract", "_k_repel", "_k_boundary", "_noise_sigma", "_repel_radius")

    def __init__(
        self,
        k_attract: float = 0.15,
        k_repel: float = 0.8,
        k_boundary: float = 0.5,
        noise_sigma: float = 0.3,
        repel_radius: float = 20.0,
    ) -> None:
        self._k_attract = k_attract
        self._k_repel = k_repel
        self._k_boundary = k_boundary
        self._noise_sigma = noise_sigma
        self._repel_radius = repel_radius

    def compute_sector_targets(
        self, dept_headcount: list[float], radius: float
    ) -> dict[int, tuple[float, float]]:
        """Angular partitioning: each dept gets a wedge proportional to headcount.

        Returns centroid target (x, y) at 55% of radius for each department index
        that has nonzero headcount.
        """
        total = sum(dept_headcount)
        if total <= 0:
            return {}

        targets: dict[int, tuple[float, float]] = {}
        angle = 0.0
        target_dist = radius * 0.55

        for dept_idx, count in enumerate(dept_headcount):
            if count <= 0:
                continue
            wedge = 2 * np.pi * (count / total)
            mid_angle = angle + wedge / 2
            targets[dept_idx] = (
                target_dist * np.cos(mid_angle),
                target_dist * np.sin(mid_angle),
            )
            angle += wedge

        return targets

    def step(
        self,
        positions: np.ndarray,
        dept_indices: np.ndarray,
        sector_targets: dict[int, tuple[float, float]],
        organism_radius: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Compute one force step. Returns new positions array (n_cells, 2)."""
        n = len(positions)
        if n == 0:
            return positions.copy()

        forces = np.zeros_like(positions)

        # 1. Intra-department cohesion: spring toward sector target centroid
        for dept_idx, target in sector_targets.items():
            mask = dept_indices == dept_idx
            if not np.any(mask):
                continue
            dept_positions = positions[mask]
            centroid = dept_positions.mean(axis=0)
            # Blend actual centroid with sector target for stable clustering
            effective_target = np.array(target) * 0.5 + centroid * 0.5
            forces[mask] += self._k_attract * (effective_target - dept_positions)

        # 2. Inter-department repulsion: inverse-square between different-dept cells
        if n > 1:
            # Pairwise displacement vectors: diff[i, j] = positions[i] - positions[j]
            diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=2)
            dist_sq_safe = np.maximum(dist_sq, 1e-6)

            # Mask: only repel cells of different departments within cutoff
            same_dept = dept_indices[:, np.newaxis] == dept_indices[np.newaxis, :]
            within_cutoff = dist_sq < self._repel_radius**2
            repel_mask = ~same_dept & within_cutoff
            np.fill_diagonal(repel_mask, False)

            # F = k_repel * (pos_i - pos_j) / dist^2
            repel_force = self._k_repel * diff / dist_sq_safe[:, :, np.newaxis]
            repel_force[~repel_mask] = 0.0
            forces += repel_force.sum(axis=1)

        # 3. Boundary containment: soft wall at 70% radius
        dists = np.sqrt(np.sum(positions**2, axis=1))
        boundary_limit = organism_radius * 0.7
        overshoot = dists - boundary_limit
        outside = overshoot > 0
        if np.any(outside):
            directions = positions[outside] / np.maximum(dists[outside, np.newaxis], 1e-10)
            forces[outside] -= self._k_boundary * overshoot[outside, np.newaxis] * directions

        # 4. Brownian noise
        forces += rng.normal(0, self._noise_sigma, size=positions.shape)

        return positions + forces
