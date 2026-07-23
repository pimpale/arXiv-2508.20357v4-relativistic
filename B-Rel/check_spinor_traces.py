#!/usr/bin/env python3
"""Numerically verify the massless reactor-antineutrino spinor traces.

The calculation uses the Dirac representation, metric (+---), and the
operators appearing in rel.tex.  In the massless limit the chiral projectors
remove the inactive helicity, so the usual v-spinor completeness relation can
be used without averaging over the incoming spin.
"""

from __future__ import annotations

import math

import numpy as np


def block(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    return np.block([[a, b], [c, d]])


zero = np.zeros((2, 2), dtype=complex)
sigma_1 = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_3 = np.array([[1, 0], [0, -1]], dtype=complex)

gamma = [np.diag([1, 1, -1, -1]).astype(complex)]
gamma.extend(block(zero, s, -s, zero) for s in (sigma_1, sigma_2, sigma_3))
gamma_5 = 1j * gamma[0] @ gamma[1] @ gamma[2] @ gamma[3]
left_twice = np.eye(4) - gamma_5


def slash(energy: float, direction: np.ndarray) -> np.ndarray:
    """Return p-slash for p = energy * (1, direction)."""

    result = energy * gamma[0]
    for index in range(3):
        result -= energy * direction[index] * gamma[index + 1]
    return result


def trace_pair(
    energy: float,
    direction: np.ndarray,
    final_energy: float,
    final_direction: np.ndarray,
) -> tuple[float, float]:
    """Return the excitation and de-excitation antineutrino traces."""

    p_slash = slash(energy, direction)
    p_prime_slash = slash(final_energy, final_direction)
    circular_minus = gamma[1] - 1j * gamma[2]
    circular_plus = gamma[1] + 1j * gamma[2]

    excitation = np.trace(
        p_slash
        @ circular_minus
        @ left_twice
        @ p_prime_slash
        @ circular_plus
        @ left_twice
    ).real
    deexcitation = np.trace(
        p_slash
        @ circular_plus
        @ left_twice
        @ p_prime_slash
        @ circular_minus
        @ left_twice
    ).real
    return float(excitation), float(deexcitation)


def unit_vector(theta: float, phi: float) -> np.ndarray:
    return np.array(
        [
            math.sin(theta) * math.cos(phi),
            math.sin(theta) * math.sin(phi),
            math.cos(theta),
        ]
    )


def main() -> None:
    rng = np.random.default_rng(250820357)
    maximum_relative_error = 0.0

    for _ in range(10_000):
        energy = rng.uniform(0.1, 10.0)
        final_energy = rng.uniform(0.1, 10.0)
        direction = unit_vector(
            math.acos(rng.uniform(-1.0, 1.0)),
            rng.uniform(0.0, 2.0 * math.pi),
        )
        final_direction = unit_vector(
            math.acos(rng.uniform(-1.0, 1.0)),
            rng.uniform(0.0, 2.0 * math.pi),
        )

        calculated = trace_pair(energy, direction, final_energy, final_direction)
        expected = (
            16.0
            * energy
            * final_energy
            * (1.0 + direction[2])
            * (1.0 - final_direction[2]),
            16.0
            * energy
            * final_energy
            * (1.0 - direction[2])
            * (1.0 + final_direction[2]),
        )

        for value, target in zip(calculated, expected):
            scale = max(abs(target), energy * final_energy * 1.0e-12)
            maximum_relative_error = max(
                maximum_relative_error, abs(value - target) / scale
            )

    print("Verified 10,000 random direction pairs.")
    print(f"Maximum scaled relative error: {maximum_relative_error:.3e}")
    print("S_+ = 16 E E' (1 + p_z_hat) (1 - p'_z_hat)")
    print("S_- = 16 E E' (1 - p_z_hat) (1 + p'_z_hat)")


if __name__ == "__main__":
    main()
