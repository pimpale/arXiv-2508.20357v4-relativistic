#!/usr/bin/env python3
"""Spectrum moments and fixed-target spin-flip rates for a 235U reactor.

The CSV is treated as a piecewise-constant yield dN/(dE dfission).  Natural
units are used for the cross section and converted to cm^2 at the end.
The example rates are illustrative and can be changed with command-line
arguments; no reactor normalization is hidden in the spectrum file.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


G_F_GEV_MINUS_2 = 1.1663787e-5
GEV_MINUS_2_TO_CM2 = 0.3893793721e-27
MEV_TO_JOULE = 1.602176634e-13
HBAR_C_MEV_CM = 1.973269804e-11
HBAR_C_GEV_CM = HBAR_C_MEV_CM * 1.0e-3


def load_bins(path: Path) -> list[tuple[float, float, float]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            (
                float(row["bin_low_MeV"]),
                float(row["bin_high_MeV"]),
                float(row["dN_dE_per_MeV_per_fission"]),
            )
            for row in reader
        ]


def moment(bins: list[tuple[float, float, float]], power: int) -> float:
    """Return integral dE E**power S(E), with E in MeV."""

    return sum(
        density
        * (upper ** (power + 1) - lower ** (power + 1))
        / (power + 1)
        for lower, upper, density in bins
    )


def shifted_second_moment(
    bins: list[tuple[float, float, float]], omega_mev: float, sign: int
) -> float:
    """Return integral S(E) (E + sign*omega)^2 with excitation threshold.

    sign=-1 is excitation and starts at E=omega.  sign=+1 is de-excitation.
    """

    total = 0.0
    for lower, upper, density in bins:
        integration_low = max(lower, omega_mev) if sign == -1 else lower
        if upper <= integration_low:
            continue
        a = integration_low
        b = upper
        total += density * (
            (b**3 - a**3) / 3.0
            + sign * omega_mev * (b**2 - a**2)
            + omega_mev**2 * (b - a)
        )
    return total


def cross_section_coefficient_cm2_per_mev2(g_a: float) -> float:
    """Coefficient in sigma = C E'^2 angular_factor."""

    # (2/pi) G_F^2 g_A^2 E_GeV^2, followed by GeV^-2 -> cm^2.
    return (
        2.0
        / math.pi
        * G_F_GEV_MINUS_2**2
        * g_a**2
        * 1.0e-6
        * GEV_MINUS_2_TO_CM2
    )


def main() -> None:
    default_csv = Path(__file__).resolve().parent.parent / "235U_nspec.csv"
    parser = argparse.ArgumentParser()
    parser.add_argument("--spectrum", type=Path, default=default_csv)
    parser.add_argument("--g-a", type=float, default=0.5)
    parser.add_argument("--omega-ev", type=float, default=1.0e-8)
    parser.add_argument("--cos-alpha", type=float, default=0.0)
    parser.add_argument("--power-gw", type=float, default=1.0)
    parser.add_argument("--distance-m", type=float, default=10.0)
    parser.add_argument("--energy-per-fission-mev", type=float, default=202.36)
    parser.add_argument("--sample-radius-cm", type=float, default=1.0)
    parser.add_argument("--spin-density-cm3", type=float, default=3.0e22)
    args = parser.parse_args()

    if not -1.0 <= args.cos_alpha <= 1.0:
        raise ValueError("--cos-alpha must be between -1 and 1")

    bins = load_bins(args.spectrum)
    moments = [moment(bins, power) for power in range(5)]
    yield_per_fission = moments[0]
    mean_energy = moments[1] / moments[0]
    rms_energy = math.sqrt(moments[2] / moments[0])
    rate_weighted_energy = moments[2] / moments[1]

    omega_mev = args.omega_ev * 1.0e-6
    excitation_integral = shifted_second_moment(bins, omega_mev, -1)
    deexcitation_integral = shifted_second_moment(bins, omega_mev, +1)
    coefficient = cross_section_coefficient_cm2_per_mev2(args.g_a)

    # Integral S(E) sigma(E) dE: cm^2 per fission.
    excitation_cross_section_yield = (
        coefficient * (1.0 + args.cos_alpha) * excitation_integral
    )
    deexcitation_cross_section_yield = (
        coefficient * (1.0 - args.cos_alpha) * deexcitation_integral
    )

    fission_energy_joule = args.energy_per_fission_mev * MEV_TO_JOULE
    fission_rate = args.power_gw * 1.0e9 / fission_energy_joule
    distance_cm = args.distance_m * 100.0
    geometric_flux_factor = fission_rate / (4.0 * math.pi * distance_cm**2)
    integrated_flux = geometric_flux_factor * yield_per_fission
    excitation_rate = geometric_flux_factor * excitation_cross_section_yield
    deexcitation_rate = geometric_flux_factor * deexcitation_cross_section_yield

    fractional_kinematic_difference = (
        4.0 * omega_mev * moments[1] / moments[2]
    )
    er_rms = (
        rms_energy * args.sample_radius_cm / HBAR_C_MEV_CM
    )
    number_of_spins = (
        4.0
        * math.pi
        / 3.0
        * args.sample_radius_cm**3
        * args.spin_density_cm3
    )
    radius_gev_inverse = args.sample_radius_cm / HBAR_C_GEV_CM
    coherent_cross_section = (
        9.0
        * G_F_GEV_MINUS_2**2
        * args.g_a**2
        * number_of_spins**2
        * (1.0 - args.cos_alpha**2)
        / (16.0 * math.pi * radius_gev_inverse**2)
        * GEV_MINUS_2_TO_CM2
    )
    coherent_rate = integrated_flux * coherent_cross_section
    incoherent_excitation_rate = number_of_spins / 2.0 * excitation_rate
    incoherent_deexcitation_rate = number_of_spins / 2.0 * deexcitation_rate

    print(f"Spectrum: {args.spectrum}")
    print(f"Bins: {len(bins)}")
    print(f"Y_0 = integral S(E)dE       = {moments[0]:.10g} /fission")
    print(f"Y_1 = integral E S(E)dE     = {moments[1]:.10g} MeV/fission")
    print(f"Y_2 = integral E^2 S(E)dE   = {moments[2]:.10g} MeV^2/fission")
    print(f"Y_3 = integral E^3 S(E)dE   = {moments[3]:.10g} MeV^3/fission")
    print(f"Y_4 = integral E^4 S(E)dE   = {moments[4]:.10g} MeV^4/fission")
    print(f"Mean energy                 = {mean_energy:.8g} MeV")
    print(f"RMS energy                  = {rms_energy:.8g} MeV")
    print(f"E_eff = Y_2/Y_1             = {rate_weighted_energy:.8g} MeV")
    print(
        "Kinematic (gamma_- - gamma_+)/gamma_0 at cos(alpha)=0 "
        f"= {fractional_kinematic_difference:.6e}"
    )
    print(
        f"Cross-section coefficient C = {coefficient:.8e} cm^2/MeV^2"
    )
    print(
        "Spectrum-averaged sigma_+  "
        f"= {excitation_cross_section_yield / yield_per_fission:.8e} cm^2"
    )
    print(
        "Spectrum-averaged sigma_-  "
        f"= {deexcitation_cross_section_yield / yield_per_fission:.8e} cm^2"
    )
    print()
    print("Illustrative point-source normalization:")
    print(f"  Thermal power             = {args.power_gw:g} GW")
    print(f"  Distance                  = {args.distance_m:g} m")
    print(
        f"  Energy per fission        = {args.energy_per_fission_mev:g} MeV"
    )
    print(f"  Fission rate              = {fission_rate:.8e} /s")
    print(f"  Integrated flux           = {integrated_flux:.8e} cm^-2 s^-1")
    print(f"  Single-spin gamma_+       = {excitation_rate:.8e} s^-1")
    print(f"  Single-spin gamma_-       = {deexcitation_rate:.8e} s^-1")
    print()
    print(
        f"At E_rms and R={args.sample_radius_cm:g} cm: E R = {er_rms:.8e}, "
        f"theta_coh ~ {1.0 / er_rms:.8e} rad"
    )
    print(f"Number of spins in sphere   = {number_of_spins:.8e}")
    print(
        "Leading coherent sigma_+,- = "
        f"{coherent_cross_section:.8e} cm^2 per ensemble"
    )
    print(f"Leading coherent Gamma_+,- = {coherent_rate:.8e} s^-1")
    print(
        f"Incoherent transverse Gamma_+ = {incoherent_excitation_rate:.8e} s^-1"
    )
    print(
        f"Incoherent transverse Gamma_- = {incoherent_deexcitation_rate:.8e} s^-1"
    )
    if incoherent_excitation_rate > 0.0:
        print(
            "Coherent/incoherent Gamma_+ = "
            f"{coherent_rate / incoherent_excitation_rate:.8g}"
        )
    if incoherent_deexcitation_rate > 0.0:
        print(
            "Coherent/incoherent Gamma_- = "
            f"{coherent_rate / incoherent_deexcitation_rate:.8g}"
        )


if __name__ == "__main__":
    main()
