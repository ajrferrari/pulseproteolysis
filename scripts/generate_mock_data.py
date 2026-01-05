#!/usr/bin/env python3
import numpy as np
import pandas as pd
from numpy.random import default_rng
import matplotlib.pyplot as plt
import argparse

# -----------------------------
# Core generators
# -----------------------------
def generate_initial_counts(N, total_counts, sigma=0.3, rng=None):
    """
    Initial library composition (no-protease) as multinomial from lognormal weights.
    """
    rng = default_rng() if rng is None else rng
    logw = rng.normal(loc=0.0, scale=sigma, size=N)
    w = np.exp(logw)
    a0 = w / w.sum()
    counts0 = rng.multinomial(int(total_counts), a0)
    return counts0, a0


def generate_thermodynamic_params(
    N,
    dGH2O_min=-2.0,
    dGH2O_max=12.0,
    m_mean=1.4,
    m_lognorm_sigma=0.2,
    rng=None,
):
    """
    Generate per-protein (dG_H2O, m, Cm) with positive m via lognormal.
    """
    rng = default_rng() if rng is None else rng

    dG_H2O = rng.uniform(dGH2O_min, dGH2O_max, size=N)

    # lognormal centered approximately at m_mean (median ~ m_mean)
    m_vals = rng.lognormal(mean=np.log(m_mean), sigma=m_lognorm_sigma, size=N)
    m_vals = np.clip(m_vals, 1e-6, None)

    Cm = dG_H2O / m_vals
    return dG_H2O, m_vals, Cm


def survival_matrix(Cm_array, m_array, conc, survival_min=0.0, RT=0.596, rng=None):
    """
    survival fraction for each protein (N) across concentrations (K).
    Returns shape (N, K).
    survival_min can be:
      - float
      - (low, high) tuple: samples ONE global floor
      - array of shape (N,) : per-protein floors
    """
    Cm_array = np.asarray(Cm_array)
    m_array = np.asarray(m_array)
    conc = np.asarray(conc)

    N = Cm_array.shape[0]

    if isinstance(survival_min, tuple):
        low, high = survival_min
        rng = default_rng() if rng is None else rng
        survival_min = rng.uniform(low, high)

    if survival_min is None:
        survival_min = 0.0

    if np.isscalar(survival_min):
        floors = float(survival_min) * np.ones(N)
    else:
        floors = np.asarray(survival_min)
        if floors.shape[0] != N:
            raise ValueError("survival_min array must have length N")

    dG = m_array[:, None] * (Cm_array[:, None] - conc[None, :])  # (N,K)
    K_eq = np.exp(-dG / RT)
    S = 1.0 / (1.0 + K_eq)
    S = np.maximum(S, floors[:, None])
    return S


def add_noise_realistic(S,
                        lognormal_sigma=0.05,
                        gaussian_sd=0.005,
                        kappa=1000,
                        rng=None):
    """
    Multiplicative (lognormal) + Beta overdispersion + additive Gaussian.
    S shape (N,K), returns (N,K) clipped to [0,1].
    """
    rng = default_rng() if rng is None else rng

    S1 = S * rng.lognormal(mean=0.0, sigma=lognormal_sigma, size=S.shape)
    S1 = np.clip(S1, 1e-6, 1 - 1e-6)

    a = S1 * kappa
    b = (1 - S1) * kappa
    S2 = rng.beta(a, b)

    S3 = S2 + rng.normal(0.0, gaussian_sd, size=S.shape)
    return np.clip(S3, 0.0, 1.0)


def simulate_pulse_counts_from_survival(
    counts0,
    survival,
    total_counts=None,
    total_counts_std=None,
    rng=None
):
    """
    counts0: (N,)
    survival: (N,K)
    Returns counts matrix (K+1, N): row0 is counts0, rows1..K are per-conc counts.
    """
    rng = default_rng() if rng is None else rng

    counts0 = np.asarray(counts0, dtype=int)
    survival = np.asarray(survival, dtype=float)
    N = counts0.size
    N2, K = survival.shape
    if N2 != N:
        raise ValueError("counts0 and survival must agree on N")

    counts = np.zeros((K + 1, N), dtype=int)
    counts[0] = counts0

    for k in range(K):
        expected = counts0 * survival[:, k]

        if total_counts is None:
            total_k = int(max(expected.sum(), 1))
        else:
            if total_counts_std is None:
                total_k = int(total_counts)
            else:
                total_k = int(rng.normal(loc=total_counts, scale=total_counts_std))
                total_k = max(total_k, 1)

        s = expected.sum()
        probs = (expected / s) if s > 0 else (np.ones(N) / N)
        counts[k + 1] = rng.multinomial(total_k, probs)

    return counts


# -----------------------------
# High-level wrapper
# -----------------------------
def generate_mock_pulse_dataset(
    N=2000,
    conc=None,
    total_counts=200_000,
    total_counts_std_frac=0.2,      # sequencing depth variation per condition
    init_sigma=0.4,                 # initial library heterogeneity
    dGH2O_min=-2.0,
    dGH2O_max=10.0,
    m_mean=1.4,
    m_lognorm_sigma=0.2,
    survival_floor_mode="per_protein",  # "per_protein" | "global" | "fixed"
    floor_low=1e-5,
    floor_high=1e-4,
    add_noise=True,
    noise_lognormal_sigma=0.05,
    noise_gaussian_sd=0.005,
    noise_kappa=1000,
    seed=42,
):
    """
    Returns:
      counts_true_df, counts_noisy_df, params_df, (S_true, S_noisy)
    """
    rng = default_rng(seed)

    if conc is None:
        conc = np.linspace(0, 8, 25)
    conc = np.asarray(conc, dtype=float)
    K = conc.size

    # params
    dG_H2O, m_vals, Cm = generate_thermodynamic_params(
        N=N,
        dGH2O_min=dGH2O_min,
        dGH2O_max=dGH2O_max,
        m_mean=m_mean,
        m_lognorm_sigma=m_lognorm_sigma,
        rng=rng,
    )

    # floors
    if survival_floor_mode == "per_protein":
        floors = rng.uniform(floor_low, floor_high, size=N)
    elif survival_floor_mode == "global":
        floors = (floor_low, floor_high)  # sampled once inside survival_matrix
    elif survival_floor_mode == "fixed":
        floors = float(floor_low)
    else:
        raise ValueError("survival_floor_mode must be per_protein|global|fixed")

    # initial counts
    counts0, a0 = generate_initial_counts(N=N, total_counts=total_counts, sigma=init_sigma, rng=rng)

    # survival
    S_true = survival_matrix(Cm, m_vals, conc, survival_min=floors, rng=rng)

    # noisy survival
    if add_noise:
        S_noisy = add_noise_realistic(
            S_true,
            lognormal_sigma=noise_lognormal_sigma,
            gaussian_sd=noise_gaussian_sd,
            kappa=noise_kappa,
            rng=rng
        )
    else:
        S_noisy = S_true.copy()

    total_counts_std = int(total_counts * total_counts_std_frac) if total_counts_std_frac is not None else None

    # counts from survival (true vs noisy)
    counts_true = simulate_pulse_counts_from_survival(
        counts0, S_true, total_counts=total_counts, total_counts_std=total_counts_std, rng=rng
    )
    counts_noisy = simulate_pulse_counts_from_survival(
        counts0, S_noisy, total_counts=total_counts, total_counts_std=total_counts_std, rng=rng
    )

    # Build tables
    protein_id = np.array([f"prot_{i:06d}" for i in range(N)])

    colnames = ["no_protease"] + [f"c_{c:.3f}M" for c in conc]
    # (K+1,N) -> (N,K+1)
    counts_true_df = pd.DataFrame(counts_true.T, index=protein_id, columns=colnames)
    counts_noisy_df = pd.DataFrame(counts_noisy.T, index=protein_id, columns=colnames)

    # params table
    if survival_floor_mode == "per_protein":
        floor_vec = np.asarray(floors)
    elif survival_floor_mode == "fixed":
        floor_vec = np.full(N, float(floors))
    else:  # global
        # we don't know the realized value unless we compute it; easiest is recompute:
        # (survival_matrix sampled it internally). We can back-calc per protein from S_true at high conc:
        # but better: just set NaN and store the tuple
        floor_vec = np.full(N, np.nan)

    params_df = pd.DataFrame({
        "protein_id": protein_id,
        "dG_H2O": dG_H2O,
        "m": m_vals,
        "Cm": Cm,
        "survival_floor": floor_vec,
    }).set_index("protein_id")

    return counts_true_df, counts_noisy_df, params_df, (S_true, S_noisy), conc


def save_dataset(
    counts_df,
    params_df,
    conc,
    out_counts_csv="mock_counts.csv",
    out_params_csv="mock_params.csv",
    out_counts_json=None,
    out_params_json=None,
):
    """
    Saves counts with proteins as rows and conc as columns, plus params table.
    """
    counts_df.to_csv(out_counts_csv)
    params_df.to_csv(out_params_csv)

    if out_counts_json is not None:
        # include protein_id as a field in JSON records
        tmp = counts_df.reset_index().rename(columns={"index": "protein_id"})
        tmp.to_json(out_counts_json, orient="records")

    if out_params_json is not None:
        tmp = params_df.reset_index()
        tmp.to_json(out_params_json, orient="records")

    # handy metadata (optional to save)
    meta = {
        "n_proteins": int(counts_df.shape[0]),
        "n_conditions": int(len(conc)),
        "concentrations_M": [float(x) for x in conc],
        "counts_columns": list(counts_df.columns),
    }
    return meta

def plot_survival_examples(conc, S_true, S_noisy, n_show=15,
                           out_png="survival_noise_examples.png"):
    """
    Saves a diagnostic plot: true vs noisy survival curves for a few proteins.
    True and noisy curves share the same color; noise is shown as dashed.
    """
    plt.figure(figsize=(5, 3.2))

    # Use a colormap to ensure consistent colors
    cmap = plt.get_cmap("tab10")
    n_show = min(n_show, S_true.shape[0])

    for i in range(n_show):
        color = cmap(i % cmap.N)

        # True curve
        plt.plot(
            conc, S_true[i],
            color=color,
            lw=1.,
            alpha=0.9,
            label="True" if i == 0 else None
        )

        # Noisy curve (same color, dashed)
        plt.plot(
            conc, S_noisy[i],
            color=color,
            lw=1.,
            ls="--",
            alpha=0.9,
            label="Noisy" if i == 0 else None
        )

    plt.xlabel("[Denaturant] (M)")
    plt.ylabel("Survival fraction")

    # Single legend entry
    plt.legend(frameon=False, loc="best")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate mock pulse-proteolysis count tables"
    )

    # -----------------------------
    # Core sizes
    # -----------------------------
    parser.add_argument("--N", type=int, default=12000,
                        help="Number of proteins (default: 12000)")
    parser.add_argument("--n_conditions", type=int, default=20,
                        help="Number of denaturant concentrations (default: 20)")
    parser.add_argument("--conc_min", type=float, default=0.0,
                        help="Minimum denaturant concentration (M)")
    parser.add_argument("--conc_max", type=float, default=8.0,
                        help="Maximum denaturant concentration (M)")

    # -----------------------------
    # Sequencing depth
    # -----------------------------
    parser.add_argument("--total_counts", type=int, default=1e6,
                        help="Total sequencing reads per condition")
    parser.add_argument("--total_counts_std_frac", type=float, default=0.2,
                        help="Relative std of sequencing depth (fraction of total_counts)")

    # -----------------------------
    # Initial library heterogeneity
    # -----------------------------
    parser.add_argument("--init_sigma", type=float, default=0.4,
                        help="Lognormal sigma for initial library composition")

    # -----------------------------
    # Thermodynamics
    # -----------------------------
    parser.add_argument("--dGH2O_min", type=float, default=-2.0)
    parser.add_argument("--dGH2O_max", type=float, default=10.0)

    parser.add_argument("--m_mean", type=float, default=1.4,
                        help="Mean m-value")
    parser.add_argument("--m_lognorm_sigma", type=float, default=0.2,
                        help="Lognormal sigma for m-values")

    # -----------------------------
    # Survival floor
    # -----------------------------
    parser.add_argument("--survival_floor_mode",
                        choices=["per_protein", "global", "fixed"],
                        default="per_protein",
                        help="How to apply survival floor")
    parser.add_argument("--floor_low", type=float, default=1e-5)
    parser.add_argument("--floor_high", type=float, default=1e-4)

    # -----------------------------
    # Noise model
    # -----------------------------
    parser.add_argument("--add_noise", action="store_true",
                        help="Add realistic noise to survival curves")
    parser.add_argument("--noise_lognormal_sigma", type=float, default=0.05)
    parser.add_argument("--noise_gaussian_sd", type=float, default=0.005)
    parser.add_argument("--noise_kappa", type=float, default=1000)

    # -----------------------------
    # Output
    # -----------------------------
    parser.add_argument("--out_prefix", type=str, default="mock_pulse",
                        help="Prefix for output files")
    parser.add_argument("--save_json", action="store_true",
                        help="Also save JSON versions")
    parser.add_argument("--plot_examples", action="store_true",
                        help="Save survival curve example plot")

    # -----------------------------
    # Reproducibility
    # -----------------------------
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()

    rng = default_rng(args.seed)

    conc = np.linspace(args.conc_min, args.conc_max, args.n_conditions)

    counts_true_df, counts_noisy_df, params_df, (S_true, S_noisy), conc = \
        generate_mock_pulse_dataset(
            N=args.N,
            conc=conc,
            total_counts=args.total_counts,
            total_counts_std_frac=args.total_counts_std_frac,
            init_sigma=args.init_sigma,
            dGH2O_min=args.dGH2O_min,
            dGH2O_max=args.dGH2O_max,
            m_mean=args.m_mean,
            m_lognorm_sigma=args.m_lognorm_sigma,
            survival_floor_mode=args.survival_floor_mode,
            floor_low=args.floor_low,
            floor_high=args.floor_high,
            add_noise=args.add_noise,
            noise_lognormal_sigma=args.noise_lognormal_sigma,
            noise_gaussian_sd=args.noise_gaussian_sd,
            noise_kappa=args.noise_kappa,
            seed=args.seed,
        )

    # -----------------------------
    # Save outputs
    # -----------------------------
    counts_csv = f"{args.out_prefix}_counts.csv"
    params_csv = f"{args.out_prefix}_params.csv"

    counts_json = f"{args.out_prefix}_counts.json" if args.save_json else None
    params_json = f"{args.out_prefix}_params.json" if args.save_json else None

    meta = save_dataset(
        counts_df=counts_noisy_df if args.add_noise else counts_true_df,
        params_df=params_df,
        conc=conc,
        out_counts_csv=counts_csv,
        out_params_csv=params_csv,
        out_counts_json=counts_json,
        out_params_json=params_json,
    )

    print("=== Dataset generated ===")
    for k, v in meta.items():
        print(f"{k}: {v}")

    # -----------------------------
    # Optional plot
    # -----------------------------
    if args.plot_examples:
        plot_path = f"{args.out_prefix}_survival_examples.png"
        plot_survival_examples(conc, S_true, S_noisy, out_png=plot_path)
        print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()

# -----------------------------
# Example usage
# -----------------------------
#    counts_true_df, counts_noisy_df, params_df, (S_true, S_noisy), conc = generate_mock_pulse_dataset(
#        N=5000,
#        conc=np.linspace(0, 8, 25),
#        total_counts=200_000,
#        total_counts_std_frac=0.2,
#        init_sigma=0.4,
#        dGH2O_min=-2,
#        dGH2O_max=10,
#        m_mean=1.4,
#        m_lognorm_sigma=0.2,
#        survival_floor_mode="per_protein",  # or "global" or "fixed"
#        floor_low=1e-5,
#        floor_high=1e-4,
#        add_noise=True,
#        noise_lognormal_sigma=0.05,
#        noise_gaussian_sd=0.005,
#        noise_kappa=1000,
#        seed=42,
#    )
#
#    # Save noisy counts (usually what you fit), plus params truth
#    meta = save_dataset(
#        counts_df=counts_noisy_df,
#        params_df=params_df,
#        conc=conc,
#        out_counts_csv="mock_counts_noisy.csv",
#        out_params_csv="mock_params_truth.csv",
#        out_counts_json="mock_counts_noisy.json",
#        out_params_json="mock_params_truth.json",
#    )
#    print("Saved. Meta:", meta)
#
#    plot_survival_examples(conc, S_true, S_noisy, n_show=10, out_png="mock_survival_noise_examples.png")
#    print("Saved plot: mock_survival_noise_examples.png")
