# Box-counting dimension of the needle star associated with the Perron
# tree, run at three raster resolutions (G = 1024, 2048, 4096) to show
# both the increase of the estimate toward d_B = 2 and the
# grid-saturation ceiling.
#
# To run it : python box_counting_perron.py [--max 14]

import argparse
import csv
import math
import time

import numpy as np
from scipy import stats

GRIDS = [1024, 2048, 4096]


def rasterise_needles(n, G):
    """2^n unit segments on a G x G binary grid, sampled at 6G points each."""
    N = 2 ** n
    angles = np.linspace(0.0, np.pi, N, endpoint=False)
    grid = np.zeros((G, G), dtype=bool)
    ts = np.linspace(0.0, 1.0, 6 * G)

    for theta in angles:
        dx = np.cos(theta) * 0.85
        dy = np.sin(theta) * 0.85
        xs = -dx + ts * 2 * dx
        ys = -dy + ts * 2 * dy
        col = np.clip(((xs + 1.0) / 2.0 * (G - 1)).astype(int), 0, G - 1)
        row = np.clip(((ys + 1.0) / 2.0 * (G - 1)).astype(int), 0, G - 1)
        grid[row, col] = True

    return grid


def box_count(grid):
    """N(eps)"""
    G = grid.shape[0]
    max_k = int(math.log2(G))
    epsilons, counts = [], []

    for k in range(1, max_k):
        bs = 2 ** k
        G2 = (G // bs) * bs
        n_b = G2 // bs
        cnt = int(grid[:G2, :G2]
                  .reshape(n_b, bs, n_b, bs)
                  .any(axis=(1, 3)).sum())
        if cnt > 1:
            epsilons.append(bs / G)
            counts.append(cnt)

    return np.array(epsilons), np.array(counts)


def estimate_dim(eps, counts, skip_low=2, skip_high=2):
    log_inv_eps = np.log(1.0 / eps)
    log_N = np.log(counts.astype(float))
    a = skip_low
    b = max(a + 2, len(eps) - skip_high)
    slope, intercept, r, _, se = stats.linregress(log_inv_eps[a:b], log_N[a:b])
    return {"dim": slope, "se": se, "r2": r ** 2}


def main():
    parser = argparse.ArgumentParser(
        description="Box-counting dimension of the Perron needle star")
    parser.add_argument("--max", type=int, default=14,
                        help="maximum level n (default: 14)")
    args = parser.parse_args()

    print(f"Grids: {GRIDS}   Levels: 1 .. {args.max}")
    print("Theoretical result: d_B = 2 (Davies 1971)\n")

    col = "  ".join(f"{'G=' + str(G):>14}" for G in GRIDS)
    header = f"{'n':>3}  {'N':>6}  {col}"
    sub = "  ".join(f"{'d_B   t(s)':>14}" for _ in GRIDS)
    print("-" * len(header))
    print(header)
    print(f"{'':>3}  {'':>6}  {sub}")
    print("-" * len(header))

    rows = []
    for n in range(1, args.max + 1):
        line = f"{n:>3}  {2**n:>6}"
        record = {"n": n, "N": 2 ** n}
        for G in GRIDS:
            t0 = time.perf_counter()
            grid = rasterise_needles(n, G)
            eps, cnts = box_count(grid)
            if len(eps) >= 4:
                fit = estimate_dim(eps, cnts)
            else:
                fit = {"dim": float("nan"), "se": float("nan"),
                       "r2": float("nan")}
            t = time.perf_counter() - t0

            line += f"  {fit['dim']:.4f} {t:5.2f}s"
            record[f"dim_B_G{G}"] = fit["dim"]
            record[f"se_G{G}"] = fit["se"]
            record[f"r2_G{G}"] = fit["r2"]
        print(line)
        rows.append(record)

    print("-" * len(header))

    print("\nSaturation summary:")
    for G in GRIDS:
        ceiling = max(r[f"dim_B_G{G}"] for r in rows)
        print(f"  G={G}: ceiling ~ {ceiling:.4f}")

    csv_filename = "box_counting_perron.csv"
    fieldnames = ["n", "N"]
    for G in GRIDS:
        fieldnames += [f"dim_B_G{G}", f"se_G{G}", f"r2_G{G}"]

    with open(csv_filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            out = {"n": r["n"], "N": r["N"]}
            for G in GRIDS:
                out[f"dim_B_G{G}"] = f"{r[f'dim_B_G{G}']:.6f}"
                out[f"se_G{G}"] = f"{r[f'se_G{G}']:.6f}"
                out[f"r2_G{G}"] = f"{r[f'r2_G{G}']:.6f}"
            writer.writerow(out)

    print(f"\nResults written to {csv_filename}")


if __name__ == "__main__":
    main()
