# Perron tree area decay. Builds the level-n Perron tree from a unit
# equilateral seed triangle by recursive pair-and-slide, measures the
# union area with shapely/GEOS, and compares two overlap schedules:
# a constant fraction alpha = 1/2 (the area floors) and the tuned
# schedule alpha_m = (m+1)/(n+1) (the area keeps decreasing). Regression fits in u = 1/log N quantify this.
#
# To run it : python perron_area_decay.py [--max 14]

import argparse
import csv
import time

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union


def seed_triangle():
    """Unit equilateral triangle, base on the x-axis."""
    return np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, np.sqrt(3) / 2],
    ])


def bisect_triangle(tri, n):
    """Split tri into 2^n thin triangles sharing the apex.

    The base (vertex 0 to vertex 1) is cut into 2^n equal pieces and each
    piece is joined to the apex (vertex 2).
    """
    base_left, base_right, apex = tri
    k = 2 ** n
    ts = np.linspace(0.0, 1.0, k + 1)
    base_pts = base_left + np.outer(ts, base_right - base_left)
    return [np.array([base_pts[i], base_pts[i + 1], apex]) for i in range(k)]


def constant_schedule(f):
    """Same overlap fraction at every merge level."""
    return lambda m, n: f


def tuned_schedule(m, n):
    """Overlap fraction (m+1)/(n+1) at merge level m of a level-n tree."""
    return (m + 1) / (n + 1)


def perron_tree(n, schedule=tuned_schedule):
    """Level-n Perron tree as a list of (3,2) triangles.
    """
    blocks = [[t] for t in bisect_triangle(seed_triangle(), n)]
    width = 1.0 / 2 ** n
    m = 0
    while len(blocks) > 1:
        f = schedule(m, n)
        merged = []
        for j in range(0, len(blocks), 2):
            left, right = blocks[j], blocks[j + 1]
            shifted = [t + np.array([-f * width, 0.0]) for t in right]
            merged.append(left + shifted)
        blocks = merged
        width *= 2
        m += 1
    return blocks[0]


def union_area(tris):
    """Area of the union (overlap counted once), via GEOS."""
    return unary_union([Polygon(t) for t in tris]).area


def r2(y, yhat):
    y = np.asarray(y)
    return 1 - np.sum((y - yhat) ** 2) / np.sum((y - np.mean(y)) ** 2)


def fit_free(u, y):
    """area = s*u + c (free intercept)."""
    s, c = np.polyfit(u, y, 1)
    return (s, c), s * u + c


def fit_origin(u, y):
    """area = s*u (through the origin)."""
    s = np.sum(u * y) / np.sum(u * u)
    return (s,), s * u


def fit_two_term(u, y):
    """area = a*u + b*u^2 (through the origin, with curvature)."""
    A = np.column_stack([u, u ** 2])
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    return (a, b), a * u + b * u ** 2


def main():
    parser = argparse.ArgumentParser(
        description="Perron tree area decay, constant vs tuned schedule")
    parser.add_argument("--max", type=int, default=14,
                        help="maximum level n (default: 14)")
    args = parser.parse_args()

    ns = list(range(0, args.max + 1))

    print(f"{'n':>3}  {'2^n':>6}  {'tuned':>10}  {'constant 1/2':>13}  "
          f"{'time':>8}")
    areas_tuned, areas_const = [], []
    for n in ns:
        t0 = time.perf_counter()
        a_t = union_area(perron_tree(n, tuned_schedule))
        a_c = union_area(perron_tree(n, constant_schedule(0.5)))
        t = time.perf_counter() - t0
        areas_tuned.append(a_t)
        areas_const.append(a_c)
        print(f"{n:>3}  {2**n:>6}  {a_t:>10.6f}  {a_c:>13.6f}  {t:>7.2f}s")

    # Regression in u = 1/log N (skip n = 0, where log N = 0). A model
    # through the origin can only fit if the area really decays to zero.
    logN = np.array(ns[1:]) * np.log(2)
    u = 1.0 / logN
    at = np.array(areas_tuned[1:])
    ac = np.array(areas_const[1:])

    (s_t, c_t), p_free_t = fit_free(u, at)
    (so_t,), p_orig_t = fit_origin(u, at)
    (a_t2, b_t2), p_two_t = fit_two_term(u, at)
    (s_c, c_c), p_free_c = fit_free(u, ac)
    (so_c,), p_orig_c = fit_origin(u, ac)

    print("\nFits (u = 1/log N):")
    print(f"  tuned    a/logN + c : a={s_t:.4f}  c={c_t:.4f}  "
          f"R^2={r2(at, p_free_t):.4f}")
    print(f"  tuned    a/logN     : a={so_t:.4f}            "
          f"R^2={r2(at, p_orig_t):.4f}")
    print(f"  tuned    a/logN + b/logN^2 : a={a_t2:.4f}  b={b_t2:.4f}  "
          f"R^2={r2(at, p_two_t):.4f}")
    print(f"  constant a/logN + c : a={s_c:.4f}  c={c_c:.4f}  "
          f"R^2={r2(ac, p_free_c):.4f}")
    print(f"  constant a/logN     : a={so_c:.4f}            "
          f"R^2={r2(ac, p_orig_c):.4f}")

    # CSV consumed directly by the pgfplots figures in the report;
    # column names (inv_logN, area_tuned, area_constant) must not change.
    decay_csv = "perron_area_decay.csv"
    with open(decay_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "N", "area_constant", "area_tuned",
                    "logN", "inv_logN"])
        for n, a_c, a_t in zip(ns, areas_const, areas_tuned):
            N = 2 ** n
            if n == 0:
                lN = iv = ""
            else:
                lN = f"{np.log(N):.6f}"
                iv = f"{1.0 / np.log(N):.6f}"
            w.writerow([n, N, f"{a_c:.6f}", f"{a_t:.6f}", lN, iv])

    fits_csv = "perron_fits.csv"
    with open(fits_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["series", "model", "coef1", "coef2", "R2"])
        w.writerow(["tuned", "a/logN + c", f"{s_t:.4f}", f"{c_t:.4f}",
                    f"{r2(at, p_free_t):.4f}"])
        w.writerow(["tuned", "a/logN", f"{so_t:.4f}", "",
                    f"{r2(at, p_orig_t):.4f}"])
        w.writerow(["tuned", "a/logN + b/logN^2", f"{a_t2:.4f}",
                    f"{b_t2:.4f}", f"{r2(at, p_two_t):.4f}"])
        w.writerow(["constant", "a/logN + c", f"{s_c:.4f}", f"{c_c:.4f}",
                    f"{r2(ac, p_free_c):.4f}"])
        w.writerow(["constant", "a/logN", f"{so_c:.4f}", "",
                    f"{r2(ac, p_orig_c):.4f}"])

    print(f"\nResults written to {decay_csv} and {fits_csv}")


if __name__ == "__main__":
    main()
