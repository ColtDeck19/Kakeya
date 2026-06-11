# Kakeya sets in F_q^2: minimum sizes via ILP (small q) and randomised
# greedy (larger q), compared against the Dvir bound C_(q+1)^2 and the
# sharp Blokhuis-Mazzocca planar minimum.
# To run the script: sage finite_field.sage.

import time
import random
import csv

TARGET_Q = [
    2, 3, 4, 5, 7, 8, 9, 11, 13, 16, 17, 19, 23, 25, 27, 29, 31,
    32, 37, 41, 43, 47, 49, 53, 59, 61, 64, 67, 71, 73, 79, 81, 83,
    89, 97
]

ILP_THRESHOLD = 5
ILP_SOLVER = "GLPK"
GREEDY_TRIALS = 500
SEED = 42


def get_lines(Fq):
    """All affine lines in Fq^2, grouped by direction.

    """
    elems = list(Fq)
    idx = {e: i for i, e in enumerate(elems)}
    directions = []

    for d in elems:
        directions.append([
            frozenset((idx[t], idx[t * d + b]) for t in elems)
            for b in elems
        ])

    # vertical lines x = b
    directions.append([
        frozenset((idx[b], idx[t]) for t in elems)
        for b in elems
    ])

    return directions, elems, idx


def verify_kakeya(K, directions):
    """Check that K contains a full line in every direction."""
    return all(any(line.issubset(K) for line in lines_d)
               for lines_d in directions)


def min_kakeya_ilp(q, Fq, solver=ILP_SOLVER):
    """Exact minimum Kakeya set via ILP (set cover with one line per
    direction). Variables: x_p (point in K), y_l (line chosen as the
    representative of its direction)..
    """
    directions, elems, idx = get_lines(Fq)
    n_pts = q * q

    all_lines = []
    dir_line_ids = []
    for lines_d in directions:
        ids = []
        for line in lines_d:
            all_lines.append(line)
            ids.append(len(all_lines) - 1)
        dir_line_ids.append(ids)

    points = [(i, j) for i in range(q) for j in range(q)]
    pt_idx = {p: k for k, p in enumerate(points)}

    prog = MixedIntegerLinearProgram(maximization=False, solver=solver)
    x = prog.new_variable(binary=True)
    y = prog.new_variable(binary=True)

    prog.set_objective(sum(x[k] for k in range(n_pts)))

    # exactly one representative line per direction
    for line_ids in dir_line_ids:
        prog.add_constraint(sum(y[li] for li in line_ids) == 1)

    # a chosen line forces its points into K
    for li, line in enumerate(all_lines):
        for pt in line:
            prog.add_constraint(x[pt_idx[pt]] >= y[li])

    prog.solve()
    x_val = prog.get_values(x)
    K = {points[k] for k in range(n_pts) if float(x_val[k]) > 0.5}
    return K, directions


def min_kakeya_greedy(q, Fq, n_trials=GREEDY_TRIALS, seed=SEED):
    """Randomised greedy heuristic. Each trial visits the q+1 directions
    in random order and picks, in each direction, a line of maximal
    overlap with the current set (ties broken randomly). Returns the best
    set over all trials. this is an upper bound on the true minimum. and is (often) not exact.
    """
    random.seed(int(seed))
    directions, elems, idx = get_lines(Fq)
    best_K = None

    for _ in range(n_trials):
        order = list(range(len(directions)))
        random.shuffle(order)
        K = set()
        for d_id in order:
            # the random term only breaks ties: it never overrides an
            # overlap difference of 1 or more
            scored = [(len(line & K) + random.random() * 0.5, line)
                      for line in directions[d_id]]
            K |= max(scored, key=lambda item: item[0])[1]
        if best_K is None or len(K) < len(best_K):
            best_K = K

    return best_K, directions


def dvir_bound_2d(q):
    """Dvir lower bound in dimension 2: binom(q+1,2) = q(q+1)/2."""
    return q * (q + 1) // 2


def blokhuis_mazzocca_min_2d(q):
    base = q * (q + 1) // 2
    return base if q % 2 == 0 else base + (q - 1) // 2

header = (
    f"{'q':>3}  {'method':>8}  {'#K':>5}  {'#K/q^2':>8}  "
    f"{'Dvir':>6}  {'BM_min':>7}  {'(BM-Dvir)/q^2':>16}  "
    f"{'valid':>5}  {'attains_BM':>10}  {'time':>8}"
)
sep = "-" * len(header)
print(sep)
print(header)
print(sep)

results = []
t_total_start = time.perf_counter()

for q in TARGET_Q:
    Fq = GF(q)
    t0 = time.perf_counter()

    if q <= ILP_THRESHOLD:
        K, directions = min_kakeya_ilp(q, Fq)
        method = "ILP"
    else:
        K, directions = min_kakeya_greedy(q, Fq)
        method = "greedy"

    elapsed = float(time.perf_counter() - t0)
    size = int(len(K))
    density = float(size) / float(q * q)
    dvir = dvir_bound_2d(q)
    bm = blokhuis_mazzocca_min_2d(q)
    gap_normalized = float(bm - dvir) / float(q * q)
    valid = verify_kakeya(K, directions)
    attains_bm = bool(valid and size == bm)

    print(
        f"{q:>3}  {method:>8}  {size:>5}  {density:>8.4f}  "
        f"{dvir:>6}  {bm:>7}  {gap_normalized:>16.6f}  "
        f"{'Y' if valid else 'N':>5}  "
        f"{'Y' if attains_bm else 'N':>10}  "
        f"{elapsed:>7.2f}s"
    )

    results.append({
        "q": q, "method": method, "size": size, "density": density,
        "dvir": dvir, "bm_min": bm, "gap_normalized": gap_normalized,
        "valid": valid, "attains_bm": attains_bm, "elapsed": elapsed,
    })

print(sep)
print()
print("For even q, BM = Dvir; for odd q, BM - Dvir = (q-1)/2.")

csv_filename = "kakeya_finite_field.csv"
with open(csv_filename, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "q", "method", "#K", "#K/q^2", "Dvir_2d", "BM_min_2d",
        "(BM-Dvir)/q^2", "valid", "attains_BM", "elapsed_s",
    ])
    writer.writeheader()
    for r in results:
        writer.writerow({
            "q": r["q"],
            "method": r["method"],
            "#K": r["size"],
            "#K/q^2": f"{r['density']:.6f}",
            "Dvir_2d": r["dvir"],
            "BM_min_2d": r["bm_min"],
            "(BM-Dvir)/q^2": f"{r['gap_normalized']:.6f}",
            "valid": "yes" if r["valid"] else "no",
            "attains_BM": "yes" if r["attains_bm"] else "no",
            "elapsed_s": f"{r['elapsed']:.6f}",
        })

print(f"Results written to {csv_filename}")
print(f"Total runtime: {float(time.perf_counter() - t_total_start):.1f}s")
