"""Evaluate every ghost on held-out levels (seeds never used in training).

Uses the NumPy ghost files, so it measures exactly what the demo shows.

    python evaluate.py --ghosts runs/main/ghosts --levels 100
    python evaluate.py --ghosts runs/main/ghosts --plot results.png
    python evaluate.py --ghosts runs/main/ghosts --only-new     # skip ghosts already in the CSV

Writes <run>/eval.csv with one row per ghost x eval set x difficulty. If the run
trained on a fixed set of levels (train.py --train-levels N), each ghost is also
evaluated on those training levels (eval_set = "train"), which is what the
generalisation experiment compares against "unseen".
"""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from procgen_rl import config as C
from procgen_rl.level import generate_level
from procgen_rl.policy import load_ghosts
from procgen_rl.sim import FINISHED, RUNNING, Runner

FIELDS = ["label", "sim_hours", "eval_set", "difficulty", "levels",
          "success_rate", "mean_progress", "mean_finish_time"]


def run_episode(policy, level, deterministic=True, rng=None, record=False):
    """Play one level. Returns (success, progress, seconds, trajectory or None)."""
    r = Runner(level)
    steps = stall = 0
    best = r.max_x
    traj = [(r.state.cx, r.state.cy)] if record else None
    while r.status == RUNNING and steps < C.MAX_EPISODE_STEPS and stall < C.STALL_STEPS:
        r.act(policy.act(r.observe(), deterministic=deterministic, rng=rng))
        steps += 1
        if record:
            traj.append((r.state.cx, r.state.cy))
        if r.max_x > best + 1e-6:
            best, stall = r.max_x, 0
        else:
            stall += 1
    return r.status == FINISHED, r.progress, r.frames / C.FPS, traj


def run_info(ghost_dir):
    """Training setup of the run that produced these ghosts (from state.json)."""
    state = Path(ghost_dir).parent / "state.json"
    if state.exists():
        s = json.loads(state.read_text())
        return int(s.get("train_levels") or 0), s.get("fixed_difficulty")
    return 0, None


def eval_sets(ghost_dir, difficulties, n_levels):
    sets = [("unseen", d, [C.EVAL_SEED_START + i for i in range(n_levels)]) for d in difficulties]
    train_levels, fixed_d = run_info(ghost_dir)
    if train_levels > 0:
        d = fixed_d if fixed_d is not None else 0.5
        sets.append(("train", d, list(range(min(train_levels, n_levels)))))
        if all(abs(d - x) > 1e-9 for x in difficulties):
            sets.append(("unseen", d, [C.EVAL_SEED_START + i for i in range(n_levels)]))
    return sets


def read_rows(path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != FIELDS:
            return []  # older format: re-evaluate everything
        return list(reader)


def write_rows(path, rows):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ghosts", default="ghosts")
    p.add_argument("--levels", type=int, default=100, help="held-out levels per difficulty")
    p.add_argument("--difficulties", default="0.25,0.5,0.75,1.0")
    p.add_argument("--stochastic", action="store_true", help="sample actions instead of argmax")
    p.add_argument("--only-new", action="store_true", help="keep existing CSV rows, evaluate missing ones")
    p.add_argument("--csv", default=None, help="output CSV (default: <ghosts>/../eval.csv)")
    p.add_argument("--plot", default=None, help="save a quick PNG plot here (needs matplotlib)")
    args = p.parse_args()

    ghosts, skipped = load_ghosts(args.ghosts)
    for name, why in skipped:
        print(f"skipped {name}: {why}")
    if not ghosts:
        raise SystemExit(f"No compatible ghosts in {args.ghosts}")
    diffs = [float(d) for d in args.difficulties.split(",")]
    sets = eval_sets(args.ghosts, diffs, args.levels)
    csv_path = Path(args.csv) if args.csv else Path(args.ghosts).parent / "eval.csv"

    rows = read_rows(csv_path) if args.only_new else []
    done = {(r["label"], r["eval_set"], f"{float(r['difficulty']):g}") for r in rows}
    level_cache = {}

    for g in ghosts:
        rng = np.random.default_rng(0)
        cells = []
        for eval_set, d, seeds in sets:
            if (g.label, eval_set, f"{d:g}") in done:
                continue
            key = (eval_set, d)
            if key not in level_cache:
                level_cache[key] = [generate_level(s, d) for s in seeds]
            results = [run_episode(g, lv, not args.stochastic, rng) for lv in level_cache[key]]
            succ = [r for r in results if r[0]]
            rows.append({
                "label": g.label, "sim_hours": g.sim_hours, "eval_set": eval_set, "difficulty": d,
                "levels": len(results),
                "success_rate": round(float(np.mean([r[0] for r in results])), 4),
                "mean_progress": round(float(np.mean([r[1] for r in results])), 4),
                "mean_finish_time": round(float(np.mean([r[2] for r in succ])), 3) if succ else "",
            })
            cells.append(f"{eval_set} d={d:g}: {rows[-1]['success_rate']:.2f}/{rows[-1]['mean_progress']:.2f}")
        if cells:
            print(f"{g.label:>7}  " + "  ".join(cells), flush=True)
            rows.sort(key=lambda r: (float(r["sim_hours"]), r["eval_set"], float(r["difficulty"])))
            write_rows(csv_path, rows)  # write after every ghost so partial results are visible
    print(f"wrote {csv_path}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        for d in diffs:
            sel = [r for r in rows if r["eval_set"] == "unseen" and float(r["difficulty"]) == d]
            ax.plot([max(float(r["sim_hours"]), 0.05) for r in sel], [float(r["success_rate"]) for r in sel],
                    marker="o", label=f"difficulty {d:g}")
        ax.set_xscale("log")
        ax.set_xlabel("simulated training hours")
        ax.set_ylabel("success rate on unseen levels")
        ax.set_ylim(0, 1.02)
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.plot, dpi=120)
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
