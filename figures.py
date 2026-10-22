"""Generate presentation figures (PNG) from training runs.

    python figures.py                                  # auto: runs/main + runs/gen_*
    python figures.py --main runs/main --out runs/presentation

Writes numbered PNGs plus captions.json and summary.md into --out. Figures whose
data does not exist yet (e.g. no generalisation runs) are skipped, so this can be
re-run at any time during training. The web app shows the result as a gallery.

Colour use: categorical slots for comparing runs or train/unseen; one blue
ordinal ramp for anything ordered (training time, difficulty). No dual axes.
"""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

from procgen_rl import config as C  # noqa: E402
from procgen_rl.level import generate_level  # noqa: E402
from procgen_rl.policy import load_ghosts  # noqa: E402
from procgen_rl.sim import Runner  # noqa: E402

# ---------------------------------------------------------------- style
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
ORDINAL = {  # validated single-hue blue ordinal ramps (light surface)
    1: ["#2a78d6"],
    2: ["#86b6ef", "#1c5cab"],
    3: ["#86b6ef", "#2a78d6", "#0d366b"],
    4: ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
    5: ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"],
}
SEQ_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "blue_seq", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
# game colours for level illustrations
SKY = "#e9f2fc"
DIRT = "#9b6b47"
GRASS = "#5aae48"
SPIKE = "#8d929e"
FLAG = "#e34948"

_INSTALLED = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
plt.rcParams.update({
    "font.family": [f for f in ("Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans") if f in _INSTALLED]
                   or ["sans-serif"],
    "font.size": 11,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "axes.titlesize": 13,
    "axes.titleweight": "semibold",
    "axes.titlelocation": "left",
    "axes.titlepad": 10,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "lines.linewidth": 2.0,
    "legend.frameon": False,
    "legend.labelcolor": INK2,
})

DPI = 160
CAPTIONS = []


def save(fig, out, name, title, caption):
    path = out / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    CAPTIONS.append({"file": name, "title": title, "caption": caption})
    print(f"  wrote {name}")


def pct_axis(ax):
    ax.set_ylim(0, 1.03)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))


NICE_HOURS = [0, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 300, 500, 1000]


def hours_axis(ax, log=True):
    """Symlog time axis (linear below 1 h) with readable fixed ticks. Call after plotting."""
    ax.set_xlabel("simulated training hours")
    if log:
        xmax = ax.dataLim.x1
        ax.set_xscale("symlog", linthresh=1.0, linscale=0.6)
        # limits computed on the linear scale would leave a huge empty band left of 0 under symlog
        ax.set_xlim(-0.08, xmax * 1.12)
        ticks = [t for t in NICE_HOURS if t <= xmax * 1.05]
        # drop ticks that would crowd on a narrow axis (keep the last one, it is the final ghost)
        px = lambda t: ax.transData.transform((t, 0))[0]  # noqa: E731
        kept = [ticks[-1]]
        for t in reversed(ticks[:-1]):
            if px(kept[-1]) - px(t) >= 30:
                kept.append(t)
        ticks = sorted(kept)
        ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))


def end_labels(ax, items, min_gap=13, dot=True):
    """Direct labels at line ends, nudged apart vertically so they never overlap.

    items: [(x, y, text, color)]. Call after the y-limits are final.
    """
    h_pts = ax.get_window_extent().height * 72 / ax.figure.dpi
    y0, y1 = ax.get_ylim()
    to_pts = lambda y: (y - y0) / (y1 - y0) * h_pts  # noqa: E731
    order = sorted(items, key=lambda it: it[1])
    placed = []
    for _, y, _, _ in order:
        p = to_pts(y)
        if placed and p - placed[-1] < min_gap:
            p = placed[-1] + min_gap
        placed.append(p)
    for (x, y, text, color), p in zip(order, placed):
        ax.annotate(text, (x, y), xytext=(7, p - to_pts(y)), textcoords="offset points", va="center",
                    color=INK2, fontsize=9.5, annotation_clip=False)
        if dot:
            ax.plot([x], [y], "o", color=color, ms=6, mec=SURFACE, mew=2, zorder=5)


# ---------------------------------------------------------------- data loading
def read_metrics(run):
    p = Path(run) / "metrics.jsonl"
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def read_eval(run, name="eval.csv"):
    p = Path(run) / name
    if not p.exists():
        return []
    with open(p, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["sim_hours"] = float(r["sim_hours"])
        r["difficulty"] = float(r["difficulty"])
        r["success_rate"] = float(r["success_rate"])
        r["mean_progress"] = float(r["mean_progress"])
        r["mean_finish_time"] = float(r["mean_finish_time"]) if r.get("mean_finish_time") else None
    return rows


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default


def series(rows, key):
    xs = [r["sim_hours"] for r in rows if r.get(key) is not None]
    ys = [r[key] for r in rows if r.get(key) is not None]
    return np.array(xs), np.array(ys)


def smooth(y, k):
    if len(y) < k or k <= 1:
        return y
    pad = np.pad(y, (k // 2, k - 1 - k // 2), mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def pick_ghosts(ghosts, n=5):
    """n ghosts spread evenly in log-time, always including first and last."""
    if len(ghosts) <= n:
        return ghosts
    hrs = np.array([math.log10(g.sim_hours + 0.1) for g in ghosts])
    targets = np.linspace(hrs[0], hrs[-1], n)
    idx = sorted({int(np.argmin(np.abs(hrs - t))) for t in targets})
    return [ghosts[i] for i in idx]


# ---------------------------------------------------------------- level drawing
def level_image(level, c0, c1, px=6):
    H = level.height
    sky = np.array(matplotlib.colors.to_rgb(SKY))
    dirt = np.array(matplotlib.colors.to_rgb(DIRT))
    grass = np.array(matplotlib.colors.to_rgb(GRASS))
    img = np.tile(sky, (H * px, (c1 - c0) * px, 1))
    g = level.grid
    for r in range(H):
        for c in range(c0, c1):
            if g[r, c] == C.SOLID:
                y, x = r * px, (c - c0) * px
                img[y:y + px, x:x + px] = dirt
                if r == 0 or g[r - 1, c] != C.SOLID:
                    img[y:y + max(1, px // 3), x:x + px] = grass
    return img


def draw_level(ax, level, c0=0, c1=None):
    c1 = level.width if c1 is None else c1
    ax.imshow(level_image(level, c0, c1), extent=(c0, c1, level.height, 0), interpolation="nearest")
    rows, cols = np.nonzero(level.grid[:, c0:c1] == C.SPIKE)
    for r, c in zip(rows, cols + c0):
        for k in (0, 0.5):
            ax.add_patch(Polygon([(c + k, r + 1), (c + k + 0.25, r + 0.5), (c + k + 0.5, r + 1)],
                                 closed=True, fc=SPIKE, ec="#5f6470", lw=0.4))
    if c0 <= level.goal_x <= c1:
        top = level.surface_row(int(level.goal_x)) or level.height
        ax.plot([level.goal_x, level.goal_x], [top, top - 4], color="#55585f", lw=1.5)
        ax.add_patch(Polygon([(level.goal_x, top - 4), (level.goal_x + 1.3, top - 3.5), (level.goal_x, top - 3)],
                             closed=True, fc=FLAG, ec="none"))
    ax.set_xlim(c0, c1)
    ax.set_ylim(level.height, 0)
    ax.set_aspect("equal")
    ax.axis("off")


# ---------------------------------------------------------------- illustrations
def fig_level_gallery(out):
    seed = C.EVAL_SEED_START + 3
    diffs = [0.0, 0.25, 0.5, 0.75, 1.0]
    fig, axes = plt.subplots(len(diffs), 1, figsize=(13, 11.5))
    for ax, d in zip(axes, diffs):
        lv = generate_level(seed, d)
        draw_level(ax, lv, 0, min(lv.width, 110))
        ax.set_anchor("W")
        chunks = sorted({c[0] for c in lv.chunks} - {"start", "finish"})
        ax.set_title(f"difficulty {d:g}   (chunks: {', '.join(chunks)})", fontsize=11, color=INK2,
                     fontweight="normal", pad=4)
    fig.suptitle("One seed, five difficulties: the generator widens gaps and unlocks spikes and platforms",
                 x=0.01, ha="left", fontsize=14, fontweight="semibold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, out, "01_level_gallery.png", "Procedural levels across difficulty",
         f"Level seed {seed} generated at five difficulty settings (first 110 columns shown). "
         "Difficulty widens gaps and height changes, and unlocks the stage 2 hazards "
         "(spikes from 0.25, floating platforms from 0.35).")


def fig_level_variety(out):
    fig, axes = plt.subplots(4, 1, figsize=(13, 9.2))
    for i, ax in enumerate(axes):
        seed = C.DEMO_SEED_START + 11 * i + 5
        lv = generate_level(seed, 0.6)
        draw_level(ax, lv, 0, min(lv.width, 110))
        ax.set_anchor("W")
        ax.set_title(f"seed {seed}", fontsize=11, color=INK2, fontweight="normal", pad=4)
    fig.suptitle("Every level is new: four random seeds at difficulty 0.6",
                 x=0.01, ha="left", fontsize=14, fontweight="semibold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, out, "02_level_variety.png", "Every level is new",
         "Four different seeds at the same difficulty. Training draws a fresh seed from a pool "
         "of one million for every episode; evaluation and demo seeds come from disjoint ranges.")


def fig_observation(out):
    from procgen_rl.solver import solve
    seed, d = C.EVAL_SEED_START + 7, 0.8
    lv = generate_level(seed, d)
    path = solve(lv) or []
    r = Runner(lv)
    spike_layer = slice(C.VIEW_H * C.VIEW_W, 2 * C.VIEW_H * C.VIEW_W)
    for a in path:
        r.act(a)
        if r.state.x > 20 and r.state.on_ground and r.observe()[spike_layer].sum() >= 2:
            break
    s = r.state
    obs = r.observe()
    grid = obs[:C.OBS_CHANNELS * C.VIEW_H * C.VIEW_W].reshape(C.OBS_CHANNELS, C.VIEW_H, C.VIEW_W)
    col, row = math.floor(s.cx), math.floor(s.cy)

    fig = plt.figure(figsize=(13, 6.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], hspace=0.35, wspace=0.12)
    ax = fig.add_subplot(gs[0, :])
    c0 = max(0, col - 12)
    draw_level(ax, lv, c0, min(lv.width, col + 30))
    ax.add_patch(Rectangle((col - C.VIEW_BEHIND, row - C.VIEW_UP), C.VIEW_W, C.VIEW_H,
                           fill=False, ec=SERIES[0], lw=2.2))
    ax.add_patch(Rectangle((s.x, s.y), C.PLAYER_W, C.PLAYER_H, fc=FLAG, ec="white", lw=1.2, zorder=4))
    ax.annotate("agent", (s.x + 0.4, s.y), xytext=(0, 14), textcoords="offset points", ha="center",
                color=INK, fontsize=10)
    ax.annotate(f"what the agent sees: {C.VIEW_W} x {C.VIEW_H} tiles\n"
                f"({C.VIEW_BEHIND} behind, {C.VIEW_AHEAD} ahead, {C.VIEW_UP} up, {C.VIEW_DOWN} down)",
                (col - C.VIEW_BEHIND + C.VIEW_W, row - C.VIEW_UP), xytext=(8, -4),
                textcoords="offset points", va="top", color=SERIES[0], fontsize=10)
    ax.set_title("The observation is an egocentric tile window, not pixels", fontsize=13)

    cmap = matplotlib.colors.ListedColormap([SURFACE, "#2a78d6"])
    names = ["solid tiles (+ walls)", "spike tiles", "goal region"]
    for i in range(3):
        a = fig.add_subplot(gs[1, i])
        a.imshow(grid[i], cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        a.add_patch(Rectangle((C.VIEW_BEHIND - 0.5, C.VIEW_UP - 0.5), 1, 1, fill=False, ec=FLAG, lw=1.6))
        a.set_xticks([])
        a.set_yticks([])
        a.grid(False)
        for sp in a.spines.values():
            sp.set_visible(True)
            sp.set_color(AXIS)
        a.set_title(f"layer {i + 1}: {names[i]}", fontsize=11, fontweight="normal", color=INK2)
    sc = obs[-C.OBS_SCALARS:]
    fig.text(0.01, -0.02,
             f"+ 7 scalars: vx {sc[0]:+.2f}, vy {sc[1]:+.2f}, on_ground {sc[2]:.0f}, can_jump {sc[3]:.0f}, "
             f"jumping {sc[4]:.0f}, frac_x {sc[5]:.2f}, frac_y {sc[6]:.2f}      "
             f"total input: 3 x {C.VIEW_H} x {C.VIEW_W} + 7 = {C.OBS_DIM} numbers (red square = agent's tile)",
             fontsize=10.5, color=INK2)
    save(fig, out, "03_observation.png", "What the agent sees",
         f"The agent observes a {C.VIEW_W}x{C.VIEW_H} tile window centred on itself, split into three binary "
         "layers (solid, spike, goal), plus 7 numbers describing its own motion. This is far cheaper "
         "than pixels and generalises across levels because it is relative to the agent.")


def _box(ax, x, y, w, h, text, fc, ec, color=INK, size=10.5, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=color,
            fontweight=weight, linespacing=1.4)


def _arrow(ax, p0, p1, color=MUTED):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14, color=color, lw=1.4))


def fig_network(out):
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    light, blue = "#eaf2fd", "#86b6ef"
    _box(ax, 0.2, 1.6, 2.3, 1.4, f"Observation\n{C.OBS_DIM} numbers\n(840 tile bits + 7 scalars)", light, blue)
    for i, (y, name) in enumerate(((2.75, "Actor (policy)"), (0.35, "Critic (value)"))):
        _box(ax, 3.3, y, 1.9, 1.3, "Linear 256\nReLU", light, blue)
        _box(ax, 5.9, y, 1.9, 1.3, "Linear 256\nReLU", light, blue)
        ax.text(3.3, y + 1.42, name, fontsize=11, color=INK, fontweight="semibold")
        _arrow(ax, (2.5, 2.3), (3.3, y + 0.65))
        _arrow(ax, (5.2, y + 0.65), (5.9, y + 0.65))
    _box(ax, 8.6, 2.75, 2.0, 1.3, "6 action logits\n(softmax)", "#fdeee6", "#eb6834")
    _box(ax, 8.6, 0.35, 2.0, 1.3, "state value V(s)", "#fdeee6", "#eb6834")
    _arrow(ax, (7.8, 3.4), (8.6, 3.4))
    _arrow(ax, (7.8, 1.0), (8.6, 1.0))
    ax.text(10.9, 3.4, "idle, left, right,\njump, left+jump,\nright+jump\n(held for 4 frames)",
            va="center", fontsize=10, color=INK2)
    ax.text(10.9, 1.0, "used only during\ntraining (PPO\nadvantage estimates)", va="center", fontsize=10, color=INK2)
    ax.set_title("PPO network: two separate 256-256 MLPs (Stable-Baselines3 MlpPolicy)", fontsize=13)
    save(fig, out, "04_network.png", "Policy network",
         "Separate actor and critic MLPs with two hidden layers of 256 ReLU units each. Only the actor is "
         "exported into ghost files; the demo runs it with NumPy, about 0.1 ms per decision.")


def fig_pipeline(out):
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    light, blue, warm, warm_ec = "#eaf2fd", "#86b6ef", "#fdeee6", "#eb6834"
    boxes = [
        (0.1, "Level generator\nseed + difficulty\n-> chunks -> level"),
        (2.7, "Gymnasium env\nphysics @ 60 fps\nobs, reward, done"),
        (5.3, "PPO training\n8 parallel envs\ncurriculum 0.1 -> 1.0"),
        (7.9, "Ghost snapshots\nat 0min, 30min, 1h,\n2h, 5h ... 300h"),
    ]
    for x, t in boxes:
        _box(ax, x, 2.2, 2.2, 1.5, t, light, blue)
    for x0, x1 in ((2.3, 2.7), (4.9, 5.3), (7.5, 7.9)):
        _arrow(ax, (x0, 2.95), (x1, 2.95))
    _box(ax, 10.6, 3.0, 2.3, 1.0, "Evaluation\nunseen levels", warm, warm_ec)
    _box(ax, 10.6, 1.6, 2.3, 1.0, "Demo\nrace vs ghosts", warm, warm_ec)
    _arrow(ax, (10.1, 3.1), (10.6, 3.5))
    _arrow(ax, (10.1, 2.7), (10.6, 2.1))
    ax.text(0.1, 1.3, "Shared code path: physics.step_player + sim.Runner are used by training, evaluation and the demo,\n"
                      "so a ghost behaves exactly as it did during training. Levels are deterministic per seed on every machine.",
            fontsize=10.5, color=INK2, va="top")
    ax.text(0.1, 0.35, "Runs in the web app on the laptop (or the Colab notebook). "
                       "Autosaves every 10 real minutes and every simulated hour; resumes after interruption.",
            fontsize=10.5, color=INK2, va="top")
    ax.set_title("System pipeline", fontsize=13)
    save(fig, out, "05_pipeline.png", "System pipeline",
         "From seeded level generation through PPO training to milestone ghosts, which feed both the "
         "quantitative evaluation and the interactive demo.")


# ---------------------------------------------------------------- training curves (main run)
def fig_learning_curve(out, run, metrics, state):
    x, succ = series(metrics, "success_rate")
    _, prog = series(metrics, "progress")
    if len(x) < 3:
        return
    k = max(1, len(x) // 150)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, smooth(prog, k), color=SERIES[0], label="mean progress through level")
    ax.plot(x, smooth(succ, k), color=SERIES[1], label="levels completed")
    for m in (state or {}).get("milestones_done", []):
        if m > 0:
            ax.axvline(m, color=GRID, lw=1, zorder=0)
    pct_axis(ax)
    hours_axis(ax)
    end_labels(ax, [(x[-1], smooth(prog, k)[-1], f"progress {smooth(prog, k)[-1]:.0%}", SERIES[0]),
                    (x[-1], smooth(succ, k)[-1], f"completed {smooth(succ, k)[-1]:.0%}", SERIES[1])])
    ax.legend(loc="upper left")
    ax.set_title("Training episodes: share of levels completed and distance covered "
                 "(last 200 episodes, at the current curriculum difficulty)")
    save(fig, out, "07_learning_curve.png", "Learning curve",
         "Rolling success and progress over the most recent 200 training episodes. Dips are expected: "
         "every time the agent masters a difficulty the curriculum makes levels harder "
         "(see the curriculum figure). Faint vertical lines mark ghost snapshots.")


def fig_curriculum(out, metrics):
    x, d = series(metrics, "difficulty")
    if len(x) < 3:
        return
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.step(x, d, where="post", color=SERIES[0])
    ax.fill_between(x, d, step="post", color=SERIES[0], alpha=0.08)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("difficulty ceiling")
    hours_axis(ax)
    ups = [(x[i], d[i]) for i in range(1, len(d)) if d[i] > d[i - 1]]
    for xi, di in ups[::max(1, len(ups) // 6)] + ups[-1:]:
        ax.annotate(f"{di:.1f} at {xi:.1f}h", (xi, di), xytext=(4, 6), textcoords="offset points",
                    fontsize=9, color=INK2)
    ax.set_title("Automatic curriculum: difficulty rises by 0.1 whenever 70% of recent levels at the ceiling are completed")
    save(fig, out, "08_curriculum.png", "Curriculum",
         "The difficulty ceiling over simulated training time. The agent starts on easy levels and is "
         "promoted when it completes 70% of the last 100 levels played at the ceiling.")


def fig_diagnostics(out, metrics):
    panels = [("episode_reward", "Episode reward"), ("steps_per_sec", "Training speed (agent steps / s)"),
              ("entropy_loss", "Entropy loss (more negative = more random)"), ("value_loss", "Value loss"),
              ("approx_kl", "Approx. KL per update"), ("explained_variance", "Explained variance of critic")]
    avail = [(k, t) for k, t in panels if len(series(metrics, k)[0]) > 3]
    if not avail:
        return
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, (k, t) in zip(axes.flat, panels):
        x, y = series(metrics, k)
        if len(x) > 3:
            ax.plot(x, smooth(y, max(1, len(x) // 100)), color=SERIES[0], lw=1.6)
        ax.set_title(t, fontsize=11)
        hours_axis(ax)
        ax.set_xlabel("sim hours", fontsize=9)
    fig.suptitle("PPO training diagnostics", x=0.01, ha="left", fontsize=14, fontweight="semibold")
    fig.tight_layout()
    save(fig, out, "12_ppo_diagnostics.png", "PPO diagnostics",
         "Health checks for the optimisation: reward should rise, entropy should fall slowly (less "
         "random actions), KL stays small (stable updates). Explained variance measures how well the critic "
         "predicts returns (1 = perfect); in this project it stays low, so the critic is the weakest part.")


# ---------------------------------------------------------------- evaluation (main run)
def fig_eval_unseen(out, ev):
    rows = [r for r in ev if r["eval_set"] == "unseen"]
    diffs = sorted({r["difficulty"] for r in rows})
    if not rows or len({r["sim_hours"] for r in rows}) < 2:
        return
    cols = ORDINAL[min(5, len(diffs))]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for key, ax, title in (("success_rate", axes[0], "Levels completed"),
                           ("mean_progress", axes[1], "Mean progress through level")):
        ends = []
        for d, c in zip(diffs, cols):
            sel = sorted((r for r in rows if r["difficulty"] == d), key=lambda r: r["sim_hours"])
            xs = [r["sim_hours"] for r in sel]
            ys = [r[key] for r in sel]
            ax.plot(xs, ys, color=c, marker="o", ms=5, mec=SURFACE, mew=1.5, label=f"difficulty {d:g}")
            ends.append((xs[-1], ys[-1], f"d={d:g}  {ys[-1]:.0%}", c))
        pct_axis(ax)
        hours_axis(ax)
        end_labels(ax, ends, dot=False)
        ax.set_title(title)
    axes[0].legend(loc="upper left")
    n = rows[0]["levels"]
    fig.suptitle(f"Every ghost tested on {n} never-seen levels per difficulty",
                 x=0.01, ha="left", fontsize=14, fontweight="semibold")
    fig.tight_layout()
    save(fig, out, "09_eval_unseen.png", "Performance on unseen levels",
         "Each point is one ghost snapshot played deterministically on held-out levels (seeds never used "
         "in training). This is the honest measure of generalisation, unlike the training curve.")


def fig_eval_heatmap(out, ev):
    rows = [r for r in ev if r["eval_set"] == "unseen"]
    if not rows:
        return
    labels = []
    for r in sorted(rows, key=lambda r: r["sim_hours"]):
        if r["label"] not in labels:
            labels.append(r["label"])
    diffs = sorted({r["difficulty"] for r in rows})
    m = np.full((len(labels), len(diffs)), np.nan)
    for r in rows:
        m[labels.index(r["label"]), diffs.index(r["difficulty"])] = r["success_rate"]
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(labels) + 1.8))
    im = ax.imshow(m, cmap=SEQ_CMAP, vmin=0, vmax=1, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(len(diffs)), [f"{d:g}" for d in diffs])
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("level difficulty")
    ax.set_ylabel("ghost (simulated training time)")
    for i in range(len(labels)):
        for j in range(len(diffs)):
            if not np.isnan(m[i, j]):
                ax.text(j, i, f"{m[i, j]:.0%}", ha="center", va="center", fontsize=9.5,
                        color="white" if m[i, j] > 0.55 else INK)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03, format=matplotlib.ticker.PercentFormatter(1.0))
    cb.outline.set_visible(False)
    ax.set_title("Levels completed on unseen levels")
    save(fig, out, "10_eval_heatmap.png", "Success matrix",
         "Success rate of every ghost (rows) at every difficulty (columns) on held-out levels.")


def fig_finish_time(out, ev):
    rows = [r for r in ev if r["eval_set"] == "unseen" and r["mean_finish_time"] and r["success_rate"] >= 0.1]
    diffs = sorted({r["difficulty"] for r in rows})
    if len({r["sim_hours"] for r in rows}) < 2:
        return
    cols = ORDINAL[min(5, len(diffs))]
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ends = []
    for d, c in zip(diffs, cols):
        sel = sorted((r for r in rows if r["difficulty"] == d), key=lambda r: r["sim_hours"])
        if len(sel) < 2:
            continue
        xs = [r["sim_hours"] for r in sel]
        ys = [r["mean_finish_time"] for r in sel]
        ax.plot(xs, ys, color=c, marker="o", ms=5, mec=SURFACE, mew=1.5, label=f"difficulty {d:g}")
        ends.append((xs[-1], ys[-1], f"d={d:g}", c))
    ax.set_ylabel("seconds to finish (completed levels)")
    ax.set_ylim(bottom=0)
    hours_axis(ax)
    end_labels(ax, ends, dot=False)
    ax.legend(loc="lower left")
    ax.set_title("Completed runs take about 17 s at every stage: training improves survival, not speed")
    save(fig, out, "14_finish_time.png", "Finish time",
         "Mean time to reach the flag on the unseen levels each ghost completed "
         "(ghosts completing fewer than 10% are omitted).")


def fig_trajectories(out, run):
    ghosts, _ = load_ghosts(Path(run) / "ghosts")
    if len(ghosts) < 2:
        return []
    chosen = pick_ghosts(ghosts, 5)
    from evaluate import run_episode
    best = None
    for i in range(30):  # find an unseen level the best ghost completes but early ones do not
        lv = generate_level(C.EVAL_SEED_START + 500 + i, 0.6)
        res = [run_episode(g, lv, record=True) for g in chosen]
        score = res[-1][0] * 10 + len({round(r[1], 1) for r in res})
        if best is None or score > best[0]:
            best = (score, lv, res)
        if res[-1][0] and score >= 14:
            break
    _, lv, res = best
    cols = ORDINAL[len(chosen)]
    fig, ax = plt.subplots(figsize=(16, 3.8))
    draw_level(ax, lv)
    n_done = 0
    for g, (ok, prog, secs, traj), c in zip(chosen, res, cols):
        t = np.array(traj)
        ax.plot(t[:, 0], t[:, 1], color=c, lw=1.8)
        # a ghost that fell into a pit ends below the frame: pin its marker and label to the bottom edge
        end = (t[-1, 0], min(t[-1, 1], lv.height - 0.4))
        ax.plot(*end, marker="o" if ok else "X", ms=8, color=c, mec="white", mew=1.2, zorder=5, clip_on=False)
        if ok:
            text, xy, off, ha, va = f"{g.label}: finished {secs:.1f}s", (lv.goal_x, 0.5), (-10, -12 * n_done), "right", "top"
            n_done += 1
        elif t[-1, 1] > lv.height - 0.5:
            text, xy, off, ha, va = f"{g.label}: fell at {prog:.0%}", end, (6, 4), "left", "bottom"
        else:
            text, xy, off, ha, va = f"{g.label}: stopped at {prog:.0%}", end, (4, -12), "left", "top"
        ax.annotate(text, xy, xytext=off, textcoords="offset points", ha=ha, va=va, annotation_clip=False,
                    fontsize=8.5, color=INK, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
                    zorder=6)
    handles = [plt.Line2D([], [], color=c, lw=2.5, label=g.label) for g, c in zip(chosen, cols)]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=len(chosen),
              title="ghost (darker = more training)", title_fontsize=9.5)
    ax.set_title(f"The same unseen level (seed {lv.seed}, difficulty 0.6) played by ghosts from different training stages")
    save(fig, out, "11_ghost_trajectories.png", "Ghost trajectories",
         "Paths of five ghost snapshots through one held-out level. X marks where a ghost died or got "
         "stuck; a dot marks reaching the flag. This is the static version of the demo's ghost race.")
    return [g.label for g in chosen]


def _demo_shot(run, shot, seed, difficulty, frames):
    env = dict(os.environ, SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy")
    cmd = [sys.executable, str(Path(__file__).with_name("demo.py")), "--ghosts", str(Path(run) / "ghosts"),
           "--spectate", "--seed", str(seed), "--difficulty", str(difficulty),
           "--frames", str(frames), "--screenshot", str(shot)]
    try:
        subprocess.run(cmd, env=env, check=True, capture_output=True, timeout=180)
        return True
    except Exception as e:
        print(f"  demo screenshot failed: {e}")
        return False


def fig_demo_screenshot(out, run, labels):
    if not labels:
        return
    # 3 s into the race: the pack is still together, so most name tags and overtaking arrows show
    if _demo_shot(run, out / "13_demo_race.png", C.DEMO_SEED_START + 42, 0.5, 300):
        CAPTIONS.append({"file": "13_demo_race.png", "title": "The demo: racing the ghosts",
                         "caption": "Screenshot of the pygame demo in spectate mode: every ghost races the same fresh "
                                    "level, with name tags, a compact leaderboard (arrows mark overtakes) and a minimap."})
        print("  wrote 13_demo_race.png")
    if _demo_shot(run, out / "17_expert_demo.png", C.DEMO_SEED_START + 7, 1.5, 560):
        CAPTIONS.append({"file": "17_expert_demo.png", "title": "Expert level in the demo",
                         "caption": "Difficulty 1.5, beyond anything seen in training: a one-tile pillar, wider gaps and "
                                    "repeated spike runs. Half of the ghosts are already out after 7 seconds."})
        print("  wrote 17_expert_demo.png")


def fig_expert(out, ev, ev_expert):
    """Completion across the training range and the expert range, for a few ghosts."""
    rows = [r for r in ev if r["eval_set"] == "unseen"] + [r for r in ev_expert if r["difficulty"] > 1.0]
    if not ev_expert:
        return
    hours = sorted({r["sim_hours"] for r in rows})
    chosen = [h for h in (10, 30, 300) if h in hours] or hours[-3:]
    cols = ORDINAL[len(chosen)]
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.axvspan(1.0, 2.05, color="#efeee9", zorder=0)
    ax.text(1.03, 0.98, "expert: beyond training", transform=ax.get_xaxis_transform(), va="top",
            fontsize=10, color=INK2)
    ends = []
    for h, c in zip(chosen, cols):
        sel = sorted((r for r in rows if r["sim_hours"] == h), key=lambda r: r["difficulty"])
        xs = [r["difficulty"] for r in sel]
        ys = [r["success_rate"] for r in sel]
        label = sel[0]["label"]
        ax.plot(xs, ys, color=c, marker="o", ms=5, mec=SURFACE, mew=1.5, label=f"{label} ghost", zorder=3)
        ends.append((xs[-1], ys[-1], f"{label}  {ys[-1]:.0%}", c))
    pct_axis(ax)
    ax.set_xlim(0.2, 2.05)
    ax.set_xticks([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
    ax.set_xlabel("level difficulty")
    end_labels(ax, ends, dot=False)
    ax.legend(loc="lower left")
    ax.set_title("Levels completed on never-seen levels, through the training range and beyond it")
    save(fig, out, "18_expert_levels.png", "Expert difficulty",
         "Difficulty above 1 uses the same tiles and physics, pushed towards what the jump can clear. "
         "No ghost was trained there: even the 300h ghost drops from 85% at 1.0 to under 10% at 2.0, so "
         "a skilled human can beat it.")


# ---------------------------------------------------------------- generalisation (cross-run)
def gen_runs(root):
    runs = []
    for d in sorted(Path(root).glob("gen_*")):
        st = read_json(d / "state.json", {})
        ev = read_eval(d)
        if not ev:
            continue
        n = int(st.get("train_levels") or 0)
        runs.append({"name": d.name, "train_levels": n, "eval": ev,
                     "fixed_d": st.get("fixed_difficulty") or 0.5,
                     "label": f"{n:,} training levels" if n else "unlimited levels"})
    runs.sort(key=lambda r: r["train_levels"] or 10**12)
    return runs


def fig_generalization(out, runs):
    if not runs:
        return {}
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.36
    summary = {}
    for i, r in enumerate(runs):
        last_h = max(e["sim_hours"] for e in r["eval"])
        d = r["fixed_d"]
        unseen = next((e["success_rate"] for e in r["eval"] if e["sim_hours"] == last_h
                       and e["eval_set"] == "unseen" and abs(e["difficulty"] - d) < 1e-9), None)
        train = next((e["success_rate"] for e in r["eval"] if e["sim_hours"] == last_h
                      and e["eval_set"] == "train"), None)
        if train is None:
            train = unseen  # unlimited levels: every training level is effectively unseen
        summary[r["label"]] = {"train": train, "unseen": unseen, "sim_hours": last_h}
        for off, v, c in ((-width / 2, train, SERIES[0]), (width / 2, unseen, SERIES[1])):
            if v is None:
                continue
            ax.bar(i + off, v, width - 0.04, color=c, edgecolor=SURFACE, linewidth=2)
            ax.text(i + off, v + 0.015, f"{v:.0%}", ha="center", va="bottom", fontsize=10, color=INK2)
    ax.set_xticks(range(len(runs)), [r["label"] for r in runs])
    pct_axis(ax)
    ax.grid(axis="x", visible=False)
    handles = [matplotlib.patches.Patch(color=SERIES[0], label="levels it trained on"),
               matplotlib.patches.Patch(color=SERIES[1], label="never-seen levels")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
    ax.set_title("Generalisation: completion rate on training levels vs never-seen levels "
                 f"(difficulty {runs[0]['fixed_d']:g}, final ghost of each run)")
    save(fig, out, "15_generalization.png", "Generalisation experiment",
         "Agents trained on a small fixed set of levels memorise them and fail on new ones; the gap "
         "closes as the number of distinct training levels grows. This is why procedural generation "
         "matters (cf. Cobbe et al., 2019, Procgen benchmark).")
    return summary


def fig_generalization_curve(out, runs):
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ends = []
    for r, c in zip(runs, SERIES):
        d = r["fixed_d"]
        sel = sorted((e for e in r["eval"] if e["eval_set"] == "unseen" and abs(e["difficulty"] - d) < 1e-9),
                     key=lambda e: e["sim_hours"])
        if len(sel) < 2:
            continue
        xs = [e["sim_hours"] for e in sel]
        ys = [e["success_rate"] for e in sel]
        ax.plot(xs, ys, color=c, marker="o", ms=5, mec=SURFACE, mew=1.5, label=r["label"])
        ends.append((xs[-1], ys[-1], r["label"], c))
    pct_axis(ax)
    hours_axis(ax)
    end_labels(ax, ends, dot=False)
    ax.legend(loc="upper left")
    ax.set_title("Completion rate on never-seen levels during training, by number of distinct training levels")
    save(fig, out, "16_generalization_curve.png", "Generalisation over training time",
         "Held-out success rate of each generalisation run's ghosts over training time.")


# ---------------------------------------------------------------- summary
def write_summary(out, main, metrics, ev, state, gen_summary):
    lines = ["# ProcGen RL - results summary", "", f"Generated {time.strftime('%Y-%m-%d %H:%M')}.", ""]
    if metrics:
        last = metrics[-1]
        wall_h = (metrics[-1]["wall"] - metrics[0]["wall"]) / 3600 if len(metrics) > 1 else 0
        lines += ["## Main run", "",
                  f"- Simulated training time: {last['sim_hours']:.1f} h ({last['timesteps']:,} agent steps)",
                  f"- Real time (approx.): {wall_h:.1f} h, about {last['steps_per_sec']:,.0f} agent steps/s",
                  f"- Final curriculum difficulty: {last['difficulty']:.1f}",
                  f"- Ghost snapshots: {len((state or {}).get('milestones_done', []))}", ""]
    unseen = [r for r in ev if r["eval_set"] == "unseen"]
    if unseen:
        last_h = max(r["sim_hours"] for r in unseen)
        lines += [f"## Final ghost on unseen levels ({unseen[0]['levels']} levels per difficulty)", "",
                  "| difficulty | completed | mean progress | mean finish time |", "|---|---|---|---|"]
        for r in sorted((r for r in unseen if r["sim_hours"] == last_h), key=lambda r: r["difficulty"]):
            ft = f"{r['mean_finish_time']:.1f} s" if r["mean_finish_time"] else "-"
            lines.append(f"| {r['difficulty']:g} | {r['success_rate']:.0%} | {r['mean_progress']:.0%} | {ft} |")
        lines.append("")
    if gen_summary:
        lines += ["## Generalisation experiment", "", "| training levels | on training levels | on unseen levels |",
                  "|---|---|---|"]
        for k, v in gen_summary.items():
            tr = f"{v['train']:.0%}" if v["train"] is not None else "-"
            un = f"{v['unseen']:.0%}" if v["unseen"] is not None else "-"
            lines.append(f"| {k} | {tr} | {un} |")
        lines.append("")
    (out / "summary.md").write_text("\n".join(lines))
    print("  wrote summary.md")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--main", default="runs/main", help="main (curriculum) run folder")
    p.add_argument("--runs-root", default="runs", help="folder containing gen_* runs")
    p.add_argument("--out", default="runs/presentation")
    p.add_argument("--skip-illustrations", action="store_true")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("illustrations")
    if not args.skip_illustrations:
        fig_level_gallery(out)
        fig_level_variety(out)
        fig_observation(out)
        fig_network(out)
        fig_pipeline(out)

    main_run = Path(args.main)
    metrics = read_metrics(main_run)
    ev = read_eval(main_run)
    state = read_json(main_run / "state.json", {})
    print(f"main run {main_run}: {len(metrics)} metric rows, {len(ev)} eval rows")
    fig_learning_curve(out, main_run, metrics, state)
    fig_curriculum(out, metrics)
    fig_eval_unseen(out, ev)
    fig_eval_heatmap(out, ev)
    labels = fig_trajectories(out, main_run) if (main_run / "ghosts").exists() else []
    fig_diagnostics(out, metrics)
    fig_demo_screenshot(out, main_run, labels)
    fig_finish_time(out, ev)
    fig_expert(out, ev, read_eval(main_run, "eval_expert.csv"))

    runs = gen_runs(args.runs_root)
    print(f"generalisation runs: {[r['name'] for r in runs]}")
    gen_summary = fig_generalization(out, runs)
    fig_generalization_curve(out, runs)
    write_summary(out, main_run, metrics, ev, state, gen_summary)

    CAPTIONS.sort(key=lambda c: c["file"])
    (out / "captions.json").write_text(json.dumps(CAPTIONS, indent=2))
    print(f"done: {len(CAPTIONS)} figures in {out}")


if __name__ == "__main__":
    main()
