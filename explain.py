"""Explainability figures: what the trained agent looks at and what it relies on.

    python explain.py                               # 300h ghost from ./ghosts -> runs/presentation
    python explain.py --ghosts runs/main/ghosts --levels 100

Writes 20_* to 26_* PNGs into --out, merges their captions into captions.json and
writes explain.json with the numbers quoted in the slides:
- 20 the observation of one real moment, as the numbers the network receives
- 21 where the agent looks: average sensitivity of each tile in the view window
- 22 sensitivity by input group (flip-one-input test, NumpyPolicy.sensitivity)
- 23 blindfold test: success on unseen levels with parts of the input removed
- 24 which actions each ghost uses
- 25 the critic's expectation during one level, for a ghost that dies and one that finishes
- 26 what single first-layer neurons respond to (their input weights)
Only numpy + matplotlib; all levels come from the held-out evaluation seed range.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import figures as F
from figures import INK, INK2, MUTED, SERIES, plt
from procgen_rl import config as C
from procgen_rl.level import generate_level
from procgen_rl.policy import N_TILE_INPUTS, load_ghosts
from procgen_rl.sim import FINISHED, RUNNING, Runner

N_TILES = C.VIEW_H * C.VIEW_W
GROUPS = (
    ("ground + walls (solid layer)", slice(0, N_TILES)),
    ("spikes (spike layer)", slice(N_TILES, 2 * N_TILES)),
    ("finish line (goal layer)", slice(2 * N_TILES, 3 * N_TILES)),
    ("own speed (vx, vy)", slice(N_TILE_INPUTS, N_TILE_INPUTS + 2)),
    ("jump state (3 flags)", slice(N_TILE_INPUTS + 2, N_TILE_INPUTS + 5)),
    ("position in tile", slice(N_TILE_INPUTS + 5, N_TILE_INPUTS + 7)),
)
EXAMPLE = (1_000_500, 0.6)      # the level of the ghost-trajectory figure
EXAMPLE_STEP = 76               # a moment just before a gap


def play(policy, level, mask=None, on_decision=None):
    """One deterministic episode with the training cut-offs. Returns (success, progress, steps)."""
    r = Runner(level)
    steps = stall = 0
    best = r.max_x
    while r.status == RUNNING and steps < C.MAX_EPISODE_STEPS and stall < C.STALL_STEPS:
        obs = r.observe()
        if mask is not None:
            obs = mask(obs)
        if on_decision is not None:
            on_decision(r, obs, steps)
        r.act(policy.act(obs))
        steps += 1
        if r.max_x > best + 1e-6:
            best, stall = r.max_x, 0
        else:
            stall += 1
    return r.status == FINISHED, r.progress, steps


def eval_levels(n, difficulty):
    return [generate_level(C.EVAL_SEED_START + i, difficulty) for i in range(n)]


# ---------------------------------------------------------------- masks for the blindfold test
def _zero(sl):
    def f(obs):
        obs = obs.copy()
        obs[sl] = 0.0
        return obs
    return f


def _sight(cols_ahead):
    """Tiles further than cols_ahead columns ahead of the agent read as empty air."""
    keep = C.VIEW_BEHIND + 1 + cols_ahead

    def f(obs):
        obs = obs.copy()
        tiles = obs[:N_TILE_INPUTS].reshape(3, C.VIEW_H, C.VIEW_W)
        tiles[:, :, keep:] = 0.0
        return obs
    return f


BLINDFOLDS = (
    ("full input (normal)", None),
    ("no position in tile", _zero(GROUPS[5][1])),
    ("no jump-state flags", _zero(GROUPS[4][1])),
    ("no own speed", _zero(GROUPS[3][1])),
    ("cannot see the finish line", _zero(GROUPS[2][1])),
    ("cannot see spikes", _zero(GROUPS[1][1])),
    ("sees only 8 tiles ahead", _sight(8)),
    ("sees only 4 tiles ahead", _sight(4)),
    ("sees only 2 tiles ahead", _sight(2)),
)


# ---------------------------------------------------------------- figures
def fig_observation(policy, out, numbers):
    seed, d = EXAMPLE
    level = generate_level(seed, d)
    r = Runner(level)
    for _ in range(EXAMPLE_STEP):
        r.act(policy.act(r.observe()))
    obs = r.observe()
    hidden, logits, probs = policy.trace(obs)
    numbers["example"] = {
        "seed": seed, "difficulty": d, "step": EXAMPLE_STEP,
        "scalars": [round(float(v), 3) for v in obs[N_TILE_INPUTS:]],
        "ones": int((obs[:N_TILE_INPUTS] > 0.5).sum()),
        "active": [int((h > 0).sum()) for h in hidden],
        "logits": [round(float(v), 2) for v in logits],
        "probs": [round(float(v), 4) for v in probs],
        "action": C.ACTION_NAMES[int(np.argmax(probs))],
        "value": round(policy.value(obs), 2) if policy.critic is not None else None,
    }

    c0, r0 = r.obs_builder.window_origin(r.state)
    fig = plt.figure(figsize=(15, 8.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.05], width_ratios=[1, 1, 1, 0.78],
                          hspace=0.28, wspace=0.12)
    ax = fig.add_subplot(gs[0, :])
    x0, x1 = max(0, c0 - 6), min(level.width, c0 + C.VIEW_W + 10)
    img = np.ones((level.height, x1 - x0, 3)) * matplotlib_rgb(F.SKY)
    for rr in range(level.height):
        for cc in range(x0, x1):
            t = level.grid[rr, cc]
            if t == C.SOLID:
                above = rr > 0 and level.grid[rr - 1, cc] == C.SOLID
                img[rr, cc - x0] = matplotlib_rgb(F.DIRT if above else F.GRASS)
            elif t == C.SPIKE:
                img[rr, cc - x0] = matplotlib_rgb(F.SPIKE)
    ax.imshow(img, extent=(x0, x1, level.height, 0), interpolation="nearest")
    s = r.state
    ax.add_patch(F.Rectangle((s.x, s.y), C.PLAYER_W, C.PLAYER_H, color=F.FLAG, zorder=4))
    ax.add_patch(F.Rectangle((c0, r0), C.VIEW_W, C.VIEW_H, fill=False, ec=INK, lw=2.2, zorder=5))
    ax.add_patch(F.Rectangle((c0 + C.VIEW_BEHIND, r0 + C.VIEW_UP), 1, 1, fill=False, ec=SERIES[0], lw=2, zorder=5))
    ax.text(c0 + 0.2, r0 - 0.35, "observation window: 20 columns x 14 rows around the agent "
            "(4 behind, 15 ahead, 6 up, 7 down)", fontsize=11, color=INK)
    ax.set_xlim(x0, x1)
    ax.set_ylim(level.height, max(0, r0 - 1.2))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title(f"One real moment: seed {seed}, difficulty {d}, decision {EXAMPLE_STEP}", fontsize=13)

    tiles = obs[:N_TILE_INPUTS].reshape(3, C.VIEW_H, C.VIEW_W)
    names = ("layer 1: solid  (inputs 1-280)", "layer 2: spike  (inputs 281-560)", "layer 3: goal  (inputs 561-840)")
    for k in range(3):
        a = fig.add_subplot(gs[1, k])
        a.imshow(tiles[k], cmap="Greys", vmin=0, vmax=1.6, interpolation="nearest")
        for rr in range(C.VIEW_H):
            for cc in range(C.VIEW_W):
                v = int(tiles[k, rr, cc] > 0.5)
                a.text(cc, rr, str(v), ha="center", va="center", fontsize=6.5,
                       color="white" if v else "#9a9a9a", family="monospace")
        a.add_patch(F.Rectangle((C.VIEW_BEHIND - 0.5, C.VIEW_UP - 0.5), 1, 1, fill=False, ec=SERIES[0], lw=2))
        a.set_title(names[k], fontsize=11, fontweight="normal", color=INK2)
        a.set_xticks([])
        a.set_yticks([])
        a.grid(False)
    a = fig.add_subplot(gs[1, 3])
    a.axis("off")
    labels = ("vx / run speed", "vy / max fall", "on_ground", "can_jump", "jumping", "x within tile", "y within tile")
    a.text(0, 1.0, "inputs 841-847: about itself", fontsize=11, color=INK2, va="top")
    for i, (lab, v) in enumerate(zip(labels, obs[N_TILE_INPUTS:])):
        a.text(0, 0.88 - i * 0.1, f"{841 + i}  {lab:<15}{float(v):+.2f}", fontsize=10.5,
               family="monospace", color=INK, va="top")
    a.text(0, 0.14, "blue square = the agent's own tile\n1 = yes, 0 = no for every tile",
           fontsize=10, color=INK2, va="top")
    F.save(fig, out, "20_observation_numbers.png", "The input, as numbers",
           "One real moment of the 300h ghost. The 20 x 14 tile window becomes three yes/no layers "
           "(840 numbers), plus 7 numbers about the agent itself: 847 numbers in total.")


def matplotlib_rgb(hex_color):
    import matplotlib.colors as mc
    return np.array(mc.to_rgb(hex_color))


def collect_sensitivity(policy, levels):
    tile_maps = np.zeros((3, C.VIEW_H, C.VIEW_W))
    combined = np.zeros((C.VIEW_H, C.VIEW_W))
    shares = np.zeros(len(GROUPS))
    act_counts = []
    n = 0

    def on_decision(r, obs, step):
        nonlocal n
        sens = policy.sensitivity(obs)
        t = sens[:N_TILE_INPUTS].reshape(3, C.VIEW_H, C.VIEW_W)
        tile_maps[:] += t
        combined[:] += t.max(axis=0)
        total = sens.sum() or 1.0
        shares[:] += [sens[s].sum() / total for _, s in GROUPS]
        act_counts.append([int((h > 0).sum()) for h in policy.trace(obs)[0]])
        n += 1

    for lv in levels:
        play(policy, lv, on_decision=on_decision)
    return tile_maps / n, combined / n, shares / n, np.array(act_counts), n


def fig_where_it_looks(combined, tile_maps, n, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw={"width_ratios": [1, 1, 1.05]})
    for a, m, t in ((axes[0], tile_maps[0], "if this tile's 'solid' value flipped"),
                    (axes[1], tile_maps[1], "if this tile's 'spike' value flipped")):
        im = a.imshow(m, cmap=F.SEQ_CMAP, interpolation="nearest")
        a.add_patch(F.Rectangle((C.VIEW_BEHIND - 0.5, C.VIEW_UP - 0.5), 1, 1, fill=False, ec=F.FLAG, lw=2.2))
        a.set_title(t, fontsize=11.5)
        a.set_xticks([0, C.VIEW_BEHIND, C.VIEW_W - 1], ["4 behind", "agent", "15 ahead"])
        a.set_yticks([0, C.VIEW_UP, C.VIEW_H - 1], ["6 up", "agent", "7 down"])
        a.grid(False)
    ax = axes[2]
    prof = combined.sum(axis=0)
    prof = prof / prof.sum()
    xs = np.arange(C.VIEW_W) - C.VIEW_BEHIND
    ax.bar(xs, prof, color=[F.FLAG if x == 0 else SERIES[0] for x in xs], width=0.8)
    ax.set_xlabel("columns from the agent (negative = behind)")
    ax.yaxis.set_major_formatter(F.matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title("share of sensitivity by column", fontsize=11.5)
    fig.suptitle(f"Where the 300h agent looks: average over {n:,} decisions on unseen levels "
                 "(darker = flipping that tile changes the decision more)",
                 x=0.01, ha="left", fontsize=13.5, fontweight="semibold")
    fig.tight_layout()
    F.save(fig, out, "21_where_it_looks.png", "Where the agent looks",
           "Average sensitivity of the decision to each tile of the view window. The agent attends to "
           "the ground just ahead and below, a few tiles in front of it; far tiles and tiles behind matter little.")
    return prof, xs


def fig_groups(shares_by_ghost, out):
    fig, ax = plt.subplots(figsize=(11, 4.6))
    names = [g[0] for g in GROUPS]
    y = np.arange(len(names))
    h = 0.8 / len(shares_by_ghost)
    ramp = F.ORDINAL[len(shares_by_ghost)]
    for i, (label, sh) in enumerate(shares_by_ghost):
        ax.barh(y + (i - (len(shares_by_ghost) - 1) / 2) * h, sh, height=h, color=ramp[i], label=f"{label} ghost")
        for yy, v in zip(y, sh):
            ax.text(v + 0.006, yy + (i - (len(shares_by_ghost) - 1) / 2) * h, f"{v:.0%}", va="center",
                    fontsize=9.5, color=INK2)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(F.matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("share of the total decision change when each input is flipped, averaged over decisions")
    ax.legend(loc="lower right")
    ax.set_title("What the decision is sensitive to, by input group", fontsize=13)
    fig.tight_layout()
    F.save(fig, out, "22_input_groups.png", "Sensitivity by input group",
           "Each input is flipped (tiles and flags 0 <-> 1; speeds and position set to 0) and the change "
           "in the action probabilities is measured; shares are averaged over all decisions on unseen levels.")


def fig_blindfold(results, out):
    fig, ax = plt.subplots(figsize=(11, 5.0))
    names = [n for n, _, _ in results]
    vals = [v for _, v, _ in results]
    y = np.arange(len(names))
    ax.barh(y, vals, color=[INK if i == 0 else SERIES[0] for i in range(len(vals))], height=0.62)
    for yy, v in zip(y, vals):
        ax.text(v + 0.01, yy, f"{v:.0%}", va="center", fontsize=10.5, color=INK)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.xaxis.set_major_formatter(F.matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("levels completed")
    ax.set_title(f"Blindfold test: 300h ghost on {results[0][2]} unseen levels at difficulty 0.75, "
                 "with part of its input removed", fontsize=13)
    fig.tight_layout()
    F.save(fig, out, "23_blindfold.png", "Blindfold test",
           "The same agent with one part of its input zeroed (removed parts read as 0, i.e. empty air or "
           "standing still). A big drop means the agent really relies on that input.")


def fig_actions(ghosts, levels, out, numbers):
    rows = []
    for g in ghosts:
        counts = np.zeros(len(C.ACTIONS))

        def on_decision(r, obs, step, g=g, counts=counts):
            counts[g.act(obs)] += 1

        for lv in levels:
            play(g, lv, on_decision=on_decision)
        rows.append((g.label, counts / counts.sum()))
    numbers["action_usage"] = {lab: [round(float(v), 4) for v in sh] for lab, sh in rows}
    fig, ax = plt.subplots(figsize=(12, 5.6))
    y = np.arange(len(rows))
    left = np.zeros(len(rows))
    cols = ["#b9b8b0", "#6250d6", SERIES[0], SERIES[2], "#e87ba4", SERIES[1]]
    for a, name in enumerate(C.ACTION_NAMES):
        v = np.array([sh[a] for _, sh in rows])
        ax.barh(y, v, left=left, color=cols[a], label=name, height=0.72)
        for yy, l, vv in zip(y, left, v):
            if vv > 0.07:
                ax.text(l + vv / 2, yy, f"{vv:.0%}", ha="center", va="center", fontsize=8.5, color="white")
        left += v
    ax.set_yticks(y, [lab for lab, _ in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(F.matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.grid(False)
    ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.07))
    ax.set_title(f"Which buttons each ghost presses ({len(levels)} unseen levels, difficulty 0.5)", fontsize=13)
    fig.tight_layout()
    F.save(fig, out, "24_action_usage.png", "Action usage by training stage",
           "Share of decisions spent on each of the 6 actions. Nobody told the agent that 'right' is the "
           "useful direction: it learned to stop using left, idle and plain jump.")


def fig_critic(ghosts, out, numbers):
    seed, d = EXAMPLE
    level = generate_level(seed, d)
    fig, ax = plt.subplots(figsize=(12, 4.4))
    numbers["critic_trace"] = {}
    for g, col in ((ghosts[0], SERIES[1]), (ghosts[1], SERIES[0])):
        ts, vs = [], []
        r = Runner(level)

        def on_decision(rr, obs, step, ts=ts, vs=vs):
            ts.append(rr.frames / C.FPS)
            vs.append(g.value(obs))

        ok, prog, steps = play(g, level, on_decision=on_decision)
        end = "finished" if ok else f"died at {prog:.0%}"
        ax.plot(ts, vs, color=col, label=f"{g.label} ghost ({end})")
        ax.plot(ts[-1], vs[-1], marker="o" if ok else "X", ms=11, color=col)
        numbers["critic_trace"][g.label] = {"result": end, "first": round(vs[0], 2), "last": round(vs[-1], 2),
                                            "min": round(min(vs), 2), "max": round(max(vs), 2)}
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xlabel("seconds into the level")
    ax.set_ylabel("critic: expected future reward")
    ax.legend(loc="lower left")
    ax.set_title(f"What the critic expects during one unseen level (seed {seed}, difficulty {d})", fontsize=13)
    fig.tight_layout()
    F.save(fig, out, "25_critic_trace.png", "The critic's expectation",
           "The critic network predicts how much reward is still to come. It drops just before a fall "
           "and stays high on a clean run: the agent 'knows' when it is in trouble.")


def fig_neurons(policy, levels, out, numbers):
    acts, obss = [], []

    def on_decision(r, obs, step):
        acts.append(policy.trace(obs)[0][0])
        obss.append(obs[:N_TILE_INPUTS])

    for lv in levels:
        play(policy, lv, on_decision=on_decision)
    A, X = np.array(acts), np.array(obss)
    # selective neurons that matter downstream: activation spread x outgoing weight size
    score = A.std(axis=0) * np.linalg.norm(policy.weights[1], axis=0)
    mean = X.mean(axis=0)
    patterns = {}
    for k in np.argsort(-score)[:120]:
        hot = A[:, k] >= np.quantile(A[:, k], 0.95)
        if hot.sum() >= 10:
            patterns[int(k)] = X[hot].mean(axis=0) - mean
    # most important first, skipping neurons that respond to the same pattern as one already shown
    top = []
    for k, v in patterns.items():
        if all(np.corrcoef(v, patterns[t])[0, 1] < 0.5 for t in top):
            top.append(k)
        if len(top) == 6:
            break
    fig, axes = plt.subplots(2, 6, figsize=(15, 4.8))
    for j, k in enumerate(top):
        diff = patterns[k].reshape(3, C.VIEW_H, C.VIEW_W)
        for row, layer, name in ((0, 0, "solid"), (1, 1, "spike")):
            a = axes[row, j]
            a.imshow(diff[layer], cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
            a.add_patch(F.Rectangle((C.VIEW_BEHIND - 0.5, C.VIEW_UP - 0.5), 1, 1, fill=False, ec=INK, lw=1.5))
            a.set_xticks([])
            a.set_yticks([])
            a.grid(False)
            if row == 0:
                a.set_title(f"neuron {k}", fontsize=10.5, fontweight="normal", color=INK2)
            if j == 0:
                a.set_ylabel(f"{name} layer", fontsize=11, color=INK2)
    fig.suptitle("What single neurons in hidden layer 1 respond to: the average view in their 5% most active "
                 "moments, minus the average view\n(red = more ground/spike here than usual, "
                 "blue = less, e.g. a gap; box = agent)",
                 x=0.01, ha="left", fontsize=12.5, fontweight="semibold")
    fig.tight_layout()
    F.save(fig, out, "26_neuron_templates.png", "What neurons respond to",
           "Nobody programmed these detectors. Training produced neurons that fire for situations such as "
           "the wall at the end of the level, flat ground under the feet, or ground dropping away ahead.")
    numbers["neurons"] = {"top": [int(k) for k in top],
                          "mean_active_h1": round(float((A > 0).sum(axis=1).mean()), 1)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ghosts", default="ghosts")
    p.add_argument("--out", default="runs/presentation")
    p.add_argument("--levels", type=int, default=100, help="levels for the blindfold test")
    p.add_argument("--sens-levels", type=int, default=25, help="levels for the sensitivity averages")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ghosts, _ = load_ghosts(args.ghosts)
    by_label = {g.label: g for g in ghosts}
    best = ghosts[-1]
    numbers = {"ghost": best.label}
    t0 = time.time()

    W = best.weights
    numbers["params_actor"] = int(sum(w.size + b.size for w, b in zip(best.weights, best.biases)))
    if best.critic is not None:
        numbers["params_critic"] = int(sum(w.size + b.size for w, b in zip(*best.critic)))
    obs = Runner(generate_level(*EXAMPLE)).observe()
    t = time.perf_counter()
    for _ in range(1000):
        best.act(obs)
    numbers["inference_ms"] = round((time.perf_counter() - t), 3)
    numbers["layers"] = [list(w.shape) for w in W]

    print("20 observation")
    fig_observation(best, out, numbers)

    print("21/22 sensitivity")
    sens_levels = eval_levels(args.sens_levels, 0.75)
    tile_maps, combined, shares, act_counts, n = collect_sensitivity(best, sens_levels)
    prof, xs = fig_where_it_looks(combined, tile_maps, n, out)
    numbers["sensitivity_decisions"] = n
    numbers["column_share"] = {int(x): round(float(v), 4) for x, v in zip(xs, prof)}
    numbers["active_neurons_mean"] = [round(float(v), 1) for v in act_counts.mean(axis=0)]
    shares_by = []
    for lab in ("10h", "300h"):
        if lab in by_label:
            sh = shares if lab == best.label else collect_sensitivity(by_label[lab], sens_levels)[2]
            shares_by.append((lab, sh))
    numbers["group_share"] = {lab: {g[0]: round(float(v), 4) for g, v in zip(GROUPS, sh)} for lab, sh in shares_by}
    fig_groups(shares_by, out)

    print("23 blindfold")
    levels = eval_levels(args.levels, 0.75)
    results = []
    for name, mask in BLINDFOLDS:
        ok = sum(play(best, lv, mask=mask)[0] for lv in levels)
        results.append((name, ok / len(levels), len(levels)))
        print(f"  {name}: {ok}/{len(levels)}")
    numbers["blindfold"] = {name: v for name, v, _ in results}
    fig_blindfold(results, out)

    print("24 actions")
    fig_actions(ghosts, eval_levels(30, 0.5), out, numbers)

    print("25 critic")
    if best.critic is not None and "5h" in by_label:
        fig_critic([by_label["5h"], best], out, numbers)

    print("26 neurons")
    fig_neurons(best, sens_levels[:10], out, numbers)

    cap_path = out / "captions.json"
    caps = json.loads(cap_path.read_text(encoding="utf-8")) if cap_path.exists() else []
    new = {c["file"] for c in F.CAPTIONS}
    caps = [c for c in caps if c["file"] not in new] + F.CAPTIONS
    caps.sort(key=lambda c: c["file"])
    cap_path.write_text(json.dumps(caps, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "explain.json").write_text(json.dumps(numbers, indent=2), encoding="utf-8")
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
