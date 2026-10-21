"""Train a PPO agent on the procedural platformer.

Runs the same way locally, on Colab/Kaggle, or launched from the web app.
Output layout (under --out):

    latest.zip          SB3 model for resuming (autosaved every --save-every-minutes
                        and every --save-every-hours of simulated time)
    state.json          curriculum difficulty + milestones already exported
    run_config.json     the arguments of the most recent launch
    status.json         live status (sim hours, speed, ETA), rewritten every few seconds
    metrics.jsonl       one line per PPO rollout: success, progress, reward, losses, ...
    checkpoints/        SB3 model at every milestone (ppo_10h.zip, ...)
    ghosts/             NumPy ghost files for the demo (ghost_0010.00h.npz, ...)
    eval.csv            held-out evaluation of each ghost (with --eval-milestones)
    tb/                 TensorBoard logs

Simulated time = agent steps * FRAME_SKIP / FPS, summed over all parallel envs.
Re-running the same command resumes from latest.zip automatically. Creating a
file named STOP in the output folder stops training cleanly (used by the web app).

Example:
    python train.py --hours 100 --out runs/main
"""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

from procgen_rl import config as C

FIXED_LEVELS_DIFFICULTY = 0.5


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="runs/main", help="output directory")
    p.add_argument("--hours", type=float, default=100.0, help="total simulated hours to train")
    p.add_argument("--milestones", default="0,0.5,1,2,5,10,20,30,50,75,100,150,200,300",
                   help="simulated hours at which ghosts are exported (comma separated)")
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--vec", choices=["auto", "subproc", "dummy"], default="auto")
    p.add_argument("--device", default="cpu", help="cpu is usually fastest for this small MLP")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fresh", action="store_true", help="ignore latest.zip and start over")
    p.add_argument("--max-wall-minutes", type=float, default=0,
                   help="stop cleanly (saving latest.zip) after this many minutes; 0 = no limit")
    p.add_argument("--save-every-hours", type=float, default=1.0,
                   help="simulated hours between latest.zip autosaves")
    p.add_argument("--save-every-minutes", type=float, default=10.0,
                   help="real minutes between latest.zip autosaves")
    p.add_argument("--eval-milestones", type=int, default=0,
                   help="evaluate each new ghost on this many held-out levels per difficulty "
                        "in a background process (0 = off)")
    # curriculum
    p.add_argument("--start-difficulty", type=float, default=0.1)
    p.add_argument("--max-difficulty", type=float, default=1.0)
    p.add_argument("--difficulty-step", type=float, default=0.1)
    p.add_argument("--promote-at", type=float, default=0.7,
                   help="success rate at the current difficulty needed to step it up")
    p.add_argument("--promote-window", type=int, default=100)
    p.add_argument("--fixed-difficulty", type=float, default=None,
                   help="disable the curriculum and train at this difficulty")
    # generalisation experiment
    p.add_argument("--train-levels", type=int, default=0,
                   help="train on only this many fixed level seeds (0 = unlimited). Implies a "
                        f"fixed difficulty ({FIXED_LEVELS_DIFFICULTY} unless --fixed-difficulty is given) "
                        "so the levels really are fixed")
    # PPO
    p.add_argument("--net", default="256,256")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--n-epochs", type=int, default=5)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--clip-range", type=float, default=0.2)
    args = p.parse_args(argv)
    if args.train_levels > 0 and args.fixed_difficulty is None:
        args.fixed_difficulty = FIXED_LEVELS_DIFFICULTY
    return args


def sim_hours(timesteps):
    return timesteps / C.AGENT_STEPS_PER_SIM_HOUR


def hours_to_steps(hours):
    return int(round(hours * C.AGENT_STEPS_PER_SIM_HOUR))


def ghost_filename(hours):
    return f"ghost_{hours:07.2f}h.npz"


def write_json(path, data):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def make_env_fn(rank, args, difficulty, level_seeds):
    def _init():
        from procgen_rl.env import PlatformerEnv
        env = PlatformerEnv(difficulty=difficulty, level_seeds=level_seeds,
                            sample_difficulty=args.fixed_difficulty is None)
        env.reset(seed=args.seed * 1000 + rank)
        return env
    return _init


def main():
    args = parse_args()
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
    import torch

    from procgen_rl.policy import export_sb3_policy, format_hours

    out = Path(args.out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "ghosts").mkdir(parents=True, exist_ok=True)
    latest_path = out / "latest.zip"
    state_path = out / "state.json"
    status_path = out / "status.json"
    metrics_path = out / "metrics.jsonl"
    stop_path = out / "STOP"
    if stop_path.exists():
        stop_path.unlink()

    resume = latest_path.exists() and not args.fresh
    if not resume and metrics_path.exists():
        metrics_path.unlink()
    state = {"difficulty": args.start_difficulty, "milestones_done": []}
    if resume and state_path.exists():
        state.update(json.loads(state_path.read_text()))
    if args.fixed_difficulty is not None:
        state["difficulty"] = args.fixed_difficulty
    state["train_levels"] = args.train_levels
    state["fixed_difficulty"] = args.fixed_difficulty
    write_json(out / "run_config.json", {**vars(args), "launched": time.strftime("%Y-%m-%d %H:%M:%S")})

    level_seeds = list(range(args.train_levels)) if args.train_levels > 0 else None

    milestones = sorted({float(m) for m in args.milestones.split(",") if m.strip()} | {args.hours})
    milestones = [m for m in milestones if m <= args.hours]

    use_subproc = args.vec == "subproc" or (args.vec == "auto" and (os.cpu_count() or 1) > 2 and args.n_envs > 1)
    fns = [make_env_fn(i, args, state["difficulty"], level_seeds) for i in range(args.n_envs)]
    venv = SubprocVecEnv(fns) if use_subproc else DummyVecEnv(fns)
    venv = VecMonitor(venv)

    net = [int(x) for x in args.net.split(",")]
    if resume:
        print(f"Resuming from {latest_path}")
        model = PPO.load(latest_path, env=venv, device=args.device,
                         tensorboard_log=str(out / "tb"))
    else:
        model = PPO(
            "MlpPolicy", venv,
            learning_rate=args.lr, n_steps=args.n_steps, batch_size=args.batch_size,
            n_epochs=args.n_epochs, gamma=args.gamma, gae_lambda=args.gae_lambda,
            ent_coef=args.ent_coef, clip_range=args.clip_range,
            policy_kwargs=dict(net_arch=dict(pi=net, vf=net), activation_fn=torch.nn.ReLU),
            tensorboard_log=str(out / "tb"), seed=args.seed, device=args.device, verbose=0,
        )
    total = hours_to_steps(args.hours)

    def save_state():
        write_json(state_path, state)

    def save_latest():
        model.save(latest_path)
        save_state()

    class TrainingCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.recent = deque(maxlen=200)          # (success, progress, reward) over all levels
            self.at_ceiling = deque(maxlen=args.promote_window)
            self.next_save_h = sim_hours(model.num_timesteps) + args.save_every_hours
            self.next_save_t = time.time() + args.save_every_minutes * 60
            self.t0 = time.time()
            self.steps0 = model.num_timesteps
            self.last_print = time.time()
            self.last_status = 0.0
            self.eval_proc = None
            self.eval_pending = False
            self.stop_reason = None

        # ---------------------------------------------------- milestones
        def export_due_milestones(self):
            h = sim_hours(self.num_timesteps)
            for m in milestones:
                if m <= h + 1e-9 and m not in state["milestones_done"]:
                    label = format_hours(m)
                    export_sb3_policy(self.model, out / "ghosts" / ghost_filename(m), m, label=label,
                                      extra={"difficulty": state["difficulty"],
                                             "train_levels": args.train_levels,
                                             "run": out.name})
                    self.model.save(out / "checkpoints" / f"ppo_{label}.zip")
                    state["milestones_done"].append(m)
                    save_state()
                    print(f"[milestone] {label}: exported ghost (difficulty {state['difficulty']:.2f})", flush=True)
                    self.eval_pending = True
            self.maybe_launch_eval()

        def maybe_launch_eval(self):
            if not (args.eval_milestones and self.eval_pending):
                return
            if self.eval_proc is not None and self.eval_proc.poll() is None:
                return  # one at a time; the next launch picks up everything new
            self.eval_pending = False
            cmd = [sys.executable, str(Path(__file__).with_name("evaluate.py")), "--ghosts", str(out / "ghosts"),
                   "--levels", str(args.eval_milestones), "--only-new"]
            log = open(out / "eval.log", "a")
            self.eval_proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)

        def finish_eval(self):
            if self.eval_proc is not None:
                self.eval_proc.wait()
            if self.eval_pending:
                self.maybe_launch_eval()
                if self.eval_proc is not None:
                    self.eval_proc.wait()

        # ---------------------------------------------------- per step
        def _on_training_start(self):
            self.export_due_milestones()

        def _on_step(self):
            for done, info in zip(self.locals["dones"], self.locals["infos"]):
                if not done:
                    continue
                ep_r = info.get("episode", {}).get("r", np.nan)
                self.recent.append((info["is_success"], info["progress"], ep_r))
                if abs(info["difficulty"] - state["difficulty"]) < 1e-6:
                    self.at_ceiling.append(info["is_success"])
            self.maybe_promote()
            self.export_due_milestones()

            h = sim_hours(self.num_timesteps)
            now = time.time()
            if h >= self.next_save_h or now >= self.next_save_t:
                save_latest()
                self.next_save_h = h + args.save_every_hours
                self.next_save_t = now + args.save_every_minutes * 60
            if now - self.last_status > 5:
                self.write_status("training")
            if now - self.last_print > 30:
                self.report()
            if stop_path.exists():
                self.stop_reason = "stop requested"
            elif args.max_wall_minutes and now - self.t0 > args.max_wall_minutes * 60:
                self.stop_reason = "wall-clock limit"
            if self.stop_reason:
                print(f"Stopping: {self.stop_reason}", flush=True)
                return False
            return True

        def maybe_promote(self):
            if args.fixed_difficulty is not None or state["difficulty"] >= args.max_difficulty - 1e-9:
                return
            if len(self.at_ceiling) < self.at_ceiling.maxlen:
                return
            if np.mean(self.at_ceiling) >= args.promote_at:
                state["difficulty"] = round(min(args.max_difficulty, state["difficulty"] + args.difficulty_step), 4)
                self.training_env.env_method("set_difficulty", state["difficulty"])
                self.at_ceiling.clear()
                save_state()
                print(f"[curriculum] difficulty -> {state['difficulty']:.2f} at "
                      f"{sim_hours(self.num_timesteps):.2f} sim h", flush=True)

        # ---------------------------------------------------- reporting
        def speed(self):
            return (model.num_timesteps - self.steps0) / max(1e-6, time.time() - self.t0)

        def recent_means(self):
            if not self.recent:
                return 0.0, 0.0, float("nan")
            a = np.array(self.recent, dtype=np.float64)
            return float(a[:, 0].mean()), float(a[:, 1].mean()), float(np.nanmean(a[:, 2]))

        def write_status(self, phase):
            self.last_status = time.time()
            sps = self.speed()
            succ, prog, _ = self.recent_means()
            write_json(status_path, {
                "phase": phase,
                "sim_hours": sim_hours(model.num_timesteps),
                "target_hours": args.hours,
                "timesteps": int(model.num_timesteps),
                "steps_per_sec": sps,
                "eta_minutes": max(0, total - model.num_timesteps) / max(sps, 1e-6) / 60,
                "difficulty": state["difficulty"],
                "success_rate": succ,
                "progress": prog,
                "milestones_done": state["milestones_done"],
                "updated": time.time(),
                "pid": os.getpid(),
            })

        def report(self):
            self.last_print = time.time()
            h = sim_hours(model.num_timesteps)
            sps = self.speed()
            succ, prog, _ = self.recent_means()
            eta = max(0, total - model.num_timesteps) / max(sps, 1e-6) / 60
            print(f"sim {h:7.2f}h | difficulty {state['difficulty']:.2f} | success {succ:.2f} | "
                  f"progress {prog:.2f} | {sps:,.0f} steps/s | eta {eta:.1f} min", flush=True)

        def _on_rollout_end(self):
            h = sim_hours(self.num_timesteps)
            succ, prog, rew = self.recent_means()
            self.logger.record("sim/hours", h)
            self.logger.record("curriculum/difficulty", state["difficulty"])
            self.logger.record("rollout/success_rate", succ)
            self.logger.record("rollout/progress", prog)
            row = {"sim_hours": round(h, 5), "timesteps": int(self.num_timesteps),
                   "wall": round(time.time(), 1), "difficulty": state["difficulty"],
                   "success_rate": round(succ, 4), "progress": round(prog, 4),
                   "episode_reward": None if np.isnan(rew) else round(rew, 4),
                   "steps_per_sec": round(self.speed(), 1)}
            # PPO losses from the previous update (recorded but not yet dumped)
            vals = self.logger.name_to_value
            for k in ("entropy_loss", "value_loss", "policy_gradient_loss", "approx_kl",
                      "clip_fraction", "explained_variance"):
                if f"train/{k}" in vals:
                    row[k] = round(float(vals[f"train/{k}"]), 6)
            with open(metrics_path, "a") as f:
                f.write(json.dumps(row) + "\n")

    remaining = total - model.num_timesteps
    print(f"Training to {args.hours:g} sim hours ({total:,} agent steps), "
          f"{remaining:,} remaining, {args.n_envs} envs ({'subproc' if use_subproc else 'dummy'}), "
          f"device {args.device}", flush=True)
    cb = TrainingCallback()
    phase = "finished"
    try:
        if remaining > 0:
            model.learn(total_timesteps=remaining, callback=cb, reset_num_timesteps=not resume,
                        tb_log_name="ppo", progress_bar=False)
        else:
            cb.init_callback(model)
            cb.export_due_milestones()
        if cb.stop_reason:
            phase = "stopped"
    except KeyboardInterrupt:
        print("Interrupted, saving latest.zip", flush=True)
        phase = "stopped"
    finally:
        save_latest()
        venv.close()
        cb.write_status("evaluating" if args.eval_milestones else phase)
        cb.finish_eval()
        cb.write_status(phase)
        if stop_path.exists():
            stop_path.unlink()
    print(f"Done at {sim_hours(model.num_timesteps):.2f} sim hours ({phase}). Ghosts in {out / 'ghosts'}", flush=True)


if __name__ == "__main__":
    main()
