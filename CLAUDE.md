# CLAUDE.md

Guidance for Claude Code (and humans) working in this repository.

## Project

Reinforcement learning on a procedurally generated 2D platformer. A PPO agent
(Stable-Baselines3) learns to traverse never-seen levels. Training runs on
Colab/Kaggle or locally; snapshots of the agent at different amounts of simulated
training time ("ghosts") are raced against a human player in a pygame demo.
Game design, mechanics, the agent interface and results are in `game_info.md`.

## Rules

- Never `git push`. The user pushes. Committing is fine only when asked.
- No emojis in code, docs, commit messages or program output.
- Keep `game_info.md` in sync when changing mechanics, level generation, rewards,
  observations, curriculum or demo controls. Update this file when the layout or
  workflow changes.
- Phase 1 scope is stage 1 terrain plus stage 2 hazards (spikes, platform hops).
  Enemies are phase 2; do not add them unless asked.

## Layout

```
procgen_rl/
  config.py     all shared constants: timing, physics, actions, observation, reward, seed ranges
  physics.py    deterministic per-frame player physics (PlayerState, step_player)
  level.py      chunk-based procedural level generator (generate_level, Level)
  sim.py        ObservationBuilder + Runner (the only code path that advances a character)
  env.py        Gymnasium PlatformerEnv (imports gymnasium; not imported by the package __init__)
  policy.py     export SB3 actor -> .npz ghost; NumpyPolicy inference; load_ghosts
  solver.py     best-first search solver, used by tests to prove levels are beatable
  bench.py      python -m procgen_rl.bench (raw env speed)
train.py        PPO training with curriculum, sim-hour milestones, autosave, resume, ghost export,
                metrics.jsonl + status.json, STOP-file graceful stop, background ghost evaluation
evaluate.py     run every ghost on held-out seeds (and on its own training levels for --train-levels runs)
                -> <run>/eval.csv; --only-new appends only missing rows
figures.py      presentation figures (PNG) + captions.json + summary.md -> runs/presentation
demo.py         pygame demo: human vs ghosts (needs only numpy + pygame)
webapp/server.py   local control room (stdlib HTTP server, port 8765): job queue, live dashboard,
                   presentation gallery, demo launcher. Frontend in webapp/static (Chart.js vendored)
run_webapp.bat  double-click launcher for the web app
notebooks/train_colab.ipynb   Colab/Kaggle training notebook (clones the repo, saves to Drive)
tests/          pytest: level determinism + solvability, env API, env/runner parity, ghost export parity
ghosts/         ghost .npz files the demo loads by default
runs/           training outputs (gitignored)
```

## Commands

```
pip install -r requirements.txt          # training + tests
pip install -r requirements-demo.txt     # demo only (numpy, pygame)
python -m pytest -q                      # ~35 s
python -m procgen_rl.bench               # env speed
python train.py --hours 100 --out runs/main            # resumes automatically if runs/main/latest.zip exists
python train.py --hours 100 --out runs/fixed10 --train-levels 10   # generalisation experiment
python evaluate.py --ghosts runs/main/ghosts --levels 100 --plot runs/main/eval.png
python figures.py                                      # runs/main + runs/gen_* -> runs/presentation
python demo.py --ghosts runs/main/ghosts               # or copy ghosts into ./ghosts and run python demo.py
python webapp/server.py --open                         # control room at http://127.0.0.1:8765
```

## Web app (control room)

- Queue state lives in `runs/webapp_queue.json`. Jobs (train / evaluate / figures) run one at a time
  as subprocesses with `CREATE_NO_WINDOW`, so they survive the server being closed. On restart, a job
  that was running is re-queued; if its train.py is still alive (pid in `status.json`) the server
  attaches to it instead of starting a second one.
- "Stop current" writes `runs/<run>/STOP`; train.py saves `latest.zip` and exits, and the queue pauses.
- After each successful training job the server queues a catch-up evaluation and a figures refresh;
  while training runs it also refreshes figures every 30 minutes (skipping the static illustrations).
- Keeps Windows awake (SetThreadExecutionState) while a job runs. It cannot prevent sleep on lid close.
- The overnight plan (`overnight_plan()` in server.py): `main` (curriculum, 300 sim h) then
  `gen_10`, `gen_100`, `gen_1000`, `gen_unlimited` (fixed difficulty 0.5, 100 sim h each). figures.py
  finds generalisation runs by the `gen_*` folder prefix.
- Chart colours follow the validated reference palette (categorical blue/orange for compared series,
  single-hue blue ordinal ramp for ordered series like difficulty or training time; no dual axes).
  figures.py uses the same palette on a light surface.

Headless demo check (renders N frames, auto-holds right+jump, saves a screenshot):
`SDL_VIDEODRIVER=dummy python demo.py --frames 300 --screenshot out.png`

## Invariants (do not break)

- **One stepping path.** Env, evaluator, solver and demo all advance characters via
  `physics.step_player`, and env, evaluator and demo all use `sim.Runner`. Ghost
  behaviour in the demo must match training exactly. `tests/test_env.py::test_runner_matches_env_step`
  guards this.
- **Ghost compatibility.** Changing anything in `config.physics_fingerprint()`
  (physics constants, frame skip, view window, actions) invalidates existing ghosts.
  The demo and evaluator skip mismatched ghosts. Retrain after such changes.
- **Deterministic levels.** `level.py` must use `random.Random` (stable across
  platforms and Python versions), not numpy RNG or hash-based randomness. The same
  seed must give the same level on Colab and on the laptop.
- **Solvability.** Generator changes must keep `tests/test_levels.py` passing. When
  adding a chunk, derive its limits from jump physics (see `max_gap`) and add it to
  the solvability test.
- **Light demo dependencies.** `import procgen_rl` must not import gymnasium, torch or
  SB3. `demo.py` and `evaluate.py` use only numpy (+ pygame / matplotlib).
- **Disjoint seeds.** Training uses seeds below 1,000,000; evaluation starts at
  1,000,000; the demo starts at 2,000,000.
- Physics uses plain Python floats with no velocity at or above 1 tile/frame; collision
  resolution relies on this.

## Training notes

- PPO with an MLP is CPU-bound (game simulation dominates); use `--device cpu`. A GPU
  does not help unless the policy becomes a CNN.
- Measured locally: about 2,700 agent steps/s with 8 subprocess envs on a 16-thread
  CPU, so 10 simulated hours take about 3.5 minutes. Colab's free tier has 2 vCPUs,
  so expect it to be slower than the laptop.
- `train.py` writes `latest.zip` + `state.json` every simulated hour, every 10 real minutes and on
  exit/interrupt, and re-running the same command resumes.
- `--train-levels N` implies a fixed difficulty (0.5 unless `--fixed-difficulty` is given), because
  a level is defined by seed and difficulty; with the curriculum the "fixed" levels would change.
- On Windows, SubprocVecEnv needs the `if __name__ == "__main__"` guard (present in
  train.py).
