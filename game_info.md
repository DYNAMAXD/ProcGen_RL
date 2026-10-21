# Game info

Design reference for the ProcGen RL platformer: mechanics, level generation, the
agent interface and the demo. Keep this file in sync with `procgen_rl/config.py`
and `procgen_rl/level.py` whenever mechanics, rewards or observations change.

## Concept

A 2D side-scroller (left to right, Mario-like). Every level is generated
procedurally from a seed, so the agent is always tested on terrain it has never
seen. A PPO agent learns to traverse levels; snapshots of the agent taken at
different amounts of simulated training time are raced against each other and
against a human player in the demo, as semi-transparent "ghosts" with name tags
(0min, 30min, 1h, 2h, 5h, 10h, ...).

Scope:
- Phase 1 (done): stage 1 terrain (flat, steps, stairs, gaps) and stage 2
  hazards (spikes, floating platform hops), curriculum, Colab training, demo.
- Phase 2 (planned): enemies (patrolling walkers, stomp or avoid), possibly
  moving platforms. See "Phase 2 notes" at the bottom.

## World and units

- Units are tiles and frames. 1 tile = 40 px in the demo at scale 1.0.
- Physics runs at 60 frames per simulated second (`FPS`).
- Levels are 16 rows tall. Row 0 is the top; y grows downward.
- Left and right of the level are solid walls; above and below are open. Falling
  below the bottom row kills the player.

## Physics (`procgen_rl/physics.py`, constants in `config.py`)

| Constant | Value | Meaning |
|---|---|---|
| PLAYER_W x PLAYER_H | 0.75 x 0.90 tiles | player hitbox |
| GRAVITY | 0.020 tiles/frame^2 | |
| JUMP_VELOCITY | 0.40 tiles/frame | jump height ~3.8 tiles |
| JUMP_CUT | 0.5 | releasing jump while rising halves vertical speed (variable jump height) |
| MAX_FALL | 0.50 tiles/frame | terminal velocity |
| RUN_SPEED | 0.15 tiles/frame | 9 tiles/s |
| GROUND_ACCEL / AIR_ACCEL | 0.015 / 0.010 | per frame |
| GROUND_FRICTION / AIR_FRICTION | 0.020 / 0.004 | per frame, when no input |
| COYOTE_FRAMES | 5 | jump still allowed shortly after leaving a ledge |
| SPIKE_MARGIN | 0.10 tiles | spike hitbox forgiveness |

Derived: jump airtime on flat ground is ~40 frames, so a full-speed jump covers ~6
tiles horizontally. Holding jump re-jumps immediately on landing.

Collision is axis-separated (move x and resolve, then move y and resolve), which is
exact because no velocity exceeds 1 tile/frame. All maths uses Python floats, so the
same inputs give bit-identical results on any machine.

## Tiles

| Id | Name | Behaviour |
|---|---|---|
| 0 | EMPTY | air |
| 1 | SOLID | ground, steps, floating platforms |
| 2 | SPIKE | kills on touch; hitbox is the lower half of the tile |

The goal is a vertical line at `level.goal_x` (flag pole with a checkered finish).
Touching it wins.

## Level generation (`procgen_rl/level.py`)

A level is: an 8-column start area, then randomly chosen chunks until 140 columns
have been placed, then a 10-column finish area with the flag. Typical width is
160 to 190 columns, around 20 seconds at full speed.

`generate_level(seed, difficulty)` is fully deterministic. It uses
`random.Random(seed)`, which gives the same output across Python versions and
platforms.

Ground surface height ranges from 2 to 9 tiles above the bottom; the start is at 3.

| Chunk | Available | Parameters (d = difficulty in [0, 1]) |
|---|---|---|
| flat | always | 3 to 8-4d columns |
| step | always | height change of 1 to 1+2d tiles, up or down |
| stairs | d > 0.2 | 2 to 4 single-tile steps, 1 to 2 columns each |
| gap | always | pit 1 to gap_limit(dh) wide; landing height change of -2 to +1 (+1 only when d > 0.5) |
| spikes | d >= 0.25 | 1 to min(3, 1+2d) spike tiles on the ground |
| platforms | d >= 0.35 | pit with 1 to 1+2d floating platforms (2 to 3 wide, 1 wide when d > 0.8), each within -2/+1 height of the previous |

Chunk weights: flat 3-2d, step 2, stairs 1, gap 1+2d, spikes 2d, platforms 2d.

Solvability rules:
- `max_gap(dh)`: 4 tiles when landing at the same height or lower, 3 tiles when 1 up,
  2 tiles when 2 or more up.
- `gap_limit = min(max_gap(dh), 1 + round(3d))`, so difficulty 0 has only 1-tile gaps.
- Steps up are at most 3 tiles (jump height is 3.8).
- Every hazard chunk starts with at least 2 tiles of runway.

`tests/test_levels.py` runs a search-based solver (`procgen_rl/solver.py`, same
actions and frame skip as the agent) on 200 held-out levels across all difficulties
and requires every one to be beatable.

## Agent interface (`procgen_rl/env.py`)

Gymnasium env `PlatformerEnv`. Every reset generates a new level.

**Actions** (Discrete 6, each held for FRAME_SKIP = 4 frames):
idle, left, right, jump, left+jump, right+jump.

**Observation** (flat float32 vector, 847 values):
- An egocentric tile window around the player's center tile: 4 columns behind,
  15 ahead (20 wide) and 6 rows up, 7 down (14 tall).
- 3 one-hot layers: solid (walls outside the level count as solid), spike, and goal
  region (columns at or past the flag). 3 x 14 x 20 = 840.
- 7 scalars: vx / RUN_SPEED, vy / MAX_FALL, on_ground, can_jump (coyote > 0),
  jumping (rising with jump held), and the player center's fractional position
  within its tile (x, y).

**Reward** per agent step:
- +0.1 per tile of new furthest-right progress (moving back and forth earns nothing)
- -0.002 time penalty
- -1.0 on death (fall or spike), episode terminates
- +5.0 on reaching the flag, episode terminates

**Truncation:** 90 agent steps (6 s) without new progress, or 1500 agent steps (100 s).

**Info:** level_seed, difficulty, progress (0 to 1), is_success, death_cause, sim_frames.

## Curriculum

The env holds a difficulty ceiling. Each level uses the ceiling 70% of the time and a
uniform random difficulty below it 30% of the time, so easier skills are not
forgotten. `train.py` raises the ceiling by 0.1 when the success rate over the last
100 episodes played at the ceiling reaches 70%. It starts at 0.1 and is capped at 1.0.
`--fixed-difficulty` disables the curriculum.

## Training and simulated time

- Algorithm: PPO (Stable-Baselines3), MlpPolicy with separate 256-256 ReLU actor and
  critic. n_steps 256 x 8 envs, batch 512, 5 epochs, lr 3e-4, gamma 0.99,
  GAE 0.95, entropy 0.01, clip 0.2.
- Simulated time = agent steps x FRAME_SKIP / FPS, summed over all parallel envs.
  1 simulated hour = 54,000 agent steps.
- Ghost milestones (simulated hours): 0, 0.5, 1, 2, 5, 10, 20, 30, 50, 75, 100, 150,
  200, 300, plus the final hour value.
- Every milestone writes `ghosts/ghost_XXXX.XXh.npz` (NumPy actor weights and
  metadata) and `checkpoints/ppo_<label>.zip` (full SB3 model).
- Autosave: `latest.zip` every simulated hour, every 10 real minutes, and on stop or exit.
- With `--eval-milestones N`, each new ghost is evaluated in a background process on N
  held-out levels per difficulty (0.25, 0.5, 0.75, 1.0) and appended to `eval.csv`.

## Experiments (overnight plan in the web app)

| Run | Levels | Difficulty | Sim hours | Purpose |
|---|---|---|---|---|
| main | unlimited | curriculum 0.1 to 1.0 | 300 | showcase run, ghosts for the demo |
| gen_10 | 10 fixed seeds | fixed 0.5 | 100 | generalisation experiment |
| gen_100 | 100 fixed seeds | fixed 0.5 | 100 | generalisation experiment |
| gen_1000 | 1,000 fixed seeds | fixed 0.5 | 100 | generalisation experiment |
| gen_unlimited | unlimited | fixed 0.5 | 100 | generalisation baseline |

The generalisation runs use a fixed difficulty because a level is defined by
(seed, difficulty); with the curriculum, the "fixed" levels would change. Each run is
evaluated on its own training levels and on never-seen levels at difficulty 0.5. The gap
between the two is the overfitting measure (as in Cobbe et al., 2019).

## Presentation figures (`figures.py` -> `runs/presentation`)

01 level gallery (one seed, five difficulties) · 02 level variety · 03 observation
window · 04 policy network · 05 system pipeline · 07 learning curve · 08 curriculum ·
09 success on unseen levels per ghost · 10 success matrix · 11 ghost trajectories on one
level · 12 PPO diagnostics · 13 demo screenshot · 14 finish time · 15 generalisation
(train vs unseen) · 16 generalisation over training time · `summary.md` with the key
numbers. Each figure has a caption in `captions.json`.

## Seed ranges

| Use | Seeds |
|---|---|
| Training | 0 to 999,999 (drawn at random per level) |
| Evaluation (`evaluate.py`) | 1,000,000 onward |
| Demo (`N` key) | 2,000,000 to 2,999,999 |

`--train-levels N` trains on seeds 0 to N-1 only. This is the generalisation
experiment: compare its ghosts against the unlimited-levels run on the same held-out
evaluation seeds.

## Ghosts and the demo (`demo.py`)

- Ghost file: `.npz` holding the actor MLP weights (W0, b0, ...) plus JSON metadata:
  label, sim_hours, timesteps, activation, difficulty at export, and a physics
  fingerprint. The demo skips ghosts whose fingerprint does not match the current
  `config.py`.
- Every ghost plays the same level live, using the shared `Runner`. It picks an
  action every 4 frames, exactly as in training. Ghosts use argmax actions by
  default; press S for sampled (stochastic) actions.
- Ghosts are marked "stuck" with the same cut-offs used in training (6 s without
  progress, or 100 s total).
- The human acts every frame (no frame skip), which is a small advantage.

Controls: Left/Right or A/D to run; Space, Up or W to jump (hold for higher); R to
restart; N for a new level; `[` and `]` to change difficulty (same seed); P to switch
between playing and spectating; G to toggle ghosts; C to switch the camera between
you and the leader; S to switch deterministic and stochastic; F to change speed
(1x, 2x, 4x); L for the leaderboard; H for help; Esc to quit.

## Results so far

Full results from the overnight plan go into `runs/presentation/summary.md`. Earlier
pilot result (evaluation difficulties 0.3/0.6/1.0, before evaluation moved to
0.25/0.5/0.75/1.0): a quick local run of 10 simulated hours (about 3.5 real minutes on a 16-thread laptop CPU)
reached curriculum difficulty 0.6. Evaluated on 50 unseen levels per difficulty
(success rate / mean progress):

| Ghost | d=0.3 | d=0.6 | d=1.0 |
|---|---|---|---|
| 0min | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| 30min | 0.32 / 0.66 | 0.02 / 0.34 | 0.00 / 0.26 |
| 1h | 0.00 / 0.24 | 0.00 / 0.21 | 0.00 / 0.16 |
| 2h | 0.14 / 0.46 | 0.06 / 0.29 | 0.00 / 0.18 |
| 5h | 0.56 / 0.82 | 0.16 / 0.49 | 0.12 / 0.46 |
| 10h | 0.98 / 0.99 | 0.76 / 0.89 | 0.42 / 0.73 |

Progress is not monotonic early on (the 1h ghost is worse than the 30min one). That is
normal for PPO and makes the ghost race more interesting.

## Phase 2 notes (enemies)

Planned approach, so the phase 1 code does not need restructuring:
- Add an `enemies` list to `Level` (spawn x, patrol range, speed), generated by a new
  `enemy` chunk gated by difficulty.
- Enemy motion must be a pure function of the frame number (for example a ping-pong
  patrol), so every racer in the demo sees the same enemies without enemies
  interacting with racers.
- Add an "enemy" observation layer, and possibly enemy velocity, which changes
  OBS_DIM and therefore requires retraining (the fingerprint check enforces this).
- Extend the solver to account for enemy timing, or keep enemy chunks provably
  avoidable (for example, always jumpable from flat ground).
