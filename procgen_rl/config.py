"""Shared constants for physics, observation, reward and timing.

Everything here is shared by training (env.py) and the demo (demo.py).
Changing any value in the PHYSICS or OBSERVATION sections makes previously
trained ghosts invalid, so retrain after editing them.
"""

# ---------------------------------------------------------------- timing
FPS = 60                    # physics frames per simulated second
FRAME_SKIP = 4              # physics frames per agent decision
SIM_FRAMES_PER_HOUR = FPS * 3600
AGENT_STEPS_PER_SIM_HOUR = SIM_FRAMES_PER_HOUR // FRAME_SKIP  # 54_000

# ---------------------------------------------------------------- tiles
EMPTY = 0
SOLID = 1
SPIKE = 2                   # hazard, occupies the lower half of its tile

LEVEL_HEIGHT = 16           # rows; row 0 is the top, y grows downward

# ---------------------------------------------------------------- physics
# Units are tiles and frames. Values below are tuned so that from flat ground
# the player can jump ~3.8 tiles high and clear a ~6 tile gap at full speed.
PLAYER_W = 0.75
PLAYER_H = 0.90

GRAVITY = 0.020
JUMP_VELOCITY = 0.40
JUMP_CUT = 0.5              # vy multiplier when jump is released while rising
MAX_FALL = 0.50             # must stay < 1 tile/frame (collision assumes it)
RUN_SPEED = 0.15            # must stay < 1 tile/frame
GROUND_ACCEL = 0.015
AIR_ACCEL = 0.010
GROUND_FRICTION = 0.020
AIR_FRICTION = 0.004
COYOTE_FRAMES = 5           # frames after leaving a ledge where jump still works
SPIKE_MARGIN = 0.10         # spike hitbox is shrunk by this much (forgiveness)

# Action table: index -> (move, jump). move in {-1, 0, 1}, jump in {0, 1}.
ACTIONS = (
    (0, 0),
    (-1, 0),
    (1, 0),
    (0, 1),
    (-1, 1),
    (1, 1),
)
ACTION_NAMES = ("idle", "left", "right", "jump", "left+jump", "right+jump")

# ---------------------------------------------------------------- observation
# Egocentric tile window around the player's center tile.
VIEW_BEHIND = 4             # columns to the left of the player's column
VIEW_AHEAD = 15             # columns to the right
VIEW_UP = 6                 # rows above the player's row
VIEW_DOWN = 7               # rows below
VIEW_W = VIEW_BEHIND + 1 + VIEW_AHEAD   # 20
VIEW_H = VIEW_UP + 1 + VIEW_DOWN        # 14
OBS_CHANNELS = 3            # solid, spike, goal
OBS_SCALARS = 7             # vx, vy, on_ground, can_jump, jumping, frac_x, frac_y
OBS_DIM = OBS_CHANNELS * VIEW_W * VIEW_H + OBS_SCALARS

# ---------------------------------------------------------------- reward
PROGRESS_REWARD = 0.1       # per tile of new furthest x
GOAL_REWARD = 5.0
DEATH_PENALTY = 1.0
TIME_PENALTY = 0.002        # per agent step
MAX_EPISODE_STEPS = 1500    # agent steps (100 simulated seconds)
STALL_STEPS = 90            # agent steps without new progress -> truncate

# ---------------------------------------------------------------- level seeds
# Disjoint seed ranges so evaluation and demo levels are never trained on.
TRAIN_SEED_RANGE = (0, 1_000_000)
EVAL_SEED_START = 1_000_000
DEMO_SEED_START = 2_000_000


def physics_fingerprint():
    """Values a ghost must have been trained with to behave correctly."""
    return {
        "fps": FPS,
        "frame_skip": FRAME_SKIP,
        "obs_dim": OBS_DIM,
        "view": [VIEW_BEHIND, VIEW_AHEAD, VIEW_UP, VIEW_DOWN],
        "physics": [PLAYER_W, PLAYER_H, GRAVITY, JUMP_VELOCITY, JUMP_CUT, MAX_FALL,
                    RUN_SPEED, GROUND_ACCEL, AIR_ACCEL, GROUND_FRICTION,
                    AIR_FRICTION, COYOTE_FRAMES, SPIKE_MARGIN],
        "actions": [list(a) for a in ACTIONS],
    }
