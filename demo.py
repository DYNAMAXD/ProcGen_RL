"""Playable demo: race against ghosts of the agent from different training stages.

    python demo.py                         # play with ghosts from ./ghosts
    python demo.py --ghosts runs/main/ghosts --difficulty 0.6
    python demo.py --spectate              # watch the ghosts only
    python demo.py --debug 300h            # one ghost alone, with its inputs, network and decisions

Every ghost is a saved policy snapshot (e.g. after 1h, 10h, 50h of simulated
training) playing the same freshly generated level live, with the exact same
physics and observation the agent was trained with. Needs only numpy + pygame.
Press H in the demo for the controls.
"""

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np
import pygame

import debug_view as DV
from procgen_rl import config as C
from procgen_rl.level import MAX_DIFFICULTY, generate_level
from procgen_rl.policy import load_ghosts
from procgen_rl.sim import DEAD, FINISHED, RUNNING, Runner
from procgen_rl.solver import solve

BASE_TILE = 40
VIEW_COLS = 32
HUD_H = 52
MINIMAP_H = 26
COUNTDOWN_FRAMES = 120
SPEEDS = (1, 2, 4, 0.5, 0.25)

SKY_TOP = (104, 164, 232)
SKY_BOTTOM = (204, 230, 250)
HILL_FAR = (160, 204, 186)
HILL_NEAR = (122, 178, 140)
CLOUD = (245, 249, 255)
GRASS = (92, 182, 74)
GRASS_DARK = (64, 142, 54)
DIRT = (152, 106, 70)
DIRT_DARK = (128, 88, 58)
SPIKE_FILL = (214, 218, 226)
SPIKE_EDGE = (86, 92, 108)
HUD_BG = (22, 26, 36)
HUD_TEXT = (232, 236, 242)
HUD_DIM = (146, 154, 170)
PANEL_BG = (16, 20, 30, 240)
PLAYER_COLOR = (232, 68, 58)
RANK_UP = (96, 204, 128)
RANK_DOWN = (232, 96, 84)
GHOST_ALPHA = 90            # ghost body opacity (the player is fully opaque)
GHOST_LABEL_ALPHA = 140     # ghost name tag opacity
MAX_LABEL_LIFTS = 1         # a crowded ghost tag moves up at most this many rows, then hides
RANK_WINDOW = 30            # standings are compared every half second (no flicker when bunched)
RANK_FLASH_FRAMES = 90      # how long an overtaking arrow stays on the leaderboard
STILL_FRAMES = 8            # the clock stops once no racer has moved for this many frames
MAX_SNAPSHOTS = 2000        # debug view: decisions you can step back (a level is at most 1500)
STUCK = "stuck"

HELP_LINES = [
    ("Left/Right or A/D", "run"),
    ("Space / Up / W", "jump (hold for higher)"),
    ("R", "restart this level"),
    ("N", "new random level"),
    ("[  ]", "difficulty down / up (above 1.0: expert)"),
    ("P", "toggle playing / spectating"),
    ("G", "toggle ghosts"),
    ("C", "camera: you / leading ghost"),
    ("S", "ghosts: deterministic / stochastic"),
    ("F", "speed 1x / 2x / 4x / 0.5x / 0.25x"),
    ("L", "toggle leaderboard"),
    ("I", "debug view: one ghost, its inputs and network"),
    ("  Tab / Shift+Tab", "debug: next / previous ghost"),
    ("  Space", "debug: pause / resume"),
    ("  Left / Right", "debug: one decision back / forward"),
    ("  O", "debug: tile importance overlay on / off"),
    ("H", "toggle this help"),
    ("Esc", "quit"),
]


def difficulty_name(d):
    if d <= 0.35:
        return "easy"
    if d <= 0.7:
        return "normal"
    if d <= 1.0:
        return "hard"
    return "expert" if d <= 1.5 else "insane"


def step_difficulty(d, up):
    """0.1 steps up to 1.0 (the training range), 0.25 steps through the expert range."""
    step = 0.25 if (d >= 1.0 - 1e-9 if up else d > 1.0 + 1e-9) else 0.1
    d = d + step if up else d - step
    return round(min(MAX_DIFFICULTY, max(0.0, d)), 2)


def ghost_colors(n):
    """Early ghosts cool blue, late ghosts warm gold."""
    cols = []
    for i in range(n):
        t = i / max(1, n - 1)
        c = pygame.Color(0)
        c.hsva = ((215 - 175 * t) % 360, 60 + 25 * t, 95, 100)
        cols.append((c.r, c.g, c.b))
    return cols


class Racer:
    def __init__(self, name, color, policy=None):
        self.name = name
        self.color = color
        self.policy = policy
        self.runner = None
        self.sprites = {}
        self.label_surf = None

    @property
    def is_human(self):
        return self.policy is None

    def reset(self, level):
        self.runner = Runner(level)
        self.move = 0
        self.jump = 0
        self.best_x = self.runner.max_x
        self.stall_frames = 0
        self.stuck = False
        self.finish_time = None
        self.rank = None
        self.rank_change = 0      # +1 moved up, -1 moved down
        self.rank_change_until = 0

    @property
    def status(self):
        return STUCK if self.stuck else self.runner.status

    @property
    def active(self):
        return self.runner.status == RUNNING and not self.stuck


class Demo:
    def __init__(self, args):
        pygame.init()
        pygame.display.set_caption("ProcGen RL - race the ghosts")
        self.scale = args.scale
        self.display_flags = pygame.SCALED if args.scaled else 0
        self.clock = pygame.time.Clock()
        self.font = self._font(18)
        self.font_small = self._font(15, bold=True)
        self.font_tiny = self._font(13, bold=True)
        self.font_big = self._font(96, bold=True)
        self.font_mid = self._font(32, bold=True)

        self.difficulty = args.difficulty
        self.seed = args.seed if args.seed is not None else self._random_seed()
        self.playing = not args.spectate
        self.show_ghosts = True
        self.show_board = True
        self.show_help = False
        self.follow_leader = args.spectate
        self.deterministic = not args.stochastic
        self.speed_idx = 0
        self.rng = np.random.default_rng(0)
        self.debug = False
        self.dbg = DV.DebugView()
        self.paused = False
        self.step_frames = 0
        self.show_heat = True

        ghost_dir = self._resolve_ghost_dir(args.ghosts)
        policies, skipped = load_ghosts(ghost_dir) if ghost_dir else ([], [])
        if args.only:
            wanted = {s.strip() for s in args.only.split(",")}
            policies = [p for p in policies if p.label in wanted]
        for name, why in skipped:
            print(f"skipped ghost {name}: {why}")
        print(f"loaded {len(policies)} ghosts from {ghost_dir}" if policies else
              "no ghosts found - train first (see README); playing solo")
        self.ghost_dir = ghost_dir
        self.ghosts = [Racer(p.label, c, p) for p, c in zip(policies, ghost_colors(len(policies)))]
        self.human = Racer("YOU", PLAYER_COLOR)
        self.dbg_index = max(0, len(self.ghosts) - 1)   # the debug view starts with the most trained ghost
        if args.debug is not None and self.ghosts:
            labels = [g.name for g in self.ghosts]
            if args.debug in labels:
                self.dbg_index = labels.index(args.debug)
            self.debug = True
        self.level = None
        self._apply_layout()
        self.load_level()

    def _apply_layout(self):
        """Window size and everything drawn at the tile size: normal race or debug view."""
        if self.debug:
            self.T = max(16, int(DV.GAME_TILE * self.scale))
            self.w = DV.GAME_COLS * self.T
        else:
            self.T = max(16, int(BASE_TILE * self.scale))
            self.w = VIEW_COLS * self.T
        self.play_h = C.LEVEL_HEIGHT * self.T
        self.h = HUD_H + self.play_h + MINIMAP_H
        size = (self.w, self.h)
        if self.debug:
            size = (self.w + DV.PANEL_W, max(DV.MIN_H, self.h + DV.DECISION_H))
        self.screen = pygame.display.set_mode(size, self.display_flags)
        for r in self.ghosts + [self.human]:
            self._build_sprites(r)
        self.bg = self._build_background()
        if self.level is not None:
            self.level_surf = self._render_level(self.level)
            self.minimap = self._render_minimap(self.level)
            target = self._camera_target()
            self.cam_x = self._camera_goal(target) if target else 0.0

    @property
    def debug_ghost(self):
        return self.ghosts[self.dbg_index] if self.debug and self.ghosts else None

    # ------------------------------------------------------------ setup
    @staticmethod
    def _font(size, bold=False):
        return pygame.font.SysFont("segoeui,helveticaneue,dejavusans,arial", size, bold=bold)

    @staticmethod
    def _random_seed():
        return random.randrange(C.DEMO_SEED_START, C.DEMO_SEED_START + 1_000_000)

    @staticmethod
    def _resolve_ghost_dir(path):
        p = Path(path)
        if p.is_dir() and (p / "ghosts").is_dir():
            p = p / "ghosts"
        if p.is_dir() and any(p.glob("*.npz")):
            return p
        runs = sorted(Path("runs").glob("*/ghosts"), key=lambda d: d.stat().st_mtime, reverse=True)
        for d in runs:
            if any(d.glob("*.npz")):
                return d
        return None

    @property
    def racers(self):
        if self.debug_ghost:
            return [self.debug_ghost]
        rs = list(self.ghosts) if self.show_ghosts else []
        if self.playing:
            rs.append(self.human)
        return rs

    def load_level(self):
        self.level = generate_level(self.seed, self.difficulty)
        if self.difficulty > 1.0:
            # expert levels are tested beatable, but check anyway and skip to the next seed if not
            for _ in range(5):
                if solve(self.level, max_expansions=200_000) is not None:
                    break
                self.seed += 1
                self.level = generate_level(self.seed, self.difficulty)
        self.level_surf = self._render_level(self.level)
        self.minimap = self._render_minimap(self.level)
        self.restart()

    def restart(self):
        for r in self.ghosts + [self.human]:
            r.reset(self.level)
        self.countdown = 0 if self.debug_ghost else COUNTDOWN_FRAMES
        self.dbg.reset()
        self.go_timer = 0
        self.frame = 0
        self.end_frame = None     # the clock stops here once the race is over
        self.last_move_frame = 0  # last frame in which any racer changed position
        self.still_frames = 0     # frames since then
        self.snapshots = []       # debug view: state before each decision, for stepping back
        target = self._camera_target()
        self.cam_x = self._camera_goal(target) if target else 0.0

    # ------------------------------------------------------------ simulation
    def update(self, keys):
        if self.debug_ghost and self.paused:
            if self.step_frames <= 0:
                return
            self.step_frames -= 1
        if self.countdown > 0:
            self.countdown -= 1
            if self.countdown == 0:
                self.go_timer = 40
            return
        if self.go_timer > 0:
            self.go_timer -= 1
        self.frame += 1
        moved = False
        for r in self.racers:
            if not r.active:
                continue
            run = r.runner
            before = (run.state.x, run.state.y)
            if r.is_human:
                r.move = (keys[pygame.K_RIGHT] or keys[pygame.K_d]) - (keys[pygame.K_LEFT] or keys[pygame.K_a])
                r.jump = int(keys[pygame.K_SPACE] or keys[pygame.K_UP] or keys[pygame.K_w])
            elif run.frames % C.FRAME_SKIP == 0:
                if r is self.debug_ghost:
                    self.snapshots.append(self._snapshot(r))
                    del self.snapshots[:-MAX_SNAPSHOTS]
                    a = self.dbg.decide(r.policy, run.observe(), deterministic=self.deterministic, rng=self.rng)
                else:
                    a = r.policy.act(run.observe(), deterministic=self.deterministic, rng=self.rng)
                r.move, r.jump = C.ACTIONS[a]
            run.frame(r.move, r.jump)
            moved = moved or (run.state.x, run.state.y) != before
            if run.status == FINISHED:
                r.finish_time = run.frames / C.FPS
            if not r.is_human:
                # same cut-offs as training: no progress for a while, or too long
                if run.max_x > r.best_x + 1e-6:
                    r.best_x, r.stall_frames = run.max_x, 0
                else:
                    r.stall_frames += 1
                if (r.stall_frames >= C.STALL_STEPS * C.FRAME_SKIP
                        or run.frames >= C.MAX_EPISODE_STEPS * C.FRAME_SKIP):
                    r.stuck = True
        if moved:
            self.last_move_frame, self.still_frames = self.frame, 0
        else:
            self.still_frames += 1
        self._update_ranks()
        if self.end_frame is None and self.race_over():
            self.end_frame = self.last_move_frame

    def _snapshot(self, r):
        """Everything one decision changes, so the debug view can step back exactly."""
        run = r.runner
        return {
            "state": run.state.copy(),
            "runner": (run.status, run.frames, run.max_x, run.death_cause),
            "racer": (r.move, r.jump, r.best_x, r.stall_frames, r.stuck, r.finish_time),
            # taken during update(), after the frame counter has already advanced for this frame
            "clock": (self.frame - 1, self.end_frame, self.last_move_frame, self.still_frames, self.go_timer),
            "debug": (self.dbg.current, list(self.dbg.history)),
            "rng": self.rng.bit_generator.state,
        }

    def step_back(self):
        """Debug view: return to just before the previous decision (paused)."""
        r = self.debug_ghost
        if not r or not self.snapshots:
            return
        s = self.snapshots.pop()
        run = r.runner
        run.state = s["state"]
        run.status, run.frames, run.max_x, run.death_cause = s["runner"]
        r.move, r.jump, r.best_x, r.stall_frames, r.stuck, r.finish_time = s["racer"]
        self.frame, self.end_frame, self.last_move_frame, self.still_frames, self.go_timer = s["clock"]
        self.dbg.current, self.dbg.history = s["debug"]
        self.rng.bit_generator.state = s["rng"]
        self.paused = True
        self.step_frames = 0

    @property
    def clock_frame(self):
        """Frame shown on the clock: it stops once every racer has stood still for STILL_FRAMES
        (and resumes if someone moves again), and for good when the race is over."""
        if self.end_frame is not None:
            return self.end_frame
        return self.last_move_frame if self.still_frames >= STILL_FRAMES else self.frame

    def race_over(self):
        """Playing: over when your run ends (finish or death). Spectating: when every ghost is done."""
        if self.debug_ghost:
            return not self.debug_ghost.active
        if self.playing:
            return not self.human.active
        return bool(self.racers) and not any(r.active for r in self.racers)

    def standings(self):
        def key(r):
            if r.status == FINISHED:
                return (0, r.finish_time)
            return (1, -r.runner.progress)
        return sorted(self.racers, key=key)

    def _update_ranks(self):
        if self.frame % RANK_WINDOW:
            return
        for i, r in enumerate(self.standings()):
            if r.rank is not None and i != r.rank and self.frame > 2 * RANK_WINDOW:
                r.rank_change = 1 if i < r.rank else -1
                r.rank_change_until = self.frame + RANK_FLASH_FRAMES
            r.rank = i

    def _camera_target(self):
        if self.debug_ghost:
            return self.debug_ghost
        if self.playing and not self.follow_leader:
            return self.human
        live = [r for r in self.racers if not r.is_human]
        if not live:
            return self.human if self.playing else None
        active = [r for r in live if r.active] or live
        return max(active, key=lambda r: r.runner.state.x)

    def _camera_goal(self, racer):
        if self.debug:
            # agent near the left edge so its whole view window (4 behind, 15 ahead) is on screen
            x = (racer.runner.state.cx - DV.AGENT_COL - 0.5) * self.T
        else:
            x = racer.runner.state.cx * self.T - self.w * 0.38
        return max(0.0, min(x, self.level.width * self.T - self.w))

    # ------------------------------------------------------------ pre-rendering
    def _build_background(self):
        bg = pygame.Surface((self.w, self.play_h))
        for y in range(self.play_h):
            t = y / self.play_h
            c = [int(a + (b - a) * t) for a, b in zip(SKY_TOP, SKY_BOTTOM)]
            pygame.draw.line(bg, c, (0, y), (self.w, y))
        return bg

    def _draw_parallax(self):
        T, cam = self.T, self.cam_x
        # clouds
        for i in range(8):
            base = i * 420 + 90
            x = (base - cam * 0.12) % (self.w + 300) - 150
            y = HUD_H + 40 + (i * 53) % 140
            for dx, dy, r in ((0, 0, 26), (28, -10, 32), (60, 0, 24), (30, 8, 26)):
                pygame.draw.circle(self.screen, CLOUD, (int(x + dx), int(y + dy)), r)
        # hills
        for color, factor, amp, base, period in ((HILL_FAR, 0.25, 70, 0.62, 520), (HILL_NEAR, 0.45, 55, 0.74, 380)):
            pts = [(0, HUD_H + self.play_h)]
            for sx in range(0, self.w + 21, 20):
                wx = sx + cam * factor
                y = base * self.play_h - amp * (0.6 * math.sin(wx / period) + 0.4 * math.sin(wx / (period * 0.43) + 1.7))
                pts.append((sx, HUD_H + y))
            pts.append((self.w, HUD_H + self.play_h))
            pygame.draw.polygon(self.screen, color, pts)

    def _render_level(self, level):
        T = self.T
        surf = pygame.Surface((level.width * T, level.height * T), pygame.SRCALPHA)
        g = level.grid
        for r in range(level.height):
            for c in range(level.width):
                t = g[r, c]
                x, y = c * T, r * T
                if t == C.SOLID:
                    above = r > 0 and g[r - 1, c] == C.SOLID
                    below = r + 1 < level.height and g[r + 1, c] == C.SOLID
                    pygame.draw.rect(surf, DIRT, (x, y, T, T))
                    if (r * 7 + c * 13) % 5 == 0:
                        pygame.draw.circle(surf, DIRT_DARK, (x + T // 3, y + T // 2 + 4), max(2, T // 10))
                    if (r * 11 + c * 5) % 7 == 0:
                        pygame.draw.circle(surf, DIRT_DARK, (x + 2 * T // 3, y + T // 3 + 6), max(2, T // 12))
                    if not above:
                        pygame.draw.rect(surf, GRASS, (x, y, T, T // 4 + 2))
                        pygame.draw.rect(surf, GRASS_DARK, (x, y + T // 4 + 2, T, 3))
                    if not below and r + 1 < level.height:
                        pygame.draw.rect(surf, DIRT_DARK, (x, y + T - 4, T, 4))
                elif t == C.SPIKE:
                    half = T // 2
                    for k in range(2):
                        x0 = x + k * half
                        pts = [(x0 + 1, y + T), (x0 + half // 2, y + half - 2), (x0 + half - 1, y + T)]
                        pygame.draw.polygon(surf, SPIKE_FILL, pts)
                        pygame.draw.polygon(surf, SPIKE_EDGE, pts, 2)
        # finish flag
        gx = int(level.goal_x * T)
        top_row = level.surface_row(int(level.goal_x)) or level.height
        gy = top_row * T
        pygame.draw.rect(surf, (235, 235, 240), (gx - 3, gy - 5 * T, 6, 5 * T))
        pygame.draw.circle(surf, (250, 210, 60), (gx, gy - 5 * T), 7)
        pygame.draw.polygon(surf, (235, 60, 70), [(gx + 3, gy - 5 * T + 6), (gx + 3 + int(1.6 * T), gy - 4.4 * T),
                                                  (gx + 3, gy - 3.8 * T)])
        sq = max(6, T // 5)
        for i in range(int(level.width * T - gx) // sq + 1):
            for j in range(2):
                if (i + j) % 2 == 0:
                    pygame.draw.rect(surf, (30, 30, 30), (gx + i * sq, gy - (j + 1) * sq, sq, sq))
                else:
                    pygame.draw.rect(surf, (245, 245, 245), (gx + i * sq, gy - (j + 1) * sq, sq, sq))
        return surf

    def _render_minimap(self, level):
        mw = self.w - 24
        surf = pygame.Surface((mw, MINIMAP_H - 10), pygame.SRCALPHA)
        hh = surf.get_height()
        for c in range(level.width):
            sr = level.surface_row(c)
            if sr is None:
                continue
            x0 = int(c / level.width * mw)
            x1 = max(x0 + 1, int((c + 1) / level.width * mw))
            top = int(sr / level.height * hh)
            pygame.draw.rect(surf, (90, 140, 80), (x0, top, x1 - x0, hh - top))
            if (level.grid[:, c] == C.SPIKE).any():
                pygame.draw.rect(surf, (220, 220, 230), (x0, max(0, top - 2), x1 - x0, 2))
        gx = int(level.goal_x / level.width * mw)
        pygame.draw.rect(surf, (235, 60, 70), (gx, 0, 2, hh))
        return surf

    def _build_sprites(self, racer):
        T = self.T
        w, h = int(C.PLAYER_W * T), int(C.PLAYER_H * T)
        rad = max(3, T // 6)
        for kind, alpha in (("live", 255 if racer.is_human else GHOST_ALPHA), ("dead", 45)):
            s = pygame.Surface((w, h), pygame.SRCALPHA)
            pygame.draw.rect(s, (*racer.color, alpha), (0, 0, w, h), border_radius=rad)
            pygame.draw.rect(s, (255, 255, 255, min(255, alpha + 40)), (0, 0, w, h), width=2, border_radius=rad)
            ex, ey, er = int(w * 0.58), int(h * 0.32), max(3, T // 9)
            for dx in (0, int(w * 0.24)):
                cx = ex + dx - int(w * 0.1)
                if kind == "dead":
                    d = er
                    pygame.draw.line(s, (40, 40, 40, 160), (cx - d, ey - d), (cx + d, ey + d), 2)
                    pygame.draw.line(s, (40, 40, 40, 160), (cx - d, ey + d), (cx + d, ey - d), 2)
                else:
                    pygame.draw.circle(s, (255, 255, 255, min(255, alpha + 80)), (cx, ey), er)
                    pygame.draw.circle(s, (20, 20, 30, min(255, alpha + 80)), (cx + er // 2, ey), max(1, er // 2))
            racer.sprites[(kind, 1)] = s
            racer.sprites[(kind, -1)] = pygame.transform.flip(s, True, False)
        text = self.font_small.render(racer.name, True, (255, 255, 255))
        pad = 5
        pill = pygame.Surface((text.get_width() + 2 * pad, text.get_height() + 2), pygame.SRCALPHA)
        dark = tuple(int(v * 0.55) for v in racer.color)
        pygame.draw.rect(pill, (*dark, 215), pill.get_rect(), border_radius=pill.get_height() // 2)
        pill.blit(text, (pad, 1))
        if not racer.is_human:
            pill.set_alpha(GHOST_LABEL_ALPHA)
        racer.label_surf = pill
        faded = pill.copy()
        faded.set_alpha(60)
        racer.label_faded = faded

    # ------------------------------------------------------------ drawing
    def draw(self):
        scr = self.screen
        target = self._camera_target()
        if target:
            self.cam_x += (self._camera_goal(target) - self.cam_x) * (0.5 if self.debug else 0.12)
        cam = int(self.cam_x)

        scr.blit(self.bg, (0, HUD_H))
        self._draw_parallax()
        scr.blit(self.level_surf, (0, HUD_H), area=pygame.Rect(cam, 0, self.w, self.play_h))
        dbg = self.debug_ghost
        if dbg:
            self.dbg.draw_world_overlay(scr, dbg.runner, self.T, cam, HUD_H, self.show_heat)

        racers = self.racers
        # draw dead/stuck first, then live ghosts, then the human on top
        order = sorted(racers, key=lambda r: (r.is_human, r.active))
        on_screen = []
        for r in order:
            s = r.runner.state
            kind = "live" if r.status in (RUNNING, FINISHED) else "dead"
            sprite = r.sprites[(kind, 1 if s.facing >= 0 else -1)]
            px = int(s.x * self.T) - cam
            py = int(s.y * self.T) + HUD_H
            if px < -self.T or px > self.w + self.T:
                continue
            scr.blit(sprite, (px, py))
            on_screen.append((r, kind, px + sprite.get_width() // 2, py))
        # name tags: you first, then the leading ghosts; a crowded tag lifts a little, then hides
        on_screen.sort(key=lambda t: (not t[0].is_human, not t[0].active, -t[0].runner.state.x))
        label_rects = []
        for r, kind, cx, py in on_screen:
            label = r.label_surf if kind == "live" else r.label_faded
            rect = pygame.Rect(cx - label.get_width() // 2, py - label.get_height() - 4,
                               label.get_width(), label.get_height())
            lifts = 0
            while rect.collidelist(label_rects) != -1 and lifts < MAX_LABEL_LIFTS:
                rect.y -= rect.height + 1
                lifts += 1
            if rect.collidelist(label_rects) != -1 and not r.is_human:
                continue
            label_rects.append(rect)
            scr.blit(label, rect)

        self._draw_minimap()
        self._draw_hud()
        if dbg:
            self.dbg.draw_panels(scr, self.w, self.h, dbg, self.paused)
        elif self.show_board and racers:
            self._draw_board()
        self._draw_center_text()
        if self.show_help:
            self._draw_help()

    def _draw_hud(self):
        scr = self.screen
        pygame.draw.rect(scr, HUD_BG, (0, 0, self.w, HUD_H))
        stopped = self.end_frame is not None
        t = self.clock_frame / C.FPS
        if self.debug:
            g = self.debug_ghost
            title = f"DEBUG  ghost {g.name}" if g else "DEBUG  no ghosts loaded"
            parts = [f"seed {self.seed}", f"difficulty {self.difficulty:g}", f"time {t:.1f}s"]
            if self.speed_idx:
                parts.append(f"speed {SPEEDS[self.speed_idx]:g}x")
            if not self.deterministic:
                parts.append("stochastic")
            scr.blit(self.font.render(title, True, HUD_TEXT), (14, 6))
            scr.blit(self.font.render("   ".join(parts), True, HUD_DIM), (14, 28))
            hint = self.font.render("Tab ghost  Space pause  I exit", True, HUD_DIM)
            scr.blit(hint, (self.w - hint.get_width() - 14, 6))
            return
        ghosts = f"{len(self.ghosts)} ghosts" if self.ghosts else "no ghosts loaded"
        diff = f"difficulty {self.difficulty:g} {difficulty_name(self.difficulty)}"
        if self.difficulty > 1.0:
            diff += " (beyond training)"
        parts = [f"seed {self.seed}", diff, f"time {t:.1f}s" + (" final" if stopped else ""), ghosts]
        if not self.playing:
            parts.append("spectating")
        if not self.deterministic:
            parts.append("stochastic")
        if self.speed_idx:
            parts.append(f"speed {SPEEDS[self.speed_idx]:g}x")
        left = "   ".join(parts)
        scr.blit(self.font.render("ProcGen RL", True, HUD_TEXT), (14, 6))
        scr.blit(self.font.render(left, True, HUD_DIM), (14, 28))
        hint = self.font.render("H help   N new level   R restart", True, HUD_DIM)
        scr.blit(hint, (self.w - hint.get_width() - 14, 6))

    def _draw_minimap(self):
        scr = self.screen
        y0 = HUD_H + self.play_h
        pygame.draw.rect(scr, HUD_BG, (0, y0, self.w, MINIMAP_H))
        scr.blit(self.minimap, (12, y0 + 5))
        mw, lw = self.minimap.get_width(), self.level.width
        vx0 = int(self.cam_x / self.T / lw * mw)
        vx1 = int((self.cam_x + self.w) / self.T / lw * mw)
        pygame.draw.rect(scr, (255, 255, 255), (12 + vx0, y0 + 3, vx1 - vx0, MINIMAP_H - 6), 1)
        for r in sorted(self.racers, key=lambda r: r.is_human):
            x = 12 + int(r.runner.state.cx / lw * mw)
            rad = 5 if r.is_human else 4
            col = r.color if r.status in (RUNNING, FINISHED) else tuple(v // 2 for v in r.color)
            pygame.draw.circle(scr, col, (x, y0 + MINIMAP_H // 2), rad)
            pygame.draw.circle(scr, (255, 255, 255), (x, y0 + MINIMAP_H // 2), rad, 1)

    def _draw_board(self):
        """Compact standings: order shows who is ahead (no place numbers); an arrow flashes
        next to anyone who just moved up or down."""
        rows = self.standings()
        line_h = 16
        width = 164
        height = 10 + line_h * len(rows)
        panel = pygame.Surface((width, height), pygame.SRCALPHA)
        pygame.draw.rect(panel, PANEL_BG[:3], panel.get_rect(), border_radius=8)
        for i, r in enumerate(rows):
            y = 5 + i * line_h
            out = r.status in (DEAD, STUCK)
            if r.is_human:
                pygame.draw.rect(panel, (255, 255, 255, 28), (3, y, width - 6, line_h), border_radius=4)
            col = tuple(v // 2 for v in r.color) if out else r.color
            pygame.draw.circle(panel, col, (11, y + line_h // 2), 4)
            name = self.font_tiny.render(r.name, True, HUD_DIM if out else HUD_TEXT)
            panel.blit(name, (20, y + (line_h - name.get_height()) // 2))
            if r.rank_change and self.frame < r.rank_change_until:
                ax, ay, s = 90, y + line_h // 2, 4
                if r.rank_change > 0:
                    pts, c = [(ax - s, ay + s - 1), (ax + s, ay + s - 1), (ax, ay - s)], RANK_UP
                else:
                    pts, c = [(ax - s, ay - s + 1), (ax + s, ay - s + 1), (ax, ay + s)], RANK_DOWN
                pygame.draw.polygon(panel, c, pts)
            if r.status == FINISHED:
                info = f"{r.finish_time:.1f}s"
            elif out:
                info = "out"
            else:
                info = f"{r.runner.progress * 100:.0f}%"
            txt = self.font_tiny.render(info, True, HUD_DIM if out else HUD_TEXT)
            panel.blit(txt, (width - txt.get_width() - 8, y + (line_h - txt.get_height()) // 2))
        self.screen.blit(panel, (self.w - width - 10, HUD_H + 10))

    def _draw_center_text(self):
        cx, cy = self.w // 2, HUD_H + self.play_h // 2 - 40
        if self.countdown > 0:
            n = math.ceil(self.countdown / (COUNTDOWN_FRAMES / 3))
            self._shadow_text(str(n), self.font_big, cx, cy)
            return
        if self.go_timer > 0:
            self._shadow_text("GO!", self.font_big, cx, cy)
        g = self.debug_ghost
        if g:
            if not g.active:
                run = g.runner
                if g.status == FINISHED:
                    why = f"Finished in {g.finish_time:.2f}s"
                elif g.status == STUCK:
                    why = "Stuck: no progress for 6 s"
                else:
                    why = "Spiked" if run.death_cause == "spike" else "Fell"
                self._shadow_text(why, self.font_mid, cx, cy)
                self._shadow_text("R replay   N new level   Tab other ghost", self.font, cx, cy + 40)
            elif self.paused:
                self._shadow_text("paused", self.font_mid, cx, HUD_H + 40)
            return
        if self.playing:
            h = self.human
            if h.status == DEAD:
                why = "Spiked!" if h.runner.death_cause == "spike" else "You fell!"
                self._shadow_text(why, self.font_mid, cx, cy)
                self._shadow_text("R retry   N new level", self.font, cx, cy + 40)
            elif h.status == FINISHED:
                place = 1 + sum(1 for g in self.racers if not g.is_human and g.status == FINISHED
                                and g.finish_time < h.finish_time)
                self._shadow_text(f"Finished in {h.finish_time:.2f}s  (place {place})", self.font_mid, cx, cy)
                self._shadow_text("R retry   N new level", self.font, cx, cy + 40)

    def _shadow_text(self, text, font, cx, cy):
        for color, off in (((20, 20, 30), 3), ((255, 255, 255), 0)):
            s = font.render(text, True, color)
            self.screen.blit(s, (cx - s.get_width() // 2 + off, cy - s.get_height() // 2 + off))

    def _draw_help(self):
        line_h = 22
        width, height = 560, 60 + line_h * len(HELP_LINES)
        panel = pygame.Surface((width, height), pygame.SRCALPHA)
        pygame.draw.rect(panel, (16, 20, 30, 225), panel.get_rect(), border_radius=12)
        panel.blit(self.font_mid.render("Controls", True, HUD_TEXT), (20, 10))
        for i, (k, v) in enumerate(HELP_LINES):
            y = 54 + i * line_h
            panel.blit(self.font.render(k, True, HUD_TEXT), (24, y))
            panel.blit(self.font.render(v, True, HUD_DIM), (210, y))
        self.screen.blit(panel, ((self.w - width) // 2, HUD_H + (self.play_h - height) // 2))

    # ------------------------------------------------------------ main loop
    def handle_key(self, key):
        if key == pygame.K_ESCAPE:
            return False
        if key == pygame.K_r:
            self.restart()
        elif key == pygame.K_n:
            self.seed = self._random_seed()
            self.load_level()
        elif key in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
            self.difficulty = step_difficulty(self.difficulty, key == pygame.K_RIGHTBRACKET)
            self.load_level()
        elif key == pygame.K_p:
            self.playing = not self.playing
            self.follow_leader = not self.playing
        elif key == pygame.K_g:
            self.show_ghosts = not self.show_ghosts
        elif key == pygame.K_c:
            self.follow_leader = not self.follow_leader
        elif key == pygame.K_s:
            self.deterministic = not self.deterministic
        elif key == pygame.K_f:
            self.speed_idx = (self.speed_idx + 1) % len(SPEEDS)
        elif key == pygame.K_l:
            self.show_board = not self.show_board
        elif key == pygame.K_i and self.ghosts:
            self.debug = not self.debug
            self.paused = False
            self._apply_layout()
            self.restart()
        elif key == pygame.K_TAB and self.debug_ghost:
            shift = pygame.key.get_mods() & pygame.KMOD_SHIFT
            self.dbg_index = (self.dbg_index + (-1 if shift else 1)) % len(self.ghosts)
            self.restart()
        elif key == pygame.K_SPACE and self.debug_ghost:
            self.paused = not self.paused
            self.step_frames = 0
        elif key == pygame.K_RIGHT and self.debug_ghost:
            self.paused = True
            self.step_frames = C.FRAME_SKIP
        elif key == pygame.K_LEFT and self.debug_ghost:
            self.step_back()
        elif key == pygame.K_o and self.debug_ghost:
            self.show_heat = not self.show_heat
        elif key == pygame.K_h:
            self.show_help = not self.show_help
        return True

    def run(self, max_frames=None, screenshot=None, auto_keys=None):
        acc = 0.0
        step_dt = 1.0 / C.FPS
        frames = 0
        running = True
        while running:
            dt = self.clock.tick(C.FPS) / 1000.0
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.KEYDOWN:
                    running = self.handle_key(e.key)
            keys = pygame.key.get_pressed() if auto_keys is None else auto_keys
            if max_frames is not None:
                dt = step_dt  # headless test runs: exactly one tick per frame
            acc = min(acc + dt * SPEEDS[self.speed_idx], 8 * step_dt)
            while acc >= step_dt:
                self.update(keys)
                acc -= step_dt
            self.draw()
            pygame.display.flip()
            frames += 1
            if max_frames is not None and frames >= max_frames:
                break
        if screenshot:
            pygame.image.save(self.screen, screenshot)
        pygame.quit()


class _AutoKeys:
    """Fake key state for headless runs: hold right and jump."""

    def __getitem__(self, k):
        return k in (pygame.K_RIGHT, pygame.K_SPACE)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ghosts", default="ghosts", help="folder of ghost .npz files (or a run folder)")
    p.add_argument("--only", default=None, help="comma separated ghost labels to show, e.g. 1h,10h,50h")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--difficulty", type=float, default=0.5,
                   help=f"0-1 is the training range; up to {MAX_DIFFICULTY:g} for expert levels beyond training")
    p.add_argument("--spectate", action="store_true", help="start without the human player")
    p.add_argument("--stochastic", action="store_true", help="ghosts sample actions instead of argmax")
    p.add_argument("--debug", nargs="?", const="", default=None, metavar="LABEL",
                   help="start in the debug view with one ghost (e.g. 300h; default the most trained)")
    p.add_argument("--scale", type=float, default=1.0, help="tile size multiplier (0.75 for small screens)")
    p.add_argument("--scaled", action="store_true", help="let SDL scale the window to the display")
    p.add_argument("--frames", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--screenshot", default=None, help=argparse.SUPPRESS)
    args = p.parse_args()
    demo = Demo(args)
    demo.run(max_frames=args.frames, screenshot=args.screenshot,
             auto_keys=_AutoKeys() if args.frames else None)
    sys.exit(0)


if __name__ == "__main__":
    main()
