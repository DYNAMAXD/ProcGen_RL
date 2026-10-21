"""Playable demo: race against ghosts of the agent from different training stages.

    python demo.py                         # play with ghosts from ./ghosts
    python demo.py --ghosts runs/main/ghosts --difficulty 0.6
    python demo.py --spectate              # watch the ghosts only

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

from procgen_rl import config as C
from procgen_rl.level import generate_level
from procgen_rl.policy import load_ghosts
from procgen_rl.sim import DEAD, FINISHED, RUNNING, Runner

BASE_TILE = 40
VIEW_COLS = 32
HUD_H = 52
MINIMAP_H = 26
COUNTDOWN_FRAMES = 120
SPEEDS = (1, 2, 4)

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
STUCK = "stuck"

HELP_LINES = [
    ("Left/Right or A/D", "run"),
    ("Space / Up / W", "jump (hold for higher)"),
    ("R", "restart this level"),
    ("N", "new random level"),
    ("[  ]", "difficulty down / up (same seed)"),
    ("P", "toggle playing / spectating"),
    ("G", "toggle ghosts"),
    ("C", "camera: you / leading ghost"),
    ("S", "ghosts: deterministic / stochastic"),
    ("F", "simulation speed 1x / 2x / 4x"),
    ("L", "toggle leaderboard"),
    ("H", "toggle this help"),
    ("Esc", "quit"),
]


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
        self.T = max(16, int(BASE_TILE * args.scale))
        self.w = VIEW_COLS * self.T
        self.play_h = C.LEVEL_HEIGHT * self.T
        self.h = HUD_H + self.play_h + MINIMAP_H
        flags = pygame.SCALED if args.scaled else 0
        self.screen = pygame.display.set_mode((self.w, self.h), flags)
        self.clock = pygame.time.Clock()
        self.font = self._font(18)
        self.font_small = self._font(15, bold=True)
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
        for r in self.ghosts + [self.human]:
            self._build_sprites(r)

        self.bg = self._build_background()
        self.load_level()

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
        rs = list(self.ghosts) if self.show_ghosts else []
        if self.playing:
            rs.append(self.human)
        return rs

    def load_level(self):
        self.level = generate_level(self.seed, self.difficulty)
        self.level_surf = self._render_level(self.level)
        self.minimap = self._render_minimap(self.level)
        self.restart()

    def restart(self):
        for r in self.ghosts + [self.human]:
            r.reset(self.level)
        self.countdown = COUNTDOWN_FRAMES
        self.go_timer = 0
        self.frame = 0
        target = self._camera_target()
        self.cam_x = self._camera_goal(target) if target else 0.0

    # ------------------------------------------------------------ simulation
    def update(self, keys):
        if self.countdown > 0:
            self.countdown -= 1
            if self.countdown == 0:
                self.go_timer = 40
            return
        if self.go_timer > 0:
            self.go_timer -= 1
        self.frame += 1
        for r in self.racers:
            if not r.active:
                continue
            run = r.runner
            if r.is_human:
                r.move = (keys[pygame.K_RIGHT] or keys[pygame.K_d]) - (keys[pygame.K_LEFT] or keys[pygame.K_a])
                r.jump = int(keys[pygame.K_SPACE] or keys[pygame.K_UP] or keys[pygame.K_w])
            elif run.frames % C.FRAME_SKIP == 0:
                a = r.policy.act(run.observe(), deterministic=self.deterministic, rng=self.rng)
                r.move, r.jump = C.ACTIONS[a]
            run.frame(r.move, r.jump)
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

    def _camera_target(self):
        if self.playing and not self.follow_leader:
            return self.human
        live = [r for r in self.racers if not r.is_human]
        if not live:
            return self.human if self.playing else None
        active = [r for r in live if r.active] or live
        return max(active, key=lambda r: r.runner.state.x)

    def _camera_goal(self, racer):
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
        for kind, alpha in (("live", 255 if racer.is_human else 135), ("dead", 60)):
            s = pygame.Surface((w, h), pygame.SRCALPHA)
            pygame.draw.rect(s, (*racer.color, alpha), (0, 0, w, h), border_radius=rad)
            pygame.draw.rect(s, (255, 255, 255, min(255, alpha + 60)), (0, 0, w, h), width=2, border_radius=rad)
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
        racer.label_surf = pill

    # ------------------------------------------------------------ drawing
    def draw(self):
        scr = self.screen
        target = self._camera_target()
        if target:
            self.cam_x += (self._camera_goal(target) - self.cam_x) * 0.12
        cam = int(self.cam_x)

        scr.blit(self.bg, (0, HUD_H))
        self._draw_parallax()
        scr.blit(self.level_surf, (0, HUD_H), area=pygame.Rect(cam, 0, self.w, self.play_h))

        racers = self.racers
        # draw dead/stuck first, then live ghosts, then the human on top
        order = sorted(racers, key=lambda r: (r.is_human, r.active))
        label_rects = []
        for r in order:
            s = r.runner.state
            kind = "live" if r.status in (RUNNING, FINISHED) else "dead"
            sprite = r.sprites[(kind, 1 if s.facing >= 0 else -1)]
            px = int(s.x * self.T) - cam
            py = int(s.y * self.T) + HUD_H
            if px < -self.T or px > self.w + self.T:
                continue
            scr.blit(sprite, (px, py))
            label = r.label_surf
            lx = px + sprite.get_width() // 2 - label.get_width() // 2
            ly = py - label.get_height() - 4
            rect = pygame.Rect(lx, ly, label.get_width(), label.get_height())
            while rect.collidelist(label_rects) != -1 and rect.y > HUD_H - 40:
                rect.y -= rect.height + 1
            label_rects.append(rect)
            if kind == "dead":
                faded = label.copy()
                faded.set_alpha(110)
                scr.blit(faded, rect)
            else:
                scr.blit(label, rect)

        self._draw_minimap()
        self._draw_hud()
        if self.show_board and racers:
            self._draw_board()
        self._draw_center_text()
        if self.show_help:
            self._draw_help()

    def _draw_hud(self):
        scr = self.screen
        pygame.draw.rect(scr, HUD_BG, (0, 0, self.w, HUD_H))
        t = self.frame / C.FPS
        mode = "playing" if self.playing else "spectating"
        ghosts = f"{len(self.ghosts)} ghosts" if self.ghosts else "no ghosts loaded"
        left = (f"seed {self.seed}   difficulty {self.difficulty:.1f}   time {t:5.1f}s   "
                f"{mode}   {ghosts}   {'stochastic' if not self.deterministic else 'deterministic'}"
                f"   speed {SPEEDS[self.speed_idx]}x")
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
        def key(r):
            if r.status == FINISHED:
                return (0, r.finish_time)
            return (1, -r.runner.progress)
        rows = sorted(self.racers, key=key)
        line_h = 21
        width = 250
        height = 30 + line_h * len(rows)
        panel = pygame.Surface((width, height), pygame.SRCALPHA)
        pygame.draw.rect(panel, PANEL_BG, panel.get_rect(), border_radius=10)
        panel.blit(self.font_small.render("LEADERBOARD", True, HUD_DIM), (12, 7))
        for i, r in enumerate(rows):
            y = 28 + i * line_h
            pygame.draw.circle(panel, r.color, (18, y + 9), 6)
            name = self.font_small.render(f"{i + 1}. {r.name}", True, HUD_TEXT)
            panel.blit(name, (30, y))
            if r.status == FINISHED:
                info = f"{r.finish_time:.1f}s"
            else:
                info = f"{r.runner.progress * 100:3.0f}%"
                if r.status == DEAD:
                    info += " died"
                elif r.status == STUCK:
                    info += " stuck"
            txt = self.font_small.render(info, True, HUD_DIM if r.status in (DEAD, STUCK) else HUD_TEXT)
            panel.blit(txt, (width - txt.get_width() - 12, y))
        self.screen.blit(panel, (self.w - width - 12, HUD_H + 12))

    def _draw_center_text(self):
        cx, cy = self.w // 2, HUD_H + self.play_h // 2 - 40
        if self.countdown > 0:
            n = math.ceil(self.countdown / (COUNTDOWN_FRAMES / 3))
            self._shadow_text(str(n), self.font_big, cx, cy)
            return
        if self.go_timer > 0:
            self._shadow_text("GO!", self.font_big, cx, cy)
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
        line_h = 26
        width, height = 520, 60 + line_h * len(HELP_LINES)
        panel = pygame.Surface((width, height), pygame.SRCALPHA)
        pygame.draw.rect(panel, (16, 20, 30, 225), panel.get_rect(), border_radius=12)
        panel.blit(self.font_mid.render("Controls", True, HUD_TEXT), (20, 10))
        for i, (k, v) in enumerate(HELP_LINES):
            y = 54 + i * line_h
            panel.blit(self.font.render(k, True, HUD_TEXT), (24, y))
            panel.blit(self.font.render(v, True, HUD_DIM), (220, y))
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
            step = 0.1 if key == pygame.K_RIGHTBRACKET else -0.1
            self.difficulty = round(min(1.0, max(0.0, self.difficulty + step)), 2)
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
    p.add_argument("--difficulty", type=float, default=0.5)
    p.add_argument("--spectate", action="store_true", help="start without the human player")
    p.add_argument("--stochastic", action="store_true", help="ghosts sample actions instead of argmax")
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
