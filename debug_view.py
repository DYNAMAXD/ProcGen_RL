"""Debug view for the demo: one ghost plays alone while its inputs and decisions are shown.

Panels (right of and below the game):
- the 840 tile inputs as the three 14 x 20 layers the agent receives, plus a map of
  which tiles the current decision depends on;
- the 7 numbers the agent gets about itself;
- what share of the decision depends on each input group (flip-one-input test,
  see NumpyPolicy.sensitivity);
- the network itself (847 -> 256 -> 256 -> 6), live: every hidden neuron's activation
  and the strongest connections behind the chosen action;
- the 6 output probabilities, how often each action was chosen so far, and the
  critic's estimate of future reward, over time.

Used by demo.py (key I or --debug). Needs only numpy + pygame.
"""

import numpy as np
import pygame

from procgen_rl import config as C

GAME_COLS = 22              # columns of level visible next to the panels
GAME_TILE = 30              # tile size in the debug layout (times --scale)
PANEL_W = 780
DECISION_H = 222            # panel under the game view
MIN_H = 780
AGENT_COL = 5               # camera keeps the agent in this column, so the whole view window is visible
HISTORY = 160               # decisions kept for the timeline
HEAT_TOP = 15               # the game view shades this many most influential tiles
HEAT_MIN = 0.05             # ... if flipping them changes the decision by at least this much
HEAT_FULL = 0.5             # ... at full strength from this much change

BG = (12, 15, 22)
BOX = (24, 29, 42)
LINE = (52, 60, 80)
TEXT = (232, 236, 242)
DIM = (146, 154, 170)
OFF = (36, 42, 58)
SOLID_ON = (236, 236, 236)
SPIKE_ON = (232, 96, 84)
GOAL_ON = (250, 210, 60)
HEAT = (255, 140, 30)
GOOD = (96, 204, 128)
BAD = (232, 96, 84)
VALUE = (120, 180, 255)

N_TILES = C.VIEW_H * C.VIEW_W
SCALAR_NAMES = ("vx  run speed", "vy  fall speed", "on_ground", "can_jump", "jumping", "x in tile", "y in tile")
GROUPS = (
    ("ground + walls", slice(0, N_TILES)),
    ("spikes", slice(N_TILES, 2 * N_TILES)),
    ("finish line", slice(2 * N_TILES, 3 * N_TILES)),
    ("own speed", slice(3 * N_TILES, 3 * N_TILES + 2)),
    ("jump state", slice(3 * N_TILES + 2, 3 * N_TILES + 5)),
    ("position in tile", slice(3 * N_TILES + 5, 3 * N_TILES + 7)),
)
ACTION_COLORS = ((150, 150, 160), (190, 120, 255), (96, 204, 128), (120, 180, 255), (240, 120, 200), (255, 190, 60))


def _font(size, bold=False):
    return pygame.font.SysFont("consolas,dejavusansmono,couriernew", size, bold=bold)


class Decision:
    """Everything the policy computed for one decision."""

    def __init__(self, policy, obs, action=None):
        self.obs = obs
        hidden, self.logits, self.probs = policy.trace(obs)
        self.hidden = hidden
        self.action = int(np.argmax(self.probs)) if action is None else action
        self.value = policy.value(obs)
        self.sens = policy.sensitivity(obs)
        total = float(self.sens.sum()) or 1.0
        self.group_share = [float(self.sens[s].sum()) / total for _, s in GROUPS]
        # per tile: the largest effect of changing that tile in any layer
        self.tile_sens = self.sens[:3 * N_TILES].reshape(3, C.VIEW_H, C.VIEW_W).max(axis=0)
        self.edges = self._edges(policy, obs)

    def _edges(self, policy, obs):
        """Strongest contributions behind the chosen action: (layer, from, to, strength) tuples.

        Layer 2: hidden-2 neuron -> chosen output, top 6 by |weight x activation|.
        Layer 1: hidden-1 neuron -> each of those, top 2.
        Layer 0: input -> each of those hidden-1 neurons, top 1 among non-zero inputs.
        """
        if len(self.hidden) != 2:
            return []
        h1, h2 = self.hidden
        W0, W1, W2 = policy.weights[:3]
        edges = []
        c2 = W2[self.action] * h2
        top2 = np.argsort(-np.abs(c2))[:6]
        m2 = np.abs(c2[top2]).max() or 1.0
        nz = np.flatnonzero(obs)
        for j in top2:
            edges.append((2, int(j), self.action, float(c2[j] / m2)))
            c1 = W1[j] * h1
            top1 = np.argsort(-np.abs(c1))[:2]
            m1 = np.abs(c1[top1]).max() or 1.0
            for k in top1:
                edges.append((1, int(k), int(j), float(c1[k] / m1)))
                if len(nz):
                    c0 = W0[k, nz] * obs[nz]
                    i = int(nz[np.argmax(np.abs(c0))])
                    edges.append((0, i, int(k), 1.0 if c0.max() >= -c0.min() else -1.0))
        return edges


class DebugView:
    def __init__(self):
        self.font = _font(14)
        self.font_b = _font(14, bold=True)
        self.font_s = _font(12)
        self.font_h = _font(15, bold=True)
        self.reset()

    def reset(self):
        self.current = None
        self.history = []                 # (value, confidence, action)
        self.counts = np.zeros(len(C.ACTIONS), dtype=int)

    def decide(self, policy, obs, deterministic=True, rng=None):
        """Run the policy with full tracing; returns the action index."""
        action = None
        if not deterministic:
            hidden, logits, probs = policy.trace(obs)
            action = int((rng or np.random.default_rng()).choice(len(probs), p=probs))
        d = Decision(policy, obs, action)
        self.current = d
        self.counts[d.action] += 1
        self.history.append((d.value, float(d.probs[d.action]), d.action))
        del self.history[:-HISTORY]
        return d.action

    # ------------------------------------------------------------ on the game view
    def draw_world_overlay(self, screen, runner, T, cam, top, show_heat):
        """Outline the agent's observation window on the level and shade the tiles it depends on."""
        c0, r0 = runner.obs_builder.window_origin(runner.state)
        x0, y0 = c0 * T - cam, r0 * T + top
        if show_heat and self.current is not None:
            heat = self.current.tile_sens
            surf = pygame.Surface((C.VIEW_W * T, C.VIEW_H * T), pygame.SRCALPHA)
            for i in np.argsort(heat, axis=None)[::-1][:HEAT_TOP]:
                r, c = divmod(int(i), C.VIEW_W)
                if heat[r, c] > HEAT_MIN:
                    a = min(1.0, heat[r, c] / HEAT_FULL)
                    surf.fill((*HEAT, int(60 + 140 * a)), (c * T, r * T, T, T))
            screen.blit(surf, (x0, y0))
        rect = pygame.Rect(x0, y0, C.VIEW_W * T, C.VIEW_H * T)
        pygame.draw.rect(screen, (255, 255, 255), rect, 2)
        pygame.draw.rect(screen, (20, 20, 30), rect.inflate(4, 4), 1)
        tag = self.font_s.render("what the agent sees: 20 x 14 tiles   orange: tiles it depends on most", True, (255, 255, 255))
        bg = pygame.Surface((tag.get_width() + 8, tag.get_height() + 2), pygame.SRCALPHA)
        bg.fill((20, 20, 30, 190))
        bg.blit(tag, (4, 1))
        screen.blit(bg, (max(0, x0 + 2), max(top + 2, y0 + 2)))

    # ------------------------------------------------------------ panels
    def draw_panels(self, screen, x0, y_decision, ghost, paused):
        """Right column at x0 (full height) and the decision panel under the game view."""
        w, h = screen.get_size()
        pygame.draw.rect(screen, BG, (x0, 0, w - x0, h))
        pygame.draw.rect(screen, BG, (0, y_decision, x0, h - y_decision))
        d = self.current
        if d is None:
            self._text(screen, "waiting for the first decision ...", x0 + 16, 16, DIM)
            return
        px = x0 + 8
        self._tiles_box(screen, d, px, 6, PANEL_W - 16, 196)
        self._scalars_box(screen, d, px, 208, 380, 168)
        self._groups_box(screen, d, px + 388, 208, PANEL_W - 16 - 388, 168)
        self._network_box(screen, d, px, 382, PANEL_W - 16, 226)
        self._timeline_box(screen, px, 614, PANEL_W - 16, h - 614 - 6)
        self._decision_box(screen, d, 8, y_decision + 6, x0 - 16, h - y_decision - 12, ghost, paused)

    def _box(self, screen, x, y, w, h, title, note=None):
        pygame.draw.rect(screen, BOX, (x, y, w, h), border_radius=6)
        self._text(screen, title, x + 10, y + 6, TEXT, self.font_h)
        if note:
            s = self.font_s.render(note, True, DIM)
            screen.blit(s, (x + w - s.get_width() - 10, y + 8))

    def _text(self, screen, text, x, y, color=TEXT, font=None):
        s = (font or self.font).render(text, True, color)
        screen.blit(s, (x, y))
        return s.get_width()

    def _tiles_box(self, screen, d, x, y, w, h):
        self._box(screen, x, y, w, h, "INPUTS 1-840  tile window", "3 layers x 14 rows x 20 columns, each 0 or 1")
        cell = 9
        gw, gh = C.VIEW_W * cell, C.VIEW_H * cell
        gap = (w - 20 - 4 * gw) // 3
        layers = d.obs[:3 * N_TILES].reshape(3, C.VIEW_H, C.VIEW_W)
        heat = d.tile_sens / max(float(d.tile_sens.max()), HEAT_MIN)
        titles = ("solid  1-280", "spike  281-560", "goal  561-840", "decision depends on")
        colors = (SOLID_ON, SPIKE_ON, GOAL_ON, HEAT)
        for k in range(4):
            gx, gy = x + 10 + k * (gw + gap), y + 48
            self._text(screen, titles[k], gx, y + 28, DIM, self.font_s)
            for r in range(C.VIEW_H):
                for c in range(C.VIEW_W):
                    if k < 3:
                        col = colors[k] if layers[k, r, c] > 0.5 else OFF
                    else:
                        a = min(1.0, heat[r, c])
                        col = tuple(int(o + (hc - o) * a) for o, hc in zip(OFF, HEAT))
                    pygame.draw.rect(screen, col, (gx + c * cell, gy + r * cell, cell - 1, cell - 1))
            # the agent's own tile
            pygame.draw.rect(screen, GOOD, (gx + C.VIEW_BEHIND * cell - 1, gy + C.VIEW_UP * cell - 1,
                                            cell + 1, cell + 1), 2)
        self._text(screen, "green box = the agent's own tile.  orange = flipping this tile would change "
                   "the decision", x + 10, y + h - 20, DIM, self.font_s)

    def _scalars_box(self, screen, d, x, y, w, h):
        self._box(screen, x, y, w, h, "INPUTS 841-847  itself")
        vals = d.obs[3 * N_TILES:]
        bar_x, bar_w = x + 150, w - 220
        for i, (name, v) in enumerate(zip(SCALAR_NAMES, vals)):
            ry = y + 30 + i * 19
            self._text(screen, name, x + 10, ry, TEXT, self.font_s)
            pygame.draw.rect(screen, OFF, (bar_x, ry + 3, bar_w, 11))
            if i < 2:   # signed, -1..1, zero in the middle
                mid = bar_x + bar_w // 2
                v = max(-1.0, min(1.0, float(v)))
                bw = int(abs(v) * bar_w / 2)
                pygame.draw.rect(screen, VALUE, (mid if v >= 0 else mid - bw, ry + 3, bw, 11))
                pygame.draw.line(screen, DIM, (mid, ry + 1), (mid, ry + 15))
            else:
                pygame.draw.rect(screen, VALUE, (bar_x, ry + 3, int(min(1.0, max(0.0, float(v))) * bar_w), 11))
            self._text(screen, f"{float(v):+.2f}", bar_x + bar_w + 8, ry, TEXT, self.font_s)

    def _groups_box(self, screen, d, x, y, w, h):
        self._box(screen, x, y, w, h, "SENSITIVITY BY INPUT GROUP")
        bar_x, bar_w = x + 138, w - 196
        order = np.argsort(d.group_share)[::-1]
        for row, g in enumerate(order):
            ry = y + 30 + row * 19
            share = d.group_share[g]
            self._text(screen, GROUPS[g][0], x + 10, ry, TEXT, self.font_s)
            pygame.draw.rect(screen, OFF, (bar_x, ry + 3, bar_w, 11))
            pygame.draw.rect(screen, HEAT, (bar_x, ry + 3, int(share * bar_w), 11))
            self._text(screen, f"{share * 100:4.0f}%", bar_x + bar_w + 6, ry, TEXT, self.font_s)
        self._text(screen, "flip each input, share of the change", x + 10, y + h - 20, DIM, self.font_s)

    def _network_box(self, screen, d, x, y, w, h):
        sizes = [len(a) for a in d.hidden]
        arch = " -> ".join(str(n) for n in [C.OBS_DIM, *sizes, len(C.ACTIONS)])
        self._box(screen, x, y, w, h, f"NETWORK (actor)  {arch}", "live: brightness = neuron activation")
        top = y + 50
        # input column: combined tile map + scalar strip
        cell = 5
        ix, iy = x + 14, top + 8
        layers = d.obs[:3 * N_TILES].reshape(3, C.VIEW_H, C.VIEW_W)
        for r in range(C.VIEW_H):
            for c in range(C.VIEW_W):
                col = OFF
                if layers[1, r, c] > 0.5:
                    col = SPIKE_ON
                elif layers[0, r, c] > 0.5:
                    col = SOLID_ON
                elif layers[2, r, c] > 0.5:
                    col = GOAL_ON
                pygame.draw.rect(screen, col, (ix + c * cell, iy + r * cell, cell, cell))
        sy = iy + C.VIEW_H * cell + 8
        scal = d.obs[3 * N_TILES:]
        for i, v in enumerate(scal):
            a = min(1.0, abs(float(v)))
            pygame.draw.rect(screen, tuple(int(o + (t - o) * a) for o, t in zip(OFF, VALUE)),
                             (ix + i * 14, sy, 12, 12))
        self._text(screen, f"input {C.OBS_DIM}", ix, top - 20, DIM, self.font_s)

        def input_pos(i):
            if i < 3 * N_TILES:
                t = i % N_TILES
                return ix + (t % C.VIEW_W) * cell + cell // 2, iy + (t // C.VIEW_W) * cell + cell // 2
            return ix + (i - 3 * N_TILES) * 14 + 6, sy + 6

        # hidden layers as 16 x 16 grids
        grids = []
        hc = 9
        gx = x + 190
        for li, act in enumerate(d.hidden):
            side = int(np.ceil(np.sqrt(len(act))))
            peak = max(float(act.max()), 1e-6)
            for n, v in enumerate(act):
                a = float(v) / peak
                col = tuple(int(o + (t - o) * a) for o, t in zip(OFF, (255, 255, 255))) if v > 0 else (30, 34, 46)
                pygame.draw.rect(screen, col, (gx + (n % side) * hc, top + (n // side) * hc, hc - 1, hc - 1))
            on = int((act > 0).sum())
            self._text(screen, f"hidden {li + 1}: {on}/{len(act)} active", gx, top - 20, DIM, self.font_s)
            grids.append((gx, side))
            gx += side * hc + 60

        def neuron_pos(layer, n):
            gx0, side = grids[layer]
            return gx0 + (n % side) * hc + hc // 2, top + (n // side) * hc + hc // 2

        # outputs
        ox = gx + 10
        oy = top + 4
        out_pos = []
        for a, name in enumerate(C.ACTION_NAMES):
            ry = oy + a * 24
            chosen = a == d.action
            pygame.draw.circle(screen, GOOD if chosen else OFF, (ox, ry + 8), 7)
            self._text(screen, f"{name:<10} {d.probs[a] * 100:5.1f}%", ox + 12, ry,
                       TEXT if chosen else DIM, self.font_b if chosen else self.font_s)
            out_pos.append((ox - 7, ry + 8))
        self._text(screen, "output 6 (softmax)", ox - 8, top - 20, DIM, self.font_s)

        # strongest contributions behind the chosen action
        for layer, src, dst, s in d.edges:
            if layer == 0:
                p0, p1 = input_pos(src), neuron_pos(0, dst)
            elif layer == 1:
                p0, p1 = neuron_pos(0, src), neuron_pos(1, dst)
            else:
                p0, p1 = neuron_pos(1, src), out_pos[dst]
            col = GOOD if s > 0 else BAD
            pygame.draw.line(screen, col, p0, p1, 1 + int(2 * min(1.0, abs(s))))
        self._text(screen, "lines: strongest connections behind the chosen action "
                   "(green pushes it up, red pushes it down)", x + 10, y + h - 20, DIM, self.font_s)

    def _timeline_box(self, screen, x, y, w, h):
        self._box(screen, x, y, w, h, "OVER TIME", f"last {HISTORY} decisions")
        gx, gy, gw, gh = x + 50, y + 30, w - 64, h - 62
        pygame.draw.rect(screen, (18, 22, 32), (gx, gy, gw, gh))
        n = len(self.history)
        if n < 2:
            return
        step = gw / (HISTORY - 1)
        vals = [v for v, _, _ in self.history if v is not None]
        lo, hi = min(-1.0, min(vals, default=0.0)), max(6.0, max(vals, default=0.0))
        if vals:
            pts = [(gx + i * step, gy + gh - (v - lo) / (hi - lo) * gh) for i, (v, _, _) in enumerate(self.history)]
            zero = gy + gh - (0 - lo) / (hi - lo) * gh
            pygame.draw.line(screen, LINE, (gx, zero), (gx + gw, zero))
            pygame.draw.lines(screen, VALUE, False, pts, 2)
            self._text(screen, f"{hi:.0f}", x + 10, gy - 6, DIM, self.font_s)
            self._text(screen, f"{lo:.0f}", x + 10, gy + gh - 8, DIM, self.font_s)
        conf = [(gx + i * step, gy + gh - c * gh) for i, (_, c, _) in enumerate(self.history)]
        pygame.draw.lines(screen, HEAT, False, conf, 1)
        for i, (_, _, a) in enumerate(self.history):
            pygame.draw.rect(screen, ACTION_COLORS[a], (gx + i * step, gy + gh + 3, max(2, step), 5))
        lx = x + 10
        lx += self._text(screen, "blue: critic's expected future reward   ", lx, y + h - 22, VALUE, self.font_s)
        lx += self._text(screen, "orange: confidence in chosen action   ", lx, y + h - 22, HEAT, self.font_s)
        self._text(screen, "strip: action taken", lx, y + h - 22, DIM, self.font_s)

    def _decision_box(self, screen, d, x, y, w, h, ghost, paused):
        title = f"OUTPUT  ghost {ghost.name}"
        self._box(screen, x, y, w, h, title, "PAUSED  (Right: one decision)" if paused else None)
        half = w // 2
        bar_x, bar_w = x + 110, half - 170
        self._text(screen, "probability now", x + 10, y + 28, DIM, self.font_s)
        for a, name in enumerate(C.ACTION_NAMES):
            ry = y + 46 + a * 22
            chosen = a == d.action
            self._text(screen, name, x + 10, ry, TEXT if chosen else DIM, self.font_b if chosen else self.font_s)
            pygame.draw.rect(screen, OFF, (bar_x, ry + 3, bar_w, 13))
            pygame.draw.rect(screen, GOOD if chosen else DIM, (bar_x, ry + 3, int(d.probs[a] * bar_w), 13))
            self._text(screen, f"{d.probs[a] * 100:5.1f}%", bar_x + bar_w + 6, ry, TEXT, self.font_s)
        total = max(1, int(self.counts.sum()))
        ux = x + half + 10
        self._text(screen, f"chosen so far ({total} decisions)", ux, y + 28, DIM, self.font_s)
        bar_x2, bar_w2 = ux + 100, half - 170
        for a, name in enumerate(C.ACTION_NAMES):
            ry = y + 46 + a * 22
            share = self.counts[a] / total
            self._text(screen, name, ux, ry, DIM, self.font_s)
            pygame.draw.rect(screen, OFF, (bar_x2, ry + 3, bar_w2, 13))
            pygame.draw.rect(screen, ACTION_COLORS[a], (bar_x2, ry + 3, int(share * bar_w2), 13))
            self._text(screen, f"{share * 100:5.1f}%", bar_x2 + bar_w2 + 6, ry, TEXT, self.font_s)
        if d.value is not None:
            self._text(screen, f"critic: expected future reward {d.value:+.2f}  "
                       "(high = expects to finish, below 0 = expects to die)",
                       x + 10, y + h - 22, VALUE, self.font_s)
