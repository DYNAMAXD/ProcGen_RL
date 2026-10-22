"""Ghost policy files: export from Stable-Baselines3 and run with NumPy only.

A ghost is a small .npz holding the actor MLP weights plus JSON metadata
(label, simulated hours, physics fingerprint). The demo and evaluator only
need NumPy to run it, so the laptop does not need PyTorch or SB3.

Ghosts may also carry the critic (value network, arrays V0.. / c0..). Only the
demo's debug view and the explainability figures use it; acting never does.
"""

import json
from pathlib import Path

import numpy as np

from . import config as C

_ACTIVATIONS = {
    "Tanh": np.tanh,
    "ReLU": lambda x: np.maximum(x, 0.0),
}

# Observation layout: OBS_CHANNELS x VIEW_H x VIEW_W one-hot tiles, then the scalars
# vx, vy, on_ground, can_jump, jumping, frac_x, frac_y. Tiles and the 3 flags are 0 or 1.
N_TILE_INPUTS = C.OBS_CHANNELS * C.VIEW_H * C.VIEW_W
_BINARY_INPUTS = np.zeros(C.OBS_DIM, dtype=bool)
_BINARY_INPUTS[:N_TILE_INPUTS] = True
_BINARY_INPUTS[N_TILE_INPUTS + 2:N_TILE_INPUTS + 5] = True


def softmax(logits):
    p = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return p / p.sum(axis=-1, keepdims=True)


def export_sb3_policy(model, path, sim_hours, label=None, extra=None):
    """Write the actor (and critic) networks of an SB3 PPO MlpPolicy to `path` (.npz)."""
    import torch.nn as nn

    policy = model.policy
    linears = [m for m in policy.mlp_extractor.policy_net if isinstance(m, nn.Linear)]
    linears.append(policy.action_net)
    critic = [m for m in policy.mlp_extractor.value_net if isinstance(m, nn.Linear)]
    critic.append(policy.value_net)
    arrays = {}
    for i, layer in enumerate(linears):
        arrays[f"W{i}"] = layer.weight.detach().cpu().numpy().astype(np.float32)
        arrays[f"b{i}"] = layer.bias.detach().cpu().numpy().astype(np.float32)
    for i, layer in enumerate(critic):
        arrays[f"V{i}"] = layer.weight.detach().cpu().numpy().astype(np.float32)
        arrays[f"c{i}"] = layer.bias.detach().cpu().numpy().astype(np.float32)
    meta = {
        "label": label or format_hours(sim_hours),
        "sim_hours": float(sim_hours),
        "timesteps": int(model.num_timesteps),
        "activation": policy.activation_fn.__name__,
        "n_layers": len(linears),
        "value_layers": len(critic),
        "fingerprint": C.physics_fingerprint(),
    }
    if extra:
        meta.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, meta=np.array(json.dumps(meta)), **arrays)
    return path


def add_critic_from_checkpoint(ghost_path, checkpoint_path):
    """Copy the critic from an SB3 checkpoint .zip into an existing ghost file.

    For ghosts exported before ghosts carried the critic. Refuses if the
    checkpoint's actor is not the ghost's actor. Needs PyTorch.
    """
    import io
    import zipfile

    import torch

    with zipfile.ZipFile(checkpoint_path) as z:
        sd = torch.load(io.BytesIO(z.read("policy.pth")), map_location="cpu", weights_only=True)
    with np.load(ghost_path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    meta = json.loads(str(arrays.pop("meta")))
    if not np.array_equal(arrays["W0"], sd["mlp_extractor.policy_net.0.weight"].numpy()):
        raise ValueError(f"{checkpoint_path} does not hold the actor of {ghost_path}")
    names = sorted((k[:-len(".weight")] for k in sd
                    if k.startswith("mlp_extractor.value_net.") and k.endswith(".weight")),
                   key=lambda k: int(k.rsplit(".", 1)[1]))
    names.append("value_net")
    for i, name in enumerate(names):
        arrays[f"V{i}"] = sd[name + ".weight"].numpy().astype(np.float32)
        arrays[f"c{i}"] = sd[name + ".bias"].numpy().astype(np.float32)
    meta["value_layers"] = len(names)
    tmp = Path(ghost_path).with_name(Path(ghost_path).stem + ".tmp.npz")
    np.savez(tmp, meta=np.array(json.dumps(meta)), **arrays)
    tmp.replace(ghost_path)


def format_hours(h):
    if h < 1:
        return f"{round(h * 60)}min"
    return f"{h:g}h"


class NumpyPolicy:
    def __init__(self, weights, biases, activation, meta, path=None, critic=None):
        self.weights = weights
        self.biases = biases
        self.activation = _ACTIVATIONS[activation]
        self.meta = meta
        self.path = path
        self.critic = critic      # (weights, biases) of the value network, or None

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["meta"]))
        n = meta["n_layers"]
        weights = [data[f"W{i}"] for i in range(n)]
        biases = [data[f"b{i}"] for i in range(n)]
        critic = None
        if meta.get("value_layers"):
            m = meta["value_layers"]
            critic = ([data[f"V{i}"] for i in range(m)], [data[f"c{i}"] for i in range(m)])
        return cls(weights, biases, meta["activation"], meta, path=str(path), critic=critic)

    @property
    def label(self):
        return self.meta["label"]

    @property
    def sim_hours(self):
        return self.meta["sim_hours"]

    def compatible(self):
        """True if the ghost was trained with the current physics/observation."""
        return self.meta.get("fingerprint") == C.physics_fingerprint()

    def logits(self, obs):
        h = obs
        last = len(self.weights) - 1
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            h = h @ W.T + b
            if i < last:
                h = self.activation(h)
        return h

    def trace(self, obs):
        """Forward pass that keeps every layer: (hidden activations, logits, probabilities)."""
        h = obs
        hidden = []
        last = len(self.weights) - 1
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            h = h @ W.T + b
            if i < last:
                h = self.activation(h)
                hidden.append(h)
        return hidden, h, softmax(h)

    def value(self, obs):
        """The critic's estimate of future reward from this state, or None without a critic."""
        if self.critic is None:
            return None
        h = obs
        weights, biases = self.critic
        last = len(weights) - 1
        for i, (W, b) in enumerate(zip(weights, biases)):
            h = h @ W.T + b
            if i < last:
                h = self.activation(h)
        return float(h[0])

    def sensitivity(self, obs):
        """How much the decision depends on each input, changing one input at a time.

        Binary inputs (the 840 tile values and the 3 flags) are flipped 0 <-> 1;
        continuous inputs (speeds, position in the tile) are set to 0. The score is
        the total variation distance between the action probabilities before and
        after: 0 = no change, 1 = a completely different decision. Returns OBS_DIM
        scores from one batched forward pass (about 10 ms).
        """
        obs = np.asarray(obs, dtype=np.float32)
        changed = np.where(_BINARY_INPUTS, 1.0 - obs, 0.0).astype(np.float32)
        delta = changed - obs
        W0, b0 = self.weights[0], self.biases[0]
        h = (obs @ W0.T + b0)[None, :] + delta[:, None] * W0.T   # one row per changed input
        for W, b in zip(self.weights[1:], self.biases[1:]):
            h = self.activation(h) @ W.T + b
        p = softmax(self.logits(obs))
        return 0.5 * np.abs(softmax(h) - p[None, :]).sum(axis=1)

    def act(self, obs, deterministic=True, rng=None):
        logits = self.logits(obs)
        if deterministic:
            return int(np.argmax(logits))
        p = softmax(logits)
        rng = rng or np.random.default_rng()
        return int(rng.choice(len(p), p=p))


def load_ghosts(directory, skip_incompatible=True):
    """Load every ghost .npz in `directory`, sorted by simulated hours."""
    ghosts, skipped = [], []
    for p in sorted(Path(directory).glob("*.npz")):
        try:
            g = NumpyPolicy.load(p)
        except Exception as e:  # corrupt or foreign file
            skipped.append((p.name, str(e)))
            continue
        if skip_incompatible and not g.compatible():
            skipped.append((p.name, "trained with different physics/observation config"))
            continue
        ghosts.append(g)
    ghosts.sort(key=lambda g: g.sim_hours)
    return ghosts, skipped
