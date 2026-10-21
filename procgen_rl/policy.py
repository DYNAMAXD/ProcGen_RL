"""Ghost policy files: export from Stable-Baselines3 and run with NumPy only.

A ghost is a small .npz holding the actor MLP weights plus JSON metadata
(label, simulated hours, physics fingerprint). The demo and evaluator only
need NumPy to run it, so the laptop does not need PyTorch or SB3.
"""

import json
from pathlib import Path

import numpy as np

from . import config as C

_ACTIVATIONS = {
    "Tanh": np.tanh,
    "ReLU": lambda x: np.maximum(x, 0.0),
}


def export_sb3_policy(model, path, sim_hours, label=None, extra=None):
    """Write the actor network of an SB3 PPO MlpPolicy to `path` (.npz)."""
    import torch.nn as nn

    policy = model.policy
    linears = [m for m in policy.mlp_extractor.policy_net if isinstance(m, nn.Linear)]
    linears.append(policy.action_net)
    arrays = {}
    for i, layer in enumerate(linears):
        arrays[f"W{i}"] = layer.weight.detach().cpu().numpy().astype(np.float32)
        arrays[f"b{i}"] = layer.bias.detach().cpu().numpy().astype(np.float32)
    meta = {
        "label": label or format_hours(sim_hours),
        "sim_hours": float(sim_hours),
        "timesteps": int(model.num_timesteps),
        "activation": policy.activation_fn.__name__,
        "n_layers": len(linears),
        "fingerprint": C.physics_fingerprint(),
    }
    if extra:
        meta.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, meta=np.array(json.dumps(meta)), **arrays)
    return path


def format_hours(h):
    if h < 1:
        return f"{round(h * 60)}min"
    return f"{h:g}h"


class NumpyPolicy:
    def __init__(self, weights, biases, activation, meta, path=None):
        self.weights = weights
        self.biases = biases
        self.activation = _ACTIVATIONS[activation]
        self.meta = meta
        self.path = path

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["meta"]))
        n = meta["n_layers"]
        weights = [data[f"W{i}"] for i in range(n)]
        biases = [data[f"b{i}"] for i in range(n)]
        return cls(weights, biases, meta["activation"], meta, path=str(path))

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

    def act(self, obs, deterministic=True, rng=None):
        logits = self.logits(obs)
        if deterministic:
            return int(np.argmax(logits))
        p = np.exp(logits - logits.max())
        p /= p.sum()
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
