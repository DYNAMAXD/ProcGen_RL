import numpy as np
import torch
from stable_baselines3 import PPO

from procgen_rl.env import PlatformerEnv
from procgen_rl.policy import NumpyPolicy, export_sb3_policy


def test_exported_ghost_matches_torch_policy(tmp_path):
    env = PlatformerEnv()
    model = PPO("MlpPolicy", env, n_steps=64, batch_size=64, device="cpu",
                policy_kwargs=dict(net_arch=dict(pi=[32, 32], vf=[32, 32]), activation_fn=torch.nn.ReLU))
    path = export_sb3_policy(model, tmp_path / "g.npz", sim_hours=2.0)
    ghost = NumpyPolicy.load(path)
    assert ghost.label == "2h" and ghost.compatible()

    obs, _ = env.reset(seed=0)
    for _ in range(50):
        with torch.no_grad():
            dist = model.policy.get_distribution(torch.as_tensor(obs[None]))
            torch_logits = dist.distribution.logits.numpy()[0]
        np_logits = ghost.logits(obs)
        # SB3 stores normalised log-probs; compare up to a constant shift
        assert np.allclose(np_logits - np_logits.max(), torch_logits - torch_logits.max(), atol=1e-4)
        assert ghost.act(obs) == int(np.argmax(torch_logits))
        obs, _, term, trunc, _ = env.step(ghost.act(obs))
        if term or trunc:
            obs, _ = env.reset()


def test_exported_critic_and_sensitivity(tmp_path):
    env = PlatformerEnv()
    model = PPO("MlpPolicy", env, n_steps=64, batch_size=64, device="cpu",
                policy_kwargs=dict(net_arch=dict(pi=[32, 32], vf=[32, 32]), activation_fn=torch.nn.ReLU))
    ghost = NumpyPolicy.load(export_sb3_policy(model, tmp_path / "g.npz", sim_hours=1.0))
    obs, _ = env.reset(seed=1)
    with torch.no_grad():
        torch_value = float(model.policy.predict_values(torch.as_tensor(obs[None]))[0, 0])
    assert abs(ghost.value(obs) - torch_value) < 1e-4

    # sensitivity must equal changing each input by hand and comparing probabilities
    sens = ghost.sensitivity(obs)
    _, _, p = ghost.trace(obs)
    for i in (0, 300, 600, 840, 842, 846):
        x = obs.copy()
        x[i] = (1.0 - x[i]) if (i < 840 or 842 <= i <= 844) else 0.0
        _, _, q = ghost.trace(x)
        assert abs(sens[i] - 0.5 * np.abs(q - p).sum()) < 1e-5
