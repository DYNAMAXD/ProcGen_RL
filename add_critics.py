"""Copy the critic (value network) from training checkpoints into ghost files.

    python add_critics.py                                   # ./ghosts from runs/main/checkpoints
    python add_critics.py --ghosts runs/gen_10/ghosts --checkpoints runs/gen_10/checkpoints

Ghosts exported by train.py now include the critic. Older ghosts only hold the
actor; this adds the critic from the matching checkpoints/ppo_<label>.zip so the
demo's debug view can show what the agent expects. Acting is unchanged. Needs PyTorch.
"""

import argparse
from pathlib import Path

from procgen_rl.policy import NumpyPolicy, add_critic_from_checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ghosts", default="ghosts")
    p.add_argument("--checkpoints", default="runs/main/checkpoints")
    args = p.parse_args()
    for path in sorted(Path(args.ghosts).glob("*.npz")):
        g = NumpyPolicy.load(path)
        if g.critic is not None:
            print(f"{path.name}: already has a critic")
            continue
        ckpt = Path(args.checkpoints) / f"ppo_{g.label}.zip"
        if not ckpt.exists():
            print(f"{path.name}: no checkpoint {ckpt}, skipped")
            continue
        add_critic_from_checkpoint(path, ckpt)
        print(f"{path.name}: critic added from {ckpt.name}")


if __name__ == "__main__":
    main()
