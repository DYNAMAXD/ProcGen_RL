# ProcGen RL

A procedurally generated 2D platformer and a PPO agent that learns to beat levels it
has never seen. In the demo you race against "ghosts": snapshots of the agent after
0 minutes, 30 minutes, 1 hour, 10 hours, ... of simulated training, all playing the
same fresh level at once.

Design details, mechanics and results: [game_info.md](game_info.md).

## Quick start (laptop)

```
pip install -r requirements-demo.txt
python demo.py
```

Run right with the arrow keys or A/D, jump with Space (hold it for a higher jump), press
N for a new level, and H for all controls. `python demo.py --spectate` just watches
the ghosts.

## Control room (web app)

Double-click `run_webapp.bat`, or run `python webapp/server.py --open`, then open
http://127.0.0.1:8765.

- **New run:** a form with every training setting, presets, and a one-click "overnight plan" (the
  main 300-hour run plus the generalisation experiment).
- **Dashboard:** live progress, ETA, speed, curriculum, and charts of training and held-out
  evaluation. It also lists every autosaved ghost and checkpoint.
- **Queue:** jobs run one after another. Training can be stopped (it saves first) and resumed
  later. Closing the web app does not stop a running training, and restarting it picks the
  training up again.
- **Presentation:** all figures for the slides with captions, a results summary, and a zip
  download. They are refreshed automatically during and after training.
- **Demo:** launch the game against any run's ghosts.

It keeps Windows awake while jobs run. On a laptop, keep it plugged in and the lid open.

## Training

On Google Colab or Kaggle: open `notebooks/train_colab.ipynb`, choose a CPU runtime
and run all cells. Runs are saved to Google Drive and resume after a disconnect. The
last cell downloads a zip of ghosts; unzip it into `ghosts/` on the laptop.

Locally:

```
pip install -r requirements.txt
python train.py --hours 100 --out runs/main
python evaluate.py --ghosts runs/main/ghosts --plot runs/main/eval.png
python demo.py --ghosts runs/main/ghosts
```

Generalisation experiment (train on only 10 fixed levels, then compare on unseen ones):

```
python train.py --hours 100 --out runs/fixed10 --train-levels 10
python evaluate.py --ghosts runs/fixed10/ghosts
```

## Tests

```
python -m pytest -q
```

The tests check that levels are deterministic and that a search-based solver can beat
every generated level. They also check that the demo's ghost stepping matches the
training env exactly, and that exported ghosts reproduce the PyTorch policy.
