"""ProcGen RL control room: a local web app to configure, run and monitor training.

    python webapp/server.py            # then open http://127.0.0.1:8765

Standard library only (plus the project's own scripts). Features:
- a persistent job queue (runs/webapp_queue.json) of train / evaluate / figures jobs,
  executed one at a time as subprocesses; survives server restarts and resumes
  interrupted training from latest.zip
- graceful stop (STOP file -> train.py saves and exits)
- keeps Windows awake while jobs run (SetThreadExecutionState)
- regenerates the presentation figures after every training job and every
  30 minutes while training runs
- JSON API for the dashboard in webapp/static
"""

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import zipfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
RUNS = ROOT / "runs"
PRESENTATION = RUNS / "presentation"
QUEUE_FILE = RUNS / "webapp_queue.json"
PY = sys.executable
FIGURES_EVERY_MIN = 30
NOISE = ("oneDNN", "absl::", "cudaGetDeviceCount", "run_backward", "_cuda_getDeviceCount",
         "Variable._execution_engine", "TF_ENABLE_ONEDNN_OPTS", "findfont")

sys.path.insert(0, str(ROOT))
from procgen_rl import config as C  # noqa: E402

# train.py flags the form may set: key -> (flag, kind)
TRAIN_FLAGS = {
    "hours": ("--hours", float), "n_envs": ("--n-envs", int), "milestones": ("--milestones", str),
    "eval_milestones": ("--eval-milestones", int), "seed": ("--seed", int),
    "save_every_minutes": ("--save-every-minutes", float), "max_wall_minutes": ("--max-wall-minutes", float),
    "start_difficulty": ("--start-difficulty", float), "max_difficulty": ("--max-difficulty", float),
    "difficulty_step": ("--difficulty-step", float), "promote_at": ("--promote-at", float),
    "promote_window": ("--promote-window", int), "fixed_difficulty": ("--fixed-difficulty", float),
    "train_levels": ("--train-levels", int), "net": ("--net", str), "lr": ("--lr", float),
    "n_steps": ("--n-steps", int), "batch_size": ("--batch-size", int), "n_epochs": ("--n-epochs", int),
    "gamma": ("--gamma", float), "gae_lambda": ("--gae-lambda", float), "ent_coef": ("--ent-coef", float),
    "clip_range": ("--clip-range", float),
}


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def safe_name(name):
    keep = "".join(ch for ch in str(name) if ch.isalnum() or ch in "-_")
    return keep[:60] or "run"


def tail(path, n=200):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 200_000))
            lines = f.read().decode("utf-8", "replace").splitlines()
    except FileNotFoundError:
        return []
    return [ln for ln in lines if not any(k in ln for k in NOISE)][-n:]


def keep_awake(on):
    """Prevent Windows idle sleep while jobs run (no-op elsewhere)."""
    if os.name != "nt":
        return False
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0)
        return bool(ctypes.windll.kernel32.SetThreadExecutionState(flags))
    except Exception:
        return False


def training_alive(run_dir):
    """True if a train.py for this run is still running (e.g. after a server restart)."""
    st = read_json(Path(run_dir) / "status.json", {})
    if st.get("phase") not in ("training", "evaluating"):
        return False
    pid = st.get("pid")
    try:
        import psutil
        if pid and psutil.pid_exists(pid):
            return "python" in psutil.Process(pid).name().lower()
        return False
    except ImportError:
        return time.time() - st.get("updated", 0) < 90


# ---------------------------------------------------------------- job manager
class JobManager:
    def __init__(self):
        self.lock = threading.RLock()
        data = read_json(QUEUE_FILE, {})
        self.jobs = data.get("jobs", [])
        self.active = data.get("active", False)
        self.settings = data.get("settings", {"main_run": "main"})
        self.next_id = max([j["id"] for j in self.jobs] + [0]) + 1
        for j in self.jobs:  # anything "running" was interrupted by a server restart
            if j["status"] == "running":
                j["status"] = "queued"
                j["note"] = "resumed after server restart"
        self.proc = None
        self.current = None
        self.awake = False
        self.figures_lock = threading.Lock()
        self.figures_state = {"running": False, "last": None, "error": None}
        self.last_figures = time.time()
        self.save()
        threading.Thread(target=self._worker, daemon=True, name="queue-worker").start()

    # ------------------------------------------------ persistence
    def save(self):
        with self.lock:
            write_json(QUEUE_FILE, {"jobs": self.jobs, "active": self.active, "settings": self.settings})

    def job(self, jid):
        return next((j for j in self.jobs if j["id"] == jid), None)

    # ------------------------------------------------ API actions
    def add(self, spec, front=False):
        with self.lock:
            jtype = spec.get("type", "train")
            run = safe_name(spec.get("run", "main"))
            args = {k: v for k, v in (spec.get("args") or {}).items() if v not in (None, "")}
            title = spec.get("title") or {
                "train": f"Train {run} ({float(args.get('hours', 100)):g} sim h)",
                "evaluate": f"Evaluate {run}",
                "figures": "Refresh presentation figures",
            }[jtype]
            job = {"id": self.next_id, "type": jtype, "run": run, "args": args, "title": title,
                   "post": bool(spec.get("post", True)), "status": "queued", "created": time.time(),
                   "started": None, "finished": None, "returncode": None, "note": ""}
            self.next_id += 1
            if front:
                idx = next((i for i, j in enumerate(self.jobs) if j["status"] == "queued"), len(self.jobs))
                self.jobs.insert(idx, job)
            else:
                self.jobs.append(job)
            self.save()
            return job

    def cancel(self, jid):
        with self.lock:
            j = self.job(jid)
            if j and j["status"] == "queued":
                j["status"] = "cancelled"
                self.save()
            elif j and j["status"] == "running":
                self.stop_current()

    def remove(self, jid):
        with self.lock:
            self.jobs = [j for j in self.jobs if not (j["id"] == jid and j["status"] != "running")]
            self.save()

    def requeue(self, jid):
        with self.lock:
            j = self.job(jid)
            if j and j["status"] in ("stopped", "failed", "cancelled", "done"):
                j.update(status="queued", started=None, finished=None, returncode=None, note="re-queued")
                self.jobs.remove(j)
                self.jobs.append(j)
                self.save()

    def set_active(self, on):
        with self.lock:
            self.active = bool(on)
            self.save()

    def stop_current(self):
        """Graceful for training (STOP file), terminate for the rest. Pauses the queue."""
        with self.lock:
            self.active = False
            j = self.current
            if j is None:
                self.save()
                return
            j["note"] = "stop requested"
            if j["type"] == "train":
                (RUNS / j["run"]).mkdir(parents=True, exist_ok=True)
                (RUNS / j["run"] / "STOP").write_text("stop")
            elif self.proc and self.proc.poll() is None:
                self.proc.terminate()
            self.save()

    # ------------------------------------------------ commands
    def command(self, job):
        run_dir = RUNS / job["run"]
        a = job["args"]
        if job["type"] == "train":
            cmd = [PY, str(ROOT / "train.py"), "--out", str(run_dir)]
            for k, v in a.items():
                if k in TRAIN_FLAGS:
                    flag, kind = TRAIN_FLAGS[k]
                    cmd += [flag, str(kind(v))]
            return cmd, run_dir / "train.log"
        if job["type"] == "evaluate":
            cmd = [PY, str(ROOT / "evaluate.py"), "--ghosts", str(run_dir / "ghosts"), "--only-new",
                   "--levels", str(int(a.get("levels", 100)))]
            return cmd, run_dir / "eval.log"
        if job["type"] == "figures":
            main = RUNS / self.settings.get("main_run", "main")
            cmd = [PY, str(ROOT / "figures.py"), "--main", str(main), "--runs-root", str(RUNS),
                   "--out", str(PRESENTATION)]
            return cmd, RUNS / "figures.log"
        raise ValueError(job["type"])

    @staticmethod
    def child_env():
        env = dict(os.environ, PYTHONUNBUFFERED="1", TF_CPP_MIN_LOG_LEVEL="3", TF_ENABLE_ONEDNN_OPTS="0",
                   PYTHONIOENCODING="utf-8")
        return env

    # ------------------------------------------------ worker
    def _next_job(self):
        with self.lock:
            if not self.active:
                return None
            return next((j for j in self.jobs if j["status"] == "queued"), None)

    def _worker(self):
        while True:
            try:
                job = self._next_job()
                if job is None:
                    if self.awake:
                        keep_awake(False)
                        self.awake = False
                    time.sleep(1.0)
                    continue
                self.awake = keep_awake(True)
                self._run(job)
            except Exception:
                traceback.print_exc()
                time.sleep(2)

    def _run(self, job):
        cmd, log_path = self.command(job)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            job.update(status="running", started=job.get("started") or time.time(), log=str(log_path))
            self.current = job
            self.save()
        run_dir = RUNS / job["run"]
        if job["type"] == "train" and training_alive(run_dir):
            # a train.py for this run survived a server restart: wait for it instead of starting another
            job["note"] = "attached to running training"
            while training_alive(run_dir):
                self._maybe_periodic_figures()
                time.sleep(5)
            st = read_json(run_dir / "status.json", {})
            rc = 0 if st.get("phase") in ("finished", "stopped") else 1
            stopped = st.get("phase") == "stopped"
        else:
            with open(log_path, "a", encoding="utf-8") as log:
                log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} job {job['id']}: {' '.join(cmd)}\n")
                log.flush()
                flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                self.proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
                                             env=self.child_env(), creationflags=flags)
                while self.proc.poll() is None:
                    if job["type"] == "train":
                        self._maybe_periodic_figures()
                    time.sleep(2)
                rc = self.proc.returncode
            st = read_json(run_dir / "status.json", {}) if job["type"] == "train" else {}
            stopped = st.get("phase") == "stopped" or job.get("note") == "stop requested"
        with self.lock:
            job["returncode"] = rc
            job["finished"] = time.time()
            job["status"] = "stopped" if stopped else ("done" if rc == 0 else "failed")
            self.current = None
            self.proc = None
            self.save()
        if job["type"] == "train" and job.get("post") and rc == 0:
            # inserted at the front in reverse order: evaluate runs next, then figures
            self.add({"type": "figures", "run": "presentation"}, front=True)
            self.add({"type": "evaluate", "run": job["run"], "args": {"levels": job["args"].get("eval_milestones") or 100},
                      "title": f"Evaluate {job['run']} (catch-up)"}, front=True)

    # ------------------------------------------------ figures
    def _maybe_periodic_figures(self):
        if time.time() - self.last_figures > FIGURES_EVERY_MIN * 60:
            self.refresh_figures_async(quick=True)

    def refresh_figures_async(self, quick=False):
        if not self.figures_lock.acquire(blocking=False):
            return False
        self.last_figures = time.time()

        def work():
            try:
                self.figures_state.update(running=True, error=None)
                cmd = [PY, str(ROOT / "figures.py"), "--main", str(RUNS / self.settings.get("main_run", "main")),
                       "--runs-root", str(RUNS), "--out", str(PRESENTATION)]
                if quick and (PRESENTATION / "01_level_gallery.png").exists():
                    cmd.append("--skip-illustrations")
                flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                with open(RUNS / "figures.log", "a", encoding="utf-8") as log:
                    log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
                    log.flush()
                    rc = subprocess.run(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
                                        env=self.child_env(), creationflags=flags).returncode
                self.figures_state["error"] = None if rc == 0 else f"figures.py exited with {rc}"
            except Exception as e:
                self.figures_state["error"] = str(e)
            finally:
                self.figures_state.update(running=False, last=time.time())
                self.figures_lock.release()

        threading.Thread(target=work, daemon=True).start()
        return True

    # ------------------------------------------------ views
    def state(self):
        with self.lock:
            cur = self.current
            status = None
            log = []
            if cur:
                if cur["type"] == "train":
                    status = read_json(RUNS / cur["run"] / "status.json")
                log = tail(cur.get("log", ""), 60)
            return {
                "active": self.active,
                "jobs": self.jobs,
                "current": cur,
                "status": status,
                "log": log,
                "keep_awake": self.awake,
                "figures": self.figures_state,
                "settings": self.settings,
                "cpu_count": os.cpu_count(),
                "server_time": time.time(),
            }


# ---------------------------------------------------------------- run info
def list_runs():
    out = []
    if not RUNS.exists():
        return out
    for d in sorted(RUNS.iterdir()):
        if not d.is_dir() or d.name in ("presentation",) or d.name.startswith("_"):
            continue
        if not ((d / "state.json").exists() or (d / "run_config.json").exists()):
            continue
        ghosts = sorted(p.name for p in (d / "ghosts").glob("*.npz")) if (d / "ghosts").exists() else []
        out.append({
            "name": d.name,
            "config": read_json(d / "run_config.json", {}),
            "state": read_json(d / "state.json", {}),
            "status": read_json(d / "status.json", {}),
            "ghosts": ghosts,
            "has_metrics": (d / "metrics.jsonl").exists(),
            "has_eval": (d / "eval.csv").exists(),
            "modified": max((p.stat().st_mtime for p in d.iterdir()), default=0),
        })
    return out


def run_metrics(name, max_points=1500):
    p = RUNS / safe_name(name) / "metrics.jsonl"
    rows = []
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    if len(rows) > max_points:
        step = len(rows) / max_points
        rows = [rows[int(i * step)] for i in range(max_points)] + [rows[-1]]
    return rows


def run_eval(name):
    import csv
    p = RUNS / safe_name(name) / "eval.csv"
    if not p.exists():
        return []
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def presentation():
    return {
        "figures": read_json(PRESENTATION / "captions.json", []),
        "summary": (PRESENTATION / "summary.md").read_text() if (PRESENTATION / "summary.md").exists() else "",
        "updated": (PRESENTATION / "captions.json").stat().st_mtime if (PRESENTATION / "captions.json").exists() else None,
    }


def overnight_plan():
    """Main showcase run + generalisation experiment."""
    common = {"n_envs": 8, "eval_milestones": 100, "save_every_minutes": 10}
    plan = [{"type": "train", "run": "main", "title": "Main run: curriculum, unlimited levels, 300 sim h",
             "args": {**common, "hours": 300}}]
    for n in (10, 100, 1000):
        plan.append({"type": "train", "run": f"gen_{n}", "title": f"Generalisation: {n} training levels, 100 sim h",
                     "args": {**common, "hours": 100, "train_levels": n, "fixed_difficulty": 0.5,
                              "milestones": "0,1,2,5,10,20,50,100"}})
    plan.append({"type": "train", "run": "gen_unlimited", "title": "Generalisation: unlimited levels, 100 sim h",
                 "args": {**common, "hours": 100, "fixed_difficulty": 0.5, "milestones": "0,1,2,5,10,20,50,100"}})
    return plan


# ---------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    server_version = "ProcGenRL/1.0"
    jobs: JobManager = None

    def log_message(self, fmt, *args):
        pass

    def _send(self, body, ctype="application/json", status=HTTPStatus.OK, headers=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path, base):
        try:
            path = path.resolve()
            path.relative_to(base.resolve())
        except Exception:
            return self._send({"error": "forbidden"}, status=HTTPStatus.FORBIDDEN)
        if not path.is_file():
            return self._send({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
        ctype = {".html": "text/html; charset=utf-8", ".js": "text/javascript", ".css": "text/css",
                 ".png": "image/png", ".json": "application/json", ".csv": "text/csv",
                 ".md": "text/markdown; charset=utf-8", ".log": "text/plain; charset=utf-8",
                 ".svg": "image/svg+xml"}.get(path.suffix.lower(), "application/octet-stream")
        self._send(path.read_bytes(), ctype)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        p = unquote(u.path)
        try:
            if p in ("/", "/index.html"):
                return self._file(STATIC / "index.html", STATIC)
            if p.startswith("/static/"):
                return self._file(STATIC / p[len("/static/"):], STATIC)
            if p.startswith("/files/"):
                return self._file(RUNS / p[len("/files/"):], RUNS)
            if p == "/api/state":
                return self._send(self.jobs.state())
            if p == "/api/runs":
                return self._send(list_runs())
            if p == "/api/metrics":
                return self._send(run_metrics(q.get("run", "main")))
            if p == "/api/eval":
                return self._send(run_eval(q.get("run", "main")))
            if p == "/api/presentation":
                return self._send(presentation())
            if p == "/api/plan":
                return self._send(overnight_plan())
            if p == "/api/log":
                j = self.jobs.job(int(q.get("job", 0)))
                return self._send({"lines": tail(j.get("log", ""), int(q.get("lines", 300))) if j else []})
            if p == "/api/defaults":
                return self._send({"agent_steps_per_sim_hour": C.AGENT_STEPS_PER_SIM_HOUR,
                                   "eval_seed_start": C.EVAL_SEED_START})
            if p == "/download/presentation.zip":
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                    for f in sorted(PRESENTATION.glob("*")):
                        if f.is_file() and not f.name.endswith(".tmp"):
                            z.write(f, f.name)
                return self._send(buf.getvalue(), "application/zip",
                                  headers={"Content-Disposition": "attachment; filename=presentation.zip"})
            return self._send({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            traceback.print_exc()
            self._send({"error": str(e)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self):
        p = urlparse(self.path).path
        try:
            body = self._body()
            J = self.jobs
            if p == "/api/jobs":
                job = J.add(body)
                if body.get("start"):
                    J.set_active(True)
                return self._send(job)
            if p == "/api/plan/overnight":
                jobs = [J.add(spec) for spec in overnight_plan()]
                J.set_active(True)
                return self._send(jobs)
            if p == "/api/queue/start":
                J.set_active(True)
                return self._send({"ok": True})
            if p == "/api/queue/pause":
                J.set_active(False)
                return self._send({"ok": True})
            if p == "/api/stop":
                J.stop_current()
                return self._send({"ok": True})
            if p.startswith("/api/jobs/"):
                parts = p.strip("/").split("/")
                jid, action = int(parts[2]), parts[3]
                {"cancel": J.cancel, "remove": J.remove, "requeue": J.requeue}[action](jid)
                return self._send({"ok": True})
            if p == "/api/figures":
                started = J.refresh_figures_async(quick=bool(body.get("quick")))
                return self._send({"started": started})
            if p == "/api/settings":
                with J.lock:
                    J.settings.update({k: safe_name(v) for k, v in body.items() if k == "main_run"})
                    J.save()
                return self._send(J.settings)
            if p == "/api/ghosts/install":
                run = safe_name(body.get("run", "main"))
                src = RUNS / run / "ghosts"
                dst = ROOT / "ghosts"
                files = sorted(src.glob("*.npz"))
                if not files:
                    return self._send({"error": f"no ghosts in runs/{run}"}, status=HTTPStatus.BAD_REQUEST)
                dst.mkdir(exist_ok=True)
                for old in dst.glob("*.npz"):
                    old.unlink()
                for f in files:
                    shutil.copy2(f, dst / f.name)
                return self._send({"installed": len(files)})
            if p == "/api/demo":
                run = safe_name(body.get("run", "main"))
                cmd = [PY, str(ROOT / "demo.py"), "--ghosts", str(RUNS / run / "ghosts"),
                       "--difficulty", str(float(body.get("difficulty", 0.5)))]
                if body.get("spectate"):
                    cmd.append("--spectate")
                if body.get("only"):
                    cmd += ["--only", str(body["only"])]
                subprocess.Popen(cmd, cwd=str(ROOT))
                return self._send({"ok": True})
            return self._send({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
        except Exception as e:
            traceback.print_exc()
            self._send({"error": str(e)}, status=HTTPStatus.BAD_REQUEST)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--open", action="store_true", help="open the browser")
    args = ap.parse_args()
    RUNS.mkdir(exist_ok=True)
    Handler.jobs = JobManager()
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"ProcGen RL control room on {url}  (Ctrl+C to quit; running training keeps going and is re-attached on restart)")
    if args.open:
        import webbrowser
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
