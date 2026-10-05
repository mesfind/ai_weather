"""Launch Panchali's obs-vs-forecast event movie for a Demo 3 run, in the background.

Each request gets its own folder under outputs/_movies/<hash>/ holding the GIF,
the track map, the log and result.json, so repeating a request is instant.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from .regions import Box
from .store import OUTPUT_DIR

APP_ROOT = Path(__file__).resolve().parent.parent
WRAPPER = APP_ROOT / "demo3" / "event_movie" / "run_event_movie.py"
MOVIES_DIR = OUTPUT_DIR / "_movies"
DATA_DIR = Path(os.environ.get("DEMO3_DATA", "/data"))  # optional preloaded obs


@dataclass(frozen=True)
class MovieRequest:
    run_path: str
    model_name: str
    kind: str                # "heat" | "precip"
    init: str                # YYYY-MM-DD
    end: str                 # YYYY-MM-DD
    box: tuple               # (lon_min, lon_max, lat_min, lat_max)
    region_name: str
    reduce: str              # mean | max | p95 | p99
    threshold: float | None  # None -> her default (high percentile of obs)

    def folder(self) -> Path:
        key = json.dumps(asdict(self), sort_keys=True, default=str)
        return MOVIES_DIR / hashlib.md5(key.encode()).hexdigest()[:12]

    def args(self) -> list[str]:
        out = self.folder()
        imerg = DATA_DIR / "IMERG"
        a = ["--kind", self.kind, "--fc-path", self.run_path, "--fc-label", self.model_name,
             "--fc-units", "mm", "--domain", *map(str, self.box),
             "--init", self.init, "--end", self.end, "--reduce", self.reduce,
             "--name", self.region_name.replace(" ", "_").lower(),
             "--title", f"{self.region_name}: {self.model_name}",
             "--outdir", str(out), "--no-mp4", "--result", str(out / "result.json")]
        if self.kind == "precip" and not imerg.is_dir():
            a += ["--obs", "era5"]          # IMERG not on this Spark yet
        if self.threshold is not None:
            a += ["--threshold", str(self.threshold)]
        return a


def request_for(run_path: str, model_name: str, kind: str, init: pd.Timestamp, lead_days: int,
                box: Box, region_name: str, reduce: str, threshold: float | None) -> MovieRequest:
    end = init + pd.Timedelta(days=lead_days)
    return MovieRequest(run_path, model_name, kind, f"{init:%Y-%m-%d}", f"{end:%Y-%m-%d}",
                        (box.lon_min, box.lon_max, box.lat_min, box.lat_max), region_name,
                        reduce, threshold)


def launch(req: MovieRequest) -> None:
    out = req.folder()
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").unlink(missing_ok=True)
    env = dict(os.environ, DEMO3_DATA=str(DATA_DIR))
    with open(out / "log.txt", "w") as log:
        proc = subprocess.Popen([sys.executable, str(WRAPPER), *req.args()], stdout=log,
                                stderr=subprocess.STDOUT, cwd=APP_ROOT, env=env,
                                start_new_session=True)
    (out / "pid").write_text(str(proc.pid))


def _alive(pid_file: Path) -> bool:
    try:
        pid = int(pid_file.read_text())
        os.kill(pid, 0)
        # a finished child that hasn't been reaped is a zombie: not alive
        return "Z" not in Path(f"/proc/{pid}/stat").read_text().split()[2]
    except (FileNotFoundError, ValueError, ProcessLookupError, PermissionError, IndexError):
        return False


def status(req: MovieRequest) -> dict:
    """{'state': 'none'|'running'|'done'|'failed', 'gif', 'png', 'error', 'log'}"""
    out = req.folder()
    res, log = out / "result.json", out / "log.txt"
    tail = "".join(log.read_text().splitlines(keepends=True)[-15:]) if log.exists() else ""
    if res.exists():
        r = json.loads(res.read_text())
        return dict(state="failed" if r.get("error") else "done", log=tail, **r)
    if not log.exists():
        return dict(state="none", log=tail)
    if _alive(out / "pid"):
        return dict(state="running", log=tail)
    return dict(state="failed", error="the movie process stopped without a result (see log)",
                gif=None, png=None, log=tail)
