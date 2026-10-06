"""Models offered in Demo 3.

One entry per model on the 2026 list. `env` names the Python environment the
model's runner executes in (models whose dependencies conflict get separate
environments inside the container). A model can run live only if its
environment exists at {ENV_ROOT}/{env}; otherwise only its saved runs load.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

ARCO_ERA5 = "ARCO ERA5 (Google)"
IFS_OPEN = "IFS analysis (ECMWF Open Data)"
# ECMWF open data on AWS starts 2024-03-01 and is current to within a few days.
IFS_MIN, IFS_MAX = date(2024, 3, 1), date.today() - timedelta(days=2)
# ECMWF open data has every input AIFS v2 needs (incl. 10 hPa levels) only from this date.
AIFS2_MIN = date(2026, 5, 13)
# Forecast variables every run aims to provide, with the names shown to participants.
VAR_LABELS = {"tp": "rainfall", "t2m": "2 m temperature", "z500": "500 hPa geopotential (z500)"}
RUNNERS_DIR = Path(__file__).resolve().parent.parent / "runners"
# Model environments inside the container: {ENV_ROOT}/{env}/bin/python
ENV_ROOT = Path(os.environ.get("DEMO3_ENV_ROOT", "/opt/envs"))


@dataclass(frozen=True)
class Model:
    key: str
    name: str
    org: str
    kind: str                     # "deterministic" | "ensemble"
    init_source: str
    env: str                      # runner environment name
    max_lead_days: int
    max_members: int = 1
    default_members: int = 1
    slow: bool = False            # suggest loading a saved run in class
    init_min: date = date(2020, 1, 1)
    init_max: date = date(2026, 4, 30)
    notes: str = ""
    missing: tuple[str, ...] = ()  # variables (VAR_LABELS keys) the model does not forecast

    @property
    def live(self) -> bool:
        """True if this container can run the model: its environment is installed and it
        has a runner (FGN, for now, only has Google's saved sample case)."""
        return ((ENV_ROOT / self.env / "bin" / "python").exists()
                and (RUNNERS_DIR / f"runner_{self.key}.py").exists())


MODELS: list[Model] = [
    Model("aifs2_single", "AIFS Single v2.0", "ECMWF", "deterministic", IFS_OPEN, "e2s018",
          max_lead_days=15, init_min=AIFS2_MIN, init_max=IFS_MAX,
          notes="Starts from ECMWF's IFS analyses. ECMWF's open data has every input AIFS v2 "
                "needs only from 13 May 2026, so earlier start dates aren't available."),
    Model("graphcast", "GraphCast", "Google DeepMind", "deterministic", ARCO_ERA5, "graphcast",
          max_lead_days=10),
    Model("aurora15", "Aurora 1.5", "Microsoft", "deterministic", ARCO_ERA5, "e2s018",
          max_lead_days=10, slow=True),
    Model("aifs2_ens", "AIFS ENS v2", "ECMWF", "ensemble", IFS_OPEN, "e2s018ens",
          max_lead_days=15, max_members=3, default_members=2, slow=True,
          init_min=AIFS2_MIN, init_max=IFS_MAX,
          notes="Starts from ECMWF's IFS analyses. ECMWF's open data has every input AIFS v2 "
                "needs only from 13 May 2026, so earlier start dates aren't available."),
    Model("atlas_crps", "Atlas CRPS", "NVIDIA", "ensemble", ARCO_ERA5, "e2s018",
          max_lead_days=10, max_members=3, default_members=2),
    Model("neuralgcm", "NeuralGCM", "Google", "ensemble", ARCO_ERA5, "neuralgcm",
          max_lead_days=10, max_members=3, default_members=2, missing=("t2m",),
          notes="2.8° (~300 km) stochastic precipitation version — the only NeuralGCM checkpoint "
                "that forecasts rain. No 2 m temperature output."),
    Model("fgn", "FGN Mini (WeatherNext 2, 1°)", "Google DeepMind", "ensemble",
          "IFS analysis (ECMWF Open Data)", "fgn",
          max_lead_days=10, max_members=3, default_members=3,
          init_min=IFS_MIN, init_max=IFS_MAX,
          notes="The full 0.25° FGN does not fit in a Spark's memory, so this is Google's 1° Mini "
                "version (cyclone-tuned weights, the only Mini Google publishes). Starting conditions "
                "are built from ECMWF open data; on 7 Oct 2024 it uses Google's own sample file."),
]

BY_KEY = {m.key: m for m in MODELS}


def missing_in_run(ds) -> set[str]:
    """Variables a saved run does not provide: listed as missing, absent, or all NaN."""
    out = set(filter(None, ds.attrs.get("missing_variables", "").split(",")))
    for v in VAR_LABELS:
        if v not in ds.data_vars or not bool(ds[v].notnull().any()):
            out.add(v)
    return out

TIMINGS_PATH = Path(__file__).resolve().parent.parent / "timings.json"


def load_timings() -> dict:
    try:
        return json.loads(TIMINGS_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def timing_label(model: Model, timings: dict | None = None) -> str:
    """Bracketed runtime shown next to a model, e.g. '(~4 min for 10 days, 3 members)'."""
    t = (timings if timings is not None else load_timings()).get(model.key)
    if not t:
        return "(not timed yet)"
    total = t.get("load_s", 0) + t.get("run_s", 0)
    if total < 90:
        dur = f"~{round(total)} s"
    elif total < 5400:
        dur = f"~{round(total / 60)} min"
    else:
        dur = f"~{total / 3600:.1f} h"
    detail = f"{t.get('lead_days', '?')} days"
    if model.kind == "ensemble":
        detail += f", {t.get('members', '?')} members"
    return f"({dur} for {detail})"
