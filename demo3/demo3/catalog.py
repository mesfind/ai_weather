"""Models offered in Demo 3.

One entry per model on the 2026 list. `env` names the Python environment the
model's runner executes in (models whose dependencies conflict get separate
environments inside the container). `status` gates the Run button: models
that are not installed yet can only be exercised with synthetic output.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

ARCO_ERA5 = "ARCO ERA5 (Google)"
IFS_OPEN = "IFS analysis (ECMWF Open Data)"
# ECMWF open data on AWS starts 2024-03-01 and is current to within a few days.
IFS_MIN, IFS_MAX = date(2024, 3, 1), date.today() - timedelta(days=2)


@dataclass(frozen=True)
class Model:
    key: str
    name: str
    org: str
    kind: str                     # "deterministic" | "ensemble"
    init_source: str
    env: str                      # runner environment name
    max_lead_days: int
    status: str = "not installed"  # "ready" | "not installed"
    max_members: int = 1
    default_members: int = 1
    slow: bool = False            # suggest loading a saved run in class
    init_min: date = date(2020, 1, 1)
    init_max: date = date(2026, 4, 30)
    notes: str = ""


MODELS: list[Model] = [
    Model("aifs2_single", "AIFS Single v2.0", "ECMWF", "deterministic", IFS_OPEN, "e2s018",
          max_lead_days=15, init_min=IFS_MIN, init_max=IFS_MAX,
          notes="Starts from IFS analyses (incl. wave fields); ARCO ERA5 lacks some inputs."),
    Model("graphcast", "GraphCast", "Google DeepMind", "deterministic", ARCO_ERA5, "graphcast",
          max_lead_days=10, status="ready"),
    Model("aurora15", "Aurora 1.5", "Microsoft", "deterministic", ARCO_ERA5, "e2s018",
          max_lead_days=10, slow=True),
    Model("aifs2_ens", "AIFS ENS v2", "ECMWF", "ensemble", IFS_OPEN, "e2s018ens",
          max_lead_days=15, max_members=3, default_members=2, slow=True,
          init_min=IFS_MIN, init_max=IFS_MAX,
          notes="Needs IFS initial conditions incl. wave fields, not ARCO."),
    Model("atlas_crps", "Atlas CRPS", "NVIDIA", "ensemble", ARCO_ERA5, "e2s018",
          max_lead_days=10, max_members=3, default_members=2),
    Model("neuralgcm", "NeuralGCM", "Google", "ensemble", ARCO_ERA5, "neuralgcm",
          max_lead_days=10, max_members=3, default_members=2,
          notes="2.8° (~300 km) stochastic precipitation version — the only NeuralGCM checkpoint "
                "that forecasts rain. No 2 m temperature output."),
    Model("fgn", "FGN Mini (WeatherNext 2, 1°)", "Google DeepMind", "ensemble",
          "HRES analysis (Google sample case)", "fgn",
          max_lead_days=7, max_members=3, default_members=3, slow=False,
          init_min=date(2024, 10, 7), init_max=date(2024, 10, 7),
          notes="The full 0.25° FGN does not fit in a Spark's memory, so this is Google's 1° Mini "
                "version. Only Google's sample start date (2024-10-07) is available until we build "
                "a converter from ECMWF open data."),
]

BY_KEY = {m.key: m for m in MODELS}

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
