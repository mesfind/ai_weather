"""Where model outputs live on the Spark, and how saved runs are reused.

Layout: {OUTPUT_DIR}/{model}/{YYYYMMDDTHH}_{lead}h_m{members}.nc
A saved run is reused for any request with the same model, init and member
count whose lead time is <= the saved one (the extra lead times are sliced off).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

OUTPUT_DIR = Path(os.environ.get("DEMO3_OUTPUT_DIR", Path(__file__).resolve().parent.parent / "outputs"))
JOBS_DIR = OUTPUT_DIR / "_jobs"

_NAME = re.compile(r"^(\d{8}T\d{2})_(\d+)h_m(\d+)(?:_synthetic)?\.nc$")


@dataclass(frozen=True)
class RunRequest:
    model: str
    init: pd.Timestamp
    lead_hours: int
    members: int
    synthetic: bool = False

    def path(self) -> Path:
        tag = "_synthetic" if self.synthetic else ""
        return OUTPUT_DIR / self.model / f"{self.init:%Y%m%dT%H}_{self.lead_hours}h_m{self.members}{tag}.nc"


def find_saved(req: RunRequest) -> Path | None:
    """Shortest saved run that covers the request, or None."""
    folder = OUTPUT_DIR / req.model
    if not folder.is_dir():
        return None
    best = None
    for p in folder.glob("*.nc"):
        m = _NAME.match(p.name)
        if not m or ("_synthetic" in p.name) != req.synthetic:
            continue
        init, lead, members = m.group(1), int(m.group(2)), int(m.group(3))
        if init == f"{req.init:%Y%m%dT%H}" and members == req.members and lead >= req.lead_hours:
            if best is None or lead < best[0]:
                best = (lead, p)
    return best[1] if best else None


def list_saved() -> pd.DataFrame:
    rows = []
    if OUTPUT_DIR.is_dir():
        for p in OUTPUT_DIR.glob("*/*.nc"):
            m = _NAME.match(p.name)
            if m:
                rows.append(dict(model=p.parent.name, init=m.group(1), lead_h=int(m.group(2)),
                                 members=int(m.group(3)), synthetic="_synthetic" in p.name,
                                 size_mb=round(p.stat().st_size / 1e6, 1), path=str(p)))
    return pd.DataFrame(rows)
