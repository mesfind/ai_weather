#!/usr/bin/env python3
"""
Run ROMP's obs_onset_analysis in its own process and save every figure as a PNG.

Usage (called by short_medium_benchmarking.py):
    python onset_worker.py <payload.pkl> <output_dir>
"""
import os
import sys

# 1. Environment variables MUST be set at line 1 before ANY C libraries load
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"
os.environ["GOTO_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["MPLBACKEND"] = "Agg"

import pickle
import resource
import warnings
from pathlib import Path



# 2. Force Matplotlib backend to headless Agg
import matplotlib
matplotlib.use("Agg")

# 3. Expand Linux C-stack size limit (prevents stack overflow on heavy array operations)
try:
    resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
except Exception:
    pass

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2

    payload_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    payload = pickle.loads(payload_path.read_bytes())
    out_dir.mkdir(parents=True, exist_ok=True)

    # ROMP may parse sys.argv; hand it a clean one
    sys.argv = [sys.argv[0]]

    import config  # noqa: F401,E402 (puts the ROMP folders on sys.path)
    import momp.app.onset_time_series as ots  # noqa: E402

    # Track generated figures without breaking Matplotlib's internal memory cleanup
    captured_figs = []

    def safe_show(*args, **kwargs):
        fig = plt.gcf()
        if fig and fig not in captured_figs:
            captured_figs.append(fig)

    plt.show = safe_show

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ots.obs_onset_analysis(
            year=payload["year"],
            lat_select=payload["lat_select"],
            lon_select=payload["lon_select"],
            **payload["kwargs"],
        )

    # Collect figures from both safe_show and active figure numbers
    figs_to_save = list(captured_figs)
    for num in plt.get_fignums():
        fig = plt.figure(num)
        if fig not in figs_to_save:
            figs_to_save.append(fig)

    saved = 0
    for fig in figs_to_save:
        if not fig.get_axes():
            continue
        fig.savefig(out_dir / f"fig_{saved:02d}.png", format="png", dpi=110, bbox_inches="tight")
        saved += 1

    plt.close("all")
    print(f"[onset_worker] saved {saved} figure(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())