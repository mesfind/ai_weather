"""Demo 3 — Running Your First AI Weather Forecast.

    streamlit run app.py

Step 1 (in the terminal) starts the container and this app. In the app:
Step 2 choose model, start date, lead time, region and use case;
Step 3 run the model (or load a saved run); then explore the results.
"""
from __future__ import annotations

import shutil
import subprocess
import time
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st
import xarray as xr

from demo3 import catalog, jobs, movie, store, theme, viz
from demo3.regions import COUNTRIES, GROUPS, Box, country_box

st.set_page_config(page_title="Demo 3 · AI Weather Forecast", page_icon="🌦️", layout="wide")
theme.apply()


@st.cache_data(ttl=60)
def gpu_name() -> str | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


@st.cache_resource(max_entries=4)
def open_output(path: str, mtime: float) -> xr.Dataset:
    return xr.open_dataset(path).load()


def event_movie_tab(path, m, full, run_req, box, region_name, lead_days, use_case):
    """Panchali's obs-vs-forecast event movie and track map for the loaded run."""
    st.caption("Animated map of observations (ERA5) next to the forecast, with the regional "
               "series building up day by day, plus a map of the event's footprint and track.")
    missing = set(filter(None, full.attrs.get("missing_variables", "").split(",")))
    kinds = [k for k, var in (("precip", "tp"), ("heat", "t2m")) if var not in missing]
    if not kinds:
        st.info(f"{m.name} has neither rainfall nor 2 m temperature in this run.")
        return
    labels = {"precip": "Rainfall (daily total)", "heat": "Heat (daily max 2 m T)"}
    default = "heat" if use_case == "Heat" and "heat" in kinds else kinds[0]
    c1, c2, c3 = st.columns([2, 2, 2])
    kind = c1.radio("Event type", kinds, index=kinds.index(default), format_func=labels.get,
                    key="movie_kind")
    reduce = c2.selectbox("Regional series", ["mean", "max", "p95", "p99"], key="movie_reduce",
                          index=1 if kind == "heat" else 0,
                          format_func={"mean": "area mean", "max": "area maximum",
                                       "p95": "95th percentile", "p99": "99th percentile"}.get)
    auto = c3.checkbox("Automatic threshold", value=True, key="movie_auto",
                       help="High percentile of the observations (p95 heat, p97.5 rain)")
    threshold = None if auto else c3.number_input(
        "Threshold (°C)" if kind == "heat" else "Threshold (mm/day)",
        value=40.0 if kind == "heat" else 50.0, key="movie_thr")
    run_days = int(full.lead_time.max()) // 24
    req = movie.request_for(path, m.name, kind, run_req.init, min(lead_days, run_days), box,
                            region_name, reduce, threshold)
    s = movie.status(req)

    if s["state"] == "none":
        if st.button("Make event movie", type="primary", key="movie_go"):
            movie.launch(req)
            st.rerun()
        st.caption("The first movie for a new date fetches ERA5 observations from the cloud and "
                   "can take a few minutes; later ones reuse the cached data.")
    elif s["state"] == "running":
        st.info("Rendering the movie… (fetching observations and drawing one frame per day)")
        with st.expander("Log"):
            st.code(s["log"] or "…")
        time.sleep(3)
        st.rerun()
    elif s["state"] == "failed":
        st.error(f"Movie failed: {s.get('error')}")
        with st.expander("Log", expanded=True):
            st.code(s["log"])
        if st.button("Try again", key="movie_retry"):
            movie.launch(req)
            st.rerun()
    else:
        if s.get("gif"):
            st.image(s["gif"], caption="Observations vs forecast, by valid day")
            st.download_button("Download movie (GIF)", Path(s["gif"]).read_bytes(),
                               file_name=Path(s["gif"]).name, mime="image/gif", key="movie_dl")
        if s.get("png"):
            st.image(s["png"], caption="Event footprint and track")
            st.download_button("Download track map (PNG)", Path(s["png"]).read_bytes(),
                               file_name=Path(s["png"]).name, mime="image/png", key="track_dl")
        st.caption("Observations are ERA5 reanalysis. AIFS and several other models train on ERA5, "
                   "so this is not a fully independent check.")


ss = st.session_state
ss.setdefault("job_id", None)
ss.setdefault("result", None)        # (path, request fields) of the run being viewed

timings = catalog.load_timings()
saved = store.list_saved()
gpu = gpu_name()

# ── sidebar: developer options ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Developer options")
    synthetic = st.toggle("Use synthetic output", value=False,
                          help="Generate fake fields in the output format instead of running a model. "
                               "For building the visualizations before the model runners exist.")
    st.caption(f"Output folder: `{store.OUTPUT_DIR}`")
    if not saved.empty:
        st.markdown("**Saved runs**")
        st.dataframe(saved.drop(columns="path"), hide_index=True, width="stretch")

# ── header ───────────────────────────────────────────────────────────────────
theme.header("AI Forecast Lab", "Demo 3", f"{'GPU: ' + gpu if gpu else 'No GPU detected'}", live=bool(gpu))
theme.page_title("Running Your First AI Weather Forecast",
                 "Pick a model, a start date and a region, run the forecast on your Spark, "
                 "and explore temperature, rainfall and onset.")

ready = [m for m in catalog.MODELS if m.live]
free_gb = shutil.disk_usage(store.OUTPUT_DIR if store.OUTPUT_DIR.exists() else "/").free / 1e9
theme.stats([
    ("Models", str(len(catalog.MODELS)), theme.BLUE),
    ("Live on this Spark", f"{len(ready)} / {len(catalog.MODELS)}", theme.TEAL),
    ("Saved runs", str(len(saved)), theme.MUTED),
    ("Free disk", f"{free_gb:,.0f} GB", theme.ORANGE),
])

# ── Step 2a: model ───────────────────────────────────────────────────────────
theme.section("Select model", "Step 2 ·")
mode = st.radio("Mode", ["Deterministic", "Probabilistic"], horizontal=True, label_visibility="collapsed")
kind = "deterministic" if mode == "Deterministic" else "ensemble"
choices = [m for m in catalog.MODELS if m.kind == kind]

model = st.selectbox("Model", choices, format_func=lambda m: f"{m.name}  {catalog.timing_label(m, timings)}",
                     label_visibility="collapsed")
def model_tags(m: catalog.Model) -> str:
    kind_tag = theme.tag(m.kind, "ens" if m.kind == "ensemble" else "det")
    return kind_tag + (theme.tag("live", "ok") if m.live else theme.tag("saved runs only", "warn"))


theme.cards([{
    "name": m.name, "org": m.org, "tags": model_tags(m),
    "note": f"{catalog.timing_label(m, timings)} · starts from {m.init_source}",
    "color": theme.PURPLE if m.kind == "ensemble" else theme.TEAL,
    "selected": m.key == model.key, "dim": not m.live and not synthetic,
} for m in choices])
if model.notes:
    st.caption(f"ℹ️ {model.notes}")
if model.slow:
    st.caption("🐢 This model is slow on the Spark. In class, load a saved run where possible.")

# ── Step 2b: forecast setup ──────────────────────────────────────────────────
theme.section("Forecast setup")
c1, c2, c3, c4 = st.columns([2, 1, 2, 2])
default_init = min(max(date(2025, 7, 1), model.init_min), model.init_max)
init_day = c1.date_input("Start date (UTC)", value=default_init, min_value=model.init_min,
                         max_value=model.init_max, key=f"init_{model.key}")
init_hour = c2.selectbox("Hour", [0, 12], format_func=lambda h: f"{h:02d}Z")
lead_days = c3.slider("Lead time (days)", 1, model.max_lead_days, min(10, model.max_lead_days))
members = (c4.number_input("Ensemble members", 1, model.max_members, model.default_members)
           if model.kind == "ensemble" else 1)
if model.kind != "ensemble":
    c4.text_input("Ensemble members", "1 (deterministic)", disabled=True)
init = pd.Timestamp(init_day) + pd.Timedelta(hours=init_hour)

# ── Step 2c: region ──────────────────────────────────────────────────────────
theme.section("Region")
r1, r2 = st.columns([1, 3])
region_mode = r1.radio("Region", ["Country", "Custom box"], label_visibility="collapsed")
if region_mode == "Country":
    group_of = {c: g for g, cs in GROUPS.items() for c in cs}
    country = r2.selectbox("Country", COUNTRIES, format_func=lambda c: f"{c}  ·  {group_of[c]}")
    box = country_box(country)
    region_name = country
    r2.caption(f"Plot area (country plus a margin): {box.label()}")
else:
    b1, b2, b3, b4 = r2.columns(4)
    box = Box(b1.number_input("Lat min", -90.0, 90.0, 3.0, 0.25),
              b2.number_input("Lat max", -90.0, 90.0, 15.0, 0.25),
              b3.number_input("Lon min", -180.0, 180.0, 33.0, 0.25),
              b4.number_input("Lon max", -180.0, 180.0, 48.0, 0.25))
    region_name = "Custom region"
    if not box.is_valid():
        r2.error("Min must be below max; latitude within ±90°, longitude within ±180°.")

# ── Step 2d: use case ────────────────────────────────────────────────────────
theme.section("What do you want to look at?")
use_case = st.segmented_control("Use case", ["Heat", "Precipitation", "Onset"], default="Precipitation",
                                label_visibility="collapsed") or "Precipitation"
settings: dict = {}
u1, u2, u3, u4 = st.columns(4)
if use_case == "Heat":
    settings["heat_threshold_c"] = u1.number_input("Temperature threshold (°C)", value=35.0, step=0.5)
elif use_case == "Precipitation":
    settings["precip_threshold_mm"] = u1.number_input("Daily rainfall threshold (mm)", value=20.0, step=1.0)
else:
    preset = u1.selectbox("Onset definition", ["Ethiopia – Kiremt (Demo 5)", "India – monsoon (to be added)"])
    # Kiremt defaults match the Demo 5 benchmarking configuration.
    settings.update(
        wet_threshold_mm=u2.number_input("Wet-spell rainfall (mm)", value=20.0, step=1.0),
        wet_spell_days=u3.number_input("over N days", 1, 10, 3),
        dry_spell_days=u4.number_input("No dry spell of (days)", 1, 21, 7),
        dry_extent_days=21, dry_threshold_mm=1.0, preset=preset)
    if not 5 <= init.month <= 9:
        st.warning("The Kiremt onset window is May–September; onset is not meaningful for this start date.")

# ── Step 3: run ──────────────────────────────────────────────────────────────
theme.section("Run the forecast", "Step 3 ·")
req = store.RunRequest(model.key, init, lead_days * 24, int(members), synthetic)
saved_path = store.find_saved(req)
theme.pills([("Model", model.name), ("Mode", mode), ("Start", f"{init:%Y-%m-%d %HZ}"),
             ("Lead", f"{lead_days} days"), ("Members", str(members)), ("Region", region_name),
             ("Use case", use_case)] + ([("Output", "synthetic")] if synthetic else []))

can_run = (model.live or synthetic) and box.is_valid()
b1, b2 = st.columns([2, 3])
if saved_path:
    clicked = b1.button("Load saved run", type="primary", width="stretch", disabled=not box.is_valid())
    b2.markdown(theme.tag("saved run found — loads instantly", "ok"), unsafe_allow_html=True)
else:
    clicked = b1.button("Run forecast", type="primary", width="stretch",
                        disabled=not can_run or ss.job_id is not None)
    b2.markdown(theme.tag(f"will run on the Spark {catalog.timing_label(model, timings)}", "det")
                if can_run else theme.tag("no saved run for these settings, and live runs aren't set up here yet", "warn"),
                unsafe_allow_html=True)

if clicked:
    if saved_path:
        ss.result = (str(saved_path), req)
    else:
        ss.job_id = jobs.launch(req)
        ss.result = None
    st.rerun()

if ss.job_id:
    s = jobs.status(ss.job_id)
    if s["state"] in ("queued", "running", "unknown"):
        st.progress(s.get("progress", 0.0), text=f"{s.get('message', '')} · {s['elapsed_s']:.0f} s elapsed")
        with st.expander("Log"):
            st.code(jobs.log_tail(ss.job_id) or "…")
        time.sleep(2)
        st.rerun()
    elif s["state"] == "done":
        ss.result = (s["output"], req)
        ss.job_id = None
        st.success(f"Forecast finished in {s.get('runtime_s', s['elapsed_s']):.0f} s.")
        st.rerun()
    else:
        st.error(f"Run failed: {s.get('message')}")
        with st.expander("Log", expanded=True):
            st.code(jobs.log_tail(ss.job_id, 40))
        if st.button("Dismiss"):
            ss.job_id = None
            st.rerun()

# ── Results ──────────────────────────────────────────────────────────────────
if ss.result:
    path, run_req = ss.result
    p = Path(path)
    if not p.exists():
        st.warning("The saved output file is gone; run the forecast again.")
    else:
        full = open_output(path, p.stat().st_mtime)
        m = catalog.BY_KEY[run_req.model]
        theme.section("Results")
        ds = viz.crop(full, box, lead_days * 24)
        if ds.sizes["lat"] < 2 or ds.sizes["lon"] < 2:
            st.error("The selected region contains no grid points.")
        else:
            ctx = viz.VizContext(model_name=m.name, kind=m.kind, init=run_req.init,
                                 lead_hours=lead_days * 24, region_name=region_name, box=box,
                                 use_case=use_case, settings=settings, synthetic=bool(full.attrs.get("synthetic")))
            if ctx.synthetic:
                st.caption("⚠️ Synthetic output — not a real forecast.")
            first = {"Heat": "Temperature", "Precipitation": "Precipitation", "Onset": "Onset"}[use_case]
            names = [first] + [n for n in viz.VIEWS if n != first]
            tabs = st.tabs(names + ["Event movie (vs observations)"])
            for tab, name in zip(tabs, names):
                with tab:
                    viz.VIEWS[name](ds, ctx)
            with tabs[-1]:
                event_movie_tab(path, m, full, run_req, box, region_name, lead_days, use_case)
