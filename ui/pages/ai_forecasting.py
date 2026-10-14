# ui/pages/ai_forecasting.py
"""
AI Forecast Demo — Running Your First AI Weather Forecast.
Integrated into Onset Forecast Platform.
"""
from __future__ import annotations

import base64
import shutil
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st
import xarray as xr

try:
    _root = Path(__file__).resolve().parent.parent  # ui/pages -> ui -> ROOT
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
        
    from apps.forecasting import catalog, compare, jobs, movie, results, store, theme
    from apps.forecasting import onset, onset_view
    from apps.forecasting.regions import COUNTRIES, GROUPS, Box, country_box
except ImportError as e:
    st.error(f"Failed to import forecasting modules: {e}")
    st.info("Please ensure the 'apps/forecasting' package exists in your project root.")
    st.stop()


@st.cache_data(ttl=60)
def gpu_name() -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


@st.cache_resource(max_entries=4)
def open_output(path: str, mtime: float) -> xr.Dataset:
    return xr.open_dataset(path).load()


def event_results(path, m, full, run_req, box, region_name, lead_days, use_case):
    """Panchali's obs-vs-forecast event movie and event-track map for the loaded run."""
    kind, var = ("heat", "t2m") if use_case == "Heat" else ("precip", "tp")
    if var in catalog.missing_in_run(full):
        other = "Precipitation" if use_case == "Heat" else "Heat"
        st.markdown(f'<div class="ibox amber">ⓘ {m.name} does not forecast {catalog.VAR_LABELS[var]}, so there is no '
                    f'{use_case.lower()} movie for this run. Switch the use case to {other}, or '
                    'choose another model.</div>', unsafe_allow_html=True)
        return

    st.caption("Animated map of observations (ERA5) next to the forecast, with the regional "
               "series building up day by day, then a map of the event's footprint and track.")
    c2, c3 = st.columns(2)
    reduce = c2.selectbox("Regional series", ["mean", "max", "p95", "p99"], key="forecast_movie_reduce",
                          index=1 if kind == "heat" else 0,
                          format_func={"mean": "area mean", "max": "area maximum",
                                       "p95": "95th percentile", "p99": "99th percentile"}.get)
    auto = c3.checkbox("Automatic threshold", value=True, key="forecast_movie_auto",
                       help="High percentile of the observations (p95 heat, p97.5 rain)")
    threshold = None if auto else c3.number_input(
        "Threshold (°C)" if kind == "heat" else "Threshold (mm/day)",
        value=40.0 if kind == "heat" else 50.0, key="forecast_movie_thr")

    run_days = int(full.lead_time.max()) // 24
    days = min(lead_days, run_days)
    obs_end = pd.Timestamp(catalog.era5_dates()[1])

    if run_req.init >= obs_end:
        st.markdown(f'<div class="ibox sky"> Observations (ERA5) currently end on {obs_end:%-d %b %Y}, so there is nothing yet '
                    f'to compare this {run_req.init:%-d %b %Y} forecast with.</div>', unsafe_allow_html=True)
        return
    if run_req.init + pd.Timedelta(days=days) > obs_end:
        days = max(1, (obs_end - run_req.init).days)
        st.markdown(f'<div class="ibox sky">ⓘ Observations (ERA5) currently end on {obs_end:%-d %b %Y}, so the movie covers '
                    f'the first {days} day{"s" if days > 1 else ""} of the forecast.</div>', unsafe_allow_html=True)

    req = movie.request_for(path, m.name, kind, run_req.init, days, box,
                            region_name, reduce, threshold)
    s = movie.status(req)

    if s["state"] == "none":
        if st.button("Make event movie", type="primary", key="forecast_movie_go"):
            movie.launch(req)
            st.rerun()
        st.caption("The first movie for a new date fetches ERA5 observations from the cloud and "
                   "can take a few minutes; later ones reuse the cached data.")
    elif s["state"] == "running":
        frac, what = movie.progress(req)
        st.progress(frac, text=what)
        with st.expander("Log", key="forecast_movie_log_expander"):
            st.code(s["log"] or "…")
        time.sleep(3)
        st.rerun()
    elif s["state"] == "failed":
        st.error(f"Movie failed: {s.get('error')}")
        with st.expander("Log", expanded=True, key="forecast_movie_fail_log"):
            st.code(s["log"])
        if st.button("Try again", key="forecast_movie_retry"):
            movie.launch(req)
            st.rerun()
    else:
        if s.get("gif"):
            gif64 = base64.b64encode(Path(s["gif"]).read_bytes()).decode()
            st.html(f'<img src="data:image/gif;base64,{gif64}" style="width:100%" '
                    'alt="Observations vs forecast, by valid day">')
            st.caption("Observations vs forecast, by valid day")
            st.download_button("Download movie (GIF)", Path(s["gif"]).read_bytes(),
                               file_name=Path(s["gif"]).name, mime="image/gif", key="forecast_movie_dl")
        if s.get("png"):
            st.image(s["png"], caption="Event footprint and track")
            st.download_button("Download track map (PNG)", Path(s["png"]).read_bytes(),
                               file_name=Path(s["png"]).name, mime="image/png", key="forecast_track_dl")
        st.caption("Observations are ERA5 reanalysis. AIFS and several other models train on ERA5, "
                   "so this is not a fully independent check.")


def render():
    """Main entry point called by app.py"""
    ss = st.session_state
    ss.setdefault("job_id", None)
    ss.setdefault("result", None)

    timings = catalog.load_timings()
    saved = store.list_saved()
    gpu = gpu_name()

    # ── Sidebar: Developer Options ───────────────────────────────────────
    with st.sidebar:
        st.markdown("### Developer options")
        synthetic = st.toggle("Use synthetic output", value=False, key="forecast_synthetic",
                              help="Generate fake fields instead of running a model. "
                                   "For building visualizations before model runners exist.")
        st.caption(f"Output folder: `{store.OUTPUT_DIR}`")
        if not saved.empty:
            st.markdown("**Saved runs**")
            st.dataframe(saved.drop(columns="path"), hide_index=True, width="stretch")

    # ── Header & Stats (Using forecast.css KPI cards) ───────────────────
    st.title("Running Your First AI Weather Forecast")
    st.caption("Pick a model, a start date and a region, run the forecast on your Spark, "
               "and compare it with what was observed.")

    ready = [m for m in catalog.MODELS if m.live]
    free_gb = shutil.disk_usage(store.OUTPUT_DIR if store.OUTPUT_DIR.exists() else "/").free / 1e9

    # FIX: Use st.columns(4) to force horizontal layout instead of CSS grid
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f'''
        <div class="forecast-kpi-card forecast-kpi-high">
            <div class="forecast-kpi-accent"></div>
            <div class="forecast-kpi-content">
                <div class="forecast-kpi-label">Total Models</div>
                <div class="forecast-kpi-value">{len(catalog.MODELS)}</div>
                <div class="forecast-kpi-subtitle">Available in catalog</div>
            </div>
        </div>''', unsafe_allow_html=True)

    with col2:
        st.markdown(f'''
        <div class="forecast-kpi-card forecast-kpi-moderate">
            <div class="forecast-kpi-accent"></div>
            <div class="forecast-kpi-content">
                <div class="forecast-kpi-label">Live on Spark</div>
                <div class="forecast-kpi-value">{len(ready)} / {len(catalog.MODELS)}</div>
                <div class="forecast-kpi-subtitle">Ready to run</div>
            </div>
        </div>''', unsafe_allow_html=True)

    with col3:
        st.markdown(f'''
        <div class="forecast-kpi-card forecast-kpi-uncertain">
            <div class="forecast-kpi-accent"></div>
            <div class="forecast-kpi-content">
                <div class="forecast-kpi-label">Saved Runs</div>
                <div class="forecast-kpi-value">{len(saved)}</div>
                <div class="forecast-kpi-subtitle">Cached locally</div>
            </div>
        </div>''', unsafe_allow_html=True)

    with col4:
        st.markdown(f'''
        <div class="forecast-kpi-card forecast-kpi-high">
            <div class="forecast-kpi-accent"></div>
            <div class="forecast-kpi-content">
                <div class="forecast-kpi-label">Free Disk Space</div>
                <div class="forecast-kpi-value">{free_gb:,.0f} <span style="font-size:1rem;color:var(--muted)">GB</span></div>
                <div class="forecast-kpi-subtitle">Available for output</div>
            </div>
        </div>''', unsafe_allow_html=True)

    # ── Step 2a: Model Selection ────────────────────────────────────────
    st.markdown('<div class="sec-hd teal"><span class="sec-hd-accent"></span>Select model</div>', unsafe_allow_html=True)
    
    mode = st.radio("Mode", ["Deterministic", "Probabilistic"], horizontal=True, 
                    label_visibility="collapsed", key="forecast_mode")
    kind = "deterministic" if mode == "Deterministic" else "ensemble"
    choices = [m for m in catalog.MODELS if m.kind == kind]

    model_key = st.selectbox("Model", [m.key for m in choices], label_visibility="collapsed",
                             format_func=lambda k: f"{catalog.BY_KEY[k].name}  "
                                                   f"{catalog.timing_label(catalog.BY_KEY[k], timings)}",
                             key="forecast_model_select")
    model = catalog.BY_KEY[model_key]

    def model_tags(m: catalog.Model) -> str:
        kind_tag = "ENS" if m.kind == "ensemble" else "DET"
        status_tag = "LIVE" if m.live else "SAVED ONLY"
        return f"{kind_tag} · {status_tag}"

    num_cols = min(len(choices), 3)
    cols = st.columns(num_cols)
    
    for idx, m in enumerate(choices):
        with cols[idx % num_cols]:
            is_selected = m.key == model.key
            accent_color = "#10B981" if m.kind == "ensemble" else "#0F766E"
            
            st.markdown(f'''
            <div class="model-card {'selected' if is_selected else ''}" style="--card-accent: {accent_color};">
                <div class="mc-title">{m.name}</div>
                <div class="mc-desc">{m.org}</div>
                <div class="mc-badge">{model_tags(m)}</div>
            </div>
            ''', unsafe_allow_html=True)

    if model.notes:
        st.markdown(f'<div class="ibox sky">ⓘ {model.notes}</div>', unsafe_allow_html=True)
    if model.slow:
        st.markdown('<div class="ibox amber">ⓘ This model is slow on the Spark. In class, load a saved run where possible.</div>', unsafe_allow_html=True)
    
    st.markdown("**What this model needs on the Spark**")
    results.requirements(model, timings, store.OUTPUT_DIR)

    # ── Step 2b: Forecast Setup ─────────────────────────────────────────
    st.markdown('<div class="sec-hd sky"><span class="sec-hd-accent"></span>Forecast setup</div>', unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns([2, 1, 2, 2])
    init_lo, init_hi = model.date_range()
    default_init = min(max(date(2025, 7, 1), init_lo), init_hi)
    init_day = c1.date_input("Start date (UTC)", value=default_init, min_value=init_lo,
                             max_value=init_hi, key=f"forecast_init_{model.key}")
    init_hour = c2.selectbox("Hour", [0, 12], format_func=lambda h: f"{h:02d}Z", key="forecast_hour")
    lead_days = c3.slider("Lead time (days)", 1, model.max_lead_days, min(10, model.max_lead_days), key="forecast_lead_days")
    members = (c4.number_input("Ensemble members", 1, model.max_members, model.default_members, key="forecast_members")
               if model.kind == "ensemble" else 1)
    if model.kind != "ensemble":
        c4.text_input("Ensemble members", "1 (deterministic)", disabled=True, key="forecast_members_text")
    init = pd.Timestamp(init_day) + pd.Timedelta(hours=init_hour)
    st.caption(f"ⓘ Start dates for {model.name}: {init_lo:%-d %b %Y} – {init_hi:%-d %b %Y}. "
               f"{model.date_reason()}")

    # ─ Step 2c: Region ─────────────────────────────────────────────────
    st.markdown('<div class="sec-hd violet"><span class="sec-hd-accent"></span>Region</div>', unsafe_allow_html=True)
    
    r1, r2 = st.columns([1, 3])
    region_mode = r1.radio("Region", ["Country", "Custom box"], label_visibility="collapsed", key="forecast_region_mode")
    if region_mode == "Country":
        group_of = {c: g for g, cs in GROUPS.items() for c in cs}
        country = r2.selectbox("Country", COUNTRIES, format_func=lambda c: f"{c}  ·  {group_of[c]}", key="forecast_country")
        box = country_box(country)
        region_name = country
        r2.caption(f"Plot area (country plus a margin): {box.label()}")
    else:
        b1, b2, b3, b4 = r2.columns(4)
        box = Box(b1.number_input("Lat min", -90.0, 90.0, 3.0, 0.25, key="forecast_lat_min"),
                  b2.number_input("Lat max", -90.0, 90.0, 15.0, 0.25, key="forecast_lat_max"),
                  b3.number_input("Lon min", -180.0, 180.0, 33.0, 0.25, key="forecast_lon_min"),
                  b4.number_input("Lon max", -180.0, 180.0, 48.0, 0.25, key="forecast_lon_max"))
        region_name = "Custom region"
        if not box.is_valid():
            r2.error("Min must be below max; latitude within ±90°, longitude within ±180°.")

    # ── Step 2d: Use Case ────────────────────────────────────────────────
    st.markdown('<div class="sec-hd amber"><span class="sec-hd-accent"></span>What do you want to look at?</div>', unsafe_allow_html=True)
    
    use_case = st.segmented_control("Use case", ["Heat", "Precipitation", "Onset"], default="Precipitation",
                                    label_visibility="collapsed", key="forecast_use_case") or "Precipitation"
    st.caption("**Heat:** daily maximum 2 m temperature (°C).  **Precipitation:** daily rainfall "
               "total (mm). **Onset:** when the rainy season starts.")

    problems = catalog.check(model, init, lead_days, int(members), box, region_name, use_case)
    for level, msg in problems:
        (st.error if level == "error" else st.warning)(msg)
    blocked = any(level == "error" for level, _ in problems)

    # ── Step 3: Run ─────────────────────────────────────────────────────
    st.markdown('<div class="sec-hd green"><span class="sec-hd-accent"></span>Run the forecast</div>', unsafe_allow_html=True)
    
    req = store.RunRequest(model.key, init, lead_days * 24, int(members), synthetic)
    saved_path = store.find_saved(req)
    remote = None if saved_path else store.find_remote(req)

    # Render run configuration tags using benchmarking.css metric-pills
    tags = [("Model", model.name), ("Mode", mode), ("Start", f"{init:%Y-%m-%d %HZ}"),
            ("Lead", f"{lead_days} days"), ("Members", str(members)), ("Region", region_name),
            ("Use case", use_case)]
    
    pills_html = '<div class="kpi-row" style="margin-top:0.5rem;">'
    for label, val in tags:
        pills_html += f'<span class="metric-pill"><strong>{label}:</strong> {val}</span>'
    if synthetic:
        pills_html += '<span class="metric-pill"><strong>Output:</strong> synthetic</span>'
    pills_html += '</div>'
    st.markdown(pills_html, unsafe_allow_html=True)

    can_run = (model.live or synthetic) and box.is_valid() and not blocked
    b1, b2 = st.columns([2, 3])
    
    if saved_path:
        clicked = b1.button("Load saved run", type="primary", width="stretch",
                            disabled=not box.is_valid() or blocked, key="forecast_load_saved")
        b2.markdown('<span class="status-chip success">✓ Saved run found — loads instantly</span>', unsafe_allow_html=True)
    elif remote:
        if b1.button(f"Download saved run ({remote[1] / 1e6:,.0f} MB)", type="primary", width="stretch",
                     disabled=not box.is_valid() or blocked, key="forecast_download_remote"):
            with st.spinner("Downloading the saved run from the shared bucket…"):
                ss.result = (str(store.download(model.key, remote[0])), req)
            st.rerun()
        b2.markdown('<span class="status-chip success">✓ Saved run in the shared bucket</span>', unsafe_allow_html=True)
        clicked = False
    else:
        clicked = b1.button("Run forecast", type="primary", width="stretch",
                            disabled=not can_run or ss.job_id is not None, key="forecast_run_btn")
        if can_run:
            b2.markdown(f'<span class="status-chip idle"> Will run on the Spark ({catalog.timing_label(model, timings)})</span>', unsafe_allow_html=True)
        elif blocked:
            b2.markdown('<span class="status-chip error">⚠ Fix the settings above first</span>', unsafe_allow_html=True)
        else:
            b2.markdown(f'<span class="status-chip error">⚠ No saved run, and {model.name} can\'t run live yet</span>', unsafe_allow_html=True)

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
            with st.expander("Log", key="forecast_job_log"):
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
            with st.expander("Log", expanded=True, key="forecast_job_fail_log"):
                st.code(jobs.log_tail(ss.job_id, 40))
            if st.button("Dismiss", key="forecast_dismiss_error"):
                ss.job_id = None
                st.rerun()

    # ── Results ─────────────────────────────────────────────────────────
    if ss.result:
        path, run_req = ss.result
        p = Path(path)
        if not p.exists():
            st.warning("The saved output file is gone; run the forecast again.")
        else:
            full = open_output(path, p.stat().st_mtime)
            m = catalog.BY_KEY[run_req.model]
            st.markdown('<div class="sec-hd teal"><span class="sec-hd-accent"></span>Results</div>', unsafe_allow_html=True)
            
            if full.attrs.get("synthetic"):
                st.markdown('<div class="ibox amber">ⓘ Synthetic output — not a real forecast.</div>', unsafe_allow_html=True)
            
            with st.expander(f"How this forecast was made · {m.name}, started {run_req.init:%-d %b %Y %HZ}", key="forecast_pipeline_expander"):
                results.pipeline(path, full, m, timings)

            first = "Onset (vs observations)" if use_case == "Onset" else "Event movie (vs observations)"
            movie_tab, maps_tab = st.tabs([first, "Forecast maps"])
            with movie_tab:
                if use_case == "Onset":
                    onset_view.view(path, p.stat().st_mtime, full, m, run_req, box, region_name, lead_days)
                else:
                    event_results(path, m, full, run_req, box, region_name, lead_days, use_case)
            with maps_tab:
                ds = results.crop(full, box, lead_days * 24)
                if ds.sizes["lat"] < 2 or ds.sizes["lon"] < 2:
                    st.error(f"{region_name} holds fewer than 2 grid points of {m.name}'s "
                             f"{m.grid_deg:g}° grid; choose a larger region.")
                else:
                    results.field_maps(ds, m, box, region_name)

            data = results.netcdf_bytes(path, p.stat().st_mtime,
                                        (box.lat_min, box.lat_max, box.lon_min, box.lon_max),
                                        lead_days * 24)
            st.download_button(f"Download this forecast for {region_name} (NetCDF, "
                               f"{len(data) / 1e6:.1f} MB)", data, mime="application/x-netcdf",
                               file_name=f"{m.key}_{run_req.init:%Y%m%dT%H}_{lead_days}d_"
                                         f"{region_name.replace(' ', '_').lower()}.nc",
                               key="forecast_download_nc")
            st.caption(f"Variables tp (mm per 6 h), t2m (K), z500 (m² s⁻²) by member and lead time. "
                       f"The full global file is on the Spark at `{p.relative_to(store.OUTPUT_DIR.parent)}` "
                       f"({p.stat().st_size / 1e6:.0f} MB).")

    # ── Compare Models ──────────────────────────────────────────────────
    st.markdown('<div class="sec-hd violet"><span class="sec-hd-accent"></span>Compare models</div>', unsafe_allow_html=True)
    st.caption(f"The same start ({init:%-d %b %Y %HZ}), lead time ({lead_days} days) and region "
               f"({region_name}) for several models, as one time series "
               + (" with each model's onset" if use_case == "Onset" else "") + ".")
    
    compare.section(init, lead_days, box, region_name, use_case, synthetic,
                    ss.get("onset_defn") or onset.DEFINITIONS[onset.default_definition(region_name)],
                    main_job_running=ss.job_id is not None)