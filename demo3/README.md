# Demo 3 — Running Your First AI Weather Forecast

A Streamlit app, running on the DGX Spark, for running AI weather models. Participants choose a model, start date, lead time, region and use case (heat / precipitation / onset), run the forecast (or load a saved run), and explore the results.

**Status:**
- All 7 models run live on the Spark from one Docker image (`docker/`), and saved runs load instantly.
- The event movie tab (forecast vs observations) is Panchali's `demo3/event_movie/forecast_event_movie.py`.
- The other result tabs are placeholders until the visualization module is plugged in (see "For the visualization module").

## Run it

**On the Spark** (aarch64; build once, about an hour, mostly compiling two libraries):

```bash
cd ~/ai_weather/demo3
docker build -t demo3 -f docker/Dockerfile .
bash docker/run.sh            # or DEV=1 bash docker/run.sh to use this folder's code live
```

`run.sh` keeps saved runs in `outputs/` and model weights (~90 GB, downloaded on each model's
first run) in `~/.cache/demo3`; set `DEMO3_CACHE` to reuse an existing cache.

**From your laptop:** open a tunnel, then browse to http://localhost:8501.

```bash
ssh -L 8501:localhost:8501 <you>@hcwfpgx
```

**After a reboot**, if the container won't start with "unresolvable CDI devices", Docker started
before the GPU list existed: `systemctl --user restart docker && docker start demo3`.

**Environments in the image** (`/opt/envs/<env>`, pinned in `docker/locks/`):
`e2s018` Atlas CRPS, Aurora 1.5, AIFS v2 single · `e2s018ens` AIFS v2 ENS ·
`graphcast` GraphCast and NeuralGCM · `fgn` FGN · `ui` the page.

## Layout

```
app.py                 Streamlit page: steps 2–3 and results
demo3/catalog.py       the 7 models (type, environment, lead-time limit, members)
demo3/regions.py       program countries → padded plotting boxes
demo3/contract.py      the output-file format every runner writes
demo3/store.py         where outputs live; reusing saved runs
demo3/jobs.py          launches runs in the background and tracks progress
demo3/viz.py           result views (PLACEHOLDERS — the plug-in point)
demo3/theme.py         styling, matching the Demo 5 platform
runners/run_model.py   runner entry point (+ synthetic output generator)
timings.json           measured Spark runtimes shown in brackets in the app
docker/                the image: Dockerfile, run.sh, per-environment package locks
outputs/               saved runs: outputs/{model}/{YYYYMMDDTHH}_{lead}h_m{members}.nc
```

## Output format (what the plots receive)

Each run is one NetCDF file covering the whole globe:

| | |
|---|---|
| dims | `ensemble, lead_time, lat, lon` (`ensemble` has size 1 for deterministic models) |
| `lead_time` | integer hours after the start time: 0, 6, 12, … |
| `lat` / `lon` | ascending; lon runs −180 to 180 |
| `tp` | precipitation accumulated over the previous 6 h, in **mm** (NaN at lead 0) |
| `t2m` | 2 m temperature, in **K** |
| `z500` | 500 hPa geopotential, in **m² s⁻²** (divide by 9.80665 to get height in m) |
| coords | `valid_time` (along lead_time), `init_time` |
| attrs | `model`, `model_name`, `init_source`, `members`, `synthetic` |

## For the visualization module

A view is a function that draws into Streamlit:

```python
def view(ds: xr.Dataset, ctx: VizContext) -> None: ...
```

- `ds` is already cropped to the chosen region and lead time.
- `ctx` holds everything else: model name and type, start time, region name and box, use case, and the thresholds the user set (`ctx.settings`, e.g. `heat_threshold_c`, `precip_threshold_mm`, `wet_threshold_mm`, `wet_spell_days`, `dry_spell_days`).
- Use `st.pyplot`, `st.plotly_chart` or `st.image` to draw.

To plug in your plots, replace the placeholder functions in `demo3/viz.py`, or point `VIEWS` at your own module. `viz.daily_precip(ds)` gives daily totals if you need them.

To get test data without a GPU, leave **Use synthetic output** on in the sidebar and press Run. This writes a fake file in exactly this format.

## Adding a model

1. Create `runners/runner_<model_key>.py` with `run(init, lead_hours, members, report) -> xr.Dataset`, returning `tp`, `t2m` and `z500` in the units above.
2. Make sure the model's environment exists at `/opt/envs/<env>/bin/python` inside the container.
3. Set `status="ready"` for that model in `demo3/catalog.py`.
4. Add its measured runtime to `timings.json`.
