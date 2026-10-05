# Demo 3 — Running Your First AI Weather Forecast

A Streamlit app, running on the DGX Spark, for running AI weather models. Participants choose a model, start date, lead time, region and use case (heat / precipitation / onset), run the forecast (or load a saved run), and explore the results.

**Status: skeleton.**
- The full app flow works end to end with *synthetic* output.
- Real model runners are added one at a time (see "Adding a model").
- The result plots are placeholders until the visualization module is plugged in (see "For the visualization module").

## Run it

**On the Spark:**

```bash
cd ~/demo3_ai_forecast
docker run -d --name demo3-ui -p 127.0.0.1:8501:8501 -v $PWD:/app -w /app \
  demo45-benchmarks streamlit run app.py --server.headless true --server.port 8501
```

**From your laptop:** open a tunnel, then browse to http://localhost:8501.

```bash
ssh -L 8501:localhost:8501 <you>@hcwfpgx
```

`demo45-benchmarks` is a temporary base image. It will be replaced by the Demo 3 image, which also holds the model environments.

## Layout

```
app.py                 Streamlit page: steps 2–3 and results
demo3/catalog.py       the 7 models (type, environment, lead-time limit, members, status)
demo3/regions.py       program countries → padded plotting boxes
demo3/contract.py      the output-file format every runner writes
demo3/store.py         where outputs live; reusing saved runs
demo3/jobs.py          launches runs in the background and tracks progress
demo3/viz.py           result views (PLACEHOLDERS — the plug-in point)
demo3/theme.py         styling, matching the Demo 5 platform
runners/run_model.py   runner entry point (+ synthetic output generator)
timings.json           measured Spark runtimes shown in brackets in the app
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
