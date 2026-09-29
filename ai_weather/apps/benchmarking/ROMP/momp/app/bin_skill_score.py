from dataclasses import asdict
from itertools import product

from momp.metrics.skill import create_score_results
from momp.graphics.heatmap import create_heatmap
from momp.graphics.reliability import plot_reliability_diagram
from momp.graphics.panel_portrait_skill import panel_portrait_bss_auc
from momp.graphics.panel_bar_skill import panel_bar_bss_rpss_auc
from momp.io.output import save_score_results
from momp.lib.control import iter_list, make_case
from momp.lib.convention import Case
from momp.lib.loader import get_cfg, get_setting
from momp.app.ens_spatial_far_mr_mae import ens_spatial_far_mr_mae_map
from momp.utils.printing import tuple_to_str
from momp.io.dict import select_key_at_level
from momp.lib.control import filter_bins_in_window
from collections import defaultdict


# NOTE: no module-level `cfg, setting = get_cfg(), get_setting()` here.
# That pattern freezes cfg/setting at import time (a function default
# argument is evaluated once, when the function is *defined*), which,
# combined with Python's sys.modules import cache, was the root cause
# of mode selection silently not taking effect. cfg/setting are now
# real parameters, fetched fresh at call time if not explicitly passed.

def skill_score_in_bins(cfg=None, setting=None):
    cfg = cfg if cfg is not None else get_cfg()
    setting = setting if setting is not None else get_setting()

    # only execute for ensemble (probabilistic) forecasts
    if not cfg.probabilistic:
        return

    result_overall = defaultdict(dict)
    result_binned = defaultdict(dict)

    layout_pool = iter_list(vars(cfg))

    for combi in product(*layout_pool):
        case = make_case(Case, combi, vars(cfg))

        print(f"{'='*50}")
        print(f"processing {case.model} onset evaluation for verification window "
                f"{case.verification_window}, case: {case.case_name}")

        day_bins_filtered = filter_bins_in_window(case.day_bins, case.verification_window)
        case.day_bins = day_bins_filtered

        case_cfg = {**asdict(setting), **asdict(case)}

        # WORKAROUND: date_filter_year (and potentially other config keys)
        # are present on cfg but not declared as fields on the Setting/Case
        # dataclasses, so dataclasses.asdict() silently drops them when
        # building case_cfg. Downstream, get_initialization_dates_monthly()
        # requires date_filter_year as a keyword-only argument and raises
        # a TypeError if it's missing. Falling back to cfg directly here
        # is a stopgap; the proper fix is adding date_filter_year (and
        # auditing for other silently-dropped keys) as a real field on
        # Setting or Case in momp/lib/convention.py.
        case_cfg.setdefault("date_filter_year", cfg.date_filter_year)

        import time
        start = time.perf_counter()

        score_results = create_score_results(**case_cfg)

        end = time.perf_counter()
        print(f"\n\n\n skill score Execution time: {end - start:.4f} seconds\n\n\n")

        if case_cfg['save_csv_score']:
            binned_data, overall_scores = save_score_results(score_results, **case_cfg)

        window_str = tuple_to_str(case.verification_window)
        result_binned[case.model][window_str] = binned_data
        result_overall[case.model][window_str] = overall_scores

        if case_cfg['plot_heatmap_bss_auc']:
            create_heatmap(score_results, **case_cfg)

        if case_cfg['plot_reliability']:
            plot_reliability_diagram(score_results["forecast_obs_df"], **case_cfg)

    max_forecast_day = cfg.max_forecast_day

    if 2 > 3:
        import pickle
        import os
        fout = os.path.join(cfg.dir_out, f"combi_binned_skill_scores_{max_forecast_day}day.pkl")
        with open(fout, "wb") as f:
            pickle.dump(result_binned, f)

        fout = os.path.join(cfg.dir_out, f"combi_overall_skill_scores_{max_forecast_day}day.pkl")
        with open(fout, "wb") as f:
            pickle.dump(result_overall, f)

    from pprint import pprint

    if case_cfg['plot_panel_heatmap_skill']:
        for verification_window in cfg.verification_window_list:
            window_str = tuple_to_str(verification_window)
            result_binned_window = select_key_at_level(result_binned, 2, window_str)
            print("\n\n\n result_binned_window = ", result_binned_window)
            pprint(result_binned_window)
            panel_portrait_bss_auc(result_binned_window, verification_window, **vars(cfg))

    if case_cfg['plot_bar_bss_rpss_auc']:
        for verification_window in cfg.verification_window_list:
            window_str = tuple_to_str(verification_window)
            result_overall_window = select_key_at_level(result_overall, 2, window_str)
            print("\n\n\n result_overall_window = ", result_overall_window)
            panel_bar_bss_rpss_auc(result_overall_window, verification_window, **vars(cfg))


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    skill_score_in_bins()