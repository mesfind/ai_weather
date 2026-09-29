#from momp.io.output import save_score_results
from momp.stats.climatology import compute_climatological_onset_dataset
from momp.stats.bins import multi_year_forecast_obs_pairs, multi_year_climatological_forecast_obs_pairs
from momp.stats.score import calculate_brier_score, calculate_auc, calculate_rps, calculate_brier_score_climatology, calculate_auc_climatology, calculate_skill_scores
from momp.utils.printing import tuple_to_str
from momp.io.output import save_ref_score_results, load_ref_score_results
#from momp.lib.control import restore_args
from momp.utils.practical import restore_args
from momp.io.dict import extract_pd_bins
from momp.stats.parallel import parallel_climatological_forecast_obs_pairs
from momp.stats.parallel import parallel_forecast_obs_pairs
from momp.stats.parallel import parallel_climatological_onset_dataset
from momp.metrics.idr_calibration import calibrate_onset_probabilities


def create_score_results(*, BS, RPS, AUC, skill_score, 
                         ref_model, ref_model_dir, ref_model_var, ref_model_file_pattern, ref_model_unit_cvt,
                         years, years_clim, obs_dir, obs_file_pattern, obs_var,
                         thresh_file, thresh_var, wet_threshold,
                         date_filter_year, init_days, start_date, end_date,
                         model_dir, model_var, unit_cvt, file_pattern,
                         wet_init, wet_spell, dry_spell, dry_threshold, dry_extent, fallback_date, mok,
                         members, onset_percentage_threshold, max_forecast_day, day_bins,
                         apply_idr_calibration=False,
                         idr_min_valid=10, idr_min_onset_events=5,
                         idr_use_parallel=False, idr_n_jobs=None,
                         **kwargs):
    """
    apply_idr_calibration : bool
        If True, forecast probabilities are recalibrated via Leave-One-
        Year-Out Isotonic Distributional Regression (see
        momp.metrics.idr_calibration) before scoring. Only affects the
        probabilistic track (BS/RPS/AUC) — the raw `predicted_prob`
        column is preserved in `forecast_obs_df` alongside the new
        `calibrated_prob`/`idr_crps`/`idr_pit`/`idr_calibrated` columns
        for inspection, regardless of this flag. The reference model's
        forecast_obs_df (climatology, or a named ref_model) is
        calibrated on the same basis so the resulting skill score
        compares like with like.
    idr_min_valid : int
        Minimum valid training samples per grid cell per LOYO fold for
        IDR to attempt a fit (passed through to idr_loyo_cv).
    idr_min_onset_events : int
        Minimum observed-onset events (y > 0.1) required in a fold's
        training data for IDR to attempt a fit.
    idr_use_parallel, idr_n_jobs : joblib parallelism passthrough for
        the IDR LOYO grid loop.
    """

#    print("="*60)
#    print("S2S MONSOON ONSET SKILL SCORE ANALYSIS")
#    print("="*60)
#    print(f"Model: {model}")
#    print(f"Years: {years}")
#    print(f"Max forecast day: {max_forecast_day}")
#    print(f"Day bins: {day_bins}")
#    print(f"MOK filter: {mok}")
#    print("="*60)

    kwargs = restore_args(create_score_results, kwargs, locals())
#    from pprint import pprint
#    print(kwargs)

    parallel = kwargs.get('parallel')

    results = {}

    print("\n1. Processing forecast model...")

    import time
    start = time.perf_counter()

    # select bins without "Day before" and "Day After"
    #forecast_obs_df = multi_year_forecast_obs_pairs(**kwargs)
    if parallel:
        forecast_obs_df_all = parallel_forecast_obs_pairs(**kwargs)
    else:
        forecast_obs_df_all = multi_year_forecast_obs_pairs(**kwargs)
    forecast_obs_df = extract_pd_bins(forecast_obs_df_all, day_bins)

    end = time.perf_counter()
    print(f"Execution time: {end - start:.4f} seconds")
#    import sys
#    sys.exit()

#    import pandas as pd
#    pd.set_option('display.max_rows', None)
#    pd.set_option('display.max_columns', None)
#    pd.set_option('display.width', None)
#    pd.set_option('display.max_colwidth', None)
#    fields = ["init_time", "lat", "lon", "bin_label", "predicted_prob", "observed_onset"]
#    print("\n\n\n forecast_obs_df_all = ", forecast_obs_df_all[fields])
#    print("\n\n\n forecast_obs_df = ", forecast_obs_df[fields])
#    import sys
#    sys.exit()

    # ------------------------------------------------------------------
    # IDR calibration (probabilistic track only). Applied here, right
    # after binning and before any metric is computed, so BS/RPS/AUC
    # below always score whatever forecast_obs_df_for_scoring ends up
    # holding — calibrated if requested, raw otherwise. The full
    # forecast_obs_df (with both raw predicted_prob and, if calibrated,
    # calibrated_prob/idr_crps/idr_pit/idr_calibrated columns) is what
    # gets stored in results["forecast_obs_df"] and passed to
    # reliability-diagram plotting, so raw vs. calibrated is always
    # inspectable regardless of which one was actually scored.
    # ------------------------------------------------------------------
    if apply_idr_calibration:
        print("\n1b. Applying IDR (Isotonic Distributional Regression) calibration...")
        start_idr = time.perf_counter()
        forecast_obs_df = calibrate_onset_probabilities(
            forecast_obs_df,
            min_valid=idr_min_valid,
            min_onset_events=idr_min_onset_events,
            use_parallel=idr_use_parallel,
            n_jobs=idr_n_jobs,
        )
        end_idr = time.perf_counter()
        print(f"IDR calibration time: {end_idr - start_idr:.4f} seconds")

        n_calibrated = int(forecast_obs_df["idr_calibrated"].sum())
        n_total = len(forecast_obs_df)
        print(f"IDR calibrated {n_calibrated}/{n_total} rows "
              f"({n_total - n_calibrated} fell back to raw predicted_prob due to "
              f"insufficient training data in their LOYO fold).")

        forecast_obs_df_for_scoring = forecast_obs_df.assign(
            predicted_prob=forecast_obs_df["calibrated_prob"]
        )
    else:
        forecast_obs_df_for_scoring = forecast_obs_df

    results["forecast_obs_df"] = forecast_obs_df

    results["BS"] = calculate_brier_score(forecast_obs_df_for_scoring) if BS else None
    results["RPS"] = calculate_rps(forecast_obs_df_for_scoring) if RPS else None
    results["AUC"] = calculate_auc(forecast_obs_df_for_scoring) if AUC else None


    print("\n2. Processing reference model...")

    results["BS_ref"], results["RPS_ref"], results["AUC_ref"] = None, None, None
    results['skill_score'] = None

    # check if ref_results exist
    #dir_out = kwargs.get("dir_out")
    #model = kwargs.get("model")
    ##verification_window = kwargs.get("verification_window")
    ##filename = f'ref_scores_{model}_{tuple_to_str(verification_window)}window_{max_forecast_day}day.csv'
    #filename = f'ref_scores_{model}_{max_forecast_day}day.csv'
    #filename = os.path.join(dir_out, f"{filename}.pkl")

    #if filename.exists():
    #    results = load_ref_score_results(filename, results)
    #    if skill_score:
    #        skill_results = calculate_skill_scores(
    #        results["BS"], results["RPS"],
    #        results["BS_ref"], results["RPS_ref"]
    #        )
    #        results["skill_results"] = skill_results
    #    return results

    if ref_model == "climatology":

        if parallel:
            clim_onset = parallel_climatological_onset_dataset(**kwargs)
        else:
            clim_onset = compute_climatological_onset_dataset(**kwargs)


        #climatology_obs_df = multi_year_climatological_forecast_obs_pairs(clim_onset, **kwargs)
        
        import time
        start = time.perf_counter()

        if parallel:
            climatology_obs_df_all = parallel_climatological_forecast_obs_pairs(clim_onset, **kwargs)
        else:
            climatology_obs_df_all = multi_year_climatological_forecast_obs_pairs(clim_onset, **kwargs)
        climatology_obs_df = extract_pd_bins(climatology_obs_df_all, day_bins)

        end = time.perf_counter()
        print(f"clim_forecast_obs_paris Execution time: {end - start:.4f} seconds")
        #import sys
        #sys.exit()

        # ------------------------------------------------------------
        # If the forecast side was IDR-calibrated, the reference
        # (climatology) side must be put on the same footing before
        # BS_ref/RPS_ref/AUC_ref are computed, or the resulting skill
        # score (forecast vs. ref) would be comparing a calibrated
        # forecast against an uncalibrated reference — not a fair
        # comparison. climatology_obs_df is expected to carry the same
        # init_time/lat/lon/bin_label/predicted_prob/observed_onset
        # shape as forecast_obs_df; if that assumption doesn't hold for
        # your climatology construction, this will raise via
        # calibrate_onset_probabilities' own column check rather than
        # silently mis-scoring.
        # ------------------------------------------------------------
        if apply_idr_calibration:
            climatology_obs_df = calibrate_onset_probabilities(
                climatology_obs_df,
                min_valid=idr_min_valid,
                min_onset_events=idr_min_onset_events,
                use_parallel=idr_use_parallel,
                n_jobs=idr_n_jobs,
            )
            climatology_obs_df_for_scoring = climatology_obs_df.assign(
                predicted_prob=climatology_obs_df["calibrated_prob"]
            )
        else:
            climatology_obs_df_for_scoring = climatology_obs_df

        results["climatology_obs_df"] = climatology_obs_df
        
#        print("\n clim_onset = ", clim_onset)
#        import pandas as pd
#        pd.set_option('display.max_rows', None)
#        pd.set_option('display.max_columns', None)
#        pd.set_option('display.width', None)
#        pd.set_option('display.max_colwidth', None)
#        fields = ["init_time", "lat", "lon", "bin_label", "predicted_prob", "observed_onset"]
#        print("\n\n\n climatology_obs_df_all = ", climatology_obs_df_all[fields])
#        print("\n\n\n climatology_obs_df = ", climatology_obs_df[fields])
#        import sys
#        sys.exit()

        if BS:
            brier_ref = calculate_brier_score_climatology(climatology_obs_df_for_scoring)
            results["BS_ref"] = brier_ref
        else:
            results["BS_ref"] = None
        
        if RPS:
            rps_ref = calculate_rps(climatology_obs_df_for_scoring)
            results["RPS_ref"] = rps_ref
        else:
            results["RPS_ref"] = None

        if AUC:
            auc_ref = calculate_auc_climatology(climatology_obs_df_for_scoring)
            results["AUC_ref"] = auc_ref
        else:
            results["AUC_ref"] = None

    else:
        results["climatology_obs_df"] = None

        kwargs_ref = {**kwargs,
                      'model': ref_model,
                      'model_dir': ref_model_dir,
                      'model_var': ref_model_var,
                      'file_pattern': ref_model_file_pattern,
                      'unit_cvt': ref_model_unit_cvt
                      }

        #ref_obs_df = multi_year_forecast_obs_pairs(**kwargs_ref)

        if parallel:
            ref_obs_df_all = parallel_forecast_obs_pairs(**kwargs)
        else:
            ref_obs_df_all = multi_year_forecast_obs_pairs(**kwargs)
        ref_obs_df = extract_pd_bins(ref_obs_df_all, day_bins)

        # Same rationale as the climatology branch above: calibrate the
        # reference model's own forecast_obs_df before scoring it, so
        # BS_ref/RPS_ref/AUC_ref are on the same footing as the
        # (possibly calibrated) forecast-side BS/RPS/AUC above.
        if apply_idr_calibration:
            ref_obs_df = calibrate_onset_probabilities(
                ref_obs_df,
                min_valid=idr_min_valid,
                min_onset_events=idr_min_onset_events,
                use_parallel=idr_use_parallel,
                n_jobs=idr_n_jobs,
            )
            ref_obs_df_for_scoring = ref_obs_df.assign(
                predicted_prob=ref_obs_df["calibrated_prob"]
            )
        else:
            ref_obs_df_for_scoring = ref_obs_df

        if BS:
            brier_ref = calculate_brier_score(ref_obs_df_for_scoring)
            results["BS_ref"] = brier_ref
        else:
            results["BS_ref"] = None
        
        if RPS:
            rps_ref = calculate_rps(ref_obs_df_for_scoring)
            results["RPS_ref"] = rps_ref
        else:
            results["RPS_ref"] = None

        if AUC:
            auc_ref = calculate_auc(ref_obs_df_for_scoring)
            results["AUC_ref"] = auc_ref
        else:
            results["AUC_ref"] = None


    #save_ref_score_results(results, filename)


    print("results[BS]  =  ", results["BS"])
    print("results[BS_ref]  =  ", results["BS_ref"])
    print("results[RPS]  =  ", results["RPS"])
    print("results[RPS_ref]  =  ", results["RPS_ref"])
    if skill_score:

        skill_results = calculate_skill_scores(
        results["BS"], results["RPS"],
        results["BS_ref"], results["RPS_ref"]
        )

        results["skill_results"] = skill_results

#    if save_csv_score:
#        save_score_results(results, model, max_forecast_day))


    return results