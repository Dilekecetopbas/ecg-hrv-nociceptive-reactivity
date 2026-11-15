import os
import numpy as np
import pandas as pd

def test_files_exist():
    assert os.path.isfile("/app/metrics.csv"), "Missing /app/metrics.csv"
    assert os.path.isfile("/app/stats.csv"), "Missing /app/stats.csv"

def test_metrics_schema_and_physiology():
    df = pd.read_csv("/app/metrics.csv")
    expected_cols = [
        "subject_id", "sex", "condition", "clean_window_ratio",
        "mean_hr_bpm", "sdnn_ms", "rmssd_ms", "hf_power_ms2", "ln_hf"
    ]
    assert list(df.columns) == expected_cols
    assert len(df) == 48
    assert not df.isna().any().any()

    df_sorted = df.sort_values(["subject_id", "condition"], ascending=[True, True]).reset_index(drop=True)
    pd.testing.assert_frame_equal(df.reset_index(drop=True), df_sorted)

    assert np.allclose(df["clean_window_ratio"], 16.0 / 18.0, atol=1e-3)
    assert ((df["mean_hr_bpm"] >= 45.0) & (df["mean_hr_bpm"] <= 130.0)).all()
    assert ((df["sdnn_ms"] >= 5.0) & (df["sdnn_ms"] <= 150.0)).all()
    assert ((df["rmssd_ms"] >= 5.0) & (df["rmssd_ms"] <= 150.0)).all()
    assert (df["hf_power_ms2"] > 10.0).all()

    base_hr = df[df["condition"] == "baseline"]["mean_hr_bpm"].mean()
    pain_hr = df[df["condition"] == "pain"]["mean_hr_bpm"].mean()
    assert pain_hr > base_hr + 5.0

    base_lnhf = df[df["condition"] == "baseline"]["ln_hf"].mean()
    pain_lnhf = df[df["condition"] == "pain"]["ln_hf"].mean()
    assert pain_lnhf < base_lnhf - 0.5

def test_stats_schema_and_significance():
    df = pd.read_csv("/app/stats.csv")
    assert list(df.columns) == ["outcome", "term", "estimate", "std_err", "z_value", "p_value"]
    assert len(df) == 16
    assert not df.isna().any().any()

    expected_outcomes = ["mean_hr_bpm", "sdnn_ms", "rmssd_ms", "ln_hf"]
    expected_terms = ["Intercept", "condition[T.pain]", "sex[T.M]", "condition[T.pain]:sex[T.M]"]
    assert df["outcome"].tolist() == [o for o in expected_outcomes for _ in expected_terms]
    assert df["term"].tolist() == expected_terms * 4

    lnhf_pain = df[(df["outcome"] == "ln_hf") & (df["term"] == "condition[T.pain]")].iloc[0]
    assert lnhf_pain["estimate"] < -1.0 and lnhf_pain["p_value"] < 0.001

    lnhf_inter = df[(df["outcome"] == "ln_hf") & (df["term"] == "condition[T.pain]:sex[T.M]")].iloc[0]
    assert lnhf_inter["estimate"] > 0.4 and lnhf_inter["p_value"] < 0.01
