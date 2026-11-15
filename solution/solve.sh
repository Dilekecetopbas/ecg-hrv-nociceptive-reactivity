#!/bin/bash
set -euo pipefail

python3 - << 'PYEOF'
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch, find_peaks, lombscargle
from scipy.interpolate import PchipInterpolator
import statsmodels.formula.api as smf

def process_ecg(filepath):
    df = pd.read_csv(filepath)
    t = df["time_sec"].to_numpy()
    ecg = df["ecg_mv"].to_numpy()
    fs = 250.0

    # 1. Zero-phase 4th-order Butterworth bandpass (5-20 Hz)
    b, a = butter(4, [5.0 / (0.5 * fs), 20.0 / (0.5 * fs)], btype="band")
    ecg_filt = filtfilt(b, a, ecg)

    # 2. Welch SQI across 18 non-overlapping 10s windows
    n_win = 18
    win_sec = 10.0
    samples_per_win = int(fs * win_sec)
    clean_win = np.zeros(n_win, dtype=bool)
    for w in range(n_win):
        seg = ecg[w * samples_per_win:(w + 1) * samples_per_win]
        freqs, psd = welch(seg, fs=fs, nperseg=512)
        m_qrs = (freqs >= 5.0) & (freqs <= 20.0)
        m_tot = (freqs >= 0.5) & (freqs <= 60.0)
        p_qrs = np.trapezoid(psd[m_qrs], freqs[m_qrs])
        p_tot = np.trapezoid(psd[m_tot], freqs[m_tot])
        sqi = p_qrs / p_tot if p_tot > 0 else 0.0
        clean_win[w] = (sqi >= 0.55)
    clean_window_ratio = float(np.mean(clean_win))

    # 3. R-peak detection
    min_dist = int(0.36 * fs)
    prom = 0.55 * np.std(ecg_filt)
    peaks, _ = find_peaks(ecg_filt, distance=min_dist, prominence=prom)
    r_times = t[peaks]

    # 4. Contiguous clean-window RR intervals
    r_win = np.clip(np.floor(r_times / win_sec).astype(int), 0, n_win - 1)
    rr_list = []
    rr_t = []
    for i in range(len(r_times) - 1):
        w1, w2 = r_win[i], r_win[i + 1]
        if clean_win[w1] and clean_win[w2] and (w2 - w1 <= 1):
            rr_list.append((r_times[i + 1] - r_times[i]) * 1000.0)
            rr_t.append(0.5 * (r_times[i] + r_times[i + 1]))

    nn = np.asarray(rr_list, dtype=float)
    nn_t = np.asarray(rr_t, dtype=float)

    # 5. Ectopic outlier correction via 5-beat centered rolling median + PCHIP
    med5 = pd.Series(nn).rolling(window=5, center=True, min_periods=1).median().to_numpy()
    outliers = np.abs(nn - med5) > (0.20 * med5)
    if np.any(outliers):
        idx = np.arange(len(nn))
        good = ~outliers
        interp = PchipInterpolator(idx[good], nn[good], extrapolate=True)
        nn[outliers] = interp(idx[outliers])

    # 6. Time-domain HRV
    mean_hr_bpm = float(60000.0 / np.mean(nn))
    sdnn_ms = float(np.std(nn, ddof=1))
    rmssd_ms = float(np.sqrt(np.mean(np.square(np.diff(nn)))))

    # 7. Parseval-normalized Lomb-Scargle HF power (0.15 - 0.40 Hz)
    f_grid = np.linspace(0.04, 0.40, 512)
    omega = 2.0 * np.pi * f_grid
    nn_centered = nn - np.mean(nn)
    pgram = lombscargle(nn_t, nn_centered, omega, normalize=False)
    total_area = np.trapezoid(pgram, f_grid)
    var_nn = np.var(nn, ddof=1)
    psd_scaled = pgram * (var_nn / total_area) if total_area > 0 else pgram
    m_hf = (f_grid >= 0.15) & (f_grid <= 0.40)
    hf_power_ms2 = float(np.trapezoid(psd_scaled[m_hf], f_grid[m_hf]))
    ln_hf = float(np.log(hf_power_ms2))

    return clean_window_ratio, mean_hr_bpm, sdnn_ms, rmssd_ms, hf_power_ms2, ln_hf

def main():
    df_subj = pd.read_csv("/data/subjects.csv")
    rows = []
    for _, srow in df_subj.iterrows():
        sid = srow["subject_id"]
        sex = srow["sex"]
        for cond in ["baseline", "pain"]:
            ratio, hr, sdnn, rmssd, hf, lnhf = process_ecg(f"/data/ecg/{sid}_{cond}.csv")
            rows.append({
                "subject_id": sid,
                "sex": sex,
                "condition": cond,
                "clean_window_ratio": ratio,
                "mean_hr_bpm": hr,
                "sdnn_ms": sdnn,
                "rmssd_ms": rmssd,
                "hf_power_ms2": hf,
                "ln_hf": lnhf,
            })

    df_metrics = pd.DataFrame(rows).sort_values(["subject_id", "condition"], ascending=[True, True])
    os.makedirs("/app", exist_ok=True)
    df_metrics.to_csv("/app/metrics.csv", index=False)

    outcomes = ["mean_hr_bpm", "sdnn_ms", "rmssd_ms", "ln_hf"]
    terms = ["Intercept", "condition[T.pain]", "sex[T.M]", "condition[T.pain]:sex[T.M]"]
    stat_rows = []
    for out in outcomes:
        formula = f"{out} ~ C(condition, Treatment(reference='baseline')) * C(sex, Treatment(reference='F'))"
        model = smf.mixedlm(formula, df_metrics, groups=df_metrics["subject_id"])
        res = model.fit(reml=True)
        pmap = {}
        for k in res.fe_params.index:
            if k == "Intercept":
                pmap["Intercept"] = k
            elif "condition" in k and "sex" not in k:
                pmap["condition[T.pain]"] = k
            elif "sex" in k and "condition" not in k:
                pmap["sex[T.M]"] = k
            elif "condition" in k and "sex" in k:
                pmap["condition[T.pain]:sex[T.M]"] = k

        for t in terms:
            rk = pmap[t]
            stat_rows.append({
                "outcome": out,
                "term": t,
                "estimate": float(res.fe_params[rk]),
                "std_err": float(res.bse_fe[rk]),
                "z_value": float(res.tvalues[rk]),
                "p_value": float(res.pvalues[rk]),
            })

    pd.DataFrame(stat_rows).to_csv("/app/stats.csv", index=False)

if __name__ == "__main__":
    main()
PYEOF
