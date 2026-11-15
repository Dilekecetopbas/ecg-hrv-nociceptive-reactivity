# Sex-Specific Autonomic Nociceptive Reactivity & HRV Mixed-Effects Pipeline (`afterquery/autonomic-hrv-lmm`)

## Overview
An end-to-end biomedical signal processing and statistical modeling pipeline analyzing 250 Hz single-lead electrocardiogram (`ECG`) recordings across resting `baseline` and nociceptive `pain` conditions in a sex-balanced cohort (`N = 24`). The pipeline performs Welch power spectral density Signal Quality Index (`SQI`) epoch gating, R-peak detection, PCHIP ectopic interval interpolation, time- and Parseval-normalized Lomb-Scargle frequency-domain heart rate variability (`HRV`) feature extraction, and Restricted Maximum Likelihood (`REML`) linear mixed-effects modeling.

## Pipeline Architecture
1. **ECG Conditioning & Welch SQI Window Gating**: Applies a 4th-order zero-phase Butterworth bandpass filter (`5.0–20.0 Hz`) and screens 10-second non-overlapping epochs using the QRS-to-total spectral power ratio (`5–20 Hz / 0.5–60 Hz >= 0.55`).
2. **R-Peak Detection & PCHIP Ectopic Correction**: Detects R-peaks (`0.36 s` refractory period), prevents cross-window interval bridging, and replaces intervals deviating `> 20%` from the centered 5-beat median using PCHIP interpolation.
3. **Time & Lomb-Scargle Frequency-Domain HRV**: Extracts `mean_hr_bpm`, `sdnn_ms`, `rmssd_ms`, and Parseval-normalized Lomb-Scargle high-frequency power (`0.15–0.40 Hz`, `hf_power_ms2` and `ln_hf`).
4. **REML Mixed-Effects Modeling**: Fits `outcome ~ condition * sex + (1 | subject_id)` via REML across all four autonomic endpoints.
