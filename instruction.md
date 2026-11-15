**Sex-Specific Autonomic Nociceptive Reactivity and Heart Rate Variability Mixed-Effects Pipeline**

You are tasked with implementing an end-to-end electrocardiogram (ECG) signal quality assessment, R-peak detection, ectopic beat correction, frequency- and time-domain heart rate variability (HRV) extraction, and Restricted Maximum Likelihood (REML) linear mixed-effects modeling pipeline across resting baseline and nociceptive pain conditions.

**1. Input Data Layout**
All input data resides under `/data/`:
- `/data/subjects.csv`: Cohort metadata table containing columns `subject_id` (`SUB01` through `SUB24`) and `sex` (`F` or `M`).
- `/data/ecg/`: Directory containing 48 single-lead ECG recordings named `<subject_id>_<condition>.csv` (`condition` in `{"baseline", "pain"}`), uniformly sampled at 250 Hz over 180 seconds (45,000 rows) with columns `time_sec` and `ecg_mv`.

**2. Signal Conditioning, Welch SQI Window Gating, and R-Peak Detection**
For each recording:
- Apply a zero-phase 4th-order Butterworth bandpass filter (`5.0–20.0 Hz`, `scipy.signal.butter` + `filtfilt`) to `ecg_mv` to isolate QRS complexes and suppress baseline wander and high-frequency muscle noise.
- Partition the 180-second recording into 18 non-overlapping 10-second windows (`[0, 10)`, `[10, 20)`, ..., `[170, 180)` seconds). For each 10-second window, estimate the power spectral density via Welch's method (`scipy.signal.welch`, `fs=250.0`, `nperseg=512`) and compute the QRS-band Signal Quality Index (`SQI`):
  $$\text{SQI} = \frac{\int_{5}^{20} P(f)\,df}{\int_{0.5}^{60} P(f)\,df}$$
  Mark any 10-second window with `SQI < 0.55` as artifact-contaminated, and record `clean_window_ratio` as `clean_windows / 18`.
- Detect R-peaks on the bandpass-filtered ECG signal using local maxima detection (`scipy.signal.find_peaks`) with a minimum refractory distance of `0.36 s` (90 samples) and a prominence threshold of `0.55 * std(ecg_filtered)`.
- Construct RR intervals (in milliseconds) strictly from consecutive R-peaks occurring within contiguous clean (`SQI >= 0.55`) 10-second windows; discard any interval spanning across a rejected window.
- Flag ectopic or missed-beat RR intervals deviating by more than `20%` from the centered 5-beat rolling median and replace them via Piecewise Cubic Hermite Interpolating Polynomial (`PCHIP`) interpolation over beat index to obtain cleaned normal-to-normal (`NN`) intervals.

**3. Time-Domain and Parseval-Normalized Lomb-Scargle Frequency-Domain HRV Extraction**
From the cleaned `NN` interval series (in milliseconds) and corresponding beat timestamps $t_k$ (in seconds) of each recording, compute:
- `mean_hr_bpm`: Mean heart rate in beats per minute (`60000.0 / mean(NN)`).
- `sdnn_ms`: Standard deviation of `NN` intervals in milliseconds (`ddof=1`).
- `rmssd_ms`: Root mean square of successive `NN` interval differences in milliseconds.
- `hf_power_ms2`: High-frequency (`0.15–0.40 Hz`) spectral power in $\text{ms}^2$ estimated on the unevenly sampled mean-centered `NN` series via the Lomb-Scargle periodogram (`scipy.signal.lombscargle`, `normalize=False`) evaluated over a uniform frequency grid of `512` points in `[0.04, 0.40] Hz` ($\omega = 2\pi f$), scaled according to Parseval's theorem so total integrated power over `[0.04, 0.40] Hz` equals the sample variance of `NN` ($\text{ms}^2$), and integrated over `[0.15, 0.40] Hz` using the trapezoidal rule.
- `ln_hf`: Natural logarithm of `hf_power_ms2` (`np.log(hf_power_ms2)`).

Save all 48 recording rows sorted by `subject_id` ascending, then `condition` ascending (`baseline` before `pain`), to `/app/metrics.csv` with exact columns:
`subject_id,sex,condition,clean_window_ratio,mean_hr_bpm,sdnn_ms,rmssd_ms,hf_power_ms2,ln_hf`

**4. REML Linear Mixed-Effects Reactivity Modeling**
Using `/app/metrics.csv`, fit four separate Restricted Maximum Likelihood (`REML`) linear mixed-effects models (`statsmodels.formula.api.mixedlm`) with subject-level random intercepts (`groups="subject_id"`):
$$\text{outcome} \sim \text{condition} \times \text{sex} + (1 \mid \text{subject\_id})$$
for each outcome in `["mean_hr_bpm", "sdnn_ms", "rmssd_ms", "ln_hf"]`, using treatment coding with reference levels `condition = "baseline"` and `sex = "F"`.

Extract the fixed-effect parameters (`Intercept`, `condition[T.pain]`, `sex[T.M]`, `condition[T.pain]:sex[T.M]`) and save all 16 rows (ordered by outcome list above, then term order above) to `/app/stats.csv` with exact columns:
`outcome,term,estimate,std_err,z_value,p_value`

You have 28800 seconds to complete this task. Do not cheat by using online solutions or hints specific to this task.
