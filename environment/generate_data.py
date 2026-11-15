import os
import numpy as np
import pandas as pd

def generate_dataset(output_dir="/data", seed=1024):
    rng = np.random.default_rng(seed)
    rec_dir = os.path.join(output_dir, "ecg")
    os.makedirs(rec_dir, exist_ok=True)

    n_subjects = 24
    subjects = []
    for i in range(1, n_subjects + 1):
        sid = f"SUB{i:02d}"
        sex = "F" if i <= 12 else "M"
        subjects.append({"subject_id": sid, "sex": sex})
    pd.DataFrame(subjects).to_csv(os.path.join(output_dir, "subjects.csv"), index=False)

    fs = 250.0
    duration_sec = 180.0
    t = np.arange(0, duration_sec, 1.0 / fs)

    for row in subjects:
        sid = row["subject_id"]
        sex = row["sex"]
        subj_rr_offset = rng.normal(0.0, 0.030)

        for cond in ["baseline", "pain"]:
            if cond == "baseline":
                base_rr = 0.85 if sex == "F" else 0.89
                hf_amp = 0.044 if sex == "F" else 0.038
                lf_amp = 0.022
            else:
                # Stronger vagal withdrawal & sympathetic arousal in females during nociceptive pain
                base_rr = 0.70 if sex == "F" else 0.80
                hf_amp = 0.015 if sex == "F" else 0.026
                lf_amp = 0.036 if sex == "F" else 0.030

            mean_rr = np.clip(base_rr + subj_rr_offset, 0.52, 1.15)
            f_hf = rng.uniform(0.22, 0.28)
            f_lf = rng.uniform(0.08, 0.12)

            r_times = [0.5]
            while r_times[-1] < duration_sec - 0.5:
                tc = r_times[-1]
                mod_hf = hf_amp * np.sin(2.0 * np.pi * f_hf * tc)
                mod_lf = lf_amp * np.sin(2.0 * np.pi * f_lf * tc + 0.7)
                jitter = rng.normal(0.0, 0.009 if cond == "pain" else 0.015)
                rr = np.clip(mean_rr + mod_hf + mod_lf + jitter, 0.45, 1.25)
                r_times.append(tc + rr)
            r_times = np.array(r_times)

            # Synthesize ECG waveform (P, QRS, T waves)
            ecg = np.zeros_like(t)
            for rt in r_times:
                dt = t - rt
                mask = (dt >= -0.25) & (dt <= 0.45)
                dtm = dt[mask]
                p_wave = 0.12 * np.exp(-0.5 * ((dtm + 0.16) / 0.022) ** 2)
                q_wave = -0.15 * np.exp(-0.5 * ((dtm + 0.022) / 0.008) ** 2)
                r_wave = 1.25 * np.exp(-0.5 * (dtm / 0.010) ** 2)
                s_wave = -0.25 * np.exp(-0.5 * ((dtm - 0.024) / 0.009) ** 2)
                t_wave = 0.28 * np.exp(-0.5 * ((dtm - 0.24) / 0.045) ** 2)
                ecg[mask] += p_wave + q_wave + r_wave + s_wave + t_wave

            baseline_wander = 0.18 * np.sin(2.0 * np.pi * 0.15 * t) + 0.08 * np.cos(2.0 * np.pi * 0.04 * t)
            ecg += baseline_wander + rng.normal(0.0, 0.02, size=t.shape)

            # Inject high-frequency EMG/motion noise bursts in windows [30, 40) and [130, 140)
            for w_start in [30.0, 130.0]:
                m_noise = (t >= w_start) & (t < w_start + 10.0)
                ecg[m_noise] += 1.65 * np.sin(2.0 * np.pi * 32.0 * t[m_noise]) + rng.normal(0.0, 0.85, m_noise.sum())

            pd.DataFrame({
                "time_sec": np.round(t, 6),
                "ecg_mv": np.round(ecg, 6),
            }).to_csv(os.path.join(rec_dir, f"{sid}_{cond}.csv"), index=False)

if __name__ == "__main__":
    generate_dataset()
