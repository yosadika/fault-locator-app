import numpy as np
import pandas as pd


def estimate_sampling_rate(df: pd.DataFrame) -> float:
    """
    Mengestimasi sampling rate dari kolom time.
    """
    time = df["time"].values

    if len(time) < 2:
        raise ValueError("Data time terlalu pendek untuk estimasi sampling rate.")

    dt = np.median(np.diff(time))

    if dt <= 0:
        raise ValueError("Kolom time tidak valid.")

    return 1.0 / dt


def calculate_rms_sliding(signal: np.ndarray, samples_per_cycle: int) -> np.ndarray:
    """
    Menghitung RMS sliding window 1 siklus.
    Output dibuat sepanjang input dengan padding NaN di awal.
    """
    signal = np.asarray(signal, dtype=float)

    if samples_per_cycle < 2:
        raise ValueError("samples_per_cycle terlalu kecil.")

    rms = np.full(len(signal), np.nan)

    for i in range(samples_per_cycle, len(signal)):
        window = signal[i - samples_per_cycle:i]
        rms[i] = np.sqrt(np.mean(window ** 2))

    return rms


def detect_fault_inception(
    df: pd.DataFrame,
    frequency: float = 50.0,
    current_threshold_multiplier: float = 2.0,
    voltage_drop_threshold: float = 0.85,
    min_prefault_cycles: int = 2,
):
    """
    Deteksi awal gangguan berdasarkan:
    - kenaikan RMS arus terhadap pre-fault
    - penurunan RMS tegangan terhadap pre-fault

    Input df wajib berisi:
    time, Va, Vb, Vc, Ia, Ib, Ic

    Output:
    dictionary berisi index dan waktu fault inception.
    """

    fs = estimate_sampling_rate(df)
    samples_per_cycle = int(round(fs / frequency))

    if samples_per_cycle < 4:
        raise ValueError("Sampling rate terlalu rendah untuk analisis 1 siklus.")

    prefault_samples = min_prefault_cycles * samples_per_cycle

    if len(df) <= prefault_samples + samples_per_cycle:
        raise ValueError("Data terlalu pendek untuk deteksi gangguan.")

    # RMS sliding untuk arus dan tegangan
    ia_rms = calculate_rms_sliding(df["Ia"].values, samples_per_cycle)
    ib_rms = calculate_rms_sliding(df["Ib"].values, samples_per_cycle)
    ic_rms = calculate_rms_sliding(df["Ic"].values, samples_per_cycle)

    va_rms = calculate_rms_sliding(df["Va"].values, samples_per_cycle)
    vb_rms = calculate_rms_sliding(df["Vb"].values, samples_per_cycle)
    vc_rms = calculate_rms_sliding(df["Vc"].values, samples_per_cycle)

    current_rms_max = np.nanmax(
        np.vstack([ia_rms, ib_rms, ic_rms]),
        axis=0
    )

    voltage_rms_min = np.nanmin(
        np.vstack([va_rms, vb_rms, vc_rms]),
        axis=0
    )

    prefault_current = np.nanmedian(
        current_rms_max[samples_per_cycle:prefault_samples]
    )

    prefault_voltage = np.nanmedian(
        voltage_rms_min[samples_per_cycle:prefault_samples]
    )

    current_pickup = prefault_current * current_threshold_multiplier
    voltage_pickup = prefault_voltage * voltage_drop_threshold

    fault_candidates = []

    for i in range(prefault_samples, len(df)):
        current_condition = current_rms_max[i] > current_pickup
        voltage_condition = voltage_rms_min[i] < voltage_pickup

        if current_condition or voltage_condition:
            fault_candidates.append(i)

    if not fault_candidates:
        return {
            "detected": False,
            "message": "Awal gangguan tidak terdeteksi otomatis. Silakan gunakan cursor manual.",
            "fs": fs,
            "samples_per_cycle": samples_per_cycle,
            "prefault_current": prefault_current,
            "prefault_voltage": prefault_voltage,
        }

    fault_index = fault_candidates[0]
    fault_time = float(df["time"].iloc[fault_index])

    return {
        "detected": True,
        "fault_index": int(fault_index),
        "fault_time": fault_time,
        "fs": fs,
        "samples_per_cycle": samples_per_cycle,
        "prefault_current": prefault_current,
        "prefault_voltage": prefault_voltage,
        "current_pickup": current_pickup,
        "voltage_pickup": voltage_pickup,
        "current_rms_max": current_rms_max,
        "voltage_rms_min": voltage_rms_min,
    }


def build_fault_window(
    df: pd.DataFrame,
    fault_index: int,
    samples_per_cycle: int,
    pre_fault_cycles: int = 2,
    post_fault_cycles: int = 4,
):
    """
    Membuat window analisis:
    - left cursor = beberapa siklus sebelum fault
    - right cursor = beberapa siklus setelah fault
    - dft cursor = 1 siklus setelah fault inception

    DFT cursor diletakkan setelah gangguan agar window DFT di sebelah kirinya
    tidak melintasi titik fault inception.
    """

    left_index = max(0, fault_index - pre_fault_cycles * samples_per_cycle)
    right_index = min(len(df) - 1, fault_index + post_fault_cycles * samples_per_cycle)

    # Cursor DFT untuk Step 4.
    # Karena DFT window berada di kiri cursor, maka cursor ini diletakkan
    # minimal 1 siklus setelah awal gangguan.
    dft_index = min(len(df) - 1, fault_index + samples_per_cycle)

    return {
        "left_index": int(left_index),
        "right_index": int(right_index),
        "fault_index": int(fault_index),
        "dft_index": int(dft_index),
        "left_time": float(df["time"].iloc[left_index]),
        "right_time": float(df["time"].iloc[right_index]),
        "fault_time": float(df["time"].iloc[fault_index]),
        "dft_time": float(df["time"].iloc[dft_index]),
    }