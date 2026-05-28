import re
from datetime import datetime

import numpy as np
import pandas as pd

from fault_detection import estimate_sampling_rate, calculate_rms_sliding


def explain_fault_type_result(result: dict, context: str = "Aplikasi"):
    fault_type = result.get("fault_type", "UNKNOWN")
    phases = result.get("faulted_phases", [])
    ground_text = (
        "melibatkan tanah/ground"
        if result.get("ground_involved")
        else "tidak menunjukkan arus tanah yang dominan"
    )

    if fault_type == "UNKNOWN":
        return (
            f"{context} belum bisa menentukan tipe gangguan dengan jelas. "
            "Cek mapping channel, threshold deteksi, dan cursor DFT."
        )

    phase_text = ", ".join(phases) if phases else "-"
    return (
        f"{context} membaca gangguan sebagai {fault_type}. "
        f"Fasa yang dianggap terganggu: {phase_text}; {ground_text}. "
        "Nilai confidence menunjukkan seberapa konsisten pola arus, tegangan, dan ground terhadap aturan klasifikasi aplikasi."
    )


def build_auto_fault_type_threshold_dataframe(settings: dict):
    rows = [
        {"Parameter": "Mode", "Value": settings.get("mode", "-")},
        {"Parameter": "Normal Voltage RMS", "Value": settings.get("normal_voltage_rms", 0.0)},
        {"Parameter": "Normal Current RMS", "Value": settings.get("normal_current_rms", 0.0)},
        {"Parameter": "Normal Ground Current RMS", "Value": settings.get("normal_ground_current_rms", 0.0)},
        {"Parameter": "Max Voltage Drop %", "Value": settings.get("max_voltage_drop_pct", 0.0)},
        {"Parameter": "Max Current Change %", "Value": settings.get("max_current_change_pct", 0.0)},
        {"Parameter": "Ground Current Rise Ratio", "Value": settings.get("ground_current_rise_ratio", 0.0)},
        {"Parameter": "Voltage Drop Threshold", "Value": settings.get("voltage_drop_threshold", 0.0)},
        {"Parameter": "Current Rise Threshold", "Value": settings.get("current_rise_threshold", 0.0)},
        {"Parameter": "Ground Current Threshold", "Value": settings.get("ground_current_threshold", 0.0)},
        {"Parameter": "Delta Current Threshold", "Value": settings.get("delta_current_threshold", 0.0)},
        {"Parameter": "Delta Voltage Threshold", "Value": settings.get("delta_voltage_threshold", 0.0)},
    ]
    return pd.DataFrame(rows)


def parse_comtrade_timestamp(value):
    if isinstance(value, datetime):
        return value

    text = str(value or "").strip()
    if not text or text == "-":
        return None

    text = text.replace("T", ",")
    text = re.sub(r"\s+", "", text)

    if "," not in text:
        return None

    date_text, time_text = text.split(",", 1)
    date_parts = date_text.split("/")

    if len(date_parts) != 3:
        return None

    try:
        first = int(date_parts[0])
        second = int(date_parts[1])
        year = int(date_parts[2])

        # COMTRADE export di lingkungan ini umumnya DD/MM/YYYY. Jika ambigu,
        # pertahankan DD/MM karena tanggal Indonesia lebih sering begitu.
        if first > 12:
            day, month = first, second
        elif second > 12:
            month, day = first, second
        else:
            day, month = first, second

        if "." in time_text:
            main_time, frac = time_text.split(".", 1)
            frac = re.sub(r"\D", "", frac)[:6].ljust(6, "0")
            time_text = f"{main_time}.{frac}"
            fmt = "%H:%M:%S.%f"
        else:
            fmt = "%H:%M:%S"

        parsed_time = datetime.strptime(time_text, fmt).time()
        return datetime(
            year,
            month,
            day,
            parsed_time.hour,
            parsed_time.minute,
            parsed_time.second,
            parsed_time.microsecond,
        )
    except Exception:
        return None


def get_absolute_event_time(metadata: dict, relative_time_s: float, mode: str):
    if mode == "cfg_trigger_time":
        return parse_comtrade_timestamp(metadata.get("cfg_trigger_time"))

    start_time = parse_comtrade_timestamp(metadata.get("cfg_start_time"))
    if start_time is None:
        return None

    return start_time + pd.to_timedelta(float(relative_time_s), unit="s").to_pytimedelta()


def calculate_time_based_fault_location(
    local_time,
    remote_time,
    line_length_km: float,
    velocity_factor: float,
):
    c_km_per_s = 299792.458
    propagation_velocity = c_km_per_s * float(velocity_factor)
    delta_t_s = (local_time - remote_time).total_seconds()
    distance_from_local_km = (float(line_length_km) + propagation_velocity * delta_t_s) / 2.0
    distance_from_remote_km = float(line_length_km) - distance_from_local_km
    one_end_travel_time_s = float(line_length_km) / max(propagation_velocity, 1e-9)

    warnings = []
    if distance_from_local_km < 0 or distance_from_local_km > float(line_length_km):
        warnings.append(
            "Hasil berada di luar panjang saluran. Cek sinkronisasi waktu, pemilihan event, atau velocity factor."
        )
    if abs(delta_t_s) > one_end_travel_time_s * 1.05:
        warnings.append(
            "Selisih waktu lebih besar dari waktu rambat ujung-ke-ujung. Ini tidak realistis untuk TWS."
        )

    return {
        "local_time": local_time,
        "remote_time": remote_time,
        "delta_t_s": delta_t_s,
        "velocity_factor": float(velocity_factor),
        "propagation_velocity_km_s": propagation_velocity,
        "one_end_travel_time_s": one_end_travel_time_s,
        "distance_from_local_km": distance_from_local_km,
        "distance_from_remote_km": distance_from_remote_km,
        "distance_from_local_percent": distance_from_local_km / max(float(line_length_km), 1e-9) * 100.0,
        "warnings": warnings,
    }


def calculate_auto_fault_detection_parameters(
    df: pd.DataFrame,
    frequency: float = 50.0,
    pre_fault_cycles: int = 2,
    nominal_phase_voltage_rms: float | None = None,
    nominal_current_rms: float | None = None,
):
    try:
        fs = estimate_sampling_rate(df)
        samples_per_cycle = max(4, int(round(fs / max(float(frequency), 1e-9))))
        prefault_samples = max(samples_per_cycle + 1, int(pre_fault_cycles) * samples_per_cycle)

        if len(df) <= prefault_samples + samples_per_cycle:
            raise ValueError("record_too_short")

        current_rms = [
            calculate_rms_sliding(df[channel].to_numpy(dtype=float), samples_per_cycle)
            for channel in ["Ia", "Ib", "Ic"]
            if channel in df.columns
        ]
        voltage_rms = [
            calculate_rms_sliding(df[channel].to_numpy(dtype=float), samples_per_cycle)
            for channel in ["Va", "Vb", "Vc"]
            if channel in df.columns
        ]

        current_rms_max = np.nanmax(np.vstack(current_rms), axis=0)
        voltage_rms_min = np.nanmin(np.vstack(voltage_rms), axis=0)

        baseline_slice = slice(samples_per_cycle, prefault_samples)
        search_slice = slice(prefault_samples, None)

        prefault_current = float(np.nanmedian(current_rms_max[baseline_slice]))
        prefault_voltage = float(np.nanmedian(voltage_rms_min[baseline_slice]))
        observed_current_max = float(np.nanmax(current_rms_max[search_slice]))
        observed_voltage_min = float(np.nanmin(voltage_rms_min[search_slice]))

        voltage_reference = prefault_voltage
        current_reference = prefault_current
        reference_mode = "prefault_rms"

        if nominal_phase_voltage_rms and nominal_phase_voltage_rms > 0:
            nominal_phase_voltage_rms = float(nominal_phase_voltage_rms)
            if prefault_voltage < 0.97 * nominal_phase_voltage_rms:
                voltage_reference = nominal_phase_voltage_rms
                reference_mode = "nominal_vt_assisted"

        if nominal_current_rms and nominal_current_rms > 0:
            nominal_current_rms = float(nominal_current_rms)
            if prefault_current > 1.20 * nominal_current_rms:
                current_reference = nominal_current_rms
                reference_mode = "nominal_ct_vt_assisted"

        current_ratio = observed_current_max / max(current_reference, 1e-9)
        voltage_ratio = observed_voltage_min / max(voltage_reference, 1e-9)

        if current_ratio > 1.05:
            current_multiplier = max(1.05, min(2.0, 1.0 + 0.35 * (current_ratio - 1.0)))
        else:
            current_multiplier = 1.50

        if voltage_ratio < 0.995:
            voltage_threshold = max(0.60, min(0.98, 1.0 - 0.35 * (1.0 - voltage_ratio)))
        else:
            voltage_threshold = 0.85

        return {
            "mode": "auto_prefault_rms",
            "current_threshold_multiplier": float(current_multiplier),
            "voltage_drop_threshold": float(voltage_threshold),
            "adaptive_threshold_sigma": 6.0,
            "superimposed_threshold_sigma": 8.0,
            "fault_detection_method": "hybrid_superimposed",
            "refine_fault_bar": True,
            "prefault_current_rms": prefault_current,
            "prefault_voltage_rms": prefault_voltage,
            "reference_current_rms": float(current_reference),
            "reference_voltage_rms": float(voltage_reference),
            "nominal_current_rms": float(nominal_current_rms or 0.0),
            "nominal_phase_voltage_rms": float(nominal_phase_voltage_rms or 0.0),
            "reference_mode": reference_mode,
            "observed_current_ratio": float(current_ratio),
            "observed_voltage_ratio": float(voltage_ratio),
            "samples_per_cycle": samples_per_cycle,
        }
    except Exception:
        return {
            "mode": "default_fallback",
            "current_threshold_multiplier": 2.0,
            "voltage_drop_threshold": 0.85,
            "adaptive_threshold_sigma": 6.0,
            "superimposed_threshold_sigma": 8.0,
            "fault_detection_method": "hybrid_superimposed",
            "refine_fault_bar": True,
            "prefault_current_rms": 0.0,
            "prefault_voltage_rms": 0.0,
            "reference_current_rms": 0.0,
            "reference_voltage_rms": 0.0,
            "nominal_current_rms": float(nominal_current_rms or 0.0),
            "nominal_phase_voltage_rms": float(nominal_phase_voltage_rms or 0.0),
            "reference_mode": "default_fallback",
            "observed_current_ratio": 0.0,
            "observed_voltage_ratio": 0.0,
            "samples_per_cycle": 0,
        }


def explain_single_ended_status(status: str):
    explanations = {
        "VALID": (
            "VALID berarti hasil numerik masih berada dalam batas kewajaran aplikasi: "
            "jarak tidak negatif, tidak melewati panjang saluran, dan indikator resistif tidak melewati batas warning. "
            "Ini bukan jaminan lokasi pasti benar, tetapi hasil layak dipakai sebagai estimasi awal."
        ),
        "CHECK": (
            "CHECK berarti hasil masih bisa dipakai sebagai indikasi, tetapi ada gejala yang perlu divalidasi "
            "dengan waveform, SOE relay, fault type, polaritas CT/CVT, dan data lapangan."
        ),
        "UNCERTAIN": (
            "UNCERTAIN berarti hasil keluar dari batas dasar, misalnya jarak negatif atau melebihi panjang saluran. "
            "Cek ulang mapping sinyal, polaritas, parameter saluran, dan pemilihan loop gangguan."
        ),
    }

    return explanations.get(status, "Status belum dikenali. Cek detail warning dan parameter input.")


def explain_two_ended_quality(quality: dict):
    score = quality.get("quality_score", 0)

    if score >= 9:
        level = "sangat baik"
    elif score >= 7:
        level = "baik, tetapi masih perlu validasi"
    elif score >= 5:
        level = "sedang dan perlu dicek ulang"
    else:
        level = "rendah, sehingga hasil perlu dianggap tidak pasti"

    return (
        f"Quality {score}/10 berarti kualitas perhitungan two-ended {level}. "
        "Skor ini terutama dipengaruhi oleh apakah jarak berada di dalam saluran, "
        "besar komponen imajiner jarak, dan mismatch tegangan fault dari dua ujung."
    )


def explain_high_resistance_result(result: dict):
    if result.get("high_resistance_suspected"):
        return (
            "Aplikasi melihat bukti gangguan resistif yang cukup kuat. "
            "Pada kondisi ini, estimasi single-ended berbasis magnitude bisa bergeser, "
            "sehingga jarak berbasis reactance/projection lebih layak dijadikan pembanding."
        )

    return (
        "Aplikasi belum melihat bukti kuat gangguan high resistance. "
        "Untuk gangguan petir atau flashover cepat, ini sering wajar karena impedansi busur dapat rendah dan durasinya singkat. "
        "Tetap validasi dengan waveform dan laporan relay."
    )


def explain_sync_warning():
    return (
        "Selisih waktu local dan remote lebih dari 1 siklus tidak selalu berarti rekaman salah. "
        "Jika jam relay tidak sinkron SNTP/GPS dan diset manual, timestamp absolut bisa berbeda. "
        "Dalam kondisi itu, fokuskan validasi pada kualitas fasor masing-masing record, fault type, dan konsistensi hasil two-ended."
    )
