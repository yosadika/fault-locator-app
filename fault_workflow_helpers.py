import contextlib
import re
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from fault_detection import (
    estimate_sampling_rate,
    calculate_rms_sliding,
    detect_fault_inception,
    build_fault_window,
)
from waveform_helpers import build_fault_window_plot


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


def _render_local_fault_cursor(
    metadata: dict,
    assigned_df_key: str,
    transformer_key: str,
    fault_window_key: str,
    fault_detection_key: str,
    key_prefix: str,
    compact: bool = False,
):
    if not compact:
        st.subheader("Fault Detection & Cursor Window")

    _trigger_ctx = st.expander("Trigger Metadata", expanded=False) if compact else contextlib.nullcontext()
    with _trigger_ctx:
        if not compact:
            st.markdown("### Trigger Metadata")
        col_t1, col_t2 = st.columns(2)
        col_t1.metric("CFG Start Time", str(metadata.get("cfg_start_time") or "-"))
        col_t2.metric("CFG Trigger Time", str(metadata.get("cfg_trigger_time") or "-"))
        st.caption(
            "Trigger timestamp dibaca dari metadata CFG jika tersedia. "
            "Fault inception tetap dideteksi dari waveform DAT untuk menentukan window DFT."
        )

    if assigned_df_key not in st.session_state:
        st.warning("Silakan lakukan Signal Assignment terlebih dahulu.")
        return

    assigned_df = st.session_state[assigned_df_key]
    local_transformer_data = st.session_state.get(transformer_key, {})

    st.markdown("### Parameter Deteksi Gangguan")

    col_fd1, col_w1, col_w2 = st.columns(3)

    with col_fd1:
        frequency = st.number_input(
            "Frekuensi Sistem (Hz)",
            value=float(metadata["frequency"]) if metadata["frequency"] else 50.0,
            min_value=40.0,
            max_value=70.0,
            step=0.001,
            format="%.5f",
            key=f"{key_prefix}_frequency",
        )

    with col_w1:
        pre_fault_cycles = st.number_input(
            "Pre-fault Window (cycles)",
            value=2,
            min_value=1,
            max_value=10,
            step=1,
            key=f"{key_prefix}_pre_fault_cycles",
        )

    with col_w2:
        post_fault_cycles = st.number_input(
            "Post-fault Window (cycles)",
            value=4,
            min_value=1,
            max_value=20,
            step=1,
            key=f"{key_prefix}_post_fault_cycles",
        )

    auto_fault_detection_settings = calculate_auto_fault_detection_parameters(
        assigned_df,
        frequency=frequency,
        pre_fault_cycles=int(pre_fault_cycles),
        nominal_phase_voltage_rms=local_transformer_data.get("nominal_phase_voltage_rms"),
        nominal_current_rms=local_transformer_data.get("nominal_current_rms"),
    )

    use_auto_fault_detection = st.checkbox(
        "Gunakan deteksi otomatis adaptif nominal + pre-fault",
        value=False,
        key=f"{key_prefix}_use_auto_fault_detection",
        help=(
            "Aplikasi memakai pre-fault RMS bila normal. Jika pre-fault terlihat sudah abnormal, "
            "aplikasi memakai referensi nominal dari VT/CT sebagai pembanding tambahan."
        ),
    )

    _thresh_ctx = st.expander("Parameter Deteksi Lanjutan", expanded=False) if compact else contextlib.nullcontext()
    with _thresh_ctx:
        col_fd2, col_fd3 = st.columns(2)
        with col_fd2:
            current_threshold_multiplier = st.number_input(
                "Multiplier Kenaikan Arus",
                value=float(auto_fault_detection_settings["current_threshold_multiplier"]),
                min_value=1.01,
                max_value=10.0,
                step=0.001,
                format="%.5f",
                disabled=use_auto_fault_detection,
                key=f"{key_prefix}_current_threshold_multiplier",
            )
        with col_fd3:
            voltage_drop_threshold = st.number_input(
                "Batas Drop Tegangan",
                value=float(auto_fault_detection_settings["voltage_drop_threshold"]),
                min_value=0.1,
                max_value=1.0,
                step=0.0001,
                format="%.5f",
                disabled=use_auto_fault_detection,
                key=f"{key_prefix}_voltage_drop_threshold",
            )

    if use_auto_fault_detection:
        current_threshold_multiplier = auto_fault_detection_settings["current_threshold_multiplier"]
        voltage_drop_threshold = auto_fault_detection_settings["voltage_drop_threshold"]

    with st.expander("Detail Parameter Deteksi Otomatis"):
        st.dataframe(
            pd.DataFrame(
                [
                    {"Parameter": key, "Value": value}
                    for key, value in auto_fault_detection_settings.items()
                ]
            ).style.format(
                {"Value": lambda x: f"{x:.6f}" if isinstance(x, (int, float)) else x}
            ),
            use_container_width=True,
        )

    with st.expander("Advanced Fault Bar Tuning"):
        use_advanced_fault_detection = st.checkbox(
            "Use Advanced Fault Detection",
            value=use_auto_fault_detection,
            help="Aktifkan hanya jika fault bar otomatis kurang presisi pada record lokal.",
            disabled=use_auto_fault_detection,
            key=f"{key_prefix}_use_advanced_fault_detection",
        )

        fault_detection_method = st.selectbox(
            "Fault Detection Method",
            ["legacy_rms", "hybrid_superimposed"],
            index=1,
            disabled=(not use_advanced_fault_detection or use_auto_fault_detection),
            help="hybrid_superimposed memakai energi perubahan satu siklus lalu divalidasi RMS.",
            key=f"{key_prefix}_fault_detection_method",
        )

        col_adv1, col_adv2, col_adv3, col_adv4 = st.columns(4)

        with col_adv1:
            adaptive_threshold_sigma = st.number_input(
                "Adaptive Threshold Sigma",
                value=6.0,
                min_value=2.0,
                max_value=20.0,
                step=0.001,
                format="%.5f",
                help="Threshold adaptif terhadap noise pre-fault. Lebih kecil = lebih sensitif.",
                disabled=(not use_advanced_fault_detection or use_auto_fault_detection),
                key=f"{key_prefix}_adaptive_threshold_sigma",
            )

        with col_adv2:
            superimposed_threshold_sigma = st.number_input(
                "Superimposed Threshold Sigma",
                value=8.0,
                min_value=2.0,
                max_value=30.0,
                step=0.001,
                format="%.5f",
                disabled=(
                    not use_advanced_fault_detection
                    or use_auto_fault_detection
                    or fault_detection_method != "hybrid_superimposed"
                ),
                help="Threshold energi superimposed terhadap baseline pre-fault.",
                key=f"{key_prefix}_superimposed_threshold_sigma",
            )

        with col_adv3:
            consecutive_samples_input = st.number_input(
                "Consecutive Samples",
                value=0,
                min_value=0,
                max_value=200,
                step=1,
                help="0 = otomatis sekitar 0.1 siklus. Nilai lebih besar menolak spike sesaat.",
                disabled=(not use_advanced_fault_detection or use_auto_fault_detection),
                key=f"{key_prefix}_consecutive_samples_input",
            )

        with col_adv4:
            refine_fault_bar = st.checkbox(
                "Refine Fault Bar",
                value=True,
                help="Backtrack dari kandidat RMS ke perubahan instantaneous awal.",
                disabled=(not use_advanced_fault_detection or use_auto_fault_detection),
                key=f"{key_prefix}_refine_fault_bar",
            )

    if use_auto_fault_detection:
        use_advanced_fault_detection = True
        fault_detection_method = auto_fault_detection_settings["fault_detection_method"]
        adaptive_threshold_sigma = auto_fault_detection_settings["adaptive_threshold_sigma"]
        superimposed_threshold_sigma = auto_fault_detection_settings["superimposed_threshold_sigma"]
        consecutive_samples_input = 0
        refine_fault_bar = auto_fault_detection_settings["refine_fault_bar"]

    detection = detect_fault_inception(
        assigned_df,
        frequency=frequency,
        current_threshold_multiplier=current_threshold_multiplier,
        voltage_drop_threshold=voltage_drop_threshold,
        min_prefault_cycles=int(pre_fault_cycles),
        adaptive_threshold_sigma=(
            adaptive_threshold_sigma
            if use_advanced_fault_detection
            else None
        ),
        consecutive_samples=(
            None
            if (
                not use_advanced_fault_detection
                or int(consecutive_samples_input) == 0
            )
            else int(consecutive_samples_input)
        ),
        refine_fault_bar=(
            refine_fault_bar
            if use_advanced_fault_detection
            else False
        ),
        method=(
            fault_detection_method
            if use_advanced_fault_detection
            else "legacy_rms"
        ),
        superimposed_threshold_sigma=superimposed_threshold_sigma,
        nominal_phase_voltage_rms=local_transformer_data.get("nominal_phase_voltage_rms"),
        nominal_current_rms=local_transformer_data.get("nominal_current_rms"),
    )

    st.session_state[fault_detection_key] = detection

    if detection["detected"]:
        st.success("Awal gangguan berhasil terdeteksi otomatis.")

        fault_window = build_fault_window(
            assigned_df,
            fault_index=detection["fault_index"],
            samples_per_cycle=detection["samples_per_cycle"],
            pre_fault_cycles=int(pre_fault_cycles),
            post_fault_cycles=int(post_fault_cycles),
        )

        st.session_state[fault_window_key] = fault_window

        col_r1, col_r2, col_r3, col_r4 = st.columns(4)

        col_r1.metric("Fault Time", f'{fault_window["fault_time"]:.6f} s')
        col_r2.metric("Left Cursor", f'{fault_window["left_time"]:.6f} s')
        col_r3.metric("Right Cursor", f'{fault_window["right_time"]:.6f} s')
        col_r4.metric("DFT Cursor", f'{fault_window["dft_time"]:.6f} s')

        st.write("Sampling Rate:", f'{detection["fs"]:.2f} Hz')
        st.write("Samples per Cycle:", detection["samples_per_cycle"])

        if detection.get("refine_fault_bar"):
            st.caption(
                "Fault bar refinement: "
                f'RMS candidate {detection["rms_fault_time"]:.6f} s -> '
                f'refined {detection["fault_time"]:.6f} s. '
                f'Confidence {detection.get("confidence_score", 0):.2f}/10, '
                f'consecutive samples {detection.get("consecutive_samples", "-")}.'
            )

        if detection.get("superimposed"):
            superimposed = detection["superimposed"]
            st.caption(
                "Superimposed detector: "
                f'threshold {superimposed["threshold"]:.6f}, '
                f'peak energy {superimposed.get("peak_energy", 0.0):.6f}.'
            )

    else:
        st.warning(detection["message"])

        st.markdown("### Manual Cursor")

        fs = detection["fs"]
        samples_per_cycle = detection["samples_per_cycle"]

        min_time = float(assigned_df["time"].min())
        max_time = float(assigned_df["time"].max())

        manual_fault_time = st.slider(
            "Pilih waktu awal gangguan manual (s)",
            min_value=min_time,
            max_value=max_time,
            value=min_time,
            step=(max_time - min_time) / 1000,
            key=f"{key_prefix}_manual_fault_time",
        )

        fault_index = int(
            (assigned_df["time"] - manual_fault_time).abs().idxmin()
        )

        fault_window = build_fault_window(
            assigned_df,
            fault_index=fault_index,
            samples_per_cycle=samples_per_cycle,
            pre_fault_cycles=int(pre_fault_cycles),
            post_fault_cycles=int(post_fault_cycles),
        )

        st.session_state[fault_window_key] = fault_window

    if fault_window_key in st.session_state:
        fault_window = st.session_state[fault_window_key]
        _info_text = (
            "Left Cursor dan Right Cursor digunakan sebagai range analisis. "
            "DFT Cursor digunakan pada Step 4 untuk mengambil fasor 1 siklus "
            "setelah awal gangguan."
        )
        _exp_label = "Detail Window Analisis" if compact else "Validasi Window Analisis"
        with st.expander(_exp_label, expanded=False):
            st.json(fault_window)
            st.info(_info_text)
            if not compact:
                display_df = assigned_df.copy()
                selected_plot = st.multiselect(
                    "Pilih sinyal untuk validasi fault window",
                    ["Va", "Vb", "Vc", "Ia", "Ib", "Ic", "IE"],
                    default=["Ia", "Ib", "Ic"],
                    key=f"{key_prefix}_fault_window_plot_channels",
                )
                fig = build_fault_window_plot(
                    display_df, fault_window, selected_plot, "Fault Detection dan Cursor Window",
                )
                _pad = (fault_window["right_time"] - fault_window["left_time"]) * 0.3
                fig.update_layout(xaxis_range=[
                    fault_window["left_time"] - _pad, fault_window["right_time"] + _pad,
                ])
                st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_fault_window_chart")

        if compact:
            display_df = assigned_df.copy()
            selected_plot = st.multiselect(
                "Pilih sinyal untuk validasi fault window",
                ["Va", "Vb", "Vc", "Ia", "Ib", "Ic", "IE"],
                default=["Ia", "Ib", "Ic"],
                key=f"{key_prefix}_fault_window_plot_channels",
            )
            fig = build_fault_window_plot(
                display_df, fault_window, selected_plot, "Fault Detection dan Cursor Window",
            )
            _pad = (fault_window["right_time"] - fault_window["left_time"]) * 0.3
            fig.update_layout(xaxis_range=[
                fault_window["left_time"] - _pad, fault_window["right_time"] + _pad,
            ])
            st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_fault_window_chart")


def _render_remote_fault_cursor(
    metadata: dict,
    assigned_df_key: str,
    transformer_key: str,
    fault_window_key: str,
    fault_detection_key: str,
    key_prefix: str,
    compact: bool = False,
):
    if not compact:
        st.subheader("Remote Fault Detection & Cursor Window")

    _trigger_ctx = st.expander("Trigger Metadata", expanded=False) if compact else contextlib.nullcontext()
    with _trigger_ctx:
        if not compact:
            st.markdown("### Trigger Metadata")
        col_t1, col_t2 = st.columns(2)
        col_t1.metric("CFG Start Time", str(metadata.get("cfg_start_time") or "-"))
        col_t2.metric("CFG Trigger Time", str(metadata.get("cfg_trigger_time") or "-"))
        st.caption(
            "Trigger timestamp dibaca dari metadata CFG remote jika tersedia. "
            "Fault inception tetap dideteksi dari waveform DAT untuk menentukan window DFT."
        )

    assigned_df = st.session_state.get(assigned_df_key)
    if assigned_df is None:
        st.info("Selesaikan Remote End > Signals terlebih dahulu.")
        return

    transformer_data = st.session_state.get(transformer_key, {})

    st.markdown("### Parameter Deteksi Gangguan")

    col_fd1, col_w1, col_w2 = st.columns(3)

    with col_fd1:
        frequency = st.number_input(
            "Frekuensi Sistem (Hz)",
            value=float(metadata.get("frequency") or 50.0),
            min_value=40.0,
            max_value=70.0,
            step=0.001,
            format="%.5f",
            key=f"{key_prefix}_frequency",
        )

    with col_w1:
        pre_fault_cycles = st.number_input(
            "Pre-fault Window (cycles)",
            value=2,
            min_value=1,
            max_value=10,
            step=1,
            key=f"{key_prefix}_pre_fault_cycles",
        )

    with col_w2:
        post_fault_cycles = st.number_input(
            "Post-fault Window (cycles)",
            value=4,
            min_value=1,
            max_value=20,
            step=1,
            key=f"{key_prefix}_post_fault_cycles",
        )

    auto_fault_detection_settings = calculate_auto_fault_detection_parameters(
        assigned_df,
        frequency=frequency,
        pre_fault_cycles=int(pre_fault_cycles),
        nominal_phase_voltage_rms=transformer_data.get("nominal_phase_voltage_rms"),
        nominal_current_rms=transformer_data.get("nominal_current_rms"),
    )

    use_auto_fault_detection = st.checkbox(
        "Gunakan deteksi otomatis adaptif nominal + pre-fault",
        value=False,
        key=f"{key_prefix}_use_auto_fault_detection",
        help=(
            "Aplikasi memakai pre-fault RMS bila normal. Jika pre-fault terlihat sudah abnormal, "
            "aplikasi memakai referensi nominal dari VT/CT sebagai pembanding tambahan."
        ),
    )

    _thresh_ctx = st.expander("Parameter Deteksi Lanjutan", expanded=False) if compact else contextlib.nullcontext()
    with _thresh_ctx:
        col_fd2, col_fd3 = st.columns(2)
        with col_fd2:
            current_threshold_multiplier = st.number_input(
                "Multiplier Kenaikan Arus",
                value=float(auto_fault_detection_settings["current_threshold_multiplier"]),
                min_value=1.01,
                max_value=10.0,
                step=0.001,
                format="%.5f",
                disabled=use_auto_fault_detection,
                key=f"{key_prefix}_current_threshold_multiplier",
            )
        with col_fd3:
            voltage_drop_threshold = st.number_input(
                "Batas Drop Tegangan",
                value=float(auto_fault_detection_settings["voltage_drop_threshold"]),
                min_value=0.1,
                max_value=1.0,
                step=0.0001,
                format="%.5f",
                disabled=use_auto_fault_detection,
                key=f"{key_prefix}_voltage_drop_threshold",
            )

    if use_auto_fault_detection:
        current_threshold_multiplier = auto_fault_detection_settings["current_threshold_multiplier"]
        voltage_drop_threshold = auto_fault_detection_settings["voltage_drop_threshold"]

    with st.expander("Detail Parameter Deteksi Otomatis"):
        st.dataframe(
            pd.DataFrame(
                [
                    {"Parameter": key, "Value": value}
                    for key, value in auto_fault_detection_settings.items()
                ]
            ).style.format(
                {"Value": lambda x: f"{x:.6f}" if isinstance(x, (int, float)) else x}
            ),
            use_container_width=True,
        )

    with st.expander("Advanced Fault Bar Tuning"):
        use_advanced_fault_detection = st.checkbox(
            "Use Advanced Fault Detection",
            value=use_auto_fault_detection,
            help="Aktifkan hanya jika fault bar otomatis kurang presisi pada record remote.",
            disabled=use_auto_fault_detection,
            key=f"{key_prefix}_use_advanced_fault_detection",
        )

        fault_detection_method = st.selectbox(
            "Fault Detection Method",
            ["legacy_rms", "hybrid_superimposed"],
            index=1,
            disabled=(not use_advanced_fault_detection or use_auto_fault_detection),
            help="hybrid_superimposed memakai energi perubahan satu siklus lalu divalidasi RMS.",
            key=f"{key_prefix}_fault_detection_method",
        )

        col_adv1, col_adv2, col_adv3, col_adv4 = st.columns(4)

        with col_adv1:
            adaptive_threshold_sigma = st.number_input(
                "Adaptive Threshold Sigma",
                value=6.0,
                min_value=2.0,
                max_value=20.0,
                step=0.001,
                format="%.5f",
                help="Threshold adaptif terhadap noise pre-fault. Lebih kecil = lebih sensitif.",
                disabled=(not use_advanced_fault_detection or use_auto_fault_detection),
                key=f"{key_prefix}_adaptive_threshold_sigma",
            )

        with col_adv2:
            superimposed_threshold_sigma = st.number_input(
                "Superimposed Threshold Sigma",
                value=8.0,
                min_value=2.0,
                max_value=30.0,
                step=0.001,
                format="%.5f",
                disabled=(
                    not use_advanced_fault_detection
                    or use_auto_fault_detection
                    or fault_detection_method != "hybrid_superimposed"
                ),
                help="Threshold energi superimposed terhadap baseline pre-fault.",
                key=f"{key_prefix}_superimposed_threshold_sigma",
            )

        with col_adv3:
            consecutive_samples_input = st.number_input(
                "Consecutive Samples",
                value=0,
                min_value=0,
                max_value=200,
                step=1,
                help="0 = otomatis sekitar 0.1 siklus. Nilai lebih besar menolak spike sesaat.",
                disabled=(not use_advanced_fault_detection or use_auto_fault_detection),
                key=f"{key_prefix}_consecutive_samples_input",
            )

        with col_adv4:
            refine_fault_bar = st.checkbox(
                "Refine Fault Bar",
                value=True,
                help="Backtrack dari kandidat RMS ke perubahan instantaneous awal.",
                disabled=(not use_advanced_fault_detection or use_auto_fault_detection),
                key=f"{key_prefix}_refine_fault_bar",
            )

    if use_auto_fault_detection:
        use_advanced_fault_detection = True
        fault_detection_method = auto_fault_detection_settings["fault_detection_method"]
        adaptive_threshold_sigma = auto_fault_detection_settings["adaptive_threshold_sigma"]
        superimposed_threshold_sigma = auto_fault_detection_settings["superimposed_threshold_sigma"]
        consecutive_samples_input = 0
        refine_fault_bar = auto_fault_detection_settings["refine_fault_bar"]

    detection = detect_fault_inception(
        assigned_df,
        frequency=frequency,
        current_threshold_multiplier=current_threshold_multiplier,
        voltage_drop_threshold=voltage_drop_threshold,
        min_prefault_cycles=int(pre_fault_cycles),
        adaptive_threshold_sigma=(
            adaptive_threshold_sigma
            if use_advanced_fault_detection
            else None
        ),
        consecutive_samples=(
            None
            if (
                not use_advanced_fault_detection
                or int(consecutive_samples_input) == 0
            )
            else int(consecutive_samples_input)
        ),
        refine_fault_bar=(
            refine_fault_bar
            if use_advanced_fault_detection
            else False
        ),
        method=(
            fault_detection_method
            if use_advanced_fault_detection
            else "legacy_rms"
        ),
        superimposed_threshold_sigma=superimposed_threshold_sigma,
        nominal_phase_voltage_rms=transformer_data.get("nominal_phase_voltage_rms"),
        nominal_current_rms=transformer_data.get("nominal_current_rms"),
    )

    st.session_state[fault_detection_key] = detection

    if detection["detected"]:
        st.success("Awal gangguan remote berhasil terdeteksi otomatis.")

        fault_window = build_fault_window(
            assigned_df,
            fault_index=detection["fault_index"],
            samples_per_cycle=detection["samples_per_cycle"],
            pre_fault_cycles=int(pre_fault_cycles),
            post_fault_cycles=int(post_fault_cycles),
        )

        st.session_state[fault_window_key] = fault_window
        samples_per_cycle = detection["samples_per_cycle"]

        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        col_r1.metric("Fault Time", f'{fault_window["fault_time"]:.6f} s')
        col_r2.metric("Left Cursor", f'{fault_window["left_time"]:.6f} s')
        col_r3.metric("Right Cursor", f'{fault_window["right_time"]:.6f} s')
        col_r4.metric("DFT Cursor", f'{fault_window["dft_time"]:.6f} s')

        st.write("Sampling Rate:", f'{detection["fs"]:.2f} Hz')
        st.write("Samples per Cycle:", detection["samples_per_cycle"])

        if detection.get("refine_fault_bar"):
            st.caption(
                "Fault bar refinement: "
                f'RMS candidate {detection["rms_fault_time"]:.6f} s -> '
                f'refined {detection["fault_time"]:.6f} s. '
                f'Confidence {detection.get("confidence_score", 0):.2f}/10, '
                f'consecutive samples {detection.get("consecutive_samples", "-")}.'
            )

        if detection.get("superimposed"):
            superimposed = detection["superimposed"]
            st.caption(
                "Superimposed detector: "
                f'threshold {superimposed["threshold"]:.6f}, '
                f'peak energy {superimposed.get("peak_energy", 0.0):.6f}.'
            )

    else:
        st.warning(detection["message"])

        st.markdown("### Manual Cursor")

        fs = detection["fs"]
        samples_per_cycle = int(round(fs / frequency))

        min_time = float(assigned_df["time"].min())
        max_time = float(assigned_df["time"].max())

        manual_fault_time = st.slider(
            "Pilih waktu awal gangguan remote manual (s)",
            min_value=min_time,
            max_value=max_time,
            value=min_time,
            step=(max_time - min_time) / 1000,
            key=f"{key_prefix}_manual_fault_time",
        )

        fault_index = int(
            (assigned_df["time"] - manual_fault_time).abs().idxmin()
        )

        fault_window = build_fault_window(
            assigned_df,
            fault_index=fault_index,
            samples_per_cycle=samples_per_cycle,
            pre_fault_cycles=int(pre_fault_cycles),
            post_fault_cycles=int(post_fault_cycles),
        )

        st.session_state[fault_window_key] = fault_window

    st.session_state["remote_samples_per_cycle"] = samples_per_cycle
    st.session_state["remote_frequency_hz"] = frequency

    if fault_window_key in st.session_state:
        fault_window = st.session_state[fault_window_key]
        _info_text = (
            "Left Cursor dan Right Cursor digunakan sebagai range analisis. "
            "DFT Cursor digunakan pada Step 4 untuk mengambil fasor 1 siklus "
            "setelah awal gangguan."
        )
        _exp_label = "Detail Window Analisis" if compact else "Validasi Window Analisis"
        with st.expander(_exp_label, expanded=False):
            st.json(fault_window)
            st.info(_info_text)
            if not compact:
                remote_plot_channels = [
                    ch for ch in ["Va", "Vb", "Vc", "Ia", "Ib", "Ic", "IE"]
                    if ch in assigned_df.columns
                ]
                remote_selected_plot = st.multiselect(
                    "Pilih sinyal remote untuk validasi fault window",
                    remote_plot_channels,
                    default=[c for c in ["Ia", "Ib", "Ic"] if c in remote_plot_channels] or remote_plot_channels[:3],
                    key=f"{key_prefix}_fault_window_plot_channels",
                )
                if remote_selected_plot:
                    fig = build_fault_window_plot(
                        assigned_df, fault_window, remote_selected_plot,
                        "Remote Fault Detection dan Cursor Window",
                    )
                    _pad = (fault_window["right_time"] - fault_window["left_time"]) * 0.3
                    fig.update_layout(xaxis_range=[
                        fault_window["left_time"] - _pad,
                        fault_window["right_time"] + _pad,
                    ])
                    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_fault_window_chart")

        if compact:
            remote_plot_channels = [
                ch for ch in ["Va", "Vb", "Vc", "Ia", "Ib", "Ic", "IE"]
                if ch in assigned_df.columns
            ]
            remote_selected_plot = st.multiselect(
                "Pilih sinyal remote untuk validasi fault window",
                remote_plot_channels,
                default=[c for c in ["Ia", "Ib", "Ic"] if c in remote_plot_channels] or remote_plot_channels[:3],
                key=f"{key_prefix}_fault_window_plot_channels",
            )
            if remote_selected_plot:
                fig = build_fault_window_plot(
                    assigned_df, fault_window, remote_selected_plot,
                    "Remote Fault Detection dan Cursor Window",
                )
                _pad = (fault_window["right_time"] - fault_window["left_time"]) * 0.3
                fig.update_layout(xaxis_range=[
                    fault_window["left_time"] - _pad,
                    fault_window["right_time"] + _pad,
                ])
                st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_fault_window_chart")


def render_fault_cursor(
    end: str,
    assigned_df_key: str,
    metadata_key: str,
    transformer_key: str,
    fault_window_key: str,
    fault_detection_key: str,
    key_prefix: str,
    compact: bool = False,
):
    """Render fault detection & cursor UI for local or remote end.

    When ``compact=True`` (used in DE tab), secondary parameters are collapsed
    by default and the validation chart is shown directly without an expander.
    """
    metadata = st.session_state.get(metadata_key, {}) or {}

    if end == "local":
        _render_local_fault_cursor(
            metadata=metadata,
            assigned_df_key=assigned_df_key,
            transformer_key=transformer_key,
            fault_window_key=fault_window_key,
            fault_detection_key=fault_detection_key,
            key_prefix=key_prefix,
            compact=compact,
        )
    else:
        _render_remote_fault_cursor(
            metadata=metadata,
            assigned_df_key=assigned_df_key,
            transformer_key=transformer_key,
            fault_window_key=fault_window_key,
            fault_detection_key=fault_detection_key,
            key_prefix=key_prefix,
            compact=compact,
        )


# ---------------------------------------------------------------------------
# Formula display helpers
# ---------------------------------------------------------------------------

def _fmt_c(z: complex, unit: str = "", prec: int = 4) -> str:
    sign = "+" if z.imag >= 0 else "−"
    mag = abs(z)
    angle = float(np.degrees(np.angle(z)))
    u = f" {unit}" if unit else ""
    return f"{z.real:.{prec}f} {sign} j{abs(z.imag):.{prec}f}{u}  (|{mag:.{prec}f}| ∠ {angle:.2f}°)"


_SE_LOOP_LATEX = {
    "AG": (r"Z_{app} = \frac{V_a}{I_a + K_0 \cdot I_0}", True),
    "BG": (r"Z_{app} = \frac{V_b}{I_b + K_0 \cdot I_0}", True),
    "CG": (r"Z_{app} = \frac{V_c}{I_c + K_0 \cdot I_0}", True),
    "AB": (r"Z_{app} = \frac{V_a - V_b}{I_a - I_b}", False),
    "BC": (r"Z_{app} = \frac{V_b - V_c}{I_b - I_c}", False),
    "CA": (r"Z_{app} = \frac{V_c - V_a}{I_c - I_a}", False),
    "V1/I1": (r"Z_{app} = \frac{V_1}{I_1}", False),
}


def render_se_formula_expander(result: dict, line_param: dict, key_suffix: str = ""):
    """Expander berisi rumus SE dan nilai aktual yang digunakan dalam perhitungan."""
    selected_loop = result.get("selected_loop", "-")
    zapp = result["Zapp"]
    loop_v = result["loop_voltage"]
    loop_i = result["loop_current"]
    z1 = line_param["Z1_per_km"]
    k0 = line_param["K0"]
    L = line_param["length_km"]
    method = result.get("recommended_method", "reactance")

    loop_latex, is_ground = _SE_LOOP_LATEX.get(
        selected_loop, (r"Z_{app} = \frac{U_{loop}}{I_{loop}}", False)
    )

    with st.expander("Rumus dan Pembahasan Perhitungan Single-Ended", expanded=False):

        st.markdown("#### 1. Impedansi Loop Gangguan")
        st.latex(loop_latex)
        if is_ground:
            st.caption(
                "K₀ adalah faktor kompensasi earth return: K₀ = (Z₀ − Z₁) / Z₁, "
                "dengan I₀ = Iᴇ / 3. "
                "Ekuivalen dengan k₀ = (Z₀ − Z₁) / (3Z₁) × Iᴇ pada notasi standar."
            )
        st.caption("*Referensi: Saha et al. (2010) Eq. 6.3; Phadke & Thorp (2009) Eq. 9.33*")

        st.markdown("#### 2. Nilai yang Digunakan")
        c1, c2 = st.columns(2)
        c1.markdown(f"**U_loop** = `{_fmt_c(loop_v, 'kV')}`")
        c2.markdown(f"**I_loop** = `{_fmt_c(loop_i, 'A')}`")
        if is_ground:
            c1, c2 = st.columns(2)
            c1.markdown(f"**K₀** = `{_fmt_c(k0)}`")
            c2.markdown("*definisi: K₀ = (Z₀ − Z₁) / Z₁*")
        c1, c2 = st.columns(2)
        c1.markdown(f"**Z₁/km** = `{_fmt_c(z1, 'Ω/km')}`")
        c2.markdown(f"**L** = `{L:.6f} km`")

        st.markdown("#### 3. Impedansi Tampak (Zapp)")
        st.latex(r"Z_{app} = \frac{U_{loop}}{I_{loop}}")
        st.markdown(f"= `{_fmt_c(zapp, 'Ω')}`")

        st.markdown("#### 4. Kalkulasi Jarak — Tiga Metode")

        d_x = result["distance_x_km"]
        st.markdown("**Reactance method** *(default)*:")
        st.latex(r"d_X = \frac{\mathrm{Im}(Z_{app})}{\mathrm{Im}(Z_1/\mathrm{km})}")
        st.markdown(
            f"= {zapp.imag:.4f} / {z1.imag:.4f} = **{d_x:.4f} km** "
            f"({d_x / L * 100:.2f}%)"
        )
        st.caption("*Referensi: Saha et al. (2010) Eq. 6.2; Phadke & Thorp (2009) Eq. 9.34*")

        d_m = result["distance_mag_km"]
        st.markdown("**Magnitude method:**")
        st.latex(r"d_{|Z|} = \frac{|Z_{app}|}{|Z_1/\mathrm{km}|}")
        st.markdown(
            f"= {abs(zapp):.4f} / {abs(z1):.4f} = **{d_m:.4f} km** "
            f"({d_m / L * 100:.2f}%)"
        )
        st.caption("*Referensi: Saha et al. (2010) Eq. 6.22; IEEE Std C37.114-2014*")

        d_p = result["distance_projection_km"]
        st.markdown("**Projection method:**")
        st.latex(r"d_{proj} = \frac{\mathrm{Re}(Z_{app}\cdot\hat{Z}_1^{\,*})}{|Z_1/\mathrm{km}|}")
        st.markdown(f"= **{d_p:.4f} km** ({d_p / L * 100:.2f}%)")
        st.caption("*Referensi: Saha et al. (2010) Eq. 6.30–6.31*")

        rec = result["recommended_distance_km"]
        st.info(
            f"▶ **Metode yang dipakai:** `{method}` → **{rec:.4f} km** "
            f"({rec / L * 100:.2f}%)"
        )

        if result.get("used_superimposed_fallback") and result.get("superimposed_distance_x_km") is not None:
            st.markdown("#### 5. Fallback: Metode Takagi (Kompensasi Rᶠ)")
            st.caption(
                "Digunakan karena hasil conventional keluar batas saluran. "
                "Metode Takagi mengeliminasi Rᶠ secara eksak tanpa asumsi Rᶠ kecil."
            )
            st.latex(
                r"d_{Takagi} = \frac{\mathrm{Im}(U_{loop}\cdot\Delta I^*)}{\mathrm{Im}(Z_1\cdot I_{loop}\cdot\Delta I^*)}"
            )
            st.caption(
                "Dasar eliminasi: Im(Rᶠ·|ΔI|²) = 0 karena Rᶠ real dan |ΔI|² real.  "
                "*Referensi: Saha et al. (2010) Eq. 6.8; Phadke & Thorp (2009) Eq. 9.36*"
            )
            d_tk = result["superimposed_distance_x_km"]
            st.markdown(f"**Hasil Takagi:** **{d_tk:.4f} km** ({d_tk / L * 100:.2f}%)")
            if result.get("superimposed_Zapp_R") is not None:
                sup_z = complex(result["superimposed_Zapp_R"], result["superimposed_Zapp_X"])
                st.markdown(f"**Zapp superimposed (ΔV/ΔI):** `{_fmt_c(sup_z, 'Ω')}`")

