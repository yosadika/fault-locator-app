import cmath
import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app_helpers import downsample_xy, downsample_dataframe_for_plot
from fault_detection import estimate_sampling_rate, calculate_rms_sliding


def build_waveform_rms_summary(df: pd.DataFrame, channels: list[str], frequency: float = 50.0):
    try:
        fs = estimate_sampling_rate(df)
        samples_per_cycle = max(4, int(round(fs / max(float(frequency), 1e-9))))
        sample_count = min(len(df), max(samples_per_cycle, 3 * samples_per_cycle))
    except Exception:
        sample_count = min(len(df), 200)

    rows = []
    for channel in channels:
        if channel not in df.columns:
            continue

        values = pd.to_numeric(df[channel].iloc[:sample_count], errors="coerce").dropna().to_numpy()
        if len(values) == 0:
            continue

        rms = float(np.sqrt(np.mean(values ** 2)))
        peak_abs = float(np.nanmax(np.abs(values)))
        peak_to_rms = peak_abs / max(rms, 1e-9)

        rows.append(
            {
                "Signal": channel,
                "RMS Awal Rekaman": rms,
                "Peak Absolut Awal": peak_abs,
                "Peak/RMS": peak_to_rms,
            }
        )

    return pd.DataFrame(rows)


def build_rms_waveform_dataframe(df: pd.DataFrame, channels: list[str], frequency: float = 50.0):
    fs = estimate_sampling_rate(df)
    samples_per_cycle = max(4, int(round(fs / max(float(frequency), 1e-9))))

    rms_df = pd.DataFrame()
    rms_df["time"] = df["time"]

    for channel in channels:
        if channel in df.columns:
            rms_df[channel] = calculate_rms_sliding(df[channel].to_numpy(dtype=float), samples_per_cycle)

    return rms_df, samples_per_cycle


def build_assigned_waveform_plot(
    df: pd.DataFrame,
    channels: list[str],
    title: str,
    display_mode: str,
    frequency: float = 50.0,
):
    if display_mode == "RMS 1 siklus":
        plot_df, samples_per_cycle = build_rms_waveform_dataframe(df, channels, frequency)
        yaxis_title = "RMS Primary Magnitude"
        caption = f"Mode RMS memakai sliding window 1 siklus ({samples_per_cycle} sampel)."
    else:
        plot_df = df
        yaxis_title = "Instantaneous Primary Magnitude (peak)"
        caption = "Mode instantaneous menampilkan nilai sample/peak seperti waveform mentah."

    plot_df = downsample_dataframe_for_plot(plot_df, "time", channels)

    fig = px.line(
        plot_df,
        x="time",
        y=channels,
        title=title,
    )
    fig.update_layout(
        xaxis_title="Time (s)",
        yaxis_title=yaxis_title,
        legend_title="Signal",
    )

    return fig, caption


def build_wavewin_style_phasor_diagram(
    phasors: dict,
    signal_names: list[str],
    title: str,
    line_color: str = "#ff00ff",
):
    fig = go.Figure()

    available_signals = [
        name for name in signal_names
        if name in phasors and "complex" in phasors[name]
    ]

    if not available_signals:
        fig.update_layout(title=title)
        return fig

    max_magnitude = max(abs(phasors[name]["complex"]) for name in available_signals)
    radial_max = max(max_magnitude * 1.18, 1.0)

    for degree in range(0, 360, 10):
        theta = math.radians(degree)
        tick_inner = radial_max * (0.965 if degree % 30 else 0.94)
        tick_outer = radial_max
        fig.add_shape(
            type="line",
            x0=tick_inner * math.cos(theta),
            y0=tick_inner * math.sin(theta),
            x1=tick_outer * math.cos(theta),
            y1=tick_outer * math.sin(theta),
            line=dict(color="#9ca3af", width=0.6 if degree % 30 else 1.0),
        )

    fig.add_shape(
        type="circle",
        x0=-radial_max,
        y0=-radial_max,
        x1=radial_max,
        y1=radial_max,
        line=dict(color="#9ca3af", width=0.8),
    )

    for degree, label in [
        (0, "0"),
        (30, "30"),
        (60, "60"),
        (90, "90"),
        (120, "120"),
        (150, "150"),
        (180, "180"),
        (210, "210"),
        (240, "240"),
        (270, "270"),
        (300, "300"),
        (330, "330"),
    ]:
        theta = math.radians(degree)
        fig.add_annotation(
            x=radial_max * 1.08 * math.cos(theta),
            y=radial_max * 1.08 * math.sin(theta),
            text=label,
            showarrow=False,
            font=dict(size=11, color="#1d4ed8"),
        )

    fig.add_shape(
        type="line",
        x0=-radial_max,
        y0=0,
        x1=radial_max,
        y1=0,
        line=dict(color="#6b7280", width=0.8, dash="dash"),
    )
    fig.add_shape(
        type="line",
        x0=0,
        y0=-radial_max,
        x1=0,
        y1=radial_max,
        line=dict(color="#6b7280", width=0.8, dash="dash"),
    )

    for signal_name in available_signals:
        z = phasors[signal_name]["complex"]
        fig.add_trace(
            go.Scatter(
                x=[0, z.real],
                y=[0, z.imag],
                mode="lines+markers+text",
                text=["", signal_name],
                textposition="middle right",
                name=signal_name,
                line=dict(color=line_color, width=2),
                marker=dict(color=line_color, size=[3, 7]),
                customdata=[
                    [signal_name, 0.0, 0.0],
                    [signal_name, abs(z), math.degrees(cmath.phase(z))],
                ],
                hovertemplate=(
                    "%{customdata[0]}<br>"
                    "RMS %{customdata[1]:.3f}<br>"
                    "Angle %{customdata[2]:.2f} deg"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=title,
        width=520,
        height=560,
        showlegend=True,
        xaxis=dict(
            range=[-radial_max * 1.18, radial_max * 1.18],
            zeroline=False,
            showgrid=False,
            visible=False,
        ),
        yaxis=dict(
            range=[-radial_max * 1.18, radial_max * 1.18],
            zeroline=False,
            showgrid=False,
            visible=False,
            scaleanchor="x",
            scaleratio=1,
        ),
        margin=dict(l=20, r=20, t=58, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#111827"),
    )

    return fig


def add_fault_window_vlines(fig, fault_window, prefix=""):
    labels = {
        "left_time": "Left Cursor",
        "fault_time": "Fault",
        "dft_time": "DFT Cursor",
        "right_time": "Right Cursor",
    }
    styles = {
        "left_time": "dash",
        "fault_time": "solid",
        "dft_time": "dot",
        "right_time": "dash",
    }
    positions = {
        "left_time": "top left",
        "fault_time": "top",
        "dft_time": "top",
        "right_time": "top right",
    }

    for key, label in labels.items():
        fig.add_vline(
            x=fault_window[key],
            line_dash=styles[key],
            annotation_text=f"{prefix}{label}",
            annotation_position=positions[key],
        )


def build_fault_window_plot(df, fault_window, selected_channels, title):
    plot_df = downsample_dataframe_for_plot(df, "time", selected_channels)

    fig = px.line(
        plot_df,
        x="time",
        y=selected_channels,
        title=title,
    )

    add_fault_window_vlines(fig, fault_window)

    fig.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Magnitude Primary",
        legend_title="Signal",
    )

    return fig


def build_synchronized_fault_plot(
    local_df,
    remote_df,
    local_fault_window,
    remote_fault_window,
    selected_channels,
    title,
    remote_time_shift_s=0.0,
):
    fig = go.Figure()

    local_time = local_df["time"] - local_fault_window["fault_time"]
    remote_time = (
        remote_df["time"]
        - remote_fault_window["fault_time"]
        + remote_time_shift_s
    )

    for channel in selected_channels:
        if channel in local_df.columns:
            local_x, local_y = downsample_xy(local_time, local_df[channel])
            fig.add_trace(
                go.Scatter(
                    x=local_x,
                    y=local_y,
                    mode="lines",
                    name=f"Local {channel}",
                    line=dict(width=1.4),
                )
            )

        if channel in remote_df.columns:
            remote_x, remote_y = downsample_xy(remote_time, remote_df[channel])
            fig.add_trace(
                go.Scatter(
                    x=remote_x,
                    y=remote_y,
                    mode="lines",
                    name=f"Remote {channel}",
                    line=dict(width=1.4, dash="dash"),
                )
            )

    sync_events = [
        (0.0, "Fault", "solid"),
        (
            local_fault_window["left_time"] - local_fault_window["fault_time"],
            "Local Left",
            "dash",
        ),
        (
            local_fault_window["dft_time"] - local_fault_window["fault_time"],
            "Local DFT",
            "dot",
        ),
        (
            local_fault_window["right_time"] - local_fault_window["fault_time"],
            "Local Right",
            "dash",
        ),
        (
            remote_fault_window["dft_time"] - remote_fault_window["fault_time"] + remote_time_shift_s,
            "Remote DFT",
            "dot",
        ),
    ]

    for x_value, label, dash in sync_events:
        fig.add_vline(
            x=x_value,
            line_dash=dash,
            annotation_text=label,
            annotation_position="top",
        )

    left_limit = min(
        local_fault_window["left_time"] - local_fault_window["fault_time"],
        remote_fault_window["left_time"] - remote_fault_window["fault_time"] + remote_time_shift_s,
    )
    right_limit = max(
        local_fault_window["right_time"] - local_fault_window["fault_time"],
        remote_fault_window["right_time"] - remote_fault_window["fault_time"] + remote_time_shift_s,
    )

    fig.update_layout(
        title=title,
        xaxis_title="Aligned Time from Local Fault (s)",
        yaxis_title="Magnitude Primary",
        legend_title="Signal",
        xaxis=dict(range=[left_limit, right_limit], autorange=False),
    )

    return fig


def estimate_waveform_time_shift_by_correlation(
    local_df,
    remote_df,
    local_fault_window,
    remote_fault_window,
    reference_channels,
    window_left_s,
    window_right_s,
    frequency=50.0,
    method="raw_correlation",
):
    local_time = np.asarray(local_df["time"] - local_fault_window["fault_time"], dtype=float)
    remote_time = np.asarray(remote_df["time"] - remote_fault_window["fault_time"], dtype=float)

    dt_candidates = []

    for time_values in [local_time, remote_time]:
        diffs = np.diff(time_values)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if len(diffs) > 0:
            dt_candidates.append(float(np.median(diffs)))

    if not dt_candidates:
        return 0.0, 0.0

    dt = min(dt_candidates)
    grid = np.arange(window_left_s, window_right_s, dt)

    if len(grid) < 8:
        return 0.0, 0.0

    samples_per_cycle = max(2, int(round((1.0 / max(float(frequency), 1e-9)) / dt)))

    def moving_average(values, window):
        if window <= 1 or len(values) < window:
            return values

        kernel = np.ones(window) / window
        return np.convolve(values, kernel, mode="same")

    def one_cycle_superimposed(values):
        delta = np.zeros_like(values)
        if len(values) > samples_per_cycle:
            delta[samples_per_cycle:] = values[samples_per_cycle:] - values[:-samples_per_cycle]
        return moving_average(np.abs(delta), max(2, samples_per_cycle // 4))

    def rms_envelope(values):
        squared = values ** 2
        return np.sqrt(np.maximum(moving_average(squared, samples_per_cycle), 0.0))

    def detect_envelope_sag_onset(grid_values, envelope_values):
        values = np.asarray(envelope_values, dtype=float)
        finite = np.isfinite(values)
        if np.count_nonzero(finite) < max(8, samples_per_cycle):
            return None

        high_level = float(np.nanpercentile(values[finite], 90))
        low_level = float(np.nanpercentile(values[finite], 10))
        if high_level <= 1e-9 or (high_level - low_level) < 0.08 * high_level:
            return None

        threshold = high_level - 0.35 * (high_level - low_level)
        below = values < threshold
        consecutive = max(2, int(round(0.25 * samples_per_cycle)))
        counter = 0

        for index, is_below in enumerate(below):
            if is_below:
                counter += 1
                if counter >= consecutive:
                    return float(grid_values[max(0, index - consecutive + 1)])
            else:
                counter = 0

        return None

    if method == "voltage_sine_sag_hybrid":
        raw_shift, raw_score = estimate_waveform_time_shift_by_correlation(
            local_df,
            remote_df,
            local_fault_window,
            remote_fault_window,
            reference_channels,
            window_left_s,
            window_right_s,
            frequency=frequency,
            method="raw_correlation",
        )

        sag_shifts = []
        for channel in reference_channels:
            if channel not in local_df.columns or channel not in remote_df.columns:
                continue

            local_values = np.interp(grid, local_time, np.asarray(local_df[channel], dtype=float))
            remote_values = np.interp(grid, remote_time, np.asarray(remote_df[channel], dtype=float))
            local_envelope = rms_envelope(local_values)
            remote_envelope = rms_envelope(remote_values)
            local_sag_time = detect_envelope_sag_onset(grid, local_envelope)
            remote_sag_time = detect_envelope_sag_onset(grid, remote_envelope)

            if local_sag_time is not None and remote_sag_time is not None:
                sag_shifts.append(local_sag_time - remote_sag_time)

        if sag_shifts:
            sag_shift = float(np.median(sag_shifts))
            blended_shift = 0.65 * sag_shift + 0.35 * raw_shift
            max_reasonable_shift = 0.5 * (window_right_s - window_left_s)
            blended_shift = max(-max_reasonable_shift, min(max_reasonable_shift, blended_shift))
            return blended_shift, max(raw_score, 0.75)

        return raw_shift, raw_score

    local_stack = []
    remote_stack = []

    for channel in reference_channels:
        if channel not in local_df.columns or channel not in remote_df.columns:
            continue

        local_values = np.interp(grid, local_time, np.asarray(local_df[channel], dtype=float))
        remote_values = np.interp(grid, remote_time, np.asarray(remote_df[channel], dtype=float))

        if method == "superimposed_energy":
            local_values = one_cycle_superimposed(local_values)
            remote_values = one_cycle_superimposed(remote_values)
        elif method == "rms_envelope":
            local_values = rms_envelope(local_values)
            remote_values = rms_envelope(remote_values)

        local_values = local_values - np.nanmean(local_values)
        remote_values = remote_values - np.nanmean(remote_values)

        local_std = np.nanstd(local_values)
        remote_std = np.nanstd(remote_values)

        if local_std < 1e-9 or remote_std < 1e-9:
            continue

        local_stack.append(local_values / local_std)
        remote_stack.append(remote_values / remote_std)

    if not local_stack:
        return 0.0, 0.0

    local_signal = np.mean(np.vstack(local_stack), axis=0)
    remote_signal = np.mean(np.vstack(remote_stack), axis=0)

    correlation = np.correlate(local_signal, remote_signal, mode="full")
    inverted_correlation = np.correlate(local_signal, -remote_signal, mode="full")

    if np.max(inverted_correlation) > np.max(correlation):
        correlation = inverted_correlation

    lag_index = int(np.argmax(correlation) - (len(remote_signal) - 1))
    shift_s = lag_index * dt
    score = float(np.max(correlation) / max(len(local_signal), 1))

    max_reasonable_shift = 0.5 * (window_right_s - window_left_s)
    shift_s = max(-max_reasonable_shift, min(max_reasonable_shift, shift_s))

    return shift_s, score


def fault_phase_to_current_channel(fault_type: str):
    fault_type = str(fault_type or "").upper()

    if "A" in fault_type:
        return "Ia"
    if "B" in fault_type:
        return "Ib"
    if "C" in fault_type:
        return "Ic"

    return None


def fault_phase_to_voltage_channel(fault_type: str):
    fault_type = str(fault_type or "").upper()

    if "A" in fault_type:
        return "Va"
    if "B" in fault_type:
        return "Vb"
    if "C" in fault_type:
        return "Vc"

    return None


def get_phasor_magnitude(phasors, signal_name):
    if not phasors or signal_name not in phasors:
        return None

    try:
        return float(phasors[signal_name]["magnitude"])
    except (TypeError, ValueError, KeyError):
        return None


def get_phasor_angle(phasors, signal_name):
    if not phasors or signal_name not in phasors:
        return None

    try:
        return float(phasors[signal_name]["angle_deg"])
    except (TypeError, ValueError, KeyError):
        return None


def build_prefault_fault_comparison_dataframe(
    fault_phasors,
    prefault_phasors,
    side_label,
):
    rows = []

    for signal_name in ["Va", "Vb", "Vc", "Ia", "Ib", "Ic", "IE"]:
        fault_mag = get_phasor_magnitude(fault_phasors, signal_name)
        prefault_mag = get_phasor_magnitude(prefault_phasors, signal_name)

        if fault_mag is None and prefault_mag is None:
            continue

        delta = None
        delta_percent = None

        if fault_mag is not None and prefault_mag is not None:
            delta = fault_mag - prefault_mag
            if abs(prefault_mag) > 1e-9:
                delta_percent = 100.0 * delta / prefault_mag

        rows.append(
            {
                "Record": side_label,
                "Signal": signal_name,
                "Pre-fault RMS": prefault_mag,
                "Fault RMS": fault_mag,
                "Delta RMS": delta,
                "Delta %": delta_percent,
                "Fault Angle deg": get_phasor_angle(fault_phasors, signal_name),
            }
        )

    return pd.DataFrame(rows)
