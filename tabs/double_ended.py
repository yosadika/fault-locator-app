import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from app_helpers import invert_current_phasors
from fault_detection import estimate_sampling_rate
from fault_type import (
    calculate_auto_fault_type_thresholds,
    detect_fault_type,
)
from fault_workflow_helpers import (
    calculate_time_based_fault_location,
    explain_sync_warning,
    explain_two_ended_quality,
    get_absolute_event_time,
    render_fault_cursor,
)
from line_analysis_helpers import (
    build_remote_single_signed_position,
    build_two_ended_comparison_dataframe,
    build_two_ended_reverse_result,
    calculate_remote_aligned_dft_index,
    choose_best_remote_dft_for_two_ended,
    classify_two_ended_operating_status,
    infer_gi_names_from_line_name,
    override_line_param_length,
    reverse_line_name,
)
from phasor import (
    add_sequence_components_to_phasor_dict,
    calculate_all_phasors,
)
from single_ended import (
    build_single_ended_result_dataframe,
    calculate_single_ended_fault_location,
)
from summary_helpers import single_ended_plot_score
from two_ended import (
    build_two_ended_result_dataframe,
    calculate_positive_sequence_two_ended,
    choose_best_remote_current_direction,
    choose_best_two_ended_adaptation,
    evaluate_two_ended_quality,
    transform_remote_phasors,
)
from waveform_helpers import (
    build_synchronized_fault_plot,
    build_wavewin_style_phasor_diagram,
    estimate_waveform_time_shift_by_correlation,
    fault_phase_to_current_channel,
    fault_phase_to_voltage_channel,
)


def render():
    st.subheader("Two-Ended Fault Locator")

    st.write(
        "Fitur ini menghitung lokasi gangguan menggunakan dua rekaman distance relay "
        "dari dua ujung saluran yang saling berhadapan. Perhitungan awal memakai "
        "positive-sequence two-ended method."
    )

    if "phasors" not in st.session_state:
        st.warning("Selesaikan dulu Step 5: Phasor Calculation untuk rekaman local end.")
        return

    if "line_param" not in st.session_state:
        st.warning("Selesaikan dulu Step 7: Line Parameter.")
        return

    st.markdown("### Fault Cursor")
    st.caption(
        "Atur fault cursor local dan remote langsung dari halaman ini tanpa berpindah ke "
        "Local End atau Remote End. Pengaturan di sini dan di halaman End tersinkron melalui "
        "session state yang sama."
    )
    with st.expander("Local Fault Cursor", expanded=False):
        render_fault_cursor(
            end="local",
            assigned_df_key="assigned_df",
            metadata_key="local_metadata",
            transformer_key="local_transformer_data",
            fault_window_key="fault_window",
            fault_detection_key="fault_detection",
            key_prefix="de_local_fc",
            compact=True,
        )
    with st.expander("Remote Fault Cursor", expanded=False):
        if "remote_assigned_df" not in st.session_state:
            st.info("Selesaikan Remote End > Signals terlebih dahulu untuk mengatur fault cursor remote.")
        else:
            render_fault_cursor(
                end="remote",
                assigned_df_key="remote_assigned_df",
                metadata_key="remote_metadata",
                transformer_key="remote_transformer_data",
                fault_window_key="remote_fault_window",
                fault_detection_key="remote_fault_detection",
                key_prefix="de_remote_fc",
                compact=True,
            )

    local_phasors = st.session_state["phasors"]
    line_param_original = st.session_state["line_param"]
    tower_length_km = st.session_state.get("tower_schedule_selected_length_km")
    tower_length_source = st.session_state.get("tower_schedule_selected_length_source", "Tower Schedule")
    length_source_options = ["line_parameter"]
    if tower_length_km is not None:
        length_source_options.append("tower_schedule")

    current_length_source = st.session_state.get("two_ended_line_length_source", "line_parameter")
    if current_length_source not in length_source_options:
        current_length_source = "line_parameter"

    if current_length_source == "tower_schedule" and tower_length_km is not None:
        line_param = override_line_param_length(
            line_param_original,
            float(tower_length_km),
            f"Tower Schedule - {tower_length_source}",
        )
    else:
        line_param = dict(line_param_original)
        line_param["length_source"] = "Line Parameter"

    st.markdown("### Local End Data")

    col_l1, col_l2, col_l3 = st.columns(3)

    col_l1.metric("Local V1 RMS", f'{local_phasors["V1"]["magnitude"]:.3f}')
    col_l2.metric("Local I1 RMS", f'{local_phasors["I1"]["magnitude"]:.3f}')
    col_l3.metric("Line Length", f'{line_param["length_km"]:.6f} km')

    missing_remote_inputs = []
    if "remote_assigned_df" not in st.session_state:
        missing_remote_inputs.append("Remote End > Signals")
    if "remote_fault_window" not in st.session_state:
        missing_remote_inputs.append("Remote End > Fault Cursor")
    if "remote_phasors" not in st.session_state:
        missing_remote_inputs.append("Remote End > Phasor")

    if missing_remote_inputs:
        st.warning(
            "Lengkapi dulu " + ", ".join(missing_remote_inputs) +
            " sebelum menjalankan double-ended calculation."
        )
        return

    local_metadata = st.session_state.get("local_metadata", {})
    remote_assigned_df = st.session_state["remote_assigned_df"]
    remote_metadata = st.session_state.get("remote_metadata", {})
    remote_phasors = st.session_state["remote_phasors"]
    remote_fault_window = st.session_state["remote_fault_window"]
    remote_detection = st.session_state.get("remote_fault_detection", {})
    remote_frequency = float(st.session_state.get("remote_frequency_hz", remote_metadata.get("frequency") or 50.0))
    remote_samples_per_cycle = int(
        st.session_state.get(
            "remote_samples_per_cycle",
            remote_detection.get("samples_per_cycle") or max(1, round(estimate_sampling_rate(remote_assigned_df) / remote_frequency)),
        )
    )
    st.markdown("### Local vs Remote Phasor Diagram")
    _col_pg, _col_ps = st.columns(2)
    with _col_pg:
        phasor_plot_group = st.radio(
            "Kelompok Fasor",
            ["Voltage", "Current"],
            horizontal=True,
            key="two_ended_phasor_plot_group",
        )
    with _col_ps:
        remote_phasor_plot_source = st.radio(
            "Sumber Fasor Remote",
            ["Uploaded remote", "Adapted remote for DE"],
            horizontal=True,
            key="two_ended_remote_phasor_plot_source",
            help=(
                "Uploaded remote menampilkan fasor sesuai rekaman remote. Adapted remote memakai fasor "
                "yang sudah dikoreksi arah/polaritas/sudut setelah perhitungan DE tersedia."
            ),
        )
    phasor_plot_signals = ["Va", "Vb", "Vc"] if phasor_plot_group == "Voltage" else ["Ia", "Ib", "Ic"]
    remote_phasors_for_plot = remote_phasors
    if (
        remote_phasor_plot_source == "Adapted remote for DE"
        and "two_ended_adapted_remote_phasors" in st.session_state
    ):
        remote_phasors_for_plot = st.session_state["two_ended_adapted_remote_phasors"]
    elif remote_phasor_plot_source == "Adapted remote for DE":
        st.caption("Adapted remote belum tersedia. Jalankan Calculate Two-Ended Fault Location terlebih dahulu.")

    col_phasor_local, col_phasor_remote = st.columns(2)
    with col_phasor_local:
        st.plotly_chart(
            build_wavewin_style_phasor_diagram(
                local_phasors,
                phasor_plot_signals,
                f"Local {phasor_plot_group} Phasors",
                line_color="#2563eb",
            ),
            use_container_width=True,
        )
    with col_phasor_remote:
        st.plotly_chart(
            build_wavewin_style_phasor_diagram(
                remote_phasors_for_plot,
                phasor_plot_signals,
                f"Remote {phasor_plot_group} Phasors",
                line_color="#ff00ff",
            ),
            use_container_width=True,
        )

    st.markdown("### Synchronization Check")

    local_fault_window = st.session_state.get("fault_window")

    if local_fault_window:
        local_fault_time = local_fault_window["fault_time"]
        remote_fault_time = remote_fault_window["fault_time"]

        local_dft_time = local_fault_window["dft_time"]
        remote_dft_time = remote_fault_window["dft_time"]

        local_samples_per_cycle = st.session_state["fault_detection"]["samples_per_cycle"]

        delta_fault_time = abs(local_fault_time - remote_fault_time)

        one_cycle_time = 1.0 / float(st.session_state["fault_detection"]["fs"]) * local_samples_per_cycle

        col_sync1, col_sync2, col_sync3 = st.columns(3)

        col_sync1.metric("Î” Fault Time", f"{delta_fault_time:.6f} s")
        col_sync2.metric("1 Cycle Time", f"{one_cycle_time:.6f} s")
        col_sync3.metric(
            "Trigger Sync Check",
            "OK" if delta_fault_time <= one_cycle_time else "Check",
        )

        if delta_fault_time > one_cycle_time:
            st.warning(
                "Perbedaan waktu fault local dan remote lebih dari 1 siklus. "
                "Cek sinkronisasi rekaman, trigger, sampling rate, atau pilih cursor manual."
            )
            st.info(explain_sync_warning())
        else:
            st.success("Perbedaan waktu fault local dan remote masih dalam batas 1 siklus.")

        st.markdown("#### Synchronized Local vs Remote Fault Waveform")
        st.caption(
            "Grafik ini menggeser waktu masing-masing rekaman sehingga fault lokal dan remote "
            "berada di t = 0 s. Untuk gangguan resistif, gunakan referensi fasa gangguan "
            "agar visualisasi tidak terlihat lebih mengikuti arus netral."
        )

        local_assigned_df = st.session_state.get("assigned_df")
        sync_plot_channels = [
            channel
            for channel in ["Va", "Vb", "Vc", "Ia", "Ib", "Ic", "IE"]
            if (
                local_assigned_df is not None
                and channel in local_assigned_df.columns
                and channel in remote_assigned_df.columns
            )
        ]
        sync_default_channels = [
            channel for channel in ["Va", "Vb", "Vc"] if channel in sync_plot_channels
        ]
        local_fault_type_for_sync = st.session_state.get("fault_type_result", {})
        remote_fault_type_for_sync = st.session_state.get("remote_fault_type_result", {})
        fault_phase_channel = (
            fault_phase_to_current_channel(local_fault_type_for_sync.get("fault_type"))
            or fault_phase_to_current_channel(remote_fault_type_for_sync.get("fault_type"))
        )
        fault_phase_voltage_channel = (
            fault_phase_to_voltage_channel(local_fault_type_for_sync.get("fault_type"))
            or fault_phase_to_voltage_channel(remote_fault_type_for_sync.get("fault_type"))
        )

        if fault_phase_voltage_channel in sync_plot_channels:
            sync_default_channels = [fault_phase_voltage_channel]
        elif fault_phase_channel in sync_plot_channels:
            sync_default_channels = [fault_phase_channel]

        sync_selected_channels = st.multiselect(
            "Pilih sinyal untuk grafik sinkronisasi local-remote",
            sync_plot_channels,
            default=sync_default_channels or sync_plot_channels[:3],
            key="sync_fault_waveform_channels",
        )

        if local_assigned_df is not None and sync_selected_channels:
            sync_reference_options = ["fault_cursor"]

            if fault_phase_channel in sync_plot_channels:
                sync_reference_options.append("fault_phase_current")

            if fault_phase_voltage_channel in sync_plot_channels:
                sync_reference_options.append("fault_phase_voltage")

            if "IE" in sync_plot_channels:
                sync_reference_options.append("ground_current")

            voltage_sync_channels = [
                channel for channel in ["Va", "Vb", "Vc"] if channel in sync_plot_channels
            ]
            if voltage_sync_channels:
                sync_reference_options.append("auto_voltage_sine")

            sync_reference_options.append("selected_channels")

            sync_reference_labels = {
                "fault_cursor": "Fault cursor only (tanpa korelasi waveform)",
                "fault_phase_current": f"Fault phase current ({fault_phase_channel})",
                "fault_phase_voltage": f"Fault phase voltage ({fault_phase_voltage_channel})",
                "ground_current": "Ground/neutral current (IE)",
                "auto_voltage_sine": "Auto voltage sine/sag alignment",
                "selected_channels": "Selected plotted channels",
            }
            default_sync_reference = "fault_cursor"

            sync_reference_mode = st.selectbox(
                "Referensi visual alignment waveform",
                sync_reference_options,
                format_func=lambda item: sync_reference_labels[item],
                index=sync_reference_options.index(default_sync_reference),
                key="sync_waveform_reference_mode",
                help=(
                    "Referensi ini dipakai untuk menggeser visualisasi. Jika opsi alignment DE diaktifkan, "
                    "cursor DFT remote juga dihitung ulang berdasarkan shift yang sama."
                ),
            )

            default_alignment_method = "rms_envelope"
            sync_alignment_method = st.selectbox(
                "Metode visual alignment",
                ["voltage_sine_sag_hybrid", "raw_correlation", "rms_envelope", "superimposed_energy"],
                format_func=lambda item: {
                    "voltage_sine_sag_hybrid": "Voltage sine + sag/drop hybrid (recommended tegangan)",
                    "superimposed_energy": "Transient/superimposed one-cycle energy (recommended resistif)",
                    "rms_envelope": "RMS envelope magnitude",
                    "raw_correlation": "Raw waveform correlation",
                }[item],
                index=["voltage_sine_sag_hybrid", "raw_correlation", "rms_envelope", "superimposed_energy"].index(default_alignment_method),
                key="sync_waveform_alignment_method",
                help=(
                    "Untuk tegangan, gunakan Voltage sine + sag/drop hybrid. Untuk arus/IE resistif, "
                    "superimposed energy biasanya lebih baik karena mengikuti perubahan fault/transient."
                ),
            )
            with st.expander("Advanced visual sync tuning", expanded=False):
                st.caption(
                    "Batas pencarian shift menentukan seberapa jauh aplikasi boleh menggeser rekaman remote "
                    "saat mencocokkan waveform. Default 0.300 s cukup untuk sebagian besar record; perbesar "
                    "hanya bila trigger local/remote terlihat sangat jauh."
                )
                sync_alignment_search_window_s = st.number_input(
                    "Batas pencarian shift alignment (s)",
                    value=0.300,
                    min_value=0.020,
                    max_value=2.000,
                    step=0.010,
                    format="%.3f",
                    key="sync_alignment_search_window_s",
                )

            sync_reference_channels = []
            if sync_reference_mode == "fault_phase_current" and fault_phase_channel:
                sync_reference_channels = [fault_phase_channel]
            elif sync_reference_mode == "fault_phase_voltage" and fault_phase_voltage_channel:
                sync_reference_channels = [fault_phase_voltage_channel]
            elif sync_reference_mode == "ground_current":
                sync_reference_channels = ["IE"]
            elif sync_reference_mode == "auto_voltage_sine":
                selected_voltage_channels = [
                    channel for channel in sync_selected_channels if channel in voltage_sync_channels
                ]
                sync_reference_channels = selected_voltage_channels or voltage_sync_channels
            elif sync_reference_mode == "selected_channels":
                sync_reference_channels = sync_selected_channels

            if (
                any(channel in voltage_sync_channels for channel in sync_selected_channels)
                and sync_reference_mode == "ground_current"
            ):
                st.warning(
                    "Grafik yang ditampilkan adalah tegangan, tetapi referensi alignment masih IE. "
                    "Untuk menyinkronkan gelombang tegangan seperti Vc, pilih "
                    "`Auto voltage sine/sag alignment` atau `Selected plotted channels`."
                )

            remote_visual_shift_s = 0.0
            correlation_score = 0.0

            if sync_reference_channels:
                left_limit = min(
                    local_fault_window["left_time"] - local_fault_window["fault_time"],
                    remote_fault_window["left_time"] - remote_fault_window["fault_time"],
                )
                right_limit = max(
                    local_fault_window["right_time"] - local_fault_window["fault_time"],
                    remote_fault_window["right_time"] - remote_fault_window["fault_time"],
                )
                remote_visual_shift_s, correlation_score = estimate_waveform_time_shift_by_correlation(
                    local_assigned_df,
                    remote_assigned_df,
                    local_fault_window,
                    remote_fault_window,
                    sync_reference_channels,
                    -sync_alignment_search_window_s,
                    sync_alignment_search_window_s,
                    frequency=remote_frequency,
                    method=sync_alignment_method,
                )

                st.caption(
                    "Visual alignment shift remote: "
                    f"{remote_visual_shift_s:+.6f} s berbasis {', '.join(sync_reference_channels)} "
                    f"dengan metode {sync_alignment_method} "
                    f"(correlation score {correlation_score:.3f})."
                )
                if correlation_score < 0.60:
                    st.warning(
                        "Skor alignment rendah. Untuk gangguan resistif, coba ubah referensi ke fasa gangguan "
                        "atau gunakan metode RMS envelope/superimposed, lalu cek ulang visual tegangan dan arus."
                    )

            sync_fig = build_synchronized_fault_plot(
                local_assigned_df,
                remote_assigned_df,
                local_fault_window,
                remote_fault_window,
                sync_selected_channels,
                "Synchronized Fault Waveform - Local vs Remote",
                remote_time_shift_s=remote_visual_shift_s,
            )
            st.plotly_chart(sync_fig, use_container_width=True)

            apply_sync_to_de = st.checkbox(
                "Gunakan alignment waveform ini untuk cursor DFT remote pada perhitungan DE",
                value=bool(sync_reference_channels),
                disabled=not bool(sync_reference_channels),
                key="apply_waveform_sync_to_de",
                help=(
                    "Default aktif. Fasor remote untuk double-ended dihitung ulang dari window DFT "
                    "yang sudah dikoreksi oleh shift alignment visual agar perhitungan DE mengikuti "
                    "rekaman yang telah disinkronkan."
                ),
            )

            if apply_sync_to_de and sync_reference_channels:
                try:
                    remote_aligned_dft_index = calculate_remote_aligned_dft_index(
                        remote_assigned_df,
                        local_fault_window,
                        remote_fault_window,
                        remote_visual_shift_s,
                    )
                    remote_aligned_phasors = calculate_all_phasors(
                        df=remote_assigned_df,
                        cursor_index=remote_aligned_dft_index,
                        samples_per_cycle=remote_samples_per_cycle,
                    )
                    remote_aligned_phasors = add_sequence_components_to_phasor_dict(
                        remote_aligned_phasors
                    )

                    st.session_state["two_ended_remote_phasors_for_calculation"] = remote_aligned_phasors
                    st.session_state["two_ended_remote_dft_index_for_calculation"] = remote_aligned_dft_index
                    st.session_state["two_ended_remote_sync_shift_s"] = remote_visual_shift_s
                    st.session_state["two_ended_remote_sync_score"] = correlation_score
                    st.session_state["two_ended_remote_sync_reference"] = ", ".join(sync_reference_channels)
                    st.session_state["two_ended_remote_sync_method"] = sync_alignment_method

                    remote_aligned_dft_time = float(remote_assigned_df["time"].iloc[remote_aligned_dft_index])
                    col_adft1, col_adft2, col_adft3 = st.columns(3)
                    col_adft1.metric("DE Remote DFT Time", f"{remote_aligned_dft_time:.6f} s")
                    col_adft2.metric("DE Remote DFT Index", remote_aligned_dft_index)
                    col_adft3.metric("Waveform Sync Score", f"{correlation_score:.3f}")
                    st.success(
                        "Fasor remote untuk perhitungan DE akan memakai cursor DFT yang sudah "
                        "dikoreksi alignment waveform."
                    )
                except Exception as sync_phasor_error:
                    st.session_state["two_ended_remote_phasors_for_calculation"] = remote_phasors
                    st.session_state["two_ended_remote_dft_index_for_calculation"] = remote_fault_window["dft_index"]
                    st.warning(
                        "Alignment waveform terdeteksi, tetapi fasor remote tersinkron tidak bisa dihitung. "
                        f"Perhitungan DE akan memakai cursor remote asli. Detail: {sync_phasor_error}"
                    )
            else:
                st.session_state["two_ended_remote_phasors_for_calculation"] = remote_phasors
                st.session_state["two_ended_remote_dft_index_for_calculation"] = remote_fault_window["dft_index"]
                st.session_state["two_ended_remote_sync_shift_s"] = 0.0
                st.session_state["two_ended_remote_sync_score"] = 0.0
                st.session_state["two_ended_remote_sync_reference"] = "fault_cursor"
                st.session_state["two_ended_remote_sync_method"] = "fault_cursor"
    else:
        st.warning("Local fault window belum tersedia.")

    st.empty()

    st.markdown("### Two-Ended Calculation")

    local_gi_label, remote_gi_label = infer_gi_names_from_line_name(
        line_param.get("line_name", "")
    )

    st.caption(
        f"Nama GI dibaca dari Line Name: {local_gi_label} sebagai sisi local dan "
        f"{remote_gi_label} sebagai sisi remote. Jika belum sesuai, ubah Line Name di tab Line Parameter."
    )

    selected_de_length_source = st.selectbox(
        "Sumber panjang line untuk kalkulasi DE",
        length_source_options,
        index=length_source_options.index(current_length_source),
        format_func=lambda value: {
            "line_parameter": (
                f"Line Parameter ({line_param_original['length_km']:.6f} km)"
            ),
            "tower_schedule": (
                f"Tower Schedule ({float(tower_length_km):.6f} km - {tower_length_source})"
                if tower_length_km is not None
                else "Tower Schedule belum tersedia"
            ),
        }[value],
        key="two_ended_line_length_source",
        help=(
            "Pilih Tower Schedule jika panjang saluran dari tabel tower lebih akurat daripada "
            "panjang pada Line Parameter. Data Tower Schedule harus dimuat dan difilter dahulu."
        ),
    )
    if selected_de_length_source == "tower_schedule" and tower_length_km is not None:
        st.info(
            f"Kalkulasi DE memakai panjang Tower Schedule: {float(tower_length_km):.6f} km. "
            "Z1_total/Z0_total dihitung ulang dari impedansi per km."
        )
    elif tower_length_km is None:
        st.caption(
            "Panjang Tower Schedule belum tersedia. Load dan filter data di tab Tower Schedule "
            "jika ingin memakai panjang saluran dari tower."
        )

    remote_direction_mode = st.selectbox(
        "Remote Record Adaptation",
        [
            "auto_adapt_record",
            "auto_current_direction_only",
            "into_line",
            "opposite_to_line",
        ],
        index=0,
        help=(
            "Auto Adapt mencoba arah arus, polaritas CT/VT, dan offset sudut antar rekaman. "
            "Direction only hanya mencoba arah arus remote. Pilih manual jika polaritas dan sinkronisasi sudah pasti."
        ),
    )

    two_ended_fault_scenario = st.selectbox(
        "Konteks rekaman dan gangguan",
        [
            "normal_internal_line_fault",
            "reverse_or_backfeed_external_fault",
        ],
        format_func=lambda value: {
            "normal_internal_line_fault": "Gangguan internal pada saluran yang direkam",
            "reverse_or_backfeed_external_fault": "Backfeed/reverse: fault eksternal atau di belakang remote",
            "sotf_parallel_or_adjacent_line": "Backfeed/reverse: fault eksternal atau di belakang remote",
        }.get(value, value),
        index=0,
        help=(
            "Pilih mode ini jika rekaman berasal dari line sehat/paralel, sedangkan gangguan diduga "
            "berada di line lain, area reverse, atau di belakang terminal remote. SOTF adalah salah satu "
            "kemungkinan penyebab, bukan syarat untuk memakai mode ini."
        ),
    )
    with st.expander("Advanced DE auto-sync experiment", expanded=False):
        st.caption(
            "Fitur ini memindai beberapa posisi DFT remote dan memilih kandidat yang terlihat paling konsisten. "
            "Karena hasil DE bisa berubah besar, default-nya nonaktif. Aktifkan hanya untuk investigasi "
            "ketika cursor remote jelas salah."
        )
        auto_de_dft_search = st.checkbox(
            "Auto-sync remote DFT cursor untuk kualitas DE terbaik",
            value=False,
            key="auto_de_dft_search",
            help=(
                "Saat Calculate DE ditekan, aplikasi memindai beberapa posisi DFT remote di sekitar fault "
                "dan memilih posisi yang memberi hasil two-ended paling konsisten."
            ),
        )
        de_dft_search_window_s = st.number_input(
            "Jendela scan DFT remote untuk DE (s)",
            value=0.300,
            min_value=0.020,
            max_value=2.000,
            step=0.010,
            format="%.3f",
            disabled=not auto_de_dft_search,
            key="de_dft_search_window_s",
        )

    if st.button("Calculate Two-Ended Fault Location"):
        try:
            remote_phasors_for_de = st.session_state.get(
                "two_ended_remote_phasors_for_calculation",
                remote_phasors,
            )
            if auto_de_dft_search:
                best_dft_sync, dft_sync_candidates = choose_best_remote_dft_for_two_ended(
                    local_phasors=local_phasors,
                    remote_df=remote_assigned_df,
                    remote_fault_window=remote_fault_window,
                    local_fault_window=local_fault_window,
                    line_param=line_param,
                    remote_samples_per_cycle=remote_samples_per_cycle,
                    remote_direction_mode=remote_direction_mode,
                    search_window_s=de_dft_search_window_s,
                )

                st.session_state["two_ended_dft_sync_candidates"] = dft_sync_candidates

                if best_dft_sync is not None:
                    remote_phasors_for_de = best_dft_sync["remote_phasors"]
                    st.session_state["two_ended_remote_phasors_for_calculation"] = remote_phasors_for_de
                    st.session_state["two_ended_remote_dft_index_for_calculation"] = best_dft_sync["remote_dft_index"]
                    st.session_state["two_ended_remote_sync_shift_s"] = best_dft_sync["implied_shift_s"]
                    st.session_state["two_ended_remote_sync_score"] = best_dft_sync["quality"]["quality_score"]
                    st.session_state["two_ended_remote_sync_reference"] = "auto_de_dft_quality_search"
                    st.session_state["two_ended_remote_sync_method"] = "dft_scan_quality"
                else:
                    st.warning(
                        "Auto-sync DFT remote tidak menemukan kandidat yang valid. "
                        "Perhitungan memakai cursor remote yang sedang aktif."
                    )
            adapted_remote_phasors = remote_phasors_for_de

            if remote_direction_mode == "auto_adapt_record":
                best_candidate, all_candidates = choose_best_two_ended_adaptation(
                    local_phasors=local_phasors,
                    remote_phasors=remote_phasors_for_de,
                    line_param=line_param,
                )

                if best_candidate["result"] is None:
                    raise ValueError("Auto adapt gagal menentukan pembacaan remote yang konsisten.")

                two_result = best_candidate["result"]
                two_quality = best_candidate["quality"]
                adapted_remote_phasors = best_candidate["adapted_remote_phasors"]

                st.session_state["two_ended_candidates"] = all_candidates
                st.session_state["two_ended_adapted_remote_phasors"] = adapted_remote_phasors

            elif remote_direction_mode == "auto_current_direction_only":
                best_candidate, all_candidates = choose_best_remote_current_direction(
                    local_phasors=local_phasors,
                    remote_phasors=remote_phasors_for_de,
                    line_param=line_param,
                )

                if best_candidate["result"] is None:
                    raise ValueError("Auto direction gagal menentukan arah arus remote.")

                two_result = best_candidate["result"]
                two_quality = best_candidate["quality"]
                adapted_remote_phasors = best_candidate["adapted_remote_phasors"]

                st.session_state["two_ended_candidates"] = all_candidates
                st.session_state["two_ended_adapted_remote_phasors"] = adapted_remote_phasors

            else:
                two_result = calculate_positive_sequence_two_ended(
                    local_phasors=local_phasors,
                    remote_phasors=remote_phasors_for_de,
                    line_param=line_param,
                    remote_current_direction=remote_direction_mode,
                )

                two_quality = evaluate_two_ended_quality(two_result, line_param)
                st.session_state.pop("two_ended_candidates", None)
                st.session_state["two_ended_adapted_remote_phasors"] = remote_phasors_for_de

            two_result.update(
                {
                    "calculation_reference_mode": "original_local_to_uploaded_remote",
                    "calculation_local_label": local_gi_label,
                    "calculation_remote_label": remote_gi_label,
                    "line_length_source": line_param.get("length_source", "Line Parameter"),
                    "line_length_km_used": line_param["length_km"],
                    "uploaded_remote_current_direction": two_result["remote_current_direction"],
                    "distance_from_original_local_km": two_result["distance_km"],
                    "distance_from_original_local_percent": two_result["distance_percent"],
                    "remote_waveform_sync_shift_s": st.session_state.get("two_ended_remote_sync_shift_s", 0.0),
                    "remote_waveform_sync_score": st.session_state.get("two_ended_remote_sync_score", 0.0),
                    "remote_waveform_sync_reference": st.session_state.get("two_ended_remote_sync_reference", "fault_cursor"),
                    "remote_waveform_sync_method": st.session_state.get("two_ended_remote_sync_method", "fault_cursor"),
                    "remote_dft_index_used": st.session_state.get(
                        "two_ended_remote_dft_index_for_calculation",
                        remote_fault_window["dft_index"],
                    ),
                }
            )

            two_reverse_result, two_reverse_quality = build_two_ended_reverse_result(
                local_phasors=local_phasors,
                adapted_remote_phasors=adapted_remote_phasors,
                normal_result=two_result,
                line_param=line_param,
                local_label=local_gi_label,
                remote_label=remote_gi_label,
            )

            two_comparison_df = build_two_ended_comparison_dataframe(
                normal_result=two_result,
                normal_quality=two_quality,
                reverse_result=two_reverse_result,
                reverse_quality=two_reverse_quality,
                local_label=local_gi_label,
                remote_label=remote_gi_label,
            )

            single_ended_compare_error = None

            try:
                local_fault_type_result = st.session_state.get("fault_type_result")

                if local_fault_type_result is None:
                    local_auto_fault_settings_for_de = calculate_auto_fault_type_thresholds(
                        local_phasors,
                        st.session_state.get("prefault_phasors"),
                    )
                    local_fault_type_result = detect_fault_type(
                        local_phasors,
                        prefault_phasors=st.session_state.get("prefault_phasors"),
                        voltage_drop_threshold=local_auto_fault_settings_for_de["voltage_drop_threshold"],
                        current_rise_threshold=local_auto_fault_settings_for_de["current_rise_threshold"],
                        ground_current_threshold=local_auto_fault_settings_for_de["ground_current_threshold"],
                        delta_current_threshold=local_auto_fault_settings_for_de["delta_current_threshold"],
                        delta_voltage_threshold=local_auto_fault_settings_for_de["delta_voltage_threshold"],
                    )

                remote_prefault_for_fault_type = (
                    transform_remote_phasors(
                        st.session_state["remote_prefault_phasors"],
                        voltage_polarity=two_result.get("remote_voltage_polarity", 1),
                        current_polarity=two_result.get("remote_current_polarity", 1),
                        angle_shift_deg=two_result.get("remote_angle_shift_deg", 0.0),
                    )
                    if "remote_prefault_phasors" in st.session_state
                    else None
                )
                remote_auto_fault_settings_for_de = calculate_auto_fault_type_thresholds(
                    adapted_remote_phasors,
                    remote_prefault_for_fault_type,
                )
                remote_fault_type_result = detect_fault_type(
                    phasors=adapted_remote_phasors,
                    prefault_phasors=remote_prefault_for_fault_type,
                    voltage_drop_threshold=remote_auto_fault_settings_for_de["voltage_drop_threshold"],
                    current_rise_threshold=remote_auto_fault_settings_for_de["current_rise_threshold"],
                    ground_current_threshold=remote_auto_fault_settings_for_de["ground_current_threshold"],
                    delta_current_threshold=remote_auto_fault_settings_for_de["delta_current_threshold"],
                    delta_voltage_threshold=remote_auto_fault_settings_for_de["delta_voltage_threshold"],
                )
                remote_single_raw_result = calculate_single_ended_fault_location(
                    phasors=adapted_remote_phasors,
                    fault_type_result=remote_fault_type_result,
                    line_param=line_param,
                    recommended_method="reactance",
                    prefault_phasors=remote_prefault_for_fault_type,
                )
                remote_single_phasors = adapted_remote_phasors
                remote_single_prefault_phasors = remote_prefault_for_fault_type

                if two_result.get("uploaded_remote_current_direction") == "opposite_to_line":
                    remote_single_phasors = invert_current_phasors(adapted_remote_phasors)
                    if remote_single_prefault_phasors is not None:
                        remote_single_prefault_phasors = invert_current_phasors(remote_single_prefault_phasors)

                local_single_result = calculate_single_ended_fault_location(
                    phasors=local_phasors,
                    fault_type_result=local_fault_type_result,
                    line_param=line_param,
                    recommended_method="reactance",
                    prefault_phasors=st.session_state.get("prefault_phasors"),
                )

                remote_single_result = calculate_single_ended_fault_location(
                    phasors=remote_single_phasors,
                    fault_type_result=remote_fault_type_result,
                    line_param=line_param,
                    recommended_method="reactance",
                    prefault_phasors=remote_single_prefault_phasors,
                )
                for se_result in (local_single_result, remote_single_result, remote_single_raw_result):
                    se_result["line_length_source"] = line_param.get("length_source", "Line Parameter")
                    se_result["line_length_km_used"] = line_param["length_km"]

                local_single_df = build_single_ended_result_dataframe(local_single_result)
                remote_single_df = build_single_ended_result_dataframe(remote_single_result)

                st.session_state["two_ended_local_single_result"] = local_single_result
                st.session_state["two_ended_remote_single_result"] = remote_single_result
                st.session_state["two_ended_remote_single_raw_result"] = remote_single_raw_result
                st.session_state["two_ended_local_single_df"] = local_single_df
                st.session_state["two_ended_remote_single_df"] = remote_single_df
                st.session_state["two_ended_local_fault_type_result"] = local_fault_type_result
                st.session_state["two_ended_remote_fault_type_result"] = remote_fault_type_result

            except Exception as single_ended_error:
                single_ended_compare_error = str(single_ended_error)
                for key in [
                    "two_ended_local_single_result",
                    "two_ended_remote_single_result",
                    "two_ended_remote_single_raw_result",
                    "two_ended_local_single_df",
                    "two_ended_remote_single_df",
                    "two_ended_local_fault_type_result",
                    "two_ended_remote_fault_type_result",
                ]:
                    st.session_state.pop(key, None)

            st.session_state["two_ended_result"] = two_result
            st.session_state["two_ended_quality"] = two_quality
            st.session_state["two_ended_reverse_result"] = two_reverse_result
            st.session_state["two_ended_reverse_quality"] = two_reverse_quality
            st.session_state["two_ended_comparison_df"] = two_comparison_df
            st.session_state["two_ended_single_ended_error"] = single_ended_compare_error
            st.session_state["two_ended_local_gi_label"] = local_gi_label
            st.session_state["two_ended_remote_gi_label"] = remote_gi_label
            st.session_state["two_ended_fault_scenario"] = two_ended_fault_scenario
            st.session_state["two_ended_operating_status"] = classify_two_ended_operating_status(
                two_result=two_result,
                two_quality=two_quality,
                line_param=line_param,
                local_single_result=st.session_state.get("two_ended_local_single_result"),
                remote_single_result=st.session_state.get("two_ended_remote_single_result"),
                scenario=two_ended_fault_scenario,
            )
            st.session_state.pop("two_ended_line_position_fig", None)
            st.success("Two-ended fault location berhasil dihitung.")

            if single_ended_compare_error:
                st.warning(
                    "Two-ended berhasil, tetapi single-ended comparison gagal: "
                    f"{single_ended_compare_error}"
                )

            st.rerun()

        except Exception as e:
            st.error("Two-ended fault location gagal.")
            st.exception(e)

    if "two_ended_result" in st.session_state:
        two_result = st.session_state["two_ended_result"]
        two_quality = st.session_state["two_ended_quality"]
        two_reverse_result = st.session_state.get("two_ended_reverse_result")
        two_reverse_quality = st.session_state.get("two_ended_reverse_quality")
        two_comparison_df = st.session_state.get("two_ended_comparison_df")
        local_gi_label = st.session_state.get("two_ended_local_gi_label", "GI Local")
        remote_gi_label = st.session_state.get("two_ended_remote_gi_label", "GI Remote")

        st.markdown("### Two-Ended Result")

        col_te1, col_te2, col_te3, col_te4 = st.columns(4)

        col_te1.metric(
            f"Original DE from {local_gi_label}",
            f'{two_result["distance_from_original_local_km"]:.3f} km',
        )

        col_te2.metric(
            f"Reverse DE from {remote_gi_label}",
            (
                f'{two_reverse_result["distance_km"]:.3f} km'
                if two_reverse_result
                else "-"
            ),
        )

        col_te3.metric(
            "Original Quality",
            f'{two_quality["quality_score"]}/10',
        )

        col_te4.metric(
            "Reverse Quality",
            f'{two_reverse_quality["quality_score"]}/10' if two_reverse_quality else "-",
        )

        st.info(explain_two_ended_quality(two_quality))

        operating_status = st.session_state.get("two_ended_operating_status")
        if operating_status:
            st.markdown("### Status Diagnostik Rekaman")
            status_text = ", ".join(operating_status.get("statuses", []))
            if operating_status.get("can_use_de_distance"):
                st.success(f"Status: {status_text}")
            else:
                st.warning(f"Status: {status_text}")
            for note in operating_status.get("notes", []):
                st.info(note)
            st.caption(operating_status.get("recommendation", ""))
        if two_comparison_df is not None:
            st.markdown("### Perbandingan Double-Ended Dua Arah")
            st.dataframe(
                two_comparison_df.style.format(
                    {
                        "Distance from Reference km": "{:.6f}",
                        f"Distance from {local_gi_label} km": "{:.6f}",
                        f"Distance from {local_gi_label} %": "{:.2f}",
                        f"Distance from {remote_gi_label} km": "{:.6f}",
                        "Quality": "{:.2f}",
                        "Mismatch Ratio": "{:.6f}",
                        "Imag Distance km": "{:.6f}",
                    }
                ),
                use_container_width=True,
            )

        two_result_df = build_two_ended_result_dataframe(two_result, two_quality)

        with st.expander("Detail Double-Ended Original Direction"):
            st.dataframe(
                two_result_df.style.format(
                    {
                        "Value": lambda x: f"{x:.6f}" if isinstance(x, (int, float)) else x
                    }
                ),
                use_container_width=True,
            )

        if two_reverse_result and two_reverse_quality:
            with st.expander("Detail Double-Ended Reverse Direction"):
                reverse_df = build_two_ended_result_dataframe(
                    two_reverse_result,
                    two_reverse_quality,
                )
                st.dataframe(
                    reverse_df.style.format(
                        {
                            "Value": lambda x: f"{x:.6f}"
                            if isinstance(x, (int, float))
                            else x
                        }
                    ),
                    use_container_width=True,
                )

        if st.session_state.get("two_ended_single_ended_error"):
            st.warning(
                "Single-ended comparison belum dapat ditampilkan: "
                f"{st.session_state['two_ended_single_ended_error']}"
            )

        if (
            "two_ended_local_single_result" in st.session_state
            and "two_ended_remote_single_result" in st.session_state
        ):
            local_single_result = st.session_state["two_ended_local_single_result"]
            remote_single_result = st.session_state["two_ended_remote_single_result"]
            local_single_df = st.session_state["two_ended_local_single_df"]
            remote_single_df = st.session_state["two_ended_remote_single_df"]
            local_fault_type_result = st.session_state.get(
                "two_ended_local_fault_type_result",
                {},
            )
            remote_fault_type_result = st.session_state.get(
                "two_ended_remote_fault_type_result",
                {},
            )

            st.markdown("### Single-Ended Comparison")

            L = line_param["length_km"]
            remote_single_position = build_remote_single_signed_position(
                line_length_km=L,
                remote_single_result=remote_single_result,
                scenario=st.session_state.get("two_ended_fault_scenario", two_ended_fault_scenario),
                two_result=two_result,
            )
            remote_single_signed_km = remote_single_position["signed_distance_from_remote_km"]
            remote_single_from_local_km = remote_single_position["distance_from_local_km"]
            remote_single_from_local_percent = remote_single_position["distance_from_local_percent"]

            col_se1, col_se2, col_se3, col_se4 = st.columns(4)

            col_se1.metric(
                f"{local_gi_label} SE",
                f'{local_single_result["recommended_distance_km"]:.3f} km',
            )
            col_se2.metric(
                f"{remote_gi_label} SE",
                f"{remote_single_signed_km:.3f} km",
            )
            col_se3.metric(
                f"{remote_gi_label} SE from {local_gi_label}",
                f"{remote_single_from_local_km:.3f} km",
            )
            col_se4.metric(
                f"{remote_gi_label} SE from {local_gi_label} %",
                f"{remote_single_from_local_percent:.2f} %",
            )

            if remote_single_position["is_reverse_external"]:
                st.caption(
                    f"Mode backfeed/reverse atau remote reverse aktif: single-ended {remote_gi_label} "
                    f"ditampilkan sebagai jarak signed negatif dari sisi remote. Konversi ke referensi "
                    f"{local_gi_label} menjadi L - (-d), sehingga posisi berada di luar ujung remote "
                    "pada line yang dianalisis."
                )
            else:
                st.caption(
                    f"Single-ended {remote_gi_label} ditampilkan sebagai jarak dari sisi remote dan "
                    f"konversinya ke referensi {local_gi_label}. Jika arah arus remote dipilih "
                    "`opposite_to_line`, arus fasor remote dibalik untuk perhitungan single-ended."
                )

            st.info(
                "Single-ended comparison adalah pembanding tambahan, bukan hasil utama double-ended. "
                "Status VALID/CHECK/UNCERTAIN pada baris local dan remote menunjukkan kewajaran estimasi satu ujung masing-masing relay. "
                "Jika pembanding ini memberi warning tetapi quality two-ended tinggi, prioritaskan hasil two-ended sambil tetap memeriksa waveform dan data relay."
            )

            se_summary_df = pd.DataFrame(
                [
                    {
                        "End": local_gi_label,
                        "Fault Type": local_fault_type_result.get("fault_type", "-"),
                        "Selected Loop": local_single_result["selected_loop"],
                        "Distance from Own End km": local_single_result["recommended_distance_km"],
                        f"Distance from {local_gi_label} km": local_single_result["recommended_distance_km"],
                        f"Distance from {local_gi_label} %": local_single_result["recommended_distance_percent"],
                        "Zapp R ohm": local_single_result["Zapp_R"],
                        "Zapp X ohm": local_single_result["Zapp_X"],
                        "Rf Est ohm": local_single_result["Rf_est_ohm"],
                        "Status": local_single_result["status"],
                    },
                    {
                        "End": remote_gi_label,
                        "Fault Type": remote_fault_type_result.get("fault_type", "-"),
                        "Selected Loop": remote_single_result["selected_loop"],
                        "Distance from Own End km": remote_single_signed_km,
                        f"Distance from {local_gi_label} km": remote_single_from_local_km,
                        f"Distance from {local_gi_label} %": remote_single_from_local_percent,
                        "Zapp R ohm": remote_single_result["Zapp_R"],
                        "Zapp X ohm": remote_single_result["Zapp_X"],
                        "Rf Est ohm": remote_single_result["Rf_est_ohm"],
                        "Status": remote_single_result["status"],
                    },
                ]
            )

            st.dataframe(
                se_summary_df.style.format(
                    {
                        "Distance from Own End km": "{:.6f}",
                        f"Distance from {local_gi_label} km": "{:.6f}",
                        f"Distance from {local_gi_label} %": "{:.2f}",
                        "Zapp R ohm": "{:.6f}",
                        "Zapp X ohm": "{:.6f}",
                        "Rf Est ohm": "{:.6f}",
                    }
                ),
                use_container_width=True,
            )

            with st.expander("Detail Single-Ended Local GI"):
                st.dataframe(
                    local_single_df.style.format(
                        {
                            "Value": lambda x: f"{x:.6f}"
                            if isinstance(x, (int, float))
                            else x
                        }
                    ),
                    use_container_width=True,
                )

            with st.expander("Detail Single-Ended Remote GI"):
                st.dataframe(
                    remote_single_df.style.format(
                        {
                            "Value": lambda x: f"{x:.6f}"
                            if isinstance(x, (int, float))
                            else x
                        }
                    ),
                    use_container_width=True,
                )

            if local_single_result["warnings"] or remote_single_result["warnings"]:
                st.markdown("### Warning Pembanding Single-Ended")
                st.info(
                    "Warning di bagian ini berasal dari perhitungan single-ended pembanding. "
                    "Artinya aplikasi melihat gejala yang perlu dicek pada estimasi satu ujung, bukan otomatis menyatakan hasil double-ended salah."
                )

            for warning in local_single_result["warnings"]:
                st.warning(f"Local GI single-ended comparison: {warning}")

            for warning in remote_single_result["warnings"]:
                st.warning(f"Remote GI single-ended comparison: {warning}")

        if two_quality["warnings"]:
            st.markdown("### Warning")
            for warning in two_quality["warnings"]:
                st.warning(warning)

        st.markdown("### Line Position Visualization")

        L = line_param["length_km"]
        line_display_name = str(line_param.get("line_name") or "").strip()
        if not line_display_name:
            line_display_name = f"{local_gi_label.replace('GI ', '')}-{remote_gi_label.replace('GI ', '')}"
        de_original_label = f"DE {line_display_name}"
        de_reverse_label = f"DE {reverse_line_name(line_display_name)}"
        marker_rows = [
            {
                "Point": de_original_label,
                "Distance km": two_result.get("distance_from_original_local_km", two_result["distance_km"]),
                "Score": two_quality["quality_score"],
                "Track": "Double-ended",
                "Color": "#2563eb",
                "Symbol": "diamond",
            }
        ]

        if two_reverse_result:
            marker_rows.append(
                {
                    "Point": de_reverse_label,
                    "Distance km": two_reverse_result["distance_from_original_local_km"],
                    "Score": two_reverse_quality["quality_score"] if two_reverse_quality else 0.0,
                    "Track": "Double-ended",
                    "Color": "#7c3aed",
                    "Symbol": "diamond-open",
                }
            )

        if "two_ended_local_single_result" in st.session_state:
            marker_rows.append(
                {
                    "Point": f"Single-ended {local_gi_label}",
                    "Distance km": st.session_state["two_ended_local_single_result"]["recommended_distance_km"],
                    "Score": single_ended_plot_score(st.session_state["two_ended_local_single_result"]),
                    "Track": "Single-ended",
                    "Color": "#059669",
                    "Symbol": "circle",
                }
            )

        if "two_ended_remote_single_result" in st.session_state:
            remote_marker_position = build_remote_single_signed_position(
                line_length_km=L,
                remote_single_result=st.session_state["two_ended_remote_single_result"],
                scenario=st.session_state.get("two_ended_fault_scenario", two_ended_fault_scenario),
                two_result=two_result,
            )
            marker_rows.append(
                {
                    "Point": f"Single-ended {remote_gi_label}",
                    "Distance km": remote_marker_position["distance_from_local_km"],
                    "Score": single_ended_plot_score(st.session_state["two_ended_remote_single_result"]),
                    "Track": "Single-ended",
                    "Color": "#d97706",
                    "Symbol": "circle-open",
                }
            )

        theme_base = st.get_option("theme.base")
        theme_background = st.get_option("theme.backgroundColor")
        if theme_base is None and theme_background:
            bg = str(theme_background).lstrip("#")
            if len(bg) >= 6:
                r = int(bg[0:2], 16)
                g = int(bg[2:4], 16)
                b = int(bg[4:6], 16)
                is_dark_theme = (0.2126 * r + 0.7152 * g + 0.0722 * b) < 128
            else:
                is_dark_theme = False
        else:
            is_dark_theme = theme_base == "dark"
        plot_template = "plotly_dark" if is_dark_theme else "plotly_white"
        paper_bg = "rgba(0,0,0,0)"
        plot_bg = "#0b1220" if is_dark_theme else "#ffffff"
        text_color = "#f8fafc" if is_dark_theme else "#111827"
        muted_text_color = "#cbd5e1" if is_dark_theme else "#475569"
        axis_title_color = "#f8fafc" if is_dark_theme else "#0f172a"
        annotation_bg = "rgba(15,23,42,0.92)" if is_dark_theme else "rgba(255,255,255,0.96)"
        annotation_border = "#94a3b8" if is_dark_theme else "#cbd5e1"
        terminal_color = "#f8fafc" if is_dark_theme else "#111827"

        sorted_marker_rows = sorted(marker_rows, key=lambda item: float(item["Distance km"]))
        min_gap_km = max(0.02 * L, 0.75)
        grouped_marker_rows = []

        for row in sorted_marker_rows:
            if (
                not grouped_marker_rows
                or abs(
                    float(row["Distance km"])
                    - float(grouped_marker_rows[-1][-1]["Distance km"])
                )
                >= min_gap_km
            ):
                grouped_marker_rows.append([row])
            else:
                grouped_marker_rows[-1].append(row)

        label_layout = {}
        double_slots = [
            (-64, -160),
            (-64, 160),
            (-100, -160),
            (-100, 160),
        ]
        single_slots = [
            (96, -190),
            (96, 190),
            (150, -190),
            (150, 190),
        ]

        for group in grouped_marker_rows:
            double_rows = [row for row in group if row["Track"] == "Double-ended"]
            single_rows = [row for row in group if row["Track"] == "Single-ended"]

            if len(group) == 1:
                row = group[0]
                label_layout[id(row)] = (
                    (-64, 0) if row["Track"] == "Double-ended" else (108, 0)
                )
            else:
                for index, row in enumerate(double_rows):
                    label_layout[id(row)] = double_slots[index % len(double_slots)]

                for index, row in enumerate(single_rows):
                    label_layout[id(row)] = single_slots[index % len(single_slots)]

        raw_marker_distances = [float(row["Distance km"]) for row in marker_rows]
        external_padding_km = max(0.05 * L, 1.0)
        x_min_km = min(0.0, min(raw_marker_distances) - external_padding_km)
        x_max_km = max(L, max(raw_marker_distances) + external_padding_km)

        for row in marker_rows:
            row["Distance km"] = float(row["Distance km"])
            row["Distance %"] = row["Distance km"] / L * 100.0
            row["Score"] = max(0.0, min(10.0, float(row["Score"])))
            short_name = (
                row["Point"]
                .replace("Double-ended", "DE")
                .replace("Single-ended", "SE")
            )
            row["Legend Name"] = short_name
            row["Label"] = (
                f"<b>{short_name}</b><br>"
                f"{row['Distance km']:.2f} km ({row['Distance %']:.1f}%)<br>"
                f"{row['Score']:.1f}/10"
            )
            row["Annotation Ay"], row["Annotation Ax"] = label_layout.get(
                id(row),
                (-64, 0) if row["Track"] == "Double-ended" else (108, 0),
            )

        marker_df = pd.DataFrame(marker_rows)

        fig_two = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.18, 0.82],
            vertical_spacing=0.08,
        )

        baseline_y = 0.0
        gi_line_y = 0.5
        gi_marker_y_base = 0.5

        fig_two.add_shape(
            type="line",
            x0=0,
            x1=L,
            y0=baseline_y,
            y1=baseline_y,
            line=dict(color=muted_text_color, width=1.2),
            row=2,
            col=1,
        )

        fig_two.add_shape(
            type="line",
            x0=0,
            x1=L,
            y0=gi_line_y,
            y1=gi_line_y,
            line=dict(color=muted_text_color, width=2),
            row=1,
            col=1,
        )

        fig_two.add_trace(
            go.Scatter(
                x=[0, L],
                y=[gi_line_y, gi_line_y],
                mode="markers",
                marker=dict(size=12, color=[terminal_color, terminal_color], symbol="square"),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        fig_two.add_annotation(
            x=0,
            y=gi_line_y,
            text=local_gi_label,
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            xshift=8,
            yshift=12,
            font=dict(color=text_color, size=13),
            row=1,
            col=1,
        )
        fig_two.add_annotation(
            x=L,
            y=gi_line_y,
            text=remote_gi_label,
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            xshift=-8,
            yshift=12,
            font=dict(color=text_color, size=13),
            row=1,
            col=1,
        )

        for index, (_, row) in enumerate(marker_df.sort_values(["Distance km", "Track"]).iterrows()):
            strip_y = gi_marker_y_base

            fig_two.add_trace(
                go.Scatter(
                    x=[row["Distance km"]],
                    y=[strip_y],
                    mode="markers",
                    marker=dict(
                        size=13,
                        color=row["Color"],
                        symbol=row["Symbol"],
                        line=dict(width=2, color=terminal_color),
                    ),
                    name=row["Legend Name"],
                    legendgroup=row["Point"],
                    showlegend=True,
                    hovertemplate=(
                        f"{row['Point']}<br>"
                        f"{row['Distance km']:.2f} km ({row['Distance %']:.1f}%)<br>"
                        f"Score {row['Score']:.1f}/10"
                        "<extra></extra>"
                    ),
                ),
                row=1,
                col=1,
            )

        x_profile = [
            x_min_km + i * (x_max_km - x_min_km) / 300.0
            for i in range(301)
        ]

        for _, row in marker_df.iterrows():
            center = float(row["Distance km"])
            score = float(row["Score"])
            curve_width = max(L * (0.09 if row["Track"] == "Double-ended" else 0.06), 2.5)
            curve_y = [
                baseline_y + score / (1.0 + abs(x - center) / curve_width)
                for x in x_profile
            ]

            fig_two.add_trace(
                go.Scatter(
                    x=x_profile,
                    y=curve_y,
                    mode="lines",
                    line=dict(color=row["Color"], width=1.5),
                    opacity=0.9,
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=row["Point"],
                ),
                row=2,
                col=1,
            )

            fig_two.add_trace(
                go.Scatter(
                    x=[row["Distance km"]],
                    y=[row["Score"]],
                    mode="markers",
                    marker=dict(
                        size=16,
                        color=row["Color"],
                        symbol=row["Symbol"],
                        line=dict(width=2, color=terminal_color),
                    ),
                    name=row["Legend Name"],
                    showlegend=False,
                    legendgroup=row["Point"],
                    customdata=[[row["Point"], row["Distance km"], row["Distance %"], row["Score"]]],
                    hovertemplate=(
                        "%{customdata[0]}<br>"
                        "%{customdata[1]:.2f} km (%{customdata[2]:.1f}%)<br>"
                        "Score %{customdata[3]:.1f}/10"
                        "<extra></extra>"
                    ),
                ),
                row=2,
                col=1,
            )

            fig_two.add_shape(
                type="line",
                x0=row["Distance km"],
                x1=row["Distance km"],
                y0=baseline_y,
                y1=row["Score"],
                line=dict(color=row["Color"], width=1.4),
                row=2,
                col=1,
            )

            fig_two.add_annotation(
                x=row["Distance km"],
                y=row["Score"],
                text=row["Label"],
                showarrow=True,
                arrowhead=2,
                arrowsize=0.8,
                arrowwidth=1.4,
                arrowcolor=row["Color"],
                ax=row["Annotation Ax"],
                ay=row["Annotation Ay"],
                bgcolor=annotation_bg,
                bordercolor=annotation_border,
                borderwidth=1,
                borderpad=4,
                font=dict(color=text_color, size=11),
                row=2,
                col=1,
            )

        fig_two.update_layout(
            title=f"Fault Location Map - {line_param.get('line_name', '')}",
            template=plot_template,
            paper_bgcolor=plot_bg,
            plot_bgcolor=plot_bg,
            font=dict(color=text_color),
            xaxis=dict(
                range=[x_min_km, x_max_km],
                autorange=False,
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                color=text_color,
            ),
            yaxis=dict(
                range=[0.0, 1.0],
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                color=text_color,
            ),
            xaxis2=dict(
                title=dict(
                    text=f"Distance from {local_gi_label} (km)",
                    font=dict(color=axis_title_color),
                ),
                range=[x_min_km, x_max_km],
                autorange=False,
                zeroline=False,
                color=text_color,
                tickfont=dict(color=muted_text_color),
                gridcolor="rgba(148,163,184,0.22)" if is_dark_theme else "rgba(148,163,184,0.35)",
            ),
            yaxis2=dict(
                title=dict(
                    text="Quality / Confidence (0-10)",
                    font=dict(color=axis_title_color),
                ),
                range=[-0.4, 11.4],
                showgrid=True,
                gridcolor="rgba(148,163,184,0.14)" if is_dark_theme else "rgba(148,163,184,0.22)",
                zeroline=False,
                color=text_color,
                tickfont=dict(color=muted_text_color),
            ),
            height=800,
            margin=dict(l=58, r=34, t=118, b=92),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.12,
                xanchor="right",
                x=1,
                bgcolor="rgba(15,23,42,0.85)" if is_dark_theme else "rgba(255,255,255,0.90)",
                bordercolor="#475569" if is_dark_theme else "#cbd5e1",
                borderwidth=1,
                font=dict(color=text_color),
            ),
        )

        fig_two.update_xaxes(
            tickfont=dict(color=muted_text_color),
            title_font=dict(color=axis_title_color),
            row=2,
            col=1,
        )
        fig_two.update_yaxes(
            tickfont=dict(color=muted_text_color),
            title_font=dict(color=axis_title_color),
            row=2,
            col=1,
        )

        st.plotly_chart(fig_two, use_container_width=True)

        if "two_ended_candidates" in st.session_state:
            st.markdown("### Auto Adaptation Candidates")

            candidate_rows = []

            for c in st.session_state["two_ended_candidates"][:50]:
                if c["result"] is None:
                    candidate_rows.append(
                        {
                            "Direction": c["direction"],
                            "VT Polarity": c.get("voltage_polarity"),
                            "CT Polarity": c.get("current_polarity"),
                            "Angle Shift deg": c.get("angle_shift_deg"),
                            "Distance km": None,
                            "Distance imag": None,
                            "Mismatch Ratio": None,
                            "Quality": None,
                            "Ranking Score": c["ranking_score"],
                            "Error": c.get("error"),
                        }
                    )
                else:
                    candidate_rows.append(
                        {
                            "Direction": c["direction"],
                            "VT Polarity": c.get("voltage_polarity", 1),
                            "CT Polarity": c.get("current_polarity", 1),
                            "Angle Shift deg": c.get("angle_shift_deg", 0.0),
                            "Distance km": c["result"]["distance_km"],
                            "Distance imag": c["result"]["distance_complex"].imag,
                            "Mismatch Ratio": c["quality"]["mismatch_ratio"],
                            "Quality": c["quality"]["quality_score"],
                            "Ranking Score": c["ranking_score"],
                            "Error": "",
                        }
                    )

            st.dataframe(pd.DataFrame(candidate_rows), use_container_width=True)

        if "two_ended_dft_sync_candidates" in st.session_state:
            with st.expander("Auto-Sync Remote DFT Candidates", expanded=False):
                dft_candidate_rows = []
                for candidate in st.session_state["two_ended_dft_sync_candidates"][:80]:
                    result = candidate.get("result", {})
                    quality = candidate.get("quality", {})
                    dft_candidate_rows.append(
                        {
                            "Remote DFT Index": candidate.get("remote_dft_index"),
                            "Remote DFT Time": candidate.get("remote_dft_time"),
                            "Implied Shift s": candidate.get("implied_shift_s"),
                            "Distance km": result.get("distance_km"),
                            "Distance imag km": result.get("distance_complex").imag if result.get("distance_complex") is not None else None,
                            "Quality": quality.get("quality_score"),
                            "Mismatch Ratio": quality.get("mismatch_ratio"),
                            "Ranking Score": candidate.get("ranking_score"),
                        }
                    )

                st.dataframe(
                    pd.DataFrame(dft_candidate_rows).style.format(
                        {
                            "Remote DFT Time": "{:.6f}",
                            "Implied Shift s": "{:.6f}",
                            "Distance km": "{:.6f}",
                            "Distance imag km": "{:.6f}",
                            "Quality": "{:.2f}",
                            "Mismatch Ratio": "{:.6f}",
                            "Ranking Score": "{:.6f}",
                        },
                        na_rep="-",
                    ),
                    use_container_width=True,
                )

        st.markdown("### Optional Time-Based / TWS Fault Locator")
        with st.expander("Hitung pembanding berbasis waktu / TWS", expanded=False):
            st.caption(
                "Metode ini opsional dan hanya layak dipakai bila kedua relay benar-benar sinkron SNTP/GPS "
                "serta event yang dibandingkan adalah arrival pertama. Hasilnya tidak otomatis menggantikan "
                "perhitungan phasor double-ended."
            )

            local_tws_fault_window = st.session_state.get("fault_window")
            remote_tws_fault_window = st.session_state.get("remote_fault_window")

            col_tws1, col_tws2 = st.columns(2)
            with col_tws1:
                tws_time_reference = st.selectbox(
                    "Referensi waktu TWS",
                    ["detected_fault_cursor", "cfg_trigger_time"],
                    format_func=lambda item: {
                        "detected_fault_cursor": "Detected fault cursor + CFG start time",
                        "cfg_trigger_time": "CFG trigger timestamp relay",
                    }[item],
                    key="optional_tws_time_reference",
                    help=(
                        "Detected cursor memakai waktu fault hasil deteksi aplikasi ditambah CFG start time. "
                        "CFG trigger memakai timestamp trigger bawaan COMTRADE."
                    ),
                )

            with col_tws2:
                tws_velocity_factor = st.number_input(
                    "Propagation Velocity Factor (x c)",
                    value=0.980,
                    min_value=0.500,
                    max_value=1.000,
                    step=0.001,
                    format="%.5f",
                    key="optional_tws_velocity_factor",
                    help="Overhead line umumnya sekitar 0.95-0.99c; kabel biasanya lebih rendah.",
                )

            if st.button("Calculate Optional TWS / Time-Based Location", key="calculate_optional_tws"):
                if local_tws_fault_window is None or remote_tws_fault_window is None:
                    st.session_state.pop("time_based_fault_location_result", None)
                    st.warning("Fault window local/remote belum lengkap. Jalankan deteksi fault local dan remote dahulu.")
                else:
                    local_tws_time = get_absolute_event_time(
                        local_metadata,
                        local_tws_fault_window["fault_time"],
                        tws_time_reference,
                    )
                    remote_tws_time = get_absolute_event_time(
                        remote_metadata,
                        remote_tws_fault_window["fault_time"],
                        tws_time_reference,
                    )

                    if local_tws_time is None or remote_tws_time is None:
                        st.session_state.pop("time_based_fault_location_result", None)
                        st.warning(
                            "Timestamp absolut local/remote belum bisa dibaca. Pastikan CFG memiliki start/trigger "
                            "timestamp yang valid atau gunakan rekaman export COMTRADE yang menyimpan timestamp lengkap."
                        )
                    else:
                        tws_result = calculate_time_based_fault_location(
                            local_tws_time,
                            remote_tws_time,
                            line_param["length_km"],
                            tws_velocity_factor,
                        )
                        st.session_state["time_based_fault_location_result"] = tws_result

            tws_result = st.session_state.get("time_based_fault_location_result")
            if tws_result:
                col_twsr1, col_twsr2, col_twsr3, col_twsr4 = st.columns(4)
                col_twsr1.metric("TWS from Local", f'{tws_result["distance_from_local_km"]:.3f} km')
                col_twsr2.metric("TWS from Remote", f'{tws_result["distance_from_remote_km"]:.3f} km')
                col_twsr3.metric("TWS Position", f'{tws_result["distance_from_local_percent"]:.2f} %')
                col_twsr4.metric("Delta t Local-Remote", f'{tws_result["delta_t_s"] * 1e6:.3f} us')

                tws_detail_df = pd.DataFrame(
                    [
                        {"Parameter": "Local absolute time", "Value": str(tws_result["local_time"])},
                        {"Parameter": "Remote absolute time", "Value": str(tws_result["remote_time"])},
                        {"Parameter": "Delta t s", "Value": tws_result["delta_t_s"]},
                        {"Parameter": "Velocity factor", "Value": tws_result["velocity_factor"]},
                        {"Parameter": "Propagation velocity km/s", "Value": tws_result["propagation_velocity_km_s"]},
                        {"Parameter": "One-end travel time s", "Value": tws_result["one_end_travel_time_s"]},
                        {"Parameter": "Distance from local km", "Value": tws_result["distance_from_local_km"]},
                        {"Parameter": "Distance from remote km", "Value": tws_result["distance_from_remote_km"]},
                    ]
                )

                st.dataframe(
                    tws_detail_df.style.format(
                        {"Value": lambda x: f"{x:.9f}" if isinstance(x, (int, float)) else x}
                    ),
                    use_container_width=True,
                )

                if tws_result["warnings"]:
                    for warning in tws_result["warnings"]:
                        st.warning(warning)
                else:
                    st.success("Hasil time-based berada di dalam panjang saluran dan selisih waktu masih realistis.")
