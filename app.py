import streamlit as st
import tempfile
import math
import cmath
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from comtrade_reader import read_comtrade
from signal_assignment import apply_signal_assignment
from fault_detection import (
    detect_fault_inception,
    build_fault_window,
    estimate_sampling_rate,
)
from phasor import (
    calculate_all_phasors,
    build_phasor_dataframe,
    calculate_sequence_components,
    add_sequence_components_to_phasor_dict,
)
from fault_type import (
    detect_fault_type,
    build_fault_type_metrics_dataframe,
)
from line_parameter import (
    normalize_line_parameter,
    build_line_parameter_dataframe,
)
from high_resistance import (
    detect_high_resistance_fault,
    build_high_resistance_dataframe,
)
from two_ended import (
    calculate_positive_sequence_two_ended,
    evaluate_two_ended_quality,
    build_two_ended_result_dataframe,
    choose_best_remote_current_direction,
    choose_best_two_ended_adaptation,
)
from auto_assignment import (
    detect_voltage_current_channels,
    detect_recorded_side,
    get_auto_transformer_data,
    build_auto_assignment_summary,
    detect_three_phase_channel_sets,
    build_channel_set_summary_dataframe,
)
from conductor_impedance_importer import (
    read_conductor_impedance_excel,
    read_conductor_impedance_database,
    detect_impedance_columns,
    extract_impedance_from_row,
    build_row_label,
)
from single_ended import (
    calculate_single_ended_fault_location,
    build_single_ended_result_dataframe,
)


def make_streamlit_safe_columns(df):
    """
    Membuat nama kolom DataFrame menjadi unik agar aman ditampilkan di Streamlit.
    """

    seen = {}
    new_columns = []

    for col in df.columns:
        col = str(col).strip()

        if col == "":
            col = "Unnamed"

        if col not in seen:
            seen[col] = 1
            new_columns.append(col)
        else:
            seen[col] += 1
            new_columns.append(f"{col}_{seen[col]}")

    safe_df = df.copy()
    safe_df.columns = new_columns

    return safe_df


def invert_current_phasors(phasors):
    inverted = {}

    for name, value in phasors.items():
        phasor_value = value["complex"]

        if name in ["Ia", "Ib", "Ic", "IE", "I0", "I1", "I2"]:
            phasor_value = -phasor_value

        inverted[name] = {
            "complex": phasor_value,
            "real": phasor_value.real,
            "imag": phasor_value.imag,
            "magnitude": abs(phasor_value),
            "angle_deg": math.degrees(cmath.phase(phasor_value)),
        }

    return inverted


def install_print_friendly_tables():
    """
    Streamlit dataframe memakai grid interaktif yang tidak ramah print.
    Wrapper ini mempertahankan tampilan interaktif di layar, lalu menambahkan
    tabel HTML khusus cetak dengan lebar kolom adaptif dan teks wrap.
    """

    if getattr(st, "_print_tables_installed", False):
        return

    original_dataframe = st.dataframe

    def _table_source(data):
        if type(data).__name__ == "Styler" and hasattr(data, "data"):
            return data.data, data.to_html()

        if isinstance(data, pd.DataFrame):
            return data, data.to_html(index=False, escape=True)

        try:
            df = pd.DataFrame(data)
            return df, df.to_html(index=False, escape=True)
        except Exception:
            return None, None

    def printable_dataframe(data=None, *args, **kwargs):
        result = original_dataframe(data, *args, **kwargs)
        source_df, table_html = _table_source(data)

        if source_df is not None and table_html:
            column_count = max(1, len(source_df.columns))
            row_count = len(source_df.index)
            density_class = "print-table-wide" if column_count >= 8 else "print-table-normal"
            if column_count >= 14:
                density_class = "print-table-ultrawide"

            st.markdown(
                f"""
                <div class="print-table-wrapper {density_class}"
                     data-print-columns="{column_count}"
                     data-print-rows="{row_count}">
                    {table_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

        return result

    st.dataframe = printable_dataframe
    st._print_tables_installed = True


st.set_page_config(
    page_title="Transmission Fault Locator",
    layout="wide"
)

st.markdown(
    """
    <style>
    .print-table-wrapper {
        display: none;
    }

    @media print {
        @page {
            size: A4 landscape;
            margin: 10mm;
        }

        html,
        body,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        .block-container {
            width: 100% !important;
            max-width: none !important;
            overflow: visible !important;
        }

        [data-testid="stSidebar"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"],
        [data-testid="stFileUploader"],
        [data-testid="stDataFrame"],
        .stDataFrame,
        div[data-testid="stElementToolbar"] {
            display: none !important;
        }

        .print-table-wrapper {
            display: block !important;
            width: 100% !important;
            max-width: 100% !important;
            overflow: visible !important;
            break-inside: auto;
            page-break-inside: auto;
            margin: 4mm 0 6mm 0;
        }

        .print-table-wrapper table {
            width: 100% !important;
            max-width: 100% !important;
            border-collapse: collapse !important;
            table-layout: fixed !important;
            font-size: 8.5pt !important;
            line-height: 1.25 !important;
            color: #111 !important;
            background: #fff !important;
        }

        .print-table-wrapper.print-table-wide table {
            font-size: 7.2pt !important;
        }

        .print-table-wrapper.print-table-ultrawide table {
            font-size: 6.2pt !important;
        }

        .print-table-wrapper thead {
            display: table-header-group !important;
        }

        .print-table-wrapper tr {
            break-inside: avoid;
            page-break-inside: avoid;
        }

        .print-table-wrapper th,
        .print-table-wrapper td {
            border: 0.35pt solid #777 !important;
            padding: 2.2pt 3pt !important;
            vertical-align: top !important;
            white-space: normal !important;
            overflow-wrap: anywhere !important;
            word-break: break-word !important;
            hyphens: auto !important;
            max-width: none !important;
            min-width: 0 !important;
        }

        .print-table-wrapper th {
            font-weight: 700 !important;
            background: #f0f2f6 !important;
        }

        .print-table-wrapper tbody tr:nth-child(even) td {
            background: #fafafa !important;
        }

        .print-table-wrapper .row_heading,
        .print-table-wrapper .blank {
            width: 8mm !important;
        }

        .element-container,
        .stMarkdown,
        [data-testid="column"] {
            overflow: visible !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

install_print_friendly_tables()

st.title("Transmission Fault Locator")

st.sidebar.header("Upload File COMTRADE")

cfg_file = st.sidebar.file_uploader("Upload file .cfg", type=["cfg"])
dat_file = st.sidebar.file_uploader("Upload file .dat", type=["dat"])

if cfg_file is None or dat_file is None:
    st.info("Silakan upload pasangan file .cfg dan .dat terlebih dahulu.")
    st.stop()

with tempfile.NamedTemporaryFile(delete=False, suffix=".cfg") as temp_cfg:
    temp_cfg.write(cfg_file.read())
    cfg_path = temp_cfg.name

with tempfile.NamedTemporaryFile(delete=False, suffix=".dat") as temp_dat:
    temp_dat.write(dat_file.read())
    dat_path = temp_dat.name

try:
    df, metadata = read_comtrade(cfg_path, dat_path)

    auto_assignment = detect_voltage_current_channels(df, metadata)
    auto_transformer_data = get_auto_transformer_data(metadata)
    auto_recorded_side = detect_recorded_side(metadata)

    st.session_state["auto_assignment"] = auto_assignment
    st.session_state["auto_transformer_data"] = auto_transformer_data
    st.session_state["auto_recorded_side"] = auto_recorded_side

    channel_sets = detect_three_phase_channel_sets(df, metadata)
    st.session_state["channel_sets"] = channel_sets

except Exception as e:
    st.error("File COMTRADE gagal dibaca.")
    st.exception(e)
    st.stop()


st.success("File COMTRADE berhasil dibaca.")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(
    [
        "1. Record Info",
        "2. Signal Assignment",
        "3. Waveform Assigned",
        "4. Fault Detection & Cursor",
        "5. Phasor Calculation",
        "6. Fault Type Detection",
        "7. Line Parameter",
        "8. High Resistance Detection",
        "9. Single-Ended Fault Locator",
        "10. Two-Ended Fault Locator",
    ]
)


with tab1:
    st.subheader("Informasi Rekaman")

    col1, col2, col3 = st.columns(3)

    col1.metric("Station Name", metadata["station_name"] or "-")
    col2.metric("Frequency", f'{metadata["frequency"] or "-"} Hz')
    col3.metric("Total Samples", metadata["total_samples"])

    st.write("Analog Channels:")
    st.write(metadata["analog_channels"])

    st.subheader("Auto-Read Metadata dari CFG")

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)

    col_m1.metric("CFG Start Time", str(metadata.get("cfg_start_time") or "-"))
    col_m2.metric("CFG Trigger Time", str(metadata.get("cfg_trigger_time") or "-"))
    col_m3.metric("VT Ratio from CFG", str(metadata.get("vt_ratio_from_cfg") or "-"))
    col_m4.metric("CT Ratio from CFG", str(metadata.get("ct_ratio_from_cfg") or "-"))

    st.subheader("Auto Signal Assignment Preview")

    auto_summary_df = build_auto_assignment_summary(
        st.session_state["auto_assignment"],
        st.session_state["auto_transformer_data"],
        metadata,
    )

    st.dataframe(auto_summary_df, use_container_width=True)

    st.subheader("Detected Three-Phase Channel Sets")

    channel_sets = st.session_state.get("channel_sets", {})
    channel_set_df = build_channel_set_summary_dataframe(channel_sets)

    if channel_set_df.empty:
        st.warning("Aplikasi belum menemukan kandidat set channel 3 fasa.")
    else:
        st.dataframe(channel_set_df, use_container_width=True)

    with st.expander("Detail Analog Metadata dari .cfg"):
        analog_meta_df = pd.DataFrame(metadata.get("analog_metadata", []))
        st.dataframe(analog_meta_df, use_container_width=True)

    st.subheader("Preview Data Original")
    st.dataframe(df.head(20), use_container_width=True)


with tab2:
    st.subheader("Signal Assignment")

    st.write(
        "Pilih channel asli dari file COMTRADE, lalu mapping ke variabel standar aplikasi."
    )

    channel_options = [col for col in df.columns if col != "time"]
    ground_options = ["None"] + channel_options

    channel_sets = st.session_state.get("channel_sets", {})
    voltage_sets = channel_sets.get("voltage_sets", [])
    current_sets = channel_sets.get("current_sets", [])
    ground_candidates = channel_sets.get("ground_candidates", [])

    st.markdown("### Auto Detected Channel Sets")

    if voltage_sets:
        voltage_set_labels = [item["label"] for item in voltage_sets]
    else:
        voltage_set_labels = ["Manual Selection"]

    if current_sets:
        current_set_labels = [item["label"] for item in current_sets]
    else:
        current_set_labels = ["Manual Selection"]

    col_set1, col_set2 = st.columns(2)

    with col_set1:
        selected_voltage_set_label = st.selectbox(
            "Pilih Set Tegangan 3 Fasa",
            ["Manual Selection"] + voltage_set_labels,
            index=1 if voltage_sets else 0,
            key="selected_voltage_set_label",
        )

    with col_set2:
        selected_current_set_label = st.selectbox(
            "Pilih Set Arus 3 Fasa",
            ["Manual Selection"] + current_set_labels,
            index=1 if current_sets else 0,
            key="selected_current_set_label",
        )


    def get_selected_set_by_label(channel_set_list, selected_label):
        for item in channel_set_list:
            if item["label"] == selected_label:
                return item
        return None


    selected_voltage_set = get_selected_set_by_label(
        voltage_sets,
        selected_voltage_set_label,
    )

    selected_current_set = get_selected_set_by_label(
        current_sets,
        selected_current_set_label,
    )

    if selected_voltage_set:
        default_va = selected_voltage_set["A"]
        default_vb = selected_voltage_set["B"]
        default_vc = selected_voltage_set["C"]
    else:
        default_va = auto_assignment.get("Va")
        default_vb = auto_assignment.get("Vb")
        default_vc = auto_assignment.get("Vc")

    if selected_current_set:
        default_ia = selected_current_set["A"]
        default_ib = selected_current_set["B"]
        default_ic = selected_current_set["C"]
    else:
        default_ia = auto_assignment.get("Ia")
        default_ib = auto_assignment.get("Ib")
        default_ic = auto_assignment.get("Ic")

    if ground_candidates:
        default_ie = ground_candidates[0]["channel"]
    else:
        default_ie = auto_assignment.get("IE")

    auto_assignment = st.session_state.get("auto_assignment", {})
    auto_transformer_data = st.session_state.get("auto_transformer_data", {})
    auto_recorded_side = st.session_state.get("auto_recorded_side", "secondary")


    def get_channel_index(channel_name, options, default_index=0):
        if channel_name in options:
            return options.index(channel_name)
        return default_index


    def get_ground_index(channel_name, options):
        if channel_name in options:
            return options.index(channel_name)
        return 0

    st.markdown("### Voltage Channel Assignment")

    col_v1, col_v2, col_v3 = st.columns(3)

    with col_v1:
        va_channel = st.selectbox(
            "Va / VL1",
            channel_options,
            index=get_channel_index(default_va, channel_options, 0),
        )

    with col_v2:
        vb_channel = st.selectbox(
            "Vb / VL2",
            channel_options,
            index=get_channel_index(
                default_vb,
                channel_options,
                1 if len(channel_options) > 1 else 0,
            ),
        )

    with col_v3:
        vc_channel = st.selectbox(
            "Vc / VL3",
            channel_options,
            index=get_channel_index(
                default_vc,
                channel_options,
                2 if len(channel_options) > 2 else 0,
            ),
        )

    st.markdown("### Current Channel Assignment")

    col_i1, col_i2, col_i3 = st.columns(3)

    default_ia_index = 3 if len(channel_options) > 3 else 0
    default_ib_index = 4 if len(channel_options) > 4 else 0
    default_ic_index = 5 if len(channel_options) > 5 else 0

    with col_i1:
        ia_channel = st.selectbox(
            "Ia / IL1",
            channel_options,
            index=get_channel_index(default_ia, channel_options, default_ia_index),
        )

    with col_i2:
        ib_channel = st.selectbox(
            "Ib / IL2",
            channel_options,
            index=get_channel_index(default_ib, channel_options, default_ib_index),
        )

    with col_i3:
        ic_channel = st.selectbox(
            "Ic / IL3",
            channel_options,
            index=get_channel_index(default_ic, channel_options, default_ic_index),
        )

    st.markdown("### Ground Current Assignment")

    ie_channel = st.selectbox(
        "IE / IN / 3I0 jika tersedia",
        ground_options,
        index=0
    )

    selected_assignment_channels = [
        va_channel,
        vb_channel,
        vc_channel,
        ia_channel,
        ib_channel,
        ic_channel,
    ]

    duplicate_channels = [
        ch for ch in selected_assignment_channels
        if selected_assignment_channels.count(ch) > 1
    ]

    if duplicate_channels:
        st.error(
            "Ada channel yang dipilih lebih dari satu kali pada Va/Vb/Vc/Ia/Ib/Ic: "
            + ", ".join(sorted(set(duplicate_channels)))
            + ". Periksa kembali Signal Assignment."
        )
    else:
        st.success("Signal Assignment Va/Vb/Vc/Ia/Ib/Ic tidak memiliki duplikasi channel.")

    st.info(
        "Jika IE/IN tidak dipilih, aplikasi akan menghitung residual current: "
        "IE = Ia + Ib + Ic, lalu I0 = IE / 3."
    )

    st.markdown("### Transformer Data")

    recorded_side_options = ["secondary", "primary"]
    recorded_side_default_index = recorded_side_options.index(auto_recorded_side) if auto_recorded_side in recorded_side_options else 0

    recorded_side = st.radio(
        "Nilai pada file COMTRADE direkam sebagai:",
        recorded_side_options,
        index=recorded_side_default_index,
        horizontal=True,
    )

    col_ct1, col_ct2, col_vt1, col_vt2 = st.columns(4)

    with col_ct1:
        ct_primary = st.number_input(
            "CT Primary (A)",
            value=float(auto_transformer_data.get("ct_primary", 800.0)),
        )

    with col_ct2:
        ct_secondary = st.number_input(
            "CT Secondary (A)",
            value=float(auto_transformer_data.get("ct_secondary", 1.0)),
        )

    with col_vt1:
        vt_primary = st.number_input(
            "VT/CVT Primary (V)",
            value=float(auto_transformer_data.get("vt_primary", 150000.0)),
        )

    with col_vt2:
        vt_secondary = st.number_input(
            "VT/CVT Secondary (V)",
            value=float(auto_transformer_data.get("vt_secondary", 100.0)),
        )
    
    st.caption(
        f"Auto ratio source: CT = {auto_transformer_data.get('ct_ratio_source', '-')}, "
        f"VT = {auto_transformer_data.get('vt_ratio_source', '-')}. "
        "Tetap validasi manual karena tidak semua file CFG menyimpan primary/secondary dengan benar."
    )

    assigned_df = apply_signal_assignment(
        df=df,
        va_channel=va_channel,
        vb_channel=vb_channel,
        vc_channel=vc_channel,
        ia_channel=ia_channel,
        ib_channel=ib_channel,
        ic_channel=ic_channel,
        ie_channel=ie_channel,
        recorded_side=recorded_side,
        ct_primary=ct_primary,
        ct_secondary=ct_secondary,
        vt_primary=vt_primary,
        vt_secondary=vt_secondary,
    )

    st.session_state["assigned_df"] = assigned_df

    st.success("Signal assignment berhasil dibuat.")

    st.subheader("Preview Data Setelah Mapping")
    st.dataframe(assigned_df.head(20), use_container_width=True)

    st.subheader("Ringkasan Mapping")

    mapping_summary = {
        "Voltage Set": selected_voltage_set_label,
        "Current Set": selected_current_set_label,
        "Va": va_channel,
        "Vb": vb_channel,
        "Vc": vc_channel,
        "Ia": ia_channel,
        "Ib": ib_channel,
        "Ic": ic_channel,
        "IE": ie_channel if ie_channel != "None" else "Calculated: Ia + Ib + Ic",
        "Recorded Side": recorded_side,
        "CT Ratio": f"{ct_primary}/{ct_secondary}",
        "VT Ratio": f"{vt_primary}/{vt_secondary}",
    }

    st.json(mapping_summary)


with tab3:
    st.subheader("Waveform Hasil Signal Assignment")

    if "assigned_df" not in st.session_state:
        st.warning("Silakan lakukan Signal Assignment terlebih dahulu.")
        st.stop()

    assigned_df = st.session_state["assigned_df"]

    signal_groups = {
        "Tegangan 3 Fasa": ["Va", "Vb", "Vc"],
        "Arus 3 Fasa": ["Ia", "Ib", "Ic"],
        "Ground Current": ["IE", "I0"],
        "Semua": ["Va", "Vb", "Vc", "Ia", "Ib", "Ic", "IE", "I0"],
    }

    selected_group = st.selectbox(
        "Pilih kelompok sinyal",
        list(signal_groups.keys())
    )

    selected_channels = signal_groups[selected_group]

    fig = px.line(
        assigned_df,
        x="time",
        y=selected_channels,
        title=f"Waveform {selected_group}"
    )

    fig.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Primary Magnitude",
        legend_title="Signal"
    )

    st.plotly_chart(fig, use_container_width=True)


with tab4:
    st.subheader("Fault Detection & Cursor Window")

    st.markdown("### Trigger Metadata")

    col_t1, col_t2 = st.columns(2)

    col_t1.metric("CFG Start Time", str(metadata.get("cfg_start_time") or "-"))
    col_t2.metric("CFG Trigger Time", str(metadata.get("cfg_trigger_time") or "-"))

    st.caption(
        "Trigger timestamp dibaca dari metadata CFG jika tersedia. "
        "Fault inception tetap dideteksi dari waveform DAT untuk menentukan window DFT."
    )

    if "assigned_df" not in st.session_state:
        st.warning("Silakan lakukan Signal Assignment terlebih dahulu.")
        st.stop()

    assigned_df = st.session_state["assigned_df"]

    st.markdown("### Parameter Deteksi Gangguan")

    col_fd1, col_fd2, col_fd3 = st.columns(3)

    with col_fd1:
        frequency = st.number_input(
            "Frekuensi Sistem (Hz)",
            value=float(metadata["frequency"]) if metadata["frequency"] else 50.0,
            min_value=40.0,
            max_value=70.0,
            step=0.1
        )

    with col_fd2:
        current_threshold_multiplier = st.number_input(
            "Multiplier Kenaikan Arus",
            value=2.0,
            min_value=1.1,
            max_value=10.0,
            step=0.1
        )

    with col_fd3:
        voltage_drop_threshold = st.number_input(
            "Batas Drop Tegangan",
            value=0.85,
            min_value=0.1,
            max_value=1.0,
            step=0.01
        )

    col_w1, col_w2 = st.columns(2)

    with col_w1:
        pre_fault_cycles = st.number_input(
            "Pre-fault Window (cycles)",
            value=2,
            min_value=1,
            max_value=10,
            step=1
        )

    with col_w2:
        post_fault_cycles = st.number_input(
            "Post-fault Window (cycles)",
            value=4,
            min_value=1,
            max_value=20,
            step=1
        )

    with st.expander("Advanced Fault Bar Tuning"):
        use_advanced_fault_detection = st.checkbox(
            "Use Advanced Fault Detection",
            value=False,
            help="Aktifkan hanya jika fault bar otomatis kurang presisi pada record lokal.",
        )

        fault_detection_method = st.selectbox(
            "Fault Detection Method",
            ["legacy_rms", "hybrid_superimposed"],
            index=1,
            disabled=not use_advanced_fault_detection,
            help="hybrid_superimposed memakai energi perubahan satu siklus lalu divalidasi RMS.",
        )

        col_adv1, col_adv2, col_adv3, col_adv4 = st.columns(4)

        with col_adv1:
            adaptive_threshold_sigma = st.number_input(
                "Adaptive Threshold Sigma",
                value=6.0,
                min_value=2.0,
                max_value=20.0,
                step=0.5,
                help="Threshold adaptif terhadap noise pre-fault. Lebih kecil = lebih sensitif.",
                disabled=not use_advanced_fault_detection,
            )

        with col_adv2:
            superimposed_threshold_sigma = st.number_input(
                "Superimposed Threshold Sigma",
                value=8.0,
                min_value=2.0,
                max_value=30.0,
                step=0.5,
                disabled=(
                    not use_advanced_fault_detection
                    or fault_detection_method != "hybrid_superimposed"
                ),
                help="Threshold energi superimposed terhadap baseline pre-fault.",
            )

        with col_adv3:
            consecutive_samples_input = st.number_input(
                "Consecutive Samples",
                value=0,
                min_value=0,
                max_value=200,
                step=1,
                help="0 = otomatis sekitar 0.1 siklus. Nilai lebih besar menolak spike sesaat.",
                disabled=not use_advanced_fault_detection,
            )

        with col_adv4:
            refine_fault_bar = st.checkbox(
                "Refine Fault Bar",
                value=True,
                help="Backtrack dari kandidat RMS ke perubahan instantaneous awal.",
                disabled=not use_advanced_fault_detection,
            )

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
    )

    st.session_state["fault_detection"] = detection

    if detection["detected"]:
        st.success("Awal gangguan berhasil terdeteksi otomatis.")

        fault_window = build_fault_window(
            assigned_df,
            fault_index=detection["fault_index"],
            samples_per_cycle=detection["samples_per_cycle"],
            pre_fault_cycles=int(pre_fault_cycles),
            post_fault_cycles=int(post_fault_cycles),
        )

        st.session_state["fault_window"] = fault_window

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
            step=(max_time - min_time) / 1000
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

        st.session_state["fault_window"] = fault_window

    st.markdown("### Validasi Window Analisis")

    if "fault_window" in st.session_state:
        fault_window = st.session_state["fault_window"]

        st.json(fault_window)

        st.info(
            "Left Cursor dan Right Cursor digunakan sebagai range analisis. "
            "DFT Cursor digunakan pada Step 4 untuk mengambil fasor 1 siklus "
            "setelah awal gangguan."
        )

        display_df = assigned_df.copy()

        selected_plot = st.multiselect(
            "Pilih sinyal untuk validasi fault window",
            ["Va", "Vb", "Vc", "Ia", "Ib", "Ic", "IE"],
            default=["Ia", "Ib", "Ic"]
        )

        fig = px.line(
            display_df,
            x="time",
            y=selected_plot,
            title="Fault Detection dan Cursor Window"
        )

        fig.add_vline(
            x=fault_window["left_time"],
            line_dash="dash",
            annotation_text="Left Cursor",
            annotation_position="top left"
        )

        fig.add_vline(
            x=fault_window["fault_time"],
            line_dash="solid",
            annotation_text="Fault",
            annotation_position="top"
        )

        fig.add_vline(
            x=fault_window["dft_time"],
            line_dash="dot",
            annotation_text="DFT Cursor",
            annotation_position="top"
        )

        fig.add_vline(
            x=fault_window["right_time"],
            line_dash="dash",
            annotation_text="Right Cursor",
            annotation_position="top right"
        )

        fig.update_layout(
            xaxis_title="Time (s)",
            yaxis_title="Magnitude Primary",
            legend_title="Signal"
        )

        st.plotly_chart(fig, use_container_width=True)


with tab5:
    st.subheader("Phasor Calculation")

    if "assigned_df" not in st.session_state:
        st.warning("Silakan lakukan Signal Assignment terlebih dahulu.")
        st.stop()

    if "fault_window" not in st.session_state:
        st.warning("Silakan lakukan Fault Detection & Cursor terlebih dahulu.")
        st.stop()

    assigned_df = st.session_state["assigned_df"]
    fault_window = st.session_state["fault_window"]
    detection = st.session_state["fault_detection"]

    st.markdown("### DFT Window Setting")

    dft_index_default = fault_window["dft_index"]
    samples_per_cycle = detection["samples_per_cycle"]

    col_p1, col_p2, col_p3 = st.columns(3)

    with col_p1:
        st.metric("Samples per Cycle", samples_per_cycle)

    with col_p2:
        st.metric("DFT Cursor Time", f'{fault_window["dft_time"]:.6f} s')

    with col_p3:
        dft_window_start_index = dft_index_default - samples_per_cycle
        dft_window_start_time = assigned_df["time"].iloc[dft_window_start_index]
        st.metric("DFT Window Start", f"{dft_window_start_time:.6f} s")

    st.info(
        "Fasor dihitung memakai full-cycle DFT. Window DFT berada di sebelah kiri "
        "DFT Cursor dengan panjang 1 siklus."
    )

    use_manual_dft_cursor = st.checkbox("Gunakan DFT Cursor manual")

    if use_manual_dft_cursor:
        min_index = samples_per_cycle
        max_index = len(assigned_df) - 1

        manual_dft_time = st.slider(
            "Pilih DFT Cursor manual (s)",
            min_value=float(assigned_df["time"].iloc[min_index]),
            max_value=float(assigned_df["time"].iloc[max_index]),
            value=float(fault_window["dft_time"]),
            step=float((assigned_df["time"].max() - assigned_df["time"].min()) / 1000),
        )

        dft_index = int((assigned_df["time"] - manual_dft_time).abs().idxmin())
    else:
        dft_index = dft_index_default

    try:
        phasors = calculate_all_phasors(
            df=assigned_df,
            cursor_index=dft_index,
            samples_per_cycle=samples_per_cycle,
        )

        phasor_df = build_phasor_dataframe(phasors)

        st.session_state["phasors"] = phasors
        st.session_state["phasor_df"] = phasor_df
        st.session_state["dft_index"] = dft_index

        st.success("Perhitungan fasor berhasil.")

        st.markdown("### Tabel Fasor RMS Fundamental")

        st.dataframe(
            phasor_df.style.format(
                {
                    "Magnitude RMS": "{:.4f}",
                    "Angle Deg": "{:.2f}",
                    "Real": "{:.4f}",
                    "Imag": "{:.4f}",
                }
            ),
            use_container_width=True,
        )

        sequence, sequence_df = calculate_sequence_components(phasors)

        st.session_state["sequence_components"] = sequence
        st.session_state["sequence_df"] = sequence_df

        sequence, sequence_df = calculate_sequence_components(phasors)

        st.session_state["sequence_components"] = sequence
        st.session_state["sequence_df"] = sequence_df

        # Tambahkan komponen simetris ke dictionary phasors
        # agar bisa dipakai oleh Two-Ended Fault Locator
        phasors = add_sequence_components_to_phasor_dict(phasors)
        st.session_state["phasors"] = phasors

        st.markdown("### Komponen Simetris")

        st.dataframe(
            sequence_df.style.format(
                {
                    "Magnitude RMS": "{:.4f}",
                    "Angle Deg": "{:.2f}",
                    "Real": "{:.4f}",
                    "Imag": "{:.4f}",
                }
            ),
            use_container_width=True,
        )

        st.markdown("### Validasi Window DFT pada Waveform")

        dft_window_start_index = dft_index - samples_per_cycle
        dft_window_end_index = dft_index

        dft_window_start_time = float(assigned_df["time"].iloc[dft_window_start_index])
        dft_window_end_time = float(assigned_df["time"].iloc[dft_window_end_index])

        selected_dft_plot = st.multiselect(
            "Pilih sinyal untuk validasi window DFT",
            ["Va", "Vb", "Vc", "Ia", "Ib", "Ic"],
            default=["Ia", "Ib", "Ic"],
        )

        fig_dft = px.line(
            assigned_df,
            x="time",
            y=selected_dft_plot,
            title="DFT Window pada Waveform",
        )

        fig_dft.add_vrect(
            x0=dft_window_start_time,
            x1=dft_window_end_time,
            opacity=0.2,
            line_width=0,
            annotation_text="DFT Window 1 Cycle",
            annotation_position="top left",
        )

        fig_dft.add_vline(
            x=float(assigned_df["time"].iloc[dft_index]),
            line_dash="dot",
            annotation_text="DFT Cursor",
            annotation_position="top",
        )

        fig_dft.add_vline(
            x=fault_window["fault_time"],
            line_dash="solid",
            annotation_text="Fault",
            annotation_position="top",
        )

        fig_dft.update_layout(
            xaxis_title="Time (s)",
            yaxis_title="Magnitude Primary",
            legend_title="Signal",
        )

        st.plotly_chart(fig_dft, use_container_width=True)

        st.markdown("### Phasor Diagram")

        phasor_group = st.radio(
            "Pilih diagram fasor",
            ["Voltage", "Current"],
            horizontal=True,
        )

        if phasor_group == "Voltage":
            signals_to_plot = ["Va", "Vb", "Vc"]
        else:
            signals_to_plot = ["Ia", "Ib", "Ic"]

        fig_phasor = go.Figure()

        for signal_name in signals_to_plot:
            z = phasors[signal_name]["complex"]

            fig_phasor.add_trace(
                go.Scatter(
                    x=[0, z.real],
                    y=[0, z.imag],
                    mode="lines+markers+text",
                    text=["", signal_name],
                    textposition="top center",
                    name=signal_name,
                )
            )

        max_abs = max(
            abs(phasors[s]["real"]) + abs(phasors[s]["imag"])
            for s in signals_to_plot
        )

        if max_abs == 0:
            max_abs = 1

        fig_phasor.update_layout(
            title=f"Phasor Diagram - {phasor_group}",
            xaxis_title="Real",
            yaxis_title="Imaginary",
            xaxis=dict(range=[-max_abs, max_abs], zeroline=True),
            yaxis=dict(range=[-max_abs, max_abs], zeroline=True, scaleanchor="x", scaleratio=1),
            showlegend=True,
        )

        st.plotly_chart(fig_phasor, use_container_width=True)

    except Exception as e:
        st.error("Perhitungan fasor gagal.")
        st.exception(e)


with tab6:
    st.subheader("Fault Type Detection")

    if "phasors" not in st.session_state:
        st.warning("Silakan lakukan Phasor Calculation terlebih dahulu.")
        st.stop()

    phasors = st.session_state["phasors"]

    st.markdown("### Parameter Deteksi Jenis Gangguan")

    col_ft1, col_ft2, col_ft3 = st.columns(3)

    with col_ft1:
        voltage_drop_threshold_ft = st.number_input(
            "Voltage Drop Threshold",
            value=0.80,
            min_value=0.10,
            max_value=1.00,
            step=0.01,
            help="Fasa dianggap drop jika Vphase <= threshold × Vmax."
        )

    with col_ft2:
        current_rise_threshold_ft = st.number_input(
            "Current Rise Threshold",
            value=1.50,
            min_value=1.05,
            max_value=10.00,
            step=0.05,
            help="Fasa dianggap faulted jika Iphase >= threshold × Imin."
        )

    with col_ft3:
        ground_current_threshold_ft = st.number_input(
            "Ground Current Threshold",
            value=0.20,
            min_value=0.01,
            max_value=1.00,
            step=0.01,
            help="Ground fault jika IE/Imax atau I0/Iavg melebihi threshold."
        )

    fault_type_result = detect_fault_type(
        phasors=phasors,
        voltage_drop_threshold=voltage_drop_threshold_ft,
        current_rise_threshold=current_rise_threshold_ft,
        ground_current_threshold=ground_current_threshold_ft,
    )

    st.session_state["fault_type_result"] = fault_type_result

    st.markdown("### Hasil Deteksi")

    col_res1, col_res2, col_res3 = st.columns(3)

    col_res1.metric("Fault Type", fault_type_result["fault_type"])
    col_res2.metric(
        "Ground Involved",
        "Yes" if fault_type_result["ground_involved"] else "No"
    )
    col_res3.metric(
        "Confidence",
        f'{fault_type_result["confidence"]}/10'
    )

    st.write(
        "Faulted Phases:",
        ", ".join(fault_type_result["faulted_phases"])
        if fault_type_result["faulted_phases"] else "-"
    )

    st.markdown("### Metrik Deteksi")

    metrics_df = build_fault_type_metrics_dataframe(fault_type_result)

    st.dataframe(metrics_df, use_container_width=True)

    st.markdown("### Grafik Perbandingan Fasor RMS")

    metrics = fault_type_result["metrics"]

    voltage_bar_df = pd.DataFrame(
        {
            "Phase": ["A", "B", "C"],
            "Voltage RMS": [metrics["Va"], metrics["Vb"], metrics["Vc"]],
        }
    )

    current_bar_df = pd.DataFrame(
        {
            "Phase": ["A", "B", "C", "Ground IE"],
            "Current RMS": [
                metrics["Ia"],
                metrics["Ib"],
                metrics["Ic"],
                metrics["IE"],
            ],
        }
    )

    fig_vbar = px.bar(
        voltage_bar_df,
        x="Phase",
        y="Voltage RMS",
        title="Perbandingan Tegangan RMS per Fasa",
        text_auto=".2f",
    )

    st.plotly_chart(fig_vbar, use_container_width=True)

    fig_ibar = px.bar(
        current_bar_df,
        x="Phase",
        y="Current RMS",
        title="Perbandingan Arus RMS per Fasa dan Ground",
        text_auto=".2f",
    )

    st.plotly_chart(fig_ibar, use_container_width=True)

    st.markdown("### Koreksi Manual")

    manual_fault_type = st.selectbox(
        "Jika hasil otomatis kurang tepat, pilih jenis gangguan manual",
        [
            "AUTO",
            "AG", "BG", "CG",
            "AB", "BC", "CA",
            "ABG", "BCG", "CAG",
            "ABC", "ABCG",
            "UNKNOWN",
        ],
        index=0,
    )

    if manual_fault_type != "AUTO":
        fault_type_result["fault_type"] = manual_fault_type
        fault_type_result["manual_override"] = True
        st.session_state["fault_type_result"] = fault_type_result
        st.success(f"Fault type dikoreksi manual menjadi: {manual_fault_type}")
    else:
        fault_type_result["manual_override"] = False
        st.session_state["fault_type_result"] = fault_type_result

    st.info(
        "Fault type ini akan dipakai pada Step 6 untuk memilih rumus loop impedansi. "
        "Untuk gangguan fasa-tanah digunakan loop AG/BG/CG dengan kompensasi tanah. "
        "Untuk gangguan fasa-fasa digunakan loop AB/BC/CA."
    )


with tab7:
        st.subheader("Line Parameter Input & Converter")

        st.write(
            "Masukkan parameter saluran secara manual. "
            "Semua parameter akan dinormalisasi menjadi Z1_per_km, Z0_per_km, K0, Z1_total, dan Z0_total."
        )

        st.markdown("### Sumber Data Parameter")

        line_parameter_source = st.radio(
            "Pilih sumber parameter saluran",
            [
                "Input Manual",
                "Database Excel Line Data",
                "Database Excel Cable Data",
            ],
            horizontal=True,
        )

        excel_impedance_data = None

        if line_parameter_source in ["Database Excel Line Data", "Database Excel Cable Data"]:
            use_cable_database = line_parameter_source == "Database Excel Cable Data"
            database_file_path = (
                "database/cable_data.xlsx"
                if use_cable_database
                else "database/line_data.xlsx"
            )
            database_title = (
                "Database Excel Data Impedansi Konduktor"
                if use_cable_database
                else "Database Excel Data Impedansi Saluran"
            )
            database_preview_title = (
                "Preview Database Cable Impedance"
                if use_cable_database
                else "Preview Database Line Impedance"
            )
            selected_row_label_text = (
                "Pilih jenis konduktor dari cable_data.xlsx"
                if use_cable_database
                else "Pilih baris data saluran / BAY PHT"
            )

            st.markdown(f"### {database_title}")

            st.info(
                "Aplikasi membaca data impedansi dari file lokal: "
                f"`{database_file_path}`, sheet: `line_impedance`."
            )

            if use_cable_database:
                st.warning(
                    "Gunakan opsi ini jika nama line tidak tersedia di line_data.xlsx. "
                    "Aplikasi akan mengambil Z1/Z0 per km dari data konduktor, "
                    "sedangkan nama line dan panjang saluran tetap diisi pada form Basic Line Data."
                )

            try:
                conductor_df = read_conductor_impedance_database(
                    file_path=database_file_path,
                    sheet_name="line_impedance",
                )

                conductor_df = make_streamlit_safe_columns(conductor_df)

                if use_cable_database:
                    detected_for_unique = detect_impedance_columns(conductor_df)
                    unique_subset = [
                        detected_for_unique.get(key)
                        for key in [
                            "conductor_type",
                            "circuit_count",
                            "z1_real",
                            "z1_imag",
                            "z0_real",
                            "z0_imag",
                            "z1_abs",
                            "z1_angle",
                            "z0_abs",
                            "z0_angle",
                        ]
                        if detected_for_unique.get(key)
                    ]

                    if unique_subset:
                        conductor_df = (
                            conductor_df
                            .drop_duplicates(subset=unique_subset)
                            .reset_index(drop=True)
                        )

                st.session_state["line_database_df"] = conductor_df

                st.markdown(f"#### {database_preview_title}")

                st.dataframe(conductor_df, use_container_width=True, height=300)

                detected_columns = detect_impedance_columns(conductor_df)

                st.markdown("#### Kolom yang Terdeteksi Otomatis")
                st.json(detected_columns)

                st.markdown("#### Koreksi Mapping Kolom Jika Perlu")

                all_columns = ["None"] + list(conductor_df.columns)

                def col_index(col_name):
                    if col_name in all_columns:
                        return all_columns.index(col_name)
                    return 0

                col_c1, col_c2, col_c3 = st.columns(3)

                with col_c1:
                    line_name_col = st.selectbox(
                        "Kolom Line Name / BAY PHT",
                        all_columns,
                        index=col_index(detected_columns.get("line_name")),
                        key="db_line_name_col",
                    )

                with col_c2:
                    length_col = st.selectbox(
                        "Kolom Length",
                        all_columns,
                        index=col_index(detected_columns.get("length")),
                        key="db_length_col",
                    )

                with col_c3:
                    bay_pht_col = st.selectbox(
                        "Kolom BAY PHT",
                        all_columns,
                        index=col_index(detected_columns.get("bay_pht")),
                        key="db_bay_pht_col",
                    )

                col_z1a, col_z1b, col_z1c, col_z1d = st.columns(4)

                with col_z1a:
                    z1_real_col = st.selectbox(
                        "Kolom Z1 Real / R1",
                        all_columns,
                        index=col_index(detected_columns.get("z1_real")),
                        key="db_z1_real_col",
                    )

                with col_z1b:
                    z1_imag_col = st.selectbox(
                        "Kolom Z1 Imag / X1",
                        all_columns,
                        index=col_index(detected_columns.get("z1_imag")),
                        key="db_z1_imag_col",
                    )

                with col_z1c:
                    z1_abs_col = st.selectbox(
                        "Kolom Z1 Abs",
                        all_columns,
                        index=col_index(detected_columns.get("z1_abs")),
                        key="db_z1_abs_col",
                    )

                with col_z1d:
                    z1_angle_col = st.selectbox(
                        "Kolom Z1 Angle",
                        all_columns,
                        index=col_index(detected_columns.get("z1_angle")),
                        key="db_z1_angle_col",
                    )

                col_z0a, col_z0b, col_z0c, col_z0d = st.columns(4)

                with col_z0a:
                    z0_real_col = st.selectbox(
                        "Kolom Z0 Real / R0",
                        all_columns,
                        index=col_index(detected_columns.get("z0_real")),
                        key="db_z0_real_col",
                    )

                with col_z0b:
                    z0_imag_col = st.selectbox(
                        "Kolom Z0 Imag / X0",
                        all_columns,
                        index=col_index(detected_columns.get("z0_imag")),
                        key="db_z0_imag_col",
                    )

                with col_z0c:
                    z0_abs_col = st.selectbox(
                        "Kolom Z0 Abs",
                        all_columns,
                        index=col_index(detected_columns.get("z0_abs")),
                        key="db_z0_abs_col",
                    )

                with col_z0d:
                    z0_angle_col = st.selectbox(
                        "Kolom Z0 Angle",
                        all_columns,
                        index=col_index(detected_columns.get("z0_angle")),
                        key="db_z0_angle_col",
                    )

                col_ratio1, col_ratio2, col_ratio3, col_ratio4 = st.columns(4)

                with col_ratio1:
                    ratio_gia_ct_col = st.selectbox(
                        "Kolom Ratio GI A CT",
                        all_columns,
                        index=col_index(detected_columns.get("ratio_gia_ct")),
                        key="db_ratio_gia_ct_col",
                    )

                with col_ratio2:
                    ratio_gia_vt_col = st.selectbox(
                        "Kolom Ratio GI A VT",
                        all_columns,
                        index=col_index(detected_columns.get("ratio_gia_vt")),
                        key="db_ratio_gia_vt_col",
                    )

                with col_ratio3:
                    ratio_gib_ct_col = st.selectbox(
                        "Kolom Ratio GI B CT",
                        all_columns,
                        index=col_index(detected_columns.get("ratio_gib_ct")),
                        key="db_ratio_gib_ct_col",
                    )

                with col_ratio4:
                    ratio_gib_vt_col = st.selectbox(
                        "Kolom Ratio GI B VT",
                        all_columns,
                        index=col_index(detected_columns.get("ratio_gib_vt")),
                        key="db_ratio_gib_vt_col",
                    )

                corrected_columns = {
                    "line_name": None if line_name_col == "None" else line_name_col,
                    "bay_pht": None if bay_pht_col == "None" else bay_pht_col,
                    "length": None if length_col == "None" else length_col,
                    "z1_real": None if z1_real_col == "None" else z1_real_col,
                    "z1_imag": None if z1_imag_col == "None" else z1_imag_col,
                    "z1_abs": None if z1_abs_col == "None" else z1_abs_col,
                    "z1_angle": None if z1_angle_col == "None" else z1_angle_col,
                    "z0_real": None if z0_real_col == "None" else z0_real_col,
                    "z0_imag": None if z0_imag_col == "None" else z0_imag_col,
                    "z0_abs": None if z0_abs_col == "None" else z0_abs_col,
                    "z0_angle": None if z0_angle_col == "None" else z0_angle_col,
                    "ratio_gia_ct": None if ratio_gia_ct_col == "None" else ratio_gia_ct_col,
                    "ratio_gia_vt": None if ratio_gia_vt_col == "None" else ratio_gia_vt_col,
                    "ratio_gib_ct": None if ratio_gib_ct_col == "None" else ratio_gib_ct_col,
                    "ratio_gib_vt": None if ratio_gib_vt_col == "None" else ratio_gib_vt_col,
                    "gia_name": None if use_cable_database else detected_columns.get("gia_name"),
                    "gib_name": None if use_cable_database else detected_columns.get("gib_name"),
                    "conductor_type": detected_columns.get("conductor_type"),
                    "circuit_count": detected_columns.get("circuit_count"),
                }

                if use_cable_database:
                    corrected_columns["line_name"] = None
                    corrected_columns["bay_pht"] = None
                    corrected_columns["length"] = None
                    corrected_columns["ratio_gia_ct"] = None
                    corrected_columns["ratio_gia_vt"] = None
                    corrected_columns["ratio_gib_ct"] = None
                    corrected_columns["ratio_gib_vt"] = None

                row_labels = build_row_label(conductor_df, corrected_columns)

                selected_row_label = st.selectbox(
                    selected_row_label_text,
                    row_labels,
                    key=(
                        "selected_database_cable_row"
                        if use_cable_database
                        else "selected_database_line_row"
                    ),
                )

                selected_row_index = row_labels.index(selected_row_label)
                selected_row = conductor_df.iloc[selected_row_index]

                excel_impedance_data = extract_impedance_from_row(
                    selected_row,
                    corrected_columns,
                )

                st.session_state["excel_impedance_data"] = excel_impedance_data
                st.session_state["excel_impedance_source"] = line_parameter_source

                st.markdown("#### Data Impedansi yang Dipilih")

                st.json(
                    {
                        "line_name": excel_impedance_data["line_name"],
                        "bay_pht": excel_impedance_data["bay_pht"],
                        "gi_a": excel_impedance_data["gi_a"],
                        "gi_b": excel_impedance_data["gi_b"],
                        "conductor_type": excel_impedance_data["conductor_type"],
                        "length": excel_impedance_data["length"],
                        "R1": excel_impedance_data["R1"],
                        "X1": excel_impedance_data["X1"],
                        "R0": excel_impedance_data["R0"],
                        "X0": excel_impedance_data["X0"],
                        "Z1_abs": excel_impedance_data["Z1_abs"],
                        "Z1_angle_deg": excel_impedance_data["Z1_angle_deg"],
                        "Z0_abs": excel_impedance_data["Z0_abs"],
                        "Z0_angle_deg": excel_impedance_data["Z0_angle_deg"],
                        "ratio_gia_ct": excel_impedance_data["ratio_gia_ct"],
                        "ratio_gia_vt": excel_impedance_data["ratio_gia_vt"],
                        "ratio_gib_ct": excel_impedance_data["ratio_gib_ct"],
                        "ratio_gib_vt": excel_impedance_data["ratio_gib_vt"],
                    }
                )

                if "excel_ratio_side" not in st.session_state:
                    st.session_state["excel_ratio_side"] = "Tidak gunakan dari Excel"
                
                if not use_cable_database:
                    st.markdown("#### Pilihan Ratio CT/VT dari Database")

                    ratio_side = st.radio(
                        "Gunakan ratio dari sisi GI mana?",
                        ["Tidak gunakan dari Excel", "GI A", "GI B"],
                        horizontal=True,
                        key="excel_ratio_side",
                    )
                else:
                    st.session_state["excel_ratio_side"] = "Tidak gunakan dari Excel"

            except Exception as e:
                st.error(f"Gagal membaca database {database_file_path}.")
                st.exception(e)

        if (
            line_parameter_source != "Input Manual"
            and st.session_state.get("excel_impedance_source") == line_parameter_source
            and "excel_impedance_data" in st.session_state
        ):
            excel_impedance_data = st.session_state["excel_impedance_data"]

        st.markdown("### Basic Line Data")

        col_lp1, col_lp2, col_lp3 = st.columns(3)

        with col_lp1:
            default_line_name = "SUTT 150 kV GI A - GI B"

            if excel_impedance_data and excel_impedance_data.get("line_name"):
                default_line_name = str(excel_impedance_data["line_name"])
            elif (
                line_parameter_source == "Database Excel Cable Data"
                and excel_impedance_data
                and excel_impedance_data.get("conductor_type")
            ):
                default_line_name = (
                    f"Line baru - {excel_impedance_data['conductor_type']}"
                )

            line_name = st.text_input(
                "Line Name",
                value=default_line_name,
            )

        with col_lp2:
            default_line_length = 75.0

            if excel_impedance_data and excel_impedance_data.get("length"):
                default_line_length = float(excel_impedance_data["length"])

            line_length = st.number_input(
                "Line Length",
                value=default_line_length,
                min_value=0.001,
                step=1.0,
            )

        with col_lp3:
            length_unit = st.selectbox(
                "Length Unit",
                ["km", "miles"],
                index=0,
            )

        st.markdown("### Input Format")

        col_fmt1, col_fmt2 = st.columns(2)

        with col_fmt1:
            default_impedance_input_index = 0

            impedance_input = st.radio(
                "Impedance Input",
                ["relative", "absolute"],
                index=default_impedance_input_index,
                horizontal=True,
                help="Relative = ohm/km atau ohm/mile. Absolute = total ohm saluran.",
            )

        with col_fmt2:
            base_side = st.radio(
                "Base Side",
                ["primary", "secondary"],
                horizontal=True,
                help="Primary jika impedansi dalam ohm primer. Secondary jika dari setting relay.",
            )

        st.markdown("### Transformer Ratio for Secondary Impedance Conversion")

        default_lp_ct_primary = 800.0
        default_lp_ct_secondary = 1.0
        default_lp_vt_primary = 150000.0
        default_lp_vt_secondary = 100.0

        if excel_impedance_data:
            ratio_side = st.session_state.get("excel_ratio_side", "Tidak gunakan dari Excel")

            if ratio_side == "GI A":
                ct_data = excel_impedance_data.get("ratio_gia_ct")
                vt_data = excel_impedance_data.get("ratio_gia_vt")

                if ct_data:
                    default_lp_ct_primary = float(ct_data["primary"])
                    default_lp_ct_secondary = float(ct_data["secondary"])

                if vt_data:
                    default_lp_vt_primary = float(vt_data["primary"])
                    default_lp_vt_secondary = float(vt_data["secondary"])

            elif ratio_side == "GI B":
                ct_data = excel_impedance_data.get("ratio_gib_ct")
                vt_data = excel_impedance_data.get("ratio_gib_vt")

                if ct_data:
                    default_lp_ct_primary = float(ct_data["primary"])
                    default_lp_ct_secondary = float(ct_data["secondary"])

                if vt_data:
                    default_lp_vt_primary = float(vt_data["primary"])
                    default_lp_vt_secondary = float(vt_data["secondary"])

        col_tr1, col_tr2, col_tr3, col_tr4 = st.columns(4)

        with col_tr1:
            lp_ct_primary = st.number_input(
                "Line CT Primary (A)",
                value=default_lp_ct_primary,
                min_value=0.001,
            )

        with col_tr2:
            lp_ct_secondary = st.number_input(
                "Line CT Secondary (A)",
                value=default_lp_ct_secondary,
                min_value=0.001,
            )

        with col_tr3:
            lp_vt_primary = st.number_input(
                "Line VT Primary (V)",
                value=default_lp_vt_primary,
                min_value=0.001,
            )

        with col_tr4:
            lp_vt_secondary = st.number_input(
                "Line VT Secondary (V)",
                value=default_lp_vt_secondary,
                min_value=0.001,
            )

        st.markdown("### Positive-Sequence System")

        if excel_impedance_data:
            positive_sequence_default_index = 0  # R_X
        else:
            positive_sequence_default_index = 0

        positive_sequence_mode = st.selectbox(
            "Positive-Sequence Input Mode",
            ["R_X", "Z_PHI", "X_PHI"],
            index=positive_sequence_default_index,
        )

        r1 = None
        x1 = None
        z1_mag = None
        phi1_deg = None

        if positive_sequence_mode == "R_X":
            col_z1a, col_z1b = st.columns(2)

            with col_z1a:
                default_r1 = 0.08
                default_x1 = 0.42

                if excel_impedance_data:
                    default_r1 = float(excel_impedance_data["R1"])
                    default_x1 = float(excel_impedance_data["X1"])

                r1 = st.number_input(
                    "R1 / R1' (ohm or ohm/km)",
                    value=default_r1,
                )

            with col_z1b:
                x1 = st.number_input(
                    "X1 / X1' (ohm or ohm/km)",
                    value=default_x1,
                )

        elif positive_sequence_mode == "Z_PHI":
            col_z1a, col_z1b = st.columns(2)

            with col_z1a:
                z1_mag = st.number_input(
                    "Z1 Magnitude",
                    value=0.4275,
                )

            with col_z1b:
                phi1_deg = st.number_input(
                    "Phi1 Angle (deg)",
                    value=79.22,
                )

        elif positive_sequence_mode == "X_PHI":
            col_z1a, col_z1b = st.columns(2)

            with col_z1a:
                x1 = st.number_input(
                    "X1 / X1' (ohm or ohm/km)",
                    value=0.42,
                )

            with col_z1b:
                phi1_deg = st.number_input(
                    "Phi1 Angle (deg)",
                    value=79.22,
                )

        st.markdown("### Zero-Sequence System")

        zero_sequence_mode = st.selectbox(
            "Zero-Sequence Input Mode",
            ["R0_X0", "RE_RL_XE_XL", "Z0_Z1", "KL"],
            index=0,
        )

        r0 = None
        x0 = None
        re_rl = None
        xe_xl = None
        z0_z1_mag = None
        z0_z1_angle_deg = None
        kl_mag = None
        kl_angle_deg = None

        if zero_sequence_mode == "R0_X0":
            col_z0a, col_z0b = st.columns(2)

            with col_z0a:
                default_r0 = 0.25
                default_x0 = 1.25

                if excel_impedance_data:
                    default_r0 = float(excel_impedance_data["R0"])
                    default_x0 = float(excel_impedance_data["X0"])

                r0 = st.number_input(
                    "R0 / R0' (ohm or ohm/km)",
                    value=default_r0,
                )

            with col_z0b:
                x0 = st.number_input(
                    "X0 / X0' (ohm or ohm/km)",
                    value=default_x0,
                )

        elif zero_sequence_mode == "RE_RL_XE_XL":
            col_z0a, col_z0b = st.columns(2)

            with col_z0a:
                re_rl = st.number_input(
                    "RE/RL",
                    value=3.125,
                )

            with col_z0b:
                xe_xl = st.number_input(
                    "XE/XL",
                    value=2.976,
                )

            st.info(
                "Catatan: RE/RL dan XE/XL digunakan untuk membentuk Z0 dari Z1. "
                "Pastikan definisinya sesuai dengan setting relay yang digunakan."
            )

        elif zero_sequence_mode == "Z0_Z1":
            col_z0a, col_z0b = st.columns(2)

            with col_z0a:
                z0_z1_mag = st.number_input(
                    "Z0/Z1 Magnitude",
                    value=3.0,
                )

            with col_z0b:
                z0_z1_angle_deg = st.number_input(
                    "Z0/Z1 Angle (deg)",
                    value=0.0,
                )

        elif zero_sequence_mode == "KL":
            col_z0a, col_z0b = st.columns(2)

            with col_z0a:
                kl_mag = st.number_input(
                    "kL Magnitude",
                    value=0.70,
                )

            with col_z0b:
                kl_angle_deg = st.number_input(
                    "kL Angle (deg)",
                    value=0.0,
                )

            st.warning(
                "Catatan: pada aplikasi ini kL diasumsikan sebagai ZE/Z1, "
                "dengan ZE = (Z0 - Z1) / 3. Pastikan definisi ini sesuai dengan setting relay."
            )
        
        if excel_impedance_data:
            st.success(
                "Parameter Z1 dan Z0 menggunakan data dari Excel impedansi konduktor/saluran."
            )
        else:
            st.info(
                "Parameter Z1 dan Z0 menggunakan input manual."
            )

        st.markdown("### Normalize Parameter")

        if st.button("Normalize Line Parameter"):
            try:
                line_param = normalize_line_parameter(
                    line_name=line_name,
                    length=line_length,
                    length_unit=length_unit,
                    impedance_input=impedance_input,
                    base_side=base_side,
                    positive_sequence_mode=positive_sequence_mode,
                    zero_sequence_mode=zero_sequence_mode,
                    ct_primary=lp_ct_primary,
                    ct_secondary=lp_ct_secondary,
                    vt_primary=lp_vt_primary,
                    vt_secondary=lp_vt_secondary,
                    r1=r1,
                    x1=x1,
                    z1_mag=z1_mag,
                    phi1_deg=phi1_deg,
                    r0=r0,
                    x0=x0,
                    re_rl=re_rl,
                    xe_xl=xe_xl,
                    z0_z1_mag=z0_z1_mag,
                    z0_z1_angle_deg=z0_z1_angle_deg,
                    kl_mag=kl_mag,
                    kl_angle_deg=kl_angle_deg,
                )

                line_param_df = build_line_parameter_dataframe(line_param)

                st.session_state["line_param"] = line_param
                st.session_state["line_param_df"] = line_param_df

                st.success("Parameter saluran berhasil dinormalisasi.")

            except Exception as e:
                st.error("Normalisasi parameter saluran gagal.")
                st.exception(e)

        if "line_param" in st.session_state:
            line_param = st.session_state["line_param"]
            line_param_df = st.session_state["line_param_df"]

            st.markdown("### Normalized Line Parameter")

            col_n1, col_n2, col_n3 = st.columns(3)

            col_n1.metric("Line Name", line_param["line_name"])
            col_n2.metric("Length", f'{line_param["length_km"]:.3f} km')
            col_n3.metric("Base Side", line_param["base_side"])

            st.dataframe(
                line_param_df.style.format(
                    {
                        "Real": lambda x: "" if pd.isna(x) else f"{x:.6f}",
                        "Imag": lambda x: "" if pd.isna(x) else f"{x:.6f}",
                        "Magnitude": lambda x: "" if pd.isna(x) else f"{x:.6f}",
                        "Angle Deg": lambda x: "" if pd.isna(x) else f"{x:.2f}",
                    }
                ),
                use_container_width=True,
            )

            st.markdown("### Ringkasan untuk Perhitungan Jarak Gangguan")

            z1 = line_param["Z1_per_km"]
            z0 = line_param["Z0_per_km"]
            k0 = line_param["K0"]

            st.code(
                f"""
                Z1_per_km = {z1.real:.6f} + j{z1.imag:.6f} ohm/km
                Z0_per_km = {z0.real:.6f} + j{z0.imag:.6f} ohm/km
                K0        = {k0.real:.6f} + j{k0.imag:.6f}
                Length    = {line_param["length_km"]:.3f} km
                """,
                language="text",
            )


with tab8:
    st.subheader("High Resistance Fault Detection")

    if "phasors" not in st.session_state:
        st.warning("Silakan lakukan Phasor Calculation terlebih dahulu.")
        st.stop()

    if "fault_type_result" not in st.session_state:
        st.warning("Silakan lakukan Fault Type Detection terlebih dahulu.")
        st.stop()

    if "line_param" not in st.session_state:
        st.warning("Silakan lakukan Line Parameter terlebih dahulu.")
        st.stop()

    phasors = st.session_state["phasors"]
    fault_type_result = st.session_state["fault_type_result"]
    line_param = st.session_state["line_param"]

    st.info(
        "High resistance fault biasanya membuat Zapp bergeser ke arah resistif. "
        "Karena itu aplikasi membandingkan jarak berbasis magnitude, reactance, "
        "dan proyeksi terhadap sudut Z1."
    )

    st.markdown("### Threshold Deteksi")

    col_hr1, col_hr2, col_hr3 = st.columns(3)

    with col_hr1:
        rf_threshold_ohm = st.number_input(
            "Rf Threshold (ohm primary)",
            value=10.0,
            min_value=0.1,
            step=1.0,
        )

    with col_hr2:
        angle_deviation_threshold_deg = st.number_input(
            "Angle Deviation Threshold (deg)",
            value=10.0,
            min_value=1.0,
            step=1.0,
        )

    with col_hr3:
        distance_deviation_threshold_percent = st.number_input(
            "Distance Deviation Threshold (%)",
            value=15.0,
            min_value=1.0,
            step=1.0,
        )

    try:
        hr_result = detect_high_resistance_fault(
            phasors=phasors,
            line_param=line_param,
            fault_type_result=fault_type_result,
            rf_threshold_ohm=rf_threshold_ohm,
            angle_deviation_threshold_deg=angle_deviation_threshold_deg,
            distance_deviation_threshold_percent=distance_deviation_threshold_percent,
        )

        st.session_state["high_resistance_result"] = hr_result

        st.markdown("### Hasil Deteksi")

        col_a, col_b, col_c, col_d = st.columns(4)

        col_a.metric("Selected Loop", hr_result["selected_loop"])
        col_b.metric(
            "High Resistance",
            "Suspected" if hr_result["high_resistance_suspected"] else "No",
        )
        col_c.metric("Rf Estimate", f'{hr_result["Rf_est_ohm"]:.3f} Ω')
        col_d.metric("Analysis Confidence", f'{hr_result["analysis_confidence"]}/10')

        if hr_result["high_resistance_suspected"]:
            st.warning(
                "Indikasi gangguan high resistance terdeteksi. "
                "Hasil fault location single-ended perlu diberi status UNCERTAIN."
            )
        else:
            st.success(
                "Belum ada indikasi kuat gangguan high resistance. "
                f"HR evidence score: {hr_result['evidence_score']}/10."
            )

        if hr_result["warnings"]:
            st.markdown("### Warning")
            for warning in hr_result["warnings"]:
                st.warning(warning)

        st.markdown("### Detail Perhitungan")

        hr_df = build_high_resistance_dataframe(hr_result)

        st.dataframe(
            hr_df.style.format(
                {
                    "Value": lambda x: f"{x:.6f}" if isinstance(x, (int, float)) else x
                }
            ),
            use_container_width=True,
        )

        st.markdown("### Perbandingan Metode Estimasi Jarak")

        distance_df = pd.DataFrame(
            {
                "Method": [
                    "Reactance-based",
                    "Magnitude-based",
                    "Projection-based",
                ],
                "Distance km": [
                    hr_result["distance_x_km"],
                    hr_result["distance_mag_km"],
                    hr_result["distance_projection_km"],
                ],
                "Distance %": [
                    hr_result["distance_x_percent"],
                    hr_result["distance_mag_percent"],
                    hr_result["distance_projection_percent"],
                ],
            }
        )

        st.dataframe(
            distance_df.style.format(
                {
                    "Distance km": "{:.3f}",
                    "Distance %": "{:.2f}",
                }
            ),
            use_container_width=True,
        )

        fig_dist = px.bar(
            distance_df,
            x="Method",
            y="Distance km",
            title="Perbandingan Estimasi Jarak Gangguan",
            text_auto=".2f",
        )

        st.plotly_chart(fig_dist, use_container_width=True)

        st.markdown("### R-X Position")

        z1_total = line_param["Z1_total"]
        z_app = hr_result["Zapp"]

        rx_df = pd.DataFrame(
            {
                "Point": ["Origin", "Z1 Total", "Zapp"],
                "R": [0.0, z1_total.real, z_app.real],
                "X": [0.0, z1_total.imag, z_app.imag],
            }
        )

        fig_rx = px.scatter(
            rx_df,
            x="R",
            y="X",
            text="Point",
            title="Posisi Zapp terhadap Z1 Total pada Diagram R-X",
        )

        fig_rx.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=z1_total.real,
            y1=z1_total.imag,
        )

        fig_rx.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=z_app.real,
            y1=z_app.imag,
            line=dict(dash="dash"),
        )

        fig_rx.update_traces(textposition="top center")
        fig_rx.update_layout(
            xaxis_title="R (ohm)",
            yaxis_title="X (ohm)",
        )

        st.plotly_chart(fig_rx, use_container_width=True)

        st.markdown("### Interpretasi")

        if hr_result["high_resistance_suspected"]:
            st.write(
                "Zapp cenderung bergeser ke kanan pada diagram R-X. "
                "Untuk kasus ini, estimasi jarak berbasis magnitude dapat terlihat lebih jauh "
                "daripada estimasi berbasis reactance. Gunakan distance berbasis X atau "
                "projection sebagai pembanding, dan beri label hasil sebagai uncertain."
            )
        else:
            st.write(
                "Zapp belum menunjukkan dominasi tahanan gangguan yang kuat. "
                "Namun tetap validasi dengan waveform, arus ground, dan data inspeksi lapangan."
            )

    except Exception as e:
        st.error("Analisis high resistance gagal.")
        st.exception(e)


with tab9:
    st.subheader("Single-Ended Fault Locator")

    st.write(
        "Fitur ini menghitung estimasi jarak gangguan dari satu ujung relay distance "
        "berdasarkan fasor, jenis gangguan, dan parameter saluran."
    )

    if "phasors" not in st.session_state:
        st.warning("Silakan lakukan Step 5: Phasor Calculation terlebih dahulu.")
        st.stop()

    if "fault_type_result" not in st.session_state:
        st.warning("Silakan lakukan Step 6: Fault Type Detection terlebih dahulu.")
        st.stop()

    if "line_param" not in st.session_state:
        st.warning("Silakan lakukan Step 7: Line Parameter terlebih dahulu.")
        st.stop()

    phasors = st.session_state["phasors"]
    fault_type_result = st.session_state["fault_type_result"]
    line_param = st.session_state["line_param"]

    st.markdown("### Input Perhitungan")

    col_se1, col_se2, col_se3, col_se4 = st.columns(4)

    col_se1.metric("Fault Type", fault_type_result.get("fault_type", "-"))
    col_se2.metric("Line Length", f'{line_param["length_km"]:.3f} km')
    col_se3.metric("Z1/km", f'{line_param["Z1_per_km"].real:.4f} + j{line_param["Z1_per_km"].imag:.4f}')
    col_se4.metric("K0", f'{line_param["K0"].real:.4f} + j{line_param["K0"].imag:.4f}')

    st.markdown("### Metode Rekomendasi Jarak")

    recommended_method = st.selectbox(
        "Pilih metode jarak utama",
        [
            "reactance",
            "projection",
            "magnitude",
        ],
        index=0,
        help=(
            "Reactance cocok untuk mengurangi pengaruh tahanan gangguan. "
            "Projection memproyeksikan Zapp ke arah sudut Z1. "
            "Magnitude sederhana, tetapi mudah bias untuk high resistance fault."
        ),
    )

    if st.button("Calculate Single-Ended Fault Location"):
        try:
            single_result = calculate_single_ended_fault_location(
                phasors=phasors,
                fault_type_result=fault_type_result,
                line_param=line_param,
                recommended_method=recommended_method,
            )

            single_df = build_single_ended_result_dataframe(single_result)

            st.session_state["single_ended_result"] = single_result
            st.session_state["single_ended_df"] = single_df

            st.success("Single-ended fault location berhasil dihitung.")

        except Exception as e:
            st.error("Perhitungan single-ended gagal.")
            st.exception(e)

    if "single_ended_result" in st.session_state:
        single_result = st.session_state["single_ended_result"]
        single_df = st.session_state["single_ended_df"]

        st.markdown("### Hasil Utama")

        col_r1, col_r2, col_r3, col_r4 = st.columns(4)

        col_r1.metric(
            "Recommended Distance",
            f'{single_result["recommended_distance_km"]:.3f} km',
        )

        col_r2.metric(
            "Distance %",
            f'{single_result["recommended_distance_percent"]:.2f} %',
        )

        col_r3.metric(
            "Zapp",
            f'{single_result["Zapp_R"]:.3f} + j{single_result["Zapp_X"]:.3f} Ω',
        )

        col_r4.metric(
            "Status",
            single_result["status"],
        )

        if single_result["status"] == "VALID":
            st.success("Hasil single-ended berada dalam batas normal.")
        elif single_result["status"] == "CHECK":
            st.warning("Hasil single-ended perlu dicek ulang dengan waveform, SOE, dan data lapangan.")
        else:
            st.error("Hasil single-ended tidak pasti. Cek polaritas, line parameter, dan fault type.")

        if single_result["warnings"]:
            st.markdown("### Warning")
            for warning in single_result["warnings"]:
                st.warning(warning)

        st.markdown("### Detail Perhitungan")

        st.dataframe(
            single_df.style.format(
                {
                    "Value": lambda x: f"{x:.6f}" if isinstance(x, (int, float)) else x
                }
            ),
            use_container_width=True,
        )

        st.markdown("### Perbandingan Metode Jarak")

        distance_df = pd.DataFrame(
            {
                "Method": [
                    "Magnitude",
                    "Reactance",
                    "Projection",
                    "Recommended",
                ],
                "Distance km": [
                    single_result["distance_mag_km"],
                    single_result["distance_x_km"],
                    single_result["distance_projection_km"],
                    single_result["recommended_distance_km"],
                ],
                "Distance %": [
                    single_result["distance_mag_percent"],
                    single_result["distance_x_percent"],
                    single_result["distance_projection_percent"],
                    single_result["recommended_distance_percent"],
                ],
            }
        )

        st.dataframe(
            distance_df.style.format(
                {
                    "Distance km": "{:.3f}",
                    "Distance %": "{:.2f}",
                }
            ),
            use_container_width=True,
        )

        fig_dist = px.bar(
            distance_df,
            x="Method",
            y="Distance km",
            text_auto=".2f",
            title="Perbandingan Estimasi Jarak Single-Ended",
        )

        st.plotly_chart(fig_dist, use_container_width=True)

        st.markdown("### Diagram R-X")

        z1_total = line_param["Z1_total"]
        zapp = single_result["Zapp"]
        recommended_distance = single_result["recommended_distance_km"]
        z_recommended_line = recommended_distance * line_param["Z1_per_km"]

        rx_df = pd.DataFrame(
            {
                "Point": [
                    "Origin",
                    "Z1 Total",
                    "Zapp",
                    "Projected Fault Point",
                ],
                "R": [
                    0.0,
                    z1_total.real,
                    zapp.real,
                    z_recommended_line.real,
                ],
                "X": [
                    0.0,
                    z1_total.imag,
                    zapp.imag,
                    z_recommended_line.imag,
                ],
            }
        )

        fig_rx = px.scatter(
            rx_df,
            x="R",
            y="X",
            text="Point",
            title="Single-Ended R-X Diagram",
        )

        fig_rx.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=z1_total.real,
            y1=z1_total.imag,
        )

        fig_rx.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=zapp.real,
            y1=zapp.imag,
            line=dict(dash="dash"),
        )

        fig_rx.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=z_recommended_line.real,
            y1=z_recommended_line.imag,
            line=dict(dash="dot"),
        )

        fig_rx.update_traces(textposition="top center")

        fig_rx.update_layout(
            xaxis_title="R (ohm)",
            yaxis_title="X (ohm)",
            yaxis=dict(scaleanchor="x", scaleratio=1),
        )

        st.plotly_chart(fig_rx, use_container_width=True)

        st.markdown("### Interpretasi")

        st.write(
            f"Loop yang digunakan adalah **{single_result['selected_loop']}**. "
            f"Impedansi terlihat relay adalah **{single_result['Zapp_R']:.4f} + "
            f"j{single_result['Zapp_X']:.4f} Ω**. "
            f"Jarak rekomendasi dari ujung relay adalah **{single_result['recommended_distance_km']:.3f} km** "
            f"atau **{single_result['recommended_distance_percent']:.2f}%** dari panjang saluran."
        )

        if abs(single_result["Rf_est_ohm"]) > 10:
            st.warning(
                f"Estimasi tahanan gangguan adalah {single_result['Rf_est_ohm']:.3f} Ω. "
                "Nilai ini cukup besar sehingga hasil single-ended dapat bergeser, terutama pada gangguan high resistance."
            )


with tab10:
    st.subheader("Two-Ended Fault Locator")

    st.write(
        "Fitur ini menghitung lokasi gangguan menggunakan dua rekaman distance relay "
        "dari dua ujung saluran yang saling berhadapan. Perhitungan awal memakai "
        "positive-sequence two-ended method."
    )

    if "phasors" not in st.session_state:
        st.warning("Selesaikan dulu Step 5: Phasor Calculation untuk rekaman local end.")
        st.stop()

    if "line_param" not in st.session_state:
        st.warning("Selesaikan dulu Step 7: Line Parameter.")
        st.stop()

    local_phasors = st.session_state["phasors"]
    line_param = st.session_state["line_param"]

    st.markdown("### Local End Data")

    col_l1, col_l2, col_l3 = st.columns(3)

    col_l1.metric("Local V1 RMS", f'{local_phasors["V1"]["magnitude"]:.3f}')
    col_l2.metric("Local I1 RMS", f'{local_phasors["I1"]["magnitude"]:.3f}')
    col_l3.metric("Line Length", f'{line_param["length_km"]:.3f} km')

    st.markdown("### Upload Remote End Fault Record")

    remote_cfg_file = st.file_uploader(
        "Upload remote end .cfg",
        type=["cfg"],
        key="remote_cfg_file",
    )

    remote_dat_file = st.file_uploader(
        "Upload remote end .dat",
        type=["dat"],
        key="remote_dat_file",
    )

    if remote_cfg_file is None or remote_dat_file is None:
        st.info("Upload pasangan file .cfg dan .dat dari relay ujung remote.")
        st.stop()

    remote_cfg_file.seek(0)
    remote_dat_file.seek(0)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".cfg") as temp_remote_cfg:
        temp_remote_cfg.write(remote_cfg_file.read())
        remote_cfg_path = temp_remote_cfg.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".dat") as temp_remote_dat:
        temp_remote_dat.write(remote_dat_file.read())
        remote_dat_path = temp_remote_dat.name

    try:
        remote_df, remote_metadata = read_comtrade(remote_cfg_path, remote_dat_path)

        remote_auto_assignment = detect_voltage_current_channels(
            remote_df,
            remote_metadata,
        )

        remote_auto_transformer_data = get_auto_transformer_data(remote_metadata)
        remote_auto_recorded_side = detect_recorded_side(remote_metadata)

        st.session_state["remote_auto_assignment"] = remote_auto_assignment
        st.session_state["remote_auto_transformer_data"] = remote_auto_transformer_data
        st.session_state["remote_auto_recorded_side"] = remote_auto_recorded_side

        st.success("Remote end COMTRADE berhasil dibaca dan signal assignment otomatis dibuat.")

    except Exception as e:
        st.error("Remote end COMTRADE gagal dibaca.")
        st.exception(e)
        st.stop()

    st.markdown("### Remote End Signal Assignment")

    st.markdown("### Remote Auto-Read Metadata")

    col_ram1, col_ram2, col_ram3, col_ram4 = st.columns(4)

    col_ram1.metric(
        "Remote CFG Start Time",
        str(remote_metadata.get("cfg_start_time") or "-"),
    )

    col_ram2.metric(
        "Remote CFG Trigger Time",
        str(remote_metadata.get("cfg_trigger_time") or "-"),
    )

    col_ram3.metric(
        "Remote VT Ratio from CFG",
        str(remote_metadata.get("vt_ratio_from_cfg") or "-"),
    )

    col_ram4.metric(
        "Remote CT Ratio from CFG",
        str(remote_metadata.get("ct_ratio_from_cfg") or "-"),
    )

    with st.expander("Remote Auto Signal Assignment Preview"):
        remote_auto_summary_df = build_auto_assignment_summary(
            st.session_state["remote_auto_assignment"],
            st.session_state["remote_auto_transformer_data"],
            remote_metadata,
        )

        st.dataframe(remote_auto_summary_df, use_container_width=True)

    with st.expander("Remote Analog Metadata dari .cfg"):
        remote_analog_meta_df = pd.DataFrame(
            remote_metadata.get("analog_metadata", [])
        )
        st.dataframe(remote_analog_meta_df, use_container_width=True)

    remote_channel_options = [col for col in remote_df.columns if col != "time"]
    remote_ground_options = ["None"] + remote_channel_options

    remote_auto_assignment = st.session_state.get("remote_auto_assignment", {})
    remote_auto_transformer_data = st.session_state.get(
        "remote_auto_transformer_data",
        {},
    )
    remote_auto_recorded_side = st.session_state.get(
        "remote_auto_recorded_side",
        "secondary",
    )


    def get_remote_channel_index(channel_name, options, default_index=0):
        if channel_name in options:
            return options.index(channel_name)
        return default_index


    def get_remote_ground_index(channel_name, options):
        if channel_name in options:
            return options.index(channel_name)
        return 0

    col_rv1, col_rv2, col_rv3 = st.columns(3)

    with col_rv1:
        remote_va_channel = st.selectbox(
            "Remote Va / VL1",
            remote_channel_options,
            index=get_remote_channel_index(
                remote_auto_assignment.get("Va"),
                remote_channel_options,
                0,
            ),
            key="remote_va_channel",
        )

    with col_rv2:
        remote_vb_channel = st.selectbox(
            "Remote Vb / VL2",
            remote_channel_options,
            index=get_remote_channel_index(
                remote_auto_assignment.get("Vb"),
                remote_channel_options,
                1 if len(remote_channel_options) > 1 else 0,
            ),
            key="remote_vb_channel",
        )

    with col_rv3:
        remote_vc_channel = st.selectbox(
            "Remote Vc / VL3",
            remote_channel_options,
            index=get_remote_channel_index(
                remote_auto_assignment.get("Vc"),
                remote_channel_options,
                2 if len(remote_channel_options) > 2 else 0,
            ),
            key="remote_vc_channel",
        )

    col_ri1, col_ri2, col_ri3 = st.columns(3)

    default_remote_ia_index = 3 if len(remote_channel_options) > 3 else 0
    default_remote_ib_index = 4 if len(remote_channel_options) > 4 else 0
    default_remote_ic_index = 5 if len(remote_channel_options) > 5 else 0

    with col_ri1:
        remote_ia_channel = st.selectbox(
            "Remote Ia / IL1",
            remote_channel_options,
            index=get_remote_channel_index(
                remote_auto_assignment.get("Ia"),
                remote_channel_options,
                default_remote_ia_index,
            ),
            key="remote_ia_channel",
        )

    with col_ri2:
        remote_ib_channel = st.selectbox(
            "Remote Ib / IL2",
            remote_channel_options,
            index=get_remote_channel_index(
                remote_auto_assignment.get("Ib"),
                remote_channel_options,
                default_remote_ib_index,
            ),
            key="remote_ib_channel",
        )

    with col_ri3:
        remote_ic_channel = st.selectbox(
            "Remote Ic / IL3",
            remote_channel_options,
            index=get_remote_channel_index(
                remote_auto_assignment.get("Ic"),
                remote_channel_options,
                default_remote_ic_index,
            ),
            key="remote_ic_channel",
        )

    remote_ie_channel = st.selectbox(
        "Remote IE / IN / 3I0 jika tersedia",
        remote_ground_options,
        index=0,
        key="remote_ie_channel",
    )

    st.markdown("### Remote End Transformer Data")

    remote_recorded_side_options = ["secondary", "primary"]

    remote_recorded_side_default_index = (
        remote_recorded_side_options.index(remote_auto_recorded_side)
        if remote_auto_recorded_side in remote_recorded_side_options
        else 0
    )

    remote_recorded_side = st.radio(
        "Nilai remote COMTRADE direkam sebagai:",
        remote_recorded_side_options,
        index=remote_recorded_side_default_index,
        horizontal=True,
        key="remote_recorded_side",
    )

    col_rct1, col_rct2, col_rvt1, col_rvt2 = st.columns(4)

    with col_rct1:
        remote_ct_primary = st.number_input(
            "Remote CT Primary (A)",
            value=float(remote_auto_transformer_data.get("ct_primary", 800.0)),
            min_value=0.001,
            key="remote_ct_primary",
        )

    with col_rct2:
        remote_ct_secondary = st.number_input(
            "Remote CT Secondary (A)",
            value=float(remote_auto_transformer_data.get("ct_secondary", 1.0)),
            min_value=0.001,
            key="remote_ct_secondary",
        )

    with col_rvt1:
        remote_vt_primary = st.number_input(
            "Remote VT Primary (V)",
            value=float(remote_auto_transformer_data.get("vt_primary", 150000.0)),
            min_value=0.001,
            key="remote_vt_primary",
        )

    with col_rvt2:
        remote_vt_secondary = st.number_input(
            "Remote VT Secondary (V)",
            value=float(remote_auto_transformer_data.get("vt_secondary", 100.0)),
            min_value=0.001,
            key="remote_vt_secondary",
        )

    st.caption(
        f"Remote auto ratio source: "
        f"CT = {remote_auto_transformer_data.get('ct_ratio_source', '-')}, "
        f"VT = {remote_auto_transformer_data.get('vt_ratio_source', '-')}. "
        "Tetap validasi manual karena tidak semua file CFG menyimpan primary/secondary dengan benar."
    )

    remote_assigned_df = apply_signal_assignment(
        df=remote_df,
        va_channel=remote_va_channel,
        vb_channel=remote_vb_channel,
        vc_channel=remote_vc_channel,
        ia_channel=remote_ia_channel,
        ib_channel=remote_ib_channel,
        ic_channel=remote_ic_channel,
        ie_channel=remote_ie_channel,
        recorded_side=remote_recorded_side,
        ct_primary=remote_ct_primary,
        ct_secondary=remote_ct_secondary,
        vt_primary=remote_vt_primary,
        vt_secondary=remote_vt_secondary,
    )

    st.session_state["remote_assigned_df"] = remote_assigned_df

    with st.expander("Preview Remote Assigned Data"):
        st.dataframe(remote_assigned_df.head(20), use_container_width=True)

    st.markdown("### Remote Fault Detection & Phasor")

    col_rf1, col_rf2, col_rf3 = st.columns(3)

    with col_rf1:
        remote_frequency = st.number_input(
            "Remote Frequency (Hz)",
            value=float(remote_metadata["frequency"]) if remote_metadata["frequency"] else 50.0,
            min_value=40.0,
            max_value=70.0,
            step=0.1,
            key="remote_frequency",
        )

    with col_rf2:
        remote_current_multiplier = st.number_input(
            "Remote Current Fault Multiplier",
            value=2.0,
            min_value=1.1,
            max_value=10.0,
            step=0.1,
            key="remote_current_multiplier",
        )

    with col_rf3:
        remote_voltage_threshold = st.number_input(
            "Remote Voltage Drop Threshold",
            value=0.85,
            min_value=0.1,
            max_value=1.0,
            step=0.01,
            key="remote_voltage_threshold",
        )

    remote_detection = detect_fault_inception(
        remote_assigned_df,
        frequency=remote_frequency,
        current_threshold_multiplier=remote_current_multiplier,
        voltage_drop_threshold=remote_voltage_threshold,
        min_prefault_cycles=2,
    )

    st.session_state["remote_fault_detection"] = remote_detection

    if not remote_detection["detected"]:
        st.warning(
            "Remote fault inception tidak terdeteksi otomatis. "
            "Gunakan slider manual untuk menentukan waktu fault remote."
        )

        min_remote_time = float(remote_assigned_df["time"].min())
        max_remote_time = float(remote_assigned_df["time"].max())

        manual_remote_fault_time = st.slider(
            "Pilih waktu awal gangguan remote manual (s)",
            min_value=min_remote_time,
            max_value=max_remote_time,
            value=min_remote_time,
            step=(max_remote_time - min_remote_time) / 1000,
            key="manual_remote_fault_time",
        )

        remote_fault_index = int(
            (remote_assigned_df["time"] - manual_remote_fault_time).abs().idxmin()
        )

        remote_samples_per_cycle = int(round(
            estimate_sampling_rate(remote_assigned_df) / remote_frequency
        ))

    else:
        st.success("Remote fault inception berhasil terdeteksi otomatis.")
        remote_fault_index = remote_detection["fault_index"]
        remote_samples_per_cycle = remote_detection["samples_per_cycle"]

    remote_fault_window = build_fault_window(
        remote_assigned_df,
        fault_index=remote_fault_index,
        samples_per_cycle=remote_samples_per_cycle,
        pre_fault_cycles=2,
        post_fault_cycles=4,
    )

    st.session_state["remote_fault_window"] = remote_fault_window

    col_rw1, col_rw2, col_rw3 = st.columns(3)

    col_rw1.metric("Remote Fault Time", f'{remote_fault_window["fault_time"]:.6f} s')
    col_rw2.metric("Remote DFT Time", f'{remote_fault_window["dft_time"]:.6f} s')
    col_rw3.metric("Remote Samples/Cycle", remote_samples_per_cycle)

    st.markdown("#### Remote Fault Detection & Cursor")

    col_rdc1, col_rdc2, col_rdc3, col_rdc4 = st.columns(4)

    col_rdc1.metric(
        "Detection",
        "AUTO" if remote_detection["detected"] else "MANUAL",
    )
    col_rdc2.metric("Left Cursor", f'{remote_fault_window["left_time"]:.6f} s')
    col_rdc3.metric("DFT Cursor", f'{remote_fault_window["dft_time"]:.6f} s')
    col_rdc4.metric("Right Cursor", f'{remote_fault_window["right_time"]:.6f} s')

    remote_cursor_df = pd.DataFrame(
        [
            {"Parameter": "Detected Automatically", "Value": remote_detection["detected"]},
            {"Parameter": "Fault Index", "Value": remote_fault_window["fault_index"]},
            {"Parameter": "Fault Time s", "Value": remote_fault_window["fault_time"]},
            {"Parameter": "Left Cursor Index", "Value": remote_fault_window["left_index"]},
            {"Parameter": "Left Cursor Time s", "Value": remote_fault_window["left_time"]},
            {"Parameter": "DFT Cursor Index", "Value": remote_fault_window["dft_index"]},
            {"Parameter": "DFT Cursor Time s", "Value": remote_fault_window["dft_time"]},
            {"Parameter": "Right Cursor Index", "Value": remote_fault_window["right_index"]},
            {"Parameter": "Right Cursor Time s", "Value": remote_fault_window["right_time"]},
            {"Parameter": "Samples per Cycle", "Value": remote_samples_per_cycle},
            {"Parameter": "Frequency Hz", "Value": remote_frequency},
        ]
    )

    with st.expander("Remote Fault Detection & Cursor Detail"):
        st.dataframe(
            remote_cursor_df.style.format(
                {
                    "Value": lambda x: f"{x:.6f}"
                    if isinstance(x, (int, float)) and not isinstance(x, bool)
                    else x
                }
            ),
            use_container_width=True,
        )

        remote_detection_detail_df = pd.DataFrame(
            [
                {
                    "Parameter": key,
                    "Value": value,
                }
                for key, value in remote_detection.items()
                if isinstance(value, (str, int, float, bool))
            ]
        )

        st.dataframe(
            remote_detection_detail_df.style.format(
                {
                    "Value": lambda x: f"{x:.6f}"
                    if isinstance(x, (int, float)) and not isinstance(x, bool)
                    else x
                }
            ),
            use_container_width=True,
        )

    remote_phasors = calculate_all_phasors(
        df=remote_assigned_df,
        cursor_index=remote_fault_window["dft_index"],
        samples_per_cycle=remote_samples_per_cycle,
    )

    remote_phasors = add_sequence_components_to_phasor_dict(remote_phasors)

    st.session_state["remote_phasors"] = remote_phasors

    st.markdown("#### Remote Fault Type Detection")

    st.caption(
        "Parameter di bawah ini memengaruhi klasifikasi tipe gangguan remote. "
        "Remote Current Fault Multiplier di atas hanya dipakai untuk mencari fault cursor."
    )

    col_rftp1, col_rftp2, col_rftp3 = st.columns(3)

    with col_rftp1:
        remote_fault_type_voltage_drop_threshold = st.number_input(
            "Remote Fault Type Voltage Drop Threshold",
            value=0.80,
            min_value=0.10,
            max_value=1.00,
            step=0.01,
            key="remote_fault_type_voltage_drop_threshold",
            help="Fasa remote dianggap drop jika Vphase <= threshold x Vmax.",
        )

    with col_rftp2:
        remote_fault_type_current_rise_threshold = st.number_input(
            "Remote Fault Type Current Rise Threshold",
            value=1.50,
            min_value=1.05,
            max_value=10.00,
            step=0.05,
            key="remote_fault_type_current_rise_threshold",
            help="Fasa remote dianggap faulted jika Iphase >= threshold x Imin.",
        )

    with col_rftp3:
        remote_fault_type_ground_current_threshold = st.number_input(
            "Remote Fault Type Ground Current Threshold",
            value=0.20,
            min_value=0.01,
            max_value=1.00,
            step=0.01,
            key="remote_fault_type_ground_current_threshold",
            help="Ground fault remote jika IE/Imax atau I0/Iavg melebihi threshold.",
        )

    remote_fault_type_result = detect_fault_type(
        phasors=remote_phasors,
        voltage_drop_threshold=remote_fault_type_voltage_drop_threshold,
        current_rise_threshold=remote_fault_type_current_rise_threshold,
        ground_current_threshold=remote_fault_type_ground_current_threshold,
    )
    remote_fault_type_df = build_fault_type_metrics_dataframe(remote_fault_type_result)

    st.session_state["remote_fault_type_result"] = remote_fault_type_result
    st.session_state["remote_fault_type_df"] = remote_fault_type_df

    col_rft1, col_rft2, col_rft3, col_rft4 = st.columns(4)

    col_rft1.metric("Remote Fault Type", remote_fault_type_result["fault_type"])
    col_rft2.metric(
        "Ground Involved",
        "Yes" if remote_fault_type_result["ground_involved"] else "No",
    )
    col_rft3.metric("Confidence", f'{remote_fault_type_result["confidence"]}/10')
    col_rft4.metric(
        "Faulted Phases",
        ", ".join(remote_fault_type_result["faulted_phases"])
        if remote_fault_type_result["faulted_phases"]
        else "-",
    )

    with st.expander("Remote Fault Type Detection Detail"):
        st.dataframe(remote_fault_type_df, use_container_width=True)

    remote_phasor_df = build_phasor_dataframe(remote_phasors)

    with st.expander("Remote Phasor Table"):
        st.dataframe(
            remote_phasor_df.style.format(
                {
                    "Magnitude RMS": "{:.4f}",
                    "Angle Deg": "{:.2f}",
                    "Real": "{:.4f}",
                    "Imag": "{:.4f}",
                }
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

        col_sync1.metric("Δ Fault Time", f"{delta_fault_time:.6f} s")
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
        else:
            st.success("Perbedaan waktu fault local dan remote masih dalam batas 1 siklus.")
    else:
        st.warning("Local fault window belum tersedia.")

    st.markdown("### Two-Ended Calculation")

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

    if st.button("Calculate Two-Ended Fault Location"):
        try:
            adapted_remote_phasors = remote_phasors

            if remote_direction_mode == "auto_adapt_record":
                best_candidate, all_candidates = choose_best_two_ended_adaptation(
                    local_phasors=local_phasors,
                    remote_phasors=remote_phasors,
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
                    remote_phasors=remote_phasors,
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
                    remote_phasors=remote_phasors,
                    line_param=line_param,
                    remote_current_direction=remote_direction_mode,
                )

                two_quality = evaluate_two_ended_quality(two_result, line_param)
                st.session_state.pop("two_ended_candidates", None)
                st.session_state["two_ended_adapted_remote_phasors"] = remote_phasors

            single_ended_compare_error = None

            try:
                local_fault_type_result = st.session_state.get("fault_type_result")

                if local_fault_type_result is None:
                    local_fault_type_result = detect_fault_type(local_phasors)

                remote_fault_type_result = detect_fault_type(
                    phasors=adapted_remote_phasors,
                    voltage_drop_threshold=remote_fault_type_voltage_drop_threshold,
                    current_rise_threshold=remote_fault_type_current_rise_threshold,
                    ground_current_threshold=remote_fault_type_ground_current_threshold,
                )
                remote_single_phasors = adapted_remote_phasors

                if two_result["remote_current_direction"] == "opposite_to_line":
                    remote_single_phasors = invert_current_phasors(adapted_remote_phasors)

                local_single_result = calculate_single_ended_fault_location(
                    phasors=local_phasors,
                    fault_type_result=local_fault_type_result,
                    line_param=line_param,
                    recommended_method="reactance",
                )

                remote_single_result = calculate_single_ended_fault_location(
                    phasors=remote_single_phasors,
                    fault_type_result=remote_fault_type_result,
                    line_param=line_param,
                    recommended_method="reactance",
                )

                local_single_df = build_single_ended_result_dataframe(local_single_result)
                remote_single_df = build_single_ended_result_dataframe(remote_single_result)

                st.session_state["two_ended_local_single_result"] = local_single_result
                st.session_state["two_ended_remote_single_result"] = remote_single_result
                st.session_state["two_ended_local_single_df"] = local_single_df
                st.session_state["two_ended_remote_single_df"] = remote_single_df
                st.session_state["two_ended_local_fault_type_result"] = local_fault_type_result
                st.session_state["two_ended_remote_fault_type_result"] = remote_fault_type_result

            except Exception as single_ended_error:
                single_ended_compare_error = str(single_ended_error)
                for key in [
                    "two_ended_local_single_result",
                    "two_ended_remote_single_result",
                    "two_ended_local_single_df",
                    "two_ended_remote_single_df",
                    "two_ended_local_fault_type_result",
                    "two_ended_remote_fault_type_result",
                ]:
                    st.session_state.pop(key, None)

            st.session_state["two_ended_result"] = two_result
            st.session_state["two_ended_quality"] = two_quality
            st.session_state["two_ended_single_ended_error"] = single_ended_compare_error

            st.success("Two-ended fault location berhasil dihitung.")

            if single_ended_compare_error:
                st.warning(
                    "Two-ended berhasil, tetapi single-ended comparison gagal: "
                    f"{single_ended_compare_error}"
                )

        except Exception as e:
            st.error("Two-ended fault location gagal.")
            st.exception(e)

    if "two_ended_result" in st.session_state:
        two_result = st.session_state["two_ended_result"]
        two_quality = st.session_state["two_ended_quality"]

        st.markdown("### Two-Ended Result")

        col_te1, col_te2, col_te3, col_te4 = st.columns(4)

        col_te1.metric(
            "Distance from Local",
            f'{two_result["distance_km"]:.3f} km',
        )

        col_te2.metric(
            "Distance from Remote",
            f'{two_result["distance_from_remote_km"]:.3f} km',
        )

        col_te3.metric(
            "Distance %",
            f'{two_result["distance_percent"]:.2f} %',
        )

        col_te4.metric(
            "Quality",
            f'{two_quality["quality_score"]}/10',
        )

        two_result_df = build_two_ended_result_dataframe(two_result, two_quality)

        st.dataframe(
            two_result_df.style.format(
                {
                    "Value": lambda x: f"{x:.6f}" if isinstance(x, (int, float)) else x
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
            remote_single_from_local_km = (
                L - remote_single_result["recommended_distance_km"]
            )
            remote_single_from_local_percent = remote_single_from_local_km / L * 100.0

            col_se1, col_se2, col_se3, col_se4 = st.columns(4)

            col_se1.metric(
                "Local SE from Local",
                f'{local_single_result["recommended_distance_km"]:.3f} km',
            )
            col_se2.metric(
                "Remote SE from Remote",
                f'{remote_single_result["recommended_distance_km"]:.3f} km',
            )
            col_se3.metric(
                "Remote SE from Local",
                f"{remote_single_from_local_km:.3f} km",
            )
            col_se4.metric(
                "Remote SE from Local %",
                f"{remote_single_from_local_percent:.2f} %",
            )

            st.caption(
                "Remote single-ended ditampilkan sebagai jarak dari remote end dan "
                "konversinya ke referensi local end. Jika arah arus remote dipilih "
                "`opposite_to_line`, arus fasor remote dibalik untuk perhitungan single-ended."
            )

            se_summary_df = pd.DataFrame(
                [
                    {
                        "End": "Local GI",
                        "Fault Type": local_fault_type_result.get("fault_type", "-"),
                        "Selected Loop": local_single_result["selected_loop"],
                        "Distance from Own End km": local_single_result["recommended_distance_km"],
                        "Distance from Local End km": local_single_result["recommended_distance_km"],
                        "Distance from Local End %": local_single_result["recommended_distance_percent"],
                        "Zapp R ohm": local_single_result["Zapp_R"],
                        "Zapp X ohm": local_single_result["Zapp_X"],
                        "Rf Est ohm": local_single_result["Rf_est_ohm"],
                        "Status": local_single_result["status"],
                    },
                    {
                        "End": "Remote GI",
                        "Fault Type": remote_fault_type_result.get("fault_type", "-"),
                        "Selected Loop": remote_single_result["selected_loop"],
                        "Distance from Own End km": remote_single_result["recommended_distance_km"],
                        "Distance from Local End km": remote_single_from_local_km,
                        "Distance from Local End %": remote_single_from_local_percent,
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
                        "Distance from Local End km": "{:.6f}",
                        "Distance from Local End %": "{:.2f}",
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
        d = two_result["distance_km"]
        point_names = ["Local End", "Two-Ended Fault", "Remote End"]
        point_distances = [0.0, d, L]

        if "two_ended_local_single_result" in st.session_state:
            point_names.append("Local Single-Ended")
            point_distances.append(
                st.session_state["two_ended_local_single_result"]["recommended_distance_km"]
            )

        if "two_ended_remote_single_result" in st.session_state:
            point_names.append("Remote Single-Ended")
            point_distances.append(
                L - st.session_state["two_ended_remote_single_result"]["recommended_distance_km"]
            )

        pos_df = pd.DataFrame(
            {
                "Point": point_names,
                "Distance km": point_distances,
                "Y": [0.0] * len(point_names),
            }
        )

        fig_two = px.scatter(
            pos_df,
            x="Distance km",
            y="Y",
            text="Point",
            title="Two-Ended Fault Location Along Transmission Line",
        )

        fig_two.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=L,
            y1=0,
        )

        fig_two.update_traces(textposition="top center")
        fig_two.update_layout(
            yaxis=dict(visible=False),
            xaxis_title="Distance from Local End (km)",
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
