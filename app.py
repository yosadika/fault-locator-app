import streamlit as st
import math
import cmath
import re
import textwrap
import hashlib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components

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
    calculate_auto_fault_type_thresholds,
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
    transform_remote_phasors,
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
from app_helpers import (
    MAX_PLOT_POINTS,
    validate_uploaded_extension,
    downsample_dataframe_for_plot,
    make_streamlit_safe_columns,
    invert_current_phasors,
)
from waveform_helpers import (
    build_waveform_rms_summary,
    build_assigned_waveform_plot,
    build_wavewin_style_phasor_diagram,
    build_fault_window_plot,
    build_synchronized_fault_plot,
    estimate_waveform_time_shift_by_correlation,
    fault_phase_to_current_channel,
    fault_phase_to_voltage_channel,
    build_prefault_fault_comparison_dataframe,
)
from fault_workflow_helpers import (
    explain_fault_type_result,
    build_auto_fault_type_threshold_dataframe,
    get_absolute_event_time,
    calculate_time_based_fault_location,
    calculate_auto_fault_detection_parameters,
    explain_single_ended_status,
    explain_two_ended_quality,
    explain_high_resistance_result,
    explain_sync_warning,
)
from case_storage import (
    DEFAULT_CASE_DRIVE_FOLDER_URL,
    get_config_secret,
    parse_runtime_credentials_upload,
    apply_runtime_credentials,
    extract_google_drive_folder_id,
    build_case_archive_bytes,
    restore_case_archive,
    get_restored_upload,
    upload_case_archive_to_drive,
)


DEFAULT_TOWER_SCHEDULE_URL = ""
DEFAULT_TOWER_SCHEDULE_SHEET = "tower_schedule"


from app_runtime import (
    read_comtrade_cached,
    read_google_spreadsheet_table_cached,
    read_google_spreadsheet_query_cached,
    get_google_spreadsheet_sheet_names_cached,
    install_print_friendly_tables,
)


from summary_helpers import (
    choose_summary_fault_signals,
    build_summary_focus_waveform,
    estimate_summary_disturbance_cause,
    single_ended_plot_score,
    build_summary_location_plot,
)


def build_summary_line_position_from_session():
    line_param = st.session_state.get("line_param")
    two_result = st.session_state.get("two_ended_result")
    single_result = st.session_state.get("two_ended_local_single_result") or st.session_state.get("single_ended_result")
    remote_single_result = st.session_state.get("two_ended_remote_single_result")

    if not line_param or not (single_result or remote_single_result or two_result):
        return None

    if two_result and two_result.get("line_length_km_used"):
        line_param = override_line_param_length(
            line_param,
            float(two_result["line_length_km_used"]),
            str(two_result.get("line_length_source", "Two-Ended calculation")),
        )

    fallback_local, fallback_remote = infer_gi_names_from_line_name(
        str(line_param.get("line_name") or "")
    )

    return build_summary_location_plot(
        line_param=line_param,
        local_gi_label=st.session_state.get("two_ended_local_gi_label", fallback_local),
        remote_gi_label=st.session_state.get("two_ended_remote_gi_label", fallback_remote),
        single_result=single_result,
        remote_single_result=remote_single_result,
        two_result=two_result,
        reverse_two_result=st.session_state.get("two_ended_reverse_result"),
    )


from line_analysis_helpers import (
    is_reverse_or_backfeed_scenario,
    build_remote_single_signed_position,
    classify_two_ended_operating_status,
    get_index_at_time,
    calculate_remote_aligned_dft_index,
    choose_best_remote_dft_for_two_ended,
    score_two_ended_for_local_search,
    clean_gi_name,
    infer_gi_names_from_line_name,
    reverse_line_name,
    orient_remote_as_line_current,
    build_two_ended_reverse_result,
    build_two_ended_comparison_dataframe,
    override_line_param_length,
)


def select_effective_line_param_for_calculation(line_param: dict, key_prefix: str):
    tower_length_km = st.session_state.get("tower_schedule_selected_length_km")
    tower_length_source = st.session_state.get("tower_schedule_selected_length_source", "Tower Schedule")
    length_source_options = ["line_parameter"]
    if tower_length_km is not None:
        length_source_options.append("tower_schedule")

    current_source = st.session_state.get(f"{key_prefix}_line_length_source", "line_parameter")
    if current_source not in length_source_options:
        current_source = "line_parameter"

    selected_source = st.selectbox(
        "Sumber panjang line untuk perhitungan",
        length_source_options,
        index=length_source_options.index(current_source),
        format_func=lambda value: {
            "line_parameter": f"Line Parameter ({line_param['length_km']:.6f} km)",
            "tower_schedule": (
                f"Tower Schedule ({float(tower_length_km):.6f} km - {tower_length_source})"
                if tower_length_km is not None
                else "Tower Schedule belum tersedia"
            ),
        }[value],
        key=f"{key_prefix}_line_length_source",
        help=(
            "Pilih Tower Schedule jika panjang saluran dari tabel tower lebih akurat. "
            "Data Tower Schedule harus dimuat dan difilter dahulu."
        ),
    )

    if selected_source == "tower_schedule" and tower_length_km is not None:
        effective_line_param = override_line_param_length(
            line_param,
            float(tower_length_km),
            f"Tower Schedule - {tower_length_source}",
        )
        st.info(
            f"Perhitungan memakai panjang Tower Schedule: {effective_line_param['length_km']:.6f} km. "
            "Z1_total/Z0_total dihitung ulang dari impedansi per km."
        )
        return effective_line_param

    effective_line_param = dict(line_param)
    effective_line_param["length_source"] = "Line Parameter"
    if tower_length_km is None:
        st.caption(
            "Panjang Tower Schedule belum tersedia. Load dan filter data di tab Tower Schedule "
            "jika ingin memakai panjang saluran dari tower."
        )
    return effective_line_param



from weather_services import (
    THUNDERSTORM_WEATHER_CODES,
    weather_code_label,
    translate_weather_description,
    safe_number_formatter,
    safe_display_number,
    get_openweather_lightning_api_key,
    get_xweather_credentials,
    get_accuweather_api_key,
    fetch_accuweather_current_weather,
    fetch_openweather_onecall_current_weather,
    fetch_openweather_onecall_15min_forecast,
    build_openweather_forecast_summary,
    normalize_event_time_for_api,
    format_openweather_time,
    calculate_haversine_km,
    parse_api_datetime,
    fetch_openweather_lightning_events,
    build_openweather_lightning_dataframe,
    fetch_xweather_lightning_flash_closest,
    build_xweather_lightning_dataframe,
    fetch_accuweather_lightning_radius,
    build_accuweather_lightning_dataframe,
    fetch_open_meteo_current_weather,
    fetch_open_meteo_recent_thunderstorm,
)

def get_summary_fault_event_time(mode: str):
    local_metadata = st.session_state.get("local_metadata", {})
    local_fault_window = st.session_state.get("fault_window")
    if not local_fault_window:
        return None
    return get_absolute_event_time(
        local_metadata,
        float(local_fault_window.get("fault_time", 0.0)),
        mode,
    )

from weather_ui import weather_card_html
from tower_map import (
    compact_tower_span_label,
    get_selected_fault_location_option,
    prepare_tower_map_dataframe,
    get_fault_location_map_options,
    interpolate_tower_path_location,
    get_fault_tower_segment,
    render_tower_map,
)



def render_fault_weather_lightning_summary(tower_df: pd.DataFrame, key_prefix: str = "summary_weather_lightning"):
    map_df = prepare_tower_map_dataframe(tower_df)
    selected_fault_option = get_selected_fault_location_option("summary_tower_fault")
    if map_df.empty or not selected_fault_option:
        return
    fault_location, fault_location_warning = interpolate_tower_path_location(
        map_df,
        selected_fault_option["distance_km"],
    )
    if not fault_location:
        st.info(f"Data cuaca titik gangguan belum dapat ditampilkan: {fault_location_warning}")
        return

    st.markdown("### Cuaca Terkini")
    st.caption(
        "Data cuaca diambil pada titik lokasi gangguan hasil interpolasi Tower Schedule. "
        "Data ini adalah kondisi cuaca dan prakiraan hujan dari OpenWeather."
    )
    openweather_key_source = get_openweather_lightning_api_key()
    with st.expander("Pengaturan cuaca", expanded=False):
        openweather_key_input = st.text_input(
            "OpenWeather API key",
            value=openweather_key_source,
            type="password",
            key=f"{key_prefix}_openweather_api_key_input",
            help=(
                "Aplikasi selalu memakai OpenWeather One Call 4.0 jika API key tersedia. "
                "Jika key kosong atau gagal, Open-Meteo dipakai sebagai fallback."
            ),
        ).strip()
        st.caption("Sumber default: OpenWeather One Call 4.0. Fallback otomatis: Open-Meteo.")
    openweather_key = openweather_key_input or openweather_key_source
    if openweather_key:
        st.session_state["openweather_lightning_api_key"] = openweather_key

    fault_lat, fault_lon, fault_cum_km = fault_location
    if openweather_key:
        current = fetch_openweather_onecall_current_weather(fault_lat, fault_lon, openweather_key)
        if current.get("error"):
            st.warning(f"OpenWeather One Call 4.0 gagal dibaca, memakai Open-Meteo fallback: {current['error']}")
            current = fetch_open_meteo_current_weather(fault_lat, fault_lon)
            forecast_summary = {
                "available": False,
                "summary": "Forecast One Call 4.0 tidak tersedia saat fallback aktif.",
                "items": [],
                "thunder_count": 0,
                "rain_count": 0,
                "max_pop": None,
                "total_precip_mm": 0.0,
            }
        else:
            forecast_payload = fetch_openweather_onecall_15min_forecast(fault_lat, fault_lon, openweather_key)
            if forecast_payload.get("error"):
                forecast_summary = {
                    "available": False,
                    "summary": f"Forecast One Call 4.0 belum dapat dibaca: {forecast_payload['error']}",
                    "items": [],
                    "thunder_count": 0,
                    "rain_count": 0,
                    "max_pop": None,
                    "total_precip_mm": 0.0,
                }
            else:
                forecast_summary = build_openweather_forecast_summary(forecast_payload, hours=12)
    else:
        current = fetch_open_meteo_current_weather(fault_lat, fault_lon)
        forecast_summary = {
            "available": False,
            "summary": "Isi OpenWeather API key untuk forecast 15 menit One Call 4.0.",
            "items": [],
            "thunder_count": 0,
            "rain_count": 0,
            "max_pop": None,
            "total_precip_mm": 0.0,
        }

    current_summary = (
        f"Gagal baca cuaca: {current['error']}"
        if current.get("error")
        else translate_weather_description(current.get("weather", "-"))
    )

    fault_segment = get_fault_tower_segment(map_df, selected_fault_option["distance_km"])
    if fault_segment:
        fault_label = (
            f"Span {compact_tower_span_label(fault_segment['prev'].get('SPAN', '-'))} - "
            f"{compact_tower_span_label(fault_segment['next'].get('SPAN', '-'))}"
        )
    else:
        fault_label = selected_fault_option["label"]

    weather_rows = [
        {
            "Location": f"Titik Gangguan - {selected_fault_option['label']}",
            "Tower": fault_label,
            "Distance from Fault km": 0.0,
            "Cumulative km": fault_cum_km,
            "Latitude": fault_lat,
            "Longitude": fault_lon,
            "Current Weather": current_summary,
            "Weather Code": current.get("weather_code"),
            "Weather Icon URL": current.get("weather_icon_url"),
            "Temperature C": current.get("temperature_c"),
            "Feels Like C": current.get("feels_like_c"),
            "Visibility m": current.get("visibility_m"),
            "Humidity %": current.get("humidity_pct"),
            "Rain mm": current.get("rain_mm"),
            "Precipitation mm": current.get("precipitation_mm"),
            "Cloud Cover %": current.get("cloud_cover_pct"),
            "Wind km/h": current.get("wind_speed_kmh"),
            "Wind Dir deg": current.get("wind_direction_deg"),
            "Weather Time": current.get("time"),
            "Weather Source": current.get("source", "Open-Meteo"),
            "Forecast Summary": forecast_summary,
        }
    ]

    weather_html = weather_card_html(weather_rows)
    if hasattr(st, "html"):
        st.html(weather_html)
    else:
        components.html(weather_html, height=760, scrolling=False)


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

    div[data-testid="stTabs"] [role="tablist"],
    div[data-testid="stTabs"] [data-baseweb="tab-list"] {
        background: #ffffff !important;
        border-bottom: none !important;
        box-shadow: none !important;
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
        div[data-testid="stTabs"] > div:first-child,
        div[data-testid="stTabs"] > div[role="tablist"],
        div[data-testid="stTabs"] div[role="tablist"],
        div[data-testid="stTabs"] [data-baseweb="tab-list"],
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

st.sidebar.header("Upload Local End COMTRADE")

cfg_file = st.sidebar.file_uploader("Local .cfg", key="local_cfg_file")
dat_file = st.sidebar.file_uploader("Local .dat", key="local_dat_file")

st.sidebar.divider()
st.sidebar.header("Upload Remote End COMTRADE")
st.sidebar.caption("Opsional. Diisi jika ingin menghitung double-ended.")
remote_cfg_file = st.sidebar.file_uploader("Remote .cfg", key="remote_cfg_file")
remote_dat_file = st.sidebar.file_uploader("Remote .dat", key="remote_dat_file")

st.sidebar.divider()
st.sidebar.header("Case Storage")
case_archive_file = st.sidebar.file_uploader("Load Case (.zip)", type=["zip"], key="case_archive_file")
if case_archive_file is not None:
    try:
        restore_case_archive(case_archive_file.getvalue())
        st.rerun()
    except Exception as e:
        st.sidebar.error("Case gagal dimuat.")
        st.sidebar.exception(e)

cfg_file = cfg_file or get_restored_upload("local_cfg")
dat_file = dat_file or get_restored_upload("local_dat")
remote_cfg_file = remote_cfg_file or get_restored_upload("remote_cfg")
remote_dat_file = remote_dat_file or get_restored_upload("remote_dat")

if st.session_state.get("case_restore_message"):
    st.sidebar.success(st.session_state.pop("case_restore_message"))

if not validate_uploaded_extension(cfg_file, ".cfg", "File local CFG"):
    st.stop()

if not validate_uploaded_extension(dat_file, ".dat", "File local DAT"):
    st.stop()
if cfg_file is None or dat_file is None:
    summary_empty_container = st.container()
    with summary_empty_container:
        st.subheader("Summary / Report Ringkas")
        local_upload_status = "Uploaded" if cfg_file is not None or dat_file is not None else "Not uploaded"
        remote_upload_status = "Uploaded" if remote_cfg_file is not None or remote_dat_file is not None else "Not uploaded"
        col_empty_sum1, col_empty_sum2, col_empty_sum3 = st.columns(3)
        col_empty_sum1.metric("Local Record", local_upload_status)
        col_empty_sum2.metric("Remote Record", remote_upload_status)
        col_empty_sum3.metric("Calculation", "Pending")
        if cfg_file is not None and dat_file is None:
            st.warning("File local CFG sudah diupload, tetapi file local DAT belum tersedia.")
        elif cfg_file is None and dat_file is not None:
            st.warning("File local DAT sudah diupload, tetapi file local CFG belum tersedia.")
        elif remote_cfg_file is not None or remote_dat_file is not None:
            st.info(
                "Rekaman remote sudah terdeteksi. Upload pasangan file local jika ingin menjalankan workflow "
                "fault locator utama."
            )
        else:
            st.info("Upload pasangan file COMTRADE local .cfg dan .dat untuk mulai menghitung fault locator.")
        st.markdown("### Yang akan tampil setelah data tersedia")
        st.write(
            "Tabel pre-fault/fault GI local dan GI remote, waveform fokus fault, "
            "estimasi penyebab gangguan, serta grafik SE/DE."
        )
    st.stop()

local_cfg_bytes = cfg_file.getvalue()
local_dat_bytes = dat_file.getvalue()
st.session_state["case_local_cfg_name"] = cfg_file.name
st.session_state["case_local_cfg_bytes"] = local_cfg_bytes
st.session_state["case_local_dat_name"] = dat_file.name
st.session_state["case_local_dat_bytes"] = local_dat_bytes

try:
    df, metadata = read_comtrade_cached(
        local_cfg_bytes,
        local_dat_bytes,
        cfg_file.name,
        dat_file.name,
    )

    auto_assignment = detect_voltage_current_channels(df, metadata)
    auto_transformer_data = get_auto_transformer_data(metadata)
    auto_recorded_side = detect_recorded_side(metadata)

    st.session_state["local_metadata"] = metadata
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

st.caption(
    "Workflow ringkas: upload local/remote di panel kiri -> siapkan Local End dan Remote End "
    "-> isi Line -> hitung Single-End atau Double-End."
)

tab_summary, tab0, tab_tower, tab_local, tab_remote, tab7, tab8, tab9, tab10, tab11 = st.tabs(
    [
        "Summary",
        "Setup DB",
        "Tower Schedule",
        "Local End",
        "Remote End",
        "Line",
        "HR Check",
        "Single-End",
        "Double-End",
        "R-X Locus",
    ]
)

with tab_local:
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "Record",
            "Signals",
            "Waveform",
            "Fault Cursor",
            "Phasor",
            "Fault Type",
        ]
    )


with tab_remote:
    st.subheader("Remote End")
    remote_tab_record, remote_tab_signals, remote_tab_waveform, remote_tab_cursor, remote_tab_phasor, remote_tab_fault_type = st.tabs(
        [
            "Record",
            "Signals",
            "Waveform",
            "Fault Cursor",
            "Phasor",
            "Fault Type",
        ]
    )

    remote_end_uploaded = remote_cfg_file is not None and remote_dat_file is not None
    remote_df = None
    remote_metadata = st.session_state.get("remote_metadata", {})
    remote_auto_assignment = st.session_state.get("remote_auto_assignment", {})
    remote_auto_transformer_data = st.session_state.get("remote_auto_transformer_data", {})
    remote_auto_recorded_side = st.session_state.get("remote_auto_recorded_side", "secondary")

    if remote_end_uploaded:
        cfg_ok = validate_uploaded_extension(remote_cfg_file, ".cfg", "File remote CFG")
        dat_ok = validate_uploaded_extension(remote_dat_file, ".dat", "File remote DAT")
        if cfg_ok and dat_ok:
            try:
                remote_cfg_bytes = remote_cfg_file.getvalue()
                remote_dat_bytes = remote_dat_file.getvalue()
                st.session_state["case_remote_cfg_name"] = remote_cfg_file.name
                st.session_state["case_remote_cfg_bytes"] = remote_cfg_bytes
                st.session_state["case_remote_dat_name"] = remote_dat_file.name
                st.session_state["case_remote_dat_bytes"] = remote_dat_bytes
                remote_df, remote_metadata = read_comtrade_cached(
                    remote_cfg_bytes,
                    remote_dat_bytes,
                    remote_cfg_file.name,
                    remote_dat_file.name,
                )
                remote_auto_assignment = detect_voltage_current_channels(remote_df, remote_metadata)
                remote_auto_transformer_data = get_auto_transformer_data(remote_metadata)
                remote_auto_recorded_side = detect_recorded_side(remote_metadata)
                st.session_state["remote_metadata"] = remote_metadata
                st.session_state["remote_auto_assignment"] = remote_auto_assignment
                st.session_state["remote_auto_transformer_data"] = remote_auto_transformer_data
                st.session_state["remote_auto_recorded_side"] = remote_auto_recorded_side
            except Exception as e:
                remote_df = None
                with remote_tab_record:
                    st.error("Remote end COMTRADE gagal dibaca.")
                    st.exception(e)

    with remote_tab_record:
        st.markdown("### Remote Record")
        if not remote_end_uploaded:
            st.info("Upload pasangan file .cfg dan .dat remote pada panel kiri.")
        elif remote_df is not None:
            st.success("Remote end COMTRADE berhasil dibaca.")
            col_ram1, col_ram2, col_ram3, col_ram4 = st.columns(4)
            col_ram1.metric("Remote CFG Start Time", str(remote_metadata.get("cfg_start_time") or "-"))
            col_ram2.metric("Remote CFG Trigger Time", str(remote_metadata.get("cfg_trigger_time") or "-"))
            col_ram3.metric("Remote VT Ratio from CFG", str(remote_metadata.get("vt_ratio_from_cfg") or "-"))
            col_ram4.metric("Remote CT Ratio from CFG", str(remote_metadata.get("ct_ratio_from_cfg") or "-"))
            st.write("Analog Channels:")
            st.write(remote_metadata.get("analog_channels", []))
            with st.expander("Remote Analog Metadata dari .cfg"):
                st.dataframe(pd.DataFrame(remote_metadata.get("analog_metadata", [])), use_container_width=True)
            st.subheader("Preview Data Original Remote")
            st.dataframe(remote_df.head(20), use_container_width=True)

    with remote_tab_signals:
        st.markdown("### Remote Signal Assignment")
        if remote_df is None:
            st.info("Upload dan baca rekaman remote terlebih dahulu.")
        else:
            with st.expander("Remote Auto Signal Assignment Preview", expanded=False):
                st.dataframe(
                    build_auto_assignment_summary(remote_auto_assignment, remote_auto_transformer_data, remote_metadata),
                    use_container_width=True,
                )

            remote_channel_options = [col for col in remote_df.columns if col != "time"]
            remote_ground_options = ["None"] + remote_channel_options

            def get_remote_channel_index(channel_name, options, default_index=0):
                return options.index(channel_name) if channel_name in options else default_index

            def get_remote_ground_index(channel_name, options):
                return options.index(channel_name) if channel_name in options else 0

            col_rv1, col_rv2, col_rv3 = st.columns(3)
            with col_rv1:
                remote_va_channel = st.selectbox(
                    "Remote Va / VL1",
                    remote_channel_options,
                    index=get_remote_channel_index(remote_auto_assignment.get("Va"), remote_channel_options, 0),
                    key="remote_va_channel",
                )
            with col_rv2:
                remote_vb_channel = st.selectbox(
                    "Remote Vb / VL2",
                    remote_channel_options,
                    index=get_remote_channel_index(remote_auto_assignment.get("Vb"), remote_channel_options, 1 if len(remote_channel_options) > 1 else 0),
                    key="remote_vb_channel",
                )
            with col_rv3:
                remote_vc_channel = st.selectbox(
                    "Remote Vc / VL3",
                    remote_channel_options,
                    index=get_remote_channel_index(remote_auto_assignment.get("Vc"), remote_channel_options, 2 if len(remote_channel_options) > 2 else 0),
                    key="remote_vc_channel",
                )

            col_ri1, col_ri2, col_ri3 = st.columns(3)
            with col_ri1:
                remote_ia_channel = st.selectbox(
                    "Remote Ia / IL1",
                    remote_channel_options,
                    index=get_remote_channel_index(remote_auto_assignment.get("Ia"), remote_channel_options, 3 if len(remote_channel_options) > 3 else 0),
                    key="remote_ia_channel",
                )
            with col_ri2:
                remote_ib_channel = st.selectbox(
                    "Remote Ib / IL2",
                    remote_channel_options,
                    index=get_remote_channel_index(remote_auto_assignment.get("Ib"), remote_channel_options, 4 if len(remote_channel_options) > 4 else 0),
                    key="remote_ib_channel",
                )
            with col_ri3:
                remote_ic_channel = st.selectbox(
                    "Remote Ic / IL3",
                    remote_channel_options,
                    index=get_remote_channel_index(remote_auto_assignment.get("Ic"), remote_channel_options, 5 if len(remote_channel_options) > 5 else 0),
                    key="remote_ic_channel",
                )

            remote_ie_channel = st.selectbox(
                "Remote IE / IN / 3I0 jika tersedia",
                remote_ground_options,
                index=get_remote_ground_index(remote_auto_assignment.get("IE"), remote_ground_options),
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
                    step=0.001,
                    format="%.5f",
                    key="remote_ct_primary",
                )
            with col_rct2:
                remote_ct_secondary = st.number_input(
                    "Remote CT Secondary (A)",
                    value=float(remote_auto_transformer_data.get("ct_secondary", 1.0)),
                    min_value=0.001,
                    step=0.001,
                    format="%.5f",
                    key="remote_ct_secondary",
                )
            with col_rvt1:
                remote_vt_primary = st.number_input(
                    "Remote VT Primary (V)",
                    value=float(remote_auto_transformer_data.get("vt_primary", 150000.0)),
                    min_value=0.001,
                    step=0.001,
                    format="%.5f",
                    key="remote_vt_primary",
                )
            with col_rvt2:
                remote_vt_secondary = st.number_input(
                    "Remote VT Secondary (V)",
                    value=float(remote_auto_transformer_data.get("vt_secondary", 100.0)),
                    min_value=0.001,
                    step=0.001,
                    format="%.5f",
                    key="remote_vt_secondary",
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
            st.session_state["remote_transformer_data"] = {
                "recorded_side": remote_recorded_side,
                "ct_primary": remote_ct_primary,
                "ct_secondary": remote_ct_secondary,
                "vt_primary": remote_vt_primary,
                "vt_secondary": remote_vt_secondary,
                "nominal_phase_voltage_rms": remote_vt_primary / math.sqrt(3.0),
                "nominal_current_rms": remote_ct_primary,
            }
            st.session_state["remote_ie_selected_channel"] = remote_ie_channel if remote_ie_channel != "None" else None
            st.session_state["remote_ie_source"] = (
                "measured" if remote_ie_channel != "None" else "calculated_from_3_phase_currents"
            )
            st.success("Remote signal assignment berhasil dibuat.")
            st.dataframe(remote_assigned_df.head(20), use_container_width=True)

    with remote_tab_waveform:
        st.markdown("### Remote Waveform")
        remote_assigned_df = st.session_state.get("remote_assigned_df")
        if remote_assigned_df is None:
            st.info("Selesaikan Remote End > Signals terlebih dahulu.")
        else:
            remote_signal_groups = {
                "Tegangan 3 Fasa": ["Va", "Vb", "Vc"],
                "Arus 3 Fasa": ["Ia", "Ib", "Ic"],
                "Ground Current": ["IE", "I0"],
                "Semua": ["Va", "Vb", "Vc", "Ia", "Ib", "Ic", "IE", "I0"],
            }
            remote_waveform_group = st.selectbox(
                "Pilih kelompok sinyal remote",
                list(remote_signal_groups.keys()),
                key="remote_assigned_waveform_group",
            )
            remote_waveform_channels = [
                channel for channel in remote_signal_groups[remote_waveform_group]
                if channel in remote_assigned_df.columns
            ]
            remote_waveform_display_mode = st.radio(
                "Mode tampilan waveform remote",
                ["Instantaneous / peak", "RMS 1 siklus"],
                horizontal=True,
                key="remote_assigned_waveform_display_mode",
            )
            remote_waveform_frequency = float(remote_metadata.get("frequency") or 50.0)
            remote_rms_summary_df = build_waveform_rms_summary(
                remote_assigned_df,
                remote_waveform_channels,
                frequency=remote_waveform_frequency,
            )
            if not remote_rms_summary_df.empty:
                with st.expander("Remote RMS vs Peak Awal Rekaman", expanded=False):
                    st.dataframe(
                        remote_rms_summary_df.style.format(
                            {
                                "RMS Awal Rekaman": "{:.3f}",
                                "Peak Absolut Awal": "{:.3f}",
                                "Peak/RMS": "{:.3f}",
                            }
                        ),
                        use_container_width=True,
                    )
            if remote_waveform_channels:
                remote_assigned_fig, remote_waveform_caption = build_assigned_waveform_plot(
                    remote_assigned_df,
                    remote_waveform_channels,
                    f"Remote Waveform {remote_waveform_group} - {remote_waveform_display_mode}",
                    remote_waveform_display_mode,
                    frequency=remote_waveform_frequency,
                )
                st.caption(remote_waveform_caption)
                st.plotly_chart(remote_assigned_fig, use_container_width=True)

    with remote_tab_cursor:
        st.markdown("### Remote Fault Cursor")
        remote_assigned_df = st.session_state.get("remote_assigned_df")
        if remote_assigned_df is None:
            st.info("Selesaikan Remote End > Signals terlebih dahulu.")
        else:
            remote_transformer_data = st.session_state.get("remote_transformer_data", {})
            col_rf1, col_rf2, col_rf3 = st.columns(3)
            with col_rf1:
                remote_frequency = st.number_input(
                    "Remote Frequency (Hz)",
                    value=float(remote_metadata.get("frequency") or 50.0),
                    min_value=40.0,
                    max_value=70.0,
                    step=0.001,
                    format="%.5f",
                    key="remote_frequency",
                )
            remote_auto_fault_detection_settings = calculate_auto_fault_detection_parameters(
                remote_assigned_df,
                frequency=remote_frequency,
                pre_fault_cycles=2,
                nominal_phase_voltage_rms=remote_transformer_data.get("nominal_phase_voltage_rms"),
                nominal_current_rms=remote_transformer_data.get("nominal_current_rms"),
            )
            use_remote_auto_fault_detection = st.checkbox(
                "Gunakan deteksi otomatis adaptif remote nominal + pre-fault",
                value=False,
                key="use_remote_auto_fault_detection",
            )
            with col_rf2:
                remote_current_multiplier = st.number_input(
                    "Remote Current Fault Multiplier",
                    value=float(remote_auto_fault_detection_settings["current_threshold_multiplier"]),
                    min_value=1.01,
                    max_value=10.0,
                    step=0.001,
                    format="%.5f",
                    key="remote_current_multiplier",
                    disabled=use_remote_auto_fault_detection,
                )
            with col_rf3:
                remote_voltage_threshold = st.number_input(
                    "Remote Voltage Drop Threshold",
                    value=float(remote_auto_fault_detection_settings["voltage_drop_threshold"]),
                    min_value=0.1,
                    max_value=1.0,
                    step=0.0001,
                    format="%.5f",
                    key="remote_voltage_threshold",
                    disabled=use_remote_auto_fault_detection,
                )
            if use_remote_auto_fault_detection:
                remote_current_multiplier = remote_auto_fault_detection_settings["current_threshold_multiplier"]
                remote_voltage_threshold = remote_auto_fault_detection_settings["voltage_drop_threshold"]
            remote_detection = detect_fault_inception(
                remote_assigned_df,
                frequency=remote_frequency,
                current_threshold_multiplier=remote_current_multiplier,
                voltage_drop_threshold=remote_voltage_threshold,
                min_prefault_cycles=2,
                adaptive_threshold_sigma=(
                    remote_auto_fault_detection_settings["adaptive_threshold_sigma"]
                    if use_remote_auto_fault_detection
                    else None
                ),
                refine_fault_bar=use_remote_auto_fault_detection,
                method=(
                    remote_auto_fault_detection_settings["fault_detection_method"]
                    if use_remote_auto_fault_detection
                    else "legacy_rms"
                ),
                superimposed_threshold_sigma=remote_auto_fault_detection_settings["superimposed_threshold_sigma"],
                nominal_phase_voltage_rms=remote_transformer_data.get("nominal_phase_voltage_rms"),
                nominal_current_rms=remote_transformer_data.get("nominal_current_rms"),
            )
            st.session_state["remote_fault_detection"] = remote_detection
            if not remote_detection["detected"]:
                st.warning("Remote fault inception tidak terdeteksi otomatis. Gunakan slider manual.")
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
                remote_fault_index = int((remote_assigned_df["time"] - manual_remote_fault_time).abs().idxmin())
                remote_samples_per_cycle = int(round(estimate_sampling_rate(remote_assigned_df) / remote_frequency))
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
            st.session_state["remote_samples_per_cycle"] = remote_samples_per_cycle
            st.session_state["remote_frequency_hz"] = remote_frequency
            col_rw1, col_rw2, col_rw3 = st.columns(3)
            col_rw1.metric("Remote Fault Time", f'{remote_fault_window["fault_time"]:.6f} s')
            col_rw2.metric("Remote DFT Time", f'{remote_fault_window["dft_time"]:.6f} s')
            col_rw3.metric("Remote Samples/Cycle", remote_samples_per_cycle)
            remote_plot_channels = [
                channel for channel in ["Va", "Vb", "Vc", "Ia", "Ib", "Ic", "IE"]
                if channel in remote_assigned_df.columns
            ]
            remote_selected_plot = st.multiselect(
                "Pilih sinyal remote untuk validasi fault window",
                remote_plot_channels,
                default=[channel for channel in ["Ia", "Ib", "Ic"] if channel in remote_plot_channels] or remote_plot_channels[:3],
                key="remote_fault_window_plot_channels",
            )
            if remote_selected_plot:
                remote_fault_fig = build_fault_window_plot(
                    remote_assigned_df,
                    remote_fault_window,
                    remote_selected_plot,
                    "Remote Fault Detection dan Cursor Window",
                )
                st.plotly_chart(remote_fault_fig, use_container_width=True)

    with remote_tab_phasor:
        st.markdown("### Remote Phasor")
        remote_assigned_df = st.session_state.get("remote_assigned_df")
        remote_fault_window = st.session_state.get("remote_fault_window")
        remote_samples_per_cycle = st.session_state.get("remote_samples_per_cycle")
        if remote_assigned_df is None or remote_fault_window is None or remote_samples_per_cycle is None:
            st.info("Selesaikan Remote End > Fault Cursor terlebih dahulu.")
        else:
            remote_phasors = calculate_all_phasors(
                df=remote_assigned_df,
                cursor_index=remote_fault_window["dft_index"],
                samples_per_cycle=int(remote_samples_per_cycle),
            )
            remote_phasors = add_sequence_components_to_phasor_dict(remote_phasors)
            st.session_state["remote_phasors"] = remote_phasors
            st.session_state["two_ended_remote_phasors_for_calculation"] = remote_phasors
            st.session_state["two_ended_remote_dft_index_for_calculation"] = remote_fault_window["dft_index"]
            st.session_state["two_ended_remote_sync_shift_s"] = 0.0
            st.session_state["two_ended_remote_sync_score"] = 0.0
            st.session_state["two_ended_remote_sync_reference"] = "fault_cursor"
            try:
                remote_prefault_phasors = calculate_all_phasors(
                    df=remote_assigned_df,
                    cursor_index=remote_fault_window["fault_index"],
                    samples_per_cycle=int(remote_samples_per_cycle),
                )
                remote_prefault_phasors = add_sequence_components_to_phasor_dict(remote_prefault_phasors)
                st.session_state["remote_prefault_phasors"] = remote_prefault_phasors
            except Exception:
                st.session_state.pop("remote_prefault_phasors", None)
            st.dataframe(
                build_phasor_dataframe(remote_phasors).style.format(
                    {
                        "Magnitude RMS": "{:.4f}",
                        "Angle Deg": "{:.2f}",
                        "Real": "{:.4f}",
                        "Imag": "{:.4f}",
                    }
                ),
                use_container_width=True,
            )

    with remote_tab_fault_type:
        st.markdown("### Remote Fault Type")
        remote_phasors = st.session_state.get("remote_phasors")
        if remote_phasors is None:
            st.info("Selesaikan Remote End > Phasor terlebih dahulu.")
        else:
            remote_auto_fault_settings = calculate_auto_fault_type_thresholds(
                remote_phasors,
                st.session_state.get("remote_prefault_phasors"),
            )
            use_remote_auto_fault_type_thresholds = st.toggle(
                "Gunakan threshold otomatis remote dari kondisi pre-fault",
                value=True,
                key="use_remote_auto_fault_type_thresholds",
            )
            col_rftp1, col_rftp2, col_rftp3 = st.columns(3)
            with col_rftp1:
                remote_fault_type_voltage_drop_threshold = st.number_input(
                    "Remote Fault Type Voltage Drop Threshold",
                    value=0.80,
                    min_value=0.10,
                    max_value=1.00,
                    step=0.0001,
                    format="%.5f",
                    key="remote_fault_type_voltage_drop_threshold",
                )
            with col_rftp2:
                remote_fault_type_current_rise_threshold = st.number_input(
                    "Remote Fault Type Current Rise Threshold",
                    value=1.50,
                    min_value=1.05,
                    max_value=10.00,
                    step=0.0001,
                    format="%.5f",
                    key="remote_fault_type_current_rise_threshold",
                )
            with col_rftp3:
                remote_fault_type_ground_current_threshold = st.number_input(
                    "Remote Fault Type Ground Current Threshold",
                    value=0.20,
                    min_value=0.01,
                    max_value=1.00,
                    step=0.0001,
                    format="%.5f",
                    key="remote_fault_type_ground_current_threshold",
                )
            with st.expander("Remote Advanced Resistive Fault / Delta Detection"):
                col_rftd1, col_rftd2 = st.columns(2)
                with col_rftd1:
                    remote_delta_current_threshold = st.number_input(
                        "Remote Delta Current Dominance Threshold",
                        value=0.45,
                        min_value=0.05,
                        max_value=1.00,
                        step=0.0001,
                        format="%.5f",
                        key="remote_delta_current_threshold",
                    )
                with col_rftd2:
                    remote_delta_voltage_threshold = st.number_input(
                        "Remote Delta Voltage Threshold",
                        value=0.01,
                        min_value=0.0001,
                        max_value=0.20,
                        step=0.0001,
                        format="%.5f",
                        key="remote_delta_voltage_threshold",
                    )
            if use_remote_auto_fault_type_thresholds:
                remote_fault_type_voltage_drop_threshold = remote_auto_fault_settings["voltage_drop_threshold"]
                remote_fault_type_current_rise_threshold = remote_auto_fault_settings["current_rise_threshold"]
                remote_fault_type_ground_current_threshold = remote_auto_fault_settings["ground_current_threshold"]
                remote_delta_current_threshold = remote_auto_fault_settings["delta_current_threshold"]
                remote_delta_voltage_threshold = remote_auto_fault_settings["delta_voltage_threshold"]
            remote_fault_type_result = detect_fault_type(
                phasors=remote_phasors,
                prefault_phasors=st.session_state.get("remote_prefault_phasors"),
                voltage_drop_threshold=remote_fault_type_voltage_drop_threshold,
                current_rise_threshold=remote_fault_type_current_rise_threshold,
                ground_current_threshold=remote_fault_type_ground_current_threshold,
                delta_current_threshold=remote_delta_current_threshold,
                delta_voltage_threshold=remote_delta_voltage_threshold,
            )
            remote_fault_type_result["auto_thresholds"] = remote_auto_fault_settings
            remote_fault_type_result["threshold_mode"] = "auto_prefault" if use_remote_auto_fault_type_thresholds else "manual"
            remote_fault_type_df = build_fault_type_metrics_dataframe(remote_fault_type_result)
            st.session_state["remote_fault_type_result"] = remote_fault_type_result
            st.session_state["remote_fault_type_df"] = remote_fault_type_df
            col_rft1, col_rft2, col_rft3, col_rft4 = st.columns(4)
            col_rft1.metric("Remote Fault Type", remote_fault_type_result["fault_type"])
            col_rft2.metric("Ground Involved", "Yes" if remote_fault_type_result["ground_involved"] else "No")
            col_rft3.metric("Confidence", f'{remote_fault_type_result["confidence"]}/10')
            col_rft4.metric(
                "Faulted Phases",
                ", ".join(remote_fault_type_result["faulted_phases"])
                if remote_fault_type_result["faulted_phases"]
                else "-",
            )
            st.info(explain_fault_type_result(remote_fault_type_result, context="Rekaman remote"))
            with st.expander("Remote Fault Type Detection Detail"):
                st.dataframe(remote_fault_type_df, use_container_width=True)

with tab_summary:
    summary_container = st.container()

from rx_locus import (
    normalize_locus_fault_type,
    parse_distance_setting_number,
    detect_locus_distance_setting_columns,
    sorted_nonempty_values,
    build_locus_setting_row_labels,
    extract_locus_zone_settings,
    impedance_secondary_scale_from_transformer,
    scale_locus_zone_settings,
    add_locus_zone_overlay,
    build_simple_rx_locus_trajectory,
)


def get_rx_locus_context_from_session(end_side: str):
    if "line_param" not in st.session_state:
        return None, "Silakan lakukan Line Parameter terlebih dahulu."

    line_param = st.session_state["line_param"]
    local_label, remote_label = infer_gi_names_from_line_name(line_param.get("line_name", ""))
    if end_side == "local":
        assigned_df = st.session_state.get("assigned_df")
        fault_window = st.session_state.get("fault_window")
        detection = st.session_state.get("fault_detection", {})
        fault_type_result = st.session_state.get("fault_type_result", {})
        transformer_data = st.session_state.get("local_transformer_data", {})
        label = local_label
    else:
        assigned_df = st.session_state.get("remote_assigned_df")
        fault_window = st.session_state.get("remote_fault_window")
        detection = st.session_state.get("remote_fault_detection", {})
        fault_type_result = st.session_state.get("remote_fault_type_result", {})
        transformer_data = st.session_state.get("remote_transformer_data", {})
        label = remote_label

    if assigned_df is None:
        return None, f"Selesaikan {'Local' if end_side == 'local' else 'Remote'} End > Signals terlebih dahulu."
    if fault_window is None:
        return None, f"Selesaikan {'Local' if end_side == 'local' else 'Remote'} End > Fault Cursor terlebih dahulu."

    frequency = float(detection.get("frequency", 50.0) or 50.0)
    samples_per_cycle = int(
        st.session_state.get(
            "remote_samples_per_cycle" if end_side == "remote" else "local_samples_per_cycle",
            detection.get("samples_per_cycle") or max(1, round(estimate_sampling_rate(assigned_df) / frequency)),
        )
    )

    default_loop = normalize_locus_fault_type(fault_type_result.get("fault_type", "AG"))
    if default_loop in ["ABG"]:
        default_loop = "AB"
    elif default_loop in ["BCG"]:
        default_loop = "BC"
    elif default_loop in ["CAG"]:
        default_loop = "CA"
    elif default_loop in ["ABC", "ABCG", "UNKNOWN"]:
        default_loop = "AG"

    return {
        "assigned_df": assigned_df,
        "fault_window": fault_window,
        "line_param": line_param,
        "transformer_data": transformer_data,
        "samples_per_cycle": samples_per_cycle,
        "default_loop": default_loop,
        "label": label,
    }, None


def build_locus_zone_settings_from_session(end_side: str, label: str, loop_name: str):
    if not st.session_state.get(f"rx_locus_show_zone_{end_side}", True):
        return [], {"zone_count": 0}, None

    zone_setting_base = st.session_state.get(f"rx_locus_zone_setting_base_{end_side}", "primary")
    transformer_data = st.session_state.get(
        "remote_transformer_data" if end_side == "remote" else "local_transformer_data",
        {},
    )
    secondary_scale = impedance_secondary_scale_from_transformer(transformer_data)
    if zone_setting_base == "secondary" and secondary_scale is None:
        return [], {"zone_count": 0, "zone_setting_base": zone_setting_base}, (
            "Rasio CT/VT dari Signal Assignment belum tersedia, zona secondary belum bisa dikonversi ke primary."
        )

    try:
        distance_settings_df = read_google_spreadsheet_table_cached(
            st.session_state.get("database_spreadsheet_url", ""),
            st.session_state.get("distance_settings_sheet_name", "distance_settings"),
        )
        distance_settings_df = make_streamlit_safe_columns(distance_settings_df)
    except Exception as exc:
        return [], {"zone_count": 0, "zone_setting_base": zone_setting_base}, (
            f"Setting distance relay belum dapat dibaca dari spreadsheet: {exc}"
        )

    distance_columns = detect_locus_distance_setting_columns(distance_settings_df)
    substation_col = distance_columns.get("substation")
    bay_col = distance_columns.get("bay")
    substation_options = sorted_nonempty_values(distance_settings_df, substation_col)
    substation_labels = ["Semua GI/Substation"] + substation_options
    selected_substation = st.session_state.get(f"rx_locus_substation_{end_side}")
    if selected_substation not in substation_labels:
        default_substation = label.replace("GI ", "").strip().upper()
        selected_substation = "Semua GI/Substation"
        for option in substation_labels:
            if default_substation and option.upper().replace(" ", "") == default_substation.replace(" ", ""):
                selected_substation = option
                break

    filtered_settings_df = distance_settings_df
    if selected_substation != "Semua GI/Substation" and substation_col:
        filtered_settings_df = filtered_settings_df[
            filtered_settings_df[substation_col].astype(str).str.strip() == selected_substation
        ].reset_index(drop=True)

    bay_labels = ["Semua Bay"] + sorted_nonempty_values(filtered_settings_df, bay_col)
    selected_bay = st.session_state.get(f"rx_locus_bay_{end_side}", "Semua Bay")
    if selected_bay not in bay_labels:
        selected_bay = "Semua Bay"
    if selected_bay != "Semua Bay" and bay_col:
        filtered_settings_df = filtered_settings_df[
            filtered_settings_df[bay_col].astype(str).str.strip() == selected_bay
        ].reset_index(drop=True)

    extra_filter = str(st.session_state.get(f"rx_locus_filter_{end_side}", "") or "").strip()
    if extra_filter:
        mask = filtered_settings_df.apply(
            lambda row: extra_filter.lower() in " ".join(str(value).lower() for value in row.values),
            axis=1,
        )
        filtered_settings_df = filtered_settings_df[mask].reset_index(drop=True)

    if filtered_settings_df.empty:
        return [], {
            "zone_count": 0,
            "zone_setting_base": zone_setting_base,
            "selected_substation": selected_substation,
            "selected_bay": selected_bay,
        }, "Tidak ada baris distance_settings yang cocok dengan filter."

    row_labels = build_locus_setting_row_labels(filtered_settings_df, distance_columns)
    selected_label = st.session_state.get(f"rx_locus_setting_row_{end_side}")
    if selected_label not in row_labels:
        selected_label = row_labels[0]
    selected_row = filtered_settings_df.iloc[row_labels.index(selected_label)]
    zones = extract_locus_zone_settings(selected_row, distance_columns, loop_name)
    if zone_setting_base == "secondary":
        zones = scale_locus_zone_settings(zones, 1.0 / secondary_scale)

    return zones, {
        "zone_count": int(len(zones)),
        "zone_setting_base": zone_setting_base,
        "selected_substation": selected_substation,
        "selected_bay": selected_bay,
        "selected_setting": selected_label,
    }, None


def build_rx_locus_figure_from_session(end_side: str):
    ctx, message = get_rx_locus_context_from_session(end_side)
    if message:
        return None, None, None, message

    loop_options = ["AG", "BG", "CG", "AB", "BC", "CA"]
    loop_name = st.session_state.get(f"rx_locus_loop_{end_side}", ctx["default_loop"])
    if loop_name not in loop_options:
        loop_name = ctx["default_loop"] if ctx["default_loop"] in loop_options else "AG"

    pre_cycles = float(st.session_state.get(f"rx_locus_pre_{end_side}", 2.0))
    post_cycles = float(st.session_state.get(f"rx_locus_post_{end_side}", 8.0))
    density = st.session_state.get(f"rx_locus_density_{end_side}", "1/4 cycle")
    plot_focus_mode = st.session_state.get(f"rx_locus_focus_{end_side}", "relay_zones")
    samples_per_cycle = int(ctx["samples_per_cycle"])
    step_lookup = {
        "Every sample": 1,
        "1/4 cycle": max(1, samples_per_cycle // 4),
        "1/2 cycle": max(1, samples_per_cycle // 2),
        "1 cycle": max(1, samples_per_cycle),
    }

    trajectory_df = build_simple_rx_locus_trajectory(
        assigned_df=ctx["assigned_df"],
        fault_window=ctx["fault_window"],
        samples_per_cycle=samples_per_cycle,
        line_param=ctx["line_param"],
        loop_name=loop_name,
        pre_cycles=pre_cycles,
        post_cycles=post_cycles,
        step_samples=step_lookup.get(density, max(1, samples_per_cycle // 4)),
    )
    if trajectory_df.empty:
        return None, None, None, "Trajectory R-X tidak dapat dihitung. Cek loop, window, atau sinyal yang tersedia."

    fault_time = float(ctx["fault_window"]["fault_time"])
    trajectory_df["relative_time_s"] = trajectory_df["time_s"] - fault_time
    trajectory_df["Stage"] = np.where(trajectory_df["relative_time_s"] < 0, "Pre-fault", "Fault/Post-fault")

    fig_locus = px.line(
        trajectory_df,
        x="R_ohm",
        y="X_ohm",
        color="Stage",
        markers=True,
        hover_data={"relative_time_s": ":.6f", "R_ohm": ":.4f", "X_ohm": ":.4f", "Z_mag_ohm": ":.4f", "Z_angle_deg": ":.2f"},
        title=f"R-X Locus Trajectory - {ctx['label']} - {loop_name}",
    )

    z1_total = ctx["line_param"]["Z1_total"]
    fig_locus.add_trace(
        go.Scatter(
            x=[0.0, z1_total.real],
            y=[0.0, z1_total.imag],
            mode="markers+text",
            text=["Origin", "Z Line Total"],
            textposition="top center",
            marker=dict(size=9, color="#111827"),
            name="Reference",
        )
    )
    fig_locus.add_shape(type="line", x0=0, y0=0, x1=z1_total.real, y1=z1_total.imag, line=dict(color="#111827", width=2, dash="dot"))

    dft_index = int(ctx["fault_window"].get("dft_index", ctx["fault_window"]["fault_index"]))
    dft_df = build_simple_rx_locus_trajectory(ctx["assigned_df"], ctx["fault_window"], samples_per_cycle, ctx["line_param"], loop_name, 0, 0, 1)
    dft_df = dft_df[dft_df["cursor_index"] == dft_index] if not dft_df.empty else dft_df
    if dft_df.empty:
        dft_window = {"fault_index": dft_index, "fault_time": float(ctx["assigned_df"]["time"].iloc[dft_index])}
        dft_df = build_simple_rx_locus_trajectory(ctx["assigned_df"], dft_window, samples_per_cycle, ctx["line_param"], loop_name, 0, 0, 1)
    if not dft_df.empty:
        row = dft_df.iloc[0]
        fig_locus.add_trace(
            go.Scatter(
                x=[row["R_ohm"]],
                y=[row["X_ohm"]],
                mode="markers+text",
                text=["DFT fault cursor"],
                textposition="top right",
                marker=dict(size=15, color="#ef4444", symbol="x", line=dict(width=1.5, color="#111827")),
                name="Fault cursor",
            )
        )

    locus_zone_settings, zone_meta, zone_warning = build_locus_zone_settings_from_session(
        end_side,
        ctx["label"],
        loop_name,
    )
    if locus_zone_settings:
        fig_locus = add_locus_zone_overlay(
            fig_locus,
            locus_zone_settings,
            line_angle_deg=math.degrees(cmath.phase(ctx["line_param"]["Z1_per_km"])),
        )

    fig_locus.update_layout(
        xaxis_title="R (ohm primary)",
        yaxis_title="X (ohm primary)",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        height=720,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    if plot_focus_mode == "relay_zones" and locus_zone_settings:
        line_angle = cmath.phase(ctx["line_param"]["Z1_per_km"])
        tan_angle = math.tan(line_angle) if abs(math.tan(line_angle)) > 1e-9 else None
        zone_x_values = []
        zone_y_values = []
        for zone in locus_zone_settings:
            r = float(zone["r_reach_ohm"])
            x = float(zone["x_reach_ohm"])
            center_r = x / tan_angle if tan_angle else 0.0
            reverse_r = r * 0.35
            lower_x = x * -0.15
            zone_x_values.extend([0.0, r, center_r + r, center_r - reverse_r, -reverse_r])
            zone_y_values.extend([0.0, lower_x, x, x, x * 0.35])
        pad_x = max(5.0, 0.18 * (max(zone_x_values) - min(zone_x_values)))
        pad_y = max(5.0, 0.18 * (max(zone_y_values) - min(zone_y_values)))
        fig_locus.update_xaxes(range=[min(zone_x_values) - pad_x, max(zone_x_values) + pad_x])
        fig_locus.update_yaxes(range=[min(zone_y_values) - pad_y, max(zone_y_values) + pad_y])

    meta = {
        "label": ctx["label"],
        "loop": loop_name,
        "point_count": int(len(trajectory_df)),
        "plot_focus_mode": plot_focus_mode,
        **zone_meta,
    }
    return fig_locus, trajectory_df, meta, zone_warning




with tab0:
    st.subheader("Spreadsheet Database Configuration")

    default_database_spreadsheet_url = get_config_secret("DATABASE_SPREADSHEET_URL")
    default_tower_schedule_url = get_config_secret("TOWER_SCHEDULE_SPREADSHEET_URL", DEFAULT_TOWER_SCHEDULE_URL)
    default_case_drive_folder_url = get_config_secret("CASE_DRIVE_FOLDER_URL", DEFAULT_CASE_DRIVE_FOLDER_URL)
    old_line_spreadsheet_url = get_config_secret("OLD_LINE_SPREADSHEET_URL")
    old_cable_spreadsheet_url = get_config_secret("OLD_CABLE_SPREADSHEET_URL")
    legacy_database_urls = {
        url for url in [old_line_spreadsheet_url, old_cable_spreadsheet_url] if url
    }
    default_line_sheet_name = get_config_secret("DATABASE_LINE_SHEET", "line_impedance")
    default_cable_sheet_name = get_config_secret("DATABASE_CABLE_SHEET", "cable_impedance")
    default_tower_schedule_sheet = get_config_secret("TOWER_SCHEDULE_SHEET", DEFAULT_TOWER_SCHEDULE_SHEET)

    if (
        "database_spreadsheet_url" not in st.session_state
        and default_database_spreadsheet_url
    ):
        st.session_state["database_spreadsheet_url"] = default_database_spreadsheet_url
        st.session_state["line_data_spreadsheet_url"] = default_database_spreadsheet_url
        st.session_state["cable_data_spreadsheet_url"] = default_database_spreadsheet_url
    if "tower_schedule_url" not in st.session_state and default_tower_schedule_url:
        st.session_state["tower_schedule_url"] = default_tower_schedule_url
    if "case_drive_folder_url" not in st.session_state and default_case_drive_folder_url:
        st.session_state["case_drive_folder_url"] = default_case_drive_folder_url

    if "line_data_sheet_name" not in st.session_state:
        st.session_state["line_data_sheet_name"] = default_line_sheet_name
    if "cable_data_sheet_name" not in st.session_state:
        st.session_state["cable_data_sheet_name"] = default_cable_sheet_name
    if "tower_schedule_sheet_name" not in st.session_state:
        st.session_state["tower_schedule_sheet_name"] = default_tower_schedule_sheet

    if st.session_state.get("database_spreadsheet_url") in legacy_database_urls:
        fallback_url = default_database_spreadsheet_url if default_database_spreadsheet_url not in legacy_database_urls else ""
        st.session_state["database_spreadsheet_url"] = fallback_url
        st.session_state["line_data_spreadsheet_url"] = fallback_url
        st.session_state["cable_data_spreadsheet_url"] = fallback_url

    if not any(
        st.session_state.get(key)
        for key in [
            "database_spreadsheet_url",
            "tower_schedule_url",
            "openweather_lightning_api_key",
        ]
    ):
        st.info(
            "Belum ada runtime credentials atau Streamlit secrets. Upload credentials file, isi secrets, "
            "atau masukkan URL/API key secara manual untuk memuat data otomatis."
        )

    with st.expander("Runtime credentials upload", expanded=False):
        st.caption(
            "Opsional untuk repo public: upload `credentials.toml` atau `credentials.json` agar URL spreadsheet "
            "dan API key terisi otomatis tanpa hardcode di GitHub. File hanya dibaca ke session, tidak disimpan ke disk/case ZIP."
        )
        template = textwrap.dedent(
            """
            [spreadsheet]
            database_url = "https://docs.google.com/spreadsheets/d/..."
            database_line_sheet = "line_impedance"
            database_cable_sheet = "cable_impedance"
            database_distance_sheet = "distance_settings"
            tower_schedule_url = "https://docs.google.com/spreadsheets/d/..."
            tower_schedule_sheet = "tower_schedule"

            [openweather]
            api_key = "isi_api_key_openweather"

            [case_storage]
            drive_folder_url = "https://drive.google.com/drive/folders/..."

            # Opsional untuk Google Drive/service account.
            # [google_service_account]
            # type = "service_account"
            # project_id = "..."
            # private_key_id = "..."
            # private_key = "-----BEGIN PRIVATE KEY-----\\n...\\n-----END PRIVATE KEY-----\\n"
            # client_email = "..."
            # client_id = "..."
            # token_uri = "https://oauth2.googleapis.com/token"
            """
        ).strip()
        st.download_button(
            "Download template credentials.toml",
            data=template,
            file_name="credentials.template.toml",
            mime="text/plain",
            key="download_runtime_credentials_template",
        )
        uploaded_credentials = st.file_uploader(
            "Upload credentials file",
            type=["toml", "json"],
            key="runtime_credentials_upload",
            help="Gunakan file milik user. Jangan upload file credentials ke GitHub.",
        )
        if uploaded_credentials is not None:
            fingerprint = hashlib.sha256(uploaded_credentials.getvalue()).hexdigest()
            if st.session_state.get("runtime_credentials_fingerprint") != fingerprint:
                payload, error = parse_runtime_credentials_upload(uploaded_credentials)
                if error:
                    st.error(error)
                else:
                    st.session_state["runtime_credentials"] = payload
                    st.session_state["runtime_credentials_loaded_name"] = uploaded_credentials.name
                    st.session_state["runtime_credentials_fingerprint"] = fingerprint
                    applied = apply_runtime_credentials(payload)
                    if applied:
                        st.success("Credentials diterapkan: " + ", ".join(applied))
                    else:
                        st.warning("Credentials terbaca, tetapi tidak ada field yang cocok untuk diterapkan.")
            else:
                payload = st.session_state.get("runtime_credentials")
                if not isinstance(payload, dict):
                    payload, error = parse_runtime_credentials_upload(uploaded_credentials)
                    if error:
                        st.error(error)
                        payload = None
                    elif payload is not None:
                        st.session_state["runtime_credentials"] = payload
                applied = apply_runtime_credentials(payload) if isinstance(payload, dict) else []
                if applied:
                    st.info(
                        f"Credentials aktif: {st.session_state.get('runtime_credentials_loaded_name', uploaded_credentials.name)}. "
                        "Nilai konfigurasi diterapkan ulang ke field."
                    )
                else:
                    st.info(f"Credentials aktif: {st.session_state.get('runtime_credentials_loaded_name', uploaded_credentials.name)}")
        if st.button("Clear runtime credentials from session", key="clear_runtime_credentials"):
            for key in [
                "runtime_credentials",
                "runtime_credentials_loaded_name",
                "runtime_credentials_fingerprint",
                "runtime_gdrive_service_account",
                "openweather_lightning_api_key",
                "database_spreadsheet_url",
                "line_data_spreadsheet_url",
                "cable_data_spreadsheet_url",
                "tower_schedule_url",
                "case_drive_folder_url",
                "case_drive_folder_id",
            ]:
                st.session_state.pop(key, None)
            st.success("Runtime credentials dibersihkan dari session.")

    existing_database_url = (
        st.session_state.get("database_spreadsheet_url")
        or st.session_state.get("line_data_spreadsheet_url")
        or st.session_state.get("cable_data_spreadsheet_url")
        or ""
    )
    if existing_database_url in legacy_database_urls:
        existing_database_url = default_database_spreadsheet_url if default_database_spreadsheet_url not in legacy_database_urls else ""

    st.session_state["database_spreadsheet_url"] = existing_database_url
    st.session_state["line_data_spreadsheet_url"] = existing_database_url
    st.session_state["cable_data_spreadsheet_url"] = existing_database_url

    st.caption(
        "URL spreadsheet dapat diisi manual, dari runtime credentials, Streamlit secrets, atau environment variable. "
        "Untuk repo public, jangan hardcode URL private di source code."
    )

    database_spreadsheet_url = st.text_input(
        "Database Spreadsheet URL",
        value=st.session_state.get("database_spreadsheet_url", ""),
        key="database_spreadsheet_url_input",
    )
    database_spreadsheet_url = database_spreadsheet_url.strip()
    st.session_state["database_spreadsheet_url"] = database_spreadsheet_url
    st.session_state["line_data_spreadsheet_url"] = database_spreadsheet_url
    st.session_state["cable_data_spreadsheet_url"] = database_spreadsheet_url

    col_refresh, col_line_sheet, col_cable_sheet = st.columns([1, 2, 2])

    with col_refresh:
        if st.button("Refresh Sheets", key="refresh_database_sheets"):
            try:
                available_sheets = get_google_spreadsheet_sheet_names_cached(database_spreadsheet_url)
                st.session_state["database_available_sheets"] = available_sheets
                st.session_state["line_data_available_sheets"] = available_sheets
                st.session_state["cable_data_available_sheets"] = available_sheets
                st.success("Daftar sheet berhasil dibaca.")
            except Exception as e:
                st.session_state["database_available_sheets"] = []
                st.error("Gagal membaca daftar sheet.")
                st.exception(e)

    available_sheets = st.session_state.get("database_available_sheets", [])

    def choose_database_sheet(label, sheet_key, default_sheet):
        current_sheet = st.session_state.get(sheet_key, default_sheet)

        if available_sheets:
            selected_sheet = st.selectbox(
                label,
                available_sheets,
                index=available_sheets.index(current_sheet)
                if current_sheet in available_sheets
                else 0,
                key=f"{sheet_key}_select",
            )
        else:
            selected_sheet = st.text_input(
                label,
                value=current_sheet,
                key=f"{sheet_key}_manual",
                help="Klik Refresh Sheets untuk memilih dari daftar sheet yang tersedia.",
            )

        st.session_state[sheet_key] = str(selected_sheet).strip()

    with col_line_sheet:
        choose_database_sheet("Line Data Sheet", "line_data_sheet_name", "line_impedance")

    with col_cable_sheet:
        choose_database_sheet("Cable Data Sheet", "cable_data_sheet_name", "cable_impedance")

    def preview_database_sheet(label, source_key):
        with st.expander(f"Preview {label} Spreadsheet"):
            if st.button(f"Load Preview {label}", key=f"preview_{source_key}_spreadsheet"):
                try:
                    preview_df = read_google_spreadsheet_table_cached(
                        st.session_state["database_spreadsheet_url"],
                        st.session_state[f"{source_key}_sheet_name"],
                    )
                    st.dataframe(preview_df.head(20), use_container_width=True)
                    st.caption(f"Rows: {len(preview_df)}, Columns: {len(preview_df.columns)}")
                except Exception as e:
                    st.error("Gagal membaca preview spreadsheet.")
                    st.exception(e)

    preview_database_sheet("Line Data", "line_data")
    preview_database_sheet("Cable Data", "cable_data")

    st.markdown("### Tower Schedule Database")
    st.caption("Pengaturan sumber data Tower Schedule. Halaman Tower Schedule hanya memakai konfigurasi ini.")
    tower_db_col1, tower_db_col2, tower_db_col3 = st.columns([3, 1.2, 0.8])
    with tower_db_col1:
        tower_schedule_url_setup = st.text_input(
            "Tower Schedule Spreadsheet URL",
            value=st.session_state.get("tower_schedule_url", ""),
            key="tower_schedule_url_setup_input",
        ).strip()
    with tower_db_col2:
        tower_schedule_sheet_setup = st.text_input(
            "Tower Schedule Sheet",
            value=st.session_state.get("tower_schedule_sheet_name", default_tower_schedule_sheet),
            key="tower_schedule_sheet_setup_input",
        ).strip()
    with tower_db_col3:
        st.write("")
        st.write("")
        if st.button("Clear Tower Cache", key="clear_tower_schedule_cache_setup"):
            read_google_spreadsheet_query_cached.clear()
            st.session_state.pop("tower_schedule_df", None)
            st.session_state.pop("tower_schedule_last_query", None)
            st.session_state["tower_schedule_loaded"] = False
            st.success("Cache tower schedule dibersihkan.")

    st.session_state["tower_schedule_url"] = tower_schedule_url_setup
    st.session_state["tower_schedule_sheet_name"] = tower_schedule_sheet_setup or default_tower_schedule_sheet

    st.markdown("### Case Storage")
    st.caption(
        "Simpan rekaman COMTRADE, parameter yang sudah diubah, dan hasil kalkulasi sebagai satu arsip case. "
        "Arsip ini bisa di-load kembali dari sidebar tanpa mengatur ulang workflow dari awal."
    )
    case_col1, case_col2 = st.columns([2, 3])
    with case_col1:
        case_name_input = st.text_input(
            "Case Name",
            value=st.session_state.get("case_name", st.session_state.get("line_param", {}).get("line_name", "fault_case")),
            key="case_name_input",
        ).strip()
        st.session_state["case_name"] = case_name_input or "fault_case"
    with case_col2:
        drive_folder_input = st.text_input(
            "Google Drive Folder URL / ID",
            value=st.session_state.get("case_drive_folder_url", ""),
            key="case_drive_folder_url_input",
        ).strip()
        st.session_state["case_drive_folder_url"] = drive_folder_input
        st.session_state["case_drive_folder_id"] = extract_google_drive_folder_id(st.session_state["case_drive_folder_url"])

    case_filename, case_archive_bytes = build_case_archive_bytes(st.session_state.get("case_name", "fault_case"))
    storage_col1, storage_col2 = st.columns([1, 1])
    with storage_col1:
        st.download_button(
            "Export Case ZIP",
            data=case_archive_bytes,
            file_name=case_filename,
            mime="application/zip",
            key="export_case_zip",
        )
    with storage_col2:
        if st.button("Save Case to Google Drive", key="save_case_to_google_drive"):
            try:
                uploaded_file = upload_case_archive_to_drive(
                    case_filename,
                    case_archive_bytes,
                    st.session_state.get("case_drive_folder_id", ""),
                )
                st.success(f"Case berhasil disimpan ke Google Drive: {uploaded_file.get('name')}")
                if uploaded_file.get("webViewLink"):
                    st.markdown(f"[Buka file di Google Drive]({uploaded_file['webViewLink']})")
            except Exception as e:
                st.error("Belum bisa menyimpan ke Google Drive.")
                st.exception(e)
                st.info(
                    "Pastikan dependency Google Drive sudah terpasang, kredensial service account tersedia, "
                    "dan folder Drive sudah di-share ke email service account tersebut."
                )

    st.caption(
        "Untuk mode Drive, isi folder melalui runtime credentials, Streamlit secrets, environment variable, "
        "atau input manual. Gunakan service account melalui runtime credentials, `st.secrets['gdrive_service_account']`, "
        "atau environment variable `GOOGLE_APPLICATION_CREDENTIALS`."
    )


with tab_tower:
    st.subheader("Tower Schedule")

    expected_tower_columns = [
        "SPAN",
        "JARAK",
        "KUMULATIF",
        "LATITUDE",
        "LONGITUDE",
        "SEGMENT",
        "ULTG",
        "TYPE STRING",
        "JUMLAH STRING",
    ]

    st.caption(
        "Data tower schedule dibaca dari spreadsheet terpisah. Kolom utama: "
        "SPAN, JARAK, KUMULATIF, LATITUDE, LONGITUDE, SEGMENT, ULTG, TYPE STRING, JUMLAH STRING."
    )

    st.session_state.setdefault("tower_schedule_url", DEFAULT_TOWER_SCHEDULE_URL)
    st.session_state.setdefault("tower_schedule_sheet_name", DEFAULT_TOWER_SCHEDULE_SHEET)
    st.caption(
        "Sumber data diatur di Setup DB. "
        f"Sheet aktif: {st.session_state['tower_schedule_sheet_name']}."
    )
    tower_schedule_url_configured = bool(str(st.session_state.get("tower_schedule_url", "") or "").strip())
    if not tower_schedule_url_configured:
        st.warning(
            "Link Tower Schedule Spreadsheet belum diatur. Buka tab Setup DB lalu isi "
            "`Tower Schedule Spreadsheet URL` atau upload runtime credentials terlebih dahulu."
        )
    col_tower_refresh, _ = st.columns([0.9, 5])
    with col_tower_refresh:
        st.write("")
        st.write("")
        if st.button("Reload", key="reload_tower_schedule"):
            read_google_spreadsheet_query_cached.clear()
            st.session_state.pop("tower_schedule_df", None)
            st.session_state.pop("tower_schedule_last_query", None)
            st.session_state["tower_schedule_loaded"] = False
            st.info("Cache tower schedule dibersihkan. Isi filter awal lalu klik Load / Refresh Tower Schedule.")

    tower_filter_options_df = pd.DataFrame()
    if tower_schedule_url_configured:
        try:
            tower_filter_options_df = read_google_spreadsheet_query_cached(
                st.session_state["tower_schedule_url"],
                st.session_state["tower_schedule_sheet_name"],
                "select F, G where F is not null or G is not null",
            )
            tower_filter_options_df = make_streamlit_safe_columns(tower_filter_options_df)
            tower_filter_options_df.columns = [str(col).strip() for col in tower_filter_options_df.columns]
        except Exception as e:
            st.warning("Daftar ULTG/Segment belum dapat dibaca. Gunakan input manual atau cek akses spreadsheet.")
            st.caption(str(e))

    def _preload_options_from_df(df, column_name):
        if df.empty or column_name not in df.columns:
            return ["Semua"]
        values = (
            df[column_name]
            .dropna()
            .astype(str)
            .map(str.strip)
        )
        values = [value for value in values if value and value.lower() not in ["nan", "none"]]
        return ["Semua"] + sorted(set(values), key=lambda item: item.upper())

    pre_ultg_options = _preload_options_from_df(tower_filter_options_df, "ULTG")

    tower_has_loaded_data = "tower_schedule_df" in st.session_state
    with st.expander("Filter Awal Load", expanded=not tower_has_loaded_data):
        pre_filter_col1, pre_filter_col2, pre_filter_col3 = st.columns([1, 1, 1.2])
        with pre_filter_col1:
            selected_pre_ultg = st.selectbox(
                "ULTG sebelum load",
                pre_ultg_options,
                index=(
                    pre_ultg_options.index(st.session_state.get("tower_schedule_pre_ultg", "Semua"))
                    if st.session_state.get("tower_schedule_pre_ultg", "Semua") in pre_ultg_options
                    else 0
                ),
                key="tower_schedule_pre_ultg",
                help="Isi persis sesuai nilai kolom ULTG agar Google Sheet hanya mengambil baris ULTG tersebut.",
            )
            tower_pre_ultg = "" if selected_pre_ultg == "Semua" else selected_pre_ultg

        segment_options_df = tower_filter_options_df
        if tower_pre_ultg and "ULTG" in tower_filter_options_df.columns:
            ultg_normalized = tower_filter_options_df["ULTG"].astype(str).str.strip().str.upper()
            segment_options_df = tower_filter_options_df[
                ultg_normalized == tower_pre_ultg.strip().upper()
            ]
        pre_segment_options = _preload_options_from_df(segment_options_df, "SEGMENT")
        if st.session_state.get("tower_schedule_pre_segment", "Semua") not in pre_segment_options:
            st.session_state["tower_schedule_pre_segment"] = "Semua"

        with pre_filter_col2:
            selected_pre_segment = st.selectbox(
                "Segment sebelum load",
                pre_segment_options,
                index=(
                    pre_segment_options.index(st.session_state.get("tower_schedule_pre_segment", "Semua"))
                    if st.session_state.get("tower_schedule_pre_segment", "Semua") in pre_segment_options
                    else 0
                ),
                key="tower_schedule_pre_segment",
                help="Isi persis sesuai nilai kolom SEGMENT agar Google Sheet hanya mengambil baris segment tersebut.",
            )
            tower_pre_segment = "" if selected_pre_segment == "Semua" else selected_pre_segment
        with pre_filter_col3:
            tower_load_all = st.checkbox(
                "Load semua data",
                value=False,
                key="tower_schedule_load_all",
                help="Matikan opsi ini agar load lebih ringan memakai filter awal ULTG/Segment.",
            )

        tower_load_requested = False
        load_tower_schedule = st.button("Load / Refresh Tower Schedule", key="load_tower_schedule")
        if load_tower_schedule:
            if not tower_schedule_url_configured:
                st.warning(
                    "Tidak bisa memuat Tower Schedule karena link spreadsheet belum diatur di Setup DB."
                )
            elif not tower_load_all and not tower_pre_ultg and not tower_pre_segment:
                st.warning("Isi ULTG atau Segment terlebih dahulu, atau centang Load semua data.")
            else:
                read_google_spreadsheet_query_cached.clear()
                st.session_state["tower_schedule_loaded"] = True
                tower_load_requested = True

    if not tower_schedule_url_configured and "tower_schedule_df" not in st.session_state:
        st.info("Isi konfigurasi Tower Schedule di Setup DB sebelum memuat data tower.")
    elif not st.session_state.get("tower_schedule_loaded") and "tower_schedule_df" not in st.session_state:
        st.info("Klik Load / Refresh Tower Schedule untuk membaca data tower dari spreadsheet.")
    else:
        try:
            if tower_load_requested or st.session_state.get("tower_schedule_loaded") or "tower_schedule_df" not in st.session_state:
                tower_where_clauses = []
                if not tower_load_all and tower_pre_segment:
                    tower_where_clauses.append(f"F = '{tower_pre_segment.replace(chr(39), chr(39) + chr(39))}'")
                if not tower_load_all and tower_pre_ultg:
                    tower_where_clauses.append(f"G = '{tower_pre_ultg.replace(chr(39), chr(39) + chr(39))}'")
                tower_query = "select *"
                if tower_where_clauses:
                    tower_query += " where " + " and ".join(tower_where_clauses)

                tower_df_raw = read_google_spreadsheet_query_cached(
                    st.session_state["tower_schedule_url"],
                    st.session_state["tower_schedule_sheet_name"],
                    tower_query,
                )
                tower_df = make_streamlit_safe_columns(tower_df_raw)
                tower_df.columns = [str(col).strip() for col in tower_df.columns]
                st.session_state["tower_schedule_df"] = tower_df
                st.session_state["tower_schedule_loaded"] = False
                st.session_state["tower_schedule_last_query"] = tower_query
            else:
                tower_df = st.session_state["tower_schedule_df"].copy()
            if st.session_state.get("tower_schedule_last_query"):
                st.caption(f"Query: {st.session_state['tower_schedule_last_query']}")
    
            missing_tower_columns = [
                col for col in expected_tower_columns
                if col not in tower_df.columns
            ]
            if missing_tower_columns:
                st.warning(
                    "Kolom berikut belum ditemukan persis sesuai struktur: "
                    + ", ".join(missing_tower_columns)
                )
    
            filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
    
            def _tower_options(column_name):
                if column_name not in tower_df.columns:
                    return ["Semua"]
                values = (
                    tower_df[column_name]
                    .dropna()
                    .astype(str)
                    .map(str.strip)
                )
                values = [value for value in values if value and value.lower() not in ["nan", "none"]]
                return ["Semua"] + sorted(set(values), key=lambda item: item.upper())
    
            with filter_col1:
                selected_segment = st.selectbox(
                    "Segment",
                    _tower_options("SEGMENT"),
                    key="tower_schedule_segment_filter",
                )
            with filter_col2:
                selected_ultg = st.selectbox(
                    "ULTG",
                    _tower_options("ULTG"),
                    key="tower_schedule_ultg_filter",
                )
            with filter_col3:
                selected_type_string = st.selectbox(
                    "Type String",
                    _tower_options("TYPE STRING"),
                    key="tower_schedule_type_string_filter",
                )
            with filter_col4:
                tower_search = st.text_input(
                    "Cari span / teks",
                    value="",
                    key="tower_schedule_search",
                ).strip()
    
            filtered_tower_df = tower_df.copy()
            if selected_segment != "Semua" and "SEGMENT" in filtered_tower_df.columns:
                filtered_tower_df = filtered_tower_df[
                    filtered_tower_df["SEGMENT"].astype(str).str.strip() == selected_segment
                ]
            if selected_ultg != "Semua" and "ULTG" in filtered_tower_df.columns:
                filtered_tower_df = filtered_tower_df[
                    filtered_tower_df["ULTG"].astype(str).str.strip() == selected_ultg
                ]
            if selected_type_string != "Semua" and "TYPE STRING" in filtered_tower_df.columns:
                filtered_tower_df = filtered_tower_df[
                    filtered_tower_df["TYPE STRING"].astype(str).str.strip() == selected_type_string
                ]
            if tower_search:
                search_mask = filtered_tower_df.apply(
                    lambda row: tower_search.lower() in " ".join(str(value).lower() for value in row.values),
                    axis=1,
                )
                filtered_tower_df = filtered_tower_df[search_mask]
    
            distance_col = "JARAK" if "JARAK" in filtered_tower_df.columns else None
            cumulative_col = "KUMULATIF" if "KUMULATIF" in filtered_tower_df.columns else None
            string_count_col = "JUMLAH STRING" if "JUMLAH STRING" in filtered_tower_df.columns else None
            tower_length_m = None
            tower_length_source = None
    
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            metric_col1.metric("Rows", len(filtered_tower_df))
            if distance_col:
                distance_values = pd.to_numeric(
                    filtered_tower_df[distance_col].astype(str).str.replace(",", ".", regex=False),
                    errors="coerce",
                )
                total_distance_m = float(distance_values.sum(skipna=True))
                metric_col2.metric("Total Jarak", f"{total_distance_m / 1000.0:.6f} km")
                if np.isfinite(total_distance_m) and total_distance_m > 0:
                    tower_length_m = total_distance_m
                    tower_length_source = "sum JARAK"
            else:
                metric_col2.metric("Total Jarak", "-")
            if cumulative_col:
                cumulative_values = pd.to_numeric(
                    filtered_tower_df[cumulative_col].astype(str).str.replace(",", ".", regex=False),
                    errors="coerce",
                )
                cumulative_max_m = float(cumulative_values.max(skipna=True))
                metric_col3.metric("Kumulatif Max", f"{cumulative_max_m / 1000.0:.6f} km")
                if np.isfinite(cumulative_max_m) and cumulative_max_m > 0:
                    tower_length_m = cumulative_max_m
                    tower_length_source = "max KUMULATIF"
            else:
                metric_col3.metric("Kumulatif Max", "-")
            if string_count_col:
                string_values = pd.to_numeric(
                    filtered_tower_df[string_count_col].astype(str).str.replace(",", ".", regex=False),
                    errors="coerce",
                )
                metric_col4.metric("Jumlah String", f"{string_values.sum(skipna=True):.0f}")
            else:
                metric_col4.metric("Jumlah String", "-")

            if tower_length_m is not None:
                tower_length_km = float(tower_length_m) / 1000.0
                st.session_state["tower_schedule_selected_length_km"] = tower_length_km
                st.session_state["tower_schedule_selected_length_source"] = tower_length_source
                st.session_state["tower_schedule_selected_rows"] = int(len(filtered_tower_df))
                st.session_state["tower_schedule_selected_segment"] = selected_segment
                st.session_state["tower_schedule_selected_ultg"] = selected_ultg
                st.caption(
                    f"Panjang line Tower Schedule untuk DE: {tower_length_km:.6f} km "
                    f"({tower_length_source}, {len(filtered_tower_df)} baris terfilter)."
                )
            else:
                st.session_state.pop("tower_schedule_selected_length_km", None)
                st.session_state.pop("tower_schedule_selected_length_source", None)
    
            display_columns = [col for col in expected_tower_columns if col in filtered_tower_df.columns]
            remaining_columns = [col for col in filtered_tower_df.columns if col not in display_columns]
            display_tower_df = filtered_tower_df[display_columns + remaining_columns].reset_index(drop=True)
            for meter_col in ["JARAK", "KUMULATIF"]:
                if meter_col in display_tower_df.columns:
                    km_col = f"{meter_col} km"
                    display_tower_df[km_col] = pd.to_numeric(
                        display_tower_df[meter_col].astype(str).str.replace(",", ".", regex=False),
                        errors="coerce",
                    ) / 1000.0
            st.session_state["tower_schedule_filtered_df"] = display_tower_df.copy()
    
            st.markdown("### Tower Schedule Table")
            tower_formatters = {
                col: "{:.6f}"
                for col in ["JARAK km", "KUMULATIF km"]
                if col in display_tower_df.columns
            }
            if tower_formatters:
                st.dataframe(
                    display_tower_df.style.format(tower_formatters, na_rep="-"),
                    use_container_width=True,
                    height=420,
                )
            else:
                st.dataframe(display_tower_df, use_container_width=True, height=420)
    
            if "LATITUDE" in display_tower_df.columns and "LONGITUDE" in display_tower_df.columns:
                show_tower_map = st.toggle(
                    "Tampilkan Tower Map",
                    value=False,
                    key="show_tower_schedule_map",
                    help="Aktifkan hanya jika ingin melihat koordinat tower. Map dimatikan default agar halaman lebih ringan.",
                )
                if show_tower_map:
                    st.markdown("### Tower Map")
                    render_tower_map(
                        display_tower_df,
                        key_prefix="tower_schedule",
                        include_fault_layer=True,
                        default_show_fault=True,
                        height=560,
                    )
        except Exception as e:
            st.error("Gagal membaca tower schedule dari spreadsheet.")
            st.caption("Pastikan link spreadsheet dapat diakses dan sheet `tower_schedule` tersedia.")
            st.exception(e)
    

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
        index=get_ground_index(default_ie, ground_options),
    )

    if ie_channel != "None":
        st.success(f"Channel arus netral/ground terdeteksi dan dipakai: {ie_channel}")
    else:
        st.info("Channel IE/IN/3I0 tidak dipilih. Aplikasi akan menghitung IE residual dari Ia + Ib + Ic.")

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
            step=0.001,
            format="%.5f",
        )

    with col_ct2:
        ct_secondary = st.number_input(
            "CT Secondary (A)",
            value=float(auto_transformer_data.get("ct_secondary", 1.0)),
            step=0.001,
            format="%.5f",
        )

    with col_vt1:
        vt_primary = st.number_input(
            "VT/CVT Primary (V)",
            value=float(auto_transformer_data.get("vt_primary", 150000.0)),
            step=0.001,
            format="%.5f",
        )

    with col_vt2:
        vt_secondary = st.number_input(
            "VT/CVT Secondary (V)",
            value=float(auto_transformer_data.get("vt_secondary", 100.0)),
            step=0.001,
            format="%.5f",
        )
    
    st.caption(
        f"Auto ratio source: CT = {auto_transformer_data.get('ct_ratio_source', '-')}, "
        f"VT = {auto_transformer_data.get('vt_ratio_source', '-')}. "
        "Tetap validasi manual karena tidak semua file CFG menyimpan primary/secondary dengan benar."
    )

    if recorded_side == "primary":
        st.info(
            "Mode primary aktif: nilai COMTRADE dianggap sudah dalam satuan primer, sehingga rasio CT/VT "
            "tidak dikalikan lagi pada waveform. Field CT/VT tetap ditampilkan untuk dokumentasi dan validasi."
        )
    else:
        st.info(
            "Mode secondary aktif: waveform akan dikalikan rasio CT/VT agar menjadi nilai primer. "
            "Pastikan VT/CVT Primary memakai tegangan primer nominal, misalnya 150000 V untuk sistem 150 kV, "
            "bukan 1500 jika yang dimaksud adalah rasio 1500:1."
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
    st.session_state["local_transformer_data"] = {
        "recorded_side": recorded_side,
        "ct_primary": ct_primary,
        "ct_secondary": ct_secondary,
        "vt_primary": vt_primary,
        "vt_secondary": vt_secondary,
        "nominal_phase_voltage_rms": vt_primary / math.sqrt(3.0),
        "nominal_current_rms": ct_primary,
    }
    st.session_state["local_ie_channel"] = ie_channel if ie_channel != "None" else None
    st.session_state["local_ie_source"] = (
        "measured" if ie_channel != "None" else "calculated_from_3_phase_currents"
    )

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


with summary_container:
    st.subheader("Summary / Report Ringkas")
    st.caption(
        "Halaman ini merangkum rekaman gangguan, hasil utama, dan grafik pendukung. "
        "Data yang belum dihitung akan ditampilkan sebagai Pending, bukan menyembunyikan report."
    )

    local_name = str(metadata.get("station_name") or "Local End")
    remote_loaded = "remote_metadata" in st.session_state or (remote_cfg_file is not None and remote_dat_file is not None)
    remote_status = "Uploaded" if remote_loaded else "Not uploaded"
    remote_metadata_summary = st.session_state.get("remote_metadata", {})
    remote_name = str(remote_metadata_summary.get("station_name") or "Remote End")

    col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
    col_sum1.metric("Local Record", local_name)
    col_sum2.metric("Remote Record", remote_name if remote_loaded else remote_status)
    col_sum3.metric("Samples", metadata.get("total_samples", "-"))
    col_sum4.metric("Frequency", f"{metadata.get('frequency') or '-'} Hz")

    ie_local_text = (
        f"measured ({st.session_state.get('local_ie_channel')})"
        if st.session_state.get("local_ie_source") == "measured"
        else "calculated from Ia+Ib+Ic"
    )
    ie_remote_text = (
        f"measured ({st.session_state.get('remote_ie_selected_channel')})"
        if st.session_state.get("remote_ie_source") == "measured"
        else "calculated from Ia+Ib+Ic"
    )
    st.caption(f"IE source: Local = {ie_local_text}; Remote = {ie_remote_text}.")

    st.markdown("### Calculation Status")
    status_rows = [
        {
            "Step": "Signal Assignment",
            "Status": "Done" if "assigned_df" in st.session_state else "Pending",
            "Main Result": "Local waveform mapped" if "assigned_df" in st.session_state else "-",
        },
        {
            "Step": "Fault Cursor",
            "Status": "Done" if "fault_window" in st.session_state else "Pending",
            "Main Result": (
                f'{st.session_state["fault_window"]["fault_time"]:.6f} s'
                if "fault_window" in st.session_state
                else "-"
            ),
        },
        {
            "Step": "Phasor",
            "Status": "Done" if "phasors" in st.session_state else "Pending",
            "Main Result": (
                f'V1 {st.session_state["phasors"]["V1"]["magnitude"]:.3f}, '
                f'I1 {st.session_state["phasors"]["I1"]["magnitude"]:.3f}'
                if "phasors" in st.session_state and "V1" in st.session_state["phasors"]
                else "-"
            ),
        },
        {
            "Step": "Fault Type",
            "Status": "Done" if "fault_type_result" in st.session_state else "Pending",
            "Main Result": (
                st.session_state["fault_type_result"].get("fault_type", "-")
                if "fault_type_result" in st.session_state
                else "-"
            ),
        },
        {
            "Step": "Line Parameter",
            "Status": "Done" if "line_param" in st.session_state else "Pending",
            "Main Result": (
                f'{st.session_state["line_param"].get("line_name", "-")} | '
                f'{st.session_state["line_param"]["length_km"]:.3f} km'
                if "line_param" in st.session_state
                else "-"
            ),
        },
        {
            "Step": "Single-End",
            "Status": "Done" if "single_ended_result" in st.session_state else "Pending",
            "Main Result": (
                f'{st.session_state["single_ended_result"]["recommended_distance_km"]:.3f} km '
                f'({st.session_state["single_ended_result"]["status"]})'
                if "single_ended_result" in st.session_state
                else "-"
            ),
        },
        {
            "Step": "Double-End",
            "Status": "Done" if "two_ended_result" in st.session_state else "Pending",
            "Main Result": (
                f'{st.session_state["two_ended_result"].get("distance_from_original_local_km", st.session_state["two_ended_result"].get("distance_km", 0.0)):.3f} km | '
                f'Q {st.session_state.get("two_ended_quality", {}).get("quality_score", "-")}/10'
                if "two_ended_result" in st.session_state
                else "-"
            ),
        },
    ]
    st.dataframe(pd.DataFrame(status_rows), use_container_width=True)

    st.markdown("### Key Results")
    key_col1, key_col2, key_col3, key_col4 = st.columns(4)
    fault_type_summary = st.session_state.get("fault_type_result", {})
    remote_fault_type_summary = st.session_state.get("remote_fault_type_result", {})
    single_summary = st.session_state.get("single_ended_result")
    two_summary = st.session_state.get("two_ended_result")
    two_quality_summary = st.session_state.get("two_ended_quality", {})

    key_col1.metric("Fault Type", fault_type_summary.get("fault_type", "-"))
    key_col2.metric(
        "Single-End",
        f'{single_summary["recommended_distance_km"]:.3f} km' if single_summary else "-",
    )
    key_col3.metric(
        "Double-End",
        f'{two_summary.get("distance_from_original_local_km", two_summary.get("distance_km", 0.0)):.3f} km' if two_summary else "-",
    )
    key_col4.metric(
        "DE Quality",
        f'{two_quality_summary.get("quality_score", "-")}/10'
        if two_quality_summary
        else "-",
    )

    summary_operating_status = st.session_state.get("two_ended_operating_status")
    if summary_operating_status:
        st.markdown("### Status Diagnostik DE")
        status_text = ", ".join(summary_operating_status.get("statuses", []))
        if summary_operating_status.get("can_use_de_distance"):
            st.success(f"Status: {status_text}")
        else:
            st.warning(f"Status: {status_text}")
        for note in summary_operating_status.get("notes", []):
            st.info(note)
        st.caption(summary_operating_status.get("recommendation", ""))
    st.markdown("### Perbandingan Pre-fault dan Fault")
    local_comparison_df = build_prefault_fault_comparison_dataframe(
        st.session_state.get("phasors"),
        st.session_state.get("prefault_phasors"),
        st.session_state.get("two_ended_local_gi_label", "Local"),
    )
    if not local_comparison_df.empty:
        st.markdown("#### Rekaman GI Lokal")
        st.dataframe(
            local_comparison_df.style.format(
                {
                    "Pre-fault RMS": "{:.3f}",
                    "Fault RMS": "{:.3f}",
                    "Delta RMS": "{:.3f}",
                    "Delta %": "{:.2f}",
                    "Fault Angle deg": "{:.2f}",
                },
                na_rep="-",
            ),
            use_container_width=True,
        )
    else:
        st.info("Tabel pre-fault/fault GI lokal belum tersedia. Jalankan Fault Cursor dan Phasor lokal dahulu.")

    remote_comparison_df = build_prefault_fault_comparison_dataframe(
        st.session_state.get("remote_phasors"),
        st.session_state.get("remote_prefault_phasors"),
        st.session_state.get("two_ended_remote_gi_label", "Remote"),
    )
    if not remote_comparison_df.empty:
        st.markdown("#### Rekaman GI Remote")
        st.dataframe(
            remote_comparison_df.style.format(
                {
                    "Pre-fault RMS": "{:.3f}",
                    "Fault RMS": "{:.3f}",
                    "Delta RMS": "{:.3f}",
                    "Delta %": "{:.2f}",
                    "Fault Angle deg": "{:.2f}",
                },
                na_rep="-",
            ),
            use_container_width=True,
        )
    else:
        st.info("Tabel pre-fault/fault GI remote belum tersedia. Jalankan Double-End sampai remote phasor terbaca.")

    st.markdown("### Waveform Fokus Fault Detection")
    summary_fault_type, summary_voltage_channel, summary_current_channel = choose_summary_fault_signals(
        fault_type_summary,
        remote_fault_type_summary,
    )
    local_assigned_df = st.session_state.get("assigned_df")
    remote_assigned_df = st.session_state.get("remote_assigned_df")
    local_fault_window = st.session_state.get("fault_window")
    remote_fault_window = st.session_state.get("remote_fault_window")
    summary_remote_shift_s = float(st.session_state.get("two_ended_remote_sync_shift_s", 0.0) or 0.0)
    if abs(summary_remote_shift_s) > 1e-9:
        st.caption(
            "Waveform Summary memakai shift sinkronisasi remote dari tab Double-End: "
            f"{summary_remote_shift_s:+.6f} s."
        )

    if (
        (local_assigned_df is not None and "IE" in local_assigned_df.columns)
        or (remote_assigned_df is not None and "IE" in remote_assigned_df.columns)
    ):
        neutral_channel = "IE"
    else:
        neutral_channel = "I0"

    waveform_specs = [
        (
            summary_voltage_channel,
            f"Waveform Tegangan Fasa Terganggu ({summary_voltage_channel})",
        ),
        (
            summary_current_channel,
            f"Waveform Arus Fasa Terganggu ({summary_current_channel})",
        ),
        (
            neutral_channel,
            f"Waveform Arus Netral ({neutral_channel})",
        ),
    ]

    show_summary_waveforms = st.toggle(
        "Tampilkan waveform fokus di Summary",
        value=False,
        key="show_summary_waveforms",
        help="Matikan default agar Summary tetap ringan di hosting. Aktifkan saat ingin membuat report atau validasi visual.",
    )

    if show_summary_waveforms:
        for channel_name, waveform_title in waveform_specs:
            if (
                (
                    local_assigned_df is not None
                    and local_fault_window is not None
                    and channel_name in local_assigned_df.columns
                )
                or (
                    remote_assigned_df is not None
                    and remote_fault_window is not None
                    and channel_name in remote_assigned_df.columns
                )
            ):
                st.plotly_chart(
                    build_summary_focus_waveform(
                        local_assigned_df,
                        remote_assigned_df,
                        local_fault_window,
                        remote_fault_window,
                        channel_name,
                        waveform_title,
                        remote_time_shift_s=summary_remote_shift_s,
                    ),
                    use_container_width=True,
                )
            else:
                st.info(f"Channel {channel_name} belum tersedia untuk grafik {waveform_title}.")

    st.markdown("### Estimasi Penyebab Gangguan")
    estimated_cause, estimated_cause_note = estimate_summary_disturbance_cause(
        fault_type_summary,
        st.session_state.get("high_resistance_result"),
    )
    cause_options = [
        "Auto estimate",
        "Petir",
        "Pohon",
        "Benda asing",
        "Power swing",
        "Belum diketahui",
    ]
    cause_choice = st.selectbox(
        "Penyebab gangguan untuk report",
        cause_options,
        key="summary_disturbance_cause_choice",
    )
    displayed_cause = estimated_cause if cause_choice == "Auto estimate" else cause_choice
    st.metric("Penyebab Gangguan", displayed_cause)
    st.caption(estimated_cause_note)

    st.markdown("### Grafik SE dan DE")
    summary_location_fig = build_summary_line_position_from_session()
    if summary_location_fig is not None:
        st.plotly_chart(
            summary_location_fig,
            use_container_width=True,
            key="summary_two_ended_line_position_fig",
        )
    else:
        st.info(
            "Grafik SE/DE akan muncul setelah Single-End atau Double-End selesai menghitung."
        )
    if "high_resistance_result" in st.session_state:
        st.info(explain_high_resistance_result(st.session_state["high_resistance_result"]))

    st.markdown("### Tower Map Fault Location")
    summary_tower_df = st.session_state.get("tower_schedule_filtered_df")
    if summary_tower_df is not None and not summary_tower_df.empty:
        if get_fault_location_map_options():
            render_tower_map(
                summary_tower_df,
                key_prefix="summary_tower_fault",
                include_fault_layer=True,
                default_show_fault=True,
                height=560,
                focus_on_fault=True,
            )
            render_fault_weather_lightning_summary(
                summary_tower_df,
                key_prefix="summary_weather_lightning",
            )
        else:
            st.info("Tower map tersedia, tetapi lokasi fault akan muncul setelah perhitungan DE atau SE selesai.")
            render_tower_map(
                summary_tower_df,
                key_prefix="summary_tower",
                include_fault_layer=False,
                default_show_fault=False,
                height=520,
            )
    else:
        st.info("Tower Map Summary akan muncul setelah data Tower Schedule dimuat dan difilter.")

    st.markdown("### R-X Locus Trajectory")
    summary_rx_local_label, summary_rx_remote_label = infer_gi_names_from_line_name(
        st.session_state.get("line_param", {}).get("line_name", "")
    )

    def show_summary_rx_locus(end_suffix: str, fallback_label: str):
        fig, _, meta, build_warning = build_rx_locus_figure_from_session(end_suffix)
        if fig is None:
            st.info(
                build_warning
                or f"Locus {fallback_label} akan muncul setelah data {fallback_label} lengkap."
            )
            return

        label = meta.get("label", fallback_label)
        loop = meta.get("loop", "-")
        point_count = meta.get("point_count", 0)
        zone_count = meta.get("zone_count", 0)
        if build_warning:
            st.warning(build_warning)
        st.caption(
            f"{label} | Loop {loop} | {point_count} titik trajectory | "
            f"{zone_count} zona proteksi"
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
            key=f"summary_rx_locus_{end_suffix}",
        )

    st.markdown(f"#### {summary_rx_local_label}")
    show_summary_rx_locus("local", summary_rx_local_label)

    st.markdown(f"#### {summary_rx_remote_label}")
    show_summary_rx_locus("remote", summary_rx_remote_label)

    if two_quality_summary and two_quality_summary.get("warnings"):
        st.markdown("### Double-End Warnings")
        for warning in two_quality_summary["warnings"]:
            st.warning(warning)

    st.markdown("### Report Hint")
    st.write(
        "Untuk hasil cetak ringkas, buka tab ini lalu gunakan menu browser/Streamlit Print. "
        "Untuk analisis detail, lanjutkan ke tab workflow di sebelah kanan."
    )

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
    waveform_display_mode = st.radio(
        "Mode tampilan waveform",
        ["Instantaneous / peak", "RMS 1 siklus"],
        horizontal=True,
        key="local_assigned_waveform_display_mode",
    )

    st.info(
        "Grafik ini menampilkan waveform instantaneous/peak setelah signal assignment. "
        "Angka RMS di Wavewin atau fasor aplikasi akan lebih kecil sekitar faktor sqrt(2) "
        "untuk sinyal sinus. Contoh sistem 150 kV: V fasa RMS sekitar 86.6 kV, "
        "sedangkan puncak instantaneous normal sekitar 122.5 kV."
    )

    rms_summary_df = build_waveform_rms_summary(
        assigned_df,
        selected_channels,
        frequency=float(st.session_state.get("fault_detection", {}).get("frequency", metadata.get("frequency") or 50.0)),
    )

    if not rms_summary_df.empty:
        with st.expander("Ringkasan RMS vs Peak Awal Rekaman", expanded=False):
            st.dataframe(
                rms_summary_df.style.format(
                    {
                        "RMS Awal Rekaman": "{:.3f}",
                        "Peak Absolut Awal": "{:.3f}",
                        "Peak/RMS": "{:.3f}",
                    }
                ),
                use_container_width=True,
            )

    fig, waveform_caption = build_assigned_waveform_plot(
        assigned_df,
        selected_channels,
        f"Waveform {selected_group} - {waveform_display_mode}",
        waveform_display_mode,
        frequency=float(metadata.get("frequency") or 50.0),
    )
    st.caption(waveform_caption)

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
    local_transformer_data = st.session_state.get("local_transformer_data", {})

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
        )

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
        key="use_auto_fault_detection",
        help=(
            "Aplikasi memakai pre-fault RMS bila normal. Jika pre-fault terlihat sudah abnormal, "
            "aplikasi memakai referensi nominal dari VT/CT sebagai pembanding tambahan."
        ),
    )

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
        )

        fault_detection_method = st.selectbox(
            "Fault Detection Method",
            ["legacy_rms", "hybrid_superimposed"],
            index=1,
            disabled=(not use_advanced_fault_detection or use_auto_fault_detection),
            help="hybrid_superimposed memakai energi perubahan satu siklus lalu divalidasi RMS.",
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
            )

        with col_adv4:
            refine_fault_bar = st.checkbox(
                "Refine Fault Bar",
                value=True,
                help="Backtrack dari kandidat RMS ke perubahan instantaneous awal.",
                disabled=(not use_advanced_fault_detection or use_auto_fault_detection),
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

        fig = build_fault_window_plot(
            display_df,
            fault_window,
            selected_plot,
            "Fault Detection dan Cursor Window",
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

        try:
            prefault_phasors = calculate_all_phasors(
                df=assigned_df,
                cursor_index=fault_window["fault_index"],
                samples_per_cycle=samples_per_cycle,
            )
            prefault_phasors = add_sequence_components_to_phasor_dict(prefault_phasors)
            st.session_state["prefault_phasors"] = prefault_phasors
            st.caption(
                "Pre-fault phasor tersedia. Aplikasi akan memakai perubahan fasor "
                "pre-fault ke fault untuk membantu case gangguan resistif/load-flow."
            )
        except Exception as prefault_error:
            st.session_state.pop("prefault_phasors", None)
            st.caption(
                "Pre-fault phasor tidak dapat dihitung untuk record lokal: "
                f"{prefault_error}"
            )

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

        dft_plot_df = downsample_dataframe_for_plot(assigned_df, "time", selected_dft_plot)
        fig_dft = px.line(
            dft_plot_df,
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

        st.caption(
            "Diagram polar ini menampilkan fasor RMS fundamental pada window DFT. "
            "Gunakan untuk memeriksa urutan fasa, polaritas, dan sudut antar fasa."
        )

        col_v_phasor, col_i_phasor = st.columns(2)

        with col_v_phasor:
            st.plotly_chart(
                build_wavewin_style_phasor_diagram(
                    phasors,
                    ["Va", "Vb", "Vc"],
                    "Voltage Phasors",
                    line_color="#ff00ff",
                ),
                use_container_width=True,
            )

        with col_i_phasor:
            st.plotly_chart(
                build_wavewin_style_phasor_diagram(
                    phasors,
                    ["Ia", "Ib", "Ic"],
                    "Current Phasors",
                    line_color="#2563eb",
                ),
                use_container_width=True,
            )

        with st.expander("Sequence Component Phasor Diagram"):
            col_seq_v, col_seq_i = st.columns(2)
            with col_seq_v:
                st.plotly_chart(
                    build_wavewin_style_phasor_diagram(
                        phasors,
                        ["V1", "V2", "V0"],
                        "Voltage Sequence Phasors",
                        line_color="#7c3aed",
                    ),
                    use_container_width=True,
                )
            with col_seq_i:
                st.plotly_chart(
                    build_wavewin_style_phasor_diagram(
                        phasors,
                        ["I1", "I2", "I0"],
                        "Current Sequence Phasors",
                        line_color="#d97706",
                    ),
                    use_container_width=True,
                )

    except Exception as e:
        st.error("Perhitungan fasor gagal.")
        st.exception(e)


with tab6:
    st.subheader("Fault Type Detection")

    if "phasors" not in st.session_state:
        st.warning("Silakan lakukan Phasor Calculation terlebih dahulu.")
        st.stop()

    phasors = st.session_state["phasors"]

    st.markdown("### Auto Fault Type Detection")

    local_auto_fault_settings = calculate_auto_fault_type_thresholds(
        phasors,
        st.session_state.get("prefault_phasors"),
    )
    use_auto_fault_type_thresholds = st.toggle(
        "Gunakan threshold otomatis dari kondisi pre-fault",
        value=True,
        key="use_auto_fault_type_thresholds",
        help=(
            "Aplikasi menghitung level normal tegangan/arus dari window pre-fault "
            "setelah scaling CT/VT, lalu menentukan threshold deteksi secara adaptif."
        ),
    )

    st.caption(
        "Mode otomatis membuat user tidak perlu tuning threshold. Parameter manual di bawah "
        "hanya dipakai jika mode otomatis dimatikan."
    )

    st.markdown("### Manual Fault Type Thresholds")

    col_ft1, col_ft2, col_ft3 = st.columns(3)

    with col_ft1:
        voltage_drop_threshold_ft = st.number_input(
            "Voltage Drop Threshold",
            value=0.80,
            min_value=0.10,
            max_value=1.00,
            step=0.0001,
            format="%.5f",
            help="Fasa dianggap drop jika Vphase <= threshold Ã— Vmax."
        )

    with col_ft2:
        current_rise_threshold_ft = st.number_input(
            "Current Rise Threshold",
            value=1.50,
            min_value=1.05,
            max_value=10.00,
            step=0.0001,
            format="%.5f",
            help="Fasa dianggap faulted jika Iphase >= threshold Ã— Imin."
        )

    with col_ft3:
        ground_current_threshold_ft = st.number_input(
            "Ground Current Threshold",
            value=0.20,
            min_value=0.01,
            max_value=1.00,
            step=0.0001,
            format="%.5f",
            help="Ground fault jika IE/Imax atau I0/Iavg melebihi threshold."
        )

    with st.expander("Advanced Resistive Fault / Delta Detection"):
        col_ftd1, col_ftd2 = st.columns(2)

        with col_ftd1:
            delta_current_threshold_ft = st.number_input(
                "Delta Current Dominance Threshold",
                value=0.45,
                min_value=0.05,
                max_value=1.00,
                step=0.0001,
                format="%.5f",
                help=(
                    "Fasa dianggap berubah signifikan jika delta fasornya cukup dominan "
                    "dibanding delta arus terbesar. Berguna saat arus fasa fault turun "
                    "karena load-flow."
                ),
            )

        with col_ftd2:
            delta_voltage_threshold_ft = st.number_input(
                "Delta Voltage Threshold",
                value=0.01,
                min_value=0.0001,
                max_value=0.20,
                step=0.0001,
                format="%.5f",
                help="Ambang perubahan tegangan relatif pre-fault untuk mengenali sag kecil pada high resistance fault.",
            )

    if use_auto_fault_type_thresholds:
        voltage_drop_threshold_ft = local_auto_fault_settings["voltage_drop_threshold"]
        current_rise_threshold_ft = local_auto_fault_settings["current_rise_threshold"]
        ground_current_threshold_ft = local_auto_fault_settings["ground_current_threshold"]
        delta_current_threshold_ft = local_auto_fault_settings["delta_current_threshold"]
        delta_voltage_threshold_ft = local_auto_fault_settings["delta_voltage_threshold"]

    fault_type_result = detect_fault_type(
        phasors=phasors,
        prefault_phasors=st.session_state.get("prefault_phasors"),
        voltage_drop_threshold=voltage_drop_threshold_ft,
        current_rise_threshold=current_rise_threshold_ft,
        ground_current_threshold=ground_current_threshold_ft,
        delta_current_threshold=delta_current_threshold_ft,
        delta_voltage_threshold=delta_voltage_threshold_ft,
    )
    fault_type_result["auto_thresholds"] = local_auto_fault_settings
    fault_type_result["threshold_mode"] = "auto_prefault" if use_auto_fault_type_thresholds else "manual"

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

    st.info(explain_fault_type_result(fault_type_result, context="Rekaman local"))

    with st.expander("Auto Threshold Detail"):
        st.dataframe(
            build_auto_fault_type_threshold_dataframe(local_auto_fault_settings).style.format(
                {"Value": lambda x: f"{x:.6f}" if isinstance(x, (int, float)) else x}
            ),
            use_container_width=True,
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
            "Pilih sumber parameter saluran, lalu normalisasi menjadi Z1_per_km, "
            "Z0_per_km, K0, Z1_total, dan Z0_total."
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
        local_transformer_for_line = (
            st.session_state.get("local_transformer_data")
            or st.session_state.get("auto_transformer_data", {})
        )
        transformer_ratio_source = (
            "Signal Assignment"
            if st.session_state.get("local_transformer_data")
            else "metadata CFG otomatis"
        )
        assignment_ct_primary = float(local_transformer_for_line.get("ct_primary", 800.0))
        assignment_ct_secondary = float(local_transformer_for_line.get("ct_secondary", 1.0))
        assignment_vt_primary = float(local_transformer_for_line.get("vt_primary", 150000.0))
        assignment_vt_secondary = float(local_transformer_for_line.get("vt_secondary", 100.0))

        def is_filled(value):
            if value is None:
                return False
            if isinstance(value, float) and math.isnan(value):
                return False
            if isinstance(value, str) and value.strip().lower() in ["", "nan", "none", "null"]:
                return False
            return True

        if (
            line_parameter_source in ["Database Excel Line Data", "Database Excel Cable Data"]
            and not str(st.session_state.get("database_spreadsheet_url", "") or "").strip()
        ):
            st.warning(
                "Link Database Spreadsheet belum diatur. Buka tab Setup DB lalu isi "
                "`Database Spreadsheet URL` atau upload runtime credentials terlebih dahulu. "
                "Sementara gunakan `Input Manual` untuk mengisi parameter saluran."
            )
            line_parameter_source = "Input Manual"

        if line_parameter_source in ["Database Excel Line Data", "Database Excel Cable Data"]:
            use_cable_database = line_parameter_source == "Database Excel Cable Data"
            database_source_key = "cable_data" if use_cable_database else "line_data"
            database_spreadsheet_url = st.session_state.get(
                f"{database_source_key}_spreadsheet_url",
                st.session_state.get("database_spreadsheet_url", ""),
            )
            database_spreadsheet_url = str(database_spreadsheet_url or "").strip()
            database_sheet_name = st.session_state.get(
                f"{database_source_key}_sheet_name",
                "cable_impedance" if use_cable_database else "line_impedance",
            )
            database_title = (
                "Database Spreadsheet Data Impedansi Konduktor"
                if use_cable_database
                else "Database Spreadsheet Data Impedansi Saluran"
            )
            database_preview_title = (
                "Preview Database Cable Impedance"
                if use_cable_database
                else "Preview Database Line Impedance"
            )
            selected_row_label_text = (
                "Pilih jenis konduktor dari cable_data spreadsheet"
                if use_cable_database
                else "Pilih baris data saluran / BAY PHT"
            )

            try:
                conductor_df = read_google_spreadsheet_table_cached(
                    database_spreadsheet_url,
                    database_sheet_name,
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

                detected_columns = detect_impedance_columns(conductor_df)

                with st.expander("Advanced Database Detail", expanded=False):
                    st.markdown(f"#### {database_title}")
                    st.caption(
                        "Aplikasi membaca data impedansi dari Google Spreadsheet yang dikonfigurasi "
                        f"di tab Spreadsheet Config. Sheet aktif: `{database_sheet_name}`."
                    )
                    st.caption(f"Spreadsheet URL: {database_spreadsheet_url}")
                    if use_cable_database:
                        st.warning(
                            "Gunakan opsi ini jika nama line tidak tersedia di line_data spreadsheet. "
                            "Aplikasi mengambil Z1/Z0 per km dari data konduktor. "
                            "Jika spreadsheet tidak memiliki panjang saluran, aplikasi memakai nilai fallback."
                        )
                    st.markdown(f"#### {database_preview_title}")
                    st.dataframe(conductor_df, use_container_width=True, height=300)
                    st.markdown("#### Kolom yang Terdeteksi Otomatis")
                    st.json(detected_columns)

                all_columns = ["None"] + list(conductor_df.columns)

                def col_index(col_name):
                    if col_name in all_columns:
                        return all_columns.index(col_name)
                    return 0

                line_name_col = detected_columns.get("line_name")
                length_col = detected_columns.get("length")
                bay_pht_col = detected_columns.get("bay_pht")
                z1_real_col = detected_columns.get("z1_real")
                z1_imag_col = detected_columns.get("z1_imag")
                z1_abs_col = detected_columns.get("z1_abs")
                z1_angle_col = detected_columns.get("z1_angle")
                z0_real_col = detected_columns.get("z0_real")
                z0_imag_col = detected_columns.get("z0_imag")
                z0_abs_col = detected_columns.get("z0_abs")
                z0_angle_col = detected_columns.get("z0_angle")
                ratio_gia_ct_col = detected_columns.get("ratio_gia_ct")
                ratio_gia_vt_col = detected_columns.get("ratio_gia_vt")
                ratio_gib_ct_col = detected_columns.get("ratio_gib_ct")
                ratio_gib_vt_col = detected_columns.get("ratio_gib_vt")

                corrected_columns = {
                    "ultg": None if use_cable_database else detected_columns.get("ultg"),
                    "gi": None if use_cable_database else detected_columns.get("gi"),
                    "line_number": None if use_cable_database else detected_columns.get("line_number"),
                    "segment": None if use_cable_database else detected_columns.get("segment"),
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

                with st.expander("Detail data impedansi yang dipilih", expanded=False):
                    st.json(
                        {
                            "ultg": excel_impedance_data.get("ultg"),
                            "gi": excel_impedance_data.get("gi"),
                            "line_number": excel_impedance_data.get("line_number"),
                            "segment": excel_impedance_data.get("segment"),
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
                
                st.session_state["excel_ratio_side"] = "Tidak gunakan dari Excel"

            except Exception as e:
                st.error("Gagal membaca database spreadsheet.")
                st.exception(e)
                st.caption(
                    "Pastikan URL benar, spreadsheet dapat diakses publik, dan nama sheet sesuai. "
                    "Ubah konfigurasi di tab Spreadsheet Config."
                )

        if (
            line_parameter_source != "Input Manual"
            and st.session_state.get("excel_impedance_source") == line_parameter_source
            and "excel_impedance_data" in st.session_state
        ):
            excel_impedance_data = st.session_state["excel_impedance_data"]

        if line_parameter_source != "Input Manual" and excel_impedance_data:
            spreadsheet_line_name = (
                excel_impedance_data.get("line_name")
                or excel_impedance_data.get("bay_pht")
            )
            spreadsheet_length = excel_impedance_data.get("length")

            if not is_filled(spreadsheet_line_name) or not is_filled(spreadsheet_length):
                st.markdown("### Lengkapi Data Saluran")
                st.caption(
                    "Data impedansi diambil dari spreadsheet. Field di bawah hanya muncul "
                    "karena belum tersedia pada baris data yang dipilih."
                )

            if is_filled(spreadsheet_line_name):
                line_name = str(spreadsheet_line_name)
            else:
                default_missing_line_name = (
                    st.session_state.get("line_param", {}).get("line_name")
                    or f"Line baru - {excel_impedance_data.get('conductor_type', 'Database')}"
                )
                line_name = st.text_input(
                    "Line Name",
                    value=str(default_missing_line_name),
                    key="spreadsheet_missing_line_name",
                    help="Nama line belum tersedia pada spreadsheet cable_data, jadi perlu diisi manual.",
                )

            if is_filled(spreadsheet_length):
                line_length = float(spreadsheet_length)
                length_unit = "km"
            else:
                default_missing_length = float(
                    st.session_state.get("line_param", {}).get("length_km", 75.0)
                    if st.session_state.get("line_param")
                    else 75.0
                )
                col_missing_length, col_missing_unit = st.columns([2, 1])
                with col_missing_length:
                    line_length = st.number_input(
                        "Line Length",
                        value=default_missing_length,
                        min_value=0.001,
                        step=0.001,
                        format="%.5f",
                        key="spreadsheet_missing_line_length",
                        help="Panjang saluran belum tersedia pada spreadsheet cable_data, jadi perlu diisi manual.",
                    )
                with col_missing_unit:
                    length_unit = st.selectbox(
                        "Length Unit",
                        ["km", "miles"],
                        index=0,
                        key="spreadsheet_missing_length_unit",
                    )

            impedance_input = "relative"
            base_side = "primary"
            lp_ct_primary = assignment_ct_primary
            lp_ct_secondary = assignment_ct_secondary
            lp_vt_primary = assignment_vt_primary
            lp_vt_secondary = assignment_vt_secondary
            positive_sequence_mode = "R_X"
            r1 = float(excel_impedance_data.get("R1") or 0.08)
            x1 = float(excel_impedance_data.get("X1") or 0.42)
            z1_mag = None
            phi1_deg = None
            zero_sequence_mode = "R0_X0"
            r0 = float(excel_impedance_data.get("R0") or 0.25)
            x0 = float(excel_impedance_data.get("X0") or 1.25)
            re_rl = None
            xe_xl = None
            z0_z1_mag = None
            z0_z1_angle_deg = None
            kl_mag = None
            kl_angle_deg = None
            st.info(
                "Parameter saluran akan dinormalisasi dari data spreadsheet yang dipilih. "
                f"Ratio CT/VT memakai data dari {transformer_ratio_source} lokal: "
                f"CT {lp_ct_primary:g}/{lp_ct_secondary:g}, "
                f"VT {lp_vt_primary:g}/{lp_vt_secondary:g}."
            )
        else:
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
                    step=0.001,
                    format="%.5f",
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
    
            default_lp_ct_primary = assignment_ct_primary
            default_lp_ct_secondary = assignment_ct_secondary
            default_lp_vt_primary = assignment_vt_primary
            default_lp_vt_secondary = assignment_vt_secondary
    
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
                    step=0.001,
                    format="%.5f",
                )
    
            with col_tr2:
                lp_ct_secondary = st.number_input(
                    "Line CT Secondary (A)",
                    value=default_lp_ct_secondary,
                    min_value=0.001,
                    step=0.001,
                    format="%.5f",
                )
    
            with col_tr3:
                lp_vt_primary = st.number_input(
                    "Line VT Primary (V)",
                    value=default_lp_vt_primary,
                    min_value=0.001,
                    step=0.001,
                    format="%.5f",
                )
    
            with col_tr4:
                lp_vt_secondary = st.number_input(
                    "Line VT Secondary (V)",
                    value=default_lp_vt_secondary,
                    min_value=0.001,
                    step=0.001,
                    format="%.5f",
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
                        step=0.00001,
                        format="%.5f",
                    )
    
                with col_z1b:
                    x1 = st.number_input(
                        "X1 / X1' (ohm or ohm/km)",
                        value=default_x1,
                        step=0.00001,
                        format="%.5f",
                    )
    
            elif positive_sequence_mode == "Z_PHI":
                col_z1a, col_z1b = st.columns(2)
    
                with col_z1a:
                    z1_mag = st.number_input(
                        "Z1 Magnitude",
                        value=0.4275,
                        step=0.00001,
                        format="%.5f",
                    )
    
                with col_z1b:
                    phi1_deg = st.number_input(
                        "Phi1 Angle (deg)",
                        value=79.22,
                        step=0.00001,
                        format="%.5f",
                    )
    
            elif positive_sequence_mode == "X_PHI":
                col_z1a, col_z1b = st.columns(2)
    
                with col_z1a:
                    x1 = st.number_input(
                        "X1 / X1' (ohm or ohm/km)",
                        value=0.42,
                        step=0.00001,
                        format="%.5f",
                    )
    
                with col_z1b:
                    phi1_deg = st.number_input(
                        "Phi1 Angle (deg)",
                        value=79.22,
                        step=0.00001,
                        format="%.5f",
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
                        step=0.00001,
                        format="%.5f",
                    )
    
                with col_z0b:
                    x0 = st.number_input(
                        "X0 / X0' (ohm or ohm/km)",
                        value=default_x0,
                        step=0.00001,
                        format="%.5f",
                    )
    
            elif zero_sequence_mode == "RE_RL_XE_XL":
                col_z0a, col_z0b = st.columns(2)
    
                with col_z0a:
                    re_rl = st.number_input(
                        "RE/RL",
                        value=3.125,
                        step=0.00001,
                        format="%.5f",
                    )
    
                with col_z0b:
                    xe_xl = st.number_input(
                        "XE/XL",
                        value=2.976,
                        step=0.00001,
                        format="%.5f",
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
                        step=0.00001,
                        format="%.5f",
                    )
    
                with col_z0b:
                    z0_z1_angle_deg = st.number_input(
                        "Z0/Z1 Angle (deg)",
                        value=0.0,
                        step=0.00001,
                        format="%.5f",
                    )
    
            elif zero_sequence_mode == "KL":
                col_z0a, col_z0b = st.columns(2)
    
                with col_z0a:
                    kl_mag = st.number_input(
                        "kL Magnitude",
                        value=0.70,
                        step=0.00001,
                        format="%.5f",
                    )
    
                with col_z0b:
                    kl_angle_deg = st.number_input(
                        "kL Angle (deg)",
                        value=0.0,
                        step=0.00001,
                        format="%.5f",
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


def resolve_end_analysis_context(end_side: str, feature_label: str):
    if "line_param" not in st.session_state:
        st.warning("Silakan lakukan Line Parameter terlebih dahulu.")
        return None

    line_param = st.session_state["line_param"]
    local_gi_label, remote_gi_label = infer_gi_names_from_line_name(
        line_param.get("line_name", "")
    )

    if end_side == "local":
        if "phasors" not in st.session_state:
            st.warning("Selesaikan Local End > Phasor terlebih dahulu.")
            return None
        if "fault_type_result" not in st.session_state:
            st.warning("Selesaikan Local End > Fault Type terlebih dahulu.")
            return None
        return {
            "label": local_gi_label,
            "phasors": st.session_state["phasors"],
            "fault_type_result": st.session_state["fault_type_result"],
            "prefault_phasors": st.session_state.get("prefault_phasors"),
            "line_param": line_param,
            "invertible": False,
        }

    if "remote_phasors" not in st.session_state:
        st.warning("Selesaikan Remote End > Phasor terlebih dahulu.")
        return None
    if "remote_fault_type_result" not in st.session_state:
        st.warning("Selesaikan Remote End > Fault Type terlebih dahulu.")
        return None

    phasors = st.session_state["remote_phasors"]
    prefault_phasors = st.session_state.get("remote_prefault_phasors")
    if st.checkbox(
        f"Balik arus remote untuk {feature_label}",
        value=False,
        key=f"{feature_label.lower().replace(' ', '_')}_invert_remote_current",
        help="Aktifkan jika arah arus rekaman remote perlu dibaca sebagai arus dari GI remote menuju line.",
    ):
        phasors = invert_current_phasors(phasors)
        if prefault_phasors is not None:
            prefault_phasors = invert_current_phasors(prefault_phasors)

    return {
        "label": remote_gi_label,
        "phasors": phasors,
        "fault_type_result": st.session_state["remote_fault_type_result"],
        "prefault_phasors": prefault_phasors,
        "line_param": line_param,
        "invertible": True,
    }


def render_high_resistance_check(end_side: str):
    ctx = resolve_end_analysis_context(end_side, "HR Check")
    if ctx is None:
        return

    suffix = end_side
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
            step=0.001,
            format="%.5f",
            key=f"hr_rf_threshold_{suffix}",
        )
    with col_hr2:
        angle_deviation_threshold_deg = st.number_input(
            "Angle Deviation Threshold (deg)",
            value=10.0,
            min_value=1.0,
            step=0.001,
            format="%.5f",
            key=f"hr_angle_threshold_{suffix}",
        )
    with col_hr3:
        distance_deviation_threshold_percent = st.number_input(
            "Distance Deviation Threshold (%)",
            value=15.0,
            min_value=1.0,
            step=0.001,
            format="%.5f",
            key=f"hr_distance_threshold_{suffix}",
        )

    try:
        hr_result = detect_high_resistance_fault(
            phasors=ctx["phasors"],
            line_param=ctx["line_param"],
            fault_type_result=ctx["fault_type_result"],
            rf_threshold_ohm=rf_threshold_ohm,
            angle_deviation_threshold_deg=angle_deviation_threshold_deg,
            distance_deviation_threshold_percent=distance_deviation_threshold_percent,
        )
        st.session_state["high_resistance_result" if end_side == "local" else "remote_high_resistance_result"] = hr_result

        st.markdown("### Hasil Deteksi")
        col_a, col_b, col_c, col_d, col_e = st.columns(5)
        col_a.metric("Location", ctx["label"])
        col_b.metric("Selected Loop", hr_result["selected_loop"])
        col_c.metric("High Resistance", "Suspected" if hr_result["high_resistance_suspected"] else "No")
        col_d.metric("Rf Estimate", f'{hr_result["Rf_est_ohm"]:.3f} Î©')
        col_e.metric("Analysis Confidence", f'{hr_result["analysis_confidence"]}/10')

        if hr_result["high_resistance_suspected"]:
            st.warning("Indikasi gangguan high resistance terdeteksi. Hasil single-ended perlu diberi status UNCERTAIN.")
        else:
            st.success(f"Belum ada indikasi kuat gangguan high resistance. HR evidence score: {hr_result['evidence_score']}/10.")

        st.info(explain_high_resistance_result(hr_result))
        for warning in hr_result.get("warnings", []):
            st.warning(warning)

        st.markdown("### Detail Perhitungan")
        st.dataframe(
            build_high_resistance_dataframe(hr_result).style.format(
                {"Value": lambda x: f"{x:.6f}" if isinstance(x, (int, float)) else x}
            ),
            use_container_width=True,
        )

        st.markdown("### Perbandingan Metode Estimasi Jarak")
        distance_df = pd.DataFrame(
            {
                "Method": ["Reactance-based", "Magnitude-based", "Projection-based"],
                "Distance km": [hr_result["distance_x_km"], hr_result["distance_mag_km"], hr_result["distance_projection_km"]],
                "Distance %": [hr_result["distance_x_percent"], hr_result["distance_mag_percent"], hr_result["distance_projection_percent"]],
            }
        )
        st.dataframe(distance_df.style.format({"Distance km": "{:.3f}", "Distance %": "{:.2f}"}), use_container_width=True)
        st.plotly_chart(px.bar(distance_df, x="Method", y="Distance km", title=f"Perbandingan Estimasi Jarak Gangguan - {ctx['label']}", text_auto=".2f"), use_container_width=True)

        st.markdown("### R-X Position")
        z1_total = ctx["line_param"]["Z1_total"]
        z_app = hr_result["Zapp"]
        rx_df = pd.DataFrame({"Point": ["Origin", "Z1 Total", "Zapp"], "R": [0.0, z1_total.real, z_app.real], "X": [0.0, z1_total.imag, z_app.imag]})
        fig_rx = px.scatter(rx_df, x="R", y="X", text="Point", title=f"Posisi Zapp terhadap Z1 Total - {ctx['label']}")
        fig_rx.add_shape(type="line", x0=0, y0=0, x1=z1_total.real, y1=z1_total.imag)
        fig_rx.add_shape(type="line", x0=0, y0=0, x1=z_app.real, y1=z_app.imag, line=dict(dash="dash"))
        fig_rx.update_traces(textposition="top center")
        fig_rx.update_layout(xaxis_title="R (ohm)", yaxis_title="X (ohm)")
        st.plotly_chart(fig_rx, use_container_width=True)
    except Exception as e:
        st.error("Analisis high resistance gagal.")
        st.exception(e)


def render_single_ended_analysis(end_side: str):
    ctx = resolve_end_analysis_context(end_side, "Single-End")
    if ctx is None:
        return

    suffix = end_side
    result_key = "single_ended_result" if end_side == "local" else "remote_single_ended_result"
    df_key = "single_ended_df" if end_side == "local" else "remote_single_ended_df"
    context_key = "single_ended_fault_context" if end_side == "local" else "remote_single_ended_fault_context"
    line_param = ctx["line_param"]
    fault_type_result = ctx["fault_type_result"]

    st.markdown("### Input Perhitungan")
    line_param = select_effective_line_param_for_calculation(
        line_param,
        f"single_ended_{suffix}",
    )
    col_se1, col_se2, col_se3, col_se4 = st.columns(4)
    col_se1.metric("End", ctx["label"])
    col_se2.metric("Fault Type", fault_type_result.get("fault_type", "-"))
    col_se3.metric("Line Length", f'{line_param["length_km"]:.6f} km')
    col_se4.metric("Z1/km", f'{line_param["Z1_per_km"].real:.4f} + j{line_param["Z1_per_km"].imag:.4f}')

    st.markdown("### Metode Rekomendasi Jarak")
    single_ended_fault_context = st.selectbox(
        "Konteks gangguan single-ended",
        ["internal_line_fault", "reverse_or_backfeed_external_fault"],
        format_func=lambda value: {
            "internal_line_fault": "Gangguan internal pada saluran yang dianalisis",
            "reverse_or_backfeed_external_fault": "Backfeed/reverse: fault eksternal atau di belakang relay",
        }.get(value, value),
        index=0,
        key=f"single_context_{suffix}",
    )
    recommended_method = st.selectbox(
        "Pilih metode jarak utama",
        ["reactance", "projection", "magnitude"],
        index=0,
        key=f"single_method_{suffix}",
    )

    if st.button("Calculate Single-Ended Fault Location", key=f"calculate_single_{suffix}"):
        try:
            single_result = calculate_single_ended_fault_location(
                phasors=ctx["phasors"],
                fault_type_result=fault_type_result,
                line_param=line_param,
                recommended_method=recommended_method,
                prefault_phasors=ctx["prefault_phasors"],
                fault_context=single_ended_fault_context,
            )
            single_result["line_length_source"] = line_param.get("length_source", "Line Parameter")
            single_result["line_length_km_used"] = line_param["length_km"]
            single_df = build_single_ended_result_dataframe(single_result)
            st.session_state[result_key] = single_result
            st.session_state[df_key] = single_df
            st.session_state[context_key] = single_ended_fault_context
            st.success("Single-ended fault location berhasil dihitung.")
        except Exception as e:
            st.error("Perhitungan single-ended gagal.")
            st.exception(e)

    if result_key not in st.session_state:
        return

    single_result = st.session_state[result_key]
    single_df = st.session_state[df_key]
    previous_length = single_result.get("line_length_km_used")
    previous_source = single_result.get("line_length_source")
    if previous_length is not None and abs(float(previous_length) - float(line_param["length_km"])) > 1e-9:
        st.warning(
            "Sumber/panjang line yang dipilih sudah berubah dari hasil Single-End tersimpan. "
            "Klik Calculate Single-Ended Fault Location ulang agar hasil memakai referensi jarak terbaru."
        )
    elif previous_source and previous_source != line_param.get("length_source", "Line Parameter"):
        st.warning(
            "Sumber panjang line yang dipilih berbeda dari hasil Single-End tersimpan. "
            "Klik Calculate Single-Ended Fault Location ulang agar metadata hasil ikut terbaru."
        )

    st.markdown("### Hasil Utama")
    col_r1, col_r2, col_r3, col_r4, col_r5 = st.columns(5)
    single_external_context = bool(single_result.get("external_context"))
    col_r1.metric("End", ctx["label"])
    col_r2.metric("Signed Distance" if single_external_context else "Recommended Distance", f'{single_result["recommended_distance_km"]:.3f} km')
    col_r3.metric("Distance %", f'{single_result["recommended_distance_percent"]:.2f} %')
    col_r4.metric("Zapp", f'{single_result["Zapp_R"]:.3f} + j{single_result["Zapp_X"]:.3f} Î©')
    col_r5.metric("Status", single_result["status"])

    if single_result["status"] == "VALID":
        st.success("Hasil single-ended berada dalam batas normal.")
    elif single_result["status"] == "CHECK":
        st.warning("Hasil single-ended perlu dicek ulang dengan waveform, SOE, dan data lapangan.")
    else:
        st.error("Hasil single-ended tidak pasti. Cek polaritas, line parameter, dan fault type.")

    st.info(explain_single_ended_status(single_result["status"]))
    if single_external_context:
        st.info("Mode backfeed/reverse aktif: jarak single-ended ditampilkan sebagai koordinat signed dari terminal relay.")
    for warning in single_result.get("warnings", []):
        st.warning(warning)

    st.markdown("### Detail Perhitungan")
    st.dataframe(single_df.style.format({"Value": lambda x: f"{x:.6f}" if isinstance(x, (int, float)) else x}), use_container_width=True)

    st.markdown("### Perbandingan Metode Jarak")
    distance_df = pd.DataFrame(
        {
            "Method": ["Magnitude", "Reactance", "Projection", "Signed/Recommended"],
            "Distance km": [single_result["distance_mag_km"], single_result["distance_x_km"], single_result["distance_projection_km"], single_result["recommended_distance_km"]],
            "Distance %": [single_result["distance_mag_percent"], single_result["distance_x_percent"], single_result["distance_projection_percent"], single_result["recommended_distance_percent"]],
        }
    )
    st.dataframe(distance_df.style.format({"Distance km": "{:.3f}", "Distance %": "{:.2f}"}), use_container_width=True)
    st.plotly_chart(px.bar(distance_df, x="Method", y="Distance km", text_auto=".2f", title=f"Perbandingan Estimasi Jarak Single-Ended - {ctx['label']}"), use_container_width=True)

    st.markdown("### Diagram R-X")
    z1_total = line_param["Z1_total"]
    zapp = single_result["Zapp"]
    z_recommended_line = single_result["recommended_distance_km"] * line_param["Z1_per_km"]
    rx_df = pd.DataFrame(
        {
            "Point": ["Origin", "Z1 Total", "Zapp", "Projected Fault Point"],
            "R": [0.0, z1_total.real, zapp.real, z_recommended_line.real],
            "X": [0.0, z1_total.imag, zapp.imag, z_recommended_line.imag],
        }
    )
    fig_rx = px.scatter(rx_df, x="R", y="X", text="Point", title=f"Single-Ended R-X Diagram - {ctx['label']}")
    fig_rx.add_shape(type="line", x0=0, y0=0, x1=z1_total.real, y1=z1_total.imag)
    fig_rx.add_shape(type="line", x0=0, y0=0, x1=zapp.real, y1=zapp.imag, line=dict(dash="dash"))
    fig_rx.add_shape(type="line", x0=0, y0=0, x1=z_recommended_line.real, y1=z_recommended_line.imag, line=dict(dash="dot"))
    fig_rx.update_traces(textposition="top center")
    fig_rx.update_layout(xaxis_title="R (ohm)", yaxis_title="X (ohm)", yaxis=dict(scaleanchor="x", scaleratio=1))
    st.plotly_chart(fig_rx, use_container_width=True)


def render_simple_rx_locus(end_side: str):
    ctx, context_message = get_rx_locus_context_from_session(end_side)
    if context_message:
        st.info(context_message)
        return

    label = ctx["label"]
    transformer_data = st.session_state.get(
        "remote_transformer_data" if end_side == "remote" else "local_transformer_data",
        {},
    )
    secondary_scale = impedance_secondary_scale_from_transformer(transformer_data)
    loop_options = ["AG", "BG", "CG", "AB", "BC", "CA"]

    st.markdown(f"### R-X Locus Trajectory - {label}")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        loop_name = st.selectbox(
            "Loop impedance",
            loop_options,
            index=loop_options.index(ctx["default_loop"]) if ctx["default_loop"] in loop_options else 0,
            key=f"rx_locus_loop_{end_side}",
        )
    with col2:
        pre_cycles = st.number_input("Pre-fault cycles", value=2.0, min_value=0.0, max_value=20.0, step=0.5, key=f"rx_locus_pre_{end_side}")
    with col3:
        post_cycles = st.number_input("Post-fault cycles", value=8.0, min_value=1.0, max_value=80.0, step=0.5, key=f"rx_locus_post_{end_side}")
    with col4:
        density = st.selectbox("Point density", ["Every sample", "1/4 cycle", "1/2 cycle", "1 cycle"], key=f"rx_locus_density_{end_side}")

    st.markdown("### Relay Zone Overlay")
    locus_zone_settings = []
    show_zone_overlay = st.toggle(
        "Tampilkan zona proteksi distance relay",
        value=True,
        key=f"rx_locus_show_zone_{end_side}",
    )
    if show_zone_overlay:
        zone_setting_base = st.selectbox(
            "Zone setting base",
            ["primary", "secondary"],
            index=0,
            format_func=lambda value: {
                "primary": "Spreadsheet zone values are primary ohm",
                "secondary": "Spreadsheet zone values are relay secondary ohm",
            }[value],
            key=f"rx_locus_zone_setting_base_{end_side}",
            help="Jika nilai X reach dan R reach di spreadsheet adalah ohm relay secondary, pilih secondary agar dikonversi ke primary ohm memakai CT/VT Signal Assignment.",
        )
        if zone_setting_base == "secondary" and secondary_scale is None:
            st.warning("Rasio CT/VT dari Signal Assignment belum tersedia, zona secondary belum bisa dikonversi ke primary.")
        elif zone_setting_base == "secondary":
            st.caption(f"Konversi zona secondary -> primary memakai faktor 1/{secondary_scale:.9f}.")

        try:
            distance_settings_df = read_google_spreadsheet_table_cached(
                st.session_state.get("database_spreadsheet_url", ""),
                st.session_state.get("distance_settings_sheet_name", "distance_settings"),
            )
            distance_settings_df = make_streamlit_safe_columns(distance_settings_df)
            distance_columns = detect_locus_distance_setting_columns(distance_settings_df)
            substation_col = distance_columns.get("substation")
            bay_col = distance_columns.get("bay")

            substation_options = sorted_nonempty_values(distance_settings_df, substation_col)
            substation_labels = ["Semua GI/Substation"] + substation_options
            default_substation = label.replace("GI ", "").strip().upper()
            default_index = 0
            for idx, option in enumerate(substation_labels):
                if default_substation and option.upper().replace(" ", "") == default_substation.replace(" ", ""):
                    default_index = idx
                    break

            col_set1, col_set2, col_set3 = st.columns([1.4, 1.6, 1.2])
            with col_set1:
                selected_substation = st.selectbox(
                    "GI / Substation",
                    substation_labels,
                    index=default_index,
                    key=f"rx_locus_substation_{end_side}",
                )

            filtered_settings_df = distance_settings_df
            if selected_substation != "Semua GI/Substation" and substation_col:
                filtered_settings_df = filtered_settings_df[
                    filtered_settings_df[substation_col].astype(str).str.strip() == selected_substation
                ].reset_index(drop=True)

            bay_labels = ["Semua Bay"] + sorted_nonempty_values(filtered_settings_df, bay_col)
            with col_set2:
                selected_bay = st.selectbox("Bay", bay_labels, key=f"rx_locus_bay_{end_side}")
            if selected_bay != "Semua Bay" and bay_col:
                filtered_settings_df = filtered_settings_df[
                    filtered_settings_df[bay_col].astype(str).str.strip() == selected_bay
                ].reset_index(drop=True)

            with col_set3:
                extra_filter = st.text_input("Cari tambahan", value="", key=f"rx_locus_filter_{end_side}").strip()
            if extra_filter:
                mask = filtered_settings_df.apply(
                    lambda row: extra_filter.lower() in " ".join(str(value).lower() for value in row.values),
                    axis=1,
                )
                filtered_settings_df = filtered_settings_df[mask].reset_index(drop=True)

            if filtered_settings_df.empty:
                st.warning("Tidak ada baris distance_settings yang cocok dengan filter.")
            else:
                row_labels = build_locus_setting_row_labels(filtered_settings_df, distance_columns)
                selected_label = st.selectbox(
                    "Pilih setting relay distance",
                    row_labels,
                    key=f"rx_locus_setting_row_{end_side}",
                )
                selected_row = filtered_settings_df.iloc[row_labels.index(selected_label)]
                locus_zone_settings = extract_locus_zone_settings(selected_row, distance_columns, loop_name)
                if zone_setting_base == "secondary":
                    locus_zone_settings = scale_locus_zone_settings(
                        locus_zone_settings,
                        1.0 / secondary_scale if secondary_scale else 1.0,
                    )

                if locus_zone_settings:
                    st.dataframe(
                        pd.DataFrame(locus_zone_settings).style.format(
                            {
                                "x_reach_ohm": "{:.3f}",
                                "r_reach_ohm": "{:.3f}",
                            }
                        ),
                        use_container_width=True,
                    )
                else:
                    st.warning("Baris setting terpilih belum memiliki X reach dan R reach yang cukup untuk Z1/Z2/Z3.")
        except Exception as e:
            st.warning("Setting distance relay belum dapat dibaca dari spreadsheet.")
            st.caption("Pastikan sheet `distance_settings` tersedia pada Database Spreadsheet URL di tab Setup DB.")
            st.exception(e)

    plot_focus_mode = st.selectbox(
        "Plot focus",
        ["relay_zones", "all_trajectory"],
        format_func=lambda value: {
            "relay_zones": "Fokus zona proteksi",
            "all_trajectory": "Tampilkan seluruh trajectory",
        }[value],
        index=0,
        key=f"rx_locus_focus_{end_side}",
    )

    fig_locus, trajectory_df, meta, build_warning = build_rx_locus_figure_from_session(end_side)
    if build_warning:
        st.warning(build_warning)
    if fig_locus is None:
        return
    summary_key_suffix = "local" if end_side == "local" else "remote"
    st.session_state[f"rx_locus_summary_fig_{summary_key_suffix}"] = fig_locus
    st.session_state[f"rx_locus_summary_meta_{summary_key_suffix}"] = meta

    st.plotly_chart(fig_locus, use_container_width=True)
    with st.expander("Trajectory Data", expanded=False):
        st.dataframe(
            trajectory_df.style.format(
                {
                    "time_s": "{:.6f}",
                    "relative_time_s": "{:.6f}",
                    "R_ohm": "{:.6f}",
                    "X_ohm": "{:.6f}",
                    "Z_mag_ohm": "{:.6f}",
                    "Z_angle_deg": "{:.3f}",
                }
            ),
            use_container_width=True,
            height=260,
        )


with tab8:
    st.subheader("High Resistance Fault Detection")
    local_hr_label, remote_hr_label = infer_gi_names_from_line_name(
        st.session_state.get("line_param", {}).get("line_name", "")
    )
    hr_local_tab, hr_remote_tab = st.tabs([local_hr_label, remote_hr_label])
    with hr_local_tab:
        render_high_resistance_check("local")
    with hr_remote_tab:
        render_high_resistance_check("remote")


with tab9:
    st.subheader("Single-Ended Fault Locator")
    st.write(
        "Fitur ini menghitung estimasi jarak gangguan dari satu ujung relay distance "
        "berdasarkan fasor, jenis gangguan, dan parameter saluran."
    )
    se_local_tab, se_remote_tab = st.tabs(["Local End", "Remote End"])
    with se_local_tab:
        render_single_ended_analysis("local")
    with se_remote_tab:
        render_single_ended_analysis("remote")


with tab11:
    st.subheader("R-X Locus")
    st.write(
        "Halaman ini menggambar trajectory apparent impedance terhadap waktu untuk GI lokal atau GI remote. "
        "Setiap titik berasal dari fasor sliding DFT satu siklus pada cursor waktu berbeda."
    )
    rx_local_tab, rx_remote_tab = st.tabs(["Local End", "Remote End"])
    with rx_local_tab:
        render_simple_rx_locus("local")
    with rx_remote_tab:
        render_simple_rx_locus("remote")


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
        st.stop()

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
    phasor_plot_group = st.radio(
        "Pilih kelompok fasor untuk perbandingan",
        ["Voltage", "Current"],
        horizontal=True,
        key="two_ended_phasor_plot_group",
    )
    phasor_plot_signals = ["Va", "Vb", "Vc"] if phasor_plot_group == "Voltage" else ["Ia", "Ib", "Ic"]

    remote_phasor_plot_source = st.radio(
        "Sumber fasor remote",
        ["Uploaded remote", "Adapted remote for DE"],
        horizontal=True,
        key="two_ended_remote_phasor_plot_source",
        help=(
            "Uploaded remote menampilkan fasor sesuai rekaman remote. Adapted remote memakai fasor "
            "yang sudah dikoreksi arah/polaritas/sudut setelah perhitungan DE tersedia."
        ),
    )
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
            if "fault_phase_voltage" in sync_reference_options:
                default_sync_reference = "fault_phase_voltage"
            elif "auto_voltage_sine" in sync_reference_options:
                default_sync_reference = "auto_voltage_sine"
            elif "fault_phase_current" in sync_reference_options:
                default_sync_reference = "fault_phase_current"
            elif "ground_current" in sync_reference_options:
                default_sync_reference = "ground_current"

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

    if False and local_fault_window:
        col_tws1, col_tws2 = st.columns(2)

        with col_tws1:
            tws_time_reference = st.selectbox(
                "Referensi waktu TWS",
                ["detected_fault_cursor", "cfg_trigger_time"],
                format_func=lambda item: {
                    "detected_fault_cursor": "Detected fault cursor + CFG start time",
                    "cfg_trigger_time": "CFG trigger timestamp relay",
                }[item],
                key="tws_time_reference",
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
                key="tws_velocity_factor",
                help="Overhead line umumnya sekitar 0.95-0.99c; kabel biasanya lebih rendah.",
            )

        local_tws_time = get_absolute_event_time(
            metadata,
            local_fault_window["fault_time"],
            tws_time_reference,
        )
        remote_tws_time = get_absolute_event_time(
            remote_metadata,
            remote_fault_window["fault_time"],
            tws_time_reference,
        )

        if local_tws_time is None or remote_tws_time is None:
            st.warning(
                "Timestamp absolut local/remote belum bisa dibaca. Pastikan CFG memiliki start/trigger timestamp "
                "yang valid atau gunakan rekaman hasil export COMTRADE yang menyimpan timestamp lengkap."
            )
        else:
            tws_result = calculate_time_based_fault_location(
                local_tws_time,
                remote_tws_time,
                line_param["length_km"],
                tws_velocity_factor,
            )
            st.session_state["time_based_fault_location_result"] = tws_result

            col_twsr1, col_twsr2, col_twsr3, col_twsr4 = st.columns(4)
            col_twsr1.metric("TWS from Local", f'{tws_result["distance_from_local_km"]:.3f} km')
            col_twsr2.metric("TWS from Remote", f'{tws_result["distance_from_remote_km"]:.3f} km')
            col_twsr3.metric("TWS Position", f'{tws_result["distance_from_local_percent"]:.2f} %')
            col_twsr4.metric("Î”t Local-Remote", f'{tws_result["delta_t_s"] * 1e6:.3f} Âµs')

            tws_detail_df = pd.DataFrame(
                [
                    {"Parameter": "Local absolute time", "Value": str(local_tws_time)},
                    {"Parameter": "Remote absolute time", "Value": str(remote_tws_time)},
                    {"Parameter": "Delta t s", "Value": tws_result["delta_t_s"]},
                    {"Parameter": "Velocity factor", "Value": tws_result["velocity_factor"]},
                    {"Parameter": "Propagation velocity km/s", "Value": tws_result["propagation_velocity_km_s"]},
                    {"Parameter": "One-end travel time s", "Value": tws_result["one_end_travel_time_s"]},
                    {"Parameter": "Distance from local km", "Value": tws_result["distance_from_local_km"]},
                    {"Parameter": "Distance from remote km", "Value": tws_result["distance_from_remote_km"]},
                ]
            )

            with st.expander("Detail Time-Based / TWS Calculation"):
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
                        metadata,
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

