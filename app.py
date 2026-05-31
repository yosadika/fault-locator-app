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
    find_column,
)
from single_ended import (
    calculate_single_ended_fault_location,
    build_single_ended_result_dataframe,
)
from app_helpers import (
    MAX_PLOT_POINTS,
    OHM,
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
    render_fault_cursor,
    render_se_formula_expander,
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
from tabs import (
    double_ended as double_ended_tab,
    line_parameter as line_parameter_tab,
    signal_assignment as signal_assignment_tab,
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

# Accordion sidebar: JS client-side — tutup semua expander lain saat satu dibuka
import streamlit.components.v1 as _stc
_stc.html("""
<script>
(function () {
    function setup() {
        var sb = window.parent.document.querySelector('[data-testid="stSidebar"]');
        if (!sb) { setTimeout(setup, 300); return; }

        function bind() {
            sb.querySelectorAll('details').forEach(function (det) {
                var summary = det.querySelector('summary');
                if (!summary || summary._acc) return;
                summary._acc = true;
                summary.addEventListener('click', function () {
                    if (!det.open) {
                        sb.querySelectorAll('details[open]').forEach(function (other) {
                            if (other !== det) {
                                    var s = other.querySelector('summary');
                                    if (s) s.click();
                                }
                        });
                    }
                });
            });
        }

        bind();
        new MutationObserver(bind).observe(sb, { childList: true, subtree: true });
    }
    setup();
})();
</script>
""", height=0)

st.title("Transmission Fault Locator")

_local_has_cfg = bool(st.session_state.get("local_cfg_file") is not None or st.session_state.get("case_local_cfg_bytes"))
_local_has_dat = bool(st.session_state.get("local_dat_file") is not None or st.session_state.get("case_local_dat_bytes"))
_local_complete = _local_has_cfg and _local_has_dat
with st.sidebar.expander("Upload Local End COMTRADE", expanded=(_local_has_cfg or _local_has_dat) and not _local_complete):
    cfg_file = st.file_uploader("Local .cfg", key="local_cfg_file")
    dat_file = st.file_uploader("Local .dat", key="local_dat_file")

_remote_has_cfg = bool(st.session_state.get("remote_cfg_file") is not None or st.session_state.get("case_remote_cfg_bytes"))
_remote_has_dat = bool(st.session_state.get("remote_dat_file") is not None or st.session_state.get("case_remote_dat_bytes"))
_remote_complete = _remote_has_cfg and _remote_has_dat
with st.sidebar.expander("Upload Remote End COMTRADE", expanded=(_remote_has_cfg or _remote_has_dat) and not _remote_complete):
    st.caption("Opsional. Diisi jika ingin menghitung double-ended.")
    remote_cfg_file = st.file_uploader("Remote .cfg", key="remote_cfg_file")
    remote_dat_file = st.file_uploader("Remote .dat", key="remote_dat_file")

_creds_db_url = (
    str(st.session_state.get("database_spreadsheet_url", "") or "").strip()
    or str(get_config_secret("DATABASE_SPREADSHEET_URL", "") or "").strip()
)
_creds_complete = bool(
    st.session_state.get("runtime_credentials_loaded_name") or _creds_db_url
)
_creds_file_present = st.session_state.get("sidebar_credentials_upload") is not None
_creds_applied = bool(st.session_state.get("runtime_credentials_loaded_name"))
with st.sidebar.expander("Credentials", expanded=_creds_file_present and not _creds_applied):
    _loaded_name = st.session_state.get("runtime_credentials_loaded_name")
    if _loaded_name:
        st.caption(f"Aktif: {_loaded_name}")
    _sb_cred = st.file_uploader(
        "credentials.toml / .json",
        type=["toml", "json"],
        key="sidebar_credentials_upload",
        label_visibility="collapsed",
        help="Upload credentials.toml atau credentials.json untuk mengisi URL database otomatis.",
    )
    if _sb_cred is not None:
        _sb_fp = hashlib.md5(_sb_cred.getvalue()).hexdigest()
        if st.session_state.get("runtime_credentials_fingerprint") != _sb_fp:
            _sb_payload, _sb_err = parse_runtime_credentials_upload(_sb_cred)
            if _sb_err:
                st.error(_sb_err)
            else:
                st.session_state["runtime_credentials"] = _sb_payload
                st.session_state["runtime_credentials_loaded_name"] = _sb_cred.name
                st.session_state["runtime_credentials_fingerprint"] = _sb_fp
                _sb_applied = apply_runtime_credentials(_sb_payload)
                st.success(
                    "Credentials diterapkan: " + ", ".join(_sb_applied)
                    if _sb_applied else "Credentials terbaca."
                )
        elif st.session_state.get("runtime_credentials"):
            apply_runtime_credentials(st.session_state["runtime_credentials"])
    st.caption("Pengaturan lengkap & clear credentials tersedia di tab Setup DB.")

with st.sidebar.expander("Filter GI / Line", expanded=False):
    _sidebar_db_url = (
        str(st.session_state.get("database_spreadsheet_url", "") or "").strip()
        or str(get_config_secret("DATABASE_SPREADSHEET_URL", "") or "").strip()
    )
    _sidebar_line_sheet = st.session_state.get("line_data_sheet_name", "line_impedance")
    if not _sidebar_db_url:
        st.caption("Isi Database Spreadsheet URL di Setup DB untuk mengaktifkan filter.")
    else:
        try:
            # ULTG dan Segment → dari line_impedance
            _gf_li = read_google_spreadsheet_table_cached(_sidebar_db_url, _sidebar_line_sheet)
            _gf_li = make_streamlit_safe_columns(_gf_li)
            _c_li_ultg = find_column(_gf_li, ["ULTG"])
            _c_li_seg  = find_column(_gf_li, ["SEGMENT"])

            # GI, Bay, Line → dari distance_settings
            _ds_sheet = st.session_state.get("distance_settings_sheet_name", "distance_settings")
            _gf_ds = read_google_spreadsheet_table_cached(_sidebar_db_url, _ds_sheet)
            _gf_ds = make_streamlit_safe_columns(_gf_ds)
            _c_ds_ultg = find_column(_gf_ds, ["ULTG"])
            _c_ds_gi   = find_column(_gf_ds, ["GI"])
            _c_ds_bay  = find_column(_gf_ds, ["BAY", "BAY PHT"])
            _c_ds_line = find_column(_gf_ds, ["LINE"])

            def _nv(v):
                """Normalisasi float-integer: '1.0'→'1', '2.0'→'2'."""
                try:
                    f = float(v)
                    if f == int(f):
                        return str(int(f))
                except (ValueError, TypeError):
                    pass
                return str(v)

            def _sf_vals(df, col):
                if col is None or col not in df.columns:
                    return []
                vals = df[col].dropna().astype(str).str.strip()
                return sorted(
                    {_nv(v) for v in vals if v and v.lower() not in ("nan", "none")},
                    key=str.upper,
                )

            def _sb_select(label, real_vals, key):
                _ph = f"Pilih {label}"
                _opts = [_ph] + real_vals
                _cur = st.session_state.get(key, _ph)
                if _cur not in _opts:
                    _cur = _ph
                return st.selectbox(
                    label, _opts,
                    index=_opts.index(_cur),
                    key=key,
                    label_visibility="collapsed",
                )

            def _active(v):
                return bool(v) and v != "Semua" and not v.startswith("Pilih ")

            def _filt(df, col, val):
                if not _active(val) or not col or col not in df.columns:
                    return df
                norm = _nv(val)
                return df[df[col].astype(str).str.strip().apply(_nv) == norm]

            # — ULTG (dari line_impedance) —
            _sel_ultg = _sb_select("ULTG", _sf_vals(_gf_li, _c_li_ultg), "sidebar_filter_ultg")
            _gf_li_u  = _filt(_gf_li, _c_li_ultg, _sel_ultg)
            _gf_ds_u  = _filt(_gf_ds, _c_ds_ultg, _sel_ultg)

            # — Segment (dari line_impedance, difilter ULTG) —
            _sb_select("Segment", _sf_vals(_gf_li_u, _c_li_seg), "sidebar_filter_segment")

            def _bay_line_opts(df, bay_col, line_col):
                """Bangun opsi gabungan 'BAY / LINE' dari dataframe."""
                if not bay_col or not line_col or bay_col not in df.columns or line_col not in df.columns:
                    return []
                d = df[[bay_col, line_col]].dropna().astype(str)
                d = d[~d[bay_col].str.strip().str.lower().isin(("nan", "none"))]
                combined = d[bay_col].str.strip() + " / " + d[line_col].str.strip().apply(_nv)
                return sorted(set(combined), key=str.upper)

            def _parse_bay_line(val):
                """Uraikan 'BAY / LINE' → (bay, line). Gunakan rsplit agar nama bay yang mengandung '/' aman."""
                if not _active(val):
                    return "", ""
                parts = val.rsplit(" / ", 1)
                return parts[0], (parts[1] if len(parts) > 1 else "")

            # — GI Lokal (dari distance_settings, difilter ULTG) —
            _sel_gi_l = _sb_select("GI Lokal", _sf_vals(_gf_ds_u, _c_ds_gi), "sidebar_filter_gi_local")
            _gf_ds_gl = _filt(_gf_ds_u, _c_ds_gi, _sel_gi_l)

            # — Bay / Line GI Lokal (gabungan, difilter GI Lokal) —
            _sel_bl_l = _sb_select("Bay / Line GI Lokal", _bay_line_opts(_gf_ds_gl, _c_ds_bay, _c_ds_line), "sidebar_filter_bay_line_local")
            _bay_l, _line_l = _parse_bay_line(_sel_bl_l)
            st.session_state["sidebar_filter_bay_local"]  = _bay_l
            st.session_state["sidebar_filter_line_local"] = _line_l

            # — GI Remote (dari distance_settings, difilter ULTG) —
            _sel_gi_r = _sb_select("GI Remote", _sf_vals(_gf_ds_u, _c_ds_gi), "sidebar_filter_gi_remote")
            _gf_ds_gr = _filt(_gf_ds_u, _c_ds_gi, _sel_gi_r)

            # — Bay / Line GI Remote (gabungan, difilter GI Remote) —
            _sel_bl_r = _sb_select("Bay / Line GI Remote", _bay_line_opts(_gf_ds_gr, _c_ds_bay, _c_ds_line), "sidebar_filter_bay_line_remote")
            _bay_r, _line_r = _parse_bay_line(_sel_bl_r)
            st.session_state["sidebar_filter_bay_remote"]  = _bay_r
            st.session_state["sidebar_filter_line_remote"] = _line_r

        except Exception as _sidebar_gi_err:
            st.caption(f"Filter GI belum dapat dimuat: {_sidebar_gi_err}")

# Sinkronisasi sidebar filters ke Tower Schedule, R-X Locus, dan Line Parameter
def _sb_ia(v):
    return bool(v) and v != "Semua" and not v.startswith("Pilih ")

_sb_ultg  = st.session_state.get("sidebar_filter_ultg", "")
_sb_seg   = st.session_state.get("sidebar_filter_segment", "")
_sb_gi_l  = st.session_state.get("sidebar_filter_gi_local", "")
_sb_bay_l = st.session_state.get("sidebar_filter_bay_local", "")
_sb_gi_r  = st.session_state.get("sidebar_filter_gi_remote", "")
_sb_bay_r = st.session_state.get("sidebar_filter_bay_remote", "")

# Tower Schedule: override langsung (bukan setdefault) selama data belum dimuat
if "tower_schedule_df" not in st.session_state:
    if _sb_ia(_sb_ultg):
        st.session_state["tower_schedule_pre_ultg"] = _sb_ultg
    if _sb_ia(_sb_seg):
        st.session_state["tower_schedule_pre_segment"] = _sb_seg

# R-X Locus: sync substation dan bay saat sidebar berubah
_sb_locus_key = f"{_sb_gi_l}|{_sb_bay_l}|{_sb_gi_r}|{_sb_bay_r}"
if _sb_locus_key != st.session_state.get("_sb_locus_sync_key", ""):
    st.session_state["_sb_locus_sync_key"] = _sb_locus_key
    if _sb_ia(_sb_gi_l):
        st.session_state["rx_locus_substation_local"] = _sb_gi_l
    if _sb_ia(_sb_bay_l):
        st.session_state["rx_locus_bay_local"] = _sb_bay_l
    if _sb_ia(_sb_gi_r):
        st.session_state["rx_locus_substation_remote"] = _sb_gi_r
    if _sb_ia(_sb_bay_r):
        st.session_state["rx_locus_bay_remote"] = _sb_bay_r

# Line Parameter: auto-select Database Excel Line Data saat sidebar GI/Segment aktif
_sb_li_key = f"{_sb_gi_l}|{_sb_seg}"
_sb_li_active = _sb_ia(_sb_gi_l) or _sb_ia(_sb_seg)
if _sb_li_key != st.session_state.get("_sb_li_sync_key", ""):
    st.session_state["_sb_li_sync_key"] = _sb_li_key
    if _sb_li_active:
        st.session_state["line_parameter_source"] = "Database Excel Line Data"
elif _sb_li_active and st.session_state.get("line_parameter_source", "Input Manual") == "Input Manual":
    # Sidebar aktif tapi source masih di default — pastikan sync terjadi (misal setelah case restore)
    st.session_state["line_parameter_source"] = "Database Excel Line Data"

st.sidebar.divider()
_case_loaded = bool(st.session_state.get("_restored_case_hash"))
with st.sidebar.expander("Case Storage", expanded=False):
    case_archive_file = st.file_uploader("Load Case (.zip)", type=["zip"], key="case_archive_file")
if case_archive_file is not None:
    import hashlib as _hashlib
    _archive_bytes = case_archive_file.getvalue()
    _archive_hash = _hashlib.md5(_archive_bytes).hexdigest()
    if st.session_state.get("_restored_case_hash") != _archive_hash:
        try:
            restore_case_archive(_archive_bytes)
            st.session_state["_restored_case_hash"] = _archive_hash
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
        render_fault_cursor(
            end="remote",
            assigned_df_key="remote_assigned_df",
            metadata_key="remote_metadata",
            transformer_key="remote_transformer_data",
            fault_window_key="remote_fault_window",
            fault_detection_key="remote_fault_detection",
            key_prefix="remote_fc",
        )

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

    _line_name = st.session_state.get("line_param", {}).get("line_name", "") or "case"
    _line_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", _line_name).strip("_") or "case"
    _auto_case_name = f"porlungcase_{_line_slug}"
    st.session_state["case_name"] = _auto_case_name

    case_filename, case_archive_bytes = build_case_archive_bytes(_auto_case_name)
    st.caption(f"File akan disimpan sebagai: `{case_filename}`")
    _col_exp, _col_ref, _col_pad = st.columns([3, 2, 7])
    with _col_exp:
        st.download_button(
            "Export Case ZIP",
            data=case_archive_bytes,
            file_name=case_filename,
            mime="application/zip",
            key="export_case_zip",
            use_container_width=True,
        )
    with _col_ref:
        st.button(
            "↺ Refresh Nama File",
            key="refresh_case_name",
            help="Perbarui nama file ZIP ke waktu terkini. Nama file menggunakan timestamp saat halaman terakhir dimuat — klik tombol ini agar timestamp mencerminkan waktu sekarang sebelum mengekspor.",
            use_container_width=True,
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

    def _norm_filter(s):
        """Normalisasi untuk matching: lowercase, strip, collapse spasi di sekitar hyphen."""
        return str(s).strip().lower().replace(" - ", "-").replace("- ", "-").replace(" -", "-")

    # Resolve sidebar-synced ULTG: exact match → pakai langsung; normalized match → pakai nilai asli spreadsheet; no match → Semua
    _pre_ultg_synced = st.session_state.get("tower_schedule_pre_ultg", "Semua")
    if _sb_ia(_pre_ultg_synced) and _pre_ultg_synced not in pre_ultg_options:
        _matched_ultg = next((o for o in pre_ultg_options if _norm_filter(o) == _norm_filter(_pre_ultg_synced)), None)
        st.session_state["tower_schedule_pre_ultg"] = _matched_ultg if _matched_ultg else "Semua"

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
        _pre_seg_synced = st.session_state.get("tower_schedule_pre_segment", "Semua")
        if _pre_seg_synced not in pre_segment_options:
            if _sb_ia(_pre_seg_synced):
                # Resolve: normalized match → pakai nilai asli spreadsheet; no match → Semua
                _matched_seg = next((o for o in pre_segment_options if _norm_filter(o) == _norm_filter(_pre_seg_synced)), None)
                st.session_state["tower_schedule_pre_segment"] = _matched_seg if _matched_seg else "Semua"
            else:
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
    signal_assignment_tab.render(df)

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
    render_fault_cursor(
        end="local",
        assigned_df_key="assigned_df",
        metadata_key="local_metadata",
        transformer_key="local_transformer_data",
        fault_window_key="fault_window",
        fault_detection_key="fault_detection",
        key_prefix="local_fc",
    )


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
    line_parameter_tab.render()

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
        col_d.metric("Rf Estimate", f'{hr_result["Rf_est_ohm"]:.3f} {OHM}')
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
    col_r4.metric("Zapp", f'{single_result["Zapp_R"]:.3f} + j{single_result["Zapp_X"]:.3f} {OHM}')
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

    render_se_formula_expander(single_result, line_param)

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
    double_ended_tab.render()
