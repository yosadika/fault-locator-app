import streamlit as st
import tempfile
import os
import math
import cmath
import re
import json
import io
import zipfile
import textwrap
from datetime import datetime, timedelta, timezone
from urllib.parse import quote
import folium
import numpy as np
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components
from streamlit_folium import st_folium

from comtrade_reader import read_comtrade
from signal_assignment import apply_signal_assignment
from fault_detection import (
    detect_fault_inception,
    build_fault_window,
    estimate_sampling_rate,
    calculate_rms_sliding,
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
    read_google_spreadsheet_table,
    get_google_spreadsheet_sheet_names,
    extract_google_spreadsheet_id,
    make_unique_columns,
    detect_impedance_columns,
    extract_impedance_from_row,
    build_row_label,
)
from single_ended import (
    calculate_single_ended_fault_location,
    build_single_ended_result_dataframe,
)


MAX_PLOT_POINTS = 6000
DEFAULT_TOWER_SCHEDULE_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "<TOWER_SCHEDULE_SPREADSHEET_ID>/edit?usp=sharing"
)
DEFAULT_TOWER_SCHEDULE_SHEET = "tower_schedule"
DEFAULT_CASE_DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/<CASE_DRIVE_FOLDER_ID>?usp=sharing"
DEFAULT_CASE_DRIVE_FOLDER_ID = "<CASE_DRIVE_FOLDER_ID>"
OPENWEATHER_LIGHTNING_ENDPOINT = "https://demo.openweathermap.org/lightning/1.0/data"
OPENWEATHER_ONECALL_CURRENT_ENDPOINT = "https://api.openweathermap.org/data/4.0/onecall/current"
OPENWEATHER_ONECALL_15MIN_ENDPOINT = "https://api.openweathermap.org/data/4.0/onecall/timeline/15min"
XWEATHER_LIGHTNING_FLASH_CLOSEST_ENDPOINT = "https://data.api.xweather.com/lightning/flash/closest"
ACCUWEATHER_BASE_URL = "https://dataservice.accuweather.com"
ACCUWEATHER_LIGHTNING_BASE_URL = "https://api.accuweather.com"
LOCAL_TIMEZONE = timezone(timedelta(hours=7))


@st.cache_data(show_spinner="Membaca file COMTRADE...")
def read_comtrade_cached(cfg_bytes: bytes, dat_bytes: bytes, cfg_name: str = "", dat_name: str = ""):
    cfg_path = None
    dat_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".cfg") as temp_cfg:
            temp_cfg.write(cfg_bytes)
            cfg_path = temp_cfg.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".dat") as temp_dat:
            temp_dat.write(dat_bytes)
            dat_path = temp_dat.name

        return read_comtrade(cfg_path, dat_path)
    finally:
        for temp_path in [cfg_path, dat_path]:
            if temp_path:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass


@st.cache_data(ttl=1800, show_spinner="Membaca Google Spreadsheet...")
def read_google_spreadsheet_table_cached(url_or_id: str, sheet_name: str):
    return read_google_spreadsheet_table(url_or_id, sheet_name)


@st.cache_data(ttl=1800, show_spinner="Membaca Tower Schedule...")
def read_google_spreadsheet_query_cached(url_or_id: str, sheet_name: str, query: str):
    spreadsheet_id = extract_google_spreadsheet_id(url_or_id)
    csv_url = (
        f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/gviz/tq"
        f"?tqx=out:csv&sheet={quote(str(sheet_name))}&tq={quote(str(query))}"
    )
    df = pd.read_csv(csv_url)
    df.columns = make_unique_columns(df.columns)
    return df


@st.cache_data(ttl=1800, show_spinner="Membaca daftar sheet...")
def get_google_spreadsheet_sheet_names_cached(url_or_id: str):
    return get_google_spreadsheet_sheet_names(url_or_id)


class RestoredUpload:
    def __init__(self, name: str, content: bytes):
        self.name = name
        self._content = content
        self.size = len(content)

    def getvalue(self):
        return self._content


CASE_FILE_KEYS = {
    "local_cfg": ("case_local_cfg_name", "case_local_cfg_bytes", "local.cfg"),
    "local_dat": ("case_local_dat_name", "case_local_dat_bytes", "local.dat"),
    "remote_cfg": ("case_remote_cfg_name", "case_remote_cfg_bytes", "remote.cfg"),
    "remote_dat": ("case_remote_dat_name", "case_remote_dat_bytes", "remote.dat"),
}

CASE_STATE_EXCLUDE_PREFIXES = (
    "local_cfg_file",
    "local_dat_file",
    "remote_cfg_file",
    "remote_dat_file",
    "summary_weather_lightning_openweather",
    "summary_weather_lightning_xweather",
)
CASE_STATE_EXCLUDE_KEYS = {
    "case_archive_file",
    "case_local_cfg_bytes",
    "case_local_dat_bytes",
    "case_remote_cfg_bytes",
    "case_remote_dat_bytes",
    "openweather_lightning_api_key",
    "xweather_client_id",
    "xweather_client_secret",
    "accuweather_api_key",
}


def extract_google_drive_folder_id(url_or_id: str):
    text = str(url_or_id or "").strip()
    if not text:
        return ""
    match = re.search(r"/folders/([A-Za-z0-9_-]+)", text)
    if match:
        return match.group(1)
    return text


def make_case_json_safe(value):
    if isinstance(value, pd.DataFrame):
        return {
            "__type__": "dataframe",
            "columns": [str(col) for col in value.columns],
            "records": make_case_json_safe(value.to_dict("records")),
        }
    if isinstance(value, pd.Series):
        return make_case_json_safe(value.to_dict())
    if isinstance(value, np.ndarray):
        return make_case_json_safe(value.tolist())
    if isinstance(value, complex):
        return {"__type__": "complex", "real": value.real, "imag": value.imag}
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if pd.isna(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (datetime,)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): make_case_json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_case_json_safe(item) for item in value]
    if isinstance(value, bytes):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def restore_case_json_value(value):
    if isinstance(value, dict) and value.get("__type__") == "dataframe":
        return pd.DataFrame(value.get("records", []), columns=value.get("columns"))
    if isinstance(value, dict) and value.get("__type__") == "complex":
        return complex(value.get("real", 0.0), value.get("imag", 0.0))
    if isinstance(value, dict):
        return {key: restore_case_json_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [restore_case_json_value(item) for item in value]
    return value


def build_case_state_snapshot():
    snapshot = {}
    for key, value in st.session_state.items():
        if key in CASE_STATE_EXCLUDE_KEYS:
            continue
        if any(str(key).startswith(prefix) for prefix in CASE_STATE_EXCLUDE_PREFIXES):
            continue
        snapshot[str(key)] = make_case_json_safe(value)
    return snapshot


def build_case_archive_bytes(case_name: str = ""):
    created_at = datetime.now().isoformat(timespec="seconds")
    case_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(case_name or "").strip()).strip("_")
    if not case_slug:
        line_name = st.session_state.get("line_param", {}).get("line_name", "case")
        case_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(line_name or "case")).strip("_") or "case"
    manifest = {
        "schema": "transmission_fault_locator_case_v1",
        "created_at": created_at,
        "case_name": case_slug,
        "drive_folder_id": st.session_state.get("case_drive_folder_id", DEFAULT_CASE_DRIVE_FOLDER_ID),
        "files": {},
    }
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for logical_name, (name_key, bytes_key, fallback_name) in CASE_FILE_KEYS.items():
            content = st.session_state.get(bytes_key)
            if content:
                filename = st.session_state.get(name_key, fallback_name)
                archive_path = f"records/{logical_name}/{filename}"
                archive.writestr(archive_path, content)
                manifest["files"][logical_name] = {
                    "name": filename,
                    "path": archive_path,
                    "size": len(content),
                }
        archive.writestr("case_state.json", json.dumps(build_case_state_snapshot(), indent=2, ensure_ascii=False))
        archive.writestr("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
    archive_buffer.seek(0)
    filename = f"{case_slug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    return filename, archive_buffer.getvalue()


def restore_case_archive(archive_bytes: bytes):
    with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as archive:
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
        state = json.loads(archive.read("case_state.json").decode("utf-8"))
        for key, value in state.items():
            st.session_state[key] = restore_case_json_value(value)
        for logical_name, (name_key, bytes_key, fallback_name) in CASE_FILE_KEYS.items():
            file_info = manifest.get("files", {}).get(logical_name)
            if file_info and file_info.get("path") in archive.namelist():
                st.session_state[name_key] = file_info.get("name", fallback_name)
                st.session_state[bytes_key] = archive.read(file_info["path"])
    st.session_state["case_restore_message"] = "Case berhasil dimuat. Aplikasi memakai file dan parameter dari arsip case."


def get_restored_upload(logical_name: str):
    name_key, bytes_key, _ = CASE_FILE_KEYS[logical_name]
    content = st.session_state.get(bytes_key)
    if not content:
        return None
    return RestoredUpload(st.session_state.get(name_key, f"{logical_name}.dat"), content)


def get_google_drive_service():
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise RuntimeError(
            "Dependency Google Drive belum terpasang. Jalankan pip install -r requirements.txt."
        ) from exc

    scopes = ["https://www.googleapis.com/auth/drive.file"]
    credentials_info = None
    try:
        credentials_info = st.secrets.get("gdrive_service_account")
    except Exception:
        credentials_info = None

    credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if credentials_info:
        credentials = service_account.Credentials.from_service_account_info(
            dict(credentials_info),
            scopes=scopes,
        )
    elif credentials_path:
        credentials = service_account.Credentials.from_service_account_file(
            credentials_path,
            scopes=scopes,
        )
    else:
        raise RuntimeError(
            "Kredensial Google Drive belum tersedia. Gunakan st.secrets['gdrive_service_account'] "
            "atau env GOOGLE_APPLICATION_CREDENTIALS, lalu share folder Drive ke email service account."
        )
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def upload_case_archive_to_drive(filename: str, archive_bytes: bytes, folder_id: str):
    try:
        from googleapiclient.http import MediaIoBaseUpload
    except ImportError as exc:
        raise RuntimeError(
            "Dependency Google Drive belum terpasang. Jalankan pip install -r requirements.txt."
        ) from exc
    service = get_google_drive_service()
    media = MediaIoBaseUpload(io.BytesIO(archive_bytes), mimetype="application/zip", resumable=False)
    metadata = {
        "name": filename,
        "parents": [folder_id],
        "mimeType": "application/zip",
    }
    return service.files().create(body=metadata, media_body=media, fields="id,name,webViewLink").execute()


def downsample_xy(x_values, y_values, max_points: int = MAX_PLOT_POINTS):
    length = len(x_values)
    if length <= max_points:
        return x_values, y_values

    step = max(1, int(math.ceil(length / max_points)))
    return x_values[::step], y_values[::step]


def validate_uploaded_extension(uploaded_file, expected_extension: str, label: str):
    if uploaded_file is None:
        return True

    file_name = str(uploaded_file.name or "").lower()
    if file_name.endswith(expected_extension.lower()):
        return True

    st.error(
        f"{label} harus berekstensi `{expected_extension}`. "
        f"File yang dipilih: `{uploaded_file.name}`. "
        "Pada smartphone, simpan file ke Files/Downloads dan pastikan ekstensi tidak berubah."
    )
    return False

def downsample_dataframe_for_plot(df: pd.DataFrame, x_col: str, y_cols, max_points: int = MAX_PLOT_POINTS):
    if df is None or len(df) <= max_points:
        return df

    selected_cols = [x_col] + [col for col in y_cols if col in df.columns]
    step = max(1, int(math.ceil(len(df) / max_points)))
    return df.loc[df.index[::step], selected_cols]


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

    def _one_based_index(df):
        if not isinstance(df, pd.DataFrame) or df.empty:
            return df
        try:
            is_default_zero_index = list(df.index) == list(range(len(df.index)))
        except Exception:
            is_default_zero_index = False
        if not is_default_zero_index:
            return df

        adjusted_df = df.copy()
        adjusted_df.index = range(1, len(adjusted_df.index) + 1)
        adjusted_df.index.name = df.index.name or "No"
        return adjusted_df

    def _one_based_display_data(data):
        if type(data).__name__ == "Styler" and hasattr(data, "data"):
            source_df = data.data
            adjusted_df = _one_based_index(source_df)
            if adjusted_df is not source_df:
                data.data.index = adjusted_df.index
            return data

        if isinstance(data, pd.DataFrame):
            return _one_based_index(data)

        return data

    def _table_source(data):
        if type(data).__name__ == "Styler" and hasattr(data, "data"):
            data = _one_based_display_data(data)
            return data.data, data.to_html()

        if isinstance(data, pd.DataFrame):
            display_df = _one_based_index(data)
            return display_df, display_df.to_html(index=True, escape=True)

        try:
            df = pd.DataFrame(data)
            display_df = _one_based_index(df)
            return display_df, display_df.to_html(index=True, escape=True)
        except Exception:
            return None, None

    def printable_dataframe(data=None, *args, **kwargs):
        display_data = _one_based_display_data(data)
        result = original_dataframe(display_data, *args, **kwargs)
        source_df, table_html = _table_source(display_data)

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


def choose_summary_fault_signals(local_fault_type_result, remote_fault_type_result):
    fault_type = (
        (local_fault_type_result or {}).get("fault_type")
        or (remote_fault_type_result or {}).get("fault_type")
        or ""
    )

    voltage_channel = fault_phase_to_voltage_channel(fault_type) or "Va"
    current_channel = fault_phase_to_current_channel(fault_type) or "Ia"

    return fault_type, voltage_channel, current_channel


def build_summary_focus_waveform(
    local_df,
    remote_df,
    local_fault_window,
    remote_fault_window,
    channel,
    title,
    seconds_before=0.08,
    seconds_after=0.12,
    remote_time_shift_s=0.0,
):
    fig = go.Figure()

    if local_df is not None and local_fault_window is not None and channel in local_df.columns:
        local_time = local_df["time"] - local_fault_window["fault_time"]
        local_mask = (local_time >= -seconds_before) & (local_time <= seconds_after)
        local_x, local_y = downsample_xy(local_time[local_mask], local_df.loc[local_mask, channel])
        fig.add_trace(
            go.Scatter(
                x=local_x,
                y=local_y,
                mode="lines",
                name=f"Local {channel}",
                line=dict(width=1.4),
            )
        )

    if remote_df is not None and remote_fault_window is not None and channel in remote_df.columns:
        remote_time = remote_df["time"] - remote_fault_window["fault_time"] + remote_time_shift_s
        remote_mask = (remote_time >= -seconds_before) & (remote_time <= seconds_after)
        remote_x, remote_y = downsample_xy(remote_time[remote_mask], remote_df.loc[remote_mask, channel])
        fig.add_trace(
            go.Scatter(
                x=remote_x,
                y=remote_y,
                mode="lines",
                name=f"Remote {channel}",
                line=dict(width=1.4, dash="dash"),
            )
        )

    fig.add_vline(
        x=0.0,
        line_dash="solid",
        annotation_text="Fault",
        annotation_position="top",
    )

    fig.update_layout(
        title=title,
        xaxis_title="Aligned Time from Fault (s)",
        yaxis_title="Instantaneous Primary Magnitude",
        legend_title="Signal",
        xaxis=dict(range=[-seconds_before, seconds_after], autorange=False),
    )

    return fig


def estimate_summary_disturbance_cause(fault_type_result, high_resistance_result):
    fault_type = str((fault_type_result or {}).get("fault_type", "")).upper()
    hr_suspected = bool((high_resistance_result or {}).get("high_resistance_suspected"))

    if hr_suspected:
        return (
            "Pohon / benda asing (indikasi resistif)",
            "Indikasi ini muncul karena pola impedansi terlihat resistif. Tetap validasi dengan inspeksi lapangan.",
        )

    if fault_type in ["ABC", "ABCG", "3PH", "3P"]:
        return (
            "Power swing / gangguan 3 fasa (perlu validasi)",
            "Gangguan tiga fasa perlu dibandingkan dengan event relay, osilasi daya, dan kondisi sistem.",
        )

    if "G" in fault_type and any(phase in fault_type for phase in ["A", "B", "C"]):
        return (
            "Petir / flashover satu fasa ke tanah (indikasi awal)",
            "Gangguan satu fasa ke tanah yang cepat sering cocok dengan flashover/petir, tetapi penyebab final tetap perlu bukti eksternal.",
        )

    return (
        "Belum dapat ditentukan otomatis",
        "Aplikasi belum melihat pola yang cukup kuat untuk mengklasifikasikan penyebab gangguan.",
    )


def single_ended_plot_score(single_result):
    if not single_result:
        return 0.0
    base_score = {
        "VALID": 9.0,
        "CHECK": 6.0,
        "UNCERTAIN": 3.0,
    }.get(str(single_result.get("status", "")).upper(), 5.0)
    warning_count = len(single_result.get("warnings", []) or [])
    return max(0.0, min(10.0, base_score - 0.4 * warning_count))


def build_summary_location_plot(
    line_param,
    local_gi_label,
    remote_gi_label,
    single_result,
    remote_single_result,
    two_result,
    reverse_two_result,
):
    if not line_param:
        return None

    line_length = float(line_param.get("length_km", 0.0) or 0.0)
    if line_length <= 0:
        return None

    points = []

    if single_result:
        points.append(
            {
                "label": f"SE {local_gi_label}",
                "distance": float(single_result.get("recommended_distance_km", 0.0)),
                "score": single_ended_plot_score(single_result),
                "symbol": "circle",
                "color": "#009e73",
            }
        )

    if remote_single_result:
        remote_position = build_remote_single_signed_position(
            line_length_km=line_length,
            remote_single_result=remote_single_result,
            scenario=st.session_state.get("two_ended_fault_scenario", "normal_internal_line_fault"),
            two_result=two_result,
        )
        points.append(
            {
                "label": f"SE {remote_gi_label}",
                "distance": remote_position["distance_from_local_km"],
                "score": single_ended_plot_score(remote_single_result),
                "symbol": "circle-open",
                "color": "#e67300",
            }
        )

    if two_result:
        points.append(
            {
                "label": f"DE {line_param.get('line_name', 'Original')}",
                "distance": float(two_result.get("distance_from_original_local_km", two_result.get("distance_km", 0.0))),
                "score": float(st.session_state.get("two_ended_quality", {}).get("quality_score", 10.0)),
                "symbol": "diamond",
                "color": "#2563eb",
            }
        )

    if reverse_two_result:
        points.append(
            {
                "label": f"DE {remote_gi_label}-{local_gi_label}",
                "distance": float(reverse_two_result.get("distance_from_original_local_km", reverse_two_result.get("distance_km", 0.0))),
                "score": float(st.session_state.get("two_ended_reverse_quality", {}).get("quality_score", 10.0)),
                "symbol": "diamond-open",
                "color": "#7c3aed",
            }
        )

    if not points:
        return None

    point_distances = [float(point["distance"]) for point in points]
    external_padding = max(0.05 * line_length, 1.0)
    x_min = min(0.0, min(point_distances) - external_padding)
    x_max = max(line_length, max(point_distances) + external_padding)

    marker_rows = []
    for point in points:
        distance = float(point["distance"])
        score = max(0.0, min(10.0, float(point["score"])))
        track = "Double-ended" if str(point["label"]).startswith("DE ") else "Single-ended"
        marker_rows.append(
            {
                "Point": point["label"],
                "Distance km": distance,
                "Distance %": 100.0 * distance / line_length,
                "Score": score,
                "Track": track,
                "Color": point["color"],
                "Symbol": point["symbol"],
                "Legend Name": point["label"],
            }
        )

    sorted_marker_rows = sorted(marker_rows, key=lambda item: float(item["Distance km"]))
    min_gap_km = max(0.02 * line_length, 0.75)
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

    for row in marker_rows:
        row["Label"] = (
            f"<b>{row['Point']}</b><br>"
            f"{row['Distance km']:.2f} km ({row['Distance %']:.1f}%)<br>"
            f"{row['Score']:.1f}/10"
        )
        row["Annotation Ay"], row["Annotation Ax"] = label_layout.get(
            id(row),
            (-64, 0) if row["Track"] == "Double-ended" else (108, 0),
        )

    marker_df = pd.DataFrame(marker_rows)
    x_profile = [
        x_min + i * (x_max - x_min) / 300.0
        for i in range(301)
    ]

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
    plot_bg = "#0b1220" if is_dark_theme else "#ffffff"
    text_color = "#f8fafc" if is_dark_theme else "#111827"
    muted_text_color = "#cbd5e1" if is_dark_theme else "#475569"
    axis_title_color = "#f8fafc" if is_dark_theme else "#0f172a"
    annotation_bg = "rgba(15,23,42,0.92)" if is_dark_theme else "rgba(255,255,255,0.96)"
    annotation_border = "#94a3b8" if is_dark_theme else "#cbd5e1"
    terminal_color = "#f8fafc" if is_dark_theme else "#111827"

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.18, 0.82],
        vertical_spacing=0.08,
    )

    fig.add_shape(
        type="line",
        x0=0,
        x1=line_length,
        y0=0.5,
        y1=0.5,
        line=dict(color=muted_text_color, width=2),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=[0, line_length],
            y=[0.5, 0.5],
            mode="markers",
            marker=dict(size=12, color=[terminal_color, terminal_color], symbol="square"),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_annotation(
        x=0,
        y=0.5,
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
    fig.add_annotation(
        x=line_length,
        y=0.5,
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

    for _, row in marker_df.sort_values(["Distance km", "Track"]).iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row["Distance km"]],
                y=[0.5],
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

    for _, row in marker_df.iterrows():
        center = float(row["Distance km"])
        score = float(row["Score"])
        curve_width = max(line_length * (0.09 if row["Track"] == "Double-ended" else 0.06), 2.5)
        curve_y = [
            score / (1.0 + abs(x - center) / curve_width)
            for x in x_profile
        ]

        fig.add_trace(
            go.Scatter(
                x=x_profile,
                y=curve_y,
                mode="lines",
                line=dict(color=row["Color"], width=1.5),
                opacity=0.9,
                hoverinfo="skip",
                legendgroup=row["Point"],
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[row["Distance km"]],
                y=[score],
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
        fig.add_shape(
            type="line",
            x0=row["Distance km"],
            x1=row["Distance km"],
            y0=0,
            y1=score,
            line=dict(color=row["Color"], width=1.4),
            row=2,
            col=1,
        )

        fig.add_annotation(
            x=row["Distance km"],
            y=score,
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

    fig.update_layout(
        title=f"Grafik SE dan DE - {line_param.get('line_name', '')}",
        template=plot_template,
        paper_bgcolor=plot_bg,
        plot_bgcolor=plot_bg,
        font=dict(color=text_color),
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
    fig.update_xaxes(
        range=[x_min, x_max],
        autorange=False,
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        color=text_color,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        range=[0.0, 1.0],
        autorange=False,
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        color=text_color,
        row=1,
        col=1,
    )
    fig.update_xaxes(
        title=dict(
            text=f"Distance from {local_gi_label} (km)",
            font=dict(color=axis_title_color),
        ),
        range=[x_min, x_max],
        autorange=False,
        zeroline=False,
        color=text_color,
        tickfont=dict(color=muted_text_color),
        gridcolor="rgba(148,163,184,0.22)" if is_dark_theme else "rgba(148,163,184,0.35)",
        row=2,
        col=1,
    )
    fig.update_yaxes(
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
        row=2,
        col=1,
    )

    return fig


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


def is_reverse_or_backfeed_scenario(scenario: str, two_result: dict | None = None):
    if scenario in ["reverse_or_backfeed_external_fault", "sotf_parallel_or_adjacent_line"]:
        return True

    remote_direction = str(
        (two_result or {}).get(
            "uploaded_remote_current_direction",
            (two_result or {}).get("remote_current_direction", ""),
        )
    )
    return remote_direction == "opposite_to_line"


def build_remote_single_signed_position(
    line_length_km: float,
    remote_single_result: dict,
    scenario: str,
    two_result: dict | None = None,
):
    remote_distance = float(remote_single_result["recommended_distance_km"])

    if is_reverse_or_backfeed_scenario(scenario, two_result):
        signed_distance_from_remote = -abs(remote_distance)
    else:
        signed_distance_from_remote = remote_distance

    distance_from_local = line_length_km - signed_distance_from_remote

    return {
        "signed_distance_from_remote_km": signed_distance_from_remote,
        "distance_from_local_km": distance_from_local,
        "distance_from_local_percent": distance_from_local / line_length_km * 100.0,
        "is_reverse_external": signed_distance_from_remote < 0,
    }


def classify_two_ended_operating_status(
    two_result,
    two_quality,
    line_param,
    local_single_result=None,
    remote_single_result=None,
    scenario="normal_internal_line_fault",
):
    """
    Memberi status konteks proteksi untuk membedakan gangguan internal saluran
    dari kasus reverse/backfeed/external fault pada line paralel atau line tetangga.
    """

    statuses = []
    notes = []
    recommendation = "Hasil DE dapat dipakai sebagai estimasi utama gangguan internal saluran yang direkam."
    can_use_de_distance = True

    L = float((line_param or {}).get("length_km", 0.0) or 0.0)
    distance = float((two_result or {}).get("distance_from_original_local_km", (two_result or {}).get("distance_km", 0.0)) or 0.0)
    remote_direction = str(
        (two_result or {}).get(
            "uploaded_remote_current_direction",
            (two_result or {}).get("remote_current_direction", "into_line"),
        )
    )
    quality_score = float((two_quality or {}).get("quality_score", 0.0) or 0.0)

    if scenario in ["reverse_or_backfeed_external_fault", "sotf_parallel_or_adjacent_line"]:
        statuses.extend(
            [
                "BACKFEED_OR_REVERSE_FAULT_SUSPECTED",
                "EXTERNAL_TO_IMPORTED_LINE_SUSPECTED",
                "DE_NOT_APPLICABLE_FOR_IMPORTED_LINE",
            ]
        )
        can_use_de_distance = False
        notes.append(
            "Mode backfeed/reverse aktif: rekaman yang dianalisis mungkin berasal dari line sehat/berbeban, sedangkan fault berada pada line paralel, line tetangga, atau peralatan di belakang terminal remote."
        )

    if remote_direction == "opposite_to_line":
        statuses.append("REMOTE_REVERSE_FAULT")
        notes.append(
            "Arah arus remote yang paling konsisten adalah opposite_to_line. Ini cocok dengan relay remote yang melihat fault pada zona reverse/belakang terminal."
        )
        if scenario != "normal_internal_line_fault":
            can_use_de_distance = False

    if L > 0 and (distance < -0.002 * L or distance > L * 1.002):
        statuses.append("DE_NOT_APPLICABLE_FOR_IMPORTED_LINE")
        can_use_de_distance = False
        notes.append(
            "Jarak DE berada di luar panjang saluran, sehingga pola ini lebih cocok diperlakukan sebagai external/reverse event atau kesalahan referensi rekaman."
        )

    if quality_score < 6.0 and scenario != "normal_internal_line_fault":
        statuses.append("DE_NOT_APPLICABLE_FOR_IMPORTED_LINE")
        can_use_de_distance = False

    if not statuses:
        statuses.append("NORMAL_INTERNAL_LINE_FAULT")

    # Buang duplikasi dengan tetap menjaga urutan kemunculan.
    statuses = list(dict.fromkeys(statuses))

    if not can_use_de_distance:
        recommendation = (
            "Jangan jadikan jarak DE dari rekaman ini sebagai jarak gangguan utama. "
            "Gunakan hasil single-ended local/remote sebagai pembanding arah dan besaran, "
            "lalu validasi dengan rekaman line yang benar-benar terganggu, event CB, SOE, dan proteksi reverse remote. Jika event ini terjadi saat energize, catat sebagai kemungkinan SOTF."
        )
    elif "REMOTE_REVERSE_FAULT" in statuses:
        recommendation = (
            "Ada indikasi remote reverse. Pakai hasil DE secara hati-hati dan cek apakah rekaman berasal dari saluran yang sama dengan saluran fault."
        )

    return {
        "primary_status": statuses[0],
        "statuses": statuses,
        "can_use_de_distance": can_use_de_distance,
        "recommendation": recommendation,
        "notes": notes,
        "remote_current_direction": remote_direction,
        "scenario": scenario,
    }

def get_index_at_time(df, time_value: float):
    return int((df["time"] - time_value).abs().idxmin())


def calculate_remote_aligned_dft_index(
    remote_df,
    local_fault_window,
    remote_fault_window,
    remote_time_shift_s,
):
    """
    Visual plot memakai remote_time - remote_fault_time + shift.
    Agar DFT remote berada di offset yang sama dengan DFT lokal, waktu cursor
    remote harus dikoreksi dengan arah shift yang berlawanan.
    """

    local_dft_offset = local_fault_window["dft_time"] - local_fault_window["fault_time"]
    aligned_remote_dft_time = (
        remote_fault_window["fault_time"]
        + local_dft_offset
        - remote_time_shift_s
    )

    return get_index_at_time(remote_df, aligned_remote_dft_time)


def choose_best_remote_dft_for_two_ended(
    local_phasors,
    remote_df,
    remote_fault_window,
    local_fault_window,
    line_param,
    remote_samples_per_cycle,
    remote_direction_mode,
    search_window_s=0.30,
):
    base_index = int(remote_fault_window["dft_index"])
    time_values = np.asarray(remote_df["time"], dtype=float)
    base_time = float(time_values[base_index])
    min_time = base_time - float(search_window_s)
    max_time = base_time + float(search_window_s)

    step = max(1, int(round(remote_samples_per_cycle / 4)))
    min_index = int(np.searchsorted(time_values, min_time, side="left"))
    max_index = int(np.searchsorted(time_values, max_time, side="right")) - 1
    min_index = max(remote_samples_per_cycle, min_index)
    max_index = min(len(remote_df) - 1, max_index)

    if min_index > max_index:
        return None, []

    candidate_indices = list(range(min_index, max_index + 1, step))
    candidate_indices.extend([base_index, min_index, max_index])
    candidate_indices = sorted(set(index for index in candidate_indices if min_index <= index <= max_index))

    candidates = []

    for candidate_index in candidate_indices:
        try:
            candidate_phasors = calculate_all_phasors(
                df=remote_df,
                cursor_index=candidate_index,
                samples_per_cycle=remote_samples_per_cycle,
            )
            candidate_phasors = add_sequence_components_to_phasor_dict(candidate_phasors)

            if remote_direction_mode == "auto_adapt_record":
                best_candidate, _ = choose_best_two_ended_adaptation(
                    local_phasors=local_phasors,
                    remote_phasors=candidate_phasors,
                    line_param=line_param,
                    angle_step_deg=5.0,
                )

                if best_candidate["result"] is None:
                    continue

                result = best_candidate["result"]
                quality = best_candidate["quality"]
                adapted_remote_phasors = best_candidate["adapted_remote_phasors"]
                ranking_score = best_candidate["ranking_score"]

            elif remote_direction_mode == "auto_current_direction_only":
                best_candidate, _ = choose_best_remote_current_direction(
                    local_phasors=local_phasors,
                    remote_phasors=candidate_phasors,
                    line_param=line_param,
                )

                if best_candidate["result"] is None:
                    continue

                result = best_candidate["result"]
                quality = best_candidate["quality"]
                adapted_remote_phasors = best_candidate["adapted_remote_phasors"]
                ranking_score = best_candidate["ranking_score"]

            else:
                result = calculate_positive_sequence_two_ended(
                    local_phasors=local_phasors,
                    remote_phasors=candidate_phasors,
                    line_param=line_param,
                    remote_current_direction=remote_direction_mode,
                )
                quality = evaluate_two_ended_quality(result, line_param)
                adapted_remote_phasors = candidate_phasors
                ranking_score = score_two_ended_for_local_search(result, quality, line_param)

            candidate_time = float(time_values[candidate_index])
            local_dft_offset = local_fault_window["dft_time"] - local_fault_window["fault_time"]
            remote_dft_offset = candidate_time - remote_fault_window["fault_time"]
            implied_shift_s = local_dft_offset - remote_dft_offset

            candidates.append(
                {
                    "remote_dft_index": int(candidate_index),
                    "remote_dft_time": candidate_time,
                    "remote_phasors": candidate_phasors,
                    "adapted_remote_phasors": adapted_remote_phasors,
                    "result": result,
                    "quality": quality,
                    "ranking_score": ranking_score,
                    "implied_shift_s": implied_shift_s,
                }
            )
        except Exception:
            continue

    candidates = sorted(candidates, key=lambda item: item["ranking_score"])
    return (candidates[0] if candidates else None), candidates


def score_two_ended_for_local_search(result, quality, line_param):
    L = float(line_param["length_km"])
    d = float(result["distance_km"])
    outside_km = max(0.0, -d, d - L)
    outside_penalty = outside_km / max(L, 1e-9) * 1000.0
    imag_penalty = abs(result["distance_complex"].imag) / max(L, 1e-9) * 100.0
    mismatch_penalty = quality.get("mismatch_ratio", 0.0) * 100.0
    quality_penalty = (10.0 - quality.get("quality_score", 0.0)) * 10.0
    return outside_penalty + imag_penalty + mismatch_penalty + quality_penalty


def clean_gi_name(raw_name: str, fallback: str):
    name = str(raw_name or "").strip()
    if not name:
        return fallback

    name = re.sub(r"(?i)\bGI\b", "", name)
    name = re.sub(r"[^A-Za-z0-9]+", " ", name).strip()

    if not name:
        return fallback

    return f"GI {name.upper()}"


def infer_gi_names_from_line_name(line_name: str):
    cleaned = str(line_name or "").strip()
    cleaned = cleaned.split("#", 1)[0]

    if "-" in cleaned:
        left, right = cleaned.split("-", 1)
        return (
            clean_gi_name(left, "GI Local"),
            clean_gi_name(right, "GI Remote"),
        )

    parts = re.split(r"\s+(?:to|ke|s/d|sd)\s+", cleaned, flags=re.IGNORECASE)
    if len(parts) >= 2:
        return (
            clean_gi_name(parts[0], "GI Local"),
            clean_gi_name(parts[1], "GI Remote"),
        )

    return "GI Local", "GI Remote"


def reverse_line_name(line_name: str):
    cleaned = str(line_name or "").strip()
    if not cleaned:
        return "Reverse Line"

    base, sep, suffix = cleaned.partition("#")
    suffix = f"{sep}{suffix}" if sep else ""

    if "-" in base:
        left, right = base.split("-", 1)
        return f"{right.strip()}-{left.strip()}{suffix}"

    parts = re.split(r"\s+(to|ke|s/d|sd)\s+", base, flags=re.IGNORECASE)
    if len(parts) >= 3:
        return f"{parts[2].strip()} {parts[1].strip()} {parts[0].strip()}{suffix}"

    return f"{cleaned} reverse"


def orient_remote_as_line_current(remote_phasors: dict, remote_direction: str):
    if remote_direction == "opposite_to_line":
        return invert_current_phasors(remote_phasors)

    return remote_phasors


def build_two_ended_reverse_result(
    local_phasors: dict,
    adapted_remote_phasors: dict,
    normal_result: dict,
    line_param: dict,
    local_label: str,
    remote_label: str,
):
    remote_direction = normal_result["remote_current_direction"]
    reverse_local_phasors = orient_remote_as_line_current(
        adapted_remote_phasors,
        remote_direction,
    )

    reverse_result = calculate_positive_sequence_two_ended(
        local_phasors=reverse_local_phasors,
        remote_phasors=local_phasors,
        line_param=line_param,
        remote_current_direction="into_line",
    )
    reverse_quality = evaluate_two_ended_quality(reverse_result, line_param)
    L = line_param["length_km"]

    reverse_result.update(
        {
            "calculation_reference_mode": "uploaded_remote_to_original_local",
            "calculation_local_label": remote_label,
            "calculation_remote_label": local_label,
            "uploaded_remote_current_direction": remote_direction,
            "distance_from_original_local_km": L - reverse_result["distance_km"],
            "distance_from_original_local_percent": (
                (L - reverse_result["distance_km"]) / L * 100.0
            ),
        }
    )

    return reverse_result, reverse_quality


def build_two_ended_comparison_dataframe(
    normal_result: dict,
    normal_quality: dict,
    reverse_result: dict,
    reverse_quality: dict,
    local_label: str,
    remote_label: str,
):
    rows = [
        {
            "Method": f"Double-ended {local_label} -> {remote_label}",
            "Reference Side": local_label,
            "Distance from Reference km": normal_result["distance_km"],
            f"Distance from {local_label} km": normal_result["distance_from_original_local_km"],
            f"Distance from {local_label} %": normal_result["distance_from_original_local_percent"],
            f"Distance from {remote_label} km": normal_result["distance_from_remote_km"],
            "Quality": normal_quality["quality_score"],
            "Mismatch Ratio": normal_quality["mismatch_ratio"],
            "Imag Distance km": normal_result["distance_complex"].imag,
            "Warnings": "; ".join(normal_quality["warnings"]) if normal_quality["warnings"] else "-",
        },
        {
            "Method": f"Double-ended {remote_label} -> {local_label}",
            "Reference Side": remote_label,
            "Distance from Reference km": reverse_result["distance_km"],
            f"Distance from {local_label} km": reverse_result["distance_from_original_local_km"],
            f"Distance from {local_label} %": reverse_result["distance_from_original_local_percent"],
            f"Distance from {remote_label} km": reverse_result["distance_km"],
            "Quality": reverse_quality["quality_score"],
            "Mismatch Ratio": reverse_quality["mismatch_ratio"],
            "Imag Distance km": reverse_result["distance_complex"].imag,
            "Warnings": "; ".join(reverse_quality["warnings"]) if reverse_quality["warnings"] else "-",
        },
    ]

    return pd.DataFrame(rows)


def override_line_param_length(line_param: dict, length_km: float, source_label: str):
    effective = dict(line_param)
    effective["length_km"] = float(length_km)
    effective["Z1_total"] = effective["Z1_per_km"] * float(length_km)
    effective["Z0_total"] = effective["Z0_per_km"] * float(length_km)
    effective["length_source"] = source_label
    return effective


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


def map_display_value(value, decimals=None, suffix=""):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    if decimals is not None:
        numeric_value = pd.to_numeric(str(value).replace(",", "."), errors="coerce")
        if pd.notna(numeric_value):
            return f"{float(numeric_value):.{decimals}f}{suffix}"
    return f"{value}{suffix}"


def map_detail_table(rows, title=None, compact=True):
    title_html = (
        f"<div style='font-weight:700; margin-bottom:6px;'>{title}</div>"
        if title
        else ""
    )
    value_cell_style = (
        "padding:1px 0; font-weight:600; white-space:nowrap;"
        if compact
        else "padding:2px 0; font-weight:600; white-space:normal; overflow-wrap:anywhere;"
    )
    body_html = "".join(
        "<tr>"
        f"<td style='padding:2px 14px 2px 0; color:#475569; white-space:nowrap; vertical-align:top;'>{label}</td>"
        f"<td style='{value_cell_style}'>{value}</td>"
        "</tr>"
        for label, value in rows
    )
    return (
        "<div style='font-size:12px; line-height:1.35; min-width:260px; max-width:420px;'>"
        f"{title_html}"
        "<table style='border-collapse:collapse; width:100%; table-layout:auto;'>"
        f"{body_html}"
        "</table>"
        "</div>"
    )


def compact_tower_span_label(span_value):
    text = str(span_value or "").strip()
    if not text:
        return "-"
    hash_match = re.search(r"#\s*([A-Za-z0-9_-]+)", text)
    if hash_match:
        return f"#{hash_match.group(1)}"
    parts = text.split()
    return parts[-1] if parts else text


def google_maps_action_links(lat, lon, label="Location"):
    if lat is None or lon is None:
        return "-"
    lat_float = float(lat)
    lon_float = float(lon)
    label_query = quote(f"{label} {lat_float:.7f},{lon_float:.7f}")
    coord_query = f"{lat_float:.7f},{lon_float:.7f}"
    open_url = f"https://www.google.com/maps/search/?api=1&query={coord_query}&query_place_id={label_query}"
    direction_url = f"https://www.google.com/maps/dir/?api=1&destination={coord_query}&travelmode=driving"
    return (
        f"<a href='{open_url}' target='_blank' rel='noopener noreferrer'>Open Maps</a>"
        " &nbsp;|&nbsp; "
        f"<a href='{direction_url}' target='_blank' rel='noopener noreferrer'>Directions</a>"
    )


def fault_label_anchor_from_segment(fault_segment):
    if not fault_segment:
        return (-16, 12), "left"
    prev_lat = float(fault_segment["prev"]["lat"])
    prev_lon = float(fault_segment["prev"]["lon"])
    next_lat = float(fault_segment["next"]["lat"])
    next_lon = float(fault_segment["next"]["lon"])
    dx = next_lon - prev_lon
    dy = next_lat - prev_lat

    # Tempatkan label pada sisi yang paling menjauh dari arah garis span supaya
    # tidak menumpuk dengan label nomor tower yang mengikuti jalur.
    if abs(dx) >= abs(dy):
        if dy >= 0:
            return (-18, 76), "below"
        return (-18, -16), "above"
    if dx >= 0:
        return (226, 20), "left"
    return (-18, 20), "right"


WMO_WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}
THUNDERSTORM_WEATHER_CODES = {95, 96, 99}


def weather_code_label(code):
    if code is None or pd.isna(code):
        return "-"
    try:
        return WMO_WEATHER_CODES.get(int(code), f"WMO {int(code)}")
    except (TypeError, ValueError):
        return "-"


def safe_number_formatter(decimals=2):
    def _formatter(value):
        if value is None or pd.isna(value):
            return "-"
        try:
            return f"{float(value):.{decimals}f}"
        except (TypeError, ValueError):
            return str(value)

    return _formatter


def safe_display_number(value, decimals=1, suffix=""):
    if value is None:
        return "-"
    try:
        if pd.isna(value):
            return "-"
    except (TypeError, ValueError):
        pass
    try:
        return f"{float(value):.{decimals}f}{suffix}"
    except (TypeError, ValueError):
        return str(value)


def get_openweather_lightning_api_key():
    key = str(st.session_state.get("openweather_lightning_api_key", "") or "").strip()
    if key:
        return key

    try:
        key = str(st.secrets.get("OPENWEATHER_API_KEY", "") or "").strip()
        if key:
            return key
    except Exception:
        pass

    return str(os.environ.get("OPENWEATHER_API_KEY", "") or "").strip()


def get_xweather_credentials():
    client_id = str(st.session_state.get("xweather_client_id", "") or "").strip()
    client_secret = str(st.session_state.get("xweather_client_secret", "") or "").strip()
    if client_id and client_secret:
        return client_id, client_secret

    try:
        client_id = client_id or str(st.secrets.get("XWEATHER_CLIENT_ID", "") or "").strip()
        client_secret = client_secret or str(st.secrets.get("XWEATHER_CLIENT_SECRET", "") or "").strip()
    except Exception:
        pass

    client_id = client_id or str(os.environ.get("XWEATHER_CLIENT_ID", "") or "").strip()
    client_secret = client_secret or str(os.environ.get("XWEATHER_CLIENT_SECRET", "") or "").strip()
    return client_id, client_secret


def get_accuweather_api_key():
    key = str(st.session_state.get("accuweather_api_key", "") or "").strip()
    if key:
        return key

    try:
        key = str(st.secrets.get("ACCUWEATHER_API_KEY", "") or "").strip()
        if key:
            return key
    except Exception:
        pass

    return str(os.environ.get("ACCUWEATHER_API_KEY", "") or "").strip()


def _accuweather_value(unit_container: dict | None, preferred_unit: str = "Metric"):
    if not isinstance(unit_container, dict):
        return None
    preferred = unit_container.get(preferred_unit)
    if isinstance(preferred, dict):
        return preferred.get("Value")
    for value in unit_container.values():
        if isinstance(value, dict) and "Value" in value:
            return value.get("Value")
    return None


def _accuweather_get(url: str, api_key: str, params: dict | None = None):
    params = dict(params or {})
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(url, params=params, headers=headers, timeout=10)
    if response.status_code in [401, 403]:
        legacy_params = dict(params)
        legacy_params["apikey"] = api_key
        response = requests.get(url, params=legacy_params, timeout=10)
    return response


@st.cache_data(ttl=600, show_spinner=False)
def fetch_accuweather_current_weather(lat: float, lon: float, api_key: str, language: str = "id-id"):
    try:
        location_response = _accuweather_get(
            f"{ACCUWEATHER_BASE_URL}/locations/v1/geoposition/search",
            api_key,
            params={
                "q": f"{float(lat):.6f},{float(lon):.6f}",
                "language": language,
                "details": "false",
            },
        )
        if location_response.status_code != 200:
            try:
                payload = location_response.json()
            except Exception:
                payload = {"message": location_response.text}
            return {
                "error": payload.get("message", "AccuWeather location lookup failed."),
                "status_code": location_response.status_code,
            }
        location_payload = location_response.json()
        location_key = location_payload.get("Key")
        if not location_key:
            return {"error": "AccuWeather location key tidak ditemukan.", "status_code": None}

        current_response = _accuweather_get(
            f"{ACCUWEATHER_BASE_URL}/currentconditions/v1/{location_key}",
            api_key,
            params={
                "language": language,
                "details": "true",
            },
        )
        if current_response.status_code != 200:
            try:
                payload = current_response.json()
            except Exception:
                payload = {"message": current_response.text}
            return {
                "error": payload.get("message", "AccuWeather current conditions failed."),
                "status_code": current_response.status_code,
            }

        records = current_response.json()
        current = records[0] if records else {}
        wind = current.get("Wind", {}) or {}
        wind_speed = wind.get("Speed", {}) if isinstance(wind, dict) else {}
        wind_direction = wind.get("Direction", {}) if isinstance(wind, dict) else {}
        precip_1h = current.get("Precip1hr", {}) or {}

        return {
            "time": current.get("LocalObservationDateTime"),
            "weather_code": current.get("WeatherIcon"),
            "weather": current.get("WeatherText", "-"),
            "temperature_c": _accuweather_value(current.get("Temperature"), "Metric"),
            "humidity_pct": current.get("RelativeHumidity"),
            "precipitation_mm": _accuweather_value(precip_1h, "Metric"),
            "rain_mm": _accuweather_value(precip_1h, "Metric"),
            "cloud_cover_pct": current.get("CloudCover"),
            "wind_speed_kmh": _accuweather_value(wind_speed, "Metric"),
            "wind_direction_deg": wind_direction.get("Degrees") if isinstance(wind_direction, dict) else None,
            "source": "AccuWeather Core Weather",
        }
    except Exception as exc:
        return {"error": str(exc), "status_code": None}


@st.cache_data(ttl=600, show_spinner=False)
def fetch_openweather_onecall_current_weather(lat: float, lon: float, api_key: str, lang: str = "id"):
    try:
        response = requests.get(
            OPENWEATHER_ONECALL_CURRENT_ENDPOINT,
            params={
                "lat": float(lat),
                "lon": float(lon),
                "units": "metric",
                "lang": lang,
                "appid": api_key,
            },
            timeout=10,
        )
        if response.status_code != 200:
            try:
                payload = response.json()
            except Exception:
                payload = {"message": response.text}
            return {
                "error": payload.get("message", "OpenWeather One Call request failed."),
                "status_code": response.status_code,
            }

        payload = response.json()
        records = payload.get("data", [])
        current = records[0] if records else {}
        weather = current.get("weather", [])
        weather_item = weather[0] if weather else {}
        rain = current.get("rain", {}) or {}
        snow = current.get("snow", {}) or {}
        wind_speed_ms = current.get("wind_speed")
        wind_speed_kmh = None
        if wind_speed_ms is not None:
            wind_speed_kmh = float(wind_speed_ms) * 3.6

        return {
            "time": datetime.fromtimestamp(current.get("dt"), tz=timezone.utc).astimezone(LOCAL_TIMEZONE).isoformat(timespec="minutes")
            if current.get("dt") is not None
            else None,
            "weather_code": weather_item.get("id"),
            "weather": weather_item.get("description") or weather_item.get("main") or "-",
            "temperature_c": current.get("temp"),
            "feels_like_c": current.get("feels_like"),
            "humidity_pct": current.get("humidity"),
            "pressure_hpa": current.get("pressure"),
            "visibility_m": current.get("visibility"),
            "precipitation_mm": rain.get("1h", 0.0) or snow.get("1h", 0.0),
            "rain_mm": rain.get("1h", 0.0),
            "cloud_cover_pct": current.get("clouds"),
            "wind_speed_kmh": wind_speed_kmh,
            "wind_direction_deg": current.get("wind_deg"),
            "weather_icon_code": weather_item.get("icon"),
            "weather_icon_url": (
                f"https://openweathermap.org/img/wn/{weather_item.get('icon')}@4x.png"
                if weather_item.get("icon")
                else None
            ),
            "source": "OpenWeather One Call 4.0",
        }
    except Exception as exc:
        return {"error": str(exc), "status_code": None}


@st.cache_data(ttl=600, show_spinner=False)
def fetch_openweather_onecall_15min_forecast(lat: float, lon: float, api_key: str, lang: str = "id"):
    try:
        response = requests.get(
            OPENWEATHER_ONECALL_15MIN_ENDPOINT,
            params={
                "lat": float(lat),
                "lon": float(lon),
                "units": "metric",
                "lang": lang,
                "appid": api_key,
            },
            timeout=10,
        )
        if response.status_code != 200:
            try:
                payload = response.json()
            except Exception:
                payload = {"message": response.text}
            return {
                "error": payload.get("message", "OpenWeather One Call timeline request failed."),
                "status_code": response.status_code,
            }
        return response.json()
    except Exception as exc:
        return {"error": str(exc), "status_code": None}


def _forecast_record_time(record: dict):
    timestamp = record.get("dt")
    if timestamp is not None:
        try:
            return datetime.fromtimestamp(timestamp, tz=timezone.utc).astimezone(LOCAL_TIMEZONE)
        except (TypeError, ValueError, OSError):
            pass
    return parse_api_datetime(record.get("date") or record.get("dateTime") or record.get("time"))


def _forecast_precipitation_mm(record: dict):
    precipitation = record.get("precipitation")
    if precipitation is not None:
        try:
            return float(precipitation)
        except (TypeError, ValueError):
            pass
    total = 0.0
    found = False
    for key in ["rain", "snow"]:
        value = record.get(key)
        if isinstance(value, dict):
            for nested_key in ["15min", "1h", "3h"]:
                if nested_key in value:
                    try:
                        total += float(value[nested_key])
                        found = True
                    except (TypeError, ValueError):
                        pass
        elif value is not None:
            try:
                total += float(value)
                found = True
            except (TypeError, ValueError):
                pass
    return total if found else 0.0


def build_openweather_forecast_summary(payload: dict, hours: int = 12):
    records = payload.get("data", []) if isinstance(payload, dict) else []
    if not records:
        return {
            "available": False,
            "summary": "Forecast 15 menit belum tersedia.",
            "items": [],
            "thunder_count": 0,
            "rain_count": 0,
            "max_pop": None,
            "total_precip_mm": 0.0,
        }

    now_local = datetime.now(tz=LOCAL_TIMEZONE)
    cutoff = now_local + timedelta(hours=int(hours))
    selected = []
    for record in records:
        record_time = _forecast_record_time(record)
        if record_time is None:
            continue
        if record_time.tzinfo is None:
            record_time = record_time.replace(tzinfo=LOCAL_TIMEZONE)
        if now_local - timedelta(minutes=20) <= record_time <= cutoff:
            selected.append((record_time, record))

    if not selected:
        selected = [
            (_forecast_record_time(record) or now_local, record)
            for record in records[: min(len(records), int(hours * 4))]
        ]

    hourly_buckets = {}

    for record_time, record in selected:
        minutes_ahead = (record_time - now_local).total_seconds() / 60.0
        if minutes_ahead <= 0:
            continue
        hour_offset = int(math.ceil(minutes_ahead / 60.0))
        if hour_offset < 1 or hour_offset > int(hours):
            continue
        hour_key = now_local + timedelta(hours=hour_offset)
        hour_key = hour_key.replace(second=0, microsecond=0)
        weather = record.get("weather", [])
        weather_item = weather[0] if weather else {}
        code = weather_item.get("id")
        description = weather_item.get("description") or weather_item.get("main") or weather_code_label(code)
        pop = record.get("pop")
        try:
            pop_value = float(pop)
            if pop_value > 1.0:
                pop_value = pop_value / 100.0
        except (TypeError, ValueError):
            pop_value = None
        precipitation_mm = _forecast_precipitation_mm(record)
        is_thunder = False
        try:
            is_thunder = 200 <= int(code) <= 232
        except (TypeError, ValueError):
            pass

        bucket = hourly_buckets.setdefault(
            hour_key,
            {
                "time": hour_key.strftime("%H:%M"),
                "records": 0,
                "pop_values": [],
                "precip_mm": 0.0,
                "thunder_count": 0,
                "temperature_values": [],
                "description_counts": {},
                "weather_code": code,
                "icon_url": (
                    f"https://openweathermap.org/img/wn/{weather_item.get('icon')}@2x.png"
                    if weather_item.get("icon")
                    else None
                ),
            },
        )
        bucket["records"] += 1
        if pop_value is not None:
            bucket["pop_values"].append(pop_value)
        bucket["precip_mm"] += precipitation_mm
        if is_thunder:
            bucket["thunder_count"] += 1
        try:
            bucket["temperature_values"].append(float(record.get("temp")))
        except (TypeError, ValueError):
            pass
        desc_key = str(description or "-").title()
        bucket["description_counts"][desc_key] = bucket["description_counts"].get(desc_key, 0) + 1
        if pop_value is not None and pop_value == max(bucket["pop_values"], default=pop_value):
            bucket["weather_code"] = code
            bucket["icon_url"] = (
                f"https://openweathermap.org/img/wn/{weather_item.get('icon')}@2x.png"
                if weather_item.get("icon")
                else bucket.get("icon_url")
            )

    timeline = []
    thunder_slots = []
    rain_slots = []
    pops = []
    total_precip = 0.0
    for hour_key in sorted(hourly_buckets):
        bucket = hourly_buckets[hour_key]
        pop_value = max(bucket["pop_values"]) if bucket["pop_values"] else None
        if pop_value is not None:
            pops.append(pop_value)
        total_precip += bucket["precip_mm"]
        if bucket["thunder_count"]:
            thunder_slots.append(hour_key)
        if bucket["precip_mm"] > 0.0 or (pop_value is not None and pop_value >= 0.3):
            rain_slots.append(hour_key)
        description = max(bucket["description_counts"], key=bucket["description_counts"].get) if bucket["description_counts"] else "-"
        temp_avg = (
            sum(bucket["temperature_values"]) / len(bucket["temperature_values"])
            if bucket["temperature_values"]
            else None
        )
        timeline.append(
            {
                "time": bucket["time"],
                "description": description,
                "weather_code": bucket["weather_code"],
                "icon_url": bucket["icon_url"],
                "pop_pct": round(pop_value * 100.0) if pop_value is not None else None,
                "precip_mm": bucket["precip_mm"],
                "temperature_c": temp_avg,
                "is_thunder": bool(bucket["thunder_count"]),
            }
        )

    max_pop = max(pops) if pops else None
    if rain_slots:
        summary = f"Peluang hujan masih ada mulai sekitar {rain_slots[0].strftime('%H:%M')}."
    else:
        summary = f"Tidak ada sinyal hujan kuat dalam {hours} jam ke depan."

    return {
        "available": True,
        "summary": summary,
        "items": timeline[:12],
        "thunder_count": len(thunder_slots),
        "rain_count": len(rain_slots),
        "max_pop": max_pop,
        "total_precip_mm": total_precip,
    }


def normalize_event_time_for_api(event_time: datetime):
    if event_time is None:
        return None
    if event_time.tzinfo is None:
        return event_time.replace(tzinfo=LOCAL_TIMEZONE)
    return event_time


def format_openweather_time(value: datetime):
    value = normalize_event_time_for_api(value)
    if value is None:
        return None
    return value.isoformat(timespec="seconds")


def calculate_haversine_km(lat1, lon1, lat2, lon2):
    radius_earth_km = 6371.0088
    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    d_phi = math.radians(float(lat2) - float(lat1))
    d_lambda = math.radians(float(lon2) - float(lon1))
    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    )
    return 2.0 * radius_earth_km * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


def parse_api_datetime(value):
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


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


@st.cache_data(ttl=300, show_spinner=False)
def fetch_openweather_lightning_events(
    lat: float,
    lon: float,
    radius_km: float,
    start_date: str,
    end_date: str,
    api_key: str,
):
    try:
        response = requests.get(
            OPENWEATHER_LIGHTNING_ENDPOINT,
            params={
                "lat": float(lat),
                "lon": float(lon),
                "radius": float(radius_km),
                "start_date": start_date,
                "end_date": end_date,
                "apikey": api_key,
            },
            timeout=12,
        )
        if response.status_code != 200:
            try:
                payload = response.json()
            except Exception:
                payload = {"message": response.text}
            return {
                "error": payload.get("message", "OpenWeather Lightning request failed."),
                "status_code": response.status_code,
                "payload": payload,
            }
        return response.json()
    except Exception as exc:
        return {"error": str(exc), "status_code": None}


def build_openweather_lightning_dataframe(payload: dict, fault_lat: float, fault_lon: float, event_time: datetime | None):
    events = payload.get("lightnings", []) if isinstance(payload, dict) else []
    rows = []
    event_time_api = normalize_event_time_for_api(event_time)

    for event in events:
        strike_lat = event.get("lat")
        strike_lon = event.get("lon")
        if strike_lat is None or strike_lon is None:
            continue

        strike_time = parse_api_datetime(event.get("datetime"))
        delta_minutes = None
        if strike_time is not None and event_time_api is not None:
            delta_minutes = (strike_time - event_time_api.astimezone(timezone.utc)).total_seconds() / 60.0

        rows.append(
            {
                "Lightning Time UTC": event.get("datetime"),
                "Delta from Fault min": delta_minutes,
                "Distance from Fault km": calculate_haversine_km(fault_lat, fault_lon, strike_lat, strike_lon),
                "Latitude": strike_lat,
                "Longitude": strike_lon,
                "Quality": event.get("quality"),
                "Location Error km": event.get("error"),
                "ID": event.get("id"),
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(
        ["Distance from Fault km", "Delta from Fault min"],
        na_position="last",
    ).reset_index(drop=True)


@st.cache_data(ttl=120, show_spinner=False)
def fetch_xweather_lightning_flash_closest(
    lat: float,
    lon: float,
    radius_km: float,
    limit: int,
    client_id: str,
    client_secret: str,
):
    try:
        response = requests.get(
            XWEATHER_LIGHTNING_FLASH_CLOSEST_ENDPOINT,
            params={
                "p": f"{float(lat):.6f},{float(lon):.6f}",
                "format": "json",
                "radius": f"{min(float(radius_km), 40.0):.0f}km",
                "minradius": "0km",
                "limit": int(limit),
                "client_id": client_id,
                "client_secret": client_secret,
            },
            timeout=12,
        )
        try:
            payload = response.json()
        except Exception:
            payload = {"success": False, "error": {"description": response.text}}

        if response.status_code != 200 or not payload.get("success", False):
            error = payload.get("error") or {}
            return {
                "error": error.get("description") or error.get("message") or "Xweather Lightning request failed.",
                "status_code": response.status_code,
                "payload": payload,
            }
        return payload
    except Exception as exc:
        return {"error": str(exc), "status_code": None}


def build_xweather_lightning_dataframe(payload: dict, fault_lat: float, fault_lon: float, event_time: datetime | None):
    events = payload.get("response", []) if isinstance(payload, dict) else []
    rows = []
    event_time_api = normalize_event_time_for_api(event_time)

    for event in events:
        loc = event.get("loc", {}) or {}
        ob = event.get("ob", {}) or {}
        strike_lat = loc.get("lat")
        strike_lon = loc.get("long")
        if strike_lat is None or strike_lon is None:
            continue

        strike_time_iso = ob.get("dateTimeISO")
        strike_time = parse_api_datetime(strike_time_iso)
        delta_minutes = None
        if strike_time is not None and event_time_api is not None:
            delta_minutes = (strike_time - event_time_api.astimezone(timezone.utc)).total_seconds() / 60.0

        rows.append(
            {
                "Lightning Time UTC": strike_time_iso,
                "Age seconds": ob.get("age"),
                "Delta from Fault min": delta_minutes,
                "Distance from Fault km": calculate_haversine_km(fault_lat, fault_lon, strike_lat, strike_lon),
                "Latitude": strike_lat,
                "Longitude": strike_lon,
                "Received Time UTC": event.get("recDateTimeISO"),
                "ID": event.get("id"),
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(
        ["Distance from Fault km", "Age seconds"],
        na_position="last",
    ).reset_index(drop=True)


@st.cache_data(ttl=120, show_spinner=False)
def fetch_accuweather_lightning_radius(
    lat: float,
    lon: float,
    radius_km: float,
    time_interval_minutes: int,
    api_key: str,
    strike_type: str = "all",
):
    try:
        interval = int(time_interval_minutes)
        if interval not in [5, 15, 30, 60, 120]:
            interval = 15
        radius_miles = min(float(radius_km) / 1.609344, 60.0)
        params = {
            "apikey": api_key,
            "q": f"{float(lat):.6f},{float(lon):.6f}",
            "distanceRadius": f"{radius_miles:.2f}",
            "offset": "-1",
        }
        if strike_type in ["cg", "ic"]:
            params["strikeType"] = strike_type

        response = requests.get(
            f"{ACCUWEATHER_LIGHTNING_BASE_URL}/lightning/v1/{interval}min/geoposition/radius.geojson",
            params=params,
            timeout=12,
        )
        try:
            payload = response.json()
        except Exception:
            payload = {"message": response.text}

        if response.status_code != 200:
            return {
                "error": payload.get("message", "AccuWeather Lightning request failed."),
                "status_code": response.status_code,
                "payload": payload,
            }
        return payload
    except Exception as exc:
        return {"error": str(exc), "status_code": None}


def build_accuweather_lightning_dataframe(payload: dict, fault_lat: float, fault_lon: float, event_time: datetime | None):
    features = payload.get("features", []) if isinstance(payload, dict) else []
    rows = []
    event_time_api = normalize_event_time_for_api(event_time)

    for feature in features:
        geometry = feature.get("geometry", {}) or {}
        properties = feature.get("properties", {}) or {}
        coordinates = geometry.get("coordinates", [])
        if not isinstance(coordinates, list) or len(coordinates) < 2:
            continue
        strike_lon = coordinates[0]
        strike_lat = coordinates[1]

        strike_time_iso = properties.get("date")
        strike_time = parse_api_datetime(strike_time_iso)
        delta_minutes = None
        if strike_time is not None and event_time_api is not None:
            delta_minutes = (strike_time - event_time_api.astimezone(timezone.utc)).total_seconds() / 60.0

        rows.append(
            {
                "Lightning Time UTC": strike_time_iso,
                "Delta from Fault min": delta_minutes,
                "Distance from Fault km": calculate_haversine_km(fault_lat, fault_lon, strike_lat, strike_lon),
                "Latitude": strike_lat,
                "Longitude": strike_lon,
                "Strike Type": properties.get("strikeType") or "-",
                "Peak Current A": properties.get("peakCurrent"),
                "Source ID": properties.get("sourceId"),
                "ID": properties.get("id"),
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(
        ["Distance from Fault km", "Delta from Fault min"],
        na_position="last",
    ).reset_index(drop=True)


def weather_icon_for_code(code):
    try:
        code = int(code)
    except (TypeError, ValueError):
        return "?"
    if 200 <= code <= 232:
        return "!!"
    if 300 <= code <= 321:
        return ".."
    if 500 <= code <= 531:
        return "//"
    if 600 <= code <= 622:
        return "SN"
    if 700 <= code <= 781:
        return "~~"
    if code == 800:
        return "SUN"
    if 801 <= code <= 804:
        return "CL"
    if code in THUNDERSTORM_WEATHER_CODES:
        return "!!"
    if code in {61, 63, 65, 66, 67, 80, 81, 82}:
        return "//"
    if code in {51, 53, 55, 56, 57}:
        return ".."
    if code in {45, 48}:
        return "~~"
    if code in {2, 3}:
        return "CL"
    if code in {0, 1}:
        return "SUN"
    return "WX"


def weather_theme_for_code(code):
    try:
        code = int(code)
    except (TypeError, ValueError):
        return "neutral"
    if code in THUNDERSTORM_WEATHER_CODES or 200 <= code <= 232:
        return "storm"
    if code in {61, 63, 65, 66, 67, 80, 81, 82} or 500 <= code <= 531:
        return "rain"
    if code in {45, 48} or 700 <= code <= 781:
        return "mist"
    if code in {2, 3} or 801 <= code <= 804:
        return "cloud"
    if code in {0, 1, 800}:
        return "clear"
    return "neutral"


def weather_symbol_for_code(code):
    try:
        code = int(code)
    except (TypeError, ValueError):
        return "☁"
    if 200 <= code <= 232 or code in THUNDERSTORM_WEATHER_CODES:
        return "⚡"
    if 300 <= code <= 321:
        return "☂"
    if 500 <= code <= 531:
        return "☔"
    if 600 <= code <= 622:
        return "❄"
    if 700 <= code <= 781:
        return "≋"
    if code == 800 or code in {0, 1}:
        return "☀"
    if code in {801, 2}:
        return "⛅"
    return "☁"


def build_weather_trend_svg(items):
    usable = []
    for item in items:
        try:
            temp = float(item.get("temperature_c"))
        except (TypeError, ValueError):
            continue
        try:
            rain = float(item.get("precip_mm") or 0.0)
        except (TypeError, ValueError):
            rain = 0.0
        usable.append((temp, rain, str(item.get("time", "-"))))

    if len(usable) < 2:
        return ""

    width = 520
    height = 110
    x_start = 18
    x_step = (width - 36) / max(len(usable) - 1, 1)
    temps = [item[0] for item in usable]
    min_temp = min(temps)
    max_temp = max(temps)
    temp_span = max(max_temp - min_temp, 1.0)
    points = []
    labels = []
    rain_bars = []

    for idx, (temp, rain, time_label) in enumerate(usable):
        x = x_start + idx * x_step
        y = 68 - ((temp - min_temp) / temp_span) * 36
        points.append(f"{x:.1f},{y:.1f}")
        labels.append((x, y, temp, time_label))
        bar_height = min(32.0, max(3.0, rain * 9.0)) if rain > 0 else 3.0
        rain_bars.append(
            f"<rect x='{x - 5:.1f}' y='{82 - bar_height:.1f}' width='10' height='{bar_height:.1f}' rx='4' />"
        )

    grid_lines = "".join(
        f"<line x1='{x_start + idx * x_step:.1f}' y1='18' x2='{x_start + idx * x_step:.1f}' y2='92' />"
        for idx in range(len(usable))
    )
    circles = "".join(
        f"<circle cx='{point.split(',')[0]}' cy='{point.split(',')[1]}' r='3' />"
        for point in points
    )
    text_labels = "".join(
        f"<text x='{x:.1f}' y='{max(12.0, y - 8):.1f}' text-anchor='middle'>{temp:.0f}°</text>"
        for x, y, temp, _ in labels
    )
    time_labels = "".join(
        f"<text x='{x:.1f}' y='104' text-anchor='middle'>{time_label}</text>"
        for x, _, _, time_label in labels
    )
    return (
        "<svg class='weather-trend-svg' viewBox='0 0 520 110' role='img' aria-label='Grafik tren suhu dan hujan'>"
        f"<g class='weather-trend-grid'>{grid_lines}</g>"
        f"<g class='weather-trend-rain'>{''.join(rain_bars)}</g>"
        f"<polyline class='weather-trend-line' points='{' '.join(points)}' />"
        f"<g class='weather-trend-points'>{circles}</g>"
        f"<g class='weather-trend-labels'>{text_labels}</g>"
        f"<g class='weather-trend-axis'>{time_labels}</g>"
        "</svg>"
    )


def build_rain_chance_bars(items):
    bars = ""
    max_pop = 1
    for item in items:
        try:
            max_pop = max(max_pop, int(item.get("pop_pct") or 0))
        except (TypeError, ValueError):
            pass
    for item in items:
        try:
            pop_num = int(item.get("pop_pct") or 0)
        except (TypeError, ValueError):
            pop_num = 0
        bar_height = 12 + (pop_num / max_pop) * 58
        bars += (
            "<div class='rain-bar-wrap'>"
            f"<i style='height:{bar_height:.0f}px'></i>"
            f"<span>{pop_num}%</span>"
            f"<small>{item.get('time', '-')}</small>"
            "</div>"
        )
    return bars


def weather_card_html(weather_rows):
    cards_html = []
    for row in weather_rows:
        thunder_text = row.get("Last Thunderstorm Indication", "-")
        storm_class = "weather-storm-muted"
        if row.get("Last Thunderstorm Time"):
            thunder_text = f"{row.get('Last Thunderstorm Time')} | {row.get('Last Thunderstorm Weather', '-')}"
            storm_class = "weather-storm-active"
        theme = weather_theme_for_code(row.get("Weather Code"))
        icon_url = row.get("Weather Icon URL")
        if icon_url:
            visual_html = f"<img class='weather-hero-img' src='{icon_url}' alt='{row.get('Current Weather', 'Weather')}' />"
        else:
            visual_html = f"<div class='weather-hero-fallback'>{weather_icon_for_code(row.get('Weather Code'))}</div>"
        visual_caption = str(row.get("Current Weather") or "-").title()
        forecast = row.get("Forecast Summary") or {}
        forecast_items = forecast.get("items", [])
        forecast_items_html = ""
        temp_values = []
        precip_values = []
        for item in forecast_items:
            temp_value = item.get("temperature_c")
            precip_value = item.get("precip_mm") or 0.0
            try:
                temp_values.append(float(temp_value))
            except (TypeError, ValueError):
                pass
            try:
                precip_values.append(float(precip_value))
            except (TypeError, ValueError):
                precip_value = 0.0
            item_icon = (
                f"<img src='{item.get('icon_url')}' alt='{item.get('description', 'Forecast')}' />"
                if item.get("icon_url")
                else f"<strong>{weather_icon_for_code(item.get('weather_code'))}</strong>"
            )
            pop_label = f"{item.get('pop_pct')}%" if item.get("pop_pct") is not None else "-"
            temp_label = safe_display_number(temp_value, 0, " C")
            precip_label = safe_display_number(precip_value, 1, " mm") if precip_value else ""
            risk_value = item.get("pop_pct") if item.get("pop_pct") is not None else 0
            precip_height = min(28, max(3, float(precip_value) * 8.0)) if precip_value else 3
            forecast_items_html += (
                "<div class='weather-forecast-item'>"
                f"<span>{item.get('time', '-')}</span>"
                f"{item_icon}"
                f"<b>{temp_label}</b>"
                f"<em>{item.get('description', '-')}</em>"
                f"<small>{pop_label}</small>"
                "<div class='weather-precip-track'>"
                f"<i style='height:{precip_height}px'></i>"
                "</div>"
                f"<label>{precip_label}</label>"
                f"<mark>{risk_value}</mark>"
                "</div>"
            )
        temp_min = min(temp_values) if temp_values else None
        temp_max = max(temp_values) if temp_values else None
        temp_range = (
            f"{safe_display_number(temp_min, 0, ' C')} - {safe_display_number(temp_max, 0, ' C')}"
            if temp_min is not None and temp_max is not None
            else "-"
        )
        forecast_summary = forecast.get("summary", "Forecast belum tersedia.")
        forecast_pop = (
            f"{forecast.get('max_pop') * 100.0:.0f}%"
            if forecast.get("max_pop") is not None
            else "-"
        )
        forecast_precip = safe_display_number(forecast.get("total_precip_mm", 0.0), 2, " mm")
        trend_svg = build_weather_trend_svg(forecast_items)
        visibility_km = None
        try:
            visibility_km = float(row.get("Visibility m")) / 1000.0
        except (TypeError, ValueError):
            visibility_km = None
        cards_html.append(
            textwrap.dedent(f"""
            <div class="weather-card weather-theme-{theme}">
                <div class="weather-effect weather-effect-{theme}"></div>
                <div class="weather-glass">
                    <div class="weather-top">
                        <div class="weather-now">
                            <div class="weather-visual-badge">{visual_html}</div>
                            <div>
                                <div class="weather-desc">{row.get('Current Weather', '-')}</div>
                                <div class="weather-location">{row.get('Location', '-')}</div>
                                <div class="weather-tower">{row.get('Tower', '-')}</div>
                            </div>
                        </div>
                        <div class="weather-temp-block">
                            <div class="weather-temp">{safe_display_number(row.get('Temperature C'), 1, ' C')}</div>
                            <div class="weather-range">{temp_range}</div>
                        </div>
                    </div>

                    <div class="weather-attribute-list">
                        <div><span>Jarak pandang</span><strong>{safe_display_number(visibility_km, 1, ' km')}</strong></div>
                        <div><span>Terasa seperti</span><strong>{safe_display_number(row.get('Feels Like C'), 1, ' C')}</strong></div>
                        <div><span>Kelembapan</span><strong>{safe_display_number(row.get('Humidity %'), 0, '%')}</strong></div>
                        <div><span>Angin</span><strong>{safe_display_number(row.get('Wind km/h'), 1, ' km/h')}</strong></div>
                        <div><span>Hujan saat ini</span><strong>{safe_display_number(row.get('Rain mm'), 2, ' mm')}</strong></div>
                        <div><span>Tutupan awan</span><strong>{safe_display_number(row.get('Cloud Cover %'), 0, '%')}</strong></div>
                    </div>

                    <div class="weather-forecast-row">
                        <div class="weather-forecast-copy">
                            <span>Prakiraan 12 jam</span>
                            <strong>{forecast_summary}</strong>
                            <small>{forecast_signal} | Peluang hujan tertinggi {forecast_pop} | Hujan total {forecast_precip}</small>
                        </div>
                        {trend_svg}
                    </div>

                    <div class="weather-forecast-strip">
                        {forecast_items_html}
                    </div>

                    <div class="weather-storm {storm_class}">
                        <span>Indikasi thunderstorm</span>
                        <strong>{thunder_text}</strong>
                    </div>

                    <div class="weather-footer">
                        <span>Kumulatif {safe_display_number(row.get('Cumulative km'), 3, ' km')}</span>
                        <span>{safe_display_number(row.get('Latitude'), 6)}, {safe_display_number(row.get('Longitude'), 6)}</span>
                        <span>Update {row.get('Weather Time') or '-'} | {row.get('Weather Source', '-')}</span>
                    </div>
                </div>
            </div>
            """)
        )
    return (
        textwrap.dedent("""
        <style>
        .weather-card-wrap {
            display: grid;
            grid-template-columns: minmax(0, 1fr);
            gap: 14px;
            margin: 12px 0 14px 0;
        }
        .weather-card {
            border: 1px solid rgba(148, 163, 184, 0.45);
            border-radius: 8px;
            padding: 14px 16px;
            background: linear-gradient(135deg, #f8fafc 0%, #e0f2fe 100%);
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.10);
            color: #0f172a;
            break-inside: avoid;
            page-break-inside: avoid;
            overflow: hidden;
            position: relative;
        }
        .weather-layout {
            display: grid;
            grid-template-columns: minmax(360px, 0.9fr) minmax(440px, 1.1fr);
            gap: 12px;
            align-items: stretch;
        }
        .weather-current-panel,
        .weather-forecast-panel {
            border-radius: 8px;
            border: 1px solid rgba(203, 213, 225, 0.72);
            background: rgba(255, 255, 255, 0.52);
            padding: 12px;
            height: 100%;
        }
        .weather-current-panel {
            display: flex;
            flex-direction: column;
        }
        .weather-forecast-panel {
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
        }
        .weather-theme-clear {
            background: linear-gradient(135deg, #fff7ed 0%, #dbeafe 100%);
        }
        .weather-theme-cloud {
            background: linear-gradient(135deg, #f8fafc 0%, #dbeafe 56%, #e2e8f0 100%);
        }
        .weather-theme-rain {
            background: linear-gradient(135deg, #eef6ff 0%, #c7d2fe 100%);
        }
        .weather-theme-storm {
            background: linear-gradient(135deg, #eef2ff 0%, #fed7aa 100%);
        }
        .weather-theme-mist {
            background: linear-gradient(135deg, #f8fafc 0%, #e5e7eb 100%);
        }
        .weather-hero,
        .weather-meta {
            display: flex;
            justify-content: space-between;
            gap: 18px;
            align-items: flex-start;
        }
        .weather-hero {
            min-height: 112px;
            align-items: center;
            padding: 0;
            border-bottom: 0;
        }
        .weather-copy {
            min-width: 0;
            flex: 1;
        }
        .weather-role {
            color: #ef4444;
            font-size: 13px;
            font-weight: 700;
            margin-bottom: 3px;
        }
        .weather-tower {
            font-size: 14px;
            font-weight: 700;
            line-height: 1.25;
        }
        .weather-temp {
            font-size: 42px;
            line-height: 1;
            font-weight: 800;
            margin-top: 18px;
        }
        .weather-desc {
            color: #334155;
            font-size: 16px;
            margin-top: 4px;
            text-transform: capitalize;
        }
        .weather-visual-card {
            width: 154px;
            border-radius: 8px;
            padding: 8px;
            background: rgba(255, 255, 255, 0.58);
            border: 1px solid rgba(255, 255, 255, 0.78);
            box-shadow: 0 18px 45px rgba(15,23,42,0.13);
            flex: 0 0 auto;
        }
        .weather-visual {
            width: 100%;
            height: 110px;
            border-radius: 8px;
            display: grid;
            place-items: center;
            background:
                radial-gradient(circle at 50% 38%, rgba(255,255,255,0.94), rgba(255,255,255,0.42) 54%, rgba(226,232,240,0.42));
        }
        .weather-hero-img {
            width: 104px;
            height: 104px;
            object-fit: contain;
            filter: drop-shadow(0 10px 12px rgba(15, 23, 42, 0.18));
        }
        .weather-visual-caption {
            margin-top: 5px;
            text-align: center;
            font-size: 12px;
            font-weight: 700;
            color: #0f172a;
        }
        .weather-hero-fallback {
            width: 96px;
            height: 96px;
            border-radius: 50%;
            display: grid;
            place-items: center;
            background: #0f172a;
            color: #ffffff;
            font-size: 24px;
            font-weight: 800;
            letter-spacing: 0;
        }
        .weather-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 8px;
            margin-top: 10px;
        }
        .weather-grid div {
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid rgba(203, 213, 225, 0.75);
            border-radius: 8px;
            padding: 8px 9px;
        }
        .weather-grid span,
        .weather-storm span {
            display: block;
            color: #64748b;
            font-size: 11px;
            margin-bottom: 1px;
        }
        .weather-grid strong,
        .weather-storm strong {
            font-size: 13px;
        }
        .weather-storm {
            margin-top: 8px;
            border-radius: 8px;
            padding: 8px 10px;
            border: 1px solid rgba(203, 213, 225, 0.8);
            background: rgba(255, 255, 255, 0.72);
        }
        .weather-forecast-head {
            margin-bottom: 8px;
        }
        .weather-forecast-head span,
        .weather-forecast-metrics span {
            display: block;
            color: #64748b;
            font-size: 11px;
            margin-bottom: 2px;
        }
        .weather-forecast-head strong {
            font-size: 14px;
        }
        .weather-forecast-metrics {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 7px;
            margin-bottom: 8px;
        }
        .weather-forecast-metrics div {
            border-radius: 8px;
            border: 1px solid rgba(203, 213, 225, 0.75);
            background: rgba(248, 250, 252, 0.85);
            padding: 7px 8px;
        }
        .weather-alert-pill {
            margin: 0 0 8px 0;
            padding: 7px 9px;
            border-radius: 8px;
            background: rgba(15, 23, 42, 0.06);
            border: 1px solid rgba(148, 163, 184, 0.32);
            color: #0f172a;
            font-size: 12px;
            font-weight: 700;
        }
        .weather-forecast-strip {
            display: flex;
            gap: 8px;
            margin-top: 0;
            overflow-x: auto;
            padding-bottom: 2px;
            flex: 1;
        }
        .weather-forecast-item {
            flex: 0 0 86px;
            min-width: 86px;
            border-radius: 8px;
            border: 1px solid rgba(203, 213, 225, 0.72);
            background: linear-gradient(180deg, rgba(255,255,255,0.86), rgba(241,245,249,0.78));
            padding: 7px 6px 6px 6px;
            text-align: center;
            display: grid;
            grid-template-rows: auto 30px auto auto auto 30px;
            align-content: center;
            gap: 1px;
        }
        .weather-forecast-item span,
        .weather-forecast-item small {
            display: block;
            color: #475569;
            font-size: 10px;
        }
        .weather-forecast-item em {
            display: block;
            color: #0f172a;
            font-size: 10px;
            font-style: normal;
            font-weight: 700;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .weather-forecast-item b {
            display: block;
            color: #0f172a;
            font-size: 13px;
            line-height: 1;
        }
        .weather-forecast-item img {
            width: 30px;
            height: 30px;
            object-fit: contain;
            margin: -3px auto -3px auto;
        }
        .weather-forecast-item strong {
            display: grid;
            place-items: center;
            width: 28px;
            height: 28px;
            margin: 0 auto;
            border-radius: 50%;
            background: #0f172a;
            color: #ffffff;
            font-size: 9px;
        }
        .weather-precip-track {
            width: 16px;
            height: 30px;
            margin: 1px auto 0 auto;
            display: flex;
            align-items: flex-end;
            justify-content: center;
            border-radius: 999px;
            background: rgba(203, 213, 225, 0.45);
            overflow: hidden;
        }
        .weather-precip-track i {
            display: block;
            width: 100%;
            min-height: 3px;
            border-radius: 999px;
            background: linear-gradient(180deg, #60a5fa, #2563eb);
        }
        .weather-storm-active {
            border-color: rgba(234, 88, 12, 0.55);
            background: rgba(255, 237, 213, 0.9);
        }
        .weather-storm-muted strong {
            color: #475569;
        }
        .weather-meta,
        .weather-time {
            color: #64748b;
            font-size: 11px;
            margin-top: 8px;
        }
        @media print {
            .weather-card-wrap {
                grid-template-columns: 1fr;
            }
            .weather-card {
                box-shadow: none;
            }
        }
        @media (max-width: 980px) {
            .weather-layout {
                grid-template-columns: 1fr;
                align-items: start;
            }
            .weather-current-panel,
            .weather-forecast-panel {
                height: auto;
            }
        }
        @media (max-width: 640px) {
            .weather-card {
                padding: 10px;
            }
            .weather-current-panel,
            .weather-forecast-panel {
                padding: 10px;
                min-width: 0;
            }
            .weather-hero {
                align-items: flex-start;
                min-height: 0;
                gap: 10px;
            }
            .weather-visual-card {
                width: 92px;
                padding: 8px;
            }
            .weather-visual {
                height: 76px;
            }
            .weather-hero-img {
                width: 72px;
                height: 72px;
            }
            .weather-visual-caption {
                font-size: 10px;
            }
            .weather-temp {
                font-size: 34px;
            }
            .weather-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .weather-forecast-head {
                grid-template-columns: 1fr;
            }
            .weather-forecast-metrics {
                grid-template-columns: repeat(3, minmax(96px, 1fr));
                overflow-x: auto;
                padding-bottom: 2px;
            }
            .weather-forecast-metrics div {
                min-width: 96px;
            }
            .weather-forecast-strip {
                display: flex;
                gap: 7px;
                overflow-x: auto;
                max-width: 100%;
                flex: 0 0 auto;
            }
            .weather-forecast-item {
                flex: 0 0 78px;
                min-width: 78px;
            }
            .weather-meta {
                display: grid;
                grid-template-columns: 1fr;
                gap: 2px;
            }
            .weather-time {
                overflow-wrap: anywhere;
            }
        }
        .weather-card {
            border-radius: 16px;
            padding: 0;
            color: #f8fafc;
            background:
                radial-gradient(circle at 66% -14%, rgba(255, 239, 163, 0.95), rgba(255, 239, 163, 0.05) 15%, transparent 25%),
                radial-gradient(circle at 0% 100%, rgba(239, 68, 68, 0.72), transparent 32%),
                linear-gradient(135deg, #0f7caf 0%, #0f5d82 46%, #153d5a 100%);
            border: 1px solid rgba(255, 255, 255, 0.22);
            box-shadow: 0 18px 46px rgba(15, 23, 42, 0.22);
            overflow: hidden;
            position: relative;
        }
        .weather-theme-cloud,
        .weather-theme-mist {
            background:
                radial-gradient(circle at 66% -14%, rgba(255, 239, 163, 0.78), rgba(255, 239, 163, 0.04) 14%, transparent 25%),
                radial-gradient(circle at 0% 100%, rgba(239, 68, 68, 0.62), transparent 32%),
                linear-gradient(135deg, #1675a7 0%, #195779 48%, #213d56 100%);
        }
        .weather-theme-rain,
        .weather-theme-storm {
            background:
                radial-gradient(circle at 64% -16%, rgba(250, 204, 21, 0.52), rgba(250, 204, 21, 0.03) 14%, transparent 25%),
                radial-gradient(circle at 0% 100%, rgba(220, 38, 38, 0.66), transparent 32%),
                linear-gradient(135deg, #0f5276 0%, #173b5a 48%, #202b44 100%);
        }
        .weather-effect {
            position: absolute;
            inset: 0;
            pointer-events: none;
            opacity: 0.28;
            background-image:
                linear-gradient(90deg, rgba(255,255,255,0.08) 1px, transparent 1px),
                linear-gradient(rgba(255,255,255,0.06) 1px, transparent 1px);
            background-size: 58px 58px;
            mask-image: linear-gradient(180deg, transparent, #000 18%, #000 74%, transparent);
        }
        .weather-glass {
            position: relative;
            z-index: 1;
            padding: 22px 28px 20px 28px;
            border-radius: 16px;
            background: rgba(15, 23, 42, 0.12);
            backdrop-filter: blur(6px);
        }
        .weather-top {
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto;
            gap: 18px;
            align-items: start;
        }
        .weather-now {
            display: flex;
            align-items: center;
            gap: 18px;
            min-width: 0;
        }
        .weather-visual-badge {
            width: 72px;
            height: 72px;
            border-radius: 50%;
            display: grid;
            place-items: center;
            flex: 0 0 auto;
            background: rgba(255, 255, 255, 0.10);
        }
        .weather-visual-badge .weather-hero-img {
            width: 82px;
            height: 82px;
            object-fit: contain;
            filter: drop-shadow(0 12px 16px rgba(0,0,0,0.25));
        }
        .weather-visual-badge .weather-hero-fallback {
            width: 58px;
            height: 58px;
            font-size: 13px;
            background: rgba(255, 255, 255, 0.16);
            color: #ffffff;
        }
        .weather-desc {
            color: #ffffff;
            font-size: 30px;
            line-height: 1.05;
            margin: 0;
            font-weight: 800;
            text-transform: capitalize;
        }
        .weather-location {
            color: rgba(255, 255, 255, 0.68);
            font-size: 14px;
            font-weight: 700;
            margin-top: 2px;
        }
        .weather-tower {
            color: rgba(255, 255, 255, 0.72);
            font-size: 12px;
            margin-top: 5px;
        }
        .weather-temp-block {
            text-align: right;
        }
        .weather-temp {
            color: #ffffff;
            font-size: 34px;
            line-height: 1;
            font-weight: 800;
            margin: 0;
        }
        .weather-range {
            color: rgba(255, 255, 255, 0.58);
            font-size: 13px;
            margin-top: 5px;
            font-weight: 700;
        }
        .weather-attribute-list {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 8px 24px;
            margin-top: 20px;
        }
        .weather-attribute-list div {
            display: flex;
            justify-content: space-between;
            gap: 12px;
            color: rgba(255, 255, 255, 0.78);
            font-size: 13px;
            font-weight: 650;
        }
        .weather-attribute-list strong {
            color: #ffffff;
            white-space: nowrap;
        }
        .weather-forecast-row {
            display: grid;
            grid-template-columns: minmax(260px, 0.9fr) minmax(360px, 1.1fr);
            gap: 16px;
            align-items: center;
            margin-top: 18px;
        }
        .weather-forecast-copy span,
        .weather-forecast-copy small {
            display: block;
            color: rgba(255, 255, 255, 0.66);
            font-size: 12px;
        }
        .weather-forecast-copy strong {
            display: block;
            color: #ffffff;
            font-size: 15px;
            margin: 5px 0 6px 0;
        }
        .weather-trend-svg {
            width: 100%;
            height: 110px;
            display: block;
            overflow: visible;
        }
        .weather-trend-grid line {
            stroke: rgba(255, 255, 255, 0.18);
            stroke-dasharray: 4 5;
        }
        .weather-trend-rain rect {
            fill: rgba(96, 165, 250, 0.75);
        }
        .weather-trend-line {
            fill: none;
            stroke: #f59e0b;
            stroke-width: 4;
            stroke-linecap: round;
            stroke-linejoin: round;
        }
        .weather-trend-points circle {
            fill: #111827;
            stroke: #f59e0b;
            stroke-width: 2;
        }
        .weather-forecast-strip {
            display: flex;
            gap: 14px;
            overflow-x: auto;
            padding: 12px 2px 8px 2px;
            margin-top: 2px;
        }
        .weather-forecast-item {
            flex: 0 0 74px;
            min-width: 74px;
            border: 0;
            border-radius: 0;
            background: transparent;
            color: rgba(255, 255, 255, 0.74);
            padding: 0;
            display: grid;
            grid-template-rows: 18px 34px 18px 16px 16px 28px;
            justify-items: center;
            gap: 1px;
            text-align: center;
        }
        .weather-forecast-item span,
        .weather-forecast-item small {
            color: rgba(255, 255, 255, 0.72);
            font-size: 11px;
        }
        .weather-forecast-item img {
            width: 34px;
            height: 34px;
            object-fit: contain;
            filter: drop-shadow(0 8px 10px rgba(0,0,0,0.25));
            margin: 0;
        }
        .weather-forecast-item b {
            color: #ffffff;
            font-size: 13px;
        }
        .weather-forecast-item em {
            max-width: 70px;
            color: rgba(255, 255, 255, 0.62);
            font-size: 10px;
        }
        .weather-forecast-item strong {
            background: rgba(255, 255, 255, 0.15);
            color: #fff;
        }
        .weather-precip-track {
            width: 22px;
            height: 28px;
            background: rgba(255, 255, 255, 0.14);
        }
        .weather-precip-track i {
            background: linear-gradient(180deg, #7dd3fc, #2563eb);
        }
        .weather-storm {
            margin-top: 12px;
            border-radius: 999px;
            padding: 8px 12px;
            border: 1px solid rgba(255, 255, 255, 0.16);
            background: rgba(15, 23, 42, 0.16);
        }
        .weather-storm span {
            color: rgba(255, 255, 255, 0.58);
            font-size: 11px;
        }
        .weather-storm strong {
            color: #ffffff;
            font-size: 12px;
        }
        .weather-storm-active {
            border-color: rgba(251, 191, 36, 0.72);
            background: rgba(251, 146, 60, 0.18);
        }
        .weather-footer {
            display: flex;
            justify-content: space-between;
            gap: 12px;
            margin-top: 12px;
            color: rgba(255, 255, 255, 0.54);
            font-size: 11px;
            flex-wrap: wrap;
        }
        @media (max-width: 900px) {
            .weather-glass {
                padding: 18px;
            }
            .weather-top,
            .weather-forecast-row {
                grid-template-columns: 1fr;
            }
            .weather-temp-block {
                text-align: left;
            }
            .weather-attribute-list {
                grid-template-columns: 1fr;
                gap: 7px;
            }
        }
        @media (max-width: 640px) {
            .weather-card {
                border-radius: 10px;
            }
            .weather-glass {
                padding: 14px;
                border-radius: 10px;
            }
            .weather-now {
                gap: 12px;
            }
            .weather-visual-badge {
                width: 54px;
                height: 54px;
            }
            .weather-visual-badge .weather-hero-img {
                width: 64px;
                height: 64px;
            }
            .weather-desc {
                font-size: 22px;
            }
            .weather-temp {
                font-size: 30px;
            }
            .weather-forecast-strip {
                gap: 10px;
            }
            .weather-forecast-item {
                flex-basis: 66px;
                min-width: 66px;
            }
        }
        /* Home Assistant-style compact forecast card override */
        .weather-card-wrap {
            place-items: start center;
            margin: 10px 0 12px 0;
        }
        .weather-card {
            width: min(100%, 760px);
            border-radius: 14px;
            background:
                radial-gradient(circle at 70% -20%, rgba(255, 236, 148, 0.86), rgba(255, 236, 148, 0.05) 16%, transparent 28%),
                radial-gradient(circle at -5% 115%, rgba(239, 68, 68, 0.62), transparent 30%),
                linear-gradient(135deg, #1277a6 0%, #145875 50%, #163f58 100%);
            box-shadow: 0 14px 34px rgba(15, 23, 42, 0.20);
            border: 1px solid rgba(255, 255, 255, 0.18);
        }
        .weather-glass {
            padding: 18px 20px 16px 20px;
            background: rgba(15, 23, 42, 0.08);
            border-radius: 14px;
        }
        .weather-top {
            grid-template-columns: minmax(0, 1fr) auto;
            gap: 14px;
        }
        .weather-now {
            gap: 14px;
        }
        .weather-visual-badge {
            width: 58px;
            height: 58px;
            background: rgba(255, 255, 255, 0.11);
        }
        .weather-visual-badge .weather-hero-img {
            width: 68px;
            height: 68px;
        }
        .weather-desc {
            font-size: 23px;
            letter-spacing: 0;
        }
        .weather-location {
            font-size: 12px;
            margin-top: 3px;
        }
        .weather-tower {
            font-size: 11px;
            margin-top: 2px;
        }
        .weather-temp {
            font-size: 29px;
        }
        .weather-range {
            font-size: 11px;
        }
        .weather-attribute-list {
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 7px 22px;
            margin-top: 16px;
        }
        .weather-attribute-list div {
            font-size: 12px;
        }
        .weather-forecast-row {
            grid-template-columns: minmax(0, 0.7fr) minmax(280px, 1fr);
            gap: 14px;
            margin-top: 16px;
        }
        .weather-forecast-copy span,
        .weather-forecast-copy small {
            font-size: 11px;
        }
        .weather-forecast-copy strong {
            font-size: 13px;
            margin: 3px 0 5px 0;
        }
        .weather-trend-svg {
            height: 76px;
        }
        .weather-forecast-strip {
            gap: 12px;
            padding: 8px 0 6px 0;
            margin-top: 4px;
            scrollbar-width: thin;
        }
        .weather-forecast-item {
            flex-basis: 56px;
            min-width: 56px;
            grid-template-rows: 16px 28px 16px 14px 14px 22px;
        }
        .weather-forecast-item span,
        .weather-forecast-item small,
        .weather-forecast-item em {
            font-size: 9px;
        }
        .weather-forecast-item img {
            width: 28px;
            height: 28px;
        }
        .weather-forecast-item b {
            font-size: 11px;
        }
        .weather-precip-track {
            width: 18px;
            height: 22px;
        }
        .weather-storm {
            margin-top: 8px;
            padding: 7px 10px;
        }
        .weather-footer {
            margin-top: 8px;
            font-size: 10px;
        }
        @media (max-width: 820px) {
            .weather-card {
                width: 100%;
            }
            .weather-forecast-row {
                grid-template-columns: 1fr;
            }
            .weather-trend-svg {
                height: 70px;
            }
        }
        @media (max-width: 520px) {
            .weather-glass {
                padding: 14px;
            }
            .weather-top {
                grid-template-columns: 1fr;
            }
            .weather-temp-block {
                text-align: left;
            }
            .weather-attribute-list {
                grid-template-columns: 1fr;
            }
            .weather-desc {
                font-size: 20px;
            }
            .weather-forecast-item {
                flex-basis: 54px;
                min-width: 54px;
            }
        }
        /* Close visual match to troinine/ha-weather-forecast-card demo */
        .weather-card-wrap {
            place-items: start center;
            margin: 10px 0;
        }
        .weather-card {
            width: min(100%, 552px);
            min-height: 428px;
            border-radius: 14px;
            color: #ffffff;
            background:
                radial-gradient(circle at 64% -8%, rgba(255, 248, 178, 0.96) 0 7%, rgba(255, 248, 178, 0.20) 10%, transparent 19%),
                radial-gradient(circle at -8% 105%, rgba(239, 68, 68, 0.82), rgba(239, 68, 68, 0.34) 18%, transparent 38%),
                linear-gradient(135deg, rgba(15, 132, 184, 0.96), rgba(20, 83, 115, 0.94) 52%, rgba(28, 65, 91, 0.95));
            border: 1px solid rgba(255, 255, 255, 0.18);
            box-shadow: 0 18px 40px rgba(0, 0, 0, 0.24);
        }
        .weather-effect {
            opacity: 0.18;
            background-image:
                linear-gradient(90deg, rgba(255,255,255,0.07) 1px, transparent 1px),
                linear-gradient(rgba(255,255,255,0.055) 1px, transparent 1px);
            background-size: 44px 44px;
        }
        .weather-glass {
            min-height: 428px;
            padding: 26px 34px 26px 34px;
            background: rgba(15, 23, 42, 0.06);
            border-radius: 14px;
            display: grid;
            grid-template-rows: auto auto 1fr auto auto;
            gap: 0;
        }
        .weather-top {
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto;
            align-items: start;
            gap: 16px;
        }
        .weather-now {
            display: flex;
            align-items: center;
            gap: 18px;
        }
        .weather-visual-badge {
            width: 54px;
            height: 54px;
            background: transparent;
            border-radius: 50%;
        }
        .weather-visual-badge .weather-hero-img {
            width: 64px;
            height: 64px;
            filter: brightness(1.2) drop-shadow(0 8px 12px rgba(0,0,0,0.18));
        }
        .weather-visual-badge .weather-hero-fallback {
            width: 52px;
            height: 52px;
            background: #facc15;
            color: #1f2937;
        }
        .weather-desc {
            font-size: 30px;
            line-height: 1;
            font-weight: 800;
            color: #ffffff;
            margin: 0;
        }
        .weather-location {
            font-size: 14px;
            color: rgba(255, 255, 255, 0.66);
            font-weight: 700;
            margin-top: 4px;
        }
        .weather-tower {
            display: none;
        }
        .weather-temp {
            font-size: 30px;
            line-height: 1;
            color: #ffffff;
            font-weight: 800;
            margin: 0;
        }
        .weather-range {
            font-size: 13px;
            color: rgba(255, 255, 255, 0.48);
            font-weight: 800;
            margin-top: 8px;
        }
        .weather-attribute-list {
            margin-top: 28px;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px 46px;
        }
        .weather-attribute-list div {
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto;
            gap: 10px;
            color: rgba(255, 255, 255, 0.76);
            font-size: 13px;
            font-weight: 800;
        }
        .weather-attribute-list strong {
            color: #ffffff;
            white-space: nowrap;
        }
        .weather-forecast-row {
            display: grid;
            grid-template-columns: 0.92fr 1.08fr;
            gap: 18px;
            align-items: end;
            margin-top: 22px;
        }
        .weather-forecast-copy span {
            color: rgba(255,255,255,0.56);
            font-size: 12px;
        }
        .weather-forecast-copy strong {
            display: block;
            color: #ffffff;
            font-size: 13px;
            line-height: 1.25;
            margin-top: 6px;
        }
        .weather-forecast-copy small {
            display: block;
            color: rgba(255,255,255,0.58);
            font-size: 10px;
            margin-top: 6px;
        }
        .weather-trend-svg {
            width: 100%;
            height: 112px;
        }
        .weather-trend-grid line {
            stroke: rgba(255,255,255,0.18);
            stroke-dasharray: 5 5;
        }
        .weather-trend-rain rect {
            fill: rgba(96, 165, 250, 0.62);
        }
        .weather-trend-line {
            fill: none;
            stroke: #f59e0b;
            stroke-width: 4;
            stroke-linecap: round;
            stroke-linejoin: round;
        }
        .weather-trend-points circle {
            fill: #11314a;
            stroke: #f59e0b;
            stroke-width: 2;
        }
        .weather-forecast-strip {
            display: flex;
            gap: 10px;
            overflow-x: auto;
            padding: 0 0 6px 0;
            margin-top: -106px;
            width: 100%;
            scrollbar-width: none;
        }
        .weather-forecast-strip::-webkit-scrollbar {
            display: none;
        }
        .weather-forecast-item {
            flex: 0 0 50px;
            min-width: 50px;
            background: transparent;
            border: 0;
            color: #ffffff;
            padding: 0;
            display: grid;
            grid-template-rows: 18px 26px 18px 13px 13px 28px 16px 24px;
            justify-items: center;
            text-align: center;
            gap: 0;
        }
        .weather-forecast-item span {
            color: rgba(255,255,255,0.54);
            font-size: 12px;
            font-weight: 800;
        }
        .weather-forecast-item img {
            width: 28px;
            height: 28px;
            object-fit: contain;
            filter: brightness(1.2) drop-shadow(0 6px 8px rgba(0,0,0,0.18));
        }
        .weather-forecast-item b {
            color: #ffffff;
            font-size: 12px;
            line-height: 1;
        }
        .weather-forecast-item em {
            max-width: 48px;
            color: rgba(255,255,255,0.48);
            font-size: 9px;
            font-style: normal;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .weather-forecast-item small {
            color: rgba(255,255,255,0.60);
            font-size: 10px;
        }
        .weather-precip-track {
            width: 18px;
            height: 28px;
            border-radius: 10px;
            background: rgba(255,255,255,0.20);
            display: flex;
            align-items: flex-end;
            justify-content: center;
            overflow: hidden;
        }
        .weather-precip-track i {
            width: 100%;
            min-height: 3px;
            background: linear-gradient(180deg, #93c5fd, #3b82f6);
        }
        .weather-forecast-item label {
            color: rgba(255,255,255,0.76);
            font-size: 10px;
            font-weight: 800;
            min-height: 14px;
        }
        .weather-forecast-item mark {
            display: grid;
            place-items: center;
            width: 24px;
            height: 24px;
            border-radius: 50%;
            color: #ffffff;
            font-size: 11px;
            font-weight: 800;
            background: rgba(15,23,42,0.38);
            border: 3px solid #f59e0b;
            padding: 0;
        }
        .weather-storm {
            margin-top: 8px;
            border-radius: 999px;
            padding: 8px 12px;
            background: rgba(15,23,42,0.20);
            border: 1px solid rgba(245,158,11,0.80);
        }
        .weather-storm span {
            color: rgba(255,255,255,0.54);
            font-size: 10px;
        }
        .weather-storm strong {
            display: block;
            color: #ffffff;
            font-size: 11px;
            margin-top: 2px;
        }
        .weather-footer {
            display: flex;
            justify-content: space-between;
            gap: 10px;
            margin-top: 8px;
            color: rgba(255,255,255,0.46);
            font-size: 10px;
            flex-wrap: wrap;
        }
        @media (max-width: 640px) {
            .weather-card {
                width: 100%;
                min-height: 0;
            }
            .weather-glass {
                min-height: 0;
                padding: 16px;
            }
            .weather-top,
            .weather-forecast-row {
                grid-template-columns: 1fr;
            }
            .weather-temp-block {
                text-align: left;
            }
            .weather-attribute-list {
                grid-template-columns: 1fr;
                gap: 8px;
                margin-top: 18px;
            }
            .weather-forecast-strip {
                margin-top: 10px;
            }
            .weather-trend-svg {
                height: 86px;
            }
        }
        /* Transmission fault report weather card */
        .weather-card-wrap {
            place-items: stretch;
            margin: 10px 0 12px 0;
        }
        .weather-card {
            width: 100%;
            min-height: 0;
            border-radius: 10px;
            color: #0f172a;
            background: linear-gradient(135deg, #f8fafc 0%, #dbeafe 100%);
            border: 1px solid rgba(148, 163, 184, 0.46);
            box-shadow: 0 10px 26px rgba(15, 23, 42, 0.11);
        }
        .weather-effect {
            display: none;
        }
        .weather-glass {
            min-height: 0;
            padding: 16px;
            border-radius: 10px;
            background: transparent;
            display: grid;
            grid-template-rows: auto;
        }
        .weather-top {
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto;
            gap: 16px;
            align-items: start;
            padding-bottom: 12px;
            border-bottom: 1px solid rgba(148, 163, 184, 0.32);
        }
        .weather-now {
            display: flex;
            align-items: center;
            gap: 14px;
            min-width: 0;
        }
        .weather-visual-badge {
            width: 64px;
            height: 64px;
            border-radius: 10px;
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid rgba(203, 213, 225, 0.75);
            box-shadow: inset 0 1px 8px rgba(255,255,255,0.55);
        }
        .weather-visual-badge .weather-hero-img {
            width: 62px;
            height: 62px;
            filter: drop-shadow(0 8px 10px rgba(15,23,42,0.16));
        }
        .weather-visual-badge .weather-hero-fallback {
            width: 46px;
            height: 46px;
            background: #0f172a;
            color: #ffffff;
            font-size: 11px;
        }
        .weather-desc {
            color: #0f172a;
            font-size: 24px;
            line-height: 1.1;
            font-weight: 800;
            margin: 0;
            text-transform: capitalize;
        }
        .weather-location {
            color: #ef4444;
            font-size: 12px;
            font-weight: 800;
            margin-top: 5px;
        }
        .weather-tower {
            display: block;
            color: #334155;
            font-size: 12px;
            font-weight: 700;
            margin-top: 3px;
        }
        .weather-temp-block {
            text-align: right;
            min-width: 120px;
        }
        .weather-temp {
            color: #0f172a;
            font-size: 34px;
            line-height: 1;
            font-weight: 850;
            margin: 0;
        }
        .weather-range {
            color: #64748b;
            font-size: 12px;
            font-weight: 700;
            margin-top: 5px;
        }
        .weather-attribute-list {
            display: grid;
            grid-template-columns: repeat(6, minmax(0, 1fr));
            gap: 8px;
            margin-top: 12px;
        }
        .weather-attribute-list div {
            display: block;
            border-radius: 8px;
            border: 1px solid rgba(203, 213, 225, 0.76);
            background: rgba(255, 255, 255, 0.70);
            padding: 8px 9px;
            color: #64748b;
            font-size: 11px;
            font-weight: 650;
        }
        .weather-attribute-list strong {
            display: block;
            margin-top: 2px;
            color: #0f172a;
            font-size: 13px;
            white-space: nowrap;
        }
        .weather-forecast-row {
            display: grid;
            grid-template-columns: minmax(0, 0.75fr) minmax(320px, 1.25fr);
            gap: 14px;
            align-items: center;
            margin-top: 12px;
        }
        .weather-forecast-copy {
            border-radius: 8px;
            border: 1px solid rgba(203, 213, 225, 0.76);
            background: rgba(255, 255, 255, 0.64);
            padding: 10px;
            min-width: 0;
        }
        .weather-forecast-copy span,
        .weather-forecast-copy small {
            display: block;
            color: #64748b;
            font-size: 11px;
        }
        .weather-forecast-copy strong {
            display: block;
            color: #0f172a;
            font-size: 13px;
            line-height: 1.3;
            margin: 4px 0 5px 0;
        }
        .weather-trend-svg {
            height: 88px;
            width: 100%;
            border-radius: 8px;
            border: 1px solid rgba(203, 213, 225, 0.76);
            background: rgba(255, 255, 255, 0.52);
        }
        .weather-trend-grid line {
            stroke: rgba(100, 116, 139, 0.22);
            stroke-dasharray: 4 5;
        }
        .weather-trend-rain rect {
            fill: rgba(59, 130, 246, 0.46);
        }
        .weather-trend-line {
            fill: none;
            stroke: #f59e0b;
            stroke-width: 4;
            stroke-linecap: round;
            stroke-linejoin: round;
        }
        .weather-trend-points circle {
            fill: #ffffff;
            stroke: #f59e0b;
            stroke-width: 2;
        }
        .weather-forecast-strip {
            display: flex;
            gap: 8px;
            overflow-x: auto;
            padding: 10px 0 4px 0;
            margin-top: 2px;
            scrollbar-width: thin;
        }
        .weather-forecast-item {
            flex: 0 0 82px;
            min-width: 82px;
            border-radius: 8px;
            border: 1px solid rgba(203, 213, 225, 0.78);
            background: rgba(255, 255, 255, 0.70);
            color: #0f172a;
            padding: 7px 6px;
            display: grid;
            grid-template-rows: 16px 30px 16px 15px 14px 22px 12px 22px;
            justify-items: center;
            gap: 1px;
            text-align: center;
        }
        .weather-forecast-item span,
        .weather-forecast-item small {
            color: #64748b;
            font-size: 10px;
            font-weight: 700;
        }
        .weather-forecast-item img {
            width: 30px;
            height: 30px;
            object-fit: contain;
            filter: drop-shadow(0 6px 8px rgba(15,23,42,0.14));
        }
        .weather-forecast-item b {
            color: #0f172a;
            font-size: 12px;
            line-height: 1;
        }
        .weather-forecast-item em {
            max-width: 72px;
            color: #334155;
            font-size: 10px;
            font-style: normal;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .weather-forecast-item label {
            color: #475569;
            font-size: 9px;
            font-weight: 700;
            min-height: 10px;
        }
        .weather-forecast-item mark {
            display: grid;
            place-items: center;
            width: 22px;
            height: 22px;
            border-radius: 50%;
            color: #0f172a;
            background: #ffffff;
            border: 3px solid #f59e0b;
            font-size: 10px;
            font-weight: 800;
            padding: 0;
        }
        .weather-precip-track {
            width: 18px;
            height: 22px;
            border-radius: 999px;
            background: rgba(203, 213, 225, 0.58);
            display: flex;
            align-items: flex-end;
            justify-content: center;
            overflow: hidden;
        }
        .weather-precip-track i {
            width: 100%;
            min-height: 3px;
            background: linear-gradient(180deg, #60a5fa, #2563eb);
        }
        .weather-storm {
            margin-top: 10px;
            border-radius: 8px;
            padding: 8px 10px;
            background: rgba(255, 237, 213, 0.82);
            border: 1px solid rgba(249, 115, 22, 0.62);
        }
        .weather-storm span {
            display: block;
            color: #9a3412;
            font-size: 11px;
            margin-bottom: 2px;
        }
        .weather-storm strong {
            color: #0f172a;
            font-size: 12px;
        }
        .weather-footer {
            display: flex;
            justify-content: space-between;
            gap: 10px;
            margin-top: 9px;
            color: #64748b;
            font-size: 10px;
            flex-wrap: wrap;
        }
        @media (max-width: 1000px) {
            .weather-attribute-list {
                grid-template-columns: repeat(3, minmax(0, 1fr));
            }
            .weather-forecast-row {
                grid-template-columns: 1fr;
            }
        }
        @media (max-width: 640px) {
            .weather-glass {
                padding: 12px;
            }
            .weather-top {
                grid-template-columns: 1fr;
            }
            .weather-temp-block {
                text-align: left;
            }
            .weather-attribute-list {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .weather-desc {
                font-size: 21px;
            }
            .weather-forecast-item {
                flex-basis: 78px;
                min-width: 78px;
            }
        }
        </style>
        <div class="weather-card-wrap">
        """)
        + "".join(cards_html)
        + "</div>"
    )


@st.cache_data(ttl=900, show_spinner=False)
def fetch_open_meteo_current_weather(lat: float, lon: float):
    try:
        response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": float(lat),
                "longitude": float(lon),
                "current": ",".join(
                    [
                        "temperature_2m",
                        "relative_humidity_2m",
                        "precipitation",
                        "rain",
                        "weather_code",
                        "cloud_cover",
                        "wind_speed_10m",
                        "wind_direction_10m",
                    ]
                ),
                "timezone": "Asia/Jakarta",
                "forecast_days": 1,
            },
            timeout=8,
        )
        response.raise_for_status()
        current = response.json().get("current", {})
        code = current.get("weather_code")
        return {
            "time": current.get("time"),
            "weather_code": code,
            "weather": weather_code_label(code),
            "temperature_c": current.get("temperature_2m"),
            "humidity_pct": current.get("relative_humidity_2m"),
            "precipitation_mm": current.get("precipitation"),
            "rain_mm": current.get("rain"),
            "cloud_cover_pct": current.get("cloud_cover"),
            "wind_speed_kmh": current.get("wind_speed_10m"),
            "wind_direction_deg": current.get("wind_direction_10m"),
        }
    except Exception as exc:
        return {"error": str(exc)}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_open_meteo_recent_thunderstorm(lat: float, lon: float, past_days: int = 7):
    try:
        response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": float(lat),
                "longitude": float(lon),
                "hourly": "weather_code,precipitation",
                "past_days": int(past_days),
                "forecast_days": 1,
                "timezone": "Asia/Jakarta",
            },
            timeout=8,
        )
        response.raise_for_status()
        hourly = response.json().get("hourly", {})
        times = hourly.get("time", [])
        codes = hourly.get("weather_code", [])
        precipitation = hourly.get("precipitation", [])
        for idx in range(min(len(times), len(codes)) - 1, -1, -1):
            try:
                code = int(codes[idx])
            except (TypeError, ValueError):
                continue
            if code in THUNDERSTORM_WEATHER_CODES:
                rain_value = precipitation[idx] if idx < len(precipitation) else None
                return {
                    "time": times[idx],
                    "weather_code": code,
                    "weather": weather_code_label(code),
                    "precipitation_mm": rain_value,
                }
        return {"time": None, "weather": f"Tidak ada indikasi thunderstorm {past_days} hari terakhir"}
    except Exception as exc:
        return {"error": str(exc)}


def get_selected_fault_location_option(key_prefix: str = "summary_tower_fault"):
    fault_options = get_fault_location_map_options()
    if not fault_options:
        return None
    option_keys = [option["key"] for option in fault_options]
    default_key = "de" if "de" in option_keys else option_keys[0]
    selected_key = st.session_state.get(f"{key_prefix}_fault_source", default_key)
    if selected_key not in option_keys:
        selected_key = default_key
    return next(option for option in fault_options if option["key"] == selected_key)


def get_fault_adjacent_tower_rows(map_df: pd.DataFrame, distance_km: float):
    if map_df.empty or "_cum_km" not in map_df.columns:
        return []
    segment = get_fault_tower_segment(map_df, distance_km)
    if segment:
        return [
            ("Tower A", segment["prev"]),
            ("Tower B", segment["next"]),
        ]
    path_df = map_df.dropna(subset=["lat", "lon", "_cum_km"]).copy()
    if path_df.empty:
        return []
    path_df["_fault_abs_km"] = (path_df["_cum_km"].astype(float) - float(distance_km)).abs()
    return [
        (f"Tower terdekat {idx + 1}", row)
        for idx, (_, row) in enumerate(path_df.sort_values("_fault_abs_km").head(2).iterrows())
    ]


def weather_card_html(weather_rows):
    cards_html = []
    for row in weather_rows:
        forecast = row.get("Forecast Summary") or {}
        forecast_items = forecast.get("items", [])
        theme = weather_theme_for_code(row.get("Weather Code"))
        icon_url = row.get("Weather Icon URL")
        if icon_url:
            visual_html = f"<img class='weather-icon-main' src='{icon_url}' alt='{row.get('Current Weather', 'Weather')}' />"
        else:
            visual_html = f"<div class='weather-icon-fallback'>{weather_icon_for_code(row.get('Weather Code'))}</div>"

        thunder_text = row.get("Last Thunderstorm Indication", "-")
        storm_class = "weather-storm-muted"
        if row.get("Last Thunderstorm Time"):
            thunder_text = f"{row.get('Last Thunderstorm Time')} | {row.get('Last Thunderstorm Weather', '-')}"
            storm_class = "weather-storm-active"

        forecast_items_html = ""
        temp_values = []
        for item in forecast_items[:8]:
            temp_value = item.get("temperature_c")
            try:
                temp_values.append(float(temp_value))
            except (TypeError, ValueError):
                pass
            item_icon = (
                f"<img src='{item.get('icon_url')}' alt='{item.get('description', 'Forecast')}' />"
                if item.get("icon_url")
                else f"<strong>{weather_icon_for_code(item.get('weather_code'))}</strong>"
            )
            pop_label = f"{item.get('pop_pct')}%" if item.get("pop_pct") is not None else "-"
            forecast_items_html += (
                "<div class='forecast-item'>"
                f"<span>{item.get('time', '-')}</span>"
                f"{item_icon}"
                f"<b>{safe_display_number(temp_value, 0, '°C')}</b>"
                f"<em>{item.get('description', '-')}</em>"
                f"<small>{pop_label}</small>"
                "</div>"
            )

        temp_range = "-"
        if temp_values:
            temp_range = f"{safe_display_number(min(temp_values), 0, ' C')} - {safe_display_number(max(temp_values), 0, ' C')}"
        forecast_pop = f"{forecast.get('max_pop') * 100.0:.0f}%" if forecast.get("max_pop") is not None else "-"
        forecast_precip = safe_display_number(forecast.get("total_precip_mm", 0.0), 2, " mm")
        forecast_signal = "Ada potensi petir" if forecast.get("thunder_count", 0) else "Tidak ada sinyal petir kuat"

        cards_html.append(
            f"""
            <div class="weather-card weather-theme-{theme}">
                <section class="current-panel">
                    <div class="current-head">
                        <div>
                            <div class="weather-role">{row.get('Location', '-')}</div>
                            <div class="weather-title">{row.get('Tower', '-')}</div>
                            <div class="weather-temp">{safe_display_number(row.get('Temperature C'), 1, ' C')}</div>
                            <div class="weather-desc">{row.get('Current Weather', '-')}</div>
                        </div>
                        <div class="weather-visual">
                            {visual_html}
                            <div>{str(row.get('Current Weather') or '-').title()}</div>
                        </div>
                    </div>
                    <div class="metric-grid">
                        <div><span>Hujan saat ini</span><strong>{safe_display_number(row.get('Rain mm'), 2, ' mm')}</strong></div>
                        <div><span>Kelembapan</span><strong>{safe_display_number(row.get('Humidity %'), 0, '%')}</strong></div>
                        <div><span>Tutupan awan</span><strong>{safe_display_number(row.get('Cloud Cover %'), 0, '%')}</strong></div>
                        <div><span>Kecepatan angin</span><strong>{safe_display_number(row.get('Wind km/h'), 1, ' km/h')}</strong></div>
                    </div>
                    <div class="storm-box {storm_class}">
                        <span>Indikasi thunderstorm</span>
                        <strong>{thunder_text}</strong>
                    </div>
                    <div class="weather-meta">
                        <span>Kumulatif {safe_display_number(row.get('Cumulative km'), 3, ' km')}</span>
                        <span>{safe_display_number(row.get('Latitude'), 6)}, {safe_display_number(row.get('Longitude'), 6)}</span>
                    </div>
                    <div class="weather-time">Update cuaca: {row.get('Weather Time') or '-'} | {row.get('Weather Source', '-')}</div>
                </section>
                <section class="forecast-panel">
                    <div class="forecast-copy">
                        <span>Prakiraan 12 jam ke depan</span>
                        <strong>{forecast.get('summary', 'Forecast belum tersedia.')}</strong>
                    </div>
                    <div class="forecast-metrics">
                        <div><span>Peluang hujan tertinggi</span><strong>{forecast_pop}</strong></div>
                        <div><span>Perkiraan hujan total</span><strong>{forecast_precip}</strong></div>
                        <div><span>Rentang suhu</span><strong>{temp_range}</strong></div>
                        <div><span>Indikasi petir</span><strong>{forecast_signal}</strong></div>
                    </div>
                    <div class="forecast-strip">
                        {forecast_items_html}
                    </div>
                </section>
            </div>
            """
        )

    return (
        """
        <style>
        * { box-sizing: border-box; }
        body { margin: 0; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
        .weather-card-wrap {
            display: grid;
            grid-template-columns: minmax(0, 1fr);
            gap: 14px;
        }
        .weather-card {
            display: grid;
            grid-template-columns: minmax(360px, 0.92fr) minmax(420px, 1.08fr);
            gap: 12px;
            padding: 14px;
            border-radius: 10px;
            border: 1px solid rgba(148, 163, 184, 0.46);
            background: linear-gradient(135deg, #f8fafc 0%, #dbeafe 100%);
            color: #0f172a;
            box-shadow: 0 10px 26px rgba(15, 23, 42, 0.11);
        }
        .weather-theme-clear { background: linear-gradient(135deg, #fff7ed 0%, #dbeafe 100%); }
        .weather-theme-cloud { background: linear-gradient(135deg, #f8fafc 0%, #dbeafe 56%, #e2e8f0 100%); }
        .weather-theme-rain { background: linear-gradient(135deg, #eef6ff 0%, #c7d2fe 100%); }
        .weather-theme-storm { background: linear-gradient(135deg, #eef2ff 0%, #fed7aa 100%); }
        .weather-theme-mist { background: linear-gradient(135deg, #f8fafc 0%, #e5e7eb 100%); }
        .current-panel,
        .forecast-panel {
            min-width: 0;
            border-radius: 8px;
            border: 1px solid rgba(203, 213, 225, 0.72);
            background: rgba(255, 255, 255, 0.56);
            padding: 12px;
        }
        .current-head {
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto;
            gap: 14px;
            align-items: start;
            padding-bottom: 10px;
            border-bottom: 1px solid rgba(148, 163, 184, 0.28);
        }
        .weather-role {
            color: #ef4444;
            font-size: 12px;
            font-weight: 800;
        }
        .weather-title {
            color: #0f172a;
            font-size: 13px;
            font-weight: 800;
            margin-top: 3px;
        }
        .weather-temp {
            font-size: 40px;
            line-height: 1;
            font-weight: 850;
            margin-top: 18px;
        }
        .weather-desc {
            color: #334155;
            font-size: 16px;
            margin-top: 5px;
            text-transform: capitalize;
        }
        .weather-visual {
            width: 130px;
            border-radius: 8px;
            border: 1px solid rgba(203, 213, 225, 0.74);
            background: rgba(255, 255, 255, 0.70);
            display: grid;
            place-items: center;
            padding: 8px;
            text-align: center;
            color: #0f172a;
            font-size: 12px;
            font-weight: 800;
        }
        .weather-icon-main {
            width: 92px;
            height: 92px;
            object-fit: contain;
            filter: drop-shadow(0 8px 10px rgba(15,23,42,0.16));
        }
        .weather-icon-fallback {
            width: 64px;
            height: 64px;
            border-radius: 50%;
            display: grid;
            place-items: center;
            background: #0f172a;
            color: #ffffff;
            font-size: 13px;
            font-weight: 800;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 8px;
            margin-top: 10px;
        }
        .metric-grid div,
        .forecast-metrics div,
        .forecast-item {
            border-radius: 8px;
            border: 1px solid rgba(203, 213, 225, 0.76);
            background: rgba(255, 255, 255, 0.70);
        }
        .metric-grid div,
        .forecast-metrics div {
            padding: 8px 9px;
        }
        .metric-grid span,
        .forecast-metrics span,
        .storm-box span,
        .forecast-copy span {
            display: block;
            color: #64748b;
            font-size: 11px;
            margin-bottom: 2px;
        }
        .metric-grid strong,
        .forecast-metrics strong,
        .storm-box strong,
        .forecast-copy strong {
            color: #0f172a;
            font-size: 13px;
        }
        .storm-box {
            margin-top: 9px;
            border-radius: 8px;
            padding: 8px 10px;
            background: rgba(255, 237, 213, 0.82);
            border: 1px solid rgba(249, 115, 22, 0.62);
        }
        .storm-box span { color: #9a3412; }
        .weather-meta,
        .weather-time {
            display: flex;
            justify-content: space-between;
            gap: 10px;
            color: #64748b;
            font-size: 10px;
            margin-top: 8px;
            flex-wrap: wrap;
        }
        .forecast-copy {
            padding-bottom: 8px;
        }
        .forecast-copy strong {
            display: block;
            line-height: 1.3;
        }
        .forecast-metrics {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 8px;
            margin-top: 2px;
        }
        .forecast-strip {
            display: flex;
            gap: 8px;
            overflow-x: auto;
            padding-top: 10px;
            scrollbar-width: thin;
        }
        .forecast-item {
            flex: 0 0 82px;
            min-width: 82px;
            padding: 7px 6px;
            display: grid;
            grid-template-rows: 16px 30px 16px 15px 14px;
            justify-items: center;
            gap: 1px;
            text-align: center;
        }
        .forecast-item span,
        .forecast-item small {
            color: #64748b;
            font-size: 10px;
            font-weight: 700;
        }
        .forecast-item img {
            width: 30px;
            height: 30px;
            object-fit: contain;
            filter: drop-shadow(0 6px 8px rgba(15,23,42,0.14));
        }
        .forecast-item b {
            color: #0f172a;
            font-size: 12px;
            line-height: 1;
        }
        .forecast-item em {
            max-width: 72px;
            color: #334155;
            font-size: 10px;
            font-style: normal;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .forecast-item strong {
            display: grid;
            place-items: center;
            width: 26px;
            height: 26px;
            border-radius: 50%;
            background: #0f172a;
            color: #ffffff;
            font-size: 9px;
        }
        @media (max-width: 980px) {
            .weather-card {
                grid-template-columns: 1fr;
            }
            .forecast-metrics {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
        @media (max-width: 640px) {
            .weather-card {
                padding: 10px;
            }
            .current-head {
                grid-template-columns: 1fr;
            }
            .weather-visual {
                width: 100%;
                grid-template-columns: auto 1fr;
                justify-content: start;
                text-align: left;
            }
            .weather-icon-main {
                width: 58px;
                height: 58px;
            }
            .metric-grid,
            .forecast-metrics {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
        </style>
        <div class="weather-card-wrap">
        """
        + "".join(cards_html)
        + "</div>"
    )


def weather_card_html(weather_rows):
    cards_html = []
    for row in weather_rows:
        forecast = row.get("Forecast Summary") or {}
        forecast_items = forecast.get("items", [])[:8]
        current_icon = f"<div class='weather-symbol weather-symbol-{weather_theme_for_code(row.get('Weather Code'))}'>{weather_symbol_for_code(row.get('Weather Code'))}</div>"
        forecast_cards = ""
        rain_bars = ""
        max_pop_seen = 1
        for item in forecast_items:
            try:
                max_pop_seen = max(max_pop_seen, int(item.get("pop_pct") or 0))
            except (TypeError, ValueError):
                pass
        for item in forecast_items:
            item_icon = f"<strong class='weather-symbol-mini weather-symbol-{weather_theme_for_code(item.get('weather_code'))}'>{weather_symbol_for_code(item.get('weather_code'))}</strong>"
            temp_value = item.get("temperature_c")
            try:
                pop_num = int(item.get("pop_pct") or 0)
            except (TypeError, ValueError):
                pop_num = 0
            bar_height = 12 + (pop_num / max_pop_seen) * 58
            forecast_cards += (
                "<div class='daily-card'>"
                f"<span>{item.get('time', '-')}</span>"
                f"{item_icon}"
                f"<b>{safe_display_number(temp_value, 0, ' C')}</b>"
                f"<small>{item.get('description', '-')}</small>"
                "</div>"
            )
            rain_bars += (
                "<div class='rain-bar-wrap'>"
                f"<i style='height:{bar_height:.0f}px'></i>"
                f"<span>{pop_num}%</span>"
                f"<small>{item.get('time', '-')}</small>"
                "</div>"
            )
        forecast_pop = f"{forecast.get('max_pop') * 100.0:.0f}%" if forecast.get("max_pop") is not None else "-"
        forecast_precip = safe_display_number(forecast.get("total_precip_mm", 0.0), 2, " mm")
        forecast_signal = "Ada potensi petir" if forecast.get("thunder_count", 0) else "Tidak ada sinyal petir kuat"
        trend_svg = build_weather_trend_svg(forecast_items)
        cards_html.append(
            textwrap.dedent(f"""
            <div class="dashboard-weather-card">
                <aside class="dashboard-current">
                    <div class="dashboard-location">
                        <strong>{row.get('Location', '-')}</strong>
                        <span>{row.get('Tower', '-')}</span>
                    </div>
                    <div class="dashboard-update">{row.get('Weather Time') or '-'}</div>
                    <div class="dashboard-icon">{current_icon}</div>
                    <div class="dashboard-condition">{row.get('Current Weather', '-')}</div>
                    <div class="dashboard-temp">{safe_display_number(row.get('Temperature C'), 1, '°C')}</div>
                    <div class="dashboard-feels">Terasa seperti {safe_display_number(row.get('Feels Like C'), 1, '°C')}</div>
                    <div class="dashboard-metrics-list">
                        <div><span>Hujan</span><b>{safe_display_number(row.get('Rain mm'), 2, ' mm')}</b></div>
                        <div><span>Angin</span><b>{safe_display_number(row.get('Wind km/h'), 1, ' km/h')}</b></div>
                        <div><span>Kelembapan</span><b>{safe_display_number(row.get('Humidity %'), 0, '%')}</b></div>
                        <div><span>Tutupan awan</span><b>{safe_display_number(row.get('Cloud Cover %'), 0, '%')}</b></div>
                        <div><span>Kumulatif fault</span><b>{safe_display_number(row.get('Cumulative km'), 3, ' km')}</b></div>
                    </div>
                </aside>
                <main class="dashboard-forecast">
                    <section class="dashboard-tile temp-tile">
                        <header>Tren suhu, °C</header>
                        {trend_svg}
                    </section>
                    <section class="dashboard-tile rain-tile">
                        <header>Peluang hujan, %</header>
                        <div class="rain-bars">{rain_bars}</div>
                    </section>
                    <section class="dashboard-tile summary-tile compact-summary">
                        <header>Ringkasan titik gangguan</header>
                        <strong>{forecast.get('summary', 'Forecast belum tersedia.')}</strong>
                        <div><span>Peluang hujan tertinggi</span><b>{forecast_pop}</b></div>
                        <div><span>Perkiraan hujan total</span><b>{forecast_precip}</b></div>
                    </section>
                    <section class="dashboard-daily">{forecast_cards}</section>
                    <section class="dashboard-note">
                        <small>{safe_display_number(row.get('Latitude'), 6)}, {safe_display_number(row.get('Longitude'), 6)} | {row.get('Weather Source', '-')}</small>
                    </section>
                </main>
            </div>
            """).strip()
        )
    return (
        textwrap.dedent("""
        <style>
        * { box-sizing: border-box; }
        body { margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #111827; }
        .weather-card-wrap { display: grid; gap: 16px; }
        .dashboard-weather-card {
            width: min(100%, 1440px);
            margin: 0 auto;
            display: grid;
            grid-template-columns: 250px minmax(0, 1fr);
            border-radius: 14px;
            overflow: hidden;
            background: #ffffff;
            border: 1px solid rgba(148, 163, 184, 0.34);
            box-shadow: none;
        }
        .dashboard-current {
            background: linear-gradient(180deg, #fbfdff 0%, #f8fafc 100%);
            padding: 18px 22px;
            border-right: 1px solid rgba(226, 232, 240, 0.9);
        }
        .dashboard-location { display: grid; gap: 4px; font-size: 12px; color: #111827; }
        .dashboard-location span, .dashboard-update, .dashboard-feels { color: #6b7280; font-size: 12px; }
        .dashboard-update { margin-top: 14px; text-align: center; }
        .dashboard-icon { width: 104px; height: 104px; display: grid; place-items: center; margin: 12px auto 4px auto; }
        .weather-symbol { font-size: 78px; line-height: 1; filter: drop-shadow(0 6px 8px rgba(15,23,42,0.18)); }
        .weather-symbol-mini { display: block; font-size: 30px; line-height: 1; background: transparent; color: inherit; }
        .weather-symbol-clear { color: #f59e0b; }
        .weather-symbol-cloud { color: #64748b; }
        .weather-symbol-rain { color: #2563eb; }
        .weather-symbol-storm { color: #d97706; }
        .weather-symbol-mist, .weather-symbol-neutral { color: #475569; }
        .dashboard-condition { text-align: center; text-transform: capitalize; font-size: 13px; color: #111827; }
        .dashboard-temp { text-align: center; font-size: 54px; line-height: 1; font-weight: 300; color: #f59e0b; margin-top: 8px; }
        .dashboard-feels { text-align: center; margin-top: 6px; }
        .dashboard-metrics-list { display: grid; gap: 9px; margin-top: 18px; }
        .dashboard-metrics-list div { display: flex; justify-content: space-between; gap: 16px; align-items: center; font-size: 13px; }
        .dashboard-metrics-list b { color: #111827; font-weight: 700; white-space: nowrap; }
        .dashboard-metrics-list b::before { display: none; }
        .dashboard-forecast { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); grid-auto-rows: min-content; gap: 12px; padding: 18px 22px; background: #ffffff; }
        .dashboard-tile, .dashboard-note { border-radius: 8px; background: #f8fafc; border: 1px solid rgba(241, 245, 249, 0.95); padding: 14px; }
        .dashboard-tile header { font-size: 14px; color: #111827; margin-bottom: 8px; }
        .weather-trend-svg { width: 100%; height: 94px; display: block; }
        .weather-trend-grid line { stroke: rgba(148, 163, 184, 0.35); stroke-dasharray: 4 5; }
        .weather-trend-rain rect { fill: rgba(56, 189, 248, 0.45); }
        .weather-trend-line { fill: none; stroke: #eab308; stroke-width: 3; stroke-linecap: round; stroke-linejoin: round; }
        .weather-trend-points circle { fill: #f8fafc; stroke: #f59e0b; stroke-width: 2; }
        .weather-trend-labels text { fill: #334155; font-size: 20px; font-weight: 700; }
        .weather-trend-axis text { fill: #64748b; font-size: 16px; font-weight: 600; }
        .rain-bars { height: 94px; display: flex; align-items: end; justify-content: space-between; gap: 8px; padding-top: 8px; }
        .rain-bar-wrap { flex: 1; min-width: 24px; display: grid; grid-template-rows: 1fr auto auto; justify-items: center; align-items: end; gap: 4px; color: #6b7280; font-size: 10px; }
        .rain-bar-wrap i { width: 100%; max-width: 28px; display: block; border-radius: 4px 4px 0 0; background: linear-gradient(180deg, #0ea5e9, #bae6fd); }
        .summary-tile { grid-column: span 2; }
        .compact-summary { display: grid; grid-template-columns: minmax(260px, 1.7fr) repeat(2, minmax(150px, 1fr)); gap: 10px; align-items: center; padding: 10px 12px; }
        .compact-summary header { margin: 0; grid-column: 1 / -1; }
        .compact-summary strong { display: block; font-size: 14px; line-height: 1.25; }
        .compact-summary div { border-radius: 8px; background: #ffffff; border: 1px solid rgba(226,232,240,0.92); padding: 8px 10px; }
        .compact-summary span { display: block; color: #6b7280; font-size: 11px; margin-bottom: 3px; }
        .dashboard-daily { grid-column: span 2; display: grid; grid-template-columns: repeat(8, minmax(0, 1fr)); gap: 10px; padding: 4px 0 0 0; }
        .daily-card { min-height: 112px; display: grid; grid-template-columns: 1fr; grid-template-rows: auto 38px auto auto; justify-items: center; align-items: center; border-radius: 10px; background: #f8fafc; border: 1px solid rgba(226,232,240,0.95); padding: 10px 8px; text-align: center; }
        .daily-card span { color: #6b7280; font-size: 11px; }
        .daily-card img, .daily-card strong { width: 38px; height: 38px; object-fit: contain; filter: saturate(1.2) contrast(1.1); }
        .daily-card b { font-size: 20px; font-weight: 500; }
        .daily-card small { color: #9ca3af; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 100%; font-size: 10px; }
        .dashboard-note { grid-column: span 2; display: grid; gap: 4px; color: #6b7280; font-size: 11px; padding: 8px 10px; }
        .dashboard-note b { color: #111827; }
        @media (max-width: 900px) {
            .dashboard-weather-card { grid-template-columns: 1fr; }
            .dashboard-current { border-right: 0; border-bottom: 1px solid rgba(226,232,240,0.9); }
            .dashboard-forecast { grid-template-columns: 1fr; }
            .summary-tile, .dashboard-daily, .dashboard-note { grid-column: span 1; }
            .compact-summary { grid-template-columns: 1fr; }
            .compact-summary header { grid-column: auto; }
            .dashboard-daily { grid-template-columns: repeat(4, minmax(0, 1fr)); }
        }
        @media (max-width: 520px) {
            .dashboard-current, .dashboard-forecast { padding: 20px; }
            .dashboard-temp { font-size: 46px; }
            .dashboard-daily { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }
        </style>
        <div class="weather-card-wrap">
        """).strip()
        + "".join(cards_html)
        + "</div>"
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
    st.session_state["openweather_lightning_api_key"] = openweather_key_input

    fault_lat, fault_lon, fault_cum_km = fault_location
    if openweather_key_input:
        current = fetch_openweather_onecall_current_weather(fault_lat, fault_lon, openweather_key_input)
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
            forecast_payload = fetch_openweather_onecall_15min_forecast(fault_lat, fault_lon, openweather_key_input)
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

    thunderstorm = fetch_open_meteo_recent_thunderstorm(fault_lat, fault_lon, past_days=7)
    current_summary = f"Gagal baca cuaca: {current['error']}" if current.get("error") else current.get("weather", "-")
    if thunderstorm.get("error"):
        storm_summary = f"Gagal baca indikasi: {thunderstorm['error']}"
        storm_time = None
        storm_weather = storm_summary
    elif thunderstorm.get("time"):
        storm_time = thunderstorm.get("time")
        storm_weather = thunderstorm.get("weather", "-")
        storm_summary = f"{storm_time} | {storm_weather}"
    else:
        storm_time = None
        storm_weather = thunderstorm.get("weather", "-")
        storm_summary = thunderstorm.get("weather", "-")

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
            "Last Thunderstorm Indication": storm_summary,
            "Last Thunderstorm Time": storm_time,
            "Last Thunderstorm Weather": storm_weather,
        }
    ]

    st.markdown(weather_card_html(weather_rows), unsafe_allow_html=True)

def prepare_tower_map_dataframe(tower_df: pd.DataFrame):
    if tower_df is None or tower_df.empty or "LATITUDE" not in tower_df.columns or "LONGITUDE" not in tower_df.columns:
        return pd.DataFrame()
    map_df = tower_df.copy()
    map_df["lat"] = pd.to_numeric(map_df["LATITUDE"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    map_df["lon"] = pd.to_numeric(map_df["LONGITUDE"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    if "JARAK km" not in map_df.columns and "JARAK" in map_df.columns:
        map_df["JARAK km"] = pd.to_numeric(map_df["JARAK"].astype(str).str.replace(",", ".", regex=False), errors="coerce") / 1000.0
    if "KUMULATIF km" not in map_df.columns and "KUMULATIF" in map_df.columns:
        map_df["KUMULATIF km"] = pd.to_numeric(map_df["KUMULATIF"].astype(str).str.replace(",", ".", regex=False), errors="coerce") / 1000.0
    map_df = map_df.dropna(subset=["lat", "lon"])
    if "KUMULATIF km" in map_df.columns:
        map_df["_cum_km"] = pd.to_numeric(map_df["KUMULATIF km"], errors="coerce")
    elif "JARAK km" in map_df.columns:
        map_df["_cum_km"] = pd.to_numeric(map_df["JARAK km"], errors="coerce").fillna(0.0).cumsum()
    else:
        map_df["_cum_km"] = np.nan
    return map_df.reset_index(drop=True)


def get_fault_location_map_options():
    options = []
    two_result = st.session_state.get("two_ended_result")
    if two_result:
        options.append(
            {
                "key": "de",
                "label": "Double-End",
                "distance_km": float(two_result.get("distance_from_original_local_km", two_result.get("distance_km", 0.0)) or 0.0),
                "quality": st.session_state.get("two_ended_quality", {}).get("quality_score"),
            }
        )
    single_result = st.session_state.get("single_ended_result")
    if single_result:
        options.append(
            {
                "key": "se_local",
                "label": "Single-End GI Lokal",
                "distance_km": float(single_result.get("recommended_distance_km", 0.0) or 0.0),
                "status": single_result.get("status"),
            }
        )
    remote_single_result = st.session_state.get("remote_single_ended_result")
    line_length_km = st.session_state.get("tower_schedule_selected_length_km")
    if line_length_km is None and "line_param" in st.session_state:
        line_length_km = st.session_state["line_param"].get("length_km")
    if remote_single_result and line_length_km is not None:
        options.append(
            {
                "key": "se_remote",
                "label": "Single-End GI Remote",
                "distance_km": float(line_length_km) - float(remote_single_result.get("recommended_distance_km", 0.0) or 0.0),
                "status": remote_single_result.get("status"),
            }
        )
    return options


def interpolate_tower_path_location(map_df: pd.DataFrame, distance_km: float):
    if map_df.empty or "_cum_km" not in map_df.columns:
        return None, "Data kumulatif tower belum tersedia."
    path_df = map_df.dropna(subset=["lat", "lon", "_cum_km"]).sort_values("_cum_km").reset_index(drop=True)
    if path_df.empty:
        return None, "Data kumulatif tower belum dapat dibaca."
    if len(path_df) == 1:
        row = path_df.iloc[0]
        return (float(row["lat"]), float(row["lon"]), float(row["_cum_km"])), "Hanya satu titik tower tersedia; lokasi fault ditempatkan pada titik tersebut."

    target = float(distance_km)
    if target <= float(path_df.iloc[0]["_cum_km"]):
        row = path_df.iloc[0]
        return (float(row["lat"]), float(row["lon"]), float(row["_cum_km"])), "Jarak fault berada sebelum tower pertama pada data terfilter."
    if target >= float(path_df.iloc[-1]["_cum_km"]):
        row = path_df.iloc[-1]
        return (float(row["lat"]), float(row["lon"]), float(row["_cum_km"])), "Jarak fault melebihi tower terakhir pada data terfilter."

    for idx in range(1, len(path_df)):
        prev_row = path_df.iloc[idx - 1]
        next_row = path_df.iloc[idx]
        prev_cum = float(prev_row["_cum_km"])
        next_cum = float(next_row["_cum_km"])
        if prev_cum <= target <= next_cum:
            if abs(next_cum - prev_cum) < 1e-9:
                ratio = 0.0
            else:
                ratio = (target - prev_cum) / (next_cum - prev_cum)
            lat = float(prev_row["lat"]) + ratio * (float(next_row["lat"]) - float(prev_row["lat"]))
            lon = float(prev_row["lon"]) + ratio * (float(next_row["lon"]) - float(prev_row["lon"]))
            return (lat, lon, target), None
    return None, "Lokasi fault belum dapat diinterpolasi pada jalur tower."


def get_fault_tower_segment(map_df: pd.DataFrame, distance_km: float):
    if map_df.empty or "_cum_km" not in map_df.columns:
        return None
    path_df = map_df.dropna(subset=["lat", "lon", "_cum_km"]).sort_values("_cum_km").reset_index(drop=True)
    if len(path_df) < 2:
        return None

    target = float(distance_km)
    for idx in range(1, len(path_df)):
        prev_row = path_df.iloc[idx - 1]
        next_row = path_df.iloc[idx]
        prev_cum = float(prev_row["_cum_km"])
        next_cum = float(next_row["_cum_km"])
        if prev_cum <= target <= next_cum:
            ratio = 0.0 if abs(next_cum - prev_cum) < 1e-9 else (target - prev_cum) / (next_cum - prev_cum)
            return {
                "prev": prev_row,
                "next": next_row,
                "ratio": ratio,
                "span_distance_km": abs(next_cum - prev_cum),
            }
    return None


def build_nearby_fault_tower_table(map_df: pd.DataFrame, distance_km: float, window: int = 5):
    if map_df.empty or "_cum_km" not in map_df.columns:
        return pd.DataFrame()

    path_df = map_df.dropna(subset=["_cum_km"]).sort_values("_cum_km").reset_index(drop=True)
    if path_df.empty:
        return pd.DataFrame()

    target = float(distance_km)
    prev_idx = 0
    next_idx = 0
    if target <= float(path_df.iloc[0]["_cum_km"]):
        prev_idx = next_idx = 0
    elif target >= float(path_df.iloc[-1]["_cum_km"]):
        prev_idx = next_idx = len(path_df) - 1
    else:
        for idx in range(1, len(path_df)):
            if float(path_df.iloc[idx - 1]["_cum_km"]) <= target <= float(path_df.iloc[idx]["_cum_km"]):
                prev_idx = idx - 1
                next_idx = idx
                break

    start_idx = max(prev_idx - window, 0)
    end_idx = min(next_idx + window, len(path_df) - 1)
    nearby_df = path_df.iloc[start_idx : end_idx + 1].copy()
    nearby_df.insert(0, "Distance from Fault km", nearby_df["_cum_km"].astype(float) - target)
    nearby_df.insert(0, "Fault Context", "Nearby tower")
    if prev_idx == next_idx:
        nearby_df.loc[nearby_df.index == prev_idx, "Fault Context"] = "Closest tower"
    else:
        nearby_df.loc[nearby_df.index == prev_idx, "Fault Context"] = "Before fault span"
        nearby_df.loc[nearby_df.index == next_idx, "Fault Context"] = "After fault span"

    helper_columns = ["Fault Context", "Distance from Fault km"]
    original_columns = [
        col
        for col in nearby_df.columns
        if col not in helper_columns and not str(col).startswith("_") and col not in ["lat", "lon"]
    ]
    return nearby_df[helper_columns + original_columns].reset_index(drop=True)


def render_tower_map(
    tower_df: pd.DataFrame,
    key_prefix: str,
    include_fault_layer: bool = True,
    default_show_fault: bool = True,
    height: int = 560,
    focus_on_fault: bool = False,
):
    map_df = prepare_tower_map_dataframe(tower_df)
    if map_df.empty:
        st.info("Latitude/Longitude tersedia tetapi belum dapat dibaca sebagai koordinat numerik.")
        return

    fault_options = get_fault_location_map_options()
    selected_fault_option = None
    show_fault_location = False

    with st.expander("Map Settings", expanded=not focus_on_fault):
        map_opt_col1, map_opt_col2, map_opt_col3, map_opt_col4 = st.columns([1.2, 1, 1, 1.2])
        with map_opt_col1:
            tower_map_style = st.selectbox(
                "Map style",
                ["satellite", "street"],
                index=0,
                format_func=lambda value: {"satellite": "Satelit", "street": "Street map"}[value],
                key=f"{key_prefix}_map_style",
            )
        with map_opt_col2:
            tower_marker_size = st.slider(
                "Ukuran marker tower",
                min_value=2,
                max_value=10,
                value=10,
                step=1,
                key=f"{key_prefix}_marker_size",
            )
        with map_opt_col3:
            show_tower_path = st.checkbox("Tampilkan jalur", value=True, key=f"{key_prefix}_show_path")
            show_tower_labels = st.checkbox("Tampilkan label tower", value=True, key=f"{key_prefix}_show_tower_labels")
        with map_opt_col4:
            if include_fault_layer and fault_options:
                show_fault_location = st.checkbox(
                    "Tampilkan fault",
                    value=default_show_fault,
                    key=f"{key_prefix}_show_fault",
                )
                option_keys = [option["key"] for option in fault_options]
                default_key = "de" if "de" in option_keys else option_keys[0]
                selected_fault_key = st.selectbox(
                    "Sumber fault",
                    option_keys,
                    index=option_keys.index(st.session_state.get(f"{key_prefix}_fault_source", default_key))
                    if st.session_state.get(f"{key_prefix}_fault_source", default_key) in option_keys
                    else option_keys.index(default_key),
                    format_func=lambda value: next(option["label"] for option in fault_options if option["key"] == value),
                    key=f"{key_prefix}_fault_source",
                )
                selected_fault_option = next(option for option in fault_options if option["key"] == selected_fault_key)

    lat_min, lat_max = map_df["lat"].min(), map_df["lat"].max()
    lon_min, lon_max = map_df["lon"].min(), map_df["lon"].max()
    coord_span = max(float(lat_max - lat_min), float(lon_max - lon_min), 0.001)
    if coord_span < 0.02:
        map_zoom = 13
    elif coord_span < 0.08:
        map_zoom = 11
    elif coord_span < 0.2:
        map_zoom = 10
    elif coord_span < 0.6:
        map_zoom = 8
    else:
        map_zoom = 6

    tower_map = folium.Map(
        location=[float(map_df["lat"].mean()), float(map_df["lon"].mean())],
        zoom_start=map_zoom,
        tiles=None,
        control_scale=True,
    )
    folium.TileLayer(
        tiles=("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"),
        attr="Esri World Imagery",
        name="Satelit",
        overlay=False,
        control=True,
        show=tower_map_style == "satellite",
    ).add_to(tower_map)
    folium.TileLayer("OpenStreetMap", name="Street map", overlay=False, control=True, show=tower_map_style == "street").add_to(tower_map)

    tower_points = list(zip(map_df["lat"].astype(float), map_df["lon"].astype(float)))
    if show_tower_path:
        folium.PolyLine(locations=tower_points, color="#2563eb", weight=3, opacity=0.85, tooltip="Tower path").add_to(tower_map)

    tower_group = folium.FeatureGroup(name="Tower", show=True)
    for _, tower_row in map_df.iterrows():
        span_value = tower_row.get("SPAN", "-")
        cumulative_value = map_display_value(tower_row.get("KUMULATIF km", tower_row.get("KUMULATIF", "-")), decimals=2, suffix=" km")
        jarak_value = map_display_value(tower_row.get("JARAK km", tower_row.get("JARAK", "-")), decimals=2, suffix=" km")
        latitude_value = tower_row.get("LATITUDE", "-")
        longitude_value = tower_row.get("LONGITUDE", "-")
        tower_lat = float(tower_row["lat"])
        tower_lon = float(tower_row["lon"])
        ultg_value = tower_row.get("ULTG", "-")
        segment_value = tower_row.get("SEGMENT", "-")
        type_string_value = tower_row.get("TYPE STRING", "-")
        jumlah_string_value = tower_row.get("JUMLAH STRING", "-")
        tooltip_html = map_detail_table(
            [
                ("Jarak", jarak_value),
                ("Kumulatif", cumulative_value),
                ("Segment", segment_value),
                ("ULTG", ultg_value),
                ("Type", type_string_value),
                ("String", jumlah_string_value),
            ],
            title=span_value,
        )
        popup_html = map_detail_table(
            [
                ("SPAN", span_value),
                ("JARAK", jarak_value),
                ("KUMULATIF", cumulative_value),
                ("ULTG", ultg_value),
                ("SEGMENT", segment_value),
                ("TYPE STRING", type_string_value),
                ("JUMLAH STRING", jumlah_string_value),
                ("LATITUDE", latitude_value),
                ("LONGITUDE", longitude_value),
                ("Maps", google_maps_action_links(tower_lat, tower_lon, span_value)),
            ],
            compact=False,
        )
        folium.CircleMarker(
            location=[tower_lat, tower_lon],
            radius=tower_marker_size,
            color="#0f172a",
            weight=1,
            fill=True,
            fill_color="#f97316",
            fill_opacity=0.9,
            tooltip=folium.Tooltip(tooltip_html, sticky=True),
            popup=folium.Popup(popup_html, max_width=520),
        ).add_to(tower_group)
        if show_tower_labels:
            tower_label = compact_tower_span_label(span_value)
            tower_label_html = (
                "<div style='"
                "background:rgba(255,255,255,0.84);"
                "border:1px solid rgba(15,23,42,0.28);"
                "border-radius:4px;"
                "padding:1px 4px;"
                "font-size:10px;"
                "font-weight:700;"
                "color:#0f172a;"
                "white-space:nowrap;"
                "box-shadow:0 1px 2px rgba(15,23,42,0.18);"
                "'>"
                f"{tower_label}"
                "</div>"
            )
            folium.Marker(
                location=[tower_lat, tower_lon],
                icon=folium.DivIcon(
                    icon_size=(54, 16),
                    icon_anchor=(-8, 8),
                    html=tower_label_html,
                ),
            ).add_to(tower_group)
    tower_group.add_to(tower_map)

    fault_location = None
    fault_warning = None
    if include_fault_layer and selected_fault_option and show_fault_location:
        fault_location, fault_warning = interpolate_tower_path_location(map_df, selected_fault_option["distance_km"])
        if fault_location:
            fault_lat, fault_lon, plotted_distance = fault_location
            fault_segment = get_fault_tower_segment(map_df, selected_fault_option["distance_km"])
            fault_rows = [
                ("Sumber", selected_fault_option["label"]),
                ("Distance", map_display_value(selected_fault_option["distance_km"], decimals=3, suffix=" km")),
                ("Plotted", map_display_value(plotted_distance, decimals=3, suffix=" km")),
            ]
            if fault_segment:
                prev_span = fault_segment["prev"].get("SPAN", "Tower A")
                next_span = fault_segment["next"].get("SPAN", "Tower B")
                prev_cum = float(fault_segment["prev"].get("_cum_km", 0.0))
                next_cum = float(fault_segment["next"].get("_cum_km", 0.0))
                from_tower_a_km = selected_fault_option["distance_km"] - prev_cum
                to_tower_b_km = next_cum - selected_fault_option["distance_km"]
                if abs(from_tower_a_km) <= abs(to_tower_b_km):
                    nearest_tower_row = fault_segment["prev"]
                    nearest_tower_span = prev_span
                    nearest_tower_distance = from_tower_a_km
                else:
                    nearest_tower_row = fault_segment["next"]
                    nearest_tower_span = next_span
                    nearest_tower_distance = to_tower_b_km
                fault_rows.extend(
                    [
                        ("Between", f"{prev_span} - {next_span}"),
                        ("From tower A", map_display_value(from_tower_a_km, decimals=3, suffix=" km")),
                        ("To tower B", map_display_value(to_tower_b_km, decimals=3, suffix=" km")),
                        ("Nearest tower", nearest_tower_span),
                        ("Nearest dist", map_display_value(nearest_tower_distance, decimals=3, suffix=" km")),
                        ("Span length", map_display_value(fault_segment["span_distance_km"], decimals=3, suffix=" km")),
                        ("Span ratio", map_display_value(fault_segment["ratio"] * 100.0, decimals=1, suffix=" %")),
                    ]
                )
            if selected_fault_option.get("quality") is not None:
                fault_rows.append(("Quality", f"{selected_fault_option['quality']}/10"))
            if selected_fault_option.get("status"):
                fault_rows.append(("Status", selected_fault_option["status"]))
            fault_rows.extend(
                [
                    ("LATITUDE", map_display_value(fault_lat, decimals=7)),
                    ("LONGITUDE", map_display_value(fault_lon, decimals=7)),
                    ("Maps", google_maps_action_links(fault_lat, fault_lon, f"Fault {selected_fault_option['label']}")),
                ]
            )
            if fault_segment:
                fault_rows.append(
                    (
                        "Nearest Maps",
                        google_maps_action_links(
                            float(nearest_tower_row["lat"]),
                            float(nearest_tower_row["lon"]),
                            str(nearest_tower_span),
                        ),
                    )
                )
            fault_popup = map_detail_table(fault_rows, title="Fault Location", compact=False)
            fault_group = folium.FeatureGroup(name="Fault Location", show=True)
            if fault_segment:
                prev_row = fault_segment["prev"]
                next_row = fault_segment["next"]
                folium.PolyLine(
                    locations=[
                        [float(prev_row["lat"]), float(prev_row["lon"])],
                        [float(next_row["lat"]), float(next_row["lon"])],
                    ],
                    color="#dc2626",
                    weight=6,
                    opacity=0.9,
                    tooltip="Fault span between two towers",
                ).add_to(fault_group)
            fault_crosshair_html = (
                "<div style='"
                "width:22px;height:22px;"
                "position:relative;"
                "'>"
                "<div style='position:absolute;left:10px;top:0;width:2px;height:22px;background:#dc2626;'></div>"
                "<div style='position:absolute;left:0;top:10px;width:22px;height:2px;background:#dc2626;'></div>"
                "<div style='position:absolute;left:6px;top:6px;width:10px;height:10px;"
                "border:2px solid #ffffff;background:#dc2626;border-radius:50%;"
                "box-shadow:0 0 0 2px #dc2626;'></div>"
                "</div>"
            )
            folium.Marker(
                location=[fault_lat, fault_lon],
                icon=folium.DivIcon(
                    icon_size=(22, 22),
                    icon_anchor=(11, 11),
                    html=fault_crosshair_html,
                ),
                tooltip=folium.Tooltip(f"Exact Fault Point - {selected_fault_option['label']}", sticky=True),
                popup=folium.Popup(fault_popup, max_width=560),
            ).add_to(fault_group)
            fault_label_anchor, fault_label_direction = fault_label_anchor_from_segment(fault_segment)
            pointer_style = {
                "left": "left:-10px;top:18px;border-top:7px solid transparent;border-bottom:7px solid transparent;border-right:10px solid #dc2626;",
                "right": "right:-10px;top:18px;border-top:7px solid transparent;border-bottom:7px solid transparent;border-left:10px solid #dc2626;",
                "above": "left:18px;bottom:-10px;border-left:7px solid transparent;border-right:7px solid transparent;border-top:10px solid #dc2626;",
                "below": "left:18px;top:-10px;border-left:7px solid transparent;border-right:7px solid transparent;border-bottom:10px solid #dc2626;",
            }.get(fault_label_direction, "")
            fault_label_html = (
                "<div style='"
                "position:relative;"
                "background:rgba(255,255,255,0.92);"
                "border:1px solid #dc2626;"
                "border-radius:6px;"
                "box-shadow:0 1px 4px rgba(15,23,42,0.25);"
                "padding:5px 7px;"
                "font-size:12px;"
                "line-height:1.25;"
                "color:#111827;"
                "white-space:nowrap;"
                "'>"
                "<div style='font-weight:700;color:#b91c1c;'>Fault Location</div>"
                f"<div>{selected_fault_option['label']}</div>"
                f"<div>{map_display_value(selected_fault_option['distance_km'], decimals=3, suffix=' km')}</div>"
                + (
                    f"<div style='color:#475569;'>{map_display_value(fault_segment['ratio'] * 100.0, decimals=1, suffix=' %')} span</div>"
                    if fault_segment
                    else ""
                )
                + f"<div style='position:absolute;width:0;height:0;{pointer_style}'></div>"
                + "</div>"
            )
            folium.Marker(
                location=[fault_lat, fault_lon],
                icon=folium.DivIcon(
                    icon_size=(210, 70),
                    icon_anchor=fault_label_anchor,
                    html=fault_label_html,
                ),
            ).add_to(fault_group)
            fault_group.add_to(tower_map)
        if fault_warning:
            st.warning(fault_warning)

    folium_center_override = None
    folium_zoom_override = None
    if focus_on_fault and fault_location and selected_fault_option:
        fault_lat, fault_lon, _ = fault_location
        focus_segment = get_fault_tower_segment(map_df, selected_fault_option["distance_km"])
        if focus_segment:
            focus_lat_values = [
                float(focus_segment["prev"]["lat"]),
                float(focus_segment["next"]["lat"]),
                fault_lat,
            ]
            focus_lon_values = [
                float(focus_segment["prev"]["lon"]),
                float(focus_segment["next"]["lon"]),
                fault_lon,
            ]
            focus_lat_min = min(focus_lat_values)
            focus_lat_max = max(focus_lat_values)
            focus_lon_min = min(focus_lon_values)
            focus_lon_max = max(focus_lon_values)
        else:
            focus_lat_min = focus_lat_max = fault_lat
            focus_lon_min = focus_lon_max = fault_lon

        lat_pad = max((focus_lat_max - focus_lat_min) * 0.45, 0.0008)
        lon_pad = max((focus_lon_max - focus_lon_min) * 0.45, 0.0008)
        tower_map.fit_bounds(
            [
                [focus_lat_min - lat_pad, focus_lon_min - lon_pad],
                [focus_lat_max + lat_pad, focus_lon_max + lon_pad],
            ]
        )
        folium_center_override = (float(fault_lat), float(fault_lon))
        focus_span = max(
            float((focus_lat_max - focus_lat_min) + (2 * lat_pad)),
            float((focus_lon_max - focus_lon_min) + (2 * lon_pad)),
            0.001,
        )
        if focus_span < 0.01:
            folium_zoom_override = 15
        elif focus_span < 0.03:
            folium_zoom_override = 14
        elif focus_span < 0.08:
            folium_zoom_override = 13
        elif focus_span < 0.2:
            folium_zoom_override = 11
        elif focus_span < 0.6:
            folium_zoom_override = 9
        else:
            folium_zoom_override = 7
    else:
        if fault_location:
            fault_lat, fault_lon, _ = fault_location
            lat_min = min(float(lat_min), fault_lat)
            lat_max = max(float(lat_max), fault_lat)
            lon_min = min(float(lon_min), fault_lon)
            lon_max = max(float(lon_max), fault_lon)
        tower_map.fit_bounds([[float(lat_min), float(lon_min)], [float(lat_max), float(lon_max)]])
    folium_key_parts = [key_prefix, "folium"]
    if selected_fault_option:
        folium_key_parts.append(selected_fault_option["key"])
        folium_key_parts.append(f"{float(selected_fault_option['distance_km']):.3f}")
    st_folium(
        tower_map,
        key="_".join(folium_key_parts).replace(".", "_"),
        height=height,
        use_container_width=True,
        returned_objects=[],
        center=folium_center_override,
        zoom=folium_zoom_override,
    )

    if include_fault_layer and selected_fault_option and show_fault_location and fault_location:
        nearby_tower_df = build_nearby_fault_tower_table(
            map_df,
            selected_fault_option["distance_km"],
            window=5,
        )
        if not nearby_tower_df.empty:
            with st.expander("Data tower sekitar titik gangguan (-5 / +5)", expanded=focus_on_fault):
                st.caption(
                    "Tabel ini menampilkan 5 tower sebelum dan 5 tower sesudah span lokasi gangguan "
                    "berdasarkan urutan KUMULATIF km. Semua kolom yang tersedia dari spreadsheet tetap ditampilkan."
                )
                nearby_formatters = {
                    "Distance from Fault km": "{:.3f}",
                    "JARAK km": "{:.2f}",
                    "KUMULATIF km": "{:.2f}",
                }
                nearby_formatters = {
                    key: formatter
                    for key, formatter in nearby_formatters.items()
                    if key in nearby_tower_df.columns
                }
                if nearby_formatters:
                    st.dataframe(
                        nearby_tower_df.style.format(nearby_formatters, na_rep="-"),
                        use_container_width=True,
                        height=360,
                    )
                else:
                    st.dataframe(nearby_tower_df, use_container_width=True, height=360)


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

def calculate_locus_loop_impedance(phasors: dict, loop_name: str, k0: complex):
    loop_name = str(loop_name or "").upper()
    va = phasors["Va"]["complex"]
    vb = phasors["Vb"]["complex"]
    vc = phasors["Vc"]["complex"]
    ia = phasors["Ia"]["complex"]
    ib = phasors["Ib"]["complex"]
    ic = phasors["Ic"]["complex"]
    i0 = phasors.get("I0", {"complex": (ia + ib + ic) / 3.0})["complex"]

    if loop_name == "AG":
        num, den = va, ia + k0 * i0
    elif loop_name == "BG":
        num, den = vb, ib + k0 * i0
    elif loop_name == "CG":
        num, den = vc, ic + k0 * i0
    elif loop_name == "AB":
        num, den = va - vb, ia - ib
    elif loop_name == "BC":
        num, den = vb - vc, ib - ic
    elif loop_name == "CA":
        num, den = vc - va, ic - ia
    else:
        raise ValueError(f"Loop R-X tidak dikenali: {loop_name}")

    if abs(den) < 1e-9:
        raise ZeroDivisionError("Loop current terlalu kecil.")
    return num / den


def normalize_locus_fault_type(fault_type: str):
    text = str(fault_type or "").upper().replace("-", "").replace(" ", "")
    aliases = {
        "A-G": "AG",
        "B-G": "BG",
        "C-G": "CG",
        "A-B": "AB",
        "B-C": "BC",
        "C-A": "CA",
    }
    text = aliases.get(text, text)
    return text if text in ["AG", "BG", "CG", "AB", "BC", "CA", "ABG", "BCG", "CAG", "ABC", "ABCG"] else "AG"


def parse_distance_setting_number(value):
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return None if pd.isna(value) else float(value)
    text = str(value).strip()
    if not text or text.lower() in ["nan", "none", "null", "n/a", "na"]:
        return None
    match = re.search(r"[-+]?\d+(?:[.,]\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", "."))
    except ValueError:
        return None


def normalize_distance_setting_column(name: str):
    text = str(name or "").lower().strip()
    text = re.sub(r"\s+", "", text)
    for old in ["(", ")", "-", "_", "/", "\\", ".", "ohm", "Ω"]:
        text = text.replace(old, "")
    return text


def find_distance_setting_column(df: pd.DataFrame, candidates: list[str]):
    normalized = {normalize_distance_setting_column(col): col for col in df.columns}
    for candidate in candidates:
        key = normalize_distance_setting_column(candidate)
        if key in normalized:
            return normalized[key]
    for key, col in normalized.items():
        for candidate in candidates:
            candidate_key = normalize_distance_setting_column(candidate)
            if candidate_key and candidate_key in key:
                return col
    return None


def detect_locus_distance_setting_columns(df: pd.DataFrame):
    return {
        "substation": find_distance_setting_column(df, ["Substation", "GI"]),
        "bay": find_distance_setting_column(df, ["Bay"]),
        "line": find_distance_setting_column(df, ["Line"]),
        "merk": find_distance_setting_column(df, ["Merk", "Brand"]),
        "type": find_distance_setting_column(df, ["Type"]),
        "z1_x": find_distance_setting_column(df, ["Z1/X1 (ohm)", "Z1/X1", "X1"]),
        "z2_x": find_distance_setting_column(df, ["Z2/X2 (ohm)", "Z2/X2", "X2"]),
        "z3_x": find_distance_setting_column(df, ["Z3/X3 (ohm)", "Z3/X3", "X3"]),
        "z1_res_phi": find_distance_setting_column(df, ["Z1 Res Phi (ohm)", "Z1 Res Phi"]),
        "z1_res_gnd": find_distance_setting_column(df, ["Z1 Res Gnd (ohm)", "Z1 Res Gnd"]),
        "z2_res_phi": find_distance_setting_column(df, ["Z2 Res Phi (ohm)", "Z2 Res Phi"]),
        "z2_res_gnd": find_distance_setting_column(df, ["Z2 Res Gnd (ohm)", "Z2 Res Gnd"]),
        "z3_res_phi": find_distance_setting_column(df, ["Z3 Res Phi (ohm)", "Z3 Res Phi"]),
        "z3_res_gnd": find_distance_setting_column(df, ["Z3 Res Gnd (ohm)", "Z3 Res Gnd"]),
    }


def sorted_nonempty_values(df: pd.DataFrame, column_name: str | None):
    if not column_name or column_name not in df.columns:
        return []
    values = []
    for value in df[column_name].dropna().tolist():
        text = str(value).strip()
        if text and text.lower() not in ["nan", "none", "null", "n/a", "na"]:
            values.append(text)
    return sorted(set(values), key=lambda item: item.upper())


def build_locus_setting_row_labels(df: pd.DataFrame, columns: dict):
    labels = []
    seen = {}
    for idx, row in df.iterrows():
        parts = []
        for key in ["substation", "bay", "line", "merk", "type"]:
            col = columns.get(key)
            if col and col in row and str(row[col]).strip() not in ["", "nan", "None"]:
                parts.append(str(row[col]).strip())
        label = " | ".join(parts) if parts else f"Row {idx + 1}"
        seen[label] = seen.get(label, 0) + 1
        labels.append(label if seen[label] == 1 else f"{label} #{seen[label]}")
    return labels


def extract_locus_zone_settings(row, columns: dict, loop_name: str):
    is_ground = str(loop_name or "").upper() in ["AG", "BG", "CG"]
    zones = []
    for zone in [1, 2, 3]:
        x_col = columns.get(f"z{zone}_x")
        r_col = columns.get(f"z{zone}_res_gnd" if is_ground else f"z{zone}_res_phi")
        r_fallback_col = columns.get(f"z{zone}_res_phi" if is_ground else f"z{zone}_res_gnd")
        x_reach = parse_distance_setting_number(row.get(x_col)) if x_col else None
        r_reach = parse_distance_setting_number(row.get(r_col)) if r_col else None
        if r_reach is None and r_fallback_col:
            r_reach = parse_distance_setting_number(row.get(r_fallback_col))
        if x_reach is None or r_reach is None or x_reach <= 0 or r_reach <= 0:
            continue
        zones.append({"zone": f"Z{zone}", "x_reach_ohm": x_reach, "r_reach_ohm": r_reach})
    return zones


def impedance_secondary_scale_from_transformer(transformer_data: dict | None):
    if not transformer_data:
        return None
    ct_primary = parse_distance_setting_number(transformer_data.get("ct_primary"))
    ct_secondary = parse_distance_setting_number(transformer_data.get("ct_secondary"))
    vt_primary = parse_distance_setting_number(transformer_data.get("vt_primary"))
    vt_secondary = parse_distance_setting_number(transformer_data.get("vt_secondary"))
    if not all([ct_primary, ct_secondary, vt_primary, vt_secondary]):
        return None
    vtr = vt_primary / vt_secondary
    ctr = ct_primary / ct_secondary
    if abs(vtr) < 1e-9:
        return None
    return ctr / vtr


def scale_locus_zone_settings(zones: list[dict], scale: float):
    scaled = []
    for zone in zones:
        item = dict(zone)
        item["x_reach_ohm"] = float(item["x_reach_ohm"]) * scale
        item["r_reach_ohm"] = float(item["r_reach_ohm"]) * scale
        scaled.append(item)
    return scaled


def add_locus_zone_overlay(fig, zones: list[dict], line_angle_deg: float):
    colors = {
        "Z1": ("rgba(34,197,94,0.20)", "#16a34a"),
        "Z2": ("rgba(59,130,246,0.16)", "#2563eb"),
        "Z3": ("rgba(245,158,11,0.14)", "#d97706"),
    }
    tan_angle = math.tan(math.radians(line_angle_deg)) if abs(math.tan(math.radians(line_angle_deg))) > 1e-9 else None
    for zone in zones:
        name = zone["zone"]
        r = float(zone["r_reach_ohm"])
        x = float(zone["x_reach_ohm"])
        center_r = x / tan_angle if tan_angle else 0.0
        reverse_r = r * 0.35
        lower_x = x * -0.15
        xs = [0.0, r, center_r + r, center_r - reverse_r, -reverse_r, 0.0]
        ys = [0.0, lower_x, x, x, x * 0.35, 0.0]
        fill_color, line_color = colors.get(name, ("rgba(100,116,139,0.12)", "#64748b"))
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                fill="toself",
                name=f"{name} locus",
                line=dict(color=line_color, width=1.5),
                fillcolor=fill_color,
                hovertemplate=f"{name}<br>R reach {r:.3f}<br>X reach {x:.3f}<extra></extra>",
            )
        )
    return fig


def build_simple_rx_locus_trajectory(
    assigned_df: pd.DataFrame,
    fault_window: dict,
    samples_per_cycle: int,
    line_param: dict,
    loop_name: str,
    pre_cycles: float,
    post_cycles: float,
    step_samples: int,
):
    rows = []
    fault_index = int(fault_window["fault_index"])
    start_index = max(int(samples_per_cycle), fault_index - int(round(pre_cycles * samples_per_cycle)))
    end_index = min(len(assigned_df) - 1, fault_index + int(round(post_cycles * samples_per_cycle)))
    step_samples = max(1, int(step_samples))

    for cursor_index in range(start_index, end_index + 1, step_samples):
        try:
            ph = calculate_all_phasors(
                df=assigned_df,
                cursor_index=cursor_index,
                samples_per_cycle=int(samples_per_cycle),
            )
            ph = add_sequence_components_to_phasor_dict(ph)
            z_loop = calculate_locus_loop_impedance(ph, loop_name, line_param["K0"])
            if not np.isfinite(z_loop.real) or not np.isfinite(z_loop.imag):
                continue
            rows.append(
                {
                    "cursor_index": cursor_index,
                    "time_s": float(assigned_df["time"].iloc[cursor_index]),
                    "R_ohm": z_loop.real,
                    "X_ohm": z_loop.imag,
                    "Z_mag_ohm": abs(z_loop),
                    "Z_angle_deg": math.degrees(cmath.phase(z_loop)),
                }
            )
        except Exception:
            continue

    return pd.DataFrame(rows)


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

    default_database_spreadsheet_url = (
        "https://docs.google.com/spreadsheets/d/"
        "<DATABASE_SPREADSHEET_ID>/edit?usp=sharing"
    )
    old_line_spreadsheet_url = (
        "https://docs.google.com/spreadsheets/d/"
        "<OLD_LINE_SPREADSHEET_ID>/edit?usp=sharing"
    )
    old_cable_spreadsheet_url = (
        "https://docs.google.com/spreadsheets/d/"
        "<OLD_CABLE_SPREADSHEET_ID>/edit?usp=sharing"
    )

    existing_database_url = (
        st.session_state.get("database_spreadsheet_url")
        or st.session_state.get("line_data_spreadsheet_url")
        or st.session_state.get("cable_data_spreadsheet_url")
        or default_database_spreadsheet_url
    )
    if existing_database_url in [old_line_spreadsheet_url, old_cable_spreadsheet_url]:
        existing_database_url = default_database_spreadsheet_url

    st.session_state["database_spreadsheet_url"] = existing_database_url
    st.session_state["line_data_spreadsheet_url"] = existing_database_url
    st.session_state["cable_data_spreadsheet_url"] = existing_database_url

    if "line_data_sheet_name" not in st.session_state:
        st.session_state["line_data_sheet_name"] = "line_impedance"

    if "cable_data_sheet_name" not in st.session_state:
        st.session_state["cable_data_sheet_name"] = "cable_impedance"

    st.caption(
        "Spreadsheet harus dapat diakses publik atau minimal dapat dibaca melalui link. "
        "Aplikasi membaca data memakai endpoint CSV Google Sheets."
    )

    database_spreadsheet_url = st.text_input(
        "Database Spreadsheet URL",
        value=st.session_state.get("database_spreadsheet_url", default_database_spreadsheet_url),
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
            value=st.session_state.get("tower_schedule_url", DEFAULT_TOWER_SCHEDULE_URL),
            key="tower_schedule_url_setup_input",
        ).strip()
    with tower_db_col2:
        tower_schedule_sheet_setup = st.text_input(
            "Tower Schedule Sheet",
            value=st.session_state.get("tower_schedule_sheet_name", DEFAULT_TOWER_SCHEDULE_SHEET),
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

    st.session_state["tower_schedule_url"] = tower_schedule_url_setup or DEFAULT_TOWER_SCHEDULE_URL
    st.session_state["tower_schedule_sheet_name"] = tower_schedule_sheet_setup or DEFAULT_TOWER_SCHEDULE_SHEET

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
            value=st.session_state.get("case_drive_folder_url", DEFAULT_CASE_DRIVE_FOLDER_URL),
            key="case_drive_folder_url_input",
        ).strip()
        st.session_state["case_drive_folder_url"] = drive_folder_input or DEFAULT_CASE_DRIVE_FOLDER_URL
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
                    st.session_state.get("case_drive_folder_id", DEFAULT_CASE_DRIVE_FOLDER_ID),
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
        "Folder default: "
        f"`{st.session_state.get('case_drive_folder_id', DEFAULT_CASE_DRIVE_FOLDER_ID)}`. "
        "Untuk mode Drive, gunakan service account melalui `st.secrets['gdrive_service_account']` "
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
            if not tower_load_all and not tower_pre_ultg and not tower_pre_segment:
                st.warning("Isi ULTG atau Segment terlebih dahulu, atau centang Load semua data.")
            else:
                read_google_spreadsheet_query_cached.clear()
                st.session_state["tower_schedule_loaded"] = True
                tower_load_requested = True

    if not st.session_state.get("tower_schedule_loaded") and "tower_schedule_df" not in st.session_state:
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
            help="Fasa dianggap drop jika Vphase <= threshold × Vmax."
        )

    with col_ft2:
        current_rise_threshold_ft = st.number_input(
            "Current Rise Threshold",
            value=1.50,
            min_value=1.05,
            max_value=10.00,
            step=0.0001,
            format="%.5f",
            help="Fasa dianggap faulted jika Iphase >= threshold × Imin."
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

        if line_parameter_source in ["Database Excel Line Data", "Database Excel Cable Data"]:
            use_cable_database = line_parameter_source == "Database Excel Cable Data"
            database_source_key = "cable_data" if use_cable_database else "line_data"
            database_spreadsheet_url = st.session_state.get(
                f"{database_source_key}_spreadsheet_url",
                "https://docs.google.com/spreadsheets/d/"
                "<DATABASE_SPREADSHEET_ID>/edit?usp=sharing",
            )
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
        col_d.metric("Rf Estimate", f'{hr_result["Rf_est_ohm"]:.3f} Ω')
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
    col_r4.metric("Zapp", f'{single_result["Zapp_R"]:.3f} + j{single_result["Zapp_X"]:.3f} Ω')
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
            col_twsr4.metric("Δt Local-Remote", f'{tws_result["delta_t_s"] * 1e6:.3f} µs')

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
