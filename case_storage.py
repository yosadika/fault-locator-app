import io
import json
import os
import re
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

try:
    import tomllib
except ModuleNotFoundError:
    tomllib = None


DEFAULT_CASE_DRIVE_FOLDER_URL = ""
DEFAULT_CASE_DRIVE_FOLDER_ID = ""


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
    "runtime_credentials",
    "runtime_credentials_loaded_name",
    "runtime_gdrive_service_account",
    "openweather_lightning_api_key",
    "xweather_client_id",
    "xweather_client_secret",
    "accuweather_api_key",
}


def _nested_config_value(config: dict, *paths, default=""):
    if not isinstance(config, dict):
        return default
    for path in paths:
        node = config
        for key in path:
            if not isinstance(node, dict) or key not in node:
                node = None
                break
            node = node[key]
        if node not in [None, ""]:
            return node
    return default


def get_config_secret(name: str, default: str = ""):
    try:
        value = st.secrets.get(name)
        if value not in [None, ""]:
            return str(value).strip()
    except Exception:
        pass
    return str(os.environ.get(name, default) or "").strip()


def parse_runtime_credentials_upload(uploaded_file):
    if uploaded_file is None:
        return None, None
    name = str(getattr(uploaded_file, "name", "") or "credentials").strip()
    content = uploaded_file.getvalue()
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        return None, "File credentials harus berupa teks UTF-8 JSON/TOML."

    suffix = name.lower().rsplit(".", 1)[-1] if "." in name else ""
    try:
        if suffix == "json":
            payload = json.loads(text)
        elif suffix == "toml":
            if tomllib is None:
                return None, "Runtime Python ini belum mendukung TOML. Gunakan credentials.json."
            payload = tomllib.loads(text)
        else:
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                if tomllib is None:
                    raise
                payload = tomllib.loads(text)
    except Exception as exc:
        return None, f"Gagal membaca credentials: {exc}"

    if not isinstance(payload, dict):
        return None, "Isi credentials harus berupa object/dictionary."
    return payload, None


def apply_runtime_credentials(payload: dict):
    if not isinstance(payload, dict):
        return []

    applied = []
    database_url = _nested_config_value(
        payload,
        ("spreadsheet", "database_url"),
        ("spreadsheet", "database_spreadsheet_url"),
        ("spreadsheets", "database_url"),
        ("database", "spreadsheet_url"),
        ("database_spreadsheet_url",),
    )
    if database_url:
        database_url = str(database_url).strip()
        st.session_state["database_spreadsheet_url"] = database_url
        st.session_state["line_data_spreadsheet_url"] = database_url
        st.session_state["cable_data_spreadsheet_url"] = database_url
        st.session_state["database_spreadsheet_url_input"] = database_url
        applied.append("Database Spreadsheet URL")

    mapping = [
        (
            "line_data_sheet_name",
            "Line Data Sheet",
            ("spreadsheet", "database_line_sheet"),
            ("spreadsheet", "line_data_sheet"),
            ("database", "line_sheet"),
            ("line_data_sheet_name",),
        ),
        (
            "cable_data_sheet_name",
            "Cable Data Sheet",
            ("spreadsheet", "database_cable_sheet"),
            ("spreadsheet", "cable_data_sheet"),
            ("database", "cable_sheet"),
            ("cable_data_sheet_name",),
        ),
        (
            "distance_settings_sheet_name",
            "Distance Settings Sheet",
            ("spreadsheet", "database_distance_sheet"),
            ("spreadsheet", "distance_settings_sheet"),
            ("database", "distance_sheet"),
            ("distance_settings_sheet_name",),
        ),
        (
            "tower_schedule_url",
            "Tower Schedule URL",
            ("spreadsheet", "tower_schedule_url"),
            ("spreadsheet", "tower_schedule_spreadsheet_url"),
            ("spreadsheets", "tower_schedule_url"),
            ("tower_schedule", "spreadsheet_url"),
            ("tower_schedule_url",),
        ),
        (
            "tower_schedule_sheet_name",
            "Tower Schedule Sheet",
            ("spreadsheet", "tower_schedule_sheet"),
            ("tower_schedule", "sheet"),
            ("tower_schedule_sheet_name",),
        ),
        (
            "case_drive_folder_url",
            "Case Drive Folder URL",
            ("case_storage", "drive_folder_url"),
            ("drive", "folder_url"),
            ("case_drive_folder_url",),
        ),
    ]
    for state_key, label, *paths in mapping:
        value = _nested_config_value(payload, *paths)
        if value:
            value = str(value).strip()
            st.session_state[state_key] = value
            widget_key = {
                "line_data_sheet_name": "line_data_sheet_name_manual",
                "cable_data_sheet_name": "cable_data_sheet_name_manual",
                "distance_settings_sheet_name": "distance_settings_sheet_name_manual",
                "tower_schedule_url": "tower_schedule_url_setup_input",
                "tower_schedule_sheet_name": "tower_schedule_sheet_setup_input",
                "case_drive_folder_url": "case_drive_folder_url_input",
            }.get(state_key)
            if widget_key:
                st.session_state[widget_key] = value
            applied.append(label)

    openweather_key = _nested_config_value(
        payload,
        ("openweather", "api_key"),
        ("weather", "openweather_api_key"),
        ("OPENWEATHER_API_KEY",),
        ("openweather_api_key",),
    )
    if openweather_key:
        openweather_key = str(openweather_key).strip()
        st.session_state["openweather_lightning_api_key"] = openweather_key
        st.session_state["summary_weather_lightning_openweather_api_key_input"] = openweather_key
        applied.append("OpenWeather API key")

    service_account = _nested_config_value(
        payload,
        ("gdrive_service_account",),
        ("google_service_account",),
        ("service_account",),
        default=None,
    )
    if isinstance(service_account, dict):
        st.session_state["runtime_gdrive_service_account"] = service_account
        applied.append("Google service account")

    if st.session_state.get("case_drive_folder_url"):
        st.session_state["case_drive_folder_id"] = extract_google_drive_folder_id(st.session_state["case_drive_folder_url"])

    return applied


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
        credentials_info = st.session_state.get("runtime_gdrive_service_account") or st.secrets.get("gdrive_service_account")
    except Exception:
        credentials_info = st.session_state.get("runtime_gdrive_service_account")

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
