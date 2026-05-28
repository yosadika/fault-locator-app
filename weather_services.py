import math
import os
import re
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
import streamlit as st


OPENWEATHER_LIGHTNING_ENDPOINT = "https://demo.openweathermap.org/lightning/1.0/data"
OPENWEATHER_ONECALL_CURRENT_ENDPOINT = "https://api.openweathermap.org/data/4.0/onecall/current"
OPENWEATHER_ONECALL_15MIN_ENDPOINT = "https://api.openweathermap.org/data/4.0/onecall/timeline/15min"
XWEATHER_LIGHTNING_FLASH_CLOSEST_ENDPOINT = "https://data.api.xweather.com/lightning/flash/closest"
ACCUWEATHER_BASE_URL = "https://dataservice.accuweather.com"
ACCUWEATHER_LIGHTNING_BASE_URL = "https://api.accuweather.com"
LOCAL_TIMEZONE = timezone(timedelta(hours=7))


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

WEATHER_DESCRIPTION_ID = {
    "clear sky": "Cerah",
    "mainly clear": "Umumnya cerah",
    "few clouds": "Sedikit berawan",
    "scattered clouds": "Berawan sebagian",
    "broken clouds": "Berawan",
    "overcast clouds": "Mendung",
    "overcast": "Mendung",
    "partly cloudy": "Berawan sebagian",
    "clouds": "Berawan",
    "mist": "Berkabut tipis",
    "fog": "Berkabut",
    "haze": "Berkabut asap",
    "smoke": "Berasap",
    "dust": "Berdebu",
    "sand": "Berdebu pasir",
    "squalls": "Angin kencang",
    "tornado": "Tornado",
    "light rain": "Hujan ringan",
    "moderate rain": "Hujan sedang",
    "heavy intensity rain": "Hujan lebat",
    "very heavy rain": "Hujan sangat lebat",
    "extreme rain": "Hujan ekstrem",
    "freezing rain": "Hujan beku",
    "light intensity shower rain": "Hujan ringan sesaat",
    "shower rain": "Hujan sesaat",
    "heavy intensity shower rain": "Hujan sesaat lebat",
    "ragged shower rain": "Hujan sesaat tidak merata",
    "light intensity drizzle": "Gerimis ringan",
    "drizzle": "Gerimis",
    "heavy intensity drizzle": "Gerimis lebat",
    "slight rain": "Hujan ringan",
    "slight rain showers": "Hujan ringan sesaat",
    "moderate rain showers": "Hujan sedang sesaat",
    "violent rain showers": "Hujan sesaat sangat lebat",
    "light snow": "Salju ringan",
    "snow": "Salju",
    "heavy snow": "Salju lebat",
    "thunderstorm": "Hujan disertai petir",
    "thunderstorm with light rain": "Hujan ringan disertai petir",
    "thunderstorm with rain": "Hujan disertai petir",
    "thunderstorm with heavy rain": "Hujan lebat disertai petir",
}


def weather_code_label(code):
    if code is None or pd.isna(code):
        return "-"
    try:
        label = WMO_WEATHER_CODES.get(int(code), f"WMO {int(code)}")
        return translate_weather_description(label)
    except (TypeError, ValueError):
        return "-"


def translate_weather_description(value):
    text = str(value or "").strip()
    if not text or text == "-":
        return "-"
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    if normalized in WEATHER_DESCRIPTION_ID:
        return WEATHER_DESCRIPTION_ID[normalized]
    return text[:1].upper() + text[1:]


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

    key = str(st.session_state.get("summary_weather_lightning_openweather_api_key_input", "") or "").strip()
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
            "weather": translate_weather_description(weather_item.get("description") or weather_item.get("main") or "-"),
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
        description = translate_weather_description(
            weather_item.get("description") or weather_item.get("main") or weather_code_label(code)
        )
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
        desc_key = translate_weather_description(description)
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
        summary = f"Ada peluang hujan mulai sekitar {rain_slots[0].strftime('%H:%M')}."
    else:
        summary = f"Tidak ada indikasi hujan dalam {hours} jam ke depan."

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
