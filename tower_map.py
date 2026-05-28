import math
import re
from urllib.parse import quote

import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium


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
        # Label berada di kiri fault point, jadi pointer harus keluar dari sisi
        # kanan label agar mengarah kembali ke pinpoint.
        return (226, 20), "right"
    return (-18, 20), "left"


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
