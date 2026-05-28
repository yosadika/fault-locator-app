import cmath
import math
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from phasor import calculate_all_phasors, add_sequence_components_to_phasor_dict


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
    for old in ["(", ")", "-", "_", "/", "\\", ".", "ohm", "Î©"]:
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
