import re

import numpy as np
import pandas as pd

from app_helpers import invert_current_phasors
from phasor import calculate_all_phasors, add_sequence_components_to_phasor_dict
from two_ended import (
    calculate_positive_sequence_two_ended,
    evaluate_two_ended_quality,
    choose_best_remote_current_direction,
    choose_best_two_ended_adaptation,
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
