import math
import cmath
import pandas as pd


def get_complex(phasors: dict, name: str) -> complex:
    return phasors[name]["complex"]


def calculate_positive_sequence_two_ended(
    local_phasors: dict,
    remote_phasors: dict,
    line_param: dict,
    remote_current_direction: str = "into_line",
):
    """
    Two-ended fault location berbasis positive sequence.

    Asumsi:
    - Local end berada pada x = 0 km.
    - Remote end berada pada x = L km.
    - Arus local mengalir masuk ke saluran dari sisi local.
    - Arus remote idealnya juga didefinisikan masuk ke saluran dari sisi remote.

    Persamaan sederhana:
    Vlocal(x) = V1L - I1L * Z1_per_km * x
    Vremote(x) = V1R - I1R * Z1_per_km * (L - x)

    Titik gangguan berada saat:
    Vlocal(x) = Vremote(x)

    Maka:
    x = (V1L - V1R + I1R * Z1_per_km * L) / (Z1_per_km * (I1L + I1R))

    Jika arah arus remote di rekaman berlawanan, pilih opsi invert_remote.
    """

    V1L = get_complex(local_phasors, "V1")
    I1L = get_complex(local_phasors, "I1")

    V1R = get_complex(remote_phasors, "V1")
    I1R = get_complex(remote_phasors, "I1")

    if remote_current_direction == "opposite_to_line":
        I1R = -I1R

    Z1_per_km = line_param["Z1_per_km"]
    L = line_param["length_km"]

    denominator = Z1_per_km * (I1L + I1R)

    if abs(denominator) < 1e-9:
        raise ZeroDivisionError("Denominator two-ended terlalu kecil. Cek arus dan arah CT.")

    distance_complex = (
        V1L - V1R + I1R * Z1_per_km * L
    ) / denominator

    distance_km = distance_complex.real
    distance_percent = distance_km / L * 100.0

    V_fault_from_local = V1L - I1L * Z1_per_km * distance_km
    V_fault_from_remote = V1R - I1R * Z1_per_km * (L - distance_km)

    mismatch = V_fault_from_local - V_fault_from_remote

    return {
        "method": "positive_sequence_two_ended",
        "distance_complex": distance_complex,
        "distance_km": distance_km,
        "distance_percent": distance_percent,
        "distance_from_remote_km": L - distance_km,
        "distance_from_remote_percent": (L - distance_km) / L * 100.0,
        "V_fault_from_local": V_fault_from_local,
        "V_fault_from_remote": V_fault_from_remote,
        "voltage_mismatch": mismatch,
        "voltage_mismatch_magnitude": abs(mismatch),
        "remote_current_direction": remote_current_direction,
    }


def evaluate_two_ended_quality(result: dict, line_param: dict):
    """
    Memberi indikator kualitas sederhana.
    """

    L = line_param["length_km"]
    d = result["distance_km"]
    imag_d = abs(result["distance_complex"].imag)

    warnings = []
    score = 10.0

    if d < 0:
        warnings.append("Jarak negatif. Kemungkinan arah arus remote/local terbalik atau gangguan di luar saluran.")
        score -= 4.0

    if d > L:
        warnings.append("Jarak melebihi panjang saluran. Cek arah CT, mapping channel, atau kemungkinan gangguan eksternal.")
        score -= 4.0

    if imag_d > 0.05 * L:
        warnings.append("Komponen imajiner hasil jarak cukup besar. Sinkronisasi atau arah arus mungkin belum tepat.")
        score -= 2.0

    if result["voltage_mismatch_magnitude"] > 0:
        # Normalisasi kasar terhadap tegangan gangguan
        vf_mag = max(
            abs(result["V_fault_from_local"]),
            abs(result["V_fault_from_remote"]),
            1.0,
        )
        mismatch_ratio = result["voltage_mismatch_magnitude"] / vf_mag

        if mismatch_ratio > 0.10:
            warnings.append("Mismatch tegangan fault dari kedua ujung cukup besar.")
            score -= 1.5
    else:
        mismatch_ratio = 0.0

    score = max(0.0, min(10.0, score))

    return {
        "quality_score": round(score, 2),
        "warnings": warnings,
        "distance_imag_km": result["distance_complex"].imag,
        "mismatch_ratio": mismatch_ratio,
    }


def build_two_ended_result_dataframe(result: dict, quality: dict):
    rows = [
        {"Parameter": "Distance from Local End (km)", "Value": result["distance_km"]},
        {"Parameter": "Distance from Local End (%)", "Value": result["distance_percent"]},
        {"Parameter": "Distance from Remote End (km)", "Value": result["distance_from_remote_km"]},
        {"Parameter": "Distance from Remote End (%)", "Value": result["distance_from_remote_percent"]},
        {"Parameter": "Distance Complex Real", "Value": result["distance_complex"].real},
        {"Parameter": "Distance Complex Imag", "Value": result["distance_complex"].imag},
        {"Parameter": "Voltage Mismatch Magnitude", "Value": result["voltage_mismatch_magnitude"]},
        {"Parameter": "Mismatch Ratio", "Value": quality["mismatch_ratio"]},
        {"Parameter": "Quality Score 0-10", "Value": quality["quality_score"]},
        {"Parameter": "Remote Current Direction", "Value": result["remote_current_direction"]},
    ]

    return pd.DataFrame(rows)


def choose_best_remote_current_direction(local_phasors, remote_phasors, line_param):
    """
    Mencoba dua kemungkinan arah arus remote:
    1. into_line
    2. opposite_to_line

    Dipilih yang hasilnya paling masuk akal:
    - distance berada di dalam saluran
    - komponen imag kecil
    - voltage mismatch kecil
    """

    candidates = []

    for direction in ["into_line", "opposite_to_line"]:
        try:
            result = calculate_positive_sequence_two_ended(
                local_phasors=local_phasors,
                remote_phasors=remote_phasors,
                line_param=line_param,
                remote_current_direction=direction,
            )

            quality = evaluate_two_ended_quality(result, line_param)

            L = line_param["length_km"]
            d = result["distance_km"]

            inside_penalty = 0 if 0 <= d <= L else 1000
            imag_penalty = abs(result["distance_complex"].imag)
            mismatch_penalty = quality["mismatch_ratio"] * L

            ranking_score = inside_penalty + imag_penalty + mismatch_penalty

            candidates.append(
                {
                    "direction": direction,
                    "result": result,
                    "quality": quality,
                    "ranking_score": ranking_score,
                }
            )

        except Exception as e:
            candidates.append(
                {
                    "direction": direction,
                    "result": None,
                    "quality": None,
                    "ranking_score": 999999,
                    "error": str(e),
                }
            )

    candidates = sorted(candidates, key=lambda x: x["ranking_score"])
    return candidates[0], candidates