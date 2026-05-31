import math
import cmath
import pandas as pd


def get_complex(phasors: dict, name: str) -> complex:
    return phasors[name]["complex"]


CURRENT_NAMES = {"Ia", "Ib", "Ic", "IE", "I0", "I1", "I2"}
VOLTAGE_NAMES = {"Va", "Vb", "Vc", "V0", "V1", "V2"}


def _phasor_entry(value: complex):
    return {
        "complex": value,
        "real": value.real,
        "imag": value.imag,
        "magnitude": abs(value),
        "angle_deg": math.degrees(cmath.phase(value)),
    }


def transform_remote_phasors(
    phasors: dict,
    voltage_polarity: int = 1,
    current_polarity: int = 1,
    angle_shift_deg: float = 0.0,
):
    """
    Menerapkan koreksi umum pada fasor remote.

    Koreksi ini mewakili perbedaan referensi sudut antar rekaman, polaritas VT,
    dan polaritas CT sebelum rumus two-ended dijalankan.
    """

    angle_factor = cmath.exp(1j * math.radians(angle_shift_deg))
    transformed = {}

    for name, value in phasors.items():
        phasor_value = value["complex"] * angle_factor

        if name in VOLTAGE_NAMES:
            phasor_value *= voltage_polarity

        if name in CURRENT_NAMES:
            phasor_value *= current_polarity

        transformed[name] = _phasor_entry(phasor_value)

    return transformed


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
        "V1L": V1L,
        "I1L": I1L,
        "V1R": V1R,
        "I1R": I1R,
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

    boundary_margin = 0.002 * L

    if d < -boundary_margin:
        warnings.append("Jarak negatif. Kemungkinan arah arus remote/local terbalik atau gangguan di luar saluran.")
        score -= 4.0

    if d > L + boundary_margin:
        warnings.append("Jarak melebihi panjang saluran. Cek arah CT, mapping channel, atau kemungkinan gangguan eksternal.")
        score -= 4.0

    if imag_d > 0.05 * L:
        warnings.append("Komponen imajiner hasil jarak cukup besar. Sinkronisasi atau arah arus mungkin belum tepat.")
        score -= 2.0
    elif imag_d > 0.02 * L:
        warnings.append("Komponen imajiner hasil jarak masih terlihat. Validasi sinkronisasi rekaman.")
        score -= 0.5

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
        elif mismatch_ratio > 0.05:
            warnings.append("Mismatch tegangan fault kecil tetapi masih perlu divalidasi.")
            score -= 0.5
    else:
        mismatch_ratio = 0.0

    score = max(0.0, min(10.0, score))

    return {
        "quality_score": round(score, 2),
        "warnings": warnings,
        "distance_imag_km": result["distance_complex"].imag,
        "mismatch_ratio": mismatch_ratio,
    }


def score_two_ended_candidate(result: dict, quality: dict, line_param: dict):
    L = line_param["length_km"]
    d = result["distance_km"]

    outside_km = max(0.0, -d, d - L)
    outside_penalty = (outside_km / max(L, 1e-9)) * 1000.0
    imag_penalty = abs(result["distance_complex"].imag) / max(L, 1e-9) * 100.0
    mismatch_penalty = quality["mismatch_ratio"] * 100.0
    quality_bonus = (10.0 - quality["quality_score"]) * 10.0

    return outside_penalty + imag_penalty + mismatch_penalty + quality_bonus


def _calculate_adapted_candidate(
    local_phasors: dict,
    remote_phasors: dict,
    line_param: dict,
    remote_current_direction: str,
    voltage_polarity: int,
    current_polarity: int,
    angle_shift_deg: float,
):
    adapted_remote = transform_remote_phasors(
        remote_phasors,
        voltage_polarity=voltage_polarity,
        current_polarity=current_polarity,
        angle_shift_deg=angle_shift_deg,
    )

    result = calculate_positive_sequence_two_ended(
        local_phasors=local_phasors,
        remote_phasors=adapted_remote,
        line_param=line_param,
        remote_current_direction=remote_current_direction,
    )

    quality = evaluate_two_ended_quality(result, line_param)
    ranking_score = score_two_ended_candidate(result, quality, line_param)
    ranking_score += abs(angle_shift_deg) * 0.01

    if voltage_polarity < 0:
        ranking_score += 0.20

    if current_polarity < 0:
        ranking_score += 0.10

    if remote_current_direction == "opposite_to_line":
        ranking_score += 0.05

    metadata = {
        "direction": remote_current_direction,
        "remote_current_direction": remote_current_direction,
        "voltage_polarity": voltage_polarity,
        "current_polarity": current_polarity,
        "angle_shift_deg": angle_shift_deg,
        "adapted_remote_phasors": adapted_remote,
    }

    result.update(
        {
            "remote_voltage_polarity": voltage_polarity,
            "remote_current_polarity": current_polarity,
            "remote_angle_shift_deg": angle_shift_deg,
        }
    )

    return {
        **metadata,
        "result": result,
        "quality": quality,
        "ranking_score": ranking_score,
    }


def choose_best_two_ended_adaptation(
    local_phasors,
    remote_phasors,
    line_param,
    angle_step_deg: float = 1.0,
):
    """
    Mencari kombinasi pembacaan remote yang paling konsisten.

    Dicoba:
    - arah arus remote,
    - polaritas VT remote,
    - polaritas CT remote,
    - offset sudut umum antar rekaman.
    """

    coarse_candidates = []
    directions = ["into_line", "opposite_to_line"]
    polarities = [-1, 1]
    angle_count = max(1, int(round(360.0 / angle_step_deg)))
    angles = [
        -180.0 + i * (360.0 / angle_count)
        for i in range(angle_count)
    ]

    for direction in directions:
        for voltage_polarity in polarities:
            for current_polarity in polarities:
                for angle_shift_deg in angles:
                    try:
                        candidate = _calculate_adapted_candidate(
                            local_phasors=local_phasors,
                            remote_phasors=remote_phasors,
                            line_param=line_param,
                            remote_current_direction=direction,
                            voltage_polarity=voltage_polarity,
                            current_polarity=current_polarity,
                            angle_shift_deg=angle_shift_deg,
                        )
                        coarse_candidates.append(candidate)
                    except Exception as e:
                        coarse_candidates.append(
                            {
                                "direction": direction,
                                "remote_current_direction": direction,
                                "voltage_polarity": voltage_polarity,
                                "current_polarity": current_polarity,
                                "angle_shift_deg": angle_shift_deg,
                                "result": None,
                                "quality": None,
                                "ranking_score": 999999.0,
                                "error": str(e),
                            }
                        )

    valid_candidates = [c for c in coarse_candidates if c["result"] is not None]

    refined_candidates = []
    for coarse in sorted(valid_candidates, key=lambda x: x["ranking_score"])[:8]:
        center = coarse["angle_shift_deg"]
        fine_step = max(0.25, angle_step_deg / 10.0)
        for fine_i in range(-10, 11):
            angle_shift_deg = center + fine_i * fine_step
            if angle_shift_deg > 180.0:
                angle_shift_deg -= 360.0
            elif angle_shift_deg <= -180.0:
                angle_shift_deg += 360.0

            try:
                refined_candidates.append(
                    _calculate_adapted_candidate(
                        local_phasors=local_phasors,
                        remote_phasors=remote_phasors,
                        line_param=line_param,
                        remote_current_direction=coarse["remote_current_direction"],
                        voltage_polarity=coarse["voltage_polarity"],
                        current_polarity=coarse["current_polarity"],
                        angle_shift_deg=angle_shift_deg,
                    )
                )
            except Exception:
                pass

    candidates = sorted(
        valid_candidates + refined_candidates,
        key=lambda x: x["ranking_score"],
    )

    if not candidates:
        return coarse_candidates[0], coarse_candidates

    return candidates[0], candidates


def build_two_ended_result_dataframe(result: dict, quality: dict):
    rows = [
        {"Parameter": "Calculation Local Side", "Value": result.get("calculation_local_label", "Local End")},
        {"Parameter": "Calculation Remote Side", "Value": result.get("calculation_remote_label", "Remote End")},
        {"Parameter": "Distance from Local End (km)", "Value": result["distance_km"]},
        {"Parameter": "Distance from Local End (%)", "Value": result["distance_percent"]},
        {"Parameter": "Distance from Remote End (km)", "Value": result["distance_from_remote_km"]},
        {"Parameter": "Distance from Remote End (%)", "Value": result["distance_from_remote_percent"]},
        {"Parameter": "Distance from Original Local GI (km)", "Value": result.get("distance_from_original_local_km", result["distance_km"])},
        {"Parameter": "Distance from Original Local GI (%)", "Value": result.get("distance_from_original_local_percent", result["distance_percent"])},
        {"Parameter": "Distance Complex Real", "Value": result["distance_complex"].real},
        {"Parameter": "Distance Complex Imag", "Value": result["distance_complex"].imag},
        {"Parameter": "Voltage Mismatch Magnitude", "Value": result["voltage_mismatch_magnitude"]},
        {"Parameter": "Mismatch Ratio", "Value": quality["mismatch_ratio"]},
        {"Parameter": "Quality Score 0-10", "Value": quality["quality_score"]},
        {"Parameter": "Remote Current Direction", "Value": result["remote_current_direction"]},
        {"Parameter": "Uploaded Remote Current Direction", "Value": result.get("uploaded_remote_current_direction", result["remote_current_direction"])},
        {"Parameter": "Remote DFT Index Used", "Value": result.get("remote_dft_index_used", "-")},
        {"Parameter": "Remote Waveform Sync Reference", "Value": result.get("remote_waveform_sync_reference", "fault_cursor")},
        {"Parameter": "Remote Waveform Sync Method", "Value": result.get("remote_waveform_sync_method", "fault_cursor")},
        {"Parameter": "Remote Waveform Sync Shift (s)", "Value": result.get("remote_waveform_sync_shift_s", 0.0)},
        {"Parameter": "Remote Waveform Sync Score", "Value": result.get("remote_waveform_sync_score", 0.0)},
        {"Parameter": "Remote Voltage Polarity", "Value": result.get("remote_voltage_polarity", 1)},
        {"Parameter": "Remote Current Polarity", "Value": result.get("remote_current_polarity", 1)},
        {"Parameter": "Remote Angle Shift Deg", "Value": result.get("remote_angle_shift_deg", 0.0)},
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

            ranking_score = score_two_ended_candidate(result, quality, line_param)

            candidates.append(
                {
                    "direction": direction,
                    "remote_current_direction": direction,
                    "voltage_polarity": 1,
                    "current_polarity": 1,
                    "angle_shift_deg": 0.0,
                    "adapted_remote_phasors": remote_phasors,
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
