import math
import cmath
import pandas as pd


def get_complex(phasors: dict, name: str) -> complex:
    return phasors[name]["complex"]


def safe_divide(a: complex, b: complex, eps: float = 1e-9) -> complex:
    if abs(b) < eps:
        raise ZeroDivisionError("Denominator terlalu kecil untuk pembagian impedansi.")
    return a / b


def normalize_fault_type(fault_type: str) -> str:
    if not fault_type:
        return "UNKNOWN"

    fault_type = fault_type.upper().strip()

    if fault_type == "AC":
        return "CA"

    if fault_type == "ACG":
        return "CAG"

    return fault_type


def calculate_loop_impedance_for_fault(
    phasors: dict,
    fault_type: str,
    k0: complex,
):
    """
    Menghitung impedansi loop sesuai jenis gangguan.

    Untuk gangguan tanah:
    AG = Va / (Ia + K0*I0)
    BG = Vb / (Ib + K0*I0)
    CG = Vc / (Ic + K0*I0)

    Untuk gangguan antar fasa:
    AB = (Va - Vb) / (Ia - Ib)
    BC = (Vb - Vc) / (Ib - Ic)
    CA = (Vc - Va) / (Ic - Ia)

    Untuk ABC:
    menggunakan pendekatan Va/Ia sebagai fallback sederhana.
    """

    fault_type = normalize_fault_type(fault_type)

    Va = get_complex(phasors, "Va")
    Vb = get_complex(phasors, "Vb")
    Vc = get_complex(phasors, "Vc")

    Ia = get_complex(phasors, "Ia")
    Ib = get_complex(phasors, "Ib")
    Ic = get_complex(phasors, "Ic")

    I0 = get_complex(phasors, "I0")

    if fault_type in ["AG", "ABG"]:
        return safe_divide(Va, Ia + k0 * I0), "AG"

    if fault_type in ["BG", "BCG"]:
        return safe_divide(Vb, Ib + k0 * I0), "BG"

    if fault_type in ["CG", "CAG"]:
        return safe_divide(Vc, Ic + k0 * I0), "CG"

    if fault_type == "AB":
        return safe_divide(Va - Vb, Ia - Ib), "AB"

    if fault_type == "BC":
        return safe_divide(Vb - Vc, Ib - Ic), "BC"

    if fault_type == "CA":
        return safe_divide(Vc - Va, Ic - Ia), "CA"

    if fault_type in ["ABC", "ABCG"]:
        return safe_divide(Va, Ia), "ABC"

    # Fallback: coba semua loop dan pilih yang magnitude impedansinya paling kecil.
    candidates = {}

    candidates["AG"] = safe_divide(Va, Ia + k0 * I0)
    candidates["BG"] = safe_divide(Vb, Ib + k0 * I0)
    candidates["CG"] = safe_divide(Vc, Ic + k0 * I0)
    candidates["AB"] = safe_divide(Va - Vb, Ia - Ib)
    candidates["BC"] = safe_divide(Vb - Vc, Ib - Ic)
    candidates["CA"] = safe_divide(Vc - Va, Ic - Ia)

    selected_loop = min(candidates, key=lambda x: abs(candidates[x]))

    return candidates[selected_loop], selected_loop


def calculate_distance_by_reactance(z_app: complex, z1_per_km: complex) -> float:
    """
    Estimasi jarak berdasarkan komponen X.

    Pada gangguan resistif, magnitude Zapp sering membesar karena tambahan Rf.
    Oleh karena itu distance berbasis X biasanya lebih masuk akal untuk indikasi awal.
    """

    if abs(z1_per_km.imag) < 1e-9:
        return 0.0

    return z_app.imag / z1_per_km.imag


def calculate_distance_by_magnitude(z_app: complex, z1_per_km: complex) -> float:
    if abs(z1_per_km) < 1e-9:
        return 0.0

    return abs(z_app) / abs(z1_per_km)


def calculate_distance_by_projection(z_app: complex, z1_per_km: complex) -> float:
    """
    Proyeksi Zapp ke arah sudut Z1.
    Cocok sebagai pembanding tambahan.
    """

    if abs(z1_per_km) < 1e-9:
        return 0.0

    unit_z1 = z1_per_km / abs(z1_per_km)
    projection = (z_app * unit_z1.conjugate()).real

    return projection / abs(z1_per_km)


def angle_deg(z: complex) -> float:
    return math.degrees(cmath.phase(z))


def estimate_fault_resistance(
    z_app: complex,
    z1_per_km: complex,
    distance_km_x: float,
):
    """
    Estimasi tahanan gangguan sederhana.

    Zline_est = distance_x * Z1_per_km
    Rf_est = real(Zapp - Zline_est)

    Ini bukan model final untuk semua kasus, tetapi cukup baik sebagai indikator awal
    gangguan high resistance pada single-ended fault locator.
    """

    z_line_est = distance_km_x * z1_per_km
    residual_z = z_app - z_line_est

    return residual_z.real, residual_z


def detect_high_resistance_fault(
    phasors: dict,
    line_param: dict,
    fault_type_result: dict,
    rf_threshold_ohm: float = 10.0,
    angle_deviation_threshold_deg: float = 10.0,
    distance_deviation_threshold_percent: float = 15.0,
):
    """
    Deteksi indikasi gangguan high resistance.

    Output:
    - high_resistance_suspected
    - Rf_est
    - Zapp
    - distance_x
    - distance_mag
    - distance_projection
    - confidence
    - warning messages
    """

    fault_type = normalize_fault_type(fault_type_result.get("fault_type", "UNKNOWN"))

    z1_per_km = line_param["Z1_per_km"]
    k0 = line_param["K0"]
    line_length_km = line_param["length_km"]

    z_app, selected_loop = calculate_loop_impedance_for_fault(
        phasors=phasors,
        fault_type=fault_type,
        k0=k0,
    )

    distance_x_km = calculate_distance_by_reactance(z_app, z1_per_km)
    distance_mag_km = calculate_distance_by_magnitude(z_app, z1_per_km)
    distance_proj_km = calculate_distance_by_projection(z_app, z1_per_km)

    rf_est_ohm, residual_z = estimate_fault_resistance(
        z_app=z_app,
        z1_per_km=z1_per_km,
        distance_km_x=distance_x_km,
    )

    z1_angle = angle_deg(z1_per_km)
    z_app_angle = angle_deg(z_app)

    angle_deviation = abs(z1_angle - z_app_angle)

    if angle_deviation > 180:
        angle_deviation = 360 - angle_deviation

    distance_deviation_percent = abs(distance_mag_km - distance_x_km) / max(
        line_length_km,
        1e-9,
    ) * 100.0

    indicators = {
        "rf_large": rf_est_ohm >= rf_threshold_ohm,
        "angle_more_resistive": angle_deviation >= angle_deviation_threshold_deg,
        "distance_methods_diverge": distance_deviation_percent >= distance_deviation_threshold_percent,
        "distance_x_inside_line": 0.0 <= distance_x_km <= 1.2 * line_length_km,
    }

    evidence_score = 0.0

    if indicators["rf_large"]:
        evidence_score += 4.0

    if indicators["angle_more_resistive"]:
        evidence_score += 2.0

    if indicators["distance_methods_diverge"]:
        evidence_score += 2.0

    if indicators["distance_x_inside_line"]:
        evidence_score += 1.0

    if "G" in fault_type:
        evidence_score += 1.0

    evidence_score = min(evidence_score, 10.0)

    analysis_confidence = 10.0

    if not indicators["distance_x_inside_line"]:
        analysis_confidence -= 3.0

    if distance_x_km < 0:
        analysis_confidence -= 2.0

    if distance_x_km > line_length_km:
        analysis_confidence -= 2.0

    if distance_deviation_percent >= 2.0 * distance_deviation_threshold_percent:
        analysis_confidence -= 1.5
    elif distance_deviation_percent >= distance_deviation_threshold_percent:
        analysis_confidence -= 0.75

    residual_x_ratio = abs(residual_z.imag) / max(abs(z_app), 1e-9)
    if residual_x_ratio > 0.20:
        analysis_confidence -= 1.0

    analysis_confidence = max(0.0, min(10.0, analysis_confidence))

    high_resistance_suspected = (
        indicators["rf_large"]
        and (
            indicators["angle_more_resistive"]
            or indicators["distance_methods_diverge"]
        )
    )

    warnings = []

    if high_resistance_suspected:
        warnings.append(
            "Indikasi gangguan high resistance: Rf_est besar dan Zapp bergeser ke arah resistif."
        )

    if distance_x_km < 0:
        warnings.append(
            "Distance berbasis reactance bernilai negatif. Cek polaritas CT/CVT atau arah arus."
        )

    if distance_x_km > line_length_km:
        warnings.append(
            "Distance berbasis reactance melebihi panjang saluran. Cek data saluran, fault type, atau kemungkinan external fault."
        )

    if distance_deviation_percent >= distance_deviation_threshold_percent:
        warnings.append(
            "Jarak berbasis magnitude dan reactance berbeda signifikan; hasil single-ended perlu dianggap uncertain."
        )

    return {
        "fault_type": fault_type,
        "selected_loop": selected_loop,
        "high_resistance_suspected": high_resistance_suspected,
        "confidence": round(analysis_confidence, 2),
        "analysis_confidence": round(analysis_confidence, 2),
        "evidence_score": round(evidence_score, 2),
        "Zapp": z_app,
        "Zapp_R": z_app.real,
        "Zapp_X": z_app.imag,
        "Zapp_mag": abs(z_app),
        "Zapp_angle_deg": z_app_angle,
        "Z1_angle_deg": z1_angle,
        "angle_deviation_deg": angle_deviation,
        "distance_x_km": distance_x_km,
        "distance_mag_km": distance_mag_km,
        "distance_projection_km": distance_proj_km,
        "distance_x_percent": distance_x_km / line_length_km * 100.0,
        "distance_mag_percent": distance_mag_km / line_length_km * 100.0,
        "distance_projection_percent": distance_proj_km / line_length_km * 100.0,
        "distance_deviation_percent": distance_deviation_percent,
        "Rf_est_ohm": rf_est_ohm,
        "residual_Z": residual_z,
        "residual_R": residual_z.real,
        "residual_X": residual_z.imag,
        "indicators": indicators,
        "warnings": warnings,
    }


def build_high_resistance_dataframe(result: dict):
    rows = [
        {"Metric": "Fault Type", "Value": result["fault_type"]},
        {"Metric": "Selected Loop", "Value": result["selected_loop"]},
        {
            "Metric": "High Resistance Suspected",
            "Value": result["high_resistance_suspected"],
        },
        {"Metric": "Analysis Confidence 0-10", "Value": result["analysis_confidence"]},
        {"Metric": "High Resistance Evidence 0-10", "Value": result["evidence_score"]},
        {"Metric": "Zapp R ohm", "Value": result["Zapp_R"]},
        {"Metric": "Zapp X ohm", "Value": result["Zapp_X"]},
        {"Metric": "Zapp Magnitude ohm", "Value": result["Zapp_mag"]},
        {"Metric": "Zapp Angle deg", "Value": result["Zapp_angle_deg"]},
        {"Metric": "Z1 Angle deg", "Value": result["Z1_angle_deg"]},
        {"Metric": "Angle Deviation deg", "Value": result["angle_deviation_deg"]},
        {"Metric": "Distance by X km", "Value": result["distance_x_km"]},
        {"Metric": "Distance by Magnitude km", "Value": result["distance_mag_km"]},
        {
            "Metric": "Distance by Projection km",
            "Value": result["distance_projection_km"],
        },
        {"Metric": "Distance by X %", "Value": result["distance_x_percent"]},
        {"Metric": "Distance by Magnitude %", "Value": result["distance_mag_percent"]},
        {
            "Metric": "Distance by Projection %",
            "Value": result["distance_projection_percent"],
        },
        {
            "Metric": "Distance Method Deviation %",
            "Value": result["distance_deviation_percent"],
        },
        {"Metric": "Estimated Fault Resistance ohm", "Value": result["Rf_est_ohm"]},
        {"Metric": "Residual R ohm", "Value": result["residual_R"]},
        {"Metric": "Residual X ohm", "Value": result["residual_X"]},
    ]

    return pd.DataFrame(rows)
