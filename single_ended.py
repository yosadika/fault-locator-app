import math
import cmath
import pandas as pd


def get_complex(phasors: dict, name: str) -> complex:
    return phasors[name]["complex"]


def safe_divide(numerator: complex, denominator: complex, eps: float = 1e-9):
    if abs(denominator) < eps:
        raise ZeroDivisionError("Denominator terlalu kecil untuk perhitungan impedansi.")
    return numerator / denominator


def normalize_fault_type(fault_type: str) -> str:
    if fault_type is None:
        return "UNKNOWN"

    fault_type = fault_type.upper().strip()

    if fault_type == "AC":
        return "CA"

    if fault_type == "ACG":
        return "CAG"

    return fault_type


def calculate_single_ended_loop(phasors: dict, fault_type: str, k0: complex):
    """
    Menghitung impedansi loop single-ended sesuai jenis gangguan.

    AG  : Va / (Ia + K0 * I0)
    BG  : Vb / (Ib + K0 * I0)
    CG  : Vc / (Ic + K0 * I0)

    AB  : (Va - Vb) / (Ia - Ib)
    BC  : (Vb - Vc) / (Ib - Ic)
    CA  : (Vc - Va) / (Ic - Ia)

    ABC : V1 / I1
    """

    fault_type = normalize_fault_type(fault_type)

    Va = get_complex(phasors, "Va")
    Vb = get_complex(phasors, "Vb")
    Vc = get_complex(phasors, "Vc")

    Ia = get_complex(phasors, "Ia")
    Ib = get_complex(phasors, "Ib")
    Ic = get_complex(phasors, "Ic")

    I0 = get_complex(phasors, "I0")

    # Untuk ABC, lebih stabil pakai V1 / I1 jika tersedia
    V1 = get_complex(phasors, "V1") if "V1" in phasors else None
    I1 = get_complex(phasors, "I1") if "I1" in phasors else None

    if fault_type == "AG":
        return {
            "selected_loop": "AG",
            "loop_voltage": Va,
            "loop_current": Ia + k0 * I0,
            "Zapp": safe_divide(Va, Ia + k0 * I0),
        }

    if fault_type == "BG":
        return {
            "selected_loop": "BG",
            "loop_voltage": Vb,
            "loop_current": Ib + k0 * I0,
            "Zapp": safe_divide(Vb, Ib + k0 * I0),
        }

    if fault_type == "CG":
        return {
            "selected_loop": "CG",
            "loop_voltage": Vc,
            "loop_current": Ic + k0 * I0,
            "Zapp": safe_divide(Vc, Ic + k0 * I0),
        }

    if fault_type == "AB":
        return {
            "selected_loop": "AB",
            "loop_voltage": Va - Vb,
            "loop_current": Ia - Ib,
            "Zapp": safe_divide(Va - Vb, Ia - Ib),
        }

    if fault_type == "BC":
        return {
            "selected_loop": "BC",
            "loop_voltage": Vb - Vc,
            "loop_current": Ib - Ic,
            "Zapp": safe_divide(Vb - Vc, Ib - Ic),
        }

    if fault_type == "CA":
        return {
            "selected_loop": "CA",
            "loop_voltage": Vc - Va,
            "loop_current": Ic - Ia,
            "Zapp": safe_divide(Vc - Va, Ic - Ia),
        }

    # Untuk two-phase-ground, gunakan loop phase-phase sebagai pendekatan lokasi awal
    if fault_type == "ABG":
        return {
            "selected_loop": "AB",
            "loop_voltage": Va - Vb,
            "loop_current": Ia - Ib,
            "Zapp": safe_divide(Va - Vb, Ia - Ib),
        }

    if fault_type == "BCG":
        return {
            "selected_loop": "BC",
            "loop_voltage": Vb - Vc,
            "loop_current": Ib - Ic,
            "Zapp": safe_divide(Vb - Vc, Ib - Ic),
        }

    if fault_type == "CAG":
        return {
            "selected_loop": "CA",
            "loop_voltage": Vc - Va,
            "loop_current": Ic - Ia,
            "Zapp": safe_divide(Vc - Va, Ic - Ia),
        }

    if fault_type in ["ABC", "ABCG"]:
        if V1 is not None and I1 is not None:
            return {
                "selected_loop": "V1/I1",
                "loop_voltage": V1,
                "loop_current": I1,
                "Zapp": safe_divide(V1, I1),
            }

        return {
            "selected_loop": "A",
            "loop_voltage": Va,
            "loop_current": Ia,
            "Zapp": safe_divide(Va, Ia),
        }

    # Fallback jika fault type tidak jelas:
    # hitung semua loop dan ambil impedansi paling kecil yang masuk akal
    candidates = {
        "AG": safe_divide(Va, Ia + k0 * I0),
        "BG": safe_divide(Vb, Ib + k0 * I0),
        "CG": safe_divide(Vc, Ic + k0 * I0),
        "AB": safe_divide(Va - Vb, Ia - Ib),
        "BC": safe_divide(Vb - Vc, Ib - Ic),
        "CA": safe_divide(Vc - Va, Ic - Ia),
    }

    selected_loop = min(candidates, key=lambda key: abs(candidates[key]))
    zapp = candidates[selected_loop]

    if selected_loop == "AG":
        loop_voltage = Va
        loop_current = Ia + k0 * I0
    elif selected_loop == "BG":
        loop_voltage = Vb
        loop_current = Ib + k0 * I0
    elif selected_loop == "CG":
        loop_voltage = Vc
        loop_current = Ic + k0 * I0
    elif selected_loop == "AB":
        loop_voltage = Va - Vb
        loop_current = Ia - Ib
    elif selected_loop == "BC":
        loop_voltage = Vb - Vc
        loop_current = Ib - Ic
    else:
        loop_voltage = Vc - Va
        loop_current = Ic - Ia

    return {
        "selected_loop": selected_loop,
        "loop_voltage": loop_voltage,
        "loop_current": loop_current,
        "Zapp": zapp,
    }


def calculate_distance_by_magnitude(zapp: complex, z1_per_km: complex):
    return abs(zapp) / abs(z1_per_km)


def calculate_distance_by_reactance(zapp: complex, z1_per_km: complex):
    if abs(z1_per_km.imag) < 1e-9:
        return 0.0
    return zapp.imag / z1_per_km.imag


def calculate_distance_by_projection(zapp: complex, z1_per_km: complex):
    """
    Proyeksi Zapp ke arah sudut Z1.
    Metode ini lebih stabil dibanding magnitude murni untuk gangguan resistif.
    """

    if abs(z1_per_km) < 1e-9:
        return 0.0

    unit_z1 = z1_per_km / abs(z1_per_km)
    projected_ohm = (zapp * unit_z1.conjugate()).real

    return projected_ohm / abs(z1_per_km)


def angle_deg(z: complex):
    return math.degrees(cmath.phase(z))


def estimate_fault_resistance(zapp: complex, z1_per_km: complex, distance_km: float):
    """
    Estimasi tahanan gangguan sederhana:
    Rf ≈ Re[Zapp - distance * Z1_per_km]
    """

    z_line = distance_km * z1_per_km
    residual = zapp - z_line

    return residual.real, residual


def calculate_single_ended_fault_location(
    phasors: dict,
    fault_type_result: dict,
    line_param: dict,
    recommended_method: str = "reactance",
):
    fault_type = normalize_fault_type(fault_type_result.get("fault_type", "UNKNOWN"))

    z1_per_km = line_param["Z1_per_km"]
    k0 = line_param["K0"]
    line_length_km = line_param["length_km"]

    loop = calculate_single_ended_loop(
        phasors=phasors,
        fault_type=fault_type,
        k0=k0,
    )

    zapp = loop["Zapp"]

    distance_mag_km = calculate_distance_by_magnitude(zapp, z1_per_km)
    distance_x_km = calculate_distance_by_reactance(zapp, z1_per_km)
    distance_proj_km = calculate_distance_by_projection(zapp, z1_per_km)

    if recommended_method == "magnitude":
        recommended_distance_km = distance_mag_km
    elif recommended_method == "projection":
        recommended_distance_km = distance_proj_km
    else:
        recommended_distance_km = distance_x_km

    recommended_distance_percent = recommended_distance_km / line_length_km * 100.0

    rf_est_ohm, residual_z = estimate_fault_resistance(
        zapp=zapp,
        z1_per_km=z1_per_km,
        distance_km=recommended_distance_km,
    )

    z1_angle = angle_deg(z1_per_km)
    zapp_angle = angle_deg(zapp)
    angle_deviation = abs(z1_angle - zapp_angle)

    if angle_deviation > 180:
        angle_deviation = 360 - angle_deviation

    warnings = []

    if recommended_distance_km < 0:
        warnings.append("Jarak bernilai negatif. Cek polaritas CT/CVT, arah arus, atau pemilihan loop.")

    if recommended_distance_km > line_length_km:
        warnings.append("Jarak melebihi panjang saluran. Cek line parameter, fault type, atau kemungkinan external fault.")

    if abs(distance_mag_km - distance_x_km) / max(line_length_km, 1e-9) * 100 > 15:
        warnings.append("Distance magnitude dan reactance berbeda signifikan. Ada indikasi gangguan resistif atau data tidak ideal.")

    if abs(rf_est_ohm) > 10:
        warnings.append("Estimasi tahanan gangguan cukup besar. Hasil single-ended perlu divalidasi.")

    if angle_deviation > 15:
        warnings.append(
            "Sudut Zapp loop single-ended menyimpang dari sudut Z1. "
            "Ini indikasi pembanding single-ended lebih resistif atau data tidak ideal, bukan warning langsung dari metode double-ended."
        )

    status = "VALID"

    if warnings:
        status = "CHECK"

    if recommended_distance_km < 0 or recommended_distance_km > line_length_km:
        status = "UNCERTAIN"

    result = {
        "method": "single_ended",
        "fault_type": fault_type,
        "selected_loop": loop["selected_loop"],
        "loop_voltage": loop["loop_voltage"],
        "loop_current": loop["loop_current"],
        "loop_voltage_mag": abs(loop["loop_voltage"]),
        "loop_current_mag": abs(loop["loop_current"]),
        "Zapp": zapp,
        "Zapp_R": zapp.real,
        "Zapp_X": zapp.imag,
        "Zapp_mag": abs(zapp),
        "Zapp_angle_deg": zapp_angle,
        "Z1_angle_deg": z1_angle,
        "angle_deviation_deg": angle_deviation,
        "distance_mag_km": distance_mag_km,
        "distance_x_km": distance_x_km,
        "distance_projection_km": distance_proj_km,
        "distance_mag_percent": distance_mag_km / line_length_km * 100.0,
        "distance_x_percent": distance_x_km / line_length_km * 100.0,
        "distance_projection_percent": distance_proj_km / line_length_km * 100.0,
        "recommended_method": recommended_method,
        "recommended_distance_km": recommended_distance_km,
        "recommended_distance_percent": recommended_distance_percent,
        "Rf_est_ohm": rf_est_ohm,
        "residual_Z": residual_z,
        "residual_R": residual_z.real,
        "residual_X": residual_z.imag,
        "status": status,
        "warnings": warnings,
    }

    return result


def build_single_ended_result_dataframe(result: dict):
    rows = [
        {"Parameter": "Fault Type", "Value": result["fault_type"]},
        {"Parameter": "Selected Loop", "Value": result["selected_loop"]},
        {"Parameter": "Status", "Value": result["status"]},
        {"Parameter": "Loop Voltage Magnitude", "Value": result["loop_voltage_mag"]},
        {"Parameter": "Loop Current Magnitude", "Value": result["loop_current_mag"]},
        {"Parameter": "Zapp R ohm", "Value": result["Zapp_R"]},
        {"Parameter": "Zapp X ohm", "Value": result["Zapp_X"]},
        {"Parameter": "Zapp Magnitude ohm", "Value": result["Zapp_mag"]},
        {"Parameter": "Zapp Angle deg", "Value": result["Zapp_angle_deg"]},
        {"Parameter": "Z1 Angle deg", "Value": result["Z1_angle_deg"]},
        {"Parameter": "Angle Deviation deg", "Value": result["angle_deviation_deg"]},
        {"Parameter": "Distance by Magnitude km", "Value": result["distance_mag_km"]},
        {"Parameter": "Distance by Reactance km", "Value": result["distance_x_km"]},
        {"Parameter": "Distance by Projection km", "Value": result["distance_projection_km"]},
        {"Parameter": "Distance by Magnitude %", "Value": result["distance_mag_percent"]},
        {"Parameter": "Distance by Reactance %", "Value": result["distance_x_percent"]},
        {"Parameter": "Distance by Projection %", "Value": result["distance_projection_percent"]},
        {"Parameter": "Recommended Method", "Value": result["recommended_method"]},
        {"Parameter": "Recommended Distance km", "Value": result["recommended_distance_km"]},
        {"Parameter": "Recommended Distance %", "Value": result["recommended_distance_percent"]},
        {"Parameter": "Estimated Fault Resistance ohm", "Value": result["Rf_est_ohm"]},
        {"Parameter": "Residual R ohm", "Value": result["residual_R"]},
        {"Parameter": "Residual X ohm", "Value": result["residual_X"]},
    ]

    return pd.DataFrame(rows)
