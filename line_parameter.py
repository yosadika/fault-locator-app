import math
import cmath
import pandas as pd


def polar_to_complex(magnitude: float, angle_deg: float) -> complex:
    angle_rad = math.radians(angle_deg)
    return complex(
        magnitude * math.cos(angle_rad),
        magnitude * math.sin(angle_rad),
    )


def rx_to_complex(r: float, x: float) -> complex:
    return complex(r, x)


def x_phi_to_complex(x: float, phi_deg: float) -> complex:
    phi_rad = math.radians(phi_deg)

    if abs(math.tan(phi_rad)) < 1e-9:
        raise ValueError("Sudut phi terlalu kecil untuk menghitung R dari X dan Phi.")

    r = x / math.tan(phi_rad)
    return complex(r, x)


def convert_length_to_km(length: float, length_unit: str) -> float:
    if length_unit == "km":
        return length

    if length_unit == "miles":
        return length * 1.609344

    raise ValueError("Satuan panjang tidak dikenali.")


def convert_impedance_to_primary(
    z_secondary: complex,
    ct_primary: float,
    ct_secondary: float,
    vt_primary: float,
    vt_secondary: float,
) -> complex:
    """
    Konversi impedansi sekunder ke primer.

    Zprimary = Zsecondary * VTR / CTR
    """

    ctr = ct_primary / ct_secondary
    vtr = vt_primary / vt_secondary

    return z_secondary * (vtr / ctr)


def build_positive_sequence_impedance(
    mode: str,
    r1: float | None = None,
    x1: float | None = None,
    z1_mag: float | None = None,
    phi1_deg: float | None = None,
) -> complex:
    """
    Membentuk Z1 dari beberapa format:
    - R_X
    - Z_PHI
    - X_PHI
    """

    if mode == "R_X":
        return rx_to_complex(r1, x1)

    if mode == "Z_PHI":
        return polar_to_complex(z1_mag, phi1_deg)

    if mode == "X_PHI":
        return x_phi_to_complex(x1, phi1_deg)

    raise ValueError("Mode positive-sequence tidak dikenali.")


def build_zero_sequence_impedance(
    mode: str,
    z1: complex,
    r0: float | None = None,
    x0: float | None = None,
    re_rl: float | None = None,
    xe_xl: float | None = None,
    z0_z1_mag: float | None = None,
    z0_z1_angle_deg: float | None = None,
    kl_mag: float | None = None,
    kl_angle_deg: float | None = None,
) -> complex:
    """
    Membentuk Z0 dari beberapa format.

    Mode:
    - R0_X0
    - RE_RL_XE_XL
    - Z0_Z1
    - KL

    Catatan:
    Dalam manual SIGRA, kL didefinisikan sebagai ground-impedance matching:
    kL = ZE / Z1.
    Untuk aplikasi ini, ZE dianggap sebagai residual earth impedance,
    sehingga Z0 = Z1 + 3ZE.
    """

    if mode == "R0_X0":
        return complex(r0, x0)

    if mode == "RE_RL_XE_XL":
        r1 = z1.real
        x1 = z1.imag

        # Berdasarkan definisi SIGRA:
        # kr = RE/RL = (R0/R1 - 1) / 3
        # kx = XE/XL = (X0/X1 - 1) / 3
        #
        # Maka:
        # R0 = R1 * (1 + 3 * RE/RL)
        # X0 = X1 * (1 + 3 * XE/XL)

        r0 = r1 * (1.0 + 3.0 * re_rl)
        x0 = x1 * (1.0 + 3.0 * xe_xl)

        return complex(r0, x0)

    if mode == "Z0_Z1":
        ratio = polar_to_complex(z0_z1_mag, z0_z1_angle_deg)
        return z1 * ratio

    if mode == "KL":
        kl = polar_to_complex(kl_mag, kl_angle_deg)

        # ZE = kL * Z1
        ze = kl * z1

        # ZE = (Z0 - Z1) / 3
        # Z0 = Z1 + 3ZE
        z0 = z1 + 3 * ze

        return z0

    raise ValueError("Mode zero-sequence tidak dikenali.")


def normalize_line_parameter(
    line_name: str,
    length: float,
    length_unit: str,
    impedance_input: str,
    base_side: str,
    positive_sequence_mode: str,
    zero_sequence_mode: str,
    ct_primary: float,
    ct_secondary: float,
    vt_primary: float,
    vt_secondary: float,
    r1: float | None = None,
    x1: float | None = None,
    z1_mag: float | None = None,
    phi1_deg: float | None = None,
    r0: float | None = None,
    x0: float | None = None,
    re_rl: float | None = None,
    xe_xl: float | None = None,
    z0_z1_mag: float | None = None,
    z0_z1_angle_deg: float | None = None,
    kl_mag: float | None = None,
    kl_angle_deg: float | None = None,
):
    """
    Menormalisasi input parameter saluran ke format internal aplikasi.

    Output utama:
    - Z1_per_km
    - Z0_per_km
    - Z1_total
    - Z0_total
    - K0
    """

    length_km = convert_length_to_km(length, length_unit)

    z1_input = build_positive_sequence_impedance(
        mode=positive_sequence_mode,
        r1=r1,
        x1=x1,
        z1_mag=z1_mag,
        phi1_deg=phi1_deg,
    )

    z0_input = build_zero_sequence_impedance(
        mode=zero_sequence_mode,
        z1=z1_input,
        r0=r0,
        x0=x0,
        re_rl=re_rl,
        xe_xl=xe_xl,
        z0_z1_mag=z0_z1_mag,
        z0_z1_angle_deg=z0_z1_angle_deg,
        kl_mag=kl_mag,
        kl_angle_deg=kl_angle_deg,
    )

    if base_side == "secondary":
        z1_input = convert_impedance_to_primary(
            z1_input,
            ct_primary,
            ct_secondary,
            vt_primary,
            vt_secondary,
        )

        z0_input = convert_impedance_to_primary(
            z0_input,
            ct_primary,
            ct_secondary,
            vt_primary,
            vt_secondary,
        )

    if impedance_input == "absolute":
        z1_total = z1_input
        z0_total = z0_input

        z1_per_km = z1_total / length_km
        z0_per_km = z0_total / length_km

    elif impedance_input == "relative":
        z1_per_km = z1_input
        z0_per_km = z0_input

        z1_total = z1_per_km * length_km
        z0_total = z0_per_km * length_km

    else:
        raise ValueError("Mode impedance input tidak dikenali.")

    k0 = (z0_per_km - z1_per_km) / z1_per_km

    return {
        "line_name": line_name,
        "length_km": length_km,
        "impedance_input": impedance_input,
        "base_side": base_side,
        "positive_sequence_mode": positive_sequence_mode,
        "zero_sequence_mode": zero_sequence_mode,
        "Z1_per_km": z1_per_km,
        "Z0_per_km": z0_per_km,
        "Z1_total": z1_total,
        "Z0_total": z0_total,
        "K0": k0,
    }


def complex_to_polar(z: complex):
    return abs(z), math.degrees(cmath.phase(z))


def build_line_parameter_dataframe(line_param: dict):
    rows = []

    for key in ["Z1_per_km", "Z0_per_km", "Z1_total", "Z0_total", "K0"]:
        z = line_param[key]
        mag, angle = complex_to_polar(z)

        rows.append(
            {
                "Parameter": key,
                "Real": float(z.real),
                "Imag": float(z.imag),
                "Magnitude": float(mag),
                "Angle Deg": float(angle),
            }
        )

    rows.append(
        {
            "Parameter": "Length km",
            "Real": float(line_param["length_km"]),
            "Imag": None,
            "Magnitude": None,
            "Angle Deg": None,
        }
    )

    return pd.DataFrame(rows)