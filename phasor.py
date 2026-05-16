import numpy as np
import pandas as pd


def calculate_full_cycle_dft_phasor(
    signal: np.ndarray,
    cursor_index: int,
    samples_per_cycle: int,
):
    """
    Menghitung fasor RMS fundamental menggunakan full-cycle DFT.

    Konsep:
    - Measuring window berada di kiri cursor_index.
    - Panjang window = 1 siklus.
    - Output berupa bilangan kompleks RMS.

    Jika cursor_index = 120 dan samples_per_cycle = 20,
    maka data yang dipakai adalah index 100 sampai 119.
    """

    if samples_per_cycle < 4:
        raise ValueError("samples_per_cycle terlalu kecil untuk DFT.")

    start_index = cursor_index - samples_per_cycle
    end_index = cursor_index

    if start_index < 0:
        raise ValueError("Cursor terlalu awal. Window DFT tidak cukup 1 siklus.")

    window = np.asarray(signal[start_index:end_index], dtype=float)

    if len(window) != samples_per_cycle:
        raise ValueError("Panjang window tidak sama dengan samples_per_cycle.")

    n = np.arange(samples_per_cycle)

    # Full-cycle DFT fundamental.
    # Faktor 2/N menghasilkan peak phasor untuk sinyal sinus.
    # Dibagi sqrt(2) agar menjadi RMS phasor.
    dft = (2.0 / samples_per_cycle) * np.sum(
        window * np.exp(-1j * 2.0 * np.pi * n / samples_per_cycle)
    )

    phasor_rms = dft / np.sqrt(2.0)

    return phasor_rms


def phasor_to_magnitude_angle(phasor: complex):
    """
    Mengubah bilangan kompleks menjadi magnitude dan sudut derajat.
    """

    magnitude = abs(phasor)
    angle_deg = np.degrees(np.angle(phasor))

    return magnitude, angle_deg


def calculate_all_phasors(
    df: pd.DataFrame,
    cursor_index: int,
    samples_per_cycle: int,
):
    """
    Menghitung fasor Va, Vb, Vc, Ia, Ib, Ic, IE, I0
    pada cursor DFT.
    """

    required_signals = ["Va", "Vb", "Vc", "Ia", "Ib", "Ic", "IE", "I0"]

    phasors = {}

    for signal_name in required_signals:
        if signal_name not in df.columns:
            continue

        phasor = calculate_full_cycle_dft_phasor(
            signal=df[signal_name].values,
            cursor_index=cursor_index,
            samples_per_cycle=samples_per_cycle,
        )

        magnitude, angle_deg = phasor_to_magnitude_angle(phasor)

        phasors[signal_name] = {
            "complex": phasor,
            "real": phasor.real,
            "imag": phasor.imag,
            "magnitude": magnitude,
            "angle_deg": angle_deg,
        }

    return phasors


def build_phasor_dataframe(phasors: dict):
    """
    Membuat DataFrame ringkasan fasor agar mudah ditampilkan di Streamlit.
    """

    rows = []

    for signal_name, value in phasors.items():
        rows.append(
            {
                "Signal": signal_name,
                "Magnitude RMS": value["magnitude"],
                "Angle Deg": value["angle_deg"],
                "Real": value["real"],
                "Imag": value["imag"],
            }
        )

    return pd.DataFrame(rows)


def calculate_sequence_components(phasors: dict):
    """
    Menghitung komponen simetris V0, V1, V2 dan I0, I1, I2
    dari fasor 3 fasa.

    a = 1∠120°
    """

    a = np.exp(1j * 2.0 * np.pi / 3.0)

    Va = phasors["Va"]["complex"]
    Vb = phasors["Vb"]["complex"]
    Vc = phasors["Vc"]["complex"]

    Ia = phasors["Ia"]["complex"]
    Ib = phasors["Ib"]["complex"]
    Ic = phasors["Ic"]["complex"]

    V0 = (Va + Vb + Vc) / 3.0
    V1 = (Va + a * Vb + (a ** 2) * Vc) / 3.0
    V2 = (Va + (a ** 2) * Vb + a * Vc) / 3.0

    I0 = (Ia + Ib + Ic) / 3.0
    I1 = (Ia + a * Ib + (a ** 2) * Ic) / 3.0
    I2 = (Ia + (a ** 2) * Ib + a * Ic) / 3.0

    sequence = {
        "V0": V0,
        "V1": V1,
        "V2": V2,
        "I0": I0,
        "I1": I1,
        "I2": I2,
    }

    rows = []

    for name, value in sequence.items():
        rows.append(
            {
                "Component": name,
                "Magnitude RMS": abs(value),
                "Angle Deg": np.degrees(np.angle(value)),
                "Real": value.real,
                "Imag": value.imag,
            }
        )

    return sequence, pd.DataFrame(rows)


def add_sequence_components_to_phasor_dict(phasors: dict):
    """
    Menambahkan V0, V1, V2, I0, I1, I2 ke dictionary phasors
    dengan format yang sama seperti Va, Vb, Vc, Ia, Ib, Ic.
    """

    sequence, _ = calculate_sequence_components(phasors)

    for name, value in sequence.items():
        phasors[name] = {
            "complex": value,
            "real": value.real,
            "imag": value.imag,
            "magnitude": abs(value),
            "angle_deg": np.degrees(np.angle(value)),
        }

    return phasors