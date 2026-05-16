import pandas as pd


def apply_signal_assignment(
    df: pd.DataFrame,
    va_channel: str,
    vb_channel: str,
    vc_channel: str,
    ia_channel: str,
    ib_channel: str,
    ic_channel: str,
    ie_channel: str | None = None,
    recorded_side: str = "secondary",
    ct_primary: float = 800.0,
    ct_secondary: float = 1.0,
    vt_primary: float = 150000.0,
    vt_secondary: float = 100.0,
):
    """
    Melakukan mapping channel COMTRADE menjadi nama standar:
    Va, Vb, Vc, Ia, Ib, Ic, IE.

    Jika recorded_side = secondary, nilai dikonversi ke primary.
    Jika recorded_side = primary, nilai dipakai langsung.

    Catatan:
    - Tegangan dikalikan VT ratio jika data sekunder.
    - Arus dikalikan CT ratio jika data sekunder.
    """

    assigned = pd.DataFrame()
    assigned["time"] = df["time"]

    ctr = ct_primary / ct_secondary
    vtr = vt_primary / vt_secondary

    if recorded_side == "secondary":
        voltage_multiplier = vtr
        current_multiplier = ctr
    else:
        voltage_multiplier = 1.0
        current_multiplier = 1.0

    assigned["Va"] = df[va_channel] * voltage_multiplier
    assigned["Vb"] = df[vb_channel] * voltage_multiplier
    assigned["Vc"] = df[vc_channel] * voltage_multiplier

    assigned["Ia"] = df[ia_channel] * current_multiplier
    assigned["Ib"] = df[ib_channel] * current_multiplier
    assigned["Ic"] = df[ic_channel] * current_multiplier

    if ie_channel and ie_channel != "None":
        assigned["IE"] = df[ie_channel] * current_multiplier
        assigned["IE_source"] = "measured"
    else:
        assigned["IE"] = assigned["Ia"] + assigned["Ib"] + assigned["Ic"]
        assigned["IE_source"] = "calculated_from_3_phase_currents"

    assigned["I0"] = assigned["IE"] / 3.0

    return assigned