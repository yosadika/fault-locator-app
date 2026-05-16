import pandas as pd


def get_mag(phasors: dict, signal_name: str) -> float:
    return float(phasors[signal_name]["magnitude"])


def detect_fault_type(
    phasors: dict,
    voltage_drop_threshold: float = 0.80,
    current_rise_threshold: float = 1.50,
    ground_current_threshold: float = 0.20,
):
    """
    Deteksi jenis gangguan berbasis fasor RMS fundamental.

    Logika utama:
    1. Cari fasa dengan arus tinggi relatif terhadap arus minimum.
    2. Cari fasa dengan tegangan drop relatif terhadap tegangan maksimum.
    3. Cek keberadaan arus tanah / zero sequence.
    4. Klasifikasikan menjadi AG, BG, CG, AB, BC, CA, ABG, BCG, CAG, ABC.

    Catatan:
    - Ini rule-based awal.
    - Nanti dapat diperkuat dengan pre-fault phasor dan negative/zero sequence.
    """

    Va = get_mag(phasors, "Va")
    Vb = get_mag(phasors, "Vb")
    Vc = get_mag(phasors, "Vc")

    Ia = get_mag(phasors, "Ia")
    Ib = get_mag(phasors, "Ib")
    Ic = get_mag(phasors, "Ic")

    IE = get_mag(phasors, "IE") if "IE" in phasors else 0.0
    I0 = get_mag(phasors, "I0") if "I0" in phasors else IE / 3.0

    voltages = {
        "A": Va,
        "B": Vb,
        "C": Vc,
    }

    currents = {
        "A": Ia,
        "B": Ib,
        "C": Ic,
    }

    max_voltage = max(voltages.values())
    min_voltage = min(voltages.values())

    max_current = max(currents.values())
    min_current = max(min(currents.values()), 1e-9)

    avg_current = sum(currents.values()) / 3.0
    avg_voltage = sum(voltages.values()) / 3.0

    # Fasa dianggap terganggu jika:
    # - arusnya tinggi dibanding arus minimum, atau
    # - tegangannya drop dibanding tegangan maksimum.
    faulted_by_current = [
        phase for phase, current in currents.items()
        if current >= current_rise_threshold * min_current
    ]

    faulted_by_voltage = [
        phase for phase, voltage in voltages.items()
        if voltage <= voltage_drop_threshold * max_voltage
    ]

    faulted_phases = sorted(
        list(set(faulted_by_current + faulted_by_voltage))
    )

    # Deteksi ground involvement.
    # Ground fault biasanya memunculkan I0/IE signifikan.
    ground_ratio_to_max_current = IE / max(max_current, 1e-9)
    ground_ratio_to_avg_current = I0 / max(avg_current, 1e-9)

    ground_involved = (
        ground_ratio_to_max_current >= ground_current_threshold
        or ground_ratio_to_avg_current >= ground_current_threshold
    )

    # Fallback jika semua fasa terdeteksi karena arus besar simetris.
    # ABC fault umumnya arus tiga fasa besar dan ground current kecil.
    current_balance_ratio = min_current / max(max_current, 1e-9)
    voltage_balance_ratio = min_voltage / max(max_voltage, 1e-9)

    is_three_phase_candidate = (
        current_balance_ratio >= 0.60
        and len(faulted_by_current) >= 2
        and not ground_involved
    )

    if is_three_phase_candidate:
        fault_type = "ABC"
        faulted_phases = ["A", "B", "C"]

    elif len(faulted_phases) == 1:
        phase = faulted_phases[0]
        if ground_involved:
            fault_type = f"{phase}G"
        else:
            # Jarang, tapi disediakan sebagai indikasi tidak pasti.
            fault_type = f"{phase}?"
    
    elif len(faulted_phases) == 2:
        pair = "".join(faulted_phases)

        if pair == "AC":
            pair = "CA"

        if ground_involved:
            fault_type = f"{pair}G"
        else:
            fault_type = pair

    elif len(faulted_phases) >= 3:
        if ground_involved:
            fault_type = "ABCG"
        else:
            fault_type = "ABC"

    else:
        fault_type = "UNKNOWN"

    confidence = calculate_fault_type_confidence(
        fault_type=fault_type,
        ground_involved=ground_involved,
        ground_ratio=ground_ratio_to_max_current,
        current_balance_ratio=current_balance_ratio,
        voltage_balance_ratio=voltage_balance_ratio,
        faulted_phases=faulted_phases,
    )

    result = {
        "fault_type": fault_type,
        "faulted_phases": faulted_phases,
        "ground_involved": ground_involved,
        "confidence": confidence,
        "metrics": {
            "Va": Va,
            "Vb": Vb,
            "Vc": Vc,
            "Ia": Ia,
            "Ib": Ib,
            "Ic": Ic,
            "IE": IE,
            "I0": I0,
            "max_voltage": max_voltage,
            "min_voltage": min_voltage,
            "max_current": max_current,
            "min_current": min_current,
            "avg_current": avg_current,
            "avg_voltage": avg_voltage,
            "ground_ratio_to_max_current": ground_ratio_to_max_current,
            "ground_ratio_to_avg_current": ground_ratio_to_avg_current,
            "current_balance_ratio": current_balance_ratio,
            "voltage_balance_ratio": voltage_balance_ratio,
            "faulted_by_current": faulted_by_current,
            "faulted_by_voltage": faulted_by_voltage,
        },
    }

    return result


def calculate_fault_type_confidence(
    fault_type: str,
    ground_involved: bool,
    ground_ratio: float,
    current_balance_ratio: float,
    voltage_balance_ratio: float,
    faulted_phases: list,
):
    """
    Skor confidence sederhana 0 sampai 10.
    Ini bukan standar relay, tetapi indikator kualitas klasifikasi awal.
    """

    score = 5.0

    if fault_type != "UNKNOWN":
        score += 1.5

    if "?" not in fault_type:
        score += 1.0

    if ground_involved and ground_ratio >= 0.25:
        score += 1.0

    if fault_type in ["ABC", "ABCG"]:
        if current_balance_ratio >= 0.70:
            score += 1.0
    else:
        if voltage_balance_ratio <= 0.90:
            score += 0.5

    if len(faulted_phases) >= 1:
        score += 0.5

    return round(min(score, 10.0), 2)


def build_fault_type_metrics_dataframe(result: dict):
    """
    Membuat DataFrame ringkasan metrik deteksi.
    """

    metrics = result["metrics"]

    rows = [
        {"Metric": "Fault Type", "Value": result["fault_type"]},
        {"Metric": "Faulted Phases", "Value": ", ".join(result["faulted_phases"])},
        {"Metric": "Ground Involved", "Value": result["ground_involved"]},
        {"Metric": "Confidence 0-10", "Value": result["confidence"]},
        {"Metric": "Va RMS", "Value": metrics["Va"]},
        {"Metric": "Vb RMS", "Value": metrics["Vb"]},
        {"Metric": "Vc RMS", "Value": metrics["Vc"]},
        {"Metric": "Ia RMS", "Value": metrics["Ia"]},
        {"Metric": "Ib RMS", "Value": metrics["Ib"]},
        {"Metric": "Ic RMS", "Value": metrics["Ic"]},
        {"Metric": "IE RMS", "Value": metrics["IE"]},
        {"Metric": "I0 RMS", "Value": metrics["I0"]},
        {
            "Metric": "Ground Ratio to Max Current",
            "Value": metrics["ground_ratio_to_max_current"],
        },
        {
            "Metric": "Current Balance Ratio",
            "Value": metrics["current_balance_ratio"],
        },
        {
            "Metric": "Voltage Balance Ratio",
            "Value": metrics["voltage_balance_ratio"],
        },
        {
            "Metric": "Faulted by Current",
            "Value": ", ".join(metrics["faulted_by_current"]),
        },
        {
            "Metric": "Faulted by Voltage",
            "Value": ", ".join(metrics["faulted_by_voltage"]),
        },
    ]

    return pd.DataFrame(rows)