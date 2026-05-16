import re
import pandas as pd
import numpy as np


VOLTAGE_KEYWORDS = [
    "VA", "VB", "VC",
    "V_A", "V_B", "V_C",
    "UA", "UB", "UC",
    "U_A", "U_B", "U_C",
    "VL1", "VL2", "VL3",
    "UL1", "UL2", "UL3",
    "UL1E", "UL2E", "UL3E",
    "VAN", "VBN", "VCN",
    "VR", "VS", "VT",
]

CURRENT_KEYWORDS = [
    "IA", "IB", "IC",
    "I_A", "I_B", "I_C",
    "IL1", "IL2", "IL3",
    "I L1", "I L2", "I L3",
    "IR", "IS", "IT",
]

GROUND_CURRENT_KEYWORDS = [
    "IE", "IN", "IG", "3I0", "I0", "RES", "RESIDUAL", "EARTH", "GROUND", "NEUTRAL"
]


TRIP_KEYWORDS = [
    "TRIP", "TRIPPING", "OPERATE", "START", "PICKUP", "PICK-UP",
    "DIST", "ZONE", "Z1", "Z2", "Z3", "FAULT", "GENERAL TRIP"
]


def normalize_name(name: str) -> str:
    name = str(name).upper()
    name = name.replace("-", "")
    name = name.replace("_", "")
    name = name.replace(" ", "")
    name = name.replace(".", "")
    name = name.replace(":", "")
    return name


def get_original_channel_name(channel_name: str) -> str:
    """
    Mengambil nama asli channel dari label unik.

    Contoh:
    A001 | IA  -> IA
    A014 | VA  -> VA
    """

    text = str(channel_name)

    if " | " in text:
        return text.split(" | ", 1)[1].strip()

    return text.strip()


def get_analog_index_from_unique_label(channel_name: str) -> int:
    """
    Mengambil index analog dari label unik.

    Contoh:
    A001 | IA -> 1
    A014 | VA -> 14
    """

    text = str(channel_name).strip()

    if text.startswith("A") and " | " in text:
        try:
            prefix = text.split(" | ", 1)[0]
            return int(prefix.replace("A", ""))
        except Exception:
            return 999999

    return 999999


def score_channel_name(name: str, candidates: list[str]) -> int:
    original_name = get_original_channel_name(name)
    n = normalize_name(original_name)

    score = 0

    for keyword in candidates:
        k = normalize_name(keyword)

        if n == k:
            score += 100

        elif k in n:
            score += 50

    return score


def detect_voltage_current_channels(df: pd.DataFrame, metadata: dict):
    """
    Deteksi otomatis channel tegangan dan arus.

    Prioritas:
    1. Nama channel dan unit dari .cfg
    2. Pola nama L1/L2/L3, A/B/C, R/S/T
    3. Magnitude statistik sebagai fallback
    """

    analog_meta = metadata.get("analog_metadata", [])
    channel_names = [c for c in df.columns if c != "time"]

    meta_by_name = {
        str(m.get("name", "")): m for m in analog_meta
    }

    scores = {}

    for ch in channel_names:
        meta = meta_by_name.get(ch, {})
        unit = str(meta.get("unit", "")).lower()

        v_score = score_channel_name(ch, VOLTAGE_KEYWORDS)
        i_score = score_channel_name(ch, CURRENT_KEYWORDS)
        g_score = score_channel_name(ch, GROUND_CURRENT_KEYWORDS)

        if "v" in unit:
            v_score += 80

        if "a" in unit:
            i_score += 80

        scores[ch] = {
            "voltage_score": v_score,
            "current_score": i_score,
            "ground_score": g_score,
            "unit": unit,
        }

    voltage_candidates = sorted(
        channel_names,
        key=lambda ch: scores[ch]["voltage_score"],
        reverse=True,
    )

    current_candidates = sorted(
        channel_names,
        key=lambda ch: scores[ch]["current_score"],
        reverse=True,
    )

    ground_candidates = sorted(
        channel_names,
        key=lambda ch: scores[ch]["ground_score"],
        reverse=True,
    )

    va, vb, vc = pick_three_phase(voltage_candidates, scores, kind="voltage")
    ia, ib, ic = pick_three_phase(current_candidates, scores, kind="current")

    ie = None
    if ground_candidates:
        best_ground = ground_candidates[0]
        if scores[best_ground]["ground_score"] > 0:
            ie = best_ground

    # Fallback magnitude jika nama channel tidak informatif
    if not all([va, vb, vc, ia, ib, ic]):
        fallback = magnitude_based_fallback(df, channel_names)
        va = va or fallback.get("Va")
        vb = vb or fallback.get("Vb")
        vc = vc or fallback.get("Vc")
        ia = ia or fallback.get("Ia")
        ib = ib or fallback.get("Ib")
        ic = ic or fallback.get("Ic")

    result = {
        "Va": va,
        "Vb": vb,
        "Vc": vc,
        "Ia": ia,
        "Ib": ib,
        "Ic": ic,
        "IE": ie,
        "scores": scores,
    }

    return result


def pick_three_phase(candidates: list[str], scores: dict, kind: str):
    """
    Memilih channel A/B/C atau L1/L2/L3 dari daftar kandidat.
    """

    threshold_key = "voltage_score" if kind == "voltage" else "current_score"

    good_candidates = [
        ch for ch in candidates
        if scores[ch][threshold_key] > 0
    ]

    selected = {
        "A": None,
        "B": None,
        "C": None,
    }

    for ch in good_candidates:
        n = normalize_name(ch)

        if selected["A"] is None and re.search(r"(A|L1|R|1)$", n):
            selected["A"] = ch
            continue

        if selected["B"] is None and re.search(r"(B|L2|S|2)$", n):
            selected["B"] = ch
            continue

        if selected["C"] is None and re.search(r"(C|L3|T|3)$", n):
            selected["C"] = ch
            continue

    # Fallback dari urutan kandidat terbaik
    remaining = [ch for ch in good_candidates if ch not in selected.values()]

    for phase in ["A", "B", "C"]:
        if selected[phase] is None and remaining:
            selected[phase] = remaining.pop(0)

    return selected["A"], selected["B"], selected["C"]


def magnitude_based_fallback(df: pd.DataFrame, channel_names: list[str]):
    """
    Fallback kasar:
    - Channel dengan magnitude RMS terbesar cenderung tegangan jika data sekunder V sekitar puluhan-ratusan
    - Channel arus bisa kecil atau besar, jadi ini hanya fallback terakhir.
    """

    stats = []

    for ch in channel_names:
        values = pd.to_numeric(df[ch], errors="coerce").dropna().values

        if len(values) == 0:
            continue

        rms = float(np.sqrt(np.mean(values ** 2)))
        peak = float(np.nanmax(np.abs(values)))

        stats.append(
            {
                "channel": ch,
                "rms": rms,
                "peak": peak,
            }
        )

    stats = sorted(stats, key=lambda x: x["rms"], reverse=True)

    fallback = {
        "Va": None,
        "Vb": None,
        "Vc": None,
        "Ia": None,
        "Ib": None,
        "Ic": None,
    }

    if len(stats) >= 6:
        # Asumsi awal: 3 magnitude terbesar tegangan, 3 berikutnya arus.
        voltage_guess = [s["channel"] for s in stats[:3]]
        current_guess = [s["channel"] for s in stats[3:6]]

        fallback["Va"], fallback["Vb"], fallback["Vc"] = voltage_guess
        fallback["Ia"], fallback["Ib"], fallback["Ic"] = current_guess

    return fallback


def detect_recorded_side(metadata: dict):
    """
    Menebak apakah data recorded primary atau secondary.
    Jika .cfg punya primary/secondary dan PS field, gunakan PS.
    Jika tidak jelas, default secondary.
    """

    analog_meta = metadata.get("analog_metadata", [])

    ps_values = []
    for ch in analog_meta:
        ps = str(ch.get("ps", "")).upper().strip()
        if ps:
            ps_values.append(ps)

    if not ps_values:
        return "secondary"

    # COMTRADE PS field biasanya P/S
    p_count = sum(1 for p in ps_values if p.startswith("P"))
    s_count = sum(1 for p in ps_values if p.startswith("S"))

    if p_count > s_count:
        return "primary"

    return "secondary"


def get_auto_transformer_data(metadata: dict):
    """
    Mengambil ratio dari .cfg jika tersedia.
    Karena metadata hanya memberi ratio, kita isi:
    VT primary = ratio, VT secondary = 1
    CT primary = ratio, CT secondary = 1

    User tetap bisa koreksi manual.
    """

    vt_ratio = metadata.get("vt_ratio_from_cfg")
    ct_ratio = metadata.get("ct_ratio_from_cfg")

    vt_primary = float(vt_ratio) if vt_ratio else 150000.0
    vt_secondary = 1.0 if vt_ratio else 100.0

    ct_primary = float(ct_ratio) if ct_ratio else 800.0
    ct_secondary = 1.0

    return {
        "ct_primary": ct_primary,
        "ct_secondary": ct_secondary,
        "vt_primary": vt_primary,
        "vt_secondary": vt_secondary,
        "ct_ratio_source": "CFG" if ct_ratio else "DEFAULT",
        "vt_ratio_source": "CFG" if vt_ratio else "DEFAULT",
    }


def detect_trigger_from_digital(df: pd.DataFrame, digital_df: pd.DataFrame | None = None):
    """
    Placeholder untuk data digital.
    Library comtrade yang kita pakai saat ini belum kita masukkan digital channel ke DataFrame.
    Nanti bisa dikembangkan untuk membaca binary trip/pickup.
    """

    return None


def build_auto_assignment_summary(auto_assignment: dict, transformer_data: dict, metadata: dict):
    rows = []

    for key in ["Va", "Vb", "Vc", "Ia", "Ib", "Ic", "IE"]:
        rows.append(
            {
                "Item": key,
                "Detected Channel": auto_assignment.get(key),
            }
        )

    rows.extend(
        [
            {"Item": "Recorded Side", "Detected Channel": detect_recorded_side(metadata)},
            {"Item": "CT Primary", "Detected Channel": transformer_data["ct_primary"]},
            {"Item": "CT Secondary", "Detected Channel": transformer_data["ct_secondary"]},
            {"Item": "VT Primary", "Detected Channel": transformer_data["vt_primary"]},
            {"Item": "VT Secondary", "Detected Channel": transformer_data["vt_secondary"]},
            {"Item": "CT Ratio Source", "Detected Channel": transformer_data["ct_ratio_source"]},
            {"Item": "VT Ratio Source", "Detected Channel": transformer_data["vt_ratio_source"]},
            {"Item": "CFG Start Time", "Detected Channel": metadata.get("cfg_start_time")},
            {"Item": "CFG Trigger Time", "Detected Channel": metadata.get("cfg_trigger_time")},
        ]
    )

    return pd.DataFrame(rows)


def detect_three_phase_channel_sets(df: pd.DataFrame, metadata: dict):
    """
    Mendeteksi beberapa kandidat set channel 3 fasa untuk tegangan dan arus.

    Output:
    {
        "voltage_sets": [
            {"label": "...", "A": "...", "B": "...", "C": "...", "score": ...}
        ],
        "current_sets": [
            {"label": "...", "A": "...", "B": "...", "C": "...", "score": ...}
        ],
        "ground_candidates": [...]
    }
    """

    channel_names = [c for c in df.columns if c != "time"]
    analog_meta = metadata.get("analog_metadata", [])

    meta_by_name = {
        str(m.get("name", "")): m for m in analog_meta
    }

    channel_info = []

    for ch in channel_names:
        original_ch = get_original_channel_name(ch)

        meta = meta_by_name.get(original_ch, {})
        unit = str(meta.get("unit", "")).lower()

        voltage_score = score_channel_name(ch, VOLTAGE_KEYWORDS)
        current_score = score_channel_name(ch, CURRENT_KEYWORDS)
        ground_score = score_channel_name(ch, GROUND_CURRENT_KEYWORDS)

        if "v" in unit:
            voltage_score += 80

        if "a" in unit:
            current_score += 80

        channel_info.append(
            {
                "channel": ch,
                "original_channel": original_ch,
                "normalized": normalize_name(original_ch),
                "unit": unit,
                "voltage_score": voltage_score,
                "current_score": current_score,
                "ground_score": ground_score,
            }
        )

    voltage_sets = build_three_phase_sets(channel_info, kind="voltage")
    current_sets = build_three_phase_sets(channel_info, kind="current")

    ground_candidates = sorted(
        [
            {
                "channel": item["channel"],
                "score": item["ground_score"],
                "unit": item["unit"],
            }
            for item in channel_info
            if item["ground_score"] > 0
        ],
        key=lambda x: x["score"],
        reverse=True,
    )

    return {
        "voltage_sets": voltage_sets,
        "current_sets": current_sets,
        "ground_candidates": ground_candidates,
    }


def build_three_phase_sets(channel_info: list, kind: str):
    """
    Membangun kandidat set A-B-C / L1-L2-L3 / R-S-T.

    Mendukung duplicate channel names:
    A001 | IA, A002 | IB, A003 | IC
    A010 | IA, A011 | IB, A012 | IC

    Akan dibuat menjadi:
    Current Set #1
    Current Set #2
    """

    score_key = "voltage_score" if kind == "voltage" else "current_score"

    candidates = [
        item for item in channel_info
        if item[score_key] > 0
    ]

    # Urutkan sesuai urutan analog channel
    candidates = sorted(
        candidates,
        key=lambda x: get_analog_index_from_unique_label(x["channel"])
    )

    groups = {}

    for item in candidates:
        phase = detect_phase_marker(item["channel"])
        base = remove_phase_marker(item["channel"])

        if phase is None:
            continue

        if base not in groups:
            groups[base] = {
                "A": [],
                "B": [],
                "C": [],
            }

        groups[base][phase].append(item)

    sets = []

    for base, phases in groups.items():
        count_a = len(phases["A"])
        count_b = len(phases["B"])
        count_c = len(phases["C"])

        set_count = min(count_a, count_b, count_c)

        for set_index in range(set_count):
            a = phases["A"][set_index]
            b = phases["B"][set_index]
            c = phases["C"][set_index]

            score = a[score_key] + b[score_key] + c[score_key]

            label = (
                f"{kind.capitalize()} Set #{set_index + 1} | {base} | "
                f"A={a['channel']}, B={b['channel']}, C={c['channel']} "
                f"| score={score}"
            )

            sets.append(
                {
                    "label": label,
                    "base": base,
                    "set_index": set_index + 1,
                    "A": a["channel"],
                    "B": b["channel"],
                    "C": c["channel"],
                    "score": score,
                    "kind": kind,
                }
            )

    # Fallback kalau tidak ada group lengkap
    if not sets:
        best = sorted(candidates, key=lambda x: x[score_key], reverse=True)[:3]

        if len(best) == 3:
            sets.append(
                {
                    "label": (
                        f"Auto fallback | "
                        f"A={best[0]['channel']}, "
                        f"B={best[1]['channel']}, "
                        f"C={best[2]['channel']}"
                    ),
                    "base": "AUTO_FALLBACK",
                    "set_index": 1,
                    "A": best[0]["channel"],
                    "B": best[1]["channel"],
                    "C": best[2]["channel"],
                    "score": sum(x[score_key] for x in best),
                    "kind": kind,
                }
            )

    sets = sorted(
        sets,
        key=lambda x: (
            -x["score"],
            x["base"],
            x["set_index"],
        )
    )

    return sets


def detect_phase_marker(channel_name: str):
    """
    Mendeteksi penanda phase:
    A/B/C, L1/L2/L3, R/S/T, 1/2/3.
    """

    original_name = get_original_channel_name(channel_name)
    n = normalize_name(original_name)

    # Prioritas eksplisit
    if n in ["VA", "UA", "IA"]:
        return "A"

    if n in ["VB", "UB", "IB"]:
        return "B"

    if n in ["VC", "UC", "IC"]:
        return "C"

    if "L1" in n:
        return "A"

    if "L2" in n:
        return "B"

    if "L3" in n:
        return "C"

    if n.endswith("A") or n.endswith("R") or n.endswith("1"):
        return "A"

    if n.endswith("B") or n.endswith("S") or n.endswith("2"):
        return "B"

    if n.endswith("C") or n.endswith("T") or n.endswith("3"):
        return "C"

    return None


def remove_phase_marker(channel_name: str):
    """
    Menghapus penanda phase untuk membuat nama dasar group.

    Contoh:
    A001 | IA -> I
    A002 | IB -> I
    A003 | IC -> I

    A005 | VA -> V
    A006 | VB -> V
    A007 | VC -> V
    """

    original_name = get_original_channel_name(channel_name)
    n = normalize_name(original_name)

    base = n

    # Bentuk umum
    for marker in ["L1", "L2", "L3"]:
        base = base.replace(marker, "")

    # Untuk VA/VB/VC dan IA/IB/IC
    if base.endswith("A") or base.endswith("B") or base.endswith("C"):
        base = base[:-1]

    if base.endswith("R") or base.endswith("S") or base.endswith("T"):
        base = base[:-1]

    if base.endswith("1") or base.endswith("2") or base.endswith("3"):
        base = base[:-1]

    if base == "":
        base = "UNKNOWN"

    return base


def build_channel_set_summary_dataframe(channel_sets: dict):
    """
    Membuat DataFrame ringkasan kandidat set channel.
    """

    rows = []

    for item in channel_sets.get("voltage_sets", []):
        rows.append(
            {
                "Type": "Voltage",
                "Base": item["base"],
                "A/L1": item["A"],
                "B/L2": item["B"],
                "C/L3": item["C"],
                "Score": item["score"],
            }
        )

    for item in channel_sets.get("current_sets", []):
        rows.append(
            {
                "Type": "Current",
                "Base": item["base"],
                "A/L1": item["A"],
                "B/L2": item["B"],
                "C/L3": item["C"],
                "Score": item["score"],
            }
        )

    for item in channel_sets.get("ground_candidates", []):
        rows.append(
            {
                "Type": "Ground",
                "Base": "-",
                "A/L1": item["channel"],
                "B/L2": "-",
                "C/L3": "-",
                "Score": item["score"],
            }
        )

    return pd.DataFrame(rows)