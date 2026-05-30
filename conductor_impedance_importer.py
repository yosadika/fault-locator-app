from pathlib import Path
import math
import re
from urllib.parse import quote
import pandas as pd


def read_conductor_impedance_excel(uploaded_file):
    """
    Membaca file Excel impedansi konduktor/saluran.

    Mendukung:
    - Header 1 baris biasa
    - Header 2 baris seperti tabel dengan grup:
      IMPEDANSI + (Ω/km) -> REAL, IMG, Ω/km abs, ANGLE rad
      IMPEDANSI 0 (Ω/km) -> REAL, IMG, Ω/km abs, ANGLE rad
      RATIO GI A -> CT, VT
      RATIO GI B -> CT, VT
    """

    excel = pd.ExcelFile(uploaded_file)
    sheets = {}

    for sheet_name in excel.sheet_names:
        df = read_sheet_smart(uploaded_file, sheet_name)
        sheets[sheet_name] = df

    return sheets


def read_sheet_smart(uploaded_file, sheet_name):
    """
    Membaca sheet dengan beberapa strategi header.
    """

    # Strategi 1: header 2 baris
    try:
        df_multi = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=[0, 1])
        df_multi = flatten_multi_header_dataframe(df_multi)

        if is_probably_valid_impedance_table(df_multi):
            return df_multi
    except Exception:
        pass

    # Strategi 2: header 1 baris
    try:
        df_single = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=0)
        df_single.columns = make_unique_columns(df_single.columns)

        if is_probably_valid_impedance_table(df_single):
            return df_single
    except Exception:
        pass

    # Strategi 3: cari baris header secara manual
    raw = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=None)

    header_row = detect_header_row(raw)

    if header_row is not None:
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=header_row)
        df.columns = make_unique_columns(df.columns)
        return df

    # fallback
    df = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=0)
    df.columns = make_unique_columns(df.columns)
    return df


def flatten_multi_header_dataframe(df):
    """
    Mengubah MultiIndex header menjadi satu baris kolom dan memastikan nama kolom unik.

    Contoh:
    ('IMPEDANSI + (Ω/km)', 'REAL') -> 'IMPEDANSI + (Ω/km) REAL'
    ('RATIO GI A', 'CT') -> 'RATIO GI A CT'
    """

    new_columns = []

    for col in df.columns:
        if isinstance(col, tuple):
            parts = []

            for item in col:
                text = str(item).strip()

                if text.lower().startswith("unnamed"):
                    continue

                if text and text != "nan":
                    parts.append(text)

            name = " ".join(parts)
        else:
            name = str(col)

        new_columns.append(clean_column_name(name))

    df.columns = make_unique_columns(new_columns)

    return df


def detect_header_row(raw_df):
    """
    Mendeteksi baris header jika Excel punya judul kosong di atas tabel.
    """

    keywords = [
        "GI",
        "BAY",
        "PHT",
        "PANJANG",
        "KONDUKTOR",
        "IMPEDANSI",
        "RATIO",
    ]

    for idx, row in raw_df.iterrows():
        row_text = " ".join([str(x).upper() for x in row.values if not pd.isna(x)])

        hit_count = sum(1 for k in keywords if k in row_text)

        if hit_count >= 3:
            return idx

    return None


def is_probably_valid_impedance_table(df):
    col_text = " ".join([str(c).upper() for c in df.columns])

    indicators = [
        "PANJANG",
        "KONDUKTOR",
        "IMPEDANSI",
        "REAL",
        "IMG",
        "RATIO",
        "BAY",
    ]

    hit = sum(1 for item in indicators if item in col_text)

    return hit >= 3


def clean_column_name(name):
    text = str(name).strip()
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)

    if text.lower().startswith("unnamed"):
        return ""

    return text


def make_unique_columns(columns):
    """
    Membuat nama kolom DataFrame menjadi unik.
    """

    seen = {}
    unique_columns = []

    for col in columns:
        col = clean_column_name(col)

        if col == "":
            col = "Unnamed"

        if col not in seen:
            seen[col] = 1
            unique_columns.append(col)
        else:
            seen[col] += 1
            unique_columns.append(f"{col}_{seen[col]}")

    return unique_columns


def normalize_column_name(name: str) -> str:
    text = str(name).lower().strip()
    text = text.replace(" ", "")
    text = text.replace("_", "")
    text = text.replace("-", "")
    text = text.replace("/", "")
    text = text.replace("\\", "")
    text = text.replace("(", "")
    text = text.replace(")", "")
    text = text.replace(".", "")
    text = text.replace(",", "")
    text = text.replace("°", "")
    text = text.replace("ω", "ohm")
    text = text.replace("Ω", "ohm")
    text = text.replace("+", "plus")
    return text


def find_column(df: pd.DataFrame, possible_names: list[str]):
    normalized_columns = {
        normalize_column_name(col): col
        for col in df.columns
        if str(col).strip() != ""
    }

    for name in possible_names:
        normalized_name = normalize_column_name(name)

        if normalized_name in normalized_columns:
            return normalized_columns[normalized_name]

    for col_norm, original_col in normalized_columns.items():
        for name in possible_names:
            normalized_name = normalize_column_name(name)

            if normalized_name and normalized_name in col_norm:
                return original_col

    return None


def detect_impedance_columns(df: pd.DataFrame):
    """
    Deteksi kolom untuk format Excel seperti contoh:
    GI | BAY PHT | PANJANG KONDUKTOR (km) | JENIS KONDUKTOR | ...
    IMPEDANSI + REAL/IMG
    IMPEDANSI 0 REAL/IMG
    RATIO GI A CT/VT
    RATIO GI B CT/VT
    """

    columns = list(df.columns)

    detected = {
        "upt": find_column(df, ["UPT"]),
        "ultg": find_column(df, ["ULTG"]),
        "gi": find_column(df, ["GI"]),
        "bay_pht": find_column(df, ["BAY PHT", "BAY", "PHT", "BAY_PHT"]),
        "line_number": find_column(df, ["LINE", "NO LINE", "SIRKIT", "CIRCUIT"]),
        "segment": find_column(df, ["SEGMENT", "NAMA SEGMENT", "NAMA SEGMEN", "SEGMEN"]),
        "line_name": find_column(
            df,
            [
                "SEGMENT",
                "NAMA SEGMENT",
                "NAMA SEGMEN",
                "SEGMEN",
                "LINE NAME",
                "NAMA SALURAN",
                "BAY PHT",
                "BAY",
                "PHT",
            ],
        ),
        "length": find_column(
            df,
            [
                "PANJANG KONDUKTOR (km)",
                "PANJANG KONDUKTOR",
                "PANJANG",
                "LENGTH",
                "LINE LENGTH",
                "KM",
            ],
        ),
        "conductor_type": find_column(
            df,
            [
                "JENIS",
                "JENIS KONDUKTOR",
                "KONDUKTOR",
                "CONDUCTOR",
                "CONDUCTOR TYPE",
                "TYPE",
            ],
        ),
        "circuit_count": find_column(
            df,
            [
                "JLH",
                "JLH SIRKIT",
                "JUMLAH SIRKIT",
                "SIRKIT",
                "CIRCUIT",
            ],
        ),
        "gia_name": find_column(df, ["GI A", "GIA"]),
        "gib_name": find_column(df, ["GI B", "GIB"]),
        "ratio_gia_ct": find_column(df, ["RATIO GI A CT", "GI A CT", "GIA CT", "CT GI A"]),
        "ratio_gia_vt": find_column(df, ["RATIO GI A VT", "GI A VT", "GIA VT", "VT GI A"]),
        "ratio_gib_ct": find_column(df, ["RATIO GI B CT", "GI B CT", "GIB CT", "CT GI B"]),
        "ratio_gib_vt": find_column(df, ["RATIO GI B VT", "GI B VT", "GIB VT", "VT GI B"]),
    }

    sequence_columns = detect_sequence_impedance_groups(columns)

    detected.update(sequence_columns)

    return detected


def detect_sequence_impedance_groups(columns):
    """
    Mendeteksi Z1 dan Z0 dari urutan grup kolom.

    Karena pada tabel contoh ada dua grup:
    IMPEDANSI + ... REAL IMG ...
    IMPEDANSI 0 ... REAL IMG ...

    Jika nama kolom hasil flatten tidak sempurna, fungsi ini tetap mencoba
    menemukan REAL/IMG berdasarkan posisi kolom.
    """

    norm_cols = [normalize_column_name(c) for c in columns]

    result = {
        "z1_real": None,
        "z1_imag": None,
        "z1_abs": None,
        "z1_angle": None,
        "z0_real": None,
        "z0_imag": None,
        "z0_abs": None,
        "z0_angle": None,
    }

    # Cari eksplisit dari nama kolom
    for col in columns:
        n = normalize_column_name(col)

        # Z1 / positive sequence
        if (
            ("impedansiplus" in n or "impedansiz1" in n or "z1" in n or "positive" in n)
            and ("real" in n or "r1" in n)
        ):
            result["z1_real"] = col

        if (
            ("impedansiplus" in n or "impedansiz1" in n or "z1" in n or "positive" in n)
            and ("img" in n or "imag" in n or "x1" in n)
        ):
            result["z1_imag"] = col

        if (
            ("impedansiplus" in n or "impedansiz1" in n or "z1" in n or "positive" in n)
            and ("abs" in n or "magnitude" in n or "ohmkmabs" in n)
        ):
            result["z1_abs"] = col

        if (
            ("impedansiplus" in n or "impedansiz1" in n or "z1" in n or "positive" in n)
            and ("angle" in n or "sudut" in n)
        ):
            result["z1_angle"] = col

        # Z0 / zero sequence
        if (
            ("impedansi0" in n or "impedansiz0" in n or "z0" in n or "zero" in n)
            and ("real" in n or "r0" in n)
        ):
            result["z0_real"] = col

        if (
            ("impedansi0" in n or "impedansiz0" in n or "z0" in n or "zero" in n)
            and ("img" in n or "imag" in n or "x0" in n)
        ):
            result["z0_imag"] = col

        if (
            ("impedansi0" in n or "impedansiz0" in n or "z0" in n or "zero" in n)
            and ("abs" in n or "magnitude" in n or "ohmkmabs" in n)
        ):
            result["z0_abs"] = col

        if (
            ("impedansi0" in n or "impedansiz0" in n or "z0" in n or "zero" in n)
            and ("angle" in n or "sudut" in n)
        ):
            result["z0_angle"] = col

    # Fallback berdasarkan posisi kolom seperti screenshot:
    # ... IMPEDANSI + REAL, IMG, complex, abs, angle,
    #     IMPEDANSI 0 REAL, IMG, complex, abs, angle
    real_img_cols = []

    for i, col in enumerate(columns):
        n = normalize_column_name(col)

        if n.endswith("real") or n == "real" or " real" in str(col).lower():
            real_img_cols.append(("real", i, col))

        if n.endswith("img") or n == "img" or " img" in str(col).lower():
            real_img_cols.append(("img", i, col))

    real_cols = [item for item in real_img_cols if item[0] == "real"]
    img_cols = [item for item in real_img_cols if item[0] == "img"]

    if result["z1_real"] is None and len(real_cols) >= 1:
        result["z1_real"] = real_cols[0][2]

    if result["z1_imag"] is None and len(img_cols) >= 1:
        result["z1_imag"] = img_cols[0][2]

    if result["z0_real"] is None and len(real_cols) >= 2:
        result["z0_real"] = real_cols[1][2]

    if result["z0_imag"] is None and len(img_cols) >= 2:
        result["z0_imag"] = img_cols[1][2]

    # Fallback untuk abs dan angle berdasarkan urutan kolom yang mengandung abs/angle
    abs_cols = []
    angle_cols = []

    for i, col in enumerate(columns):
        n = normalize_column_name(col)

        if "abs" in n or "ohmkmabs" in n:
            abs_cols.append((i, col))

        if "angle" in n:
            angle_cols.append((i, col))

    if result["z1_abs"] is None and len(abs_cols) >= 1:
        result["z1_abs"] = abs_cols[0][1]

    if result["z0_abs"] is None and len(abs_cols) >= 2:
        result["z0_abs"] = abs_cols[1][1]

    if result["z1_angle"] is None and len(angle_cols) >= 1:
        result["z1_angle"] = angle_cols[0][1]

    if result["z0_angle"] is None and len(angle_cols) >= 2:
        result["z0_angle"] = angle_cols[1][1]

    return result


def to_float(value):
    if pd.isna(value):
        return None

    text = str(value).strip()
    text = text.replace(",", ".")
    text = text.replace("Ω", "")
    text = text.replace("ohm", "")
    text = text.replace("Ohm", "")
    text = text.replace("deg", "")
    text = text.replace("°", "")

    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)

    if not match:
        return None

    try:
        return float(match.group(0))
    except Exception:
        return None


def parse_ratio(value):
    """
    Membaca ratio seperti:
    300/1
    150000/110
    150000/100

    Output:
    {
        primary,
        secondary,
        ratio
    }
    """

    if value is None or pd.isna(value):
        return None

    text = str(value).strip()
    text = text.replace(" ", "")

    if "/" in text:
        parts = text.split("/")

        if len(parts) >= 2:
            primary = to_float(parts[0])
            secondary = to_float(parts[1])

            if primary and secondary:
                return {
                    "primary": primary,
                    "secondary": secondary,
                    "ratio": primary / secondary,
                    "text": text,
                }

    number = to_float(text)

    if number:
        return {
            "primary": number,
            "secondary": 1.0,
            "ratio": number,
            "text": text,
        }

    return None


def polar_to_complex(magnitude, angle):
    """
    Angle diasumsikan radian jika nilainya kecil, derajat jika besar.

    Pada file Bapak terlihat ANGLE (rad), contohnya 1.2776.
    """

    if magnitude is None or angle is None:
        return None

    # Jika sudut <= 2*pi, anggap radian
    if abs(angle) <= 6.5:
        angle_rad = angle
    else:
        angle_rad = math.radians(angle)

    return complex(
        magnitude * math.cos(angle_rad),
        magnitude * math.sin(angle_rad),
    )


def extract_impedance_from_row(row, columns: dict):
    """
    Mengambil Z1 dan Z0 dari baris Excel.

    Prioritas:
    1. Z1 real/imag dan Z0 real/imag
    2. Z1 abs/angle dan Z0 abs/angle
    """

    z1 = None
    z0 = None

    if columns.get("z1_real") and columns.get("z1_imag"):
        r1 = to_float(row[columns["z1_real"]])
        x1 = to_float(row[columns["z1_imag"]])

        if r1 is not None and x1 is not None:
            z1 = complex(r1, x1)

    if columns.get("z0_real") and columns.get("z0_imag"):
        r0 = to_float(row[columns["z0_real"]])
        x0 = to_float(row[columns["z0_imag"]])

        if r0 is not None and x0 is not None:
            z0 = complex(r0, x0)

    if z1 is None and columns.get("z1_abs") and columns.get("z1_angle"):
        z1_abs = to_float(row[columns["z1_abs"]])
        z1_angle = to_float(row[columns["z1_angle"]])
        z1 = polar_to_complex(z1_abs, z1_angle)

    if z0 is None and columns.get("z0_abs") and columns.get("z0_angle"):
        z0_abs = to_float(row[columns["z0_abs"]])
        z0_angle = to_float(row[columns["z0_angle"]])
        z0 = polar_to_complex(z0_abs, z0_angle)

    if z1 is None:
        raise ValueError("Z1 tidak dapat dibaca dari baris Excel.")

    if z0 is None:
        raise ValueError("Z0 tidak dapat dibaca dari baris Excel.")

    line_name = None
    length = None

    upt = None
    ultg = None
    gi = None
    bay = None
    line_number = None
    segment = None
    gi_a = None
    gi_b = None
    conductor_type = None

    if columns.get("upt"):
        upt = str(row[columns["upt"]])

    if columns.get("ultg"):
        ultg = str(row[columns["ultg"]])

    if columns.get("gi"):
        gi = str(row[columns["gi"]])

    if columns.get("bay_pht"):
        bay = str(row[columns["bay_pht"]])

    if columns.get("line_number"):
        line_number = str(row[columns["line_number"]])

    if columns.get("segment"):
        segment = str(row[columns["segment"]])

    if columns.get("gia_name"):
        gi_a = str(row[columns["gia_name"]])

    if columns.get("gib_name"):
        gi_b = str(row[columns["gib_name"]])

    if columns.get("conductor_type"):
        conductor_type = str(row[columns["conductor_type"]])

    if columns.get("line_name"):
        line_name = str(row[columns["line_name"]])

    if (not gi_a or gi_a.lower() == "nan") and gi and gi.lower() != "nan":
        gi_a = gi

    if (not gi_b or gi_b.lower() == "nan") and bay and bay.lower() != "nan":
        gi_b = bay

    if not line_name or line_name.lower() == "nan":
        if segment and segment.lower() != "nan":
            line_name = segment
        elif gi_a and gi_b:
            line_name = f"{gi_a} - {gi_b}"
        elif bay:
            line_name = bay

    if columns.get("length"):
        length = to_float(row[columns["length"]])

    ratio_gia_ct = parse_ratio(row[columns["ratio_gia_ct"]]) if columns.get("ratio_gia_ct") else None
    ratio_gia_vt = parse_ratio(row[columns["ratio_gia_vt"]]) if columns.get("ratio_gia_vt") else None
    ratio_gib_ct = parse_ratio(row[columns["ratio_gib_ct"]]) if columns.get("ratio_gib_ct") else None
    ratio_gib_vt = parse_ratio(row[columns["ratio_gib_vt"]]) if columns.get("ratio_gib_vt") else None

    return {
        "upt": upt,
        "ultg": ultg,
        "gi": gi,
        "line_number": line_number,
        "segment": segment,
        "line_name": line_name,
        "bay_pht": bay,
        "gi_a": gi_a,
        "gi_b": gi_b,
        "conductor_type": conductor_type,
        "length": length,
        "Z1": z1,
        "Z0": z0,
        "R1": z1.real,
        "X1": z1.imag,
        "R0": z0.real,
        "X0": z0.imag,
        "Z1_abs": abs(z1),
        "Z1_angle_deg": math.degrees(math.atan2(z1.imag, z1.real)),
        "Z0_abs": abs(z0),
        "Z0_angle_deg": math.degrees(math.atan2(z0.imag, z0.real)),
        "ratio_gia_ct": ratio_gia_ct,
        "ratio_gia_vt": ratio_gia_vt,
        "ratio_gib_ct": ratio_gib_ct,
        "ratio_gib_vt": ratio_gib_vt,
    }


def build_row_label(df: pd.DataFrame, columns: dict):
    labels = []

    bay_col = columns.get("bay_pht")
    gi_col = columns.get("gi")
    segment_col = columns.get("segment")
    line_name_col = columns.get("line_name")
    line_number_col = columns.get("line_number")
    gia_col = columns.get("gia_name")
    gib_col = columns.get("gib_name")
    length_col = columns.get("length")
    conductor_col = columns.get("conductor_type")
    circuit_col = columns.get("circuit_count")

    for idx, row in df.iterrows():
        parts = [f"Row {idx + 1}"]

        primary_name = None

        if segment_col:
            primary_name = row[segment_col]
        elif line_name_col:
            primary_name = row[line_name_col]

        if primary_name is not None and str(primary_name).lower() != "nan":
            parts.append(str(primary_name))

        if line_number_col:
            parts.append(f"Line {row[line_number_col]}")

        if gi_col and bay_col:
            parts.append(f"{row[gi_col]} - {row[bay_col]}")
        elif bay_col:
            parts.append(f"BAY={row[bay_col]}")

        if gia_col and gib_col:
            parts.append(f"{row[gia_col]} - {row[gib_col]}")

        if length_col:
            parts.append(f"L={row[length_col]} km")

        if conductor_col:
            parts.append(str(row[conductor_col]))

        if circuit_col:
            parts.append(f"Sirkit={row[circuit_col]}")

        labels.append(" | ".join(parts))

    return labels


def read_conductor_impedance_database(
    file_path: str = "database/line_data.xlsx",
    sheet_name: str = "line_impedance",
):
    """
    Membaca database impedansi saluran dari file Excel lokal project.

    Default:
    database/line_data.xlsx
    sheet: line_impedance
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"File database tidak ditemukan: {path.resolve()}"
        )

    df = read_sheet_smart(path, sheet_name)

    return df


def extract_google_spreadsheet_id(url_or_id: str):
    text = str(url_or_id or "").strip()

    if not text:
        raise ValueError("URL/ID Google Spreadsheet kosong.")

    match = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", text)

    if match:
        return match.group(1)

    if re.fullmatch(r"[a-zA-Z0-9-_]+", text):
        return text

    raise ValueError("URL Google Spreadsheet tidak valid.")


def google_spreadsheet_csv_url(url_or_id: str, sheet_name: str):
    spreadsheet_id = extract_google_spreadsheet_id(url_or_id)
    sheet = quote(str(sheet_name or "").strip())

    if not sheet:
        raise ValueError("Nama sheet Google Spreadsheet kosong.")

    return (
        f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/gviz/tq"
        f"?tqx=out:csv&sheet={sheet}"
    )


def google_spreadsheet_metadata_url(url_or_id: str):
    spreadsheet_id = extract_google_spreadsheet_id(url_or_id)
    return f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/gviz/tq?tqx=out:json"


def get_google_spreadsheet_sheet_names(url_or_id: str):
    metadata_url = google_spreadsheet_metadata_url(url_or_id)
    import urllib.request

    with urllib.request.urlopen(metadata_url, timeout=15) as response:
        raw = response.read().decode("utf-8", errors="replace")

    matches = re.findall(r'"name"\s*:\s*"([^"]+)"', raw)
    return list(dict.fromkeys(matches))


def read_google_spreadsheet_table(url_or_id: str, sheet_name: str):
    csv_url = google_spreadsheet_csv_url(url_or_id, sheet_name)
    df = pd.read_csv(csv_url)
    df.columns = make_unique_columns(df.columns)
    return df
