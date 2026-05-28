import cmath
import math

import pandas as pd
import streamlit as st


MAX_PLOT_POINTS = 6000


def downsample_xy(x_values, y_values, max_points: int = MAX_PLOT_POINTS):
    length = len(x_values)
    if length <= max_points:
        return x_values, y_values

    step = max(1, int(math.ceil(length / max_points)))
    return x_values[::step], y_values[::step]


def validate_uploaded_extension(uploaded_file, expected_extension: str, label: str):
    if uploaded_file is None:
        return True

    file_name = str(uploaded_file.name or "").lower()
    if file_name.endswith(expected_extension.lower()):
        return True

    st.error(
        f"{label} harus berekstensi `{expected_extension}`. "
        f"File yang dipilih: `{uploaded_file.name}`. "
        "Pada smartphone, simpan file ke Files/Downloads dan pastikan ekstensi tidak berubah."
    )
    return False


def downsample_dataframe_for_plot(df: pd.DataFrame, x_col: str, y_cols, max_points: int = MAX_PLOT_POINTS):
    if df is None or len(df) <= max_points:
        return df

    selected_cols = [x_col] + [col for col in y_cols if col in df.columns]
    step = max(1, int(math.ceil(len(df) / max_points)))
    return df.loc[df.index[::step], selected_cols]


def make_streamlit_safe_columns(df):
    """
    Membuat nama kolom DataFrame menjadi unik agar aman ditampilkan di Streamlit.
    """

    seen = {}
    new_columns = []

    for col in df.columns:
        col = str(col).strip()

        if col == "":
            col = "Unnamed"

        if col not in seen:
            seen[col] = 1
            new_columns.append(col)
        else:
            seen[col] += 1
            new_columns.append(f"{col}_{seen[col]}")

    safe_df = df.copy()
    safe_df.columns = new_columns

    return safe_df

def invert_current_phasors(phasors):
    inverted = {}

    for name, value in phasors.items():
        phasor_value = value["complex"]

        if name in ["Ia", "Ib", "Ic", "IE", "I0", "I1", "I2"]:
            phasor_value = -phasor_value

        inverted[name] = {
            "complex": phasor_value,
            "real": phasor_value.real,
            "imag": phasor_value.imag,
            "magnitude": abs(phasor_value),
            "angle_deg": math.degrees(cmath.phase(phasor_value)),
        }

    return inverted
