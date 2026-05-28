import os
import tempfile
from urllib.parse import quote

import pandas as pd
import streamlit as st

from comtrade_reader import read_comtrade
from conductor_impedance_importer import (
    read_google_spreadsheet_table,
    get_google_spreadsheet_sheet_names,
    extract_google_spreadsheet_id,
    make_unique_columns,
)


@st.cache_data(show_spinner="Membaca file COMTRADE...")
def read_comtrade_cached(cfg_bytes: bytes, dat_bytes: bytes, cfg_name: str = "", dat_name: str = ""):
    cfg_path = None
    dat_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".cfg") as temp_cfg:
            temp_cfg.write(cfg_bytes)
            cfg_path = temp_cfg.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".dat") as temp_dat:
            temp_dat.write(dat_bytes)
            dat_path = temp_dat.name

        return read_comtrade(cfg_path, dat_path)
    finally:
        for temp_path in [cfg_path, dat_path]:
            if temp_path:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass


@st.cache_data(ttl=1800, show_spinner="Membaca Google Spreadsheet...")
def read_google_spreadsheet_table_cached(url_or_id: str, sheet_name: str):
    return read_google_spreadsheet_table(url_or_id, sheet_name)


@st.cache_data(ttl=1800, show_spinner="Membaca Tower Schedule...")
def read_google_spreadsheet_query_cached(url_or_id: str, sheet_name: str, query: str):
    spreadsheet_id = extract_google_spreadsheet_id(url_or_id)
    csv_url = (
        f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/gviz/tq"
        f"?tqx=out:csv&sheet={quote(str(sheet_name))}&tq={quote(str(query))}"
    )
    df = pd.read_csv(csv_url)
    df.columns = make_unique_columns(df.columns)
    return df


@st.cache_data(ttl=1800, show_spinner="Membaca daftar sheet...")
def get_google_spreadsheet_sheet_names_cached(url_or_id: str):
    return get_google_spreadsheet_sheet_names(url_or_id)


def install_print_friendly_tables():
    """
    Streamlit dataframe memakai grid interaktif yang tidak ramah print.
    Wrapper ini mempertahankan tampilan interaktif di layar, lalu menambahkan
    tabel HTML khusus cetak dengan lebar kolom adaptif dan teks wrap.
    """

    if getattr(st, "_print_tables_installed", False):
        return

    original_dataframe = st.dataframe

    def _one_based_index(df):
        if not isinstance(df, pd.DataFrame) or df.empty:
            return df
        try:
            is_default_zero_index = list(df.index) == list(range(len(df.index)))
        except Exception:
            is_default_zero_index = False
        if not is_default_zero_index:
            return df

        adjusted_df = df.copy()
        adjusted_df.index = range(1, len(adjusted_df.index) + 1)
        adjusted_df.index.name = df.index.name or "No"
        return adjusted_df

    def _one_based_display_data(data):
        if type(data).__name__ == "Styler" and hasattr(data, "data"):
            source_df = data.data
            adjusted_df = _one_based_index(source_df)
            if adjusted_df is not source_df:
                data.data.index = adjusted_df.index
            return data

        if isinstance(data, pd.DataFrame):
            return _one_based_index(data)

        return data

    def _table_source(data):
        if type(data).__name__ == "Styler" and hasattr(data, "data"):
            data = _one_based_display_data(data)
            return data.data, data.to_html()

        if isinstance(data, pd.DataFrame):
            display_df = _one_based_index(data)
            return display_df, display_df.to_html(index=True, escape=True)

        try:
            df = pd.DataFrame(data)
            display_df = _one_based_index(df)
            return display_df, display_df.to_html(index=True, escape=True)
        except Exception:
            return None, None

    def printable_dataframe(data=None, *args, **kwargs):
        display_data = _one_based_display_data(data)
        result = original_dataframe(display_data, *args, **kwargs)
        source_df, table_html = _table_source(display_data)

        if source_df is not None and table_html:
            column_count = max(1, len(source_df.columns))
            row_count = len(source_df.index)
            density_class = "print-table-wide" if column_count >= 8 else "print-table-normal"
            if column_count >= 14:
                density_class = "print-table-ultrawide"

            st.markdown(
                f"""
                <div class="print-table-wrapper {density_class}"
                     data-print-columns="{column_count}"
                     data-print-rows="{row_count}">
                    {table_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

        return result

    st.dataframe = printable_dataframe
    st._print_tables_installed = True
