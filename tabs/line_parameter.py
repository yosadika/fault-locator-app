import math

import pandas as pd
import streamlit as st

from app_helpers import make_streamlit_safe_columns
from app_runtime import read_google_spreadsheet_table_cached
from conductor_impedance_importer import (
    build_row_label,
    detect_impedance_columns,
    extract_impedance_from_row,
)
from line_parameter import (
    build_line_parameter_dataframe,
    normalize_line_parameter,
)


def render():
    st.subheader("Line Parameter Input & Converter")

    st.write(
        "Pilih sumber parameter saluran, lalu normalisasi menjadi Z1_per_km, "
        "Z0_per_km, K0, Z1_total, dan Z0_total."
    )

    st.markdown("### Sumber Data Parameter")

    line_parameter_source = st.radio(
        "Pilih sumber parameter saluran",
        [
            "Input Manual",
            "Database Excel Line Data",
            "Database Excel Cable Data",
        ],
        horizontal=True,
        key="line_parameter_source",
    )

    excel_impedance_data = None
    local_transformer_for_line = (
        st.session_state.get("local_transformer_data")
        or st.session_state.get("auto_transformer_data", {})
    )
    transformer_ratio_source = (
        "Signal Assignment"
        if st.session_state.get("local_transformer_data")
        else "metadata CFG otomatis"
    )
    assignment_ct_primary = float(local_transformer_for_line.get("ct_primary", 800.0))
    assignment_ct_secondary = float(local_transformer_for_line.get("ct_secondary", 1.0))
    assignment_vt_primary = float(local_transformer_for_line.get("vt_primary", 150000.0))
    assignment_vt_secondary = float(local_transformer_for_line.get("vt_secondary", 100.0))

    def is_filled(value):
        if value is None:
            return False
        if isinstance(value, float) and math.isnan(value):
            return False
        if isinstance(value, str) and value.strip().lower() in ["", "nan", "none", "null"]:
            return False
        return True

    if (
        line_parameter_source in ["Database Excel Line Data", "Database Excel Cable Data"]
        and not str(st.session_state.get("database_spreadsheet_url", "") or "").strip()
    ):
        st.warning(
            "Link Database Spreadsheet belum diatur. Buka tab Setup DB lalu isi "
            "`Database Spreadsheet URL` atau upload runtime credentials terlebih dahulu. "
            "Sementara gunakan `Input Manual` untuk mengisi parameter saluran."
        )
        line_parameter_source = "Input Manual"

    if line_parameter_source in ["Database Excel Line Data", "Database Excel Cable Data"]:
        use_cable_database = line_parameter_source == "Database Excel Cable Data"
        database_source_key = "cable_data" if use_cable_database else "line_data"
        database_spreadsheet_url = st.session_state.get(
            f"{database_source_key}_spreadsheet_url",
            st.session_state.get("database_spreadsheet_url", ""),
        )
        database_spreadsheet_url = str(database_spreadsheet_url or "").strip()
        database_sheet_name = st.session_state.get(
            f"{database_source_key}_sheet_name",
            "cable_impedance" if use_cable_database else "line_impedance",
        )
        database_title = (
            "Database Spreadsheet Data Impedansi Konduktor"
            if use_cable_database
            else "Database Spreadsheet Data Impedansi Saluran"
        )
        database_preview_title = (
            "Preview Database Cable Impedance"
            if use_cable_database
            else "Preview Database Line Impedance"
        )
        selected_row_label_text = (
            "Pilih jenis konduktor dari cable_data spreadsheet"
            if use_cable_database
            else "Pilih baris data saluran / BAY PHT"
        )

        try:
            conductor_df = read_google_spreadsheet_table_cached(
                database_spreadsheet_url,
                database_sheet_name,
            )

            conductor_df = make_streamlit_safe_columns(conductor_df)

            if use_cable_database:
                detected_for_unique = detect_impedance_columns(conductor_df)
                unique_subset = [
                    detected_for_unique.get(key)
                    for key in [
                        "conductor_type",
                        "circuit_count",
                        "z1_real",
                        "z1_imag",
                        "z0_real",
                        "z0_imag",
                        "z1_abs",
                        "z1_angle",
                        "z0_abs",
                        "z0_angle",
                    ]
                    if detected_for_unique.get(key)
                ]

                if unique_subset:
                    conductor_df = (
                        conductor_df
                        .drop_duplicates(subset=unique_subset)
                        .reset_index(drop=True)
                    )

            st.session_state["line_database_df"] = conductor_df

            detected_columns = detect_impedance_columns(conductor_df)

            with st.expander("Advanced Database Detail", expanded=False):
                st.markdown(f"#### {database_title}")
                st.caption(
                    "Aplikasi membaca data impedansi dari Google Spreadsheet yang dikonfigurasi "
                    f"di tab Spreadsheet Config. Sheet aktif: `{database_sheet_name}`."
                )
                st.caption(f"Spreadsheet URL: {database_spreadsheet_url}")
                if use_cable_database:
                    st.warning(
                        "Gunakan opsi ini jika nama line tidak tersedia di line_data spreadsheet. "
                        "Aplikasi mengambil Z1/Z0 per km dari data konduktor. "
                        "Jika spreadsheet tidak memiliki panjang saluran, aplikasi memakai nilai fallback."
                    )
                st.markdown(f"#### {database_preview_title}")
                st.dataframe(conductor_df, use_container_width=True, height=300)
                st.markdown("#### Kolom yang Terdeteksi Otomatis")
                st.json(detected_columns)

            line_name_col = detected_columns.get("line_name")
            length_col = detected_columns.get("length")
            bay_pht_col = detected_columns.get("bay_pht")
            z1_real_col = detected_columns.get("z1_real")
            z1_imag_col = detected_columns.get("z1_imag")
            z1_abs_col = detected_columns.get("z1_abs")
            z1_angle_col = detected_columns.get("z1_angle")
            z0_real_col = detected_columns.get("z0_real")
            z0_imag_col = detected_columns.get("z0_imag")
            z0_abs_col = detected_columns.get("z0_abs")
            z0_angle_col = detected_columns.get("z0_angle")
            ratio_gia_ct_col = detected_columns.get("ratio_gia_ct")
            ratio_gia_vt_col = detected_columns.get("ratio_gia_vt")
            ratio_gib_ct_col = detected_columns.get("ratio_gib_ct")
            ratio_gib_vt_col = detected_columns.get("ratio_gib_vt")

            corrected_columns = {
                "upt": None if use_cable_database else detected_columns.get("upt"),
                "ultg": None if use_cable_database else detected_columns.get("ultg"),
                "gi": None if use_cable_database else detected_columns.get("gi"),
                "line_number": None if use_cable_database else detected_columns.get("line_number"),
                "segment": None if use_cable_database else detected_columns.get("segment"),
                "line_name": None if line_name_col == "None" else line_name_col,
                "bay_pht": None if bay_pht_col == "None" else bay_pht_col,
                "length": None if length_col == "None" else length_col,
                "z1_real": None if z1_real_col == "None" else z1_real_col,
                "z1_imag": None if z1_imag_col == "None" else z1_imag_col,
                "z1_abs": None if z1_abs_col == "None" else z1_abs_col,
                "z1_angle": None if z1_angle_col == "None" else z1_angle_col,
                "z0_real": None if z0_real_col == "None" else z0_real_col,
                "z0_imag": None if z0_imag_col == "None" else z0_imag_col,
                "z0_abs": None if z0_abs_col == "None" else z0_abs_col,
                "z0_angle": None if z0_angle_col == "None" else z0_angle_col,
                "ratio_gia_ct": None if ratio_gia_ct_col == "None" else ratio_gia_ct_col,
                "ratio_gia_vt": None if ratio_gia_vt_col == "None" else ratio_gia_vt_col,
                "ratio_gib_ct": None if ratio_gib_ct_col == "None" else ratio_gib_ct_col,
                "ratio_gib_vt": None if ratio_gib_vt_col == "None" else ratio_gib_vt_col,
                "gia_name": None if use_cable_database else detected_columns.get("gia_name"),
                "gib_name": None if use_cable_database else detected_columns.get("gib_name"),
                "conductor_type": detected_columns.get("conductor_type"),
                "circuit_count": detected_columns.get("circuit_count"),
            }

            if use_cable_database:
                corrected_columns["line_name"] = None
                corrected_columns["bay_pht"] = None
                corrected_columns["length"] = None
                corrected_columns["ratio_gia_ct"] = None
                corrected_columns["ratio_gia_vt"] = None
                corrected_columns["ratio_gib_ct"] = None
                corrected_columns["ratio_gib_vt"] = None

            if not use_cable_database:
                _f_ultg   = st.session_state.get("sidebar_filter_ultg", "")
                _f_seg    = st.session_state.get("sidebar_filter_segment", "")
                _f_gi_l   = st.session_state.get("sidebar_filter_gi_local", "")
                _f_bay_l  = st.session_state.get("sidebar_filter_bay_local", "")
                _f_line_l = st.session_state.get("sidebar_filter_line_local", "")
                _f_gi_r   = st.session_state.get("sidebar_filter_gi_remote", "")
                _f_bay_r  = st.session_state.get("sidebar_filter_bay_remote", "")
                _f_line_r = st.session_state.get("sidebar_filter_line_remote", "")

                def _ia(v):
                    return bool(v) and v != "Semua" and not v.startswith("Pilih ")

                def _nv(v):
                    try:
                        f = float(v)
                        if f == int(f):
                            return str(int(f))
                    except (ValueError, TypeError):
                        pass
                    return str(v)

                def _col_eq(series, val):
                    return series.astype(str).str.strip().apply(_nv) == _nv(val)

                _filter_active = any(_ia(v) for v in [
                    _f_ultg, _f_seg, _f_gi_l, _f_bay_l, _f_line_l,
                    _f_gi_r, _f_bay_r, _f_line_r,
                ])
                if _filter_active:
                    _mask = pd.Series([True] * len(conductor_df), index=conductor_df.index)
                    _col_ultg = corrected_columns.get("ultg")
                    _col_seg  = corrected_columns.get("segment")
                    _col_gi   = corrected_columns.get("gi")
                    _col_bay  = corrected_columns.get("bay_pht")
                    _col_line = corrected_columns.get("line_number")

                    if _ia(_f_ultg) and _col_ultg and _col_ultg in conductor_df.columns:
                        _mask &= _col_eq(conductor_df[_col_ultg], _f_ultg)
                    if _ia(_f_seg) and _col_seg and _col_seg in conductor_df.columns:
                        _mask &= _col_eq(conductor_df[_col_seg], _f_seg)

                    if _col_gi and _col_gi in conductor_df.columns:
                        _has_local  = _ia(_f_gi_l) or _ia(_f_bay_l) or _ia(_f_line_l)
                        _has_remote = _ia(_f_gi_r) or _ia(_f_bay_r) or _ia(_f_line_r)
                        if _has_local or _has_remote:
                            _gi_s   = conductor_df[_col_gi].astype(str).str.strip()
                            _bay_s  = conductor_df[_col_bay].astype(str).str.strip() \
                                if (_col_bay and _col_bay in conductor_df.columns) else None
                            _line_s = conductor_df[_col_line].astype(str).str.strip().apply(_nv) \
                                if (_col_line and _col_line in conductor_df.columns) else None

                            _local_m = pd.Series([True] * len(conductor_df), index=conductor_df.index)
                            if _ia(_f_gi_l):
                                _local_m &= _gi_s == _f_gi_l
                            if _ia(_f_bay_l) and _bay_s is not None:
                                _local_m &= _bay_s == _f_bay_l
                            if _ia(_f_line_l) and _line_s is not None:
                                _local_m &= _line_s == _nv(_f_line_l)

                            _remote_m = pd.Series([True] * len(conductor_df), index=conductor_df.index)
                            if _ia(_f_gi_r):
                                _remote_m &= _gi_s == _f_gi_r
                            if _ia(_f_bay_r) and _bay_s is not None:
                                _remote_m &= _bay_s == _f_bay_r
                            if _ia(_f_line_r) and _line_s is not None:
                                _remote_m &= _line_s == _nv(_f_line_r)

                            if _has_local and _has_remote:
                                _mask &= _local_m | _remote_m
                            elif _has_local:
                                _mask &= _local_m
                            else:
                                _mask &= _remote_m

                    conductor_df = conductor_df[_mask].reset_index(drop=True)
                    _active_parts = [
                        f"ULTG={_f_ultg}" if _ia(_f_ultg) else None,
                        f"Segment={_f_seg}" if _ia(_f_seg) else None,
                        f"GI Lokal={_f_gi_l}" if _ia(_f_gi_l) else None,
                        f"Bay Lokal={_f_bay_l}" if _ia(_f_bay_l) else None,
                        f"Line Lokal={_f_line_l}" if _ia(_f_line_l) else None,
                        f"GI Remote={_f_gi_r}" if _ia(_f_gi_r) else None,
                        f"Bay Remote={_f_bay_r}" if _ia(_f_bay_r) else None,
                        f"Line Remote={_f_line_r}" if _ia(_f_line_r) else None,
                    ]
                    st.info(
                        "Filter sidebar aktif — "
                        + ", ".join(x for x in _active_parts if x)
                        + f". {len(conductor_df)} baris tersedia. "
                        "Ubah di expander Filter GI / Line pada sidebar."
                    )
                    if conductor_df.empty:
                        st.warning("Tidak ada data yang cocok dengan filter sidebar. Ubah filter di sidebar.")

            row_labels = build_row_label(conductor_df, corrected_columns)

            selected_row_label = st.selectbox(
                selected_row_label_text,
                row_labels,
                key=(
                    "selected_database_cable_row"
                    if use_cable_database
                    else "selected_database_line_row"
                ),
            )

            selected_row_index = row_labels.index(selected_row_label)
            selected_row = conductor_df.iloc[selected_row_index]

            excel_impedance_data = extract_impedance_from_row(
                selected_row,
                corrected_columns,
            )

            st.session_state["excel_impedance_data"] = excel_impedance_data
            st.session_state["excel_impedance_source"] = line_parameter_source

            with st.expander("Detail data impedansi yang dipilih", expanded=False):
                st.json(
                    {
                        "ultg": excel_impedance_data.get("ultg"),
                        "gi": excel_impedance_data.get("gi"),
                        "line_number": excel_impedance_data.get("line_number"),
                        "segment": excel_impedance_data.get("segment"),
                        "line_name": excel_impedance_data["line_name"],
                        "bay_pht": excel_impedance_data["bay_pht"],
                        "gi_a": excel_impedance_data["gi_a"],
                        "gi_b": excel_impedance_data["gi_b"],
                        "conductor_type": excel_impedance_data["conductor_type"],
                        "length": excel_impedance_data["length"],
                        "R1": excel_impedance_data["R1"],
                        "X1": excel_impedance_data["X1"],
                        "R0": excel_impedance_data["R0"],
                        "X0": excel_impedance_data["X0"],
                        "Z1_abs": excel_impedance_data["Z1_abs"],
                        "Z1_angle_deg": excel_impedance_data["Z1_angle_deg"],
                        "Z0_abs": excel_impedance_data["Z0_abs"],
                        "Z0_angle_deg": excel_impedance_data["Z0_angle_deg"],
                        "ratio_gia_ct": excel_impedance_data["ratio_gia_ct"],
                        "ratio_gia_vt": excel_impedance_data["ratio_gia_vt"],
                        "ratio_gib_ct": excel_impedance_data["ratio_gib_ct"],
                        "ratio_gib_vt": excel_impedance_data["ratio_gib_vt"],
                    }
                )

            if "excel_ratio_side" not in st.session_state:
                st.session_state["excel_ratio_side"] = "Tidak gunakan dari Excel"

            st.session_state["excel_ratio_side"] = "Tidak gunakan dari Excel"

        except Exception as e:
            st.error("Gagal membaca database spreadsheet.")
            st.exception(e)
            st.caption(
                "Pastikan URL benar, spreadsheet dapat diakses publik, dan nama sheet sesuai. "
                "Ubah konfigurasi di tab Spreadsheet Config."
            )

    if (
        line_parameter_source != "Input Manual"
        and st.session_state.get("excel_impedance_source") == line_parameter_source
        and "excel_impedance_data" in st.session_state
    ):
        excel_impedance_data = st.session_state["excel_impedance_data"]

    if line_parameter_source != "Input Manual" and excel_impedance_data:
        spreadsheet_line_name = (
            excel_impedance_data.get("line_name")
            or excel_impedance_data.get("bay_pht")
        )
        spreadsheet_length = excel_impedance_data.get("length")

        if not is_filled(spreadsheet_line_name) or not is_filled(spreadsheet_length):
            st.markdown("### Lengkapi Data Saluran")
            st.caption(
                "Data impedansi diambil dari spreadsheet. Field di bawah hanya muncul "
                "karena belum tersedia pada baris data yang dipilih."
            )

        if is_filled(spreadsheet_line_name):
            line_name = str(spreadsheet_line_name)
        else:
            default_missing_line_name = (
                st.session_state.get("line_param", {}).get("line_name")
                or f"Line baru - {excel_impedance_data.get('conductor_type', 'Database')}"
            )
            line_name = st.text_input(
                "Line Name",
                value=str(default_missing_line_name),
                key="spreadsheet_missing_line_name",
                help="Nama line belum tersedia pada spreadsheet cable_data, jadi perlu diisi manual.",
            )

        if is_filled(spreadsheet_length):
            line_length = float(spreadsheet_length)
            length_unit = "km"
        else:
            default_missing_length = float(
                st.session_state.get("line_param", {}).get("length_km", 75.0)
                if st.session_state.get("line_param")
                else 75.0
            )
            col_missing_length, col_missing_unit = st.columns([2, 1])
            with col_missing_length:
                line_length = st.number_input(
                    "Line Length",
                    value=default_missing_length,
                    min_value=0.001,
                    step=0.001,
                    format="%.5f",
                    key="spreadsheet_missing_line_length",
                    help="Panjang saluran belum tersedia pada spreadsheet cable_data, jadi perlu diisi manual.",
                )
            with col_missing_unit:
                length_unit = st.selectbox(
                    "Length Unit",
                    ["km", "miles"],
                    index=0,
                    key="spreadsheet_missing_length_unit",
                )

        impedance_input = "relative"
        base_side = "primary"
        lp_ct_primary = assignment_ct_primary
        lp_ct_secondary = assignment_ct_secondary
        lp_vt_primary = assignment_vt_primary
        lp_vt_secondary = assignment_vt_secondary
        positive_sequence_mode = "R_X"
        r1 = float(excel_impedance_data.get("R1") or 0.08)
        x1 = float(excel_impedance_data.get("X1") or 0.42)
        z1_mag = None
        phi1_deg = None
        zero_sequence_mode = "R0_X0"
        r0 = float(excel_impedance_data.get("R0") or 0.25)
        x0 = float(excel_impedance_data.get("X0") or 1.25)
        re_rl = None
        xe_xl = None
        z0_z1_mag = None
        z0_z1_angle_deg = None
        kl_mag = None
        kl_angle_deg = None
        st.info(
            "Parameter saluran akan dinormalisasi dari data spreadsheet yang dipilih. "
            f"Ratio CT/VT memakai data dari {transformer_ratio_source} lokal: "
            f"CT {lp_ct_primary:g}/{lp_ct_secondary:g}, "
            f"VT {lp_vt_primary:g}/{lp_vt_secondary:g}."
        )
    else:
        st.markdown("### Basic Line Data")

        col_lp1, col_lp2, col_lp3 = st.columns(3)

        with col_lp1:
            default_line_name = "SUTT 150 kV GI A - GI B"

            if excel_impedance_data and excel_impedance_data.get("line_name"):
                default_line_name = str(excel_impedance_data["line_name"])
            elif (
                line_parameter_source == "Database Excel Cable Data"
                and excel_impedance_data
                and excel_impedance_data.get("conductor_type")
            ):
                default_line_name = (
                    f"Line baru - {excel_impedance_data['conductor_type']}"
                )

            line_name = st.text_input(
                "Line Name",
                value=default_line_name,
            )

        with col_lp2:
            default_line_length = 75.0

            if excel_impedance_data and excel_impedance_data.get("length"):
                default_line_length = float(excel_impedance_data["length"])

            line_length = st.number_input(
                "Line Length",
                value=default_line_length,
                min_value=0.001,
                step=0.001,
                format="%.5f",
            )

        with col_lp3:
            length_unit = st.selectbox(
                "Length Unit",
                ["km", "miles"],
                index=0,
            )

        st.markdown("### Input Format")

        col_fmt1, col_fmt2 = st.columns(2)

        with col_fmt1:
            default_impedance_input_index = 0

            impedance_input = st.radio(
                "Impedance Input",
                ["relative", "absolute"],
                index=default_impedance_input_index,
                horizontal=True,
                help="Relative = ohm/km atau ohm/mile. Absolute = total ohm saluran.",
            )

        with col_fmt2:
            base_side = st.radio(
                "Base Side",
                ["primary", "secondary"],
                horizontal=True,
                help="Primary jika impedansi dalam ohm primer. Secondary jika dari setting relay.",
            )

        st.markdown("### Transformer Ratio for Secondary Impedance Conversion")

        default_lp_ct_primary = assignment_ct_primary
        default_lp_ct_secondary = assignment_ct_secondary
        default_lp_vt_primary = assignment_vt_primary
        default_lp_vt_secondary = assignment_vt_secondary

        if excel_impedance_data:
            ratio_side = st.session_state.get("excel_ratio_side", "Tidak gunakan dari Excel")

            if ratio_side == "GI A":
                ct_data = excel_impedance_data.get("ratio_gia_ct")
                vt_data = excel_impedance_data.get("ratio_gia_vt")

                if ct_data:
                    default_lp_ct_primary = float(ct_data["primary"])
                    default_lp_ct_secondary = float(ct_data["secondary"])

                if vt_data:
                    default_lp_vt_primary = float(vt_data["primary"])
                    default_lp_vt_secondary = float(vt_data["secondary"])

            elif ratio_side == "GI B":
                ct_data = excel_impedance_data.get("ratio_gib_ct")
                vt_data = excel_impedance_data.get("ratio_gib_vt")

                if ct_data:
                    default_lp_ct_primary = float(ct_data["primary"])
                    default_lp_ct_secondary = float(ct_data["secondary"])

                if vt_data:
                    default_lp_vt_primary = float(vt_data["primary"])
                    default_lp_vt_secondary = float(vt_data["secondary"])

        col_tr1, col_tr2, col_tr3, col_tr4 = st.columns(4)

        with col_tr1:
            lp_ct_primary = st.number_input(
                "Line CT Primary (A)",
                value=default_lp_ct_primary,
                min_value=0.001,
                step=0.001,
                format="%.5f",
            )

        with col_tr2:
            lp_ct_secondary = st.number_input(
                "Line CT Secondary (A)",
                value=default_lp_ct_secondary,
                min_value=0.001,
                step=0.001,
                format="%.5f",
            )

        with col_tr3:
            lp_vt_primary = st.number_input(
                "Line VT Primary (V)",
                value=default_lp_vt_primary,
                min_value=0.001,
                step=0.001,
                format="%.5f",
            )

        with col_tr4:
            lp_vt_secondary = st.number_input(
                "Line VT Secondary (V)",
                value=default_lp_vt_secondary,
                min_value=0.001,
                step=0.001,
                format="%.5f",
            )

        st.markdown("### Positive-Sequence System")

        if excel_impedance_data:
            positive_sequence_default_index = 0  # R_X
        else:
            positive_sequence_default_index = 0

        positive_sequence_mode = st.selectbox(
            "Positive-Sequence Input Mode",
            ["R_X", "Z_PHI", "X_PHI"],
            index=positive_sequence_default_index,
        )

        r1 = None
        x1 = None
        z1_mag = None
        phi1_deg = None

        if positive_sequence_mode == "R_X":
            col_z1a, col_z1b = st.columns(2)

            with col_z1a:
                default_r1 = 0.08
                default_x1 = 0.42

                if excel_impedance_data:
                    default_r1 = float(excel_impedance_data["R1"])
                    default_x1 = float(excel_impedance_data["X1"])

                r1 = st.number_input(
                    "R1 / R1' (ohm or ohm/km)",
                    value=default_r1,
                    step=0.00001,
                    format="%.5f",
                )

            with col_z1b:
                x1 = st.number_input(
                    "X1 / X1' (ohm or ohm/km)",
                    value=default_x1,
                    step=0.00001,
                    format="%.5f",
                )

        elif positive_sequence_mode == "Z_PHI":
            col_z1a, col_z1b = st.columns(2)

            with col_z1a:
                z1_mag = st.number_input(
                    "Z1 Magnitude",
                    value=0.4275,
                    step=0.00001,
                    format="%.5f",
                )

            with col_z1b:
                phi1_deg = st.number_input(
                    "Phi1 Angle (deg)",
                    value=79.22,
                    step=0.00001,
                    format="%.5f",
                )

        elif positive_sequence_mode == "X_PHI":
            col_z1a, col_z1b = st.columns(2)

            with col_z1a:
                x1 = st.number_input(
                    "X1 / X1' (ohm or ohm/km)",
                    value=0.42,
                    step=0.00001,
                    format="%.5f",
                )

            with col_z1b:
                phi1_deg = st.number_input(
                    "Phi1 Angle (deg)",
                    value=79.22,
                    step=0.00001,
                    format="%.5f",
                )

        st.markdown("### Zero-Sequence System")

        zero_sequence_mode = st.selectbox(
            "Zero-Sequence Input Mode",
            ["R0_X0", "RE_RL_XE_XL", "Z0_Z1", "KL"],
            index=0,
        )

        r0 = None
        x0 = None
        re_rl = None
        xe_xl = None
        z0_z1_mag = None
        z0_z1_angle_deg = None
        kl_mag = None
        kl_angle_deg = None

        if zero_sequence_mode == "R0_X0":
            col_z0a, col_z0b = st.columns(2)

            with col_z0a:
                default_r0 = 0.25
                default_x0 = 1.25

                if excel_impedance_data:
                    default_r0 = float(excel_impedance_data["R0"])
                    default_x0 = float(excel_impedance_data["X0"])

                r0 = st.number_input(
                    "R0 / R0' (ohm or ohm/km)",
                    value=default_r0,
                    step=0.00001,
                    format="%.5f",
                )

            with col_z0b:
                x0 = st.number_input(
                    "X0 / X0' (ohm or ohm/km)",
                    value=default_x0,
                    step=0.00001,
                    format="%.5f",
                )

        elif zero_sequence_mode == "RE_RL_XE_XL":
            col_z0a, col_z0b = st.columns(2)

            with col_z0a:
                re_rl = st.number_input(
                    "RE/RL",
                    value=3.125,
                    step=0.00001,
                    format="%.5f",
                )

            with col_z0b:
                xe_xl = st.number_input(
                    "XE/XL",
                    value=2.976,
                    step=0.00001,
                    format="%.5f",
                )

            st.info(
                "Catatan: RE/RL dan XE/XL digunakan untuk membentuk Z0 dari Z1. "
                "Pastikan definisinya sesuai dengan setting relay yang digunakan."
            )

        elif zero_sequence_mode == "Z0_Z1":
            col_z0a, col_z0b = st.columns(2)

            with col_z0a:
                z0_z1_mag = st.number_input(
                    "Z0/Z1 Magnitude",
                    value=3.0,
                    step=0.00001,
                    format="%.5f",
                )

            with col_z0b:
                z0_z1_angle_deg = st.number_input(
                    "Z0/Z1 Angle (deg)",
                    value=0.0,
                    step=0.00001,
                    format="%.5f",
                )

        elif zero_sequence_mode == "KL":
            col_z0a, col_z0b = st.columns(2)

            with col_z0a:
                kl_mag = st.number_input(
                    "kL Magnitude",
                    value=0.70,
                    step=0.00001,
                    format="%.5f",
                )

            with col_z0b:
                kl_angle_deg = st.number_input(
                    "kL Angle (deg)",
                    value=0.0,
                    step=0.00001,
                    format="%.5f",
                )

            st.warning(
                "Catatan: pada aplikasi ini kL diasumsikan sebagai ZE/Z1, "
                "dengan ZE = (Z0 - Z1) / 3. Pastikan definisi ini sesuai dengan setting relay."
            )

        if excel_impedance_data:
            st.success(
                "Parameter Z1 dan Z0 menggunakan data dari Excel impedansi konduktor/saluran."
            )
        else:
            st.info(
                "Parameter Z1 dan Z0 menggunakan input manual."
            )

    st.markdown("### Normalize Parameter")

    if st.button("Normalize Line Parameter"):
        try:
            line_param = normalize_line_parameter(
                line_name=line_name,
                length=line_length,
                length_unit=length_unit,
                impedance_input=impedance_input,
                base_side=base_side,
                positive_sequence_mode=positive_sequence_mode,
                zero_sequence_mode=zero_sequence_mode,
                ct_primary=lp_ct_primary,
                ct_secondary=lp_ct_secondary,
                vt_primary=lp_vt_primary,
                vt_secondary=lp_vt_secondary,
                r1=r1,
                x1=x1,
                z1_mag=z1_mag,
                phi1_deg=phi1_deg,
                r0=r0,
                x0=x0,
                re_rl=re_rl,
                xe_xl=xe_xl,
                z0_z1_mag=z0_z1_mag,
                z0_z1_angle_deg=z0_z1_angle_deg,
                kl_mag=kl_mag,
                kl_angle_deg=kl_angle_deg,
            )

            line_param_df = build_line_parameter_dataframe(line_param)

            st.session_state["line_param"] = line_param
            st.session_state["line_param_df"] = line_param_df

            st.success("Parameter saluran berhasil dinormalisasi.")

        except Exception as e:
            st.error("Normalisasi parameter saluran gagal.")
            st.exception(e)

    if "line_param" in st.session_state:
        line_param = st.session_state["line_param"]
        line_param_df = st.session_state["line_param_df"]

        st.markdown("### Normalized Line Parameter")

        col_n1, col_n2, col_n3 = st.columns(3)

        col_n1.metric("Line Name", line_param["line_name"])
        col_n2.metric("Length", f'{line_param["length_km"]:.3f} km')
        col_n3.metric("Base Side", line_param["base_side"])

        st.dataframe(
            line_param_df.style.format(
                {
                    "Real": lambda x: "" if pd.isna(x) else f"{x:.6f}",
                    "Imag": lambda x: "" if pd.isna(x) else f"{x:.6f}",
                    "Magnitude": lambda x: "" if pd.isna(x) else f"{x:.6f}",
                    "Angle Deg": lambda x: "" if pd.isna(x) else f"{x:.2f}",
                }
            ),
            use_container_width=True,
        )

        st.markdown("### Ringkasan untuk Perhitungan Jarak Gangguan")

        z1 = line_param["Z1_per_km"]
        z0 = line_param["Z0_per_km"]
        k0 = line_param["K0"]

        st.code(
            f"""
            Z1_per_km = {z1.real:.6f} + j{z1.imag:.6f} ohm/km
            Z0_per_km = {z0.real:.6f} + j{z0.imag:.6f} ohm/km
            K0        = {k0.real:.6f} + j{k0.imag:.6f}
            Length    = {line_param["length_km"]:.3f} km
            """,
            language="text",
        )
