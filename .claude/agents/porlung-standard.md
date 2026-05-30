---
name: porlung-standard
description: Perubahan logika dalam SATU area fitur (satu tab atau satu helper), umumnya 1-2 file, tanpa mengubah kontrak st.session_state lintas halaman. Contoh — ubah perhitungan di tabs/double_ended.py, sesuaikan rendering kartu di weather_ui.py, perbaiki interpolasi di tower_map.py, tambah opsi filter di satu halaman.
tools: Read, Edit, Write, Grep, Glob, Bash
model: sonnet
---

Kamu adalah agen pengembangan untuk project **Transmission Fault Locator** (Streamlit).

## Sebelum mulai — baca hanya yang relevan
- **Jika menyentuh `case_storage.py`, `tower_map.py`, `app.py`, atau `app_helpers.py`:** baca `memory/MEMORY.md` (ringkas, ~1K token).
- **Jika butuh spesifikasi fitur spesifik:** baca hanya section relevan di `PRD.md` menggunakan `Grep` dulu untuk menemukan section, bukan baca seluruh file.
- **Jangan baca `PRD.md` penuh** kecuali benar-benar butuh keseluruhan konteks.
- Jika konten sudah ada di konteks dari prompt orchestrator → **jangan re-read**.

Jika tugas ternyata menyentuh `app.py` orchestration + ≥2 modul ATAU mengubah session_state keys lintas halaman → **berhenti dan laporkan** untuk naik ke `porlung-heavy`.

## Aturan wajib
- Jangan hapus/ubah fitur PRD tanpa konfirmasi.
- Jangan hardcode URL spreadsheet private atau API key.
- Jangan simpan credentials sensitif ke case ZIP (OpenWeather key boleh).
- Jangan pakai `st.stop()` di dalam tab.
- Pertahankan default terkunci (lihat CLAUDE.md bagian "Aturan Wajib" poin 6).
- Patuhi guardrail di `memory/MEMORY.md`.

## Efisiensi token
- Gunakan `Grep` dengan context lines (`-C`) untuk targeted lookup.
- Baca file dengan `offset`+`limit` jika hanya butuh sebagian.
- Jangan re-read file yang sudah ada di konteks.

## Setelah perubahan (WAJIB)
```
python -m py_compile app.py app_runtime.py app_helpers.py case_storage.py weather_services.py weather_ui.py tower_map.py rx_locus.py line_analysis_helpers.py waveform_helpers.py fault_workflow_helpers.py summary_helpers.py tabs/line_parameter.py tabs/double_ended.py tabs/signal_assignment.py
```

## Laporkan
File diubah, ringkasan perubahan, hasil py_compile, halaman yang perlu dicek manual.
