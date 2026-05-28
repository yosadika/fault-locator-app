# Transmission Fault Locator - Application Flow

Dokumen ini mencatat flow runtime dan pembagian modul setelah refactor bertahap. Tujuannya agar perubahan berikutnya bisa dilakukan per bagian tanpa harus membaca seluruh `app.py`.

## Entry Point

- `app.py` tetap menjadi entry point Streamlit.
- `app.py` mengatur:
  - `st.set_page_config`
  - CSS global dan print table styling
  - sidebar upload COMTRADE local/remote
  - sidebar restore case ZIP
  - pembacaan COMTRADE local sebelum tab utama ditampilkan
  - pembuatan tab utama aplikasi

## Modul Pendukung

- `app_helpers.py`
  - Helper umum yang dipakai lintas fitur.
  - Berisi downsampling plot, validasi ekstensi upload, normalisasi kolom DataFrame untuk Streamlit, dan pembalikan phasor arus.

- `app_runtime.py`
  - Mengelola helper runtime/bootstrap yang dipakai entry point.
  - Berisi cache wrapper untuk COMTRADE dan Google Spreadsheet, query Tower Schedule via Google Visualization CSV, serta monkey-patch print-friendly `st.dataframe`.

- `case_storage.py`
  - Mengelola runtime credentials, restore/save case ZIP, dan upload case ke Google Drive.
  - Menyimpan/memulihkan file COMTRADE local/remote melalui `st.session_state`.
  - Menyaring key sensitif dan bytes file mentah agar tidak masuk ke `case_state.json`.

- `weather_services.py`
  - Mengelola helper data cuaca dan lightning.
  - Berisi akses API OpenWeather, Xweather, AccuWeather, dan fallback Open-Meteo.
  - Berisi formatter angka/deskripsi cuaca dan builder DataFrame lightning.

- `weather_ui.py`
  - Mengelola rendering HTML kartu cuaca/fault weather summary.
  - Berisi helper visual seperti icon/theme cuaca, tren suhu, peluang hujan, dan `weather_card_html()`.
  - `app.py` tetap mengatur flow render Summary dan hanya memanggil HTML yang sudah dibangun modul ini.

- `tower_map.py`
  - Mengelola helper Tower Schedule berbasis peta.
  - Berisi normalisasi koordinat tower, interpolasi lokasi fault pada jalur tower, tabel tower sekitar fault, link Google Maps, dan `render_tower_map()`.
  - Modul ini membaca `st.session_state` untuk pilihan sumber fault sama seperti implementasi lama di `app.py`.

- `rx_locus.py`
  - Mengelola helper murni R-X Locus.
  - Berisi parsing distance relay settings, ekstraksi zone reach, overlay zona proteksi, dan builder trajectory R-X dari waveform/phaser.
  - Flow widget, pembacaan spreadsheet, dan penyimpanan summary R-X Locus masih berada di `app.py`.

- `line_analysis_helpers.py`
  - Mengelola helper line dan two-ended yang tidak langsung merender UI.
  - Berisi infer nama GI dari nama line, reverse line name, status operasi DE/backfeed, pemilihan DFT remote terbaik, reverse-result DE, comparison dataframe, dan override panjang line.
  - Widget pemilihan sumber panjang line tetap berada di `app.py`.

- `waveform_helpers.py`
  - Mengelola helper waveform dan phasor visual.
  - Berisi plot assigned waveform, fault-window plot, synchronized local/remote plot, estimasi time shift waveform, diagram phasor, dan tabel perbandingan prefault/fault.
  - UI tab tetap berada di `app.py`; modul ini hanya membangun figure/dataframe pendukung.

- `fault_workflow_helpers.py`
  - Mengelola helper fault workflow yang tidak merender widget.
  - Berisi explanation text, tabel threshold fault type otomatis, parser timestamp COMTRADE, TWS time-based location, dan parameter auto fault detection.

- `summary_helpers.py`
  - Mengelola helper Summary yang membangun figure/data pendukung.
  - Berisi pemilihan sinyal fault utama, waveform fokus Summary, estimasi penyebab gangguan, scoring single-ended, dan grafik posisi SE/DE.
  - Fungsi yang membaca `st.session_state` langsung tetap berada di `app.py`.

## Runtime Flow Saat Aplikasi Dibuka

1. `app.py` memuat import, constant, cache wrapper, dan helper render yang masih berada di file utama.
2. `st.set_page_config` dijalankan.
3. Sidebar meminta upload:
   - Local `.cfg`
   - Local `.dat`
   - Remote `.cfg` opsional
   - Remote `.dat` opsional
   - Load Case `.zip`
4. Jika user memilih Load Case `.zip`, `case_storage.restore_case_archive()` memulihkan state dan file, lalu aplikasi rerun.
5. Jika file upload sidebar kosong tetapi case restore punya file tersimpan, `case_storage.get_restored_upload()` membuat object upload pengganti.
6. Aplikasi validasi ekstensi local `.cfg` dan `.dat` memakai `app_helpers.validate_uploaded_extension()`.
7. Jika local belum lengkap, aplikasi hanya menampilkan summary ringkas dan berhenti.
8. Jika local lengkap, COMTRADE dibaca dengan `read_comtrade_cached()`.
9. Auto signal assignment local dihitung dan disimpan di `st.session_state`.
10. Tab utama dibuat:
    - Summary
    - Setup DB
    - Tower Schedule
    - Local End
    - Remote End
    - Line
    - HR Check
    - Single-End
    - Double-End
    - R-X Locus

## Flow Setup DB dan Case Storage

1. Setup DB membaca default config dari:
   - `st.secrets`
   - environment variable
   - runtime credentials upload
   - input manual user
2. Runtime credentials diparse oleh `case_storage.parse_runtime_credentials_upload()`.
3. Nilai credentials diterapkan ke `st.session_state` oleh `case_storage.apply_runtime_credentials()`.
4. Case ZIP dibuat oleh `case_storage.build_case_archive_bytes()`.
5. Case ZIP dapat di-download dari browser atau diupload ke Google Drive via `case_storage.upload_case_archive_to_drive()`.

## Prinsip Refactor Berikutnya

- Pindahkan satu kelompok fungsi yang kohesif setiap tahap.
- Jangan ubah perilaku UI/kalkulasi bersamaan dengan pemindahan modul.
- Setelah setiap tahap, jalankan minimal:
  - `python -m py_compile app.py app_runtime.py app_helpers.py case_storage.py weather_services.py weather_ui.py tower_map.py rx_locus.py line_analysis_helpers.py waveform_helpers.py fault_workflow_helpers.py summary_helpers.py`
- Jika modul baru dibuat, tambahkan catatan singkat di dokumen ini.
