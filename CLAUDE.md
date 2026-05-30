# Transmission Fault Locator — Claude Context

Aplikasi Streamlit untuk analisis gangguan transmisi tenaga listrik. Membaca rekaman COMTRADE, menentukan fault type, menghitung lokasi gangguan single-end dan double-end, menggambar R-X locus, serta menampilkan tower schedule dan cuaca di titik gangguan.

**Spesifikasi lengkap:** [`PRD.md`](PRD.md)  
**Riwayat keputusan & guardrail:** [`memory/MEMORY.md`](memory/MEMORY.md) — baca sebelum mengubah kode

## Stack

- Python + Streamlit (multipage via `tabs/`)
- Pandas, NumPy, SciPy untuk kalkulasi sinyal dan impedansi
- Folium / streamlit-folium untuk peta tower
- Google Sheets API untuk line parameter dan tower schedule
- OpenWeather One Call 4.0 untuk cuaca; Open-Meteo sebagai fallback
- Google Drive API tersedia di backend tapi tidak diekspos di UI

## Struktur Modul

| File | Tanggung Jawab |
|---|---|
| `app.py` | Entry point, sidebar upload, tab utama, CSS global |
| `app_runtime.py` | Cache COMTRADE/Sheets, query tower schedule, monkey-patch dataframe |
| `app_helpers.py` | Helper umum lintas fitur (downsampling, validasi, normalisasi); konstanta `OHM = chr(0x03A9)` |
| `case_storage.py` | Runtime credentials, save/restore case ZIP; `CASE_SETTINGS_KEYS` menjamin kunci konfigurasi selalu tersimpan |
| `weather_services.py` | API cuaca (OpenWeather, Open-Meteo), formatter data |
| `weather_ui.py` | HTML kartu cuaca, icon, tren suhu, bar peluang hujan |
| `tower_map.py` | Interpolasi fault pada jalur tower, render Folium map, tabel tower dengan badge proteksi |
| `rx_locus.py` | Parse relay settings, overlay zona proteksi, trajectory R-X |
| `line_analysis_helpers.py` | Infer GI name, reverse DE, comparison dataframe, override panjang line |
| `waveform_helpers.py` | Plot waveform, phasor diagram, sync local/remote |
| `fault_workflow_helpers.py` | Explanation text, threshold fault type, TWS location, timestamp parser |
| `summary_helpers.py` | Waveform fokus Summary, scoring SE/DE, grafik posisi SE/DE |
| `tabs/line_parameter.py` | Render tab Line |
| `tabs/double_ended.py` | Render tab Double-End |
| `tabs/signal_assignment.py` | Render sub-tab Signals di Local End |

## Aturan Wajib

1. **Jangan hapus atau ubah fitur** yang terdaftar di PRD tanpa konfirmasi eksplisit dari user.
2. **Jangan hardcode** URL spreadsheet private atau API key — repo bersifat public.
3. **Jangan simpan** credentials sensitif (runtime credentials, service account, xweather/accuweather key) ke dalam case ZIP. OpenWeather API key **boleh** disimpan ke case ZIP karena dibutuhkan untuk restore konteks cuaca.
4. **Jangan gunakan `st.stop()`** di dalam tab; pakai `if/else` lokal.
5. **Fault inception ≠ DFT cursor:** inception untuk sync/trigger, DFT cursor untuk kalkulasi phasor/locus.
6. **Default yang tidak boleh diubah:**
   - Auto fault cursor detection: `off`
   - Visual alignment DE: RMS envelope magnitude
   - Zone relay base: primary ohm
   - Tower Map Summary: default fault source = DE jika tersedia
7. **Setelah perubahan apapun**, jalankan:
   ```
   python -m py_compile app.py app_runtime.py app_helpers.py case_storage.py weather_services.py weather_ui.py tower_map.py rx_locus.py line_analysis_helpers.py waveform_helpers.py fault_workflow_helpers.py summary_helpers.py tabs/line_parameter.py tabs/double_ended.py tabs/signal_assignment.py
   ```
8. **Setelah perubahan workflow**, validasi minimal: Summary, Setup DB, Local End, Remote End, Line, HR Check, Single-End, Double-End, R-X Locus.
9. **Setelah perubahan kalkulasi**, validasi: SE/DE memakai sumber panjang line yang dipilih, Tower Map fault interpolasi memakai `KUMULATIF km`, Summary tidak blank bila kalkulasi belum lengkap.

## Risiko yang Perlu Diperhatikan

- `app.py` masih besar; banyak state saling bergantung — refactor bertahap, satu kelompok fungsi per tahap.
- CSS/DOM selector Streamlit bawaan rapuh; scope selector ke class komponen.
- Folium map: gunakan `key`, `center`, `zoom` eksplisit saat ingin fokus ke titik fault.
- Summary dirender lebih awal; hasil kalkulasi yang dihitung setelahnya baru tampil pada rerun berikutnya.

## Efisiensi Token (berlaku untuk semua sesi)

- **Jangan re-read file yang sudah ada di konteks** — jika konten sudah terlihat dari turn sebelumnya, gunakan langsung untuk `Edit`.
- **Jangan re-read setelah `Edit` sukses** — `Edit`/`Write` error jika gagal; re-read konfirmasi = token mubazir.
- **Gunakan `Grep -C 3`** untuk menemukan string target sebelum edit — `-C` maksimal 5–10; lebih dari itu mubazir.
- **`Read` dengan `offset`+`limit`** jika hanya butuh sebagian file besar (app.py ~4000 baris).
- **Satu `Read` blok luas, bukan beberapa `Read` kecil yang overlap** — hitung rentang sekali, baca semua sekaligus.
- **PRD.md: jangan baca penuh kecuali diminta eksplisit** — gunakan `Grep` + `Read offset+limit` untuk section relevan saja.
- **Compile check: jalankan langsung tanpa `cd`** — working directory sudah benar sejak awal session; `cd /d` adalah sintaks PowerShell/cmd, bukan Bash.
- **Jangan spawn subagent untuk edit < 50 baris di 1 file** — cold-start subagent (~10K token) lebih mahal dari mengerjakan inline.
- **Batch edit dalam satu turn** jika ada beberapa perubahan kecil di file yang sama.
- **Mulai sesi baru** untuk topik/fitur yang tidak berkaitan — jangan akumulasi 60+ turn dalam satu sesi.
- **Verifikasi asumsi framework sebelum coding** — Edit yang kemudian di-revert adalah pemborosan terbesar; lebih murah investigasi 1 menit daripada 3 attempt salah.
- **UI behavior Streamlit: investigasi dulu, coding kemudian** — sebelum mengimplementasi show/hide/accordion, verifikasi: apakah widget menulis balik ke session_state? Apakah `expanded=` controlled atau initial-state-only? Jika tidak yakin, solusi client-side (JS) lebih reliable dari Python/session_state.
- **Tombol berdampingan di Streamlit: selalu `use_container_width=True`** — tanpanya tombol tidak mengisi lebar kolom dan tampak terpisah jauh meski kolom sudah sempit.
- **Label tombol dan teks UI: title case EYD** — "Refresh Nama File", bukan "Refresh nama file". Berlaku untuk semua label `st.button`, `st.download_button`, heading, dan caption UI.

## Pemilihan Model Dinamis (Routing)

Untuk menghemat biaya, eksekusi berat/ringan dirutekan ke model berbeda lewat **subagent**. Jalankan sesi orchestrator di **Sonnet** (seimbang), lalu delegasikan:

| Tier | Kriteria | Subagent (`.claude/agents/`) | Model |
|---|---|---|---|
| LIGHT | 1 file, sepele (typo, rename lokal, teks UI, formatting), tanpa ubah session_state | `porlung-light` | Haiku |
| STANDARD | 1-2 file dalam satu area fitur, tanpa ubah kontrak session_state lintas halaman | `porlung-standard` | Sonnet |
| HEAVY | `app.py` + ≥2 modul, ATAU ubah/tambah session_state keys lintas halaman, ATAU refactor/alur workflow | `porlung-heavy` | Opus |

- Gunakan **`/route <deskripsi tugas>`** untuk scoping + delegasi otomatis.
- Patokan HEAVY: perubahan menyentuh kunci di "Session State Penting" (PRD.md) karena itu kontrak lintas-halaman (Summary/SE/DE/Tower Map/Weather saling bergantung).
- Di perbatasan dua tier, pilih yang lebih tinggi.
- Catatan: subagent mulai dingin dan membaca ulang konteks — untuk tugas benar-benar sepele saat sesi sudah di model murah, kerjakan inline daripada spawn.
- Hook **tidak bisa** mengganti model; pemilihan hanya di awal sesi (`/model`, `opusplan`) atau di batas spawn subagent.
