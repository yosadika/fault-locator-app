# PRD — Transmission Fault Locator

Dokumen ini adalah sumber kebenaran tunggal untuk spesifikasi fitur, perilaku aplikasi, kontrak UI/backend, dan arsitektur modul. Gunakan dokumen ini sebagai referensi sebelum melakukan perubahan apapun.

**Jangan hapus atau ubah fitur di dokumen ini tanpa konfirmasi eksplisit dari user.**

---

## Tujuan Aplikasi

- Membaca rekaman gangguan COMTRADE local end dan optional remote end.
- Mengubah channel asli menjadi variabel standar `Va/Vb/Vc/Ia/Ib/Ic/IE`.
- Menentukan fault cursor, fault type, phasor DFT, high resistance indication, lokasi gangguan Single-End, lokasi gangguan Double-End, trajectory R-X, relay distance locus, tower schedule, dan report summary.
- Mendukung workflow kasus normal internal line fault, reverse/backfeed/external fault, serta rekaman remote yang butuh adaptasi polaritas/sudut/arah arus.

---

## Navigasi Utama

- `Summary`
- `Setup DB`
- `Tower Schedule`
- `Case Storage` berada di dalam `Setup DB`; restore case ZIP berada di sidebar.
- `Local End`
- `Remote End`
- `Line`
- `HR Check`
- `Single-End`
- `Double-End`
- `R-X Locus`

`Tower Schedule` hanya tampil setelah pasangan rekaman GI lokal lengkap (`.cfg` + `.dat`). Jika belum lengkap, layar awal hanya menampilkan Summary ringkas.

---

## Arsitektur Modul

### Entry Point

`app.py` tetap menjadi entry point Streamlit. Mengatur:
- `st.set_page_config`
- CSS global dan print table styling
- Sidebar upload COMTRADE local/remote dan restore case ZIP
- Pembacaan COMTRADE local sebelum tab utama ditampilkan
- Pembuatan dan orkestrasi tab utama

### Modul Pendukung

- **`app_helpers.py`** — Helper umum lintas fitur: downsampling plot, validasi ekstensi upload, normalisasi kolom DataFrame untuk Streamlit, pembalikan phasor arus.
- **`app_runtime.py`** — Cache wrapper untuk COMTRADE dan Google Spreadsheet, query Tower Schedule via Google Visualization CSV, monkey-patch print-friendly `st.dataframe`.
- **`case_storage.py`** — Runtime credentials, restore/save case ZIP. Menyaring key sensitif dan bytes file mentah agar tidak masuk `case_state.json`. Mendefinisikan `CASE_SETTINGS_KEYS` (kunci konfigurasi yang selalu disimpan) dan `_CASE_SETTINGS_WIDGET_FALLBACK` (sync widget key saat restore).
- **`weather_services.py`** — API OpenWeather, Xweather, AccuWeather, fallback Open-Meteo. Formatter angka/deskripsi cuaca dan builder DataFrame.
- **`weather_ui.py`** — Rendering HTML kartu cuaca: icon/theme, tren suhu, bar peluang hujan, `weather_card_html()`. `app.py` hanya memanggil HTML yang sudah dibangun modul ini.
- **`tower_map.py`** — Normalisasi koordinat tower, interpolasi lokasi fault pada jalur tower, tabel tower sekitar fault, link Google Maps, `render_tower_map()`. Membaca `st.session_state` untuk pilihan sumber fault.
- **`rx_locus.py`** — Parsing distance relay settings, ekstraksi zone reach, overlay zona proteksi, builder trajectory R-X dari waveform/phasor. Flow widget dan penyimpanan summary R-X masih di `app.py`.
- **`line_analysis_helpers.py`** — Infer nama GI dari nama line, reverse line name, status operasi DE/backfeed, pemilihan DFT remote terbaik, reverse-result DE, comparison dataframe, override panjang line. Widget pemilihan sumber panjang tetap di `app.py`.
- **`waveform_helpers.py`** — Plot assigned waveform, fault-window plot, sync local/remote plot, estimasi time shift waveform, diagram phasor, tabel perbandingan prefault/fault. UI tab tetap di `app.py`.
- **`fault_workflow_helpers.py`** — Explanation text, tabel threshold fault type otomatis, parser timestamp COMTRADE, TWS time-based location, parameter auto fault detection.
- **`summary_helpers.py`** — Pemilihan sinyal fault utama, waveform fokus Summary, estimasi penyebab gangguan, scoring single-ended, grafik posisi SE/DE. Fungsi yang membaca `st.session_state` langsung tetap di `app.py`.
- **`tabs/`** — Modul render per-tab, masing-masing mengekspos `render(...)` yang dipanggil `app.py`:
  - `tabs/line_parameter.py` — Tab `Line`. Signature: `render()`.
  - `tabs/double_ended.py` — Tab `Double-End`. Signature: `render()`.
  - `tabs/signal_assignment.py` — Sub-tab `Signals` di `Local End`. Signature: `render(df)`.

### Runtime Flow

1. `app.py` memuat import, constant, cache wrapper, dan helper render.
2. `st.set_page_config` dijalankan.
3. Sidebar meminta upload: Local `.cfg`, Local `.dat`, Remote `.cfg` (opsional), Remote `.dat` (opsional), Load Case `.zip`.
4. Jika user memilih Load Case `.zip`, `case_storage.restore_case_archive()` memulihkan state dan file, lalu aplikasi rerun.
5. Jika sidebar kosong tapi case restore punya file tersimpan, `case_storage.get_restored_upload()` membuat object upload pengganti.
6. Aplikasi validasi ekstensi local `.cfg` dan `.dat` via `app_helpers.validate_uploaded_extension()`.
7. Jika local belum lengkap, hanya Summary ringkas yang ditampilkan.
8. Jika local lengkap, COMTRADE dibaca dengan `read_comtrade_cached()`.
9. Auto signal assignment local dihitung dan disimpan di `st.session_state`.
10. Tab utama dibuat: Summary → Setup DB → Tower Schedule → Local End → Remote End → Line → HR Check → Single-End → Double-End → R-X Locus.

### Flow Setup DB dan Case Storage

1. Setup DB membaca default config dari: `st.secrets` → environment variable → runtime credentials upload → input manual user.
2. Runtime credentials diparse oleh `case_storage.parse_runtime_credentials_upload()`.
3. Nilai credentials diterapkan ke `st.session_state` oleh `case_storage.apply_runtime_credentials()`.
4. Case ZIP dibuat oleh `case_storage.build_case_archive_bytes()`.
5. Case ZIP dapat di-download via tombol `Export Case ZIP` di Setup DB.

### Prinsip Refactor

- Pindahkan satu kelompok fungsi yang kohesif setiap tahap.
- Jangan ubah perilaku UI/kalkulasi bersamaan dengan pemindahan modul.
- Setelah setiap tahap jalankan `py_compile` seluruh modul (lihat CLAUDE.md).
- Jika modul baru dibuat, tambahkan catatan di dokumen ini.

---

## Data Input

- Local COMTRADE wajib: `.cfg` dan `.dat`.
- Remote COMTRADE opsional: dipakai untuk Double-End, remote HR, remote SE, remote R-X locus, dan perbandingan summary.
- Sidebar menerima `Load Case (.zip)` untuk memulihkan rekaman, parameter user, dan hasil kalkulasi.
- Database spreadsheet utama: line parameter dan distance relay settings.
- Spreadsheet tower schedule terpisah: data tower, panjang line alternatif, map, dan fault location map.
- File Excel/Google Sheet conductor impedance: sumber parameter konduktor/line opsional.

---

## COMTRADE Reader

- Membaca CFG/DAT via `comtrade_reader.read_comtrade`.
- Metadata yang dipakai: station name, frequency, total samples, analog channel metadata, timestamp start/trigger CFG, rasio CT/VT jika tersedia.
- Memiliki fallback untuk variasi format COMTRADE, normalisasi file type CFG, timestamp, dan parsing ratio channel.

---

## Signal Assignment

- Auto assignment via `auto_assignment.detect_voltage_current_channels`.
- Pemilihan manual tersedia untuk `Va`, `Vb`, `Vc`, `Ia`, `Ib`, `Ic`, `IE/IN/3I0`.
- Jika `IE` tidak dipilih: `IE = Ia + Ib + Ic`, `I0 = IE / 3`.
- Recorded side:
  - `secondary`: waveform dikalikan rasio CT/VT ke satuan primer.
  - `primary`: waveform dianggap sudah primer; rasio CT/VT tetap disimpan untuk dokumentasi dan locus secondary conversion.
- Transformer data disimpan untuk local dan remote: `ct_primary`, `ct_secondary`, `vt_primary`, `vt_secondary`, nominal phase voltage RMS, nominal current RMS.
- Validasi duplikasi channel wajib: `Va/Vb/Vc/Ia/Ib/Ic` tidak boleh memakai channel yang sama lebih dari satu kali.

---

## Fault Cursor

- Deteksi fault inception via `fault_detection.detect_fault_inception`.
- **Default checkbox deteksi otomatis adaptif: off** untuk local dan remote.
- Window fault via `build_fault_window`: left cursor = beberapa siklus sebelum fault, right cursor = beberapa siklus setelah fault, DFT cursor = 1 siklus setelah fault inception.
- Metode deteksi: RMS sliding 1 siklus, kenaikan RMS arus, penurunan RMS tegangan, optional superimposed detection, optional refine fault bar.
- Auto threshold memperhatikan prefault RMS, nominal jika tersedia, kondisi prefault voltage rendah atau current tinggi.
- **Fault inception** dipakai untuk sinkronisasi visual/record. **DFT cursor** dipakai untuk phasor dan locus fault point. Jangan pertukarkan kedua peran ini.

---

## Phasor

- Full-cycle DFT pada window DFT cursor.
- `phasor.calculate_all_phasors` menghitung: phasor fundamental tiap phase, residual/zero sequence, sequence components `V0/V1/V2` dan `I0/I1/I2`.
- Sequence component memakai operator `a = 1∠120°`.
- Phasor disimpan terpisah untuk fault dan prefault.

---

## Fault Type

- Via `fault_type.detect_fault_type`.
- Input: magnitude `Va/Vb/Vc/Ia/Ib/Ic/IE/I0`, optional prefault phasor, threshold current rise/voltage drop/ground current/delta current/delta voltage.
- Fasa terganggu dari gabungan: arus tinggi relatif terhadap minimum, tegangan drop relatif terhadap maksimum, delta current/voltage dari prefault.
- Ground involvement dari: rasio `IE` terhadap arus maksimum, rasio `I0` terhadap arus rata-rata, kenaikan `IE/I0` dari prefault.
- Klasifikasi: `AG/BG/CG`, `AB/BC/CA`, `ABG/BCG/CAG`, `ABC`, `ABCG`, `UNKNOWN`, `A?/B?/C?`.
- Confidence 0–10 dari kejelasan fault type, ground ratio, balance current/voltage, jumlah fasa terganggu.

---

## Line Parameter

- Dibaca dari spreadsheet utama.
- `line_param` berisi minimal: `line_name`, `length_km`, `Z1_per_km`, `Z0_per_km`, `Z1_total`, `Z0_total`, `K0`.
- Konversi panjang via `convert_length_to_km` (meter/kilometer/mile).
- Impedansi dapat dibangun dari: R/X, magnitude/angle, X dan phi, primary/secondary dengan konversi CT/VT.
- Nama GI local dan remote diinfer dari `line_name`; jika tidak sesuai, user harus memperbaiki line parameter.

---

## Tower Schedule

- Tidak ada URL default yang di-hardcode (repo public). URL diisi dari runtime credentials, Streamlit secrets, env var, atau input manual Setup DB.
- Sheet default: `tower_schedule`.
- Kolom utama: `SPAN`, `JARAK`, `KUMULATIF`, `LATITUDE`, `LONGITUDE`, `SEGMENT`, `ULTG`, `TYPE STRING`, `JUMLAH STRING`.
- Kolom tambahan spreadsheet harus tetap dipertahankan (cleaning, proteksi petir, DGS, MGGS, TLA/NGLA, EGLA, sumur bor, MDG, DMRG, MRG, DG, dinding penahan, kerawanan binatang, dll.).
- `JARAK` dan `KUMULATIF` dari spreadsheet dianggap meter. Aplikasi menambahkan `JARAK km` dan `KUMULATIF km`.
- Tampilan km: 6 desimal pada tabel utama (berpengaruh ke kalkulasi DE/SE).
- Panjang line dari tower schedule: prioritas `max(KUMULATIF km)`, fallback `sum(JARAK km)`, disimpan ke `tower_schedule_selected_length_km`.
- Filter awal load: ULTG, Segment (difilter berdasarkan ULTG), opsi `Load semua data` tersedia tapi tidak default.
- Filter setelah load: Segment, ULTG, Type String, pencarian span/teks.

---

## Tower Map

- Memakai Folium/streamlit-folium. Default tile: Esri World Imagery/satellite. Alternatif: OpenStreetMap/street.
- Layer: Tower, Fault Location, Tower path/polyline.
- Kontrol peta dalam expander `Map Settings`: default tertutup di Summary/report, default terbuka di Tower Schedule/exploration.
- Kontrol layer bawaan Leaflet/Folium disembunyikan; pilihan layer dikendalikan oleh kontrol Streamlit.
- Marker tower: size 10, label ringkas dari kolom `SPAN` (contoh: `#0164`), nama lengkap di hover/popup.
- Popup tower: SPAN, JARAK, KUMULATIF, ULTG, SEGMENT, TYPE STRING, JUMLAH STRING, LATITUDE/LONGITUDE, link `Open Maps` dan `Directions`.
- Fault location source: default DE jika tersedia, fallback SE local. SE remote dikonversi ke `line_length - remote_distance`.
- Fault interpolation: memakai `KUMULATIF km`, mencari dua tower pengapit, koordinat dihitung dengan interpolasi linear latitude/longitude. Jika fault di luar range data, marker di ujung dan warning muncul.
- Marker fault: crosshair presisi, span pengapit di-highlight merah, label permanen menampilkan sumber/jarak/rasio span, label diposisikan adaptif menjauh dari arah span pengapit.
- Popup fault: sumber kalkulasi, distance, plotted distance, tower pengapit, jarak dari/ke tower A/B, tower terdekat, jarak ke tower terdekat, panjang span, rasio span, quality/status, lat/lon, link Maps fault dan tower terdekat.
- Summary Tower Map fokus otomatis ke dua tower pengapit dan titik fault, bukan seluruh jalur.
- Tabel `Data tower sekitar titik gangguan (-5 / +5)` di bawah Tower Map: 5 tower sebelum dan sesudah span fault. Ditampilkan sebagai tabel HTML dengan kolom: No, SPAN, TYPE STRING, JUMLAH STRING, dan **Proteksi Terpasang**.
  - Kolom Proteksi Terpasang menampilkan badge berwarna per kategori: Cleaning (hijau), Proteksi Petir — termasuk DGS/MGGS/TLA/NGLA/EGLA/Sumur Bor/MDG/MRG/DG/DMRG (kuning), Dinding Penahan Tanah (coklat), Kerawanan Binatang (merah), Proteksi Binatang (ungu).
  - Badge induk (misal "Proteksi Petir") hanya muncul jika tidak ada satu pun sub-device yang terisi.
  - Badge menampilkan tanggal pemasangan jika tersedia di spreadsheet.
  - Baris `Before fault span` / `After fault span` ditandai dengan background merah muda; `Closest tower` dengan kuning.
  - Kolom Fault Context, Distance from Fault km, JARAK, KUMULATIF, LATITUDE, LONGITUDE, SEGMENT, ULTG disembunyikan dari tabel (tetap ada di DataFrame backend).
- Cuaca terkini di titik gangguan ditampilkan di bawah Tower Map (lihat bagian Weather).

---

## High Resistance Check

- Sub-view local dan remote. Judul end memakai nama GI/lokasi.
- Loop impedance sesuai fault type:
  - ground loop: `Va/(Ia + K0*I0)`, `Vb/(Ib + K0*I0)`, `Vc/(Ic + K0*I0)`
  - phase loop: `(Va-Vb)/(Ia-Ib)`, `(Vb-Vc)/(Ib-Ic)`, `(Vc-Va)/(Ic-Ia)`
  - 3 phase fallback: `Va/Ia`
- Distance pembanding: magnitude `|Zapp|/|Z1_per_km|`, reactance `Imag(Zapp)/Imag(Z1_per_km)`, projection ke arah sudut Z1.
- Estimasi Rf: `Zline_est = distance_x * Z1_per_km`, `Rf_est = Real(Zapp - Zline_est)`.
- Indikator HR: `Rf_est >= threshold`, deviasi sudut Zapp terhadap Z1, deviasi distance magnitude vs reactance, distance keluar line.
- Confidence 0–10 dan evidence score ditampilkan.
- Simbol ohm harus tampil sebagai `Ω`, bukan karakter rusak.

---

## Single-End Fault Locator

- Sub-view local dan remote.
- Input: phasor fault, prefault phasor jika tersedia, fault type, line parameter efektif, sumber panjang line.
- Sumber panjang line: Line Parameter atau Tower Schedule jika sudah dimuat/difilter.
- Jika memakai Tower Schedule: `length_km`, `Z1_total`, `Z0_total` dihitung ulang dari `Z1_per_km/Z0_per_km`.
- Jika sumber panjang line berubah setelah hasil dihitung, halaman memberi warning agar user menghitung ulang.
- Loop impedance: sama dengan HR (ground loop, phase loop, 3-phase fallback).
- Distance method: magnitude, reactance, projection. **Recommended default: reactance.**
- Fault context: internal line fault atau reverse/backfeed external fault.
- Mode reverse/backfeed: signed distance dipertahankan; jarak negatif atau > line length tidak langsung salah.
- Superimposed fallback: untuk ground fault bila prefault tersedia, memakai delta voltage/current, dapat mengganti recommended distance jika konvensional out-of-range dan superimposed reactance masuk range.
- Status: `VALID`, `CHECK`, `UNCERTAIN`.
- Warning: jarak negatif, jarak melebihi line, magnitude vs reactance berbeda signifikan, Rf tinggi, sudut Zapp menyimpang, indikasi load-flow/backfeed.
- Hasil disimpan: `single_ended_result`, `remote_single_ended_result`, dataframe detail masing-masing.

---

## Double-End Fault Locator

- Remote setup berada di tab `Remote End`, bukan di halaman DE.
- Input: local positive sequence phasor `V1/I1`, remote `V1/I1`, line parameter efektif, remote record adaptation, scenario gangguan.
- Sumber panjang line: Line Parameter atau Tower Schedule. Jika Tower Schedule: `Z1_total = Z1_per_km * length_km`, `Z0_total = Z0_per_km * length_km`.
- Positive-sequence equation:
  - `Vlocal(x) = V1L - I1L * Z1_per_km * x`
  - `Vremote(x) = V1R - I1R * Z1_per_km * (L - x)`
  - fault point: `x = (V1L - V1R + I1R * Z1_per_km * L) / (Z1_per_km * (I1L + I1R))`
  - Jika remote current direction `opposite_to_line`, `I1R` diinversi.
- Output utama: distance complex, distance km dari local, distance percent, distance dari remote, voltage fault dari local/remote, mismatch tegangan fault, quality score.
- Quality score: penalti distance negatif/> line length/imaginary, penalti mismatch tegangan, dikunci 0–10.
- Remote adaptation: `auto_adapt_record`, `auto_current_direction_only`, manual `into_line`, manual `opposite_to_line`.
- Candidate ranking: distance keluar line, imaginary distance, mismatch ratio, quality score, penalti angle shift, penalti polaritas/arah tertentu.
- Visual sync: **referensi default `fault_cursor` (Fault cursor only)**, **metode default RMS envelope magnitude**, opsi sinkronisasi fault cursor/time/visual. Fault inception untuk alignment, DFT cursor untuk kalkulasi phasor.
- Optional TWS/time-based: `distance_from_local = (L + v * delta_t) / 2`, warning bila delta time/distance tidak realistis.
- Scenario: internal line fault atau reverse/backfeed/external fault. SOTF/parallel/adjacent line diperlakukan sebagai konteks reverse/backfeed, bukan label wajib.
- Perbandingan SE pada halaman DE memakai line parameter efektif yang sama; jika DE memakai Tower Schedule, SE comparison juga memakai panjang Tower Schedule.
- Hasil disimpan: `two_ended_result`, `two_ended_quality`, `two_ended_reverse_result`, `two_ended_reverse_quality`, `two_ended_comparison_df`, local/remote SE comparison result.

---

## R-X Locus

- Sub-view local dan remote.
- Membaca distance relay setting dari sheet `distance_settings`.
- Filter relay setting: GI/Substation, Bay, search text.
- Zone setting base: **default primary ohm**. Optional: relay secondary ohm, dikonversi ke primary via rasio CT/VT dari Signal Assignment.
- Parameter zona: `Z1/Z2/Z3 Res Ph`, `Z1/Z2/Z3 Res Gnd`, kN dan kN angle jika tersedia.
- Zona quadrilateral: X reach dan R reach/resistive reach, phase/gnd reach dipilih sesuai loop fault.
- Trajectory: apparent impedance dari waveform sepanjang window, titik DFT cursor ditandai, Z Line Total sebagai referensi.
- Plot focus: default fokus ke relay zones; dapat menampilkan trajectory penuh bila user pilih.
- Summary menampilkan R-X Locus local dan remote sebagai section terpisah agar print tidak menumpuk.

---

## Weather Summary

### Backend

- Koordinat cuaca dari titik gangguan hasil interpolasi Tower Schedule.
- Default sumber fault: Tower Map Summary (DE jika tersedia).
- OpenWeather One Call 4.0: sumber utama current weather dan forecast. Open-Meteo: fallback untuk cuaca saat ini bila OpenWeather gagal.
- Forecast OpenWeather dari timeline 15 menit, diagregasi ke bucket per 1 jam relatif terhadap waktu akses.
- Tiap bucket forecast menyimpan: waktu label, temperatur rata-rata, peluang hujan maksimum, presipitasi total, weather code/deskripsi dominan.
- Ringkasan hujan memakai probabilitas/akumulasi, bukan data petir.
- Jangan request histori thunderstorm/petir jika tidak ditampilkan.
- OpenWeather API key dibaca dari: `st.session_state["openweather_lightning_api_key"]` → widget → `st.secrets["OPENWEATHER_API_KEY"]` → env var `OPENWEATHER_API_KEY`. Input kosong tidak boleh menimpa key yang sudah tersimpan.

### UI Contract

- Judul section: `Cuaca Terkini`.
- Caption menjelaskan bahwa data adalah cuaca dari OpenWeather, bukan data sambaran petir.
- Pengaturan cuaca dalam expander tertutup.
- Satu card untuk titik gangguan, bukan dua card tower.
- Layout desktop: panel kiri (kondisi saat ini) + panel kanan (tren, forecast). Layout mobile: stack vertikal, tidak ada scroll internal.
- Panel kiri wajib: lokasi fault + span pengapit, timestamp cuaca, simbol cuaca, deskripsi bahasa Indonesia, temperatur `°C`, terasa seperti, hujan saat ini `mm`, angin `km/h`, kelembapan `%`, tutupan awan `%`, kumulatif fault `km`.
- Panel kanan wajib: tren suhu `°C`, peluang hujan `%`, ringkasan titik gangguan, peluang hujan tertinggi, perkiraan hujan total, sub-card forecast per 1 jam, footer koordinat + sumber API.
- Grafik tren: titik/batang vertikal sederhana tanpa garis penghubung; grafik hujan: bar sederhana. Tidak boleh terpotong/clipping.
- Simbol cuaca via HTML entity, bukan emoji mentah (hindari encoding mojibake).
- Terjemahan deskripsi cuaca: `Broken Clouds` → `Berawan`, `Scattered Clouds` → `Berawan sebagian`, `Overcast Clouds` → `Mendung`, `Mainly Clear` → `Umumnya cerah`, `Clear Sky` → `Cerah`, `Light Rain` → `Hujan ringan`.
- Teks ringkasan hujan: `Tidak ada indikasi hujan dalam 12 jam ke depan.` / `Ada peluang hujan mulai sekitar HH:MM.` — jangan gunakan `hujan kuat/lebat` tanpa klasifikasi intensitas.
- Jangan tampilkan widget atau label petir/badai tanpa provider lightning aktual.
- Kartu print-friendly; jika API gagal, tampilkan pesan gagal tanpa menghentikan Summary.

---

## Summary

- Tampil setelah COMTRADE lokal berhasil dibaca.
- Jika kalkulasi belum lengkap, tampilkan `Pending`, bukan halaman blank.
- Konten: metadata local/remote, status Signal Assignment/Fault Cursor/Phasor/Fault Type/Line Parameter/SE/DE, key results (fault type, SE distance, DE distance, DE quality), IE source local/remote, perbandingan prefault/fault, waveform fokus opsional, estimasi penyebab gangguan, grafik SE/DE, Tower Map Fault Location, cuaca terkini + forecast, R-X Locus local/remote, warning kualitas DE/HR.
- Grafik SE/DE: memakai hasil paling update dari session; scoring SE via status `VALID/CHECK/UNCERTAIN` + warning count; scoring DE via `quality_score`; line length mengikuti hasil DE jika memakai Tower Schedule.
- Tower Map Summary: default DE jika tersedia, fallback SE; fokus ke dua tower pengapit; Map Settings default tertutup; tabel -5/+5 tower default terbuka saat focus fault.
- Weather Summary: tampil setelah Tower Map punya data tower dan sumber fault.

---

## Case Storage

- Format: arsip ZIP berisi `records/local_cfg`, `records/local_dat`, `records/remote_cfg`, `records/remote_dat`, `manifest.json`, `case_state.json`.
- `manifest.json`: schema, timestamp, nama case, folder Drive target, daftar file.
- `case_state.json`: snapshot `st.session_state` yang JSON-safe (DataFrame → records+columns, complex → real/imag, numpy → tipe JSON; bytes rekaman tidak masuk JSON).
- Restore: dari sidebar sebelum validasi local COMTRADE; file dari ZIP dibungkus sebagai upload virtual dengan `.name` dan `.getvalue()`. Restore memakai MD5 hash untuk mencegah restore loop berulang.
- Export: tombol `Export Case ZIP` di `Setup DB > Case Storage`. Nama file digenerate otomatis: `porlungcase_{line_name}_{YYYYMMDD}_{HHMMSS}.zip`.
- Google Drive upload tidak tersedia di UI (fitur backend tersedia tapi tidak diekspos untuk menjaga stabilitas).
- **Yang disimpan ke case ZIP:** semua session state kecuali key sensitif. `CASE_SETTINGS_KEYS` menjamin kunci berikut selalu disimpan: DB URLs, sheet names, signal assignment (channel selections + CT/VT + recorded side), line param, OpenWeather API key.
- **Yang tidak disimpan:** runtime credentials file, service account, xweather/accuweather key, bytes file upload sementara.

### Runtime Credentials

- Format: file `credentials.toml` atau `credentials.json` diupload ke Setup DB.
- Hanya dibaca ke memory/session; tidak ditulis ke disk, tidak dicetak, tidak diekspor.
- Dapat mengisi otomatis: Database Spreadsheet URL, Line/Cable/Distance sheet names, Tower Schedule URL + sheet, OpenWeather API key, Google service account opsional.
- Tombol `Clear runtime credentials from session` tersedia.
- `.gitignore` harus mengecualikan `.streamlit/secrets.toml`, `credentials*.toml`, `credentials*.json`.

### Prioritas Konfigurasi

1. Uploaded runtime credentials di session
2. Streamlit secrets / `.streamlit/secrets.toml`
3. Environment variables
4. Input manual UI
5. Default demo/public (hanya jika benar-benar aman)

---

## UI/UX dan Frontend

- Aplikasi adalah tool engineering/reporting, bukan landing page. Tampilan padat, rapi, mudah discan.
- Hindari dekorasi berlebihan, shadow besar, gradient mencolok.
- Section besar memakai heading jelas dan ringkas. Data penting terlihat tanpa scroll di dalam card.
- HTML custom via `st.html` jika tersedia; fallback `components.html`. Hindari `st.markdown(..., unsafe_allow_html=True)` untuk blok HTML panjang.
- CSS scoped pada class komponen, bukan selector global Streamlit, kecuali untuk print/report.
- Card: radius kecil-menengah, border halus, background terang, tidak ada shadow besar.
- Mobile: layout card menjadi satu kolom, konten tidak terpotong, tidak ada overflow/scroll internal sebagai solusi utama, grid forecast turun menjadi 4 kolom lalu 2 kolom.
- Tabel data engineering boleh memakai Streamlit dataframe untuk eksplorasi; Summary/report memakai card/tabel HTML print-friendly.
- Jangan tambah label/widget yang menyebut data tidak tersedia (misalnya petir/badai) kecuali backend punya provider aktual.
- Istilah teknis boleh untuk protection engineer; istilah cuaca/provider diterjemahkan ke bahasa Indonesia.

### Print/Report

- Summary harus print-friendly.
- Sidebar, uploader, toolbar, kontrol eksplorasi tidak dicetak.
- Card/tabel/plot besar tidak boleh pecah buruk di tengah halaman.
- Dataframe interaktif disembunyikan saat print jika ada tabel HTML pengganti.
- Penomoran baris tabel default dari 1, bukan 0.

---

## Session State Penting

**Local:**
`assigned_df`, `local_transformer_data`, `fault_window`, `phasors`, `prefault_phasors`, `fault_type_result`

**Remote:**
`remote_assigned_df`, `remote_transformer_data`, `remote_fault_window`, `remote_phasors`, `remote_prefault_phasors`, `remote_fault_type_result`

**Line:**
`line_param`

**Tower:**
`tower_schedule_df`, `tower_schedule_filtered_df`, `tower_schedule_selected_length_km`, `tower_schedule_selected_length_source`, `tower_schedule_selected_segment`, `tower_schedule_selected_ultg`

**Case Storage:**
`case_name`, `case_drive_folder_url`, `case_drive_folder_id`, `case_local_cfg_name`, `case_local_cfg_bytes`, `case_local_dat_name`, `case_local_dat_bytes`, `case_remote_cfg_name`, `case_remote_cfg_bytes`, `case_remote_dat_name`, `case_remote_dat_bytes`

**HR:**
`high_resistance_result` (+ remote equivalent jika dihitung)

**SE:**
`single_ended_result`, `remote_single_ended_result`, `single_ended_df`, `remote_single_ended_df`

**DE:**
`two_ended_result`, `two_ended_quality`, `two_ended_reverse_result`, `two_ended_reverse_quality`, `two_ended_local_single_result`, `two_ended_remote_single_result`, `two_ended_remote_sync_shift_s`, `two_ended_remote_sync_method`, `two_ended_operating_status`

**R-X:**
`rx_locus_summary_fig_local`, `rx_locus_summary_fig_remote` (+ metadata masing-masing)

---

## External Links

- Popup tower dan fault menyediakan `Open Maps` dan `Directions` via Google Maps query/directions berbasis latitude/longitude.

---

## Risiko Teknis

- `app.py` masih besar; banyak state saling bergantung — refactor bertahap, satu kelompok fungsi per tahap.
- `st.stop()` di dalam tab dapat menghentikan render seluruh aplikasi; gunakan `if/else` lokal.
- Summary dirender lebih awal; hasil kalkulasi yang dihitung setelahnya baru tampil pada rerun berikutnya.
- CSS/DOM selector Streamlit bawaan rapuh; scope ke class komponen.
- Folium map: gunakan `key`, `center`, `zoom` eksplisit saat memaksa fokus fault.
- Banyak kalkulasi sangat bergantung pada kualitas signal assignment, polaritas CT/VT, dan kecocokan line/tower data.
