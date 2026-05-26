# Transmission Fault Locator - Feature Inventory

Dokumen ini adalah baseline fitur, logika, dan metode kalkulasi aplikasi. Saat melakukan refactor atau menambah fitur, jangan menghapus perilaku di dokumen ini tanpa konfirmasi eksplisit.

## Tujuan Aplikasi

- Membaca rekaman gangguan COMTRADE local end dan optional remote end.
- Mengubah channel asli menjadi variabel standar `Va/Vb/Vc/Ia/Ib/Ic/IE`.
- Menentukan fault cursor, fault type, phasor DFT, high resistance indication, lokasi gangguan Single-End, lokasi gangguan Double-End, trajectory R-X, relay distance locus, tower schedule, dan report summary.
- Mendukung workflow kasus normal internal line fault, reverse/backfeed/external fault, serta rekaman remote yang butuh adaptasi polaritas/sudut/arah arus.

## Navigasi Utama

- `Summary`
- `Setup DB`
- `Tower Schedule`
- `Local End`
- `Remote End`
- `Line`
- `HR Check`
- `Single-End`
- `Double-End`
- `R-X Locus`

`Tower Schedule` hanya tampil setelah pasangan rekaman GI lokal lengkap (`Local .cfg` dan `Local .dat`). Jika GI lokal belum lengkap, layar awal hanya menampilkan Summary ringkas.

## Data Input

- Local COMTRADE wajib: `.cfg` dan `.dat`.
- Remote COMTRADE opsional: `.cfg` dan `.dat`, dipakai untuk Double-End, remote HR, remote SE, remote R-X locus, dan perbandingan summary.
- Database spreadsheet utama dipakai untuk line parameter dan distance relay settings.
- Spreadsheet tower schedule terpisah dipakai untuk data tower, panjang line alternatif, map, dan fault location map.
- File Excel/Google Sheet conductor impedance dapat dibaca sebagai sumber parameter konduktor/line jika fitur import dipakai.

## COMTRADE Reader

- Membaca CFG/DAT memakai `comtrade_reader.read_comtrade`.
- Metadata yang dipakai:
  - station name
  - frequency
  - total samples
  - analog channel metadata
  - timestamp start/trigger CFG
  - rasio CT/VT jika tersedia di CFG
- Reader memiliki fallback untuk variasi format COMTRADE, normalisasi file type CFG, timestamp, dan parsing ratio channel.

## Signal Assignment

- Auto assignment memakai `auto_assignment.detect_voltage_current_channels`.
- Pemilihan manual tetap tersedia untuk `Va`, `Vb`, `Vc`, `Ia`, `Ib`, `Ic`, dan `IE/IN/3I0`.
- Jika `IE` tidak dipilih, aplikasi menghitung residual:
  - `IE = Ia + Ib + Ic`
  - `I0 = IE / 3`
- Recorded side:
  - `secondary`: waveform dikalikan rasio CT/VT ke satuan primer.
  - `primary`: waveform dianggap sudah primer; rasio CT/VT tetap disimpan untuk dokumentasi dan locus secondary conversion.
- Transformer data disimpan untuk local dan remote:
  - `ct_primary`, `ct_secondary`
  - `vt_primary`, `vt_secondary`
  - nominal phase voltage RMS
  - nominal current RMS
- Validasi duplikasi channel wajib ada: `Va/Vb/Vc/Ia/Ib/Ic` tidak boleh memakai channel yang sama lebih dari satu kali.

## Fault Cursor

- Deteksi fault inception memakai `fault_detection.detect_fault_inception`.
- Default checkbox deteksi otomatis adaptif nominal + pre-fault adalah off untuk local dan remote.
- Window fault memakai `build_fault_window`:
  - left cursor = beberapa siklus sebelum fault
  - right cursor = beberapa siklus setelah fault
  - DFT cursor = 1 siklus setelah fault inception
- Metode deteksi:
  - RMS sliding 1 siklus.
  - Kenaikan RMS arus terhadap baseline pre-fault.
  - Penurunan RMS tegangan terhadap baseline pre-fault.
  - Optional superimposed detection dari perubahan instantaneous.
  - Optional refine fault bar agar onset lebih dekat ke perubahan instantaneous.
- Auto threshold memperhatikan:
  - prefault RMS current/voltage
  - nominal phase voltage/current jika tersedia
  - kondisi prefault voltage rendah atau current tinggi.
- Fault inception dipakai untuk sinkronisasi visual/record, sedangkan DFT cursor dipakai untuk phasor dan locus fault point.

## Phasor

- Phasor dihitung dengan full-cycle DFT pada window DFT cursor.
- `phasor.calculate_all_phasors` menghitung:
  - phasor fundamental tiap phase.
  - residual/zero sequence jika IE tersedia atau dihitung.
  - sequence components `V0/V1/V2` dan `I0/I1/I2`.
- Sequence component memakai operator `a = 1∠120°`.
- Phasor disimpan terpisah untuk fault dan prefault.

## Fault Type

- Fault type memakai `fault_type.detect_fault_type`.
- Input utama:
  - magnitude `Va/Vb/Vc/Ia/Ib/Ic/IE/I0`
  - optional prefault phasor
  - threshold current rise, voltage drop, ground current, delta current, delta voltage
- Fasa terganggu ditentukan dari gabungan:
  - arus tinggi relatif terhadap arus minimum.
  - tegangan drop relatif terhadap tegangan maksimum.
  - delta current dari prefault ke fault.
  - delta voltage dari prefault ke fault.
- Ground involvement ditentukan dari:
  - rasio `IE` terhadap arus maksimum.
  - rasio `I0` terhadap arus rata-rata.
  - kenaikan `IE/I0` terhadap prefault jika prefault tersedia.
- Klasifikasi:
  - 1 fasa + ground -> `AG/BG/CG`
  - 2 fasa -> `AB/BC/CA`
  - 2 fasa + ground -> `ABG/BCG/CAG`
  - 3 fasa -> `ABC`
  - 3 fasa + ground -> `ABCG`
  - tidak jelas -> `UNKNOWN` atau `A?/B?/C?`
- Confidence 0-10 dihitung dari kejelasan fault type, ground ratio, balance current/voltage, dan jumlah fasa terganggu.

## Line Parameter

- Line parameter dibaca dari spreadsheet utama.
- Aplikasi harus menyimpan `line_param` berisi minimal:
  - `line_name`
  - `length_km`
  - `Z1_per_km`
  - `Z0_per_km`
  - `Z1_total`
  - `Z0_total`
  - `K0`
- Konversi panjang:
  - meter/kilometer/mile sesuai helper `convert_length_to_km`.
- Impedansi dapat dibangun dari:
  - R/X
  - magnitude/angle
  - X dan phi
  - primary/secondary dengan konversi CT/VT bila diperlukan.
- Nama GI local dan remote diinfer dari `line_name`; jika tidak sesuai, user harus memperbaiki line parameter.

## Tower Schedule

- Spreadsheet default:
  - URL: `https://docs.google.com/spreadsheets/d/<TOWER_SCHEDULE_SPREADSHEET_ID>/edit?usp=sharing`
  - sheet: `tower_schedule`
- URL dan sheet diatur dari `Setup DB`, bukan dari halaman Tower Schedule.
- Kolom utama:
  - `SPAN`
  - `JARAK`
  - `KUMULATIF`
  - `LATITUDE`
  - `LONGITUDE`
  - `SEGMENT`
  - `ULTG`
  - `TYPE STRING`
  - `JUMLAH STRING`
- Kolom tambahan spreadsheet harus tetap dipertahankan di tabel, termasuk tetapi tidak terbatas pada:
  - cleaning isolator dan tanggal cleaning
  - proteksi petir, DGS, MGGS, TLA/NGLA, EGLA
  - sumur bor, MDG, DMRG, MRG, DG
  - dinding penahan tanah, balok kopel, bronjong, sheet pile, shotcrete
  - kerawanan binatang, burung, kera, ular, jaring, kawat duri, dll.
- `JARAK` dan `KUMULATIF` dari spreadsheet dianggap meter.
- Aplikasi menambahkan:
  - `JARAK km`
  - `KUMULATIF km`
- Tampilan km memakai 6 desimal pada tabel utama karena berpengaruh ke kalkulasi DE/SE.
- Panjang line dari tower schedule:
  - prioritas `max(KUMULATIF km)`
  - fallback `sum(JARAK km)`
  - disimpan ke session sebagai `tower_schedule_selected_length_km`.
- Filter awal load:
  - user memilih `ULTG sebelum load` dan/atau `Segment sebelum load`.
  - daftar `Segment sebelum load` difilter berdasarkan `ULTG sebelum load`.
  - opsi `Load semua data` tersedia tetapi tidak default.
  - query Google Sheet dibatasi agar load tidak berat.
- Filter setelah load:
  - Segment
  - ULTG
  - Type String
  - pencarian span/teks.

## Tower Map

- Map memakai Folium/streamlit-folium.
- Default tile: Esri World Imagery/satellite.
- Tile alternatif: OpenStreetMap/street.
- Layer:
  - Tower
  - Fault Location
  - Tower path/polyline
- Kontrol peta dibungkus dalam expander `Map Settings`.
  - Default tertutup di Summary/report.
  - Default terbuka di Tower Schedule/exploration.
- Marker tower:
  - default size 10.
  - label tower default aktif.
  - label ringkas diambil dari kolom `SPAN`, misalnya `#0164`.
  - nama lengkap tetap ada di hover/popup.
- Popup tower menampilkan:
  - SPAN
  - JARAK
  - KUMULATIF
  - ULTG
  - SEGMENT
  - TYPE STRING
  - JUMLAH STRING
  - LATITUDE/LONGITUDE
  - link `Open Maps` dan `Directions`.
- Fault location source:
  - default DE jika hasil DE tersedia.
  - fallback SE local.
  - SE remote tersedia jika hasil remote SE ada; dikonversi menjadi jarak dari local dengan `line_length - remote_distance`.
- Fault interpolation:
  - memakai `KUMULATIF km`.
  - mencari dua tower yang mengapit jarak fault.
  - koordinat fault dihitung dengan interpolasi linear latitude/longitude di antara dua tower tersebut.
  - jika fault sebelum tower pertama atau setelah tower terakhir, marker ditempatkan di ujung data dan warning muncul.
- Marker fault:
  - crosshair presisi, bukan pin besar.
  - pusat marker tepat pada koordinat interpolasi.
  - span pengapit di-highlight merah.
  - label fault permanen menampilkan sumber, jarak, dan rasio span.
- Popup fault menampilkan:
  - sumber kalkulasi
  - distance
  - plotted distance
  - tower pengapit
  - jarak dari tower A
  - jarak ke tower B
  - tower terdekat
  - jarak ke tower terdekat
  - panjang span
  - rasio span
  - quality/status
  - latitude/longitude
  - link Maps fault
  - link Maps tower terdekat.
- Summary Tower Map fokus otomatis ke dua tower pengapit dan titik fault, bukan seluruh jalur.
- Di bawah Tower Map, aplikasi menampilkan tabel `Data tower sekitar titik gangguan (-5 / +5)`:
  - 5 tower sebelum span fault.
  - 5 tower sesudah span fault.
  - semua kolom spreadsheet tetap ditampilkan.
  - kolom tambahan `Fault Context`.
  - kolom tambahan `Distance from Fault km`.

## High Resistance Check

- Sub-view local dan remote.
- Judul end memakai nama GI/lokasi, bukan sekadar Local End/Remote End.
- Loop impedance dihitung sesuai fault type:
  - ground loop: `Va/(Ia + K0*I0)`, `Vb/(Ib + K0*I0)`, `Vc/(Ic + K0*I0)`
  - phase loop: `(Va - Vb)/(Ia - Ib)`, `(Vb - Vc)/(Ib - Ic)`, `(Vc - Va)/(Ic - Ia)`
  - 3 phase fallback: `Va/Ia`
- Distance pembanding:
  - magnitude: `|Zapp| / |Z1_per_km|`
  - reactance: `Imag(Zapp) / Imag(Z1_per_km)`
  - projection: proyeksi `Zapp` ke arah sudut `Z1`.
- Estimasi Rf:
  - `Zline_est = distance_x * Z1_per_km`
  - `Rf_est = Real(Zapp - Zline_est)`
- Indikator high resistance:
  - `Rf_est >= threshold`
  - deviasi sudut Zapp terhadap Z1
  - deviasi distance magnitude vs reactance
  - distance keluar line
- Confidence 0-10 dan evidence score ditampilkan.
- Simbol ohm harus tampil sebagai `Ω`, bukan karakter rusak.

## Single-End Fault Locator

- Sub-view local dan remote.
- Input:
  - phasor fault
  - prefault phasor jika tersedia
  - fault type
  - line parameter efektif
  - sumber panjang line.
- Sumber panjang line:
  - Line Parameter
  - Tower Schedule jika sudah dimuat/difilter.
- Jika memakai Tower Schedule, `length_km`, `Z1_total`, dan `Z0_total` dihitung ulang dari `Z1_per_km/Z0_per_km`.
- Jika sumber panjang line berubah setelah hasil dihitung, halaman memberi warning agar user menghitung ulang.
- Loop impedance sama dengan HR/relay distance:
  - ground: `Vphase / (Iphase + K0*I0)`
  - phase: `Vphase-phase / Iphase-phase`
  - 3 phase fallback.
- Distance method:
  - magnitude
  - reactance
  - projection
- Recommended method default: reactance.
- Fault context:
  - internal line fault.
  - reverse/backfeed external fault.
- Mode reverse/backfeed:
  - signed distance dipertahankan.
  - distance negatif atau > line length tidak langsung dianggap salah; dapat berarti fault di belakang relay/eksternal.
- Superimposed fallback:
  - untuk ground fault bila prefault tersedia.
  - memakai delta voltage/current dari prefault ke fault.
  - dapat mengganti recommended distance jika distance konvensional out-of-range dan superimposed reactance masuk range.
- Status:
  - `VALID`
  - `CHECK`
  - `UNCERTAIN`
- Warning:
  - jarak negatif
  - jarak melebihi line
  - magnitude vs reactance berbeda signifikan
  - estimasi Rf tinggi
  - sudut Zapp menyimpang
  - indikasi load-flow/backfeed.
- Hasil disimpan:
  - `single_ended_result`
  - `remote_single_ended_result`
  - dataframe detail masing-masing.

## Double-End Fault Locator

- Remote setup tidak berada di halaman DE; remote setup ada di `Remote End`.
- Input:
  - local positive sequence phasor `V1/I1`
  - remote positive sequence phasor `V1/I1`
  - line parameter efektif
  - remote record adaptation
  - scenario gangguan.
- Sumber panjang line:
  - Line Parameter
  - Tower Schedule jika tersedia.
- Jika Tower Schedule dipilih:
  - `length_km` diganti dengan `tower_schedule_selected_length_km`
  - `Z1_total = Z1_per_km * length_km`
  - `Z0_total = Z0_per_km * length_km`.
- Positive-sequence equation:
  - `Vlocal(x) = V1L - I1L * Z1_per_km * x`
  - `Vremote(x) = V1R - I1R * Z1_per_km * (L - x)`
  - fault point saat `Vlocal(x) = Vremote(x)`
  - `x = (V1L - V1R + I1R * Z1_per_km * L) / (Z1_per_km * (I1L + I1R))`
- Jika remote current direction `opposite_to_line`, `I1R` diinversi.
- Output utama:
  - distance complex
  - distance km dari local
  - distance percent
  - distance dari remote
  - voltage fault dari local/remote
  - mismatch tegangan fault
  - quality score.
- Quality:
  - penalti distance negatif.
  - penalti distance > line length.
  - penalti imaginary distance.
  - penalti mismatch tegangan fault.
  - score dikunci 0-10.
- Remote adaptation:
  - `auto_adapt_record`: mencoba arah arus, polaritas VT, polaritas CT, dan angle shift.
  - `auto_current_direction_only`: mencoba arah arus remote saja.
  - manual `into_line`.
  - manual `opposite_to_line`.
- Candidate ranking mempertimbangkan:
  - distance keluar line
  - imaginary distance
  - mismatch ratio
  - quality score
  - penalti angle shift
  - penalti polaritas/arah tertentu.
- Visual sync:
  - default metode visual alignment adalah RMS envelope magnitude.
  - opsi sinkronisasi fault cursor/time/visual.
  - fault inception dipakai untuk alignment record.
  - DFT cursor dipakai untuk phasor calculation.
- Optional TWS/time-based:
  - `distance_from_local = (L + propagation_velocity * delta_t) / 2`
  - `distance_from_remote = L - distance_from_local`
  - status warning bila delta time/distance tidak realistis.
- Scenario:
  - internal line fault.
  - reverse/backfeed/external fault.
  - SOTF/parallel/adjacent line diperlakukan sebagai konteks reverse/backfeed, bukan label wajib untuk semua backfeed.
- Diagnostic status:
  - backfeed/reverse mode dapat membuat DE tidak applicable sebagai lokasi utama.
  - SE local/remote dipakai sebagai pembanding arah dan besaran.
- Perbandingan SE pada halaman DE:
  - memakai line parameter efektif yang sama dengan DE.
  - jika DE memakai Tower Schedule, SE comparison juga memakai panjang Tower Schedule.
- Hasil DE disimpan:
  - `two_ended_result`
  - `two_ended_quality`
  - `two_ended_reverse_result`
  - `two_ended_reverse_quality`
  - `two_ended_comparison_df`
  - local/remote SE comparison result.

## R-X Locus

- Sub-view local dan remote.
- Membaca distance relay setting dari sheet `distance_settings`.
- Filter relay setting:
  - GI/Substation
  - Bay
  - search text.
- Zone setting base:
  - default: spreadsheet zone values are primary ohm.
  - optional: spreadsheet values are relay secondary ohm.
  - secondary dikonversi ke primary memakai rasio CT/VT dari Signal Assignment.
- Parameter zona proteksi:
  - `Z1 Res Ph`, `Z1 Res Gnd`
  - `Z2 Res Ph`, `Z2 Res Gnd`
  - `Z3 Res Ph`, `Z3 Res Gnd`
  - kN dan kN angle jika tersedia.
- Zona quadrilateral:
  - memakai X reach dan R reach/resistive reach.
  - phase/gnd reach dipilih sesuai loop fault.
  - Z1/Z2/Z3 digambar sebagai locus proteksi.
- Trajectory:
  - apparent impedance dihitung dari waveform sepanjang window.
  - loop mengikuti fault type/default loop.
  - titik DFT fault cursor ditandai.
  - Z Line Total ditampilkan sebagai referensi.
- Plot focus:
  - default fokus ke relay zones agar zona proteksi terlihat.
  - dapat menampilkan trajectory penuh bila user pilih.
- Summary menampilkan R-X Locus local dan remote sebagai section terpisah agar print tidak menumpuk.

## Summary

- Harus tampil setelah COMTRADE lokal berhasil dibaca.
- Jika kalkulasi belum lengkap, tampilkan `Pending`, bukan halaman blank.
- Menampilkan:
  - metadata local/remote
  - status Signal Assignment, Fault Cursor, Phasor, Fault Type, Line Parameter, SE, DE
  - key results fault type, SE distance, DE distance, DE quality
  - IE source local/remote
  - pre-fault/fault comparison local/remote jika tersedia
  - waveform fokus opsional
  - estimasi penyebab gangguan
  - grafik SE/DE
  - Tower Map Fault Location jika data tower tersedia
  - R-X Locus local/remote
  - warning kualitas DE/HR.
- Grafik SE/DE:
  - memakai hasil SE/DE paling update dari session.
  - scoring SE memakai status `VALID/CHECK/UNCERTAIN` dan warning count.
  - scoring DE memakai `quality_score`.
  - line length mengikuti hasil DE jika DE memakai Tower Schedule.
- Tower Map Summary:
  - default fault source DE jika tersedia.
  - fallback SE.
  - fokus ke dua tower pengapit.
  - Map Settings default tertutup.
  - tabel -5/+5 tower sekitar fault default terbuka saat focus fault.

## Print Report

- Dataframe interaktif disembunyikan saat print.
- Tabel HTML print-friendly ditampilkan sebagai pengganti.
- Penomoran baris tabel default dimulai dari 1, bukan 0, untuk tampilan aplikasi dan print-friendly table.
- Sidebar, toolbar, uploader, dan menu tab tidak dicetak.
- Section report diusahakan tidak terpotong di tengah plot/tabel.
- Plot/tabel besar harus diberi page-break atau block layout yang tidak pecah bila memungkinkan.
- Kontrol peta di Summary disembunyikan dalam expander agar report tidak terganggu.

## Session State Penting

- Local:
  - `assigned_df`
  - `local_transformer_data`
  - `fault_window`
  - `phasors`
  - `prefault_phasors`
  - `fault_type_result`
- Remote:
  - `remote_assigned_df`
  - `remote_transformer_data`
  - `remote_fault_window`
  - `remote_phasors`
  - `remote_prefault_phasors`
  - `remote_fault_type_result`
- Line:
  - `line_param`
- Tower:
  - `tower_schedule_df`
  - `tower_schedule_filtered_df`
  - `tower_schedule_selected_length_km`
  - `tower_schedule_selected_length_source`
  - `tower_schedule_selected_segment`
  - `tower_schedule_selected_ultg`
- HR:
  - `high_resistance_result`
  - remote equivalent if calculated.
- SE:
  - `single_ended_result`
  - `remote_single_ended_result`
  - `single_ended_df`
  - `remote_single_ended_df`
- DE:
  - `two_ended_result`
  - `two_ended_quality`
  - `two_ended_reverse_result`
  - `two_ended_reverse_quality`
  - `two_ended_local_single_result`
  - `two_ended_remote_single_result`
  - `two_ended_remote_sync_shift_s`
  - `two_ended_remote_sync_method`
  - `two_ended_operating_status`
- R-X:
  - `rx_locus_summary_fig_local`
  - `rx_locus_summary_fig_remote`
  - corresponding metadata.

## External Links

- Popup tower dan fault menyediakan:
  - `Open Maps`
  - `Directions`
- Link memakai Google Maps query/directions berbasis latitude/longitude.

## Risiko Teknis Saat Ini

- `app.py` masih sangat besar dan banyak state saling bergantung.
- `st.stop()` di dalam tab workflow bisa menghentikan render seluruh aplikasi karena Streamlit mengeksekusi semua tab pada setiap rerun.
- Summary dirender lebih awal agar tidak blank saat kalkulasi belum lengkap; beberapa hasil yang dibuat setelahnya dapat baru muncul pada rerun berikutnya.
- CSS dan DOM selector Streamlit bawaan cukup rapuh karena struktur Streamlit dapat berubah.
- Folium map punya state internal; gunakan `key`, `center`, dan `zoom` eksplisit saat ingin memaksa fokus fault.
- Banyak kalkulasi sangat bergantung pada kualitas signal assignment, polaritas CT/VT, dan kecocokan line/tower data.

## Guardrail Perubahan Berikutnya

- Jangan menghapus fitur di daftar ini tanpa konfirmasi eksplisit.
- Jangan mengubah arti fault inception vs DFT cursor:
  - fault inception untuk trigger/sync.
  - DFT cursor untuk phasor/locus calculation.
- Jangan mengubah default:
  - auto adaptive fault cursor checkbox off.
  - visual alignment DE = RMS envelope magnitude.
  - distance relay zone base = primary ohm.
  - Tower Schedule line length source tersedia untuk SE dan DE.
  - DE line length source berada di bagian Two-Ended Calculation.
  - Tower Map Summary default fault source = DE jika tersedia.
- Hindari `st.stop()` di dalam tab; gunakan `if/else` lokal pada tab.
- Pisahkan render fungsi per halaman sebelum refactor besar.
- Setelah perubahan, jalankan `python -m py_compile app.py`.
- Untuk perubahan workflow, cek minimal:
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
- Untuk perubahan kalkulasi, validasi minimal:
  - SE local memakai line length source yang dipilih.
  - SE remote memakai line length source yang dipilih.
  - SE comparison pada DE memakai line length source DE.
  - Tower Map fault interpolasi berada di span pengapit berdasarkan `KUMULATIF km`.
  - Summary tidak blank bila kalkulasi belum lengkap.
