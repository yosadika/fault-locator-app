---
description: Analisis lingkup tugas lalu delegasikan ke subagent dengan model sesuai kompleksitas (Haiku/Sonnet/Opus).
argument-hint: <deskripsi tugas yang ingin dikerjakan>
---

Tugas yang diminta: **$ARGUMENTS**

Jalankan prosedur routing berikut. **Lakukan scoping sendiri** (jangan delegasikan tahap ini), lalu delegasikan eksekusi ke subagent yang tepat.

## 1. Scoping
Baca secukupnya file yang relevan (jangan berlebihan) untuk memperkirakan:
- Berapa file/modul yang kemungkinan disentuh?
- Apakah menyentuh orkestrasi `app.py`?
- Apakah mengubah atau menambah kunci `st.session_state` yang dipakai lintas halaman? (rujuk bagian **"Session State Penting"** di `PRD.md`)
- Apakah ini integrasi lintas halaman (Summary / SE / DE / Tower Map / Weather / R-X Locus saling terkait)?

## 2. Tentukan tier

| Tier | Kriteria | Subagent | Model |
|---|---|---|---|
| **LIGHT** | 1 file, sepele (typo, rename lokal, teks UI, formatting), tanpa ubah session_state | `porlung-light` | Haiku |
| **STANDARD** | 1-2 file dalam satu area fitur, logika dalam satu tab/helper, tanpa ubah kontrak session_state lintas halaman | `porlung-standard` | Sonnet |
| **HEAVY** | `app.py` + ≥2 modul, ATAU ubah/tambah session_state keys lintas halaman, ATAU refactor / ubah alur workflow | `porlung-heavy` | Opus |

Jika di perbatasan dua tier, **pilih tier lebih tinggi** (lebih aman untuk menjaga kontrak state).

## 3. Konfirmasi + delegasikan
Tampilkan satu baris ringkas:
`Tier: <LIGHT/STANDARD/HEAVY> — <alasan singkat> — estimasi file: <daftar>`

Lalu spawn subagent yang sesuai via tool **Agent** (`subagent_type` = nama subagent di tabel), dengan prompt berisi deskripsi tugas lengkap **$ARGUMENTS** plus konteks scoping yang sudah kamu kumpulkan (file relevan, kontrak state yang terlibat) agar subagent tidak mulai dari nol sepenuhnya.

## Catatan
- Jika sesi utama ini sudah berjalan di Haiku dan tier-nya LIGHT, kamu boleh mengerjakannya langsung tanpa spawn (menghindari cold-start yang justru lebih mahal).
- Subagent mulai dingin dan membaca ulang `CLAUDE.md`/`PRD.md`; sertakan ringkasan scoping di prompt delegasi untuk menghemat.
- Setelah subagent selesai, relay ringkasan hasilnya (file diubah + hasil py_compile) ke user.
