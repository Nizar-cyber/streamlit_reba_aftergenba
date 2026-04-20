# REBA Ergonomics Analyzer — Web App

Konversi dari aplikasi desktop (customtkinter) ke web berbasis **Streamlit**.
Fungsi kamera realtime **dihapus**; hanya mendukung **upload foto dari galeri**.

---

## File yang Dibutuhkan

```
reba_web_app.py     ← aplikasi utama
requirements.txt    ← daftar dependensi
```

---

## Cara Deploy (3 pilihan)

### A) Lokal / Intranet Tim (paling cepat)

```bash
# 1. Buat virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependensi
pip install -r requirements.txt

# 3. Jalankan
streamlit run reba_web_app.py
```

Buka browser → `http://localhost:8501`

Agar seluruh tim bisa mengakses dari jaringan lokal:
```bash
streamlit run reba_web_app.py --server.address=0.0.0.0 --server.port=8501
```
Akses via `http://<IP-komputer-server>:8501`

---

### B) Streamlit Community Cloud (gratis, public)

1. Push kedua file ke repository GitHub (public/private)
2. Buka https://share.streamlit.io → **New app**
3. Pilih repo, branch, dan file utama (`reba_web_app.py`)
4. Klik **Deploy** — URL publik tersedia dalam ~2 menit

---

### C) Docker (untuk server produksi)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt reba_web_app.py ./
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "reba_web_app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true"]
```

```bash
docker build -t reba-app .
docker run -p 8501:8501 reba-app
```

---

## Fitur Web App

| Fitur | Status |
|---|---|
| Upload foto dari galeri | ✅ |
| Deteksi pose (MediaPipe) | ✅ |
| Kalkulasi REBA lengkap | ✅ |
| Visualisasi skeleton + label sudut | ✅ |
| Panel hasil (sudut, skor, alur perhitungan) | ✅ |
| Export Excel (berformat, berwarna) | ✅ |
| Kamera realtime | ❌ Dihapus |

---

## Catatan Teknis

- Gunakan `opencv-python-headless` (bukan `opencv-python`) agar tidak ada konflik GUI di server.
- Mediapipe ≥ 0.10 sudah mendukung Python 3.11.
- Gambar yang diupload **tidak disimpan ke server**; diproses langsung di memori.
