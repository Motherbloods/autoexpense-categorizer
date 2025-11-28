# AutoExpense Categorizer

AutoExpense Categorizer adalah proyek machine learning untuk mengklasifikasikan pengeluaran ke dalam kategori yang relevan secara otomatis. Proyek ini menyediakan alat dan kode yang diperlukan untuk membangun, melatih, dan menerapkan model klasifikasi pengeluaran.

## Fitur Utama

- **Kode untuk Membuat dan Melatih Model**: Skrip Python untuk mempersiapkan data dan melatih model klasifikasi pengeluaran.
- **Model Terlatih (.pkl)**: File model yang sudah dilatih dan siap digunakan untuk klasifikasi pengeluaran.
- **Dataset**: Dataset pengeluaran yang digunakan untuk melatih dan menguji model klasifikasi.
- **API**: Implementasi API untuk mengakses model dan melakukan prediksi terhadap data pengeluaran baru.
- **Integrasi**: Contoh integrasi dengan aplikasi seperti Telegram untuk klasifikasi pengeluaran secara real-time.

## Teknologi

- **Python**: Bahasa pemrograman utama untuk pengembangan proyek.
- **scikit-learn**: Library machine learning untuk membangun dan melatih model klasifikasi.
- **Flask/FastAPI**: Framework untuk mengimplementasikan API.
- **pandas**: Library untuk manipulasi dan analisis data.
- **numpy**: Library untuk komputasi numerik.

## Struktur Proyek

```
AutoExpense-Categorizer/
├── data/
│   ├── dataset.json
├── models/
│   └── expense_classifier.pkl      # Model yang telah dilatih
├── src/
│   ├── prepare_data.py             # Skrip untuk mempersiapkan data
│   ├── train_model.py              # Skrip untuk melatih model
│   └── predict.py                  # Skrip untuk melakukan prediksi
├── api/
│   ├── app.py                      # Implementasi API
│   └── requirements.txt            # Dependensi untuk API
├── requirements.txt                # Dependensi proyek
└── README.md                       # Dokumentasi proyek
```

## Cara Penggunaan

### Instalasi

1. Clone repositori ini:
   ```
   git clone https://github.com/username/AutoExpense-Categorizer.git
   cd AutoExpense-Categorizer
   ```

2. Buat dan aktifkan virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # Untuk Linux/Mac
   venv\Scripts\activate     # Untuk Windows
   ```

3. Install dependensi:
   ```
   pip install -r requirements.txt
   ```

### Melatih Model

Untuk melatih model dari awal:

```
python src/train_model.py
```

### Menggunakan API

1. Jalankan API:
   ```
   cd api
   python app.py
   ```

2. API akan berjalan di `http://localhost:8080` (untuk Flask) atau `http://localhost:8000` (untuk FastAPI).

3. Contoh permintaan API:
   ```
   POST /predict
   {
     "description": "Makan di restoran",
     "amount": 180800,
     "date": "2023-03-15"
   }
   ```

4. Respons:
   ```
   {
     "category": "Makanan",
     "confidence": 0.92
   }
   ```

## Kontribusi

Kontribusi untuk proyek ini sangat diterima. Silakan ikuti langkah-langkah berikut:

1. Fork repositori
2. Buat branch fitur (`git checkout -b feature/amazing-feature`)
3. Commit perubahan Anda (`git commit -m 'Menambahkan fitur amazing'`)
4. Push ke branch (`git push origin feature/amazing-feature`)
5. Buka Pull Request

## Kontak

Nama Proyek: [AutoExpense Categorizer](https://github.com/Motherbloods/autoexpense-categorizer)

Pengembang: [Habib Risky Kurniawan](mailto:habibskh06@gmail.com)
