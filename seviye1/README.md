# İstanbul Hal Ürünlerinin Makine Öğrenmesi ile Fiyat Tahmini ve Satın Alma Planlamasında Kullanılması (Seviye 1)

**Kahramanmaraş Sütçü İmam Üniversitesi — Bilgisayar Mühendisliği Bölümü — Makine Öğrenmesi Dersi Projesi**  
**Geliştiriciler:** Burak Demirel, Ebubekir Samet Gözübenli, Resul Kılınç

Bu proje; İstanbul Büyükşehir Belediyesi (İBB) Hal Müdürlüğü tarafından yayımlanan 2016–2026 dönemi sebze, meyve ve ithal ürün fiyat bültenlerini kullanarak bir sonraki hal işlem gününün **minimum (`hedef_min`)** ve **maksimum (`hedef_max`)** fiyatlarını tahmin eden, veri sızıntısından korunmuş bir makine öğrenmesi hattı ve çoklu ürün sepeti destekli Türkçe bir **Satın Alma Karar Destek Sistemi** sunar.

Projenin tüm aşamaları (Aşama 1 – Aşama 8) tamamlanmış, 47 birim ve entegrasyon testi ile 4 bağımsız doğrulama betiği (`dogrula_asama1.py`, `dogrula_asama2.py`, `dogrula_asama3.py`, `dogrula_asama4_6.py`) başarıyla doğrulanmıştır. Kapsamlı akademik rapor `PROJE_RAPORU.md` dosyasındadır.

---

## 1. Hızlı Kurulum ve Çalıştırma

### Bağımlılıkların Kurulumu
```bash
pip install -r requirements.txt
```

### Web Uygulamasını (Streamlit) Başlatma
Kaydedilmiş modelleri ve doğrulanmış veri çıktılarını yeniden eğitim yapmadan doğrudan yükleyen Türkçe web uygulamasını başlatmak için:
```bash
streamlit run app.py
```

### Tüm Birim ve Entegrasyon Testlerini Çalıştırma (47 Test)
```bash
python -m unittest discover -s . -p "test_*.py" -v
```

### Bağımsız Doğrulama Betiklerini Çalıştırma
```bash
python dogrula_asama1.py
python dogrula_asama2.py
python dogrula_asama3.py
python dogrula_asama4_6.py
```

---

## 2. Aşama Bazlı Çalıştırma Komutları

Orijinal CSV dosyaları değiştirilmeden tüm aşamalar deterministik olarak yeniden üretilebilir:

1. **Aşama 1 — Veri Analizi ve Temizleme (`382.897` ham kayıt $\rightarrow$ `380.985` temiz gözlem, `299` seri):**
   ```bash
   python asama1_veri.py
   ```
2. **Aşama 2 — Özellik Mühendisliği ve Kronolojik Bölme (`54` özellik, `354.896` etiketli örnek):**
   ```bash
   python asama2_ozellikler.py
   ```
   - Eğitim (`<= 2024-10-26`): `303.394` örnek
   - Doğrulama (`2024-10-27` – `2025-10-26`): `30.361` örnek
   - Test (`2025-10-27` – `2026-10-26`): `21.141` örnek
   - Son gün (`2026-10-26`) özellikleri hazır aktif seri: `91` seri
3. **Aşama 3 — Naive Baseline Referans Değerlendirmesi:**
   ```bash
   python asama3_baseline.py
   ```
4. **Aşama 4 — Makine Öğrenmesi Modelleri ve Fiyat Sürekliliği Araştırması:**
   ```bash
   python asama4_modeller.py
   ```
   - `Linear Regression`, `Random Forest Regressor` ve `XGBoost Regressor` modellerini fiyat farkı ($\Delta y(t+1) = y(t+1) - y(t)$) formülasyonuyla eğitir, `ciktilar/asama4/modeller/*.joblib` ve `model_kayit_defteri.json` dosyalarını üretir.
   - `2025-10-03` yapısal fiyat sürekliliği kırılmasını analiz ederek `ciktilar/asama4/FIYAT_SUREKLILIGI_RAPORU.md` raporunu oluşturur.
5. **Aşama 5 — Model Karşılaştırma ve Görsel Raporlama:**
   ```bash
   python asama5_karsilastirma.py
   ```
   - 4 modeli doğrulama ve test kümelerinde karşılaştırır ve `ciktilar/asama5/grafikler/` altında 6 yüksek çözünürlüklü PNG grafik üretir.
6. **Aşama 6 — Satın Alma Karar Destek ve Çoklu Ürün Sepeti Simülasyonu:**
   ```bash
   python asama6_satin_alma.py
   ```

---

## 3. Dosya ve Klasör Yapısı

- `asama1_veri.py`, `dogrula_asama1.py`, `test_asama1.py` — Aşama 1 veri temizliği ve testleri
- `asama2_ozellikler.py`, `dogrula_asama2.py`, `test_asama2.py` — Aşama 2 özellik mühendisliği ve testleri
- `asama3_baseline.py`, `dogrula_asama3.py`, `test_asama3.py` — Aşama 3 Naive Baseline ve testleri
- `tahmin_servisi.py` — Ortak ön işleme (`SeriesHistoryStatsEncoder`), model paketleme (`ResidualTargetModelArtifact`), `min <= max` tutarlılık denetimi ve tahmin servisi (`HalPredictionService`)
- `asama4_modeller.py` — Aşama 4 model eğitimi, hiperparametre seçimi ve fiyat sürekliliği analizi
- `asama5_karsilastirma.py` — Aşama 5 çok düzeyli model karşılaştırması ve grafik üretimi
- `asama6_satin_alma.py` — Aşama 6 tekil ürün ve çoklu ürün sepeti satın alma karar destek modülü
- `app.py` — 6 sayfalı Türkçe Streamlit & Plotly web uygulaması
- `dogrula_asama4_6.py`, `test_asama4_8.py` — Aşama 4–8 bağımsız doğrulama ve 12 kapsamlı entegrasyon testi
- `PROJE_RAPORU.md` — 20 bölümlük kapsamlı akademik proje raporu
- `ciktilar/asama1..6/` — Tüm aşamaların doğrulanmış veri, model, manifesto ve rapor çıktıları
