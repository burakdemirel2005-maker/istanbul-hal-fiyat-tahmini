# 🥬 İstanbul Hal Ürünlerinin Makine Öğrenmesi ile Fiyat Tahmini ve Satın Alma Karar Destek Sistemi

**Kahramanmaraş Sütçü İmam Üniversitesi — Bilgisayar Mühendisliği Bölümü — Makine Öğrenmesi Dersi Projesi**  
**Proje Ekibi:** Burak Demirel • Ebubekir Samet Gözübenli • Resul Kılınç

Bu depo; İstanbul Büyükşehir Belediyesi (İBB) Hal Müdürlüğü tarafından yayımlanan **2016–2026** dönemi sebze, meyve ve ithal ürün fiyat bültenlerini (`382.897` ham kayıt $\rightarrow$ `380.985` temiz gözlem, `299` ürün serisi) kullanarak bir sonraki hal işlem gününün **minimum (`hedef_min`)** ve **maksimum (`hedef_max`)** fiyatlarını tahmin eden makine öğrenmesi modellerini (`Naive Baseline`, `Linear Regression`, `Random Forest`, `XGBoost`), **Satın Alma Karar Destek Sistemini** ve **6 sayfalı Türkçe Streamlit web uygulamasını** içerir.

---

## 🚀 Uygulamayı Açma Yöntemleri

### Yöntem 1: Bilgisayarınızda (Yerel Ortamda) Çalıştırma

1. **Depoyu klonlayın ve proje klasörüne girin:**
   ```bash
   git clone https://github.com/burakdemirel2005-maker/istanbul-hal-fiyat-tahmini.git
   cd istanbul-hal-fiyat-tahmini/seviye1
   ```

2. **Sanal ortam oluşturun ve kütüphaneleri yükleyin:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # Windows için: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Streamlit Web Uygulamasını başlatın:**
   ```bash
   streamlit run app.py
   ```
   Komutu çalıştırdığınızda tarayıcınızda otomatik olarak **`http://localhost:8501`** adresi açılacaktır.

---

### Yöntem 2: GitHub Üzerinden Ücretsiz Bulutta Açma (Streamlit Community Cloud)

Bilgisayarınıza hiçbir şey kurmadan, doğrudan GitHub deponuz üzerinden herkese açık bir web linki oluşturmak için:

1. [https://share.streamlit.io](https://share.streamlit.io) adresine gidin ve **GitHub hesabınızla (`burakdemirel2005-maker`)** giriş yapın.
2. Sağ üstteki **"Create app"** (veya **"New app"**) butonuna tıklayın.
3. Açılan ekranda şu bilgileri seçin:
   - **Repository:** `burakdemirel2005-maker/istanbul-hal-fiyat-tahmini`
   - **Branch:** `main`
   - **Main file path:** `seviye1/app.py`
4. **"Deploy!"** butonuna tıklayın.
5. Yaklaşık 1–2 dakika içinde uygulamanız bulutta derlenecek ve size özel `https://....streamlit.app` uzantılı canlı web linki açılacaktır. Eğitilmiş modeller (`seviye1/ciktilar/asama4/modeller/*.joblib`) ve sıkıştırılmış veri dosyaları (`temiz_veri.csv.gz`) depoda hazır olduğu için yeniden model eğitimi yapmadan anında açılır.

---

### Yöntem 3: GitHub Codespaces ile Tarayıcıda Açma

1. GitHub depo sayfasında yeşil **`<> Code`** butonuna tıklayın.
2. **`Codespaces`** sekmesine geçip **`Create codespace on main`** butonuna basın.
3. Açılan bulut terminalinde şu iki komutu çalıştırın:
   ```bash
   pip install -r requirements.txt
   streamlit run seviye1/app.py
   ```
4. Sağ altta çıkan **"Open in Browser"** bildirimine tıklayarak uygulamayı doğrudan tarayıcıda görüntüleyin.

---

## 📊 Proje Özeti ve Model Performansı

| Bölüm | Hedef | Model | MAE (TL) | RMSE (TL) | MSE | R² | Baseline RMSE İyileşme (%) |
|---|---|---|---:|---:|---:|---:|---:|
| **Doğrulama** (`N=30.361`) | `hedef_min` | **Naive Baseline** | 0.5839 | 3.9134 | 15.3144 | 0.994031 | +0.00% |
| **Doğrulama** (`N=30.361`) | `hedef_min` | **Linear Regression** | 0.7563 | 3.9161 | 15.3355 | 0.994023 | -0.07% |
| **Doğrulama** (`N=30.361`) | `hedef_min` | **Random Forest (Ana Model)** | 0.6467 | **3.9067** | **15.2627** | **0.994052** | **+0.17%** |
| **Doğrulama** (`N=30.361`) | `hedef_min` | **XGBoost** | **0.6098** | **3.9117** | **15.3016** | **0.994036** | **+0.04%** |
| **Doğrulama** (`N=30.361`) | `hedef_max` | **Naive Baseline** | 0.7222 | 4.5402 | 20.6137 | 0.995196 | +0.00% |
| **Doğrulama** (`N=30.361`) | `hedef_max` | **Linear Regression** | 0.9267 | 4.5524 | 20.7247 | 0.995171 | -0.27% |
| **Doğrulama** (`N=30.361`) | `hedef_max` | **Random Forest (Ana Model)** | 0.8074 | **4.5305** | **20.5256** | **0.995217** | **+0.21%** |
| **Doğrulama** (`N=30.361`) | `hedef_max` | **XGBoost** | **0.7557** | **4.5370** | **20.5848** | **0.995203** | **+0.07%** |
| **Test** (`N=21.141`) | `hedef_min` | **Naive Baseline** | **5.0700** | 16.3458 | 267.1857 | 0.974446 | +0.00% |
| **Test** (`N=21.141`) | `hedef_min` | **Linear Regression** | 5.0781 | **15.7254** | **247.2881** | **0.976349** | **+3.80%** |
| **Test** (`N=21.141`) | `hedef_min` | **Random Forest (Ana Model)** | 5.1005 | **16.3276** | **266.5910** | **0.974503** | **+0.11%** |
| **Test** (`N=21.141`) | `hedef_min` | **XGBoost** | 5.0909 | 16.3535 | 267.4360 | 0.974422 | -0.05% |
| **Test** (`N=21.141`) | `hedef_max` | **Naive Baseline** | 6.0526 | 18.4987 | 342.2001 | 0.980335 | +0.00% |
| **Test** (`N=21.141`) | `hedef_max` | **Linear Regression** | **5.9220** | **17.7083** | **313.5851** | **0.981979** | **+4.27%** |
| **Test** (`N=21.141`) | `hedef_max` | **Random Forest (Ana Model)** | 6.0666 | **18.4709** | **341.1751** | **0.980393** | **+0.15%** |
| **Test** (`N=21.141`) | `hedef_max` | **XGBoost** | 6.0749 | 18.5216 | 343.0507 | 0.980286 | -0.12% |

---

## 📁 Depo İçeriği

- [`seviye1/app.py`](seviye1/app.py) — 6 sayfalı Türkçe Streamlit & Plotly web uygulaması
- [`seviye1/PROJE_RAPORU.md`](seviye1/PROJE_RAPORU.md) — 20 bölümlük kapsamlı akademik proje raporu
- [`seviye1/asama1_veri.py`](seviye1/asama1_veri.py) — Aşama 1 veri analizi ve temizleme
- [`seviye1/asama2_ozellikler.py`](seviye1/asama2_ozellikler.py) — Aşama 2 özellik mühendisliği ve kronolojik bölme
- [`seviye1/asama3_baseline.py`](seviye1/asama3_baseline.py) — Aşama 3 Naive Baseline referans modeli
- [`seviye1/asama4_modeller.py`](seviye1/asama4_modeller.py) & [`seviye1/tahmin_servisi.py`](seviye1/tahmin_servisi.py) — Aşama 4 ML modelleri ve tahmin servisi
- [`seviye1/asama5_karsilastirma.py`](seviye1/asama5_karsilastirma.py) — Aşama 5 model karşılaştırma ve grafik üretimi
- [`seviye1/asama6_satin_alma.py`](seviye1/asama6_satin_alma.py) — Aşama 6 satın alma karar destek ve çoklu ürün sepeti modülü
- [`seviye1/test_asama4_8.py`](seviye1/test_asama4_8.py) — 47 birim ve entegrasyon testi (`python -m unittest discover -s seviye1 -p "test_*.py" -v`)
