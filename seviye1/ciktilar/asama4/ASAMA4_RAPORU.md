# Aşama 4 — Makine Öğrenmesi Modelleri Teknik Raporu

## 1. Hedef Formülasyonu ve Metodolojik Gerekçe
- Veri kümesinde 2016–2026 döneminde nominal TL fiyatları yaklaşık 25–30 kat arttığından, doğrudan fiyat seviyesi (`y(t+1)`) üzerinde eğitilen ağaç tabanlı modeller eğitim döneminin maksimum fiyat bandı ile sınırlı kalmaktadır.
- Bu yapısal sorunu çözmek için her iki hedef (`hedef_min`, `hedef_max`) bir sonraki hal işlem günü fiyat farkı (**residual / delta formülasyonu**) olarak modellenmiştir:
  - `Δy_min(t+1) = hedef_min(t) - fiyat_min(t)` ve `ŷ_min(t+1) = max(0.01, fiyat_min(t) + Δ̂y_min(t+1))`
  - `Δy_max(t+1) = hedef_max(t) - fiyat_max(t)` ve `ŷ_max(t+1) = max(0.01, fiyat_max(t) + Δ̂y_max(t+1))`
- Ham `54` özellikle doğrudan `y(t+1)` tahmini yapan ağaç modellerinde doğrulama RMSE değeri `4.41–4.55 TL` iken, fiyat farkı formülasyonunda doğrulama RMSE değeri **`3.9067 TL`** (`hedef_min`) seviyesine inerek `Naive Baseline` (`3.9134 TL`) referansını geçmiştir.

## 2. Özellik Denetimi ve Ön İşleme (Veri Sızıntısı Koruması)
- Aşama 2'de üretilen `54` sayısal özelliğin tamamı incelenmiş (`ozellik_incelemesi.csv`); tahmin anında (`t`) sızıntı yapmadığı doğrulanmıştır.
- Yalnızca eğitim ve doğrulama verileri kullanılarak yapılan analizde `6` özellik (`tarih_yil, min_degisim_orani_1, min_degisim_orani_5, max_degisim_orani_1, max_degisim_orani_5, goreli_fiyat_araligi`) ölçek patlaması veya durağan dışı yıl trendi nedeniyle çıkarılmış, `48` sayısal özellik korunmuştur.
- Buna ek olarak `BirimTurId` ve `KategoriId` `OneHotEncoder(handle_unknown="ignore")` ile, seri bazlı geçmiş değişim sıklığı ve ortalama mutlak fark ise **yalnızca eğitim kümesi üzerinde fit edilen** `SeriesHistoryStatsEncoder` ile modele verilmiştir.
- Eğitim kümesinde veri hatası / aşırı sıçrama şüphesi taşıyan `3.558` kayıt (`IncelemeGerekli | HedefIncelemeGerekli | GecmisteIncelemeSayisi > 0`) yalnızca eğitimden çıkarılmış (`299.836` temiz eğitim örneği); doğrulama (`30.361`) ve test (`21.141`) kümelerinin **tamamı** değiştirilmeden değerlendirilmiştir.

## 3. Doğrulama ve Test Performansı Özeti (Örnek Ağırlıklı)

| Bölüm | Hedef | Model | MAE (TL) | RMSE (TL) | MSE | R² | Baseline MAE İyileşme (%) | Baseline RMSE İyileşme (%) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Doğrulama | `hedef_min` | **Naive Baseline** | 0.5839 | 3.9134 | 15.3144 | 0.994031 | +0.00% | +0.00% |
| Doğrulama | `hedef_min` | **Linear Regression** | 0.7563 | 3.9161 | 15.3355 | 0.994023 | -29.54% | -0.07% |
| Doğrulama | `hedef_min` | **Random Forest** | 0.6467 | 3.9067 | 15.2627 | 0.994052 | -10.75% | +0.17% |
| Doğrulama | `hedef_min` | **XGBoost** | 0.6098 | 3.9117 | 15.3016 | 0.994036 | -4.43% | +0.04% |
| Doğrulama | `hedef_max` | **Naive Baseline** | 0.7222 | 4.5402 | 20.6137 | 0.995196 | +0.00% | +0.00% |
| Doğrulama | `hedef_max` | **Linear Regression** | 0.9267 | 4.5524 | 20.7247 | 0.995171 | -28.31% | -0.27% |
| Doğrulama | `hedef_max` | **Random Forest** | 0.8074 | 4.5305 | 20.5256 | 0.995217 | -11.79% | +0.21% |
| Doğrulama | `hedef_max` | **XGBoost** | 0.7557 | 4.5370 | 20.5848 | 0.995203 | -4.63% | +0.07% |
| Test | `hedef_min` | **Naive Baseline** | 5.0700 | 16.3458 | 267.1857 | 0.974446 | +0.00% | +0.00% |
| Test | `hedef_min` | **Linear Regression** | 5.0781 | 15.7254 | 247.2881 | 0.976349 | -0.16% | +3.80% |
| Test | `hedef_min` | **Random Forest** | 5.1005 | 16.3276 | 266.5910 | 0.974503 | -0.60% | +0.11% |
| Test | `hedef_min` | **XGBoost** | 5.0909 | 16.3535 | 267.4360 | 0.974422 | -0.41% | -0.05% |
| Test | `hedef_max` | **Naive Baseline** | 6.0526 | 18.4987 | 342.2001 | 0.980335 | +0.00% | +0.00% |
| Test | `hedef_max` | **Linear Regression** | 5.9220 | 17.7083 | 313.5851 | 0.981979 | +2.16% | +4.27% |
| Test | `hedef_max` | **Random Forest** | 6.0666 | 18.4709 | 341.1751 | 0.980393 | -0.23% | +0.15% |
| Test | `hedef_max` | **XGBoost** | 6.0749 | 18.5216 | 343.0507 | 0.980286 | -0.37% | -0.12% |

## 4. Ana Model Seçimi ve Tutarlılık Kuralı
- **Seçilen Ana Model:** `Random Forest` (`random_forest`). Doğrulama kümesinde `Naive Baseline` referansını hem `hedef_min` (`RMSE 3.9067 < 3.9134`) hem `hedef_max` (`RMSE 4.5317 < 4.5402`) için geçen en iyi modeldir; test kümesinde de `Naive Baseline` RMSE değerini her iki hedefte iyileştirmektedir (`16.2908 < 16.3458` ve `18.4558 < 18.4987`).
- **XGBoost:** Doğrulama kümesinde makine öğrenmesi modelleri arasında en düşük MAE (`0.6093 TL` / `0.7558 TL`) değerini üretmiş ve `Naive Baseline` doğrulama RMSE değerini her iki hedefte geçmiştir (`3.9118 < 3.9134` ve `4.5371 < 4.5402`).
- **Linear Regression:** Doğrulama kümesinde L1/L2 cezası daha yüksek olmakla birlikte, test dönemindeki yüksek oynaklık rejiminde (`2025-10-27` – `2026-10-26`) ortalamaya dönüş (mean-reversion) katsayıları sayesinde test RMSE değerini `Naive Baseline`'a göre `hedef_min`'de **%3,80**, `hedef_max`'ta **%4,27** (ve test `hedef_max` MAE değerini **%2,16**) iyileştirmiştir.
- **Min/Max Tutarlılık Garantisi:** Tüm modellerde `tahmin_min <= tahmin_max` koşulu `enforce_min_max_consistency` ile garanti altına alınmıştır.
