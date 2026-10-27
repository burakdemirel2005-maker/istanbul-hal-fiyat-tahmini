# Aşama 5 — Model Karşılaştırma ve Değerlendirme Raporu

## 1. Karşılaştırma Protokolü
- Tüm modeller (`Naive Baseline`, `Linear Regression`, `Random Forest`, `XGBoost`) **birebir aynı** doğrulama (`30.361` kayıt, `152` seri) ve test (`21.141` kayıt, `123` seri) örnekleri üzerinde değerlendirilmiştir.
- Model seçimi ve hiperparametre kilitlemesi yalnızca doğrulama kümesi üzerinde tamamlanmış, test kümesi tek seferlik nihai performans raporlaması için kullanılmıştır.

## 2. Ana Karşılaştırma Tablosu (Örnek Ağırlıklı ve Seri Eşit Ağırlıklı)

| Bölüm | Hedef | Model | MAE (TL) | RMSE (TL) | MSE | R² | Baseline MAE İyileşme (%) | Baseline RMSE İyileşme (%) | Seri Ort. MAE | Seri Ort. RMSE | Baseline'dan İyi Seri Sayısı |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dogrulama | `hedef_min` | **Naive Baseline** | 0.5839 | 3.9134 | 15.3144 | 0.994031 | +0.00% | +0.00% | 0.5958 | 2.6239 | 0/152 |
| dogrulama | `hedef_min` | **Linear Regression** | 0.7563 | 3.9161 | 15.3355 | 0.994023 | -29.54% | -0.07% | 0.7997 | 2.6536 | 52/152 |
| dogrulama | `hedef_min` | **Random Forest** | 0.6467 | 3.9067 | 15.2627 | 0.994052 | -10.75% | +0.17% | 0.6715 | 2.6384 | 84/152 |
| dogrulama | `hedef_min` | **XGBoost** | 0.6098 | 3.9117 | 15.3016 | 0.994036 | -4.43% | +0.04% | 0.6252 | 2.6281 | 83/152 |
| dogrulama | `hedef_max` | **Naive Baseline** | 0.7222 | 4.5402 | 20.6137 | 0.995196 | +0.00% | +0.00% | 0.7478 | 3.0725 | 0/152 |
| dogrulama | `hedef_max` | **Linear Regression** | 0.9267 | 4.5524 | 20.7247 | 0.995171 | -28.31% | -0.27% | 0.9940 | 3.1050 | 49/152 |
| dogrulama | `hedef_max` | **Random Forest** | 0.8074 | 4.5305 | 20.5256 | 0.995217 | -11.79% | +0.21% | 0.8579 | 3.0965 | 97/152 |
| dogrulama | `hedef_max` | **XGBoost** | 0.7557 | 4.5370 | 20.5848 | 0.995203 | -4.63% | +0.07% | 0.7977 | 3.0841 | 107/152 |
| test | `hedef_min` | **Naive Baseline** | 5.0700 | 16.3458 | 267.1857 | 0.974446 | +0.00% | +0.00% | 7.3697 | 10.9339 | 0/123 |
| test | `hedef_min` | **Linear Regression** | 5.0781 | 15.7254 | 247.2881 | 0.976349 | -0.16% | +3.80% | 7.2995 | 10.5612 | 99/123 |
| test | `hedef_min` | **Random Forest** | 5.1005 | 16.3276 | 266.5910 | 0.974503 | -0.60% | +0.11% | 7.4049 | 10.9073 | 93/123 |
| test | `hedef_min` | **XGBoost** | 5.0909 | 16.3535 | 267.4360 | 0.974422 | -0.41% | -0.05% | 7.3960 | 10.9303 | 80/123 |
| test | `hedef_max` | **Naive Baseline** | 6.0526 | 18.4987 | 342.2001 | 0.980335 | +0.00% | +0.00% | 8.8331 | 12.6695 | 0/123 |
| test | `hedef_max` | **Linear Regression** | 5.9220 | 17.7083 | 313.5851 | 0.981979 | +2.16% | +4.27% | 8.5917 | 12.0906 | 118/123 |
| test | `hedef_max` | **Random Forest** | 6.0666 | 18.4709 | 341.1751 | 0.980393 | -0.23% | +0.15% | 8.8468 | 12.6115 | 109/123 |
| test | `hedef_max` | **XGBoost** | 6.0749 | 18.5216 | 343.0507 | 0.980286 | -0.37% | -0.12% | 8.8610 | 12.6672 | 84/123 |

## 3. Eğitimde Görülen ve Görülmeyen Seriler Bazında Performans

| Bölüm | Hedef | Alt Grup | Model | Örnek | Seri | MAE (TL) | RMSE (TL) | Baseline RMSE İyileşme (%) |
|---|---|---|---|---:|---:|---:|---:|---:|
| dogrulama | `hedef_min` | Eğitimde Görülmedi | **Naive Baseline** | 260 | 2 | 0.2385 | 1.2341 | +0.00% |
| dogrulama | `hedef_min` | Eğitimde Görülmedi | **Linear Regression** | 260 | 2 | 0.3236 | 1.2275 | +0.54% |
| dogrulama | `hedef_min` | Eğitimde Görülmedi | **Random Forest** | 260 | 2 | 0.2975 | 1.2240 | +0.82% |
| dogrulama | `hedef_min` | Eğitimde Görülmedi | **XGBoost** | 260 | 2 | 0.2680 | 1.2307 | +0.28% |
| dogrulama | `hedef_min` | Eğitimde Görüldü | **Naive Baseline** | 30,101 | 150 | 0.5869 | 3.9285 | +0.00% |
| dogrulama | `hedef_min` | Eğitimde Görüldü | **Linear Regression** | 30,101 | 150 | 0.7601 | 3.9313 | -0.07% |
| dogrulama | `hedef_min` | Eğitimde Görüldü | **Random Forest** | 30,101 | 150 | 0.6497 | 3.9219 | +0.17% |
| dogrulama | `hedef_min` | Eğitimde Görüldü | **XGBoost** | 30,101 | 150 | 0.6127 | 3.9269 | +0.04% |
| dogrulama | `hedef_max` | Eğitimde Görülmedi | **Naive Baseline** | 260 | 2 | 0.3154 | 1.6172 | +0.00% |
| dogrulama | `hedef_max` | Eğitimde Görülmedi | **Linear Regression** | 260 | 2 | 0.4450 | 1.5951 | +1.37% |
| dogrulama | `hedef_max` | Eğitimde Görülmedi | **Random Forest** | 260 | 2 | 0.3617 | 1.6065 | +0.66% |
| dogrulama | `hedef_max` | Eğitimde Görülmedi | **XGBoost** | 260 | 2 | 0.3355 | 1.6127 | +0.28% |
| dogrulama | `hedef_max` | Eğitimde Görüldü | **Naive Baseline** | 30,101 | 150 | 0.7258 | 4.5573 | +0.00% |
| dogrulama | `hedef_max` | Eğitimde Görüldü | **Linear Regression** | 30,101 | 150 | 0.9309 | 4.5696 | -0.27% |
| dogrulama | `hedef_max` | Eğitimde Görüldü | **Random Forest** | 30,101 | 150 | 0.8112 | 4.5476 | +0.21% |
| dogrulama | `hedef_max` | Eğitimde Görüldü | **XGBoost** | 30,101 | 150 | 0.7593 | 4.5541 | +0.07% |
| test | `hedef_min` | Eğitimde Görülmedi | **Naive Baseline** | 376 | 4 | 23.2447 | 67.7867 | +0.00% |
| test | `hedef_min` | Eğitimde Görülmedi | **Linear Regression** | 376 | 4 | 22.3941 | 64.6101 | +4.69% |
| test | `hedef_min` | Eğitimde Görülmedi | **Random Forest** | 376 | 4 | 23.2472 | 67.7727 | +0.02% |
| test | `hedef_min` | Eğitimde Görülmedi | **XGBoost** | 376 | 4 | 23.2629 | 67.8293 | -0.06% |
| test | `hedef_min` | Eğitimde Görüldü | **Naive Baseline** | 20,765 | 119 | 4.7409 | 13.7412 | +0.00% |
| test | `hedef_min` | Eğitimde Görüldü | **Linear Regression** | 20,765 | 119 | 4.7645 | 13.2732 | +3.41% |
| test | `hedef_min` | Eğitimde Görüldü | **Random Forest** | 20,765 | 119 | 4.7719 | 13.7204 | +0.15% |
| test | `hedef_min` | Eğitimde Görüldü | **XGBoost** | 20,765 | 119 | 4.7619 | 13.7466 | -0.04% |
| test | `hedef_max` | Eğitimde Görülmedi | **Naive Baseline** | 376 | 4 | 22.9601 | 63.7598 | +0.00% |
| test | `hedef_max` | Eğitimde Görülmedi | **Linear Regression** | 376 | 4 | 22.4903 | 61.0378 | +4.27% |
| test | `hedef_max` | Eğitimde Görülmedi | **Random Forest** | 376 | 4 | 22.9807 | 63.7242 | +0.06% |
| test | `hedef_max` | Eğitimde Görülmedi | **XGBoost** | 376 | 4 | 22.9938 | 63.7836 | -0.04% |
| test | `hedef_max` | Eğitimde Görüldü | **Naive Baseline** | 20,765 | 119 | 5.7465 | 16.5766 | +0.00% |
| test | `hedef_max` | Eğitimde Görüldü | **Linear Regression** | 20,765 | 119 | 5.6220 | 15.8683 | +4.27% |
| test | `hedef_max` | Eğitimde Görüldü | **Random Forest** | 20,765 | 119 | 5.7603 | 16.5476 | +0.18% |
| test | `hedef_max` | Eğitimde Görüldü | **XGBoost** | 20,765 | 119 | 5.7686 | 16.6011 | -0.15% |

## 4. En Önemli Özellikler (İlk 10 — `hedef_min`)

| Sıra | Random Forest (Göreli Önem %) | XGBoost (Göreli Önem %) | Linear Regression (|Standart Katsayı| %) |
|---:|---|---|---|
| 1 | `tarih_hafta_gunu` (%9.33) | `min_lag_5` (%7.14) | `min_rolling_5_mean` (%14.01) |
| 2 | `min_fark_1` (%7.49) | `min_lag_10` (%6.91) | `min_lag_5` (%7.17) |
| 3 | `seri_egitim_mutlak_fark_min` (%3.72) | `tarih_hafta_gunu` (%5.19) | `min_rolling_10_min` (%5.79) |
| 4 | `max_fark_1` (%3.60) | `min_fark_1` (%4.10) | `min_rolling_20_mean` (%5.64) |
| 5 | `min_lag_5` (%3.03) | `min_rolling_20_min` (%4.00) | `max_rolling_5_max` (%5.43) |
| 6 | `seri_egitim_mutlak_fark_max` (%2.94) | `min_rolling_20_std` (%3.84) | `min_rolling_10_max` (%4.89) |
| 7 | `min_rolling_20_std` (%2.68) | `max_rolling_5_min` (%3.75) | `min_rolling_5_min` (%4.61) |
| 8 | `min_rolling_20_mean` (%2.65) | `min_lag_20` (%3.55) | `min_lag_1` (%4.45) |
| 9 | `min_rolling_5_mean` (%2.63) | `seri_egitim_mutlak_fark_min` (%3.54) | `fiyat_min` (%4.45) |
| 10 | `max_rolling_10_std` (%2.35) | `min_rolling_10_max` (%3.24) | `min_lag_10` (%4.02) |

## 5. Üretilen Görseller (`ciktilar/asama5/grafikler/`)
- `model_mae_karsilastirma.png`
- `model_rmse_karsilastirma.png`
- `gercek_vs_tahmin_ornek_urunler.png`
- `zamana_gore_hata_degisimi.png`
- `urun_bazli_hata_dagilimi.png`
- `ozellik_onem_duzeyleri.png`
