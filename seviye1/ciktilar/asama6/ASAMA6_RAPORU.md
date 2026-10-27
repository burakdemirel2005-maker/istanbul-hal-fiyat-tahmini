# Aşama 6 — Satın Alma Karar Destek Sistemi Raporu

## 1. Karar Destek Mimarisi ve Belirsizlik İlkeleri
- Sistem; ürün serisi (`SeriId`), satın alma miktarı (`miktar > 0`), fiyat türü (`minimum`, `maksimum`, `aritmetik_orta`) ve isteğe bağlı bütçe (`butce`) girdilerini doğrular.
- Model tahminleri kesin fiyat veya tasarruf garantisi olarak sunulmaz; her tahminin yanında ilgili ürün serisinin Test Dönemi (`2025-10-27` – `2026-10-26`) gerçek **MAE** ve **RMSE** değerleri ve miktar bazlı belirsizlik bandı (`± MAE × miktar`) gösterilir.
- Tahmini değişim yüzdesi `%0,75`'in altında veya ürünün geçmiş MAE değerinin `%15`'inden küçük olduğunda sistem yapay alarm üretmez ve **Düşük Değişim Beklentisi** sınıfını atar.

## 2. Örnek Çoklu Ürün Sepeti Simülasyonu (`Domates`, `Salatalık`, `Patates`, `Soğan (Kuru)`)
- **Kullanılan Model:** `Random Forest` (`Aritmetik Ortalama Fiyat`)
- **Toplam Mevcut Maliyet:** `18,045.00 TL`
- **Toplam Tahmini Maliyet:** `18,054.02 TL`
- **Beklenen Toplam Fark:** `+9.02 TL` (`%+0.05`)
- **Toplam MAE Belirsizlik Bandı:** `±1,393.31 TL`
- **Bütçe Durumu:** Toplam tahmini sepet maliyeti (18,054.02 TL) bütçe (25,000.00 TL) içindedir (kalan: 6,945.98 TL).

| Ürün | Birim Açıklaması | Miktar | Mevcut Birim (TL) | Tahmini Birim (TL) | Mevcut Maliyet (TL) | Tahmini Maliyet (TL) | Fark (TL) | Değişim (%) | Geçmiş MAE (TL) | Beklenti Sınıfı |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **Domates** | Birim kodu 1 — adı doğrulanmadı | 100 | 61.00 | 61.08 | 6,100.00 | 6,108.42 | +8.42 | %+0.14 | 3.81 | Düşük Değişim Beklentisi |
| **Salatalık I.** | Birim kodu 1 — adı doğrulanmadı | 80 | 40.00 | 40.05 | 3,200.00 | 3,204.12 | +4.12 | %+0.13 | 3.04 | Düşük Değişim Beklentisi |
| **Patates** | Birim kodu 1 — adı doğrulanmadı | 150 | 29.50 | 29.53 | 4,425.00 | 4,430.15 | +5.15 | %+0.12 | 1.47 | Düşük Değişim Beklentisi |
| **Soğan (Taze)** | Birim kodu 1 — adı doğrulanmadı | 120 | 36.00 | 35.93 | 4,320.00 | 4,311.33 | -8.67 | %-0.20 | 4.57 | Düşük Değişim Beklentisi |
