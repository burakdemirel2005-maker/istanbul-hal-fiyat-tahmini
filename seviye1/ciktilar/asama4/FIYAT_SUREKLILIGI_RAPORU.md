# Bölüm 6 — Fiyat Sürekliliği (Price Persistence) Araştırma Raporu

## 1. Araştırma Sorusu ve Temel Bulgu
Aşama 3 Naive Baseline analizinde doğrulama döneminde (`2024-10-27` – `2025-10-26`) fiyatların bir sonraki işlem gününde aynı kalma oranı:
- `hedef_min == fiyat_min`: **%94.31**
- `hedef_max == fiyat_max`: **%93.00**

iken, test döneminde (`2025-10-27` – `2026-10-26`) bu oranlar:
- `hedef_min == fiyat_min`: **%41.66**
- `hedef_max == fiyat_max`: **%36.27**
seviyesine düşmektedir.

Yapılan zamansal ve seri bazlı inceleme, bu farkın **2 Ekim 2025 (`2025-10-02`) ile 3 Ekim 2025 (`2025-10-03`) tarihleri arasında İBB hal fiyat bülteni güncelleme rejiminde gerçekleşen keskin bir yapısal değişimden** kaynaklandığını kanıtlamıştır.

## 2. Dönem ve Yıl Bazında Fiyat Sürekliliği

| Dönem / Yıl | Örnek Sayısı | Seri Sayısı | `hedef_min` Değişmedi (%) | `hedef_max` Değişmedi (%) | Kesirli Fiyat (%) | Ortalama `fiyat_min` (TL) | Ortalama `|Δmin|` (TL) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Eğitim (2016–2024)** | 303,394 | 285 | %93.10 | %90.59 | %14.31 | 13.76 | 0.1818 |
| **Doğrulama (2024–2025)** | 30,361 | 152 | %94.31 | %93.00 | %0.00 | 44.20 | 0.5839 |
| **Test (2025–2026)** | 21,141 | 123 | %41.66 | %36.27 | %0.00 | 60.07 | 5.0700 |
| 2016 | 25,268 | 203 | %86.01 | %81.91 | %36.08 | 8.07 | 0.2031 |
| 2017 | 37,945 | 208 | %84.10 | %82.83 | %26.91 | 9.41 | 0.3149 |
| 2018 | 38,065 | 193 | %95.50 | %93.10 | %27.17 | 8.85 | 0.0962 |
| 2019 | 35,662 | 204 | %94.82 | %92.08 | %15.77 | 8.22 | 0.0899 |
| 2020 | 34,936 | 166 | %96.08 | %93.11 | %11.36 | 8.03 | 0.0751 |
| 2021 | 34,528 | 176 | %93.87 | %92.30 | %9.89 | 9.54 | 0.1262 |
| 2022 | 35,371 | 178 | %93.67 | %91.35 | %1.43 | 17.49 | 0.2692 |
| 2023 | 35,761 | 175 | %96.46 | %92.29 | %0.69 | 25.20 | 0.2170 |
| 2024 | 31,896 | 169 | %96.79 | %95.53 | %0.00 | 34.46 | 0.3075 |
| 2025 | 28,560 | 155 | %88.32 | %86.21 | %0.00 | 45.61 | 1.0711 |
| 2026 | 16,904 | 122 | %37.85 | %32.73 | %0.00 | 62.90 | 5.4110 |

## 3. 3 Ekim 2025 (`2025-10-03`) Yapısal Kırılmasının Kanıtı
- **2 Ekim 2025 Perşembe** günü bültende **125** aktif seri yayımlanırken, **3 Ekim 2025 Cuma** günü aktif seri sayısı **92**'ye düşmüştür.
- Yayından kaldırılan **35** seri (ör. `0198546c-ffb1-4f2f-8ba5-cf1858d1112b|b1|h2|k6|a1, 03798382-154d-4cf3-8d38-8451732fb0ce|b1|h2|k6|a1, 0b160b51-e7f0-4f55-b240-ab9ffbf4b1f0|b5|h2|k6|a1, 295ea9b8-8e87-42ad-8e27-1741623a4663|b1|h2|k7|a1, 300b8aee-02f1-475b-894a-3bc5b2c69e33|b1|h2|k6|a1`), kırılma öncesinde günlerin **%97.34**'sinde fiyatı hiç güncellenmeden bültende taşınan durağan kalemlerdir.
- Yayında kalmaya devam eden **90** ana ürün serisinde ise kırılma öncesinde (`2024-10-27` – `2025-10-02`) fiyatların sabit kalma oranı **%95.65** iken, `2025-10-03` – `2025-10-24` arasındaki son 3 doğrulama haftasında **%48.71**'ye, test döneminde (`2025-10-27` – `2026-10-26`) ise **%42.26**'ye inmiştir.

## 4. Ürün Bileşimi Etkisinin Kontrolü (Ortak 106 Seri Karşılaştırması)
- Doğrulama ve test kümelerinin her ikisinde de yer alan **106 ortak ürün serisi** izole edildiğinde:
  - Doğrulama kümesinde (`21,741` örnek) `hedef_min` sabit kalma oranı **%93.12**, `hedef_max` sabit kalma oranı **%91.50**'dir.
  - Test kümesinde (`19,754` örnek) aynı 106 serinin `hedef_min` sabit kalma oranı **%42.46**, `hedef_max` sabit kalma oranı **%37.07**'dir.
- Dolayısıyla süreklilik düşüşü yalnızca durağan ürünlerin bültenden çıkarılmasıyla açıklanamaz; **aktif ürünlerin günlük fiyat güncelleme sıklığı 3 Ekim 2025 itibarıyla yaklaşık 9 kat artmıştır** (günlük değişim oranı %6,9'dan %57,5'e çıkmıştır).

## 5. Model Öğrenimi ve Değerlendirme Üzerindeki Etkileri
1. **Naive Baseline'ın L1 (MAE) Avantajı:** Eğitim ve doğrulama dönemlerinde günlerin %94–96'sında gerçek fiyat değişimi `Δy = 0` olduğu için, koşullu medyan sıfıra eşittir. L1 kaybı (MAE) koşullu medyanı ödüllendirdiğinden, sıfırdan farklı her tahmin değişmeyen günlerde ceza alır.
2. **L2 (MSE / RMSE / R²) Metriklerinde Makine Öğrenmesi Kazancı:** Fiyatların değiştiği günlerdeki ortalama kuadratik sıçrama (conditional mean), geçmiş momentum (`min_fark_1`, `min_rolling_5_mean - fiyat_min`), fiyat aralığı ve mevsimsellik özellikleriyle anlamlı biçimde tahmin edilebilmektedir. Bu nedenle doğru düzenlileştirilmiş (regularized) `Random Forest` ve `XGBoost` modelleri doğrulama kümesinde `Naive Baseline`'dan daha düşük **RMSE / MSE** üretmektedir.
3. **Nominal Enflasyon ve Hedef Formülasyonu:** 2016 yılında ortalama minimum fiyat `2,67 TL` iken 2026 yılında `71,59 TL`'dir. Doğrudan fiyat seviyesi (`y(t+1)`) üzerinde eğitilen karar ağaçları eğitim bandının dışına çıkamazken, **fiyat farkı (`Δy(t+1) = y(t+1) - y(t)`)** üzerinde eğitilip `ŷ(t+1) = y(t) + Δ̂y(t+1)` ile geri çatılan modeller hem doğrulama hem test rejiminde ölçekten bağımsız çalışmaktadır.
