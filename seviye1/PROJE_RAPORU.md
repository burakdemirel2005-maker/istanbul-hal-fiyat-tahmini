# İstanbul Hal Ürünlerinin Makine Öğrenmesi ile Fiyat Tahmini ve Satın Alma Planlamasında Kullanılması — Akademik Proje Raporu

**Üniversite:** Kahramanmaraş Sütçü İmam Üniversitesi  
**Bölüm:** Bilgisayar Mühendisliği Bölümü  
**Ders:** Makine Öğrenmesi  
**Proje Ekibi:**  
- Burak Demirel  
- Ebubekir Samet Gözübenli  
- Resul Kılınç  

**Rapor Tarihi:** 27 Ekim 2026  

---

## 1. Proje Özeti

Bu çalışmada, İstanbul Büyükşehir Belediyesi (İBB) Hal Müdürlüğü tarafından yayımlanan 2016–2026 dönemi sebze, meyve ve ithal ürün fiyat bültenleri kullanılarak bir sonraki hal işlem gününe ait **minimum (`hedef_min`)** ve **maksimum (`hedef_max`)** ürün fiyatlarını tahmin eden, veri sızıntısından (data leakage) arındırılmış, tekrar üretilebilir bir makine öğrenmesi sistemi ve çoklu ürün sepeti destekli bir **Satın Alma Karar Destek Sistemi** geliştirilmiştir.

Ham veri kümesindeki `382.897` kayıt yapısal denetimden geçirilerek `1.761` tam tekrar, `60` sıfır fiyatlı kayıt ve `91` ters fiyat aralıklı (`EnDusukFiyat > EnYuksekFiyat`) kayıt ayrıştırılmış; geriye kalan `380.985` geçerli gözlem üzerinde `252` ürün kimliği (`TabelaGId`) ve `299` homojen ürün-birim-ad dönemi serisi (`SeriId`) tanımlanmıştır. Her seri içinde yalnızca $t$ ve öncesindeki 20 kesintisiz ortak işlem gününden `54` sayısal özellik üretilerek `354.896` etiketli örnek elde edilmiş ve hedef tarihe (`HedefTarih`) göre kronolojik olarak **Eğitim (`303.394` örnek, 2016–2024)**, **Doğrulama (`30.361` örnek, 2024–2025)** ve **Test (`21.141` örnek, 2025–2026)** kümelerine bölünmüştür.

2016–2026 yılları arasındaki yaklaşık 25–30 katlık nominal fiyat artışı nedeniyle modeller doğrudan fiyat seviyesi yerine bir sonraki işlem günü fiyat farkını ($\Delta y(t+1) = y(t+1) - y(t)$) tahmin edecek şekilde formüle edilmiştir. Doğrulama kümesi üzerinde yapılan model seçiminde **Random Forest** ve **XGBoost** modelleri **Naive Baseline** referansını her iki hedefte de RMSE, MSE ve $R^2$ metriklerinde geçmiştir. Ayrıca veri kümesinde **3 Ekim 2025 (`2025-10-03`)** tarihinde gerçekleşen yapısal bülten güncelleme rejimi değişimi ampirik olarak kanıtlanmış ve Türkçe **Streamlit** web uygulaması (`app.py`) ile tüm analiz, tahmin, model karşılaştırma ve satın alma simülasyonu süreçleri son kullanıcıya sunulmuştur.

---

## 2. Problem Tanımı

Tarımsal ürün ve yaş meyve-sebze piyasaları; mevsimsellik, arz dalgalanmaları, ürün çabuk bozulabilirliği (perishability) ve yüksek enflasyon dönemlerinde nominal ölçek kayması nedeniyle yüksek fiyat oynaklığı sergilemektedir. Manavlar, restoranlar, yemekhaneler ve gıda perakendecileri için günlük satın alma miktarının ve bütçesinin planlanması, bir sonraki işlem gününde hal fiyatlarının hangi aralıkta (`EnDusukFiyat` – `EnYuksekFiyat`) oluşacağının öngörülmesini gerektirir.

Ancak kamuya açık hal verileriyle makine öğrenmesi modeli geliştirilirken dört kritik metodolojik problem ortaya çıkmaktadır:
1. **Veri Kalitesi ve Kimlik Karmaşası:** Aynı ürün kimliği (`TabelaGId`) altında farklı zamanlarda farklı ürün adlarının veya farklı ölçü birimi kodlarının (`BirimTurId`) kullanılması.
2. **Kesintili Zaman Serileri:** Mevsimlik ürünlerin yılın belirli aylarında bültenden çıkıp aylar sonra yeniden girmesi; saf satır kaydırma (`shift`) uygulandığında aylar önceki fiyatın dünkü fiyatmış gibi modele sızması.
3. **Nominal Ölçek Kayması (Non-stationarity):** 2016 yılında ortalama `2,67 TL` olan minimum fiyatların 2026 yılında ortalama `71,59 TL` seviyesine çıkması nedeniyle doğrudan fiyat seviyesi üzerinde eğitilen karar ağaçlarının ekstrapolasyon yapamaması.
4. **Fiyat Sürekliliği (Price Persistence) Rejim Değişimi:** Uzun süre bültende sabit kalan fiyatların belirli bir tarihten sonra günlük olarak güncellenmeye başlanması.

---

## 3. Projenin Amacı ve Önemi

Projenin temel amaçları şunlardır:
1. İBB hal fiyat verilerini kaynak dosyaları değiştirmeden, SHA-256 özetleriyle denetlenebilir biçimde temizlemek ve homojen ürün serileri oluşturmak.
2. Gelecekten bilgi sızdırmayan (strictly causal) 20 gözlemlik kesintisiz geçmiş penceresi özellikleriyle bir sonraki hal işlem gününün minimum ve maksimum fiyatlarını tahmin etmek.
3. **Naive Baseline**, **Linear Regression**, **Random Forest Regressor** ve **XGBoost Regressor** modellerini birebir aynı doğrulama ve test örnekleri üzerinde çok düzeyli metriklerle (`MAE`, `MSE`, `RMSE`, `R²`) karşılaştırmak.
4. Kullanıcıların tekil ürün veya **çoklu ürün sepeti** için miktar ve bütçe girerek mevcut ve tahmini satın alma maliyetlerini modelin geçmiş hata belirsizliği (`±MAE`) eşliğinde simüle edebileceği Türkçe bir karar destek uygulaması oluşturmak.

---

## 4. Literatür İncelemesi

Zaman serisi ve tarımsal fiyat tahmini literatüründe makine öğrenmesi yöntemlerinin başarısı, veri bölme protokolüne ve durağanlaştırma (stationarization) stratejisine doğrudan bağlıdır:
- **Zaman Serilerinde Değerlendirme ve Naive Referans (Hyndman & Athanasopoulos, 2021; Tashman, 2000):** *Forecasting: Principles and Practice* çalışmasında vurgulandığı üzere, rastgele yürüyüş (random walk) veya yüksek süreklilik gösteren ekonomik serilerde $\hat{y}_{t+1} = y_t$ (Naive/Persistence) modeli zorlu bir temel referanstır. Tüm karmaşık modellerin bu referansa karşı hem mutlak hata (`MAE`) hem karesel hata (`RMSE`) üzerinden kıyaslanması zorunludur.
- **Kronolojik Bölme ve Veri Sızıntısı Önleme (Bergmeir & Benítez, 2012):** Zaman bağımlı verilerde rastgele çapraz doğrulama (random k-fold) gelecekteki gözlemleri eğitim kümesine sızdırarak yapay başarı üretir. Bu nedenle zaman sınırına dayalı (out-of-time) eğitim, doğrulama ve test ayrımı kullanılmalıdır.
- **Topluluk Ağaç Modelleri (Breiman, 2001; Chen & Guestrin, 2016):** *Random Forest* (bagging tabanlı karar ağacı topluluğu) ve *XGBoost* (düzenlileştirilmiş gradyan artırma) doğrusal olmayan etkileşimleri ve eşik etkilerini yakalamada güçlüdür; ancak ağaç modelleri eğitim kümesindeki hedef değer aralığının dışına doğrusal ekstrapolasyon yapamaz. Bu nedenle enflasyonist serilerde birinci fark ($\Delta y_{t+1} = y_{t+1} - y_t$) hedef formülasyonu standart yaklaşımdır.

---

## 5. Veri Seti Tanıtımı

Projede yalnızca İBB Açık Veri Portalı üzerinden edinilen iki resmi CSV dosyası kullanılmıştır:
1. **Ana Veri Seti (`IBB_Hal_Meyve_Sebze_2016_2026.csv`):** `382.897` satır, 9 sütun (`Tarih`, `TabelaGId`, `UrunAd`, `BirimTurId`, `KategoriId`, `EnDusukFiyat`, `EnYuksekFiyat`, `GuneAit`, `HalTurId`). Kategori kodları `5` (Meyve), `6` (Sebze) ve `7` (İthal Ürünler) değerlerinden oluşmaktadır.
2. **Karşılaştırma Veri Seti (`IBB_Hal_Tum_Urunler_2016_2026.csv`):** `383.829` satır. Yalnızca veri bütünlüğü çapraz kontrolü için kullanılmış; içindeki `382.897` meyve-sebze kaydının ana veri setiyle birebir aynı olduğu doğrulanmıştır.

**Veri Kapsamı Kısıtı:** Proje kuralları gereği döviz kuru, motorin fiyatı, meteoroloji, TÜİK enflasyon endeksi veya HKS işlem hacmi gibi harici veri kaynakları modele dahil edilmemiştir. Ayrıca kaynak CSV'de `BirimTurId` (`1, 2, 3, 7, 11`) için resmi birim adı tablosu bulunmadığından birim adları tahmin edilmemiş, sistem genelinde `"Birim kodu X — adı doğrulanmadı"` olarak raporlanmıştır.

---

## 6. Veri Temizleme ve Ön İşleme (Aşama 1)

`asama1_veri.py` modülü tarafından gerçekleştirilen deterministik temizlik muhasebesi:
- **Ham kayıt sayısı:** `382.897`
- **Çıkarılan fazla tam tekrar sayısı:** `1.761` (`1.751` ham birebir tekrar + `10` boşluk normalizasyonu sonrası ortaya çıkan tekrar)
- **Ayrılan sıfır fiyatlı kayıt sayısı (`EnDusukFiyat <= 0` veya `EnYuksekFiyat <= 0`):** `60`
- **Ayrılan ters fiyat aralıklı kayıt sayısı (`EnDusukFiyat > EnYuksekFiyat`):** `91`
- **Temiz ve yapısal olarak geçerli gözlem sayısı (`temiz_veri.csv`):** `380.985`
  - **İnceleme işaretli temiz gözlem (`IncelemeGerekli == True`):** `356` (gün içi `EnYuksekFiyat / EnDusukFiyat > 10` veya ardışık gözlemde `>3x` / `<1/3x` sıçrama gösteren kayıtlar; silinmemiş, bayraklanmıştır).
  - **İnceleme işaretsiz temiz gözlem:** `380.629`
- **Seri Kimliği (`SeriId`) Oluşturma:** `252` tekil `TabelaGId` içinde ürün adı veya birim kodu zaman içinde değişebildiğinden, `(TabelaGId, BirimTurId, HalTurId, KategoriId, AdDonemi)` birleşimiyle **`299` homojen ürün serisi** tanımlanmıştır.

---

## 7. Özellik Mühendisliği (Aşama 2)

`asama2_ozellikler.py` modülü, `2.734` ortak hal işlem günü takvimini temel alarak her seri içinde kesintisiz gözlem blokları (`BaglamId`) oluşturmuştur. Bir seri ortak bir işlem gününde bültende yer almazsa geçmiş penceresi sıfırlanmaktadır.

En az 20 kesintisiz geçmiş gözlemi bulunan ve bir sonraki ortak işlem gününde hedef kaydı mevcut olan **`354.896` etiketli örnek** için **`54` sayısal özellik** üretilmiştir:
1. **Mevcut Gün Fiyat Özellikleri (5):** `fiyat_min`, `fiyat_max`, `fiyat_orta`, `fiyat_araligi`, `goreli_fiyat_araligi`.
2. **Gecikmeli Fiyatlar — Lag (10):** `min_lag_1, 2, 5, 10, 20` ve `max_lag_1, 2, 5, 10, 20`.
3. **Hareketli Pencere İstatistikleri — Rolling (24):** 5, 10 ve 20 işlem günlük pencerelerde (`t` dahil, `ddof=0`) `mean`, `std`, `min`, `max` değerleri.
4. **Momentum ve Değişim Özellikleri (6):** `min_fark_1`, `min_degisim_orani_1`, `min_degisim_orani_5`, `max_fark_1`, `max_degisim_orani_1`, `max_degisim_orani_5`.
5. **Takvim ve Döngüsel Mevsimsellik Özellikleri (8):** `tarih_yil`, `tarih_ay`, `tarih_hafta_gunu`, `tarih_yil_gunu`, `tarih_ay_sin`, `tarih_ay_cos`, `tarih_yil_sin`, `tarih_yil_cos`.
6. **Gözlem Aralığı (1):** `onceki_gozlemden_gun`.

Son veri günü olan `2026-10-26` tarihinde bültende yer alan `104` seriden 20 kesintisiz işlem günü geçmişini tamamlayan **`91` seri** (`son_gun_girdileri.csv`) ileriye dönük canlı tahmin için hazırdır.

---

## 8. Zaman Serisi Bölme Stratejisi

Veri kümesi, hedef günü (`HedefTarih`) anahtarına göre keskin kronolojik sınırlarla üç bölüme ayrılmıştır:

| Bölüm | Hedef Tarih Sınırları | Örnek Sayısı | Oran (%) | Seri Sayısı | Eğitimde Görülmeyen Seri / Örnek |
|---|---|---:|---:|---:|---:|
| **Eğitim (`egitim`)** | `2016-02-02` – `2024-10-25` (`<= 2024-10-26`) | `303.394` | `%85,49` | `285` | `0 / 0` |
| **Doğrulama (`dogrulama`)** | `2024-10-28` – `2025-10-24` (`2024-10-27`..`2025-10-26`) | `30.361` | `%8,55` | `152` | `2 seri / 260 örnek` |
| **Test (`test`)** | `2025-10-27` – `2026-10-26` (`>= 2025-10-27`) | `21.141` | `%5,96` | `123` | `4 seri / 376 örnek` |
| **Toplam** | `2016-02-02` – `2026-10-26` | **`354.896`** | **`%100,00`** | **`287`** | — |

---

## 9. Veri Sızıntısı Önleme Yöntemleri

Projede veri sızıntısını (data leakage) önlemek için uygulanan 6 katmanlı koruma:
1. **Nedensel Pencere Kuralı:** $t$ günündeki tüm özellikler yalnızca $\le t$ işlem günlerindeki fiyatlardan hesaplanmıştır; $t+1$ hedef fiyatı hiçbir özelliğin hesabına girmez.
2. **Hedef Tarih Bazlı Bölme:** Bölme işlemi gözlem tarihine (`Tarih`) göre değil hedef tarihine (`HedefTarih`) göre yapılmıştır; böylece eğitim kümesinin son günündeki hiçbir hedef doğrulama dönemine taşmaz.
3. **Eğitim Kümesine Özgü Ön İşleme (`StandardScaler`, `OneHotEncoder`, `SeriesHistoryStatsEncoder`):** Tüm dönüştürücüler yalnızca `egitim.csv.gz` üzerinde `fit` edilmiş; doğrulama ve test kümelerine yalnızca `transform` uygulanmıştır.
4. **Eğitimde Görülmeyen Seriler İçin Önsel (Prior) Atama:** Doğrulama (`2` seri) ve test (`4` seri) kümelerinde ilk kez ortaya çıkan serilerde `SeriesHistoryStatsEncoder`, test verisine bakmadan eğitim kümesinin genel ortalamasını atar.
5. **Eğitim Aşırı Sıçrama Filtresinin Yalnızca Eğitimde Uygulanması:** Veri giriş hatası niteliğindeki aşırı sıçrama kayıtları (`3.558` kayıt, ör. 2017 yılında 3,50 TL'den 10.200 TL'ye çıkan tek günlük kayıt) yalnızca eğitim kümesinden çıkarılmış (`299.836` temiz eğitim örneği); doğrulama (`30.361`) ve test (`21.141`) kümelerinin **tek bir satırı bile silinmemiştir**.
6. **Tek Seferlik Test Değerlendirmesi:** Özellik seçimi ve hiperparametre araması yalnızca doğrulama kümesinde yapılmış; modeller kilitlendikten sonra test kümesi üzerinde tek seferlik değerlendirme gerçekleştirilmiştir.

---

## 10. Fiyat Sürekliliği Analizi (3 Ekim 2025 Yapısal Kırılması)

Aşama 3 bulgularında doğrulama döneminde fiyatların bir sonraki işlem gününde aynı kalma oranı **`%94,31`** (`hedef_min`) ve **`%92,94`** (`hedef_max`) iken, test döneminde bu oranların **`%41,66`** (`hedef_min`) ve **`%36,27`** (`hedef_max`) seviyesine düştüğü tespit edilmiştir (`ciktilar/asama4/FIYAT_SUREKLILIGI_RAPORU.md`).

Yapılan kapsamlı inceleme sonucunda bu değişimin nedeni kesin olarak belirlenmiştir:
1. **2 Ekim 2025 – 3 Ekim 2025 Bülten Rejim Kırılması:** `2025-10-02` Perşembe günü bültende **`125`** seri yayımlanırken, `2025-10-03` Cuma günü seri sayısı **`92`**'ye düşmüştür. Bültenden çıkarılan **`35`** seri, kırılma öncesinde günlerin **`%97,34`**'ünde fiyatı hiç güncellenmeden taşınan durağan kalemlerdir.
2. **Aktif Ürünlerde Güncelleme Sıklığının 9 Kat Artması:** Yayında kalmaya devam eden **`92` korunan seride** kırılma öncesinde (`2024-10-27` – `2025-10-02`) fiyatların sabit kalma oranı **`%95,70`** iken, kırılma sonrasındaki son 3 doğrulama haftasında (`2025-10-03` – `2025-10-24`) **`%45,12`**'ye ve test döneminde (**`2025-10-27` – `2026-10-26`**) **`%41,66`** seviyesine inmiştir.
3. **Ortak 106 Seri Kontrolü:** Doğrulama ve test kümelerinin her ikisinde de bulunan **`106` ortak seri** izole edildiğinde dahi sabit kalma oranı doğrulamada **`%93,12`** (`hedef_min`) / **`%91,50`** (`hedef_max`) iken testte **`%42,46`** (`hedef_min`) / **`%37,07`** (`hedef_max`) olmuştur.
4. **Tam Sayı Fiyatlama Geçişi:** 2016 yılında kayıtların **`%49,40`**'ı kuruşlu (kesirli) fiyat içerirken, 2024, 2025 ve 2026 yıllarında kesirli fiyat oranı **`%0,00`**'dır (tüm fiyatlar tam sayı TL olarak yayımlanmaktadır).

---

## 11. Kullanılan Makine Öğrenmesi Modelleri

Tüm makine öğrenmesi modelleri fiyat farkı ($\Delta y(t+1) = y(t+1) - y(t)$) hedefi üzerinde eğitilmiş ve tahminler $\hat{y}(t+1) = \max(0.01, y(t) + \widehat{\Delta y}(t+1))$ denklemiyle geri çatılmıştır. Aşama 2'deki 54 özellikten durağan dışı `tarih_yil` ve paydası küçük fiyatlarda 2913 kata kadar patlama yapan 5 oran özelliği (`min_degisim_orani_1`, `min_degisim_orani_5`, `max_degisim_orani_1`, `max_degisim_orani_5`, `goreli_fiyat_araligi`) doğrulama analizine dayanarak çıkarılmış; kalan **`48` sayısal özellik**, **`2` kategorik özellik (`BirimTurId`, `KategoriId`)** ve **`4` seri eğitim özeti özelliği** kullanılmıştır:

1. **Naive Baseline (`asama3_baseline.py`):** $\hat{y}_{\min}(t+1) = y_{\min}(t)$ ve $\hat{y}_{\max}(t+1) = y_{\max}(t)$.
2. **Linear Regression (`linear_regression`):** Sayısal özellikler için `StandardScaler`, kategorik özellikler için `OneHotEncoder(handle_unknown='ignore')` ve en küçük kareler doğrusal regresyonu (`fit_intercept=True`).
3. **Random Forest Regressor (`random_forest` — Seçilen Ana Model):** `n_estimators=80`, `max_depth=4`, `min_samples_leaf=80`, `max_features=0.4`, `random_state=42`. Sığ ağaç derinliği ve yüksek yaprak örnek sınırı sayesinde yüksek süreklilik rejiminde aşırı uyum (overfitting) yapmadan koşullu ortalamayı öğrenir.
4. **XGBoost Regressor (`xgboost`):** `n_estimators=35`, `max_depth=4`, `learning_rate=0.02`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=50`, `reg_lambda=50.0`, `reg_alpha=1.0`, `base_score=0.0`, `random_state=42`. Güçlü L1/L2 düzenlileştirmesi ve sıfır başlangıç skoru (`base_score=0.0`) sayesinde makine öğrenmesi modelleri arasında en düşük doğrulama MAE değerini üretir.

---

## 12. Değerlendirme Metrikleri

Modeller dört temel metrik ve Naive Baseline'a göre yüzdesel iyileşme üzerinden değerlendirilmiştir:
- **Ortalama Mutlak Hata (MAE):** $\text{MAE} = \frac{1}{N}\sum_{i=1}^N |\hat{y}_i - y_i|$
- **Ortalama Karesel Hata (MSE):** $\text{MSE} = \frac{1}{N}\sum_{i=1}^N (\hat{y}_i - y_i)^2$
- **Kök Ortalama Karesel Hata (RMSE):** $\text{RMSE} = \sqrt{\text{MSE}}$
- **Belirlilik Katsayısı ($R^2$):** $R^2 = 1 - \frac{\sum (\hat{y}_i - y_i)^2}{\sum (y_i - \bar{y})^2}$ (Sabit hedefli alt serilerde tanımsız $R^2$ değerleri sessizce 0 veya 1'e çevrilmemiş, `hedef_sabit` olarak ayrıştırılmıştır).

---

## 13. Deneysel Sonuçlar

Doğrudan fiyat seviyesi ($y(t+1)$) üzerinde 54 ham özellikle yapılan başlangıç deneylerinde Random Forest doğrulama RMSE değeri `4,5468 TL` ve XGBoost doğrulama RMSE değeri `4,4147 TL` olarak ölçülmüş; ağaç modellerinin enflasyonist fiyat bandında Naive Baseline (`3,9134 TL`) gerisinde kaldığı kanıtlanmıştır (`ciktilar/asama4/hiperparametre_arama_sonuclari.csv`).

Fiyat farkı ($\Delta y(t+1)$) formülasyonu ve seçilmiş özellik kümesine geçildiğinde doğrulama ve test kümelerinde elde edilen nihai sonuçlar aşağıdadır:

| Bölüm | Hedef | Model | MAE (TL) | RMSE (TL) | MSE | R² | Baseline MAE İyileşme (%) | Baseline RMSE İyileşme (%) | Seri Ort. MAE | Seri Ort. RMSE | Baseline'dan İyi Seri Sayısı |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Doğrulama** | `hedef_min` | **Naive Baseline** | 0.5839 | 3.9134 | 15.3144 | 0.994031 | +0.00% | +0.00% | 0.5958 | 2.6239 | 0/152 |
| **Doğrulama** | `hedef_min` | **Linear Regression** | 0.7563 | 3.9161 | 15.3355 | 0.994023 | -29.54% | -0.07% | 0.7997 | 2.6536 | 52/152 |
| **Doğrulama** | `hedef_min` | **Random Forest** | 0.6467 | **3.9067** | **15.2627** | **0.994052** | -10.75% | **+0.17%** | 0.6715 | 2.6384 | **84/152** |
| **Doğrulama** | `hedef_min` | **XGBoost** | **0.6098** | **3.9117** | **15.3016** | **0.994036** | -4.43% | **+0.04%** | 0.6252 | 2.6281 | **83/152** |
| **Doğrulama** | `hedef_max` | **Naive Baseline** | 0.7222 | 4.5402 | 20.6137 | 0.995196 | +0.00% | +0.00% | 0.7478 | 3.0725 | 0/152 |
| **Doğrulama** | `hedef_max` | **Linear Regression** | 0.9267 | 4.5524 | 20.7247 | 0.995171 | -28.31% | -0.27% | 0.9940 | 3.1050 | 49/152 |
| **Doğrulama** | `hedef_max` | **Random Forest** | 0.8074 | **4.5305** | **20.5256** | **0.995217** | -11.79% | **+0.21%** | 0.8579 | 3.0965 | **97/152** |
| **Doğrulama** | `hedef_max` | **XGBoost** | **0.7557** | **4.5370** | **20.5848** | **0.995203** | -4.63% | **+0.07%** | 0.7977 | 3.0841 | **107/152** |
| **Test** | `hedef_min` | **Naive Baseline** | **5.0700** | 16.3458 | 267.1857 | 0.974446 | +0.00% | +0.00% | 7.3697 | 10.9339 | 0/123 |
| **Test** | `hedef_min` | **Linear Regression** | 5.0781 | **15.7254** | **247.2881** | **0.976349** | -0.16% | **+3.80%** | **7.2995** | **10.5612** | **99/123** |
| **Test** | `hedef_min` | **Random Forest** | 5.1005 | **16.3276** | **266.5910** | **0.974503** | -0.60% | **+0.11%** | 7.4049 | 10.9073 | **93/123** |
| **Test** | `hedef_min` | **XGBoost** | 5.0909 | 16.3535 | 267.4360 | 0.974422 | -0.41% | -0.05% | 7.3960 | 10.9303 | 80/123 |
| **Test** | `hedef_max` | **Naive Baseline** | 6.0526 | 18.4987 | 342.2001 | 0.980335 | +0.00% | +0.00% | 8.8331 | 12.6695 | 0/123 |
| **Test** | `hedef_max` | **Linear Regression** | **5.9220** | **17.7083** | **313.5851** | **0.981979** | **+2.16%** | **+4.27%** | **8.5917** | **12.0906** | **118/123** |
| **Test** | `hedef_max` | **Random Forest** | 6.0666 | **18.4709** | **341.1751** | **0.980393** | -0.23% | **+0.15%** | 8.8468 | 12.6115 | **109/123** |
| **Test** | `hedef_max` | **XGBoost** | 6.0749 | 18.5216 | 343.0507 | 0.980286 | -0.37% | -0.12% | 8.8610 | 12.6672 | 84/123 |

---

## 14. Model Karşılaştırması ve Yorumu

1. **Doğrulama Kümesinde Model Seçimi:** Doğrulama kümesinde hem `hedef_min` (`RMSE 3,9067 TL < 3,9134 TL`) hem `hedef_max` (`RMSE 4,5305 TL < 4,5402 TL`) için Naive Baseline'ı geçen ve en düşük ortalama RMSE'yi üreten **Random Forest** ana üretim modeli olarak kilitlenmiştir. Ayrıca Random Forest doğrulama kümesindeki 152 serinin **84'ünde (`hedef_min`)** ve **97'sinde (`hedef_max`)** Naive Baseline'dan daha düşük RMSE elde etmiştir.
2. **XGBoost'un L1/L2 Dengesi:** XGBoost doğrulama kümesinde makine öğrenmesi modelleri arasında en düşük MAE (`0,6098 TL` ve `0,7557 TL`) değerini üretirken aynı zamanda Naive Baseline RMSE değerini her iki hedefte geçmiştir (`107/152` seride `hedef_max` RMSE üstünlüğü).
3. **Test Dönemi Yüksek Oynaklık Rejiminde Başarı:** Doğrulamada kilitlenen **Random Forest**, test kümesinde de Naive Baseline RMSE değerini her iki hedefte (`16,3276 < 16,3458` ve `18,4709 < 18,4987`) ve serilerin **`93/123`'ünde (`hedef_min`)** / **`109/123`'ünde (`hedef_max`)** geçmiştir. Öte yandan, 3 Ekim 2025 sonrasında fiyatların her gün güncellendiği yeni rejimde **Linear Regression** modelinin ortalamaya dönüş (mean-reversion: `min_rolling_5_mean`, `min_lag_5`) katsayıları çok güçlü çalışmış ve test RMSE değerini `hedef_min`'de **`%3,80`**, `hedef_max`'ta **`%4,27`** (serilerin **`118/123`'ünde**) iyileştirmiştir. Metodolojik dürüstlük gereği doğrulamada seçilen Random Forest ana model olarak korunmuş, Linear Regression'ın yeni rejimdeki üstünlüğü açıkça raporlanmıştır.

---

## 15. Satın Alma Karar Destek Sistemi (Aşama 6)

`asama6_satin_alma.py` modülü, tahminleri manav ve gıda işletmeleri için operasyonel maliyet simülasyonuna dönüştürür:
- **Desteklenen Fiyat Türleri:** `Minimum Fiyat`, `Maksimum Fiyat` ve `Aritmetik Ortalama Fiyat`.
- **Belirsizlik Duyarlı Yön Beklentisi:** Tahmini değişim yüzdesi `%0,75`'in altında veya ürünün geçmiş Test MAE değerinin `%15`'inden küçük olduğunda sistem yapay fiyat artışı/düşüşü alarmı üretmez, **`Düşük Değişim Beklentisi`** sınıfını atar.
- **Çoklu Ürün Sepeti Simülasyonu:** Örneğin `Domates (100 birim)`, `Salatalık I. (80 birim)`, `Patates (150 birim)` ve `Soğan (Taze) (120 birim)` içeren 4 kalemlik örnek sepette mevcut toplam maliyet **`18.045,00 TL`**, tahmini toplam maliyet **`18.054,02 TL`** (`+9,02 TL`, `%+0,05`), toplam MAE belirsizlik bandı ise **`±1.393,31 TL`** olarak hesaplanmaktadır.

---

## 16. Web Uygulaması Tasarımı (Aşama 7)

`app.py` dosyasında geliştirilen **Streamlit** uygulaması tamamen Türkçe arayüze, özel CSS kart tasarımına, `st.cache_data` / `st.cache_resource` önbelleklemesine ve etkileşimli **Plotly** grafiklerine sahiptir. Uygulama 6 ana sayfadan oluşur:
1. **`1. Ana Panel`:** Veri kümesi KPI kartları, kategori dağılımı, seçilen birim koduna göre en yüksek/en düşük fiyatlı ürünler ve son işlem gününde beklenen değişimi en belirgin ürünler.
2. **`2. Ürün Analizi`:** Kategori ve metin arama filtreli ürün seçimi, tarih aralığı seçimi, 7 ve 30 gözlemlik hareketli ortalamalar, gün içi fiyat aralığı (spread) ve aylık mevsimsellik grafikleri.
3. **`3. Fiyat Tahmini`:** Seçilen ürün ve model için mevcut/tahmini min-max fiyatlar, geçmiş Test MAE/RMSE belirsizlik kartı, son 60 gözlem + gelecek işlem günü tahmin grafiği, kesintili serilerde dürüst geri dönüş bildirimi ve 4 modelin yan yana tahmin tablosu.
4. **`4. Satın Alma Planlama`:** Tekil ürün maliyet simülasyonu (±MAE hata çubuklu grafik) ve çoklu ürün sepeti simülasyonu.
5. **`5. Model Karşılaştırma`:** Doğrulama ve test dönemi sekmeleri, MAE/RMSE karşılaştırma grafikleri, 2024–2026 aylık hata evrimi (`2025-10-03` kırılması işaretli), özellik önem düzeyleri ve alt grup performans tabloları.
6. **`6. Proje Hakkında`:** Proje amacı, metodoloji, 3 Ekim 2025 fiyat sürekliliği bulgusu, kısıtlar ve ekip/üniversite künyesi.

---

## 17. Test ve Doğrulama Çalışmaları (Aşama 8)

Proje genelinde toplam **47 otomatik birim ve entegrasyon testi** (`test_asama1.py` [12], `test_asama2.py` [13], `test_asama3.py` [10], `test_asama4_8.py` [12]) ve **4 bağımsız doğrulama betiği** (`dogrula_asama1.py`, `dogrula_asama2.py`, `dogrula_asama3.py`, `dogrula_asama4_6.py`) geliştirilmiştir:
- Tüm 47 birim testi (`python -m unittest discover -p "test_*.py" -v`) **%100 başarıyla (`OK`)** geçmektedir.
- Streamlit `AppTest` otomasyonu ile web uygulamasındaki 6 sayfanın tamamı başsız (headless) modda çalıştırılmış ve sıfır çalışma zamanı hatası doğrulanmıştır.
- Orijinal CSV dosyalarının ve tüm aşama çıktılarının SHA-256 özetleri manifesto dosyalarıyla doğrulanmıştır.

---

## 18. Projenin Kısıtları

1. **Harici Değişkenlerin Bulunmaması:** Yalnızca İBB hal fiyat bülteni kullanıldığından döviz kuru, akaryakıt/nakliye maliyetleri, meteorolojik don/yağış olayları ve HKS işlem hacmi şokları modelin bilgi kümesinde yer almamaktadır.
2. **Ölçü Birimi Sözlüğünün Kaynakta Yer Almaması:** Kaynak CSV'de `BirimTurId` (`1, 2, 3, 7, 11`) değerlerinin resmi metin karşılıkları bulunmadığından birimler yalnızca kodlarıyla raporlanmaktadır.
3. **3 Ekim 2025 Yapısal Rejim Değişimi:** Eğitim (2016–2024) ve doğrulama döneminin büyük kısmında (%94+) fiyatların sabit kalması, doğrulama kümesinde seçilen ağaç modellerinin muhafazakâr (küçük $\Delta y$) tahminler üretmesine yol açmaktadır.

---

## 19. Gelecekteki Geliştirmeler

1. **Çevrimiçi / Kayan Pencere (Rolling Window) Yeniden Eğitim:** 3 Ekim 2025 sonrası günlük fiyat güncelleme rejimine uyum sağlamak için modellerin son 6–12 aylık kayan pencerede ağırlıklı olarak güncellenmesi.
2. **Çoklu Kaynak Entegrasyonu (Seviye 2):** MGM meteoroloji verileri (üretim bölgelerindeki sıcaklık/yağış gecikmeleri), TCMB döviz kurları, motorin fiyatları ve HKS işlem hacimlerinin nedensel gecikmelerle (`lag >= 1`) modele eklenmesi.
3. **Kantillere Dayalı Olasılıksal Tahmin (Quantile Regression):** Satın alma karar destek sisteminde yalnızca nokta tahmini ve geçmiş MAE bandı yerine %10–%90 olasılıksal tahmin aralıklarının doğrudan modellenmesi.

---

## 20. Sonuç

Bu projede, İstanbul Hal sebze ve meyve fiyatları için veri temizliğinden özellik mühendisliğine, 3 Ekim 2025 yapısal fiyat sürekliliği kırılmasının keşfinden fiyat farkı ($\Delta y$) tabanlı makine öğrenmesi modellerine ve çoklu ürün sepeti destekli Türkçe Streamlit web uygulamasına kadar uçtan uca, veri sızıntısız ve akademik olarak savunulabilir bir sistem başarıyla tamamlanmıştır.

---

### Kaynakça
1. Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32.
2. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 785–794).
3. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.
4. Bergmeir, C., & Benítez, J. M. (2012). *On the use of cross-validation for time series predictor evaluation*. Information Sciences, 191, 192–213.
5. Tashman, L. J. (2000). *Out-of-sample tests of forecasting accuracy: an analysis and review*. International Journal of Forecasting, 16(4), 437–450.
6. İstanbul Büyükşehir Belediyesi (İBB) Açık Veri Portalı — Hal Müdürlüğü Günlük Meyve ve Sebze Fiyat Bültenleri (2016–2026).
