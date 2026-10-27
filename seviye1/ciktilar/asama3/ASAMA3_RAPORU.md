# Aşama 3 — Naive Baseline Değerlendirme Raporu

Çalıştırma zamanı: 2026-10-27T19:18:17.430800+03:00

Bu aşamada yalnızca **Naive Baseline (son gözlenen fiyatı taşıma / persistence)** yöntemi uygulanmıştır. Hiçbir makine öğrenmesi modeli (Linear Regression, Random Forest, XGBoost vb.) eğitilmemiş, parametre veya hiperparametre öğrenilmemiş, Aşama 2 özellikleri veya kronolojik bölüm sınırları değiştirilmemiştir.

## 1. Yöntem Tanımı ve Tahmin Zamanı

Uygulanan deterministik kural:

- $\hat{y}_{\min}(t+1) = \text{fiyat\_min}(t)$
- $\hat{y}_{\max}(t+1) = \text{fiyat\_max}(t)$
- Hata işareti tanımı: $\text{hata} = \text{tahmin} - \text{gerçek}$ (pozitif hata: tahmin gerçek hedefin üzerinde / gerçek fiyat düştü; negatif hata: tahmin gerçek hedefin altında / gerçek fiyat yükseldi; sıfır hata: fiyat değişmedi).

**Tahmin anı ve hedef anlamı:** Mevcut deney, $t$ gününün İBB hal fiyatları yayımlandıktan sonra bir sonraki ortak kayıtlı meyve-sebze veri gününün ($t+1$) minimum ve maksimum fiyatını tahmin eder. Bu nedenle $t$ gününün minimum (`fiyat_min`) ve maksimum (`fiyat_max`) fiyatları tahmin anında biliniyor kabul edilir. Bu kurgu, $t$ sabahı henüz $t$ fiyatları ilan edilmeden yapılan tahminden farklıdır. Kaynak veride resmî hal işlem takvimi bulunmadığı için "bir sonraki hal işlem günü", ham meyve-sebze dosyasında kayıt bulunan bir sonraki ortak veri günüyle temsil edilmektedir. Bir serinin tam sonraki ortak veri gününde kaydı yoksa uzaktaki sonraki kayda atlanmaz.

## 2. Akademik Yöntem Gerekçesi ve Kaynak Notları

1. **Hyndman, R. J. ve Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.), OTexts.**
   - **Bağlantı:** `https://otexts.com/fpp3/` (`simple-methods.html`, `accuracy.html`, `prediction-intervals.html`)
   - **İncelenen bölüm:** Bölüm 5 (The forecaster's toolbox: Simple forecasting methods, Evaluating point forecast accuracy, Prediction intervals).
   - **Projeye uygulanabilir ders:** Naive yöntem ($\hat{y}_{T+h|T} = y_T$), ekonomik ve fiyat zaman serilerinde zorunlu temel karşılaştırma referansıdır (benchmark). Ölçeğe bağlı hatalar (MAE, RMSE) hedefle aynı birimde (veya MSE için birimin karesinde) olduğundan farklı birimli ya da çok farklı fiyat seviyeli seriler arasında doğrudan tek bir evrensel ortalama olarak yorumlanmamalıdır.
   - **Doğrudan aktarılamayacak yönler:** Kitaptaki düzenli zaman aralıklı `tsibble` örnekleri, düzensiz kayıt boşlukları ve ad/birim dönemleri içeren İBB paneline doğrudan uygulanamaz; kesinti sonrası pencere sıfırlama kuralımız korunmuştur.
   - **Erişim sınırlılığı:** Açık erişimli ders kitabı; erişim kısıtı yok.

2. **Resmî `scikit-learn` Belgeleri (Sürüm 1.9.1).**
   - **Bağlantı:** `https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics`, `https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html`, `https://scikit-learn.org/stable/common_pitfalls.html#data-leakage`
   - **İncelenen bölüm:** Regresyon metrikleri (`r2_score`, `mean_absolute_error`, `mean_squared_error`), gecikmeli zaman serisi özellikleri ve veri sızıntısı önleme ilkeleri.
   - **Projeye uygulanabilir ders:** `sklearn.metrics.r2_score` fonksiyonu varsayılan `force_finite=True` ayarıyla sabit hedefli (`SS_tot == 0`) serilerde tanımsız olan $R^2$ değerini tam isabette `1.0`, hatada `0.0` değerine dönüştürür; $n < 2$ durumunda ise `NaN` döndürür. Bu projede varsayılan sessiz dönüşüm kabul edilmemiş; `gozlem_yetersiz` ($n < 2$), `hedef_sabit` ($\text{SS}_{\text{tot}} = 0$) ve `hesaplanabilir` ($n \ge 2, \text{SS}_{\text{tot}} > 0$) durumları açıkça ayrılmıştır.
   - **Doğrudan aktarılamayacak yönler:** Standart `TimeSeriesSplit` tekil düzenli seriler içindir; çok serili panelimizde ayrım satır sırasına göre değil ortak `HedefTarih` sınırlarına göre yapılmıştır.
   - **Erişim sınırlılığı:** Yerel kurulu paket ve resmî dokümantasyon doğrulandı; erişim kısıtı yok.

3. **Bulut, M. (2025). Forecasting prices of salad vegetables using statistical and artificial intelligence techniques. *Ciência Rural*, 55(11), e20240444.**
   - **Bağlantı:** `https://www.scielo.br/j/cr/a/WZs4CVKzPLzSkwbHDKsj5zd/?lang=en` (DOI: `10.1590/0103-8478cr20240444`)
   - **İncelenen bölüm:** Veri kapsamı, yöntem (ARIMA, CNN, LSTM, RF) ve hata metriklerinin sunumu.
   - **Projeye uygulanabilir ders:** Türkiye hal/sebze fiyatlarında çoklu model karşılaştırmasının ürün bazında raporlanması gerektiğini gösterir.
   - **Doğrudan aktarılamayacak yönler:** Makalede kullanılan HKS işlem hacmi bu projenin kesin veri kapsamı dışındadır; ayrıca makaledeki özet/metin tarih farklılıkları ve ölçekleme belirsizlikleri nedeniyle bildirilen doğruluk değerleri veri setimize referans alınamaz.
   - **Erişim sınırlılığı:** Açık erişimli yayıncı tam metni incelendi; ham veri eki yayımlanmamış.

4. **Adanacıoğlu, H. ve Yercan, M. (2012). An analysis of tomato prices at wholesale level in Turkey: an application of SARIMA model. *Custos e @gronegócio on line*, 8(4), 52–75.**
   - **Bağlantı:** `https://www.custoseagronegocioonline.com.br/numero4v8/Tomato%20english.pdf`
   - **İncelenen bölüm:** Türkiye toptan domates fiyatlarında mevsimsel dalgalanma ve istatistiksel zaman serisi analizi.
   - **Projeye uygulanabilir ders:** Toptan sebze fiyatlarında güçlü mevsimsel bileşenlerin varlığını doğrulayarak ürün bazlı zaman serisi izlemesini destekler.
   - **Doğrudan aktarılamayacak yönler:** Çalışma aylık veriler üzerinedir; aylık SARIMA bulguları günlük minimum/maksimum tahmin görevine doğrudan taşınamaz.
   - **Erişim sınırlılığı:** Açık erişimli PDF tam metni incelendi.

5. **Sari, M., Duran, S., Kutlu, H., Guloglu, B. ve Atik, Z. (2024). Various optimized machine learning techniques to predict agricultural commodity prices. *Neural Computing and Applications*, 36, 11439–11459.**
   - **Bağlantı:** `https://link.springer.com/article/10.1007/s00521-024-09679-x` (DOI: `10.1007/s00521-024-09679-x`)
   - **İncelenen bölüm:** 11 tarımsal üründe istatistiksel ve makine öğrenmesi modellerinin karşılaştırılması.
   - **Projeye uygulanabilir ders:** Çok sayıda üründe standart hata ölçütleriyle (RMSE, MAE, $R^2$) karşılaştırma yapılması; tek bir karmaşık modelin her üründe üstün gelemeyeceğinin görülmesi.
   - **Doğrudan aktarılamayacak yönler:** Genetik algoritma/ELM optimizasyonu kapsamımız dışındadır; her karmaşık yöntemin Naive referansı geçeceği varsayılamaz.
   - **Erişim sınırlılığı:** Açık erişim yayıncı sayfası incelendi; veri seti yalnızca yazardan talep üzerine erişilebilir durumda.

6. **Grinsztajn, L., Oyallon, E. ve Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? *NeurIPS Datasets and Benchmarks*.**
   - **Bağlantı:** `https://proceedings.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html`
   - **İncelenen bölüm:** Tablosal veri setlerinde ağaç tabanlı yöntemler ile derin öğrenme karşılaştırması.
   - **Projeye uygulanabilir ders:** Sonraki aşamada (Aşama 4) gecikmeli tablosal özellikler üzerinde Random Forest ve XGBoost ile başlamanın metodolojik gerekçesini oluşturur.
   - **Doğrudan aktarılamayacak yönler:** Genel tablosal benchmark bulgusudur; zaman serilerinde ağaç modellerinin Naive Baseline'ı otomatik olarak geçeceğini garanti etmez.
   - **Erişim sınırlılığı:** Resmî konferans bildirisi açık erişimli.

7. **Montero-Manso, P. ve Hyndman, R. J. (2021). Principles and Algorithms for Forecasting Groups of Time Series: Locality and Globality. *arXiv:2008.00444*.**
   - **Bağlantı:** `https://arxiv.org/abs/2008.00444`
   - **İncelenen bölüm:** Çoklu zaman serilerinde yerel (her seriye ayrı) ve global (ortak) modelleme ilkeleri.
   - **Projeye uygulanabilir ders:** Seri bazında (`SeriId`) yerel değerlendirme ile ortak kapsam arasındaki farkın korunmasını ve eğitimde görülmeyen serilerin açıkça izlenmesini gerekçelendirir.
   - **Doğrudan aktarılamayacak yönler:** Bu aşamada herhangi bir yerel veya global model eğitilmemiştir.
   - **Erişim sınırlılığı:** Yazarların arXiv tam metni açık erişimli.

## 3. Ana Değerlendirme Sonuçları (Doğrulama ve Test)

Aşağıdaki özet tabloda hem **satır ağırlıklı (örnek ağırlıklı)** hem de **seri eşit ağırlıklı (her seriye eşit ağırlık veren)** metrikler birlikte sunulmuştur. Ürünlerin ölçü birimleri (`BirimTurId`) ve fiyat seviyeleri farklı olduğundan, satır ağırlıklı MAE/RMSE değerleri evrensel bir "TL/kg ortalaması" olarak yorumlanmamalıdır. Aynı şekilde **Havuzlanmış $R^2$**, ürünler arası fiyat seviyesi farklarını (ör. düşük fiyatlı yeşillikler ile yüksek fiyatlı ithal ürünler arasındaki varyansı) içerdiği için bireysel ürünlerdeki tahmin başarısını temsil etmez; asıl referans seri bazlı (`seri_metrikleri.csv`) sonuçlardır.

| Bölüm | Hedef | Örnek | Seri | Satır MAE | Satır MSE | Satır RMSE | Havuz R² | Seri Ort. MAE | Seri Medyan MAE | Seri Medyan R² | Hesaplanabilir R² Seri | Sabit Hedef Seri | Negatif R² Seri | Değişmedi Oranı |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dogrulama | hedef_min | 30.361 | 152 | 0.5839 | 15.3144 | 3.9134 | 0.9940 | 0.5958 | 0.3689 | 0.9359 | 128/152 | 24 | 3 | %94.31 |
| dogrulama | hedef_max | 30.361 | 152 | 0.7222 | 20.6137 | 4.5402 | 0.9952 | 0.7478 | 0.4640 | 0.9439 | 129/152 | 23 | 0 | %93.00 |
| test | hedef_min | 21.141 | 123 | 5.0700 | 267.1857 | 16.3458 | 0.9744 | 7.3697 | 3.7380 | 0.8078 | 123/123 | 0 | 14 | %41.66 |
| test | hedef_max | 21.141 | 123 | 6.0526 | 342.2001 | 18.4987 | 0.9803 | 8.8331 | 4.2541 | 0.8021 | 123/123 | 0 | 15 | %36.27 |

### Dönemler Arası Fiyat Sürekliliği Bulgusu

- **Doğrulama döneminde (27.10.2024–26.10.2025; 30.361 örnek, 152 seri):** Ardışık ortak veri günlerinde minimum fiyatların **%94.31**'si, maksimum fiyatların **%93.00**'si bir önceki veri günüyle aynı kalmıştır. Ayrıca doğrulama döneminde `hedef_min` için **24 seride**, `hedef_max` için **23 seride** hedef fiyat dönem boyunca hiç değişmemiştir (`R2Durumu = hedef_sabit`, $\text{SS}_{\text{tot}} = 0$, $\text{SS}_{\text{res}} = 0$). `scikit-learn` varsayılanı bu tanımsız durumlara `1.0` atarken, değerlendirmemizde bu seriler `hedef_sabit` olarak ayrılmış ve sayısal $R^2$ ortalamalarına katılmamıştır.
- **Test döneminde (27.10.2025–26.10.2026; 21.141 örnek, 123 seri):** Fiyatların bir sonraki ortak veri gününde aynı kalma oranı minimum fiyatlarda **%41.66**'ye, maksimum fiyatlarda **%36.27**'ye düşmüştür. Test döneminde sabit hedefli seri yoktur (`hedef_sabit = 0`); 123 serinin tamamında $R^2$ hesaplanabilir durumdadır. Günlük fiyat değişim sıklığının ve fiyat seviyelerinin son 12 ayda belirgin biçimde artması nedeniyle, test dönemi satır ağırlıklı MAE değeri doğrulama dönemine kıyasla daha yüksektir (`hedef_min` için 0.5839'den 5.0700'ye; `hedef_max` için 0.7222'den 6.0526'ye).
- **Negatif $R^2$ görülen seriler:** Naive Baseline her zaman pozitif $R^2$ üretmez; hedef varyansının düşük olduğu veya ani sıçramaların/geri dönüşlerin yaşandığı serilerde $\text{SS}_{\text{res}} > \text{SS}_{\text{tot}}$ olduğunda $R^2 < 0$ çıkar. Doğrulamada `hedef_min` için 3 seride (maksimum hedefte 0 seride), testte ise `hedef_min` için 14 seride ve `hedef_max` için 15 seride negatif $R^2$ gözlenmiştir.

## 4. Eğitimde Görülme Durumu ve Kalite İşaretlerine Göre Kapsam Özeti

Ana değerlendirme evreni (`tum_uygun_ornekler`) değiştirilmeden, ilerideki yerel modellerle adil karşılaştırma yapılabilmesi için `EgitimdeGoruldu` kırılımı ve kalite işaretlerine (`IncelemeGerekli`, `HedefIncelemeGerekli`, `GecmisteIncelemeSayisi`) göre betimleyici alt kümeler ayrıca raporlanmıştır:

| Bölüm | Kapsam | Hedef | Örnek | Seri | Satır MAE | Satır RMSE | Seri Ort. MAE | Seri Medyan MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dogrulama | egitimde_gorulen_seriler | hedef_min | 30.101 | 150 | 0.5869 | 3.9285 | 0.6005 | 0.3817 |
| dogrulama | egitimde_gorulen_seriler | hedef_max | 30.101 | 150 | 0.7258 | 4.5573 | 0.7530 | 0.4640 |
| dogrulama | egitimde_gorulmeyen_seriler | hedef_min | 260 | 2 | 0.2385 | 1.2341 | 0.2424 | 0.2424 |
| dogrulama | egitimde_gorulmeyen_seriler | hedef_max | 260 | 2 | 0.3154 | 1.6172 | 0.3559 | 0.3559 |
| dogrulama | mevcut_ve_hedef_isaretsiz | hedef_min | 30.347 | 152 | 0.5702 | 3.8090 | 0.5847 | 0.3689 |
| dogrulama | mevcut_ve_hedef_isaretsiz | hedef_max | 30.347 | 152 | 0.7068 | 4.4113 | 0.7353 | 0.4640 |
| dogrulama | mevcut_veya_hedef_isaretli | hedef_min | 14 | 6 | 30.1429 | 41.9796 | 27.6667 | 30.0000 |
| dogrulama | mevcut_veya_hedef_isaretli | hedef_max | 14 | 6 | 34.2143 | 50.2245 | 31.1667 | 33.7500 |
| dogrulama | gecmis_penceresi_isaretsiz | hedef_min | 30.232 | 152 | 0.5762 | 3.8409 | 0.5818 | 0.3673 |
| dogrulama | gecmis_penceresi_isaretsiz | hedef_max | 30.232 | 152 | 0.7111 | 4.4657 | 0.7315 | 0.4640 |
| dogrulama | gecmis_penceresi_isaretli | hedef_min | 129 | 19 | 2.3876 | 12.1247 | 6.2977 | 0.0000 |
| dogrulama | gecmis_penceresi_isaretli | hedef_max | 129 | 19 | 3.3333 | 13.3411 | 7.4181 | 0.0000 |
| test | egitimde_gorulen_seriler | hedef_min | 20.765 | 119 | 4.7409 | 13.7412 | 6.2817 | 3.7380 |
| test | egitimde_gorulen_seriler | hedef_max | 20.765 | 119 | 5.7465 | 16.5766 | 7.8592 | 4.2541 |
| test | egitimde_gorulmeyen_seriler | hedef_min | 376 | 4 | 23.2447 | 67.7867 | 39.7390 | 6.3027 |
| test | egitimde_gorulmeyen_seriler | hedef_max | 376 | 4 | 22.9601 | 63.7598 | 37.8069 | 7.5453 |
| test | mevcut_ve_hedef_isaretsiz | hedef_min | 21.141 | 123 | 5.0700 | 16.3458 | 7.3697 | 3.7380 |
| test | mevcut_ve_hedef_isaretsiz | hedef_max | 21.141 | 123 | 6.0526 | 18.4987 | 8.8331 | 4.2541 |
| test | mevcut_veya_hedef_isaretli | hedef_min | 0 | 0 | - | - | - | - |
| test | mevcut_veya_hedef_isaretli | hedef_max | 0 | 0 | - | - | - | - |
| test | gecmis_penceresi_isaretsiz | hedef_min | 21.121 | 123 | 5.0648 | 16.3341 | 7.3721 | 3.7380 |
| test | gecmis_penceresi_isaretsiz | hedef_max | 21.121 | 123 | 6.0479 | 18.4943 | 8.8275 | 4.2541 |
| test | gecmis_penceresi_isaretli | hedef_min | 20 | 10 | 10.5500 | 25.9162 | 18.1000 | 5.0000 |
| test | gecmis_penceresi_isaretli | hedef_max | 20 | 10 | 11.0500 | 22.6197 | 19.9333 | 5.0000 |

### Eğitimde Görülmeyen Seriler

Doğrulamada **2 seri (260 örnek)**, testte **4 seri (376 örnek)** eğitim bölümünde (`egitim.csv.gz`) hiç yer almamaktadır. Naive Baseline parametre eğitimi gerektirmediğinden bu seriler için de doğrudan tahmin üretebilmektedir; ancak Aşama 4'te ürün/seri başına yerel model eğitildiğinde bu seriler için eğitim verisi bulunmayacaktır:

| Bolum | SeriId | UrunAd | BirimTurId | BirimAciklamasi | Kategori | HedefTuru | OrnekSayisi | IlkHedefTarihi | SonHedefTarihi | MAE | MSE | RMSE | R2 | R2Durumu | DegismediOrani |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dogrulama | 1b84d414-b4b8-4f46-aec3-3fb739842207/b1/h2/k5/a2 | Trabzon Hurması | 1 | Birim kodu 1 — adı doğrulanmadı | Meyve | hedef_min | 97 | 2024-11-13 | 2025-03-28 | 0.2577 | 2.3196 | 1.5230 | 0.9602 | hesaplanabilir | %96.91 |
| dogrulama | 1b84d414-b4b8-4f46-aec3-3fb739842207/b1/h2/k5/a2 | Trabzon Hurması | 1 | Birim kodu 1 — adı doğrulanmadı | Meyve | hedef_max | 97 | 2024-11-13 | 2025-03-28 | 0.5155 | 5.1546 | 2.2704 | 0.7927 | hesaplanabilir | %94.85 |
| dogrulama | 4deda22d-a3bc-4c91-9317-f1aa86802153/b1/h2/k6/a1 | Fesleğen | 1 | Birim kodu 1 — adı doğrulanmadı | Sebze | hedef_min | 163 | 2025-02-21 | 2025-10-24 | 0.2270 | 1.0491 | 1.0242 | 0.7123 | hesaplanabilir | %94.48 |
| dogrulama | 4deda22d-a3bc-4c91-9317-f1aa86802153/b1/h2/k6/a1 | Fesleğen | 1 | Birim kodu 1 — adı doğrulanmadı | Sebze | hedef_max | 163 | 2025-02-21 | 2025-10-24 | 0.1963 | 1.1043 | 1.0509 | 0.8252 | hesaplanabilir | %95.71 |
| test | 1b84d414-b4b8-4f46-aec3-3fb739842207/b1/h2/k5/a2 | Trabzon Hurması | 1 | Birim kodu 1 — adı doğrulanmadı | Meyve | hedef_min | 103 | 2025-11-11 | 2026-04-08 | 10.9709 | 363.1068 | 19.0554 | 0.6821 | hesaplanabilir | %28.16 |
| test | 1b84d414-b4b8-4f46-aec3-3fb739842207/b1/h2/k5/a2 | Trabzon Hurması | 1 | Birim kodu 1 — adı doğrulanmadı | Meyve | hedef_max | 103 | 2025-11-11 | 2026-04-08 | 11.7961 | 368.2039 | 19.1886 | 0.5228 | hesaplanabilir | %15.53 |
| test | 4deda22d-a3bc-4c91-9317-f1aa86802153/b1/h2/k6/a1 | Fesleğen | 1 | Birim kodu 1 — adı doğrulanmadı | Sebze | hedef_min | 197 | 2025-10-27 | 2026-08-17 | 1.6345 | 12.2944 | 3.5063 | 0.9022 | hesaplanabilir | %70.56 |
| test | 4deda22d-a3bc-4c91-9317-f1aa86802153/b1/h2/k6/a1 | Fesleğen | 1 | Birim kodu 1 — adı doğrulanmadı | Sebze | hedef_max | 197 | 2025-10-27 | 2026-08-17 | 3.2944 | 35.8020 | 5.9835 | 0.9175 | hesaplanabilir | %48.73 |
| test | 4deda22d-a3bc-4c91-9317-f1aa86802153/b5/h2/k6/a1 | Fesleğen | 5 | Birim kodu 5 — adı doğrulanmadı | Sebze | hedef_min | 26 | 2026-09-21 | 2026-10-26 | 1.2308 | 2.7692 | 1.6641 | -0.5692 | hesaplanabilir | %30.77 |
| test | 4deda22d-a3bc-4c91-9317-f1aa86802153/b5/h2/k6/a1 | Fesleğen | 5 | Birim kodu 5 — adı doğrulanmadı | Sebze | hedef_max | 26 | 2026-09-21 | 2026-10-26 | 1.5769 | 3.8846 | 1.9709 | -0.6042 | hesaplanabilir | %23.08 |
| test | a30c0224-1b2f-44a7-b985-7213c2308de6/b1/h2/k7/a1 | Ahududu | 1 | Birim kodu 1 — adı doğrulanmadı | İthal Ürünler | hedef_min | 50 | 2025-11-05 | 2026-09-16 | 145.1200 | 33756.8400 | 183.7303 | 0.1667 | hesaplanabilir | %8.00 |
| test | a30c0224-1b2f-44a7-b985-7213c2308de6/b1/h2/k7/a1 | Ahududu | 1 | Birim kodu 1 — adı doğrulanmadı | İthal Ürünler | hedef_max | 50 | 2025-11-05 | 2026-09-16 | 134.5600 | 29669.5200 | 172.2484 | 0.2034 | hesaplanabilir | %8.00 |

Nedenleri:
- `1b84d414-b4b8-4f46-aec3-3fb739842207|b1|h2|k5|a2` (`Trabzon Hurması`, Birim kodu 1): Eğitim döneminde `Hurma` (`AdDonemi=1`) adıyla yer almış, `2024-10-14` tarihinde başlayan `AdDonemi=2` döneminde ilk 20 gözlemlik geçmiş penceresi tamamlandığında ilk hedef tarihi `2024-11-13` (doğrulama dönemi) olmuştur.
- `4deda22d-a3bc-4c91-9317-f1aa86802153|b1|h2|k6|a1` (`Fesleğen`, Birim kodu 1): Kaynak veride ilk kez `2025-01-23` tarihinde gözlenmiştir.
- `4deda22d-a3bc-4c91-9317-f1aa86802153|b5|h2|k6|a1` (`Fesleğen`, Birim kodu 5): `2026-08-18` tarihinde birim kodu 5 olarak ayrı seriye ayrılmıştır.
- `a30c0224-1b2f-44a7-b985-7213c2308de6|b1|h2|k7|a1` (`Ahududu`, Birim kodu 1): İlk kez `2025-10-06` tarihinde gözlenmiş, 20 gözlemlik geçmiş şartı sonrası ilk etiketli hedefi `2025-11-05` (test dönemi) olmuştur.

## 5. Ölçü Birimi Kodu ve Kategori Bazında Bulgular

Kaynak CSV'de ölçü birimi adları (ör. kilogram, adet, bağ, kasa) yer almadığından ve İBB birim sözlüğü servisi doğrulanamadığından, birimler `"Birim kodu X — adı doğrulanmadı"` şeklinde korunmuştur:

| Bölüm | Grup | Hedef | Örnek | Seri | Satır MAE | Satır RMSE | Seri Ort. MAE | Seri Medyan MAE | Seri Medyan R² |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dogrulama | Birim kodu 1 — adı doğrulanmadı | hedef_min | 24.792 | 128 | 0.6496 | 4.1994 | 0.6493 | 0.4452 | 0.9377 |
| dogrulama | Birim kodu 1 — adı doğrulanmadı | hedef_max | 24.792 | 128 | 0.8026 | 4.8750 | 0.8156 | 0.4742 | 0.9449 |
| dogrulama | Birim kodu 3 — adı doğrulanmadı | hedef_min | 3.199 | 14 | 0.4392 | 2.8562 | 0.4664 | 0.3673 | 0.9303 |
| dogrulama | Birim kodu 3 — adı doğrulanmadı | hedef_max | 3.199 | 14 | 0.5486 | 3.2971 | 0.5801 | 0.6020 | 0.9440 |
| dogrulama | Birim kodu 5 — adı doğrulanmadı | hedef_min | 2.141 | 9 | 0.0738 | 0.5775 | 0.0728 | 0.0655 | 0.9278 |
| dogrulama | Birim kodu 5 — adı doğrulanmadı | hedef_max | 2.141 | 9 | 0.1032 | 0.7364 | 0.1014 | 0.0939 | 0.9222 |
| dogrulama | Birim kodu 7 — adı doğrulanmadı | hedef_min | 229 | 1 | 0.2620 | 2.0368 | 0.2620 | 0.2620 | 0.9856 |
| dogrulama | Birim kodu 7 — adı doğrulanmadı | hedef_max | 229 | 1 | 0.2402 | 1.7793 | 0.2402 | 0.2402 | 0.9917 |
| dogrulama | Kategori 5 (Meyve) | hedef_min | 8.103 | 51 | 0.8209 | 5.9597 | 0.8123 | 0.4694 | 0.9604 |
| dogrulama | Kategori 5 (Meyve) | hedef_max | 8.103 | 51 | 1.0118 | 7.0859 | 1.0445 | 0.4955 | 0.9627 |
| dogrulama | Kategori 6 (Sebze) | hedef_min | 19.180 | 87 | 0.5339 | 2.9133 | 0.5254 | 0.3433 | 0.9252 |
| dogrulama | Kategori 6 (Sebze) | hedef_max | 19.180 | 87 | 0.6644 | 3.2371 | 0.6478 | 0.4323 | 0.9249 |
| dogrulama | Kategori 7 (İthal Ürünler) | hedef_min | 3.078 | 14 | 0.2713 | 2.1611 | 0.2444 | 0.0000 | 0.9489 |
| dogrulama | Kategori 7 (İthal Ürünler) | hedef_max | 3.078 | 14 | 0.3200 | 2.4199 | 0.2883 | 0.0000 | 0.9543 |
| test | Birim kodu 1 — adı doğrulanmadı | hedef_min | 17.929 | 106 | 5.4704 | 17.5431 | 7.9459 | 3.9682 | 0.8179 |
| test | Birim kodu 1 — adı doğrulanmadı | hedef_max | 17.929 | 106 | 6.4817 | 19.7791 | 9.4564 | 4.4267 | 0.8059 |
| test | Birim kodu 3 — adı doğrulanmadı | hedef_min | 1.993 | 11 | 4.0472 | 8.0282 | 5.3372 | 2.2582 | 0.7950 |
| test | Birim kodu 3 — adı doğrulanmadı | hedef_max | 1.993 | 11 | 5.2584 | 10.4448 | 7.0321 | 2.9836 | 0.7918 |
| test | Birim kodu 5 — adı doğrulanmadı | hedef_min | 1.219 | 6 | 0.8532 | 1.3646 | 0.9176 | 0.8648 | 0.6815 |
| test | Birim kodu 5 — adı doğrulanmadı | hedef_max | 1.219 | 6 | 1.0402 | 1.5562 | 1.1247 | 1.0656 | 0.7960 |
| test | Kategori 5 (Meyve) | hedef_min | 6.633 | 47 | 5.6273 | 16.4621 | 7.3599 | 4.9459 | 0.7668 |
| test | Kategori 5 (Meyve) | hedef_max | 6.633 | 47 | 6.6633 | 19.7341 | 8.9436 | 6.0250 | 0.7152 |
| test | Kategori 6 (Sebze) | hedef_min | 13.184 | 68 | 3.1322 | 6.3292 | 3.6280 | 2.2534 | 0.8587 |
| test | Kategori 6 (Sebze) | hedef_max | 13.184 | 68 | 3.8365 | 7.2367 | 4.3266 | 3.1204 | 0.8871 |
| test | Kategori 7 (İthal Ürünler) | hedef_min | 1.324 | 8 | 21.5733 | 50.0971 | 39.2321 | 9.4251 | 0.0914 |
| test | Kategori 7 (İthal Ürünler) | hedef_max | 1.324 | 8 | 25.0604 | 54.6956 | 46.4898 | 13.5500 | 0.2097 |

## 6. Örnek Seri Bazlı Sonuçlar (En Yüksek Örnek Sayısına Sahip 10 Seri)

Tüm 152 doğrulama serisi ve 123 test serisinin minimum ve maksimum hedefleri için tam metrikler `seri_metrikleri.csv` dosyasındadır (toplam 550 seri-hedef satırı). Aşağıda örnek sayısı en yüksek 10 seri hem `hedef_min` hem `hedef_max` için ayrı ayrı gösterilmiştir:

### Doğrulama Dönemi (`hedef_min`, ilk 10 seri)

| SeriId | UrunAd | BirimAciklamasi | Kategori | OrnekSayisi | MAE | MSE | RMSE | R2 | R2Durumu | DegismediOrani |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 03f24cf5-adfe-4405-8398-a957d5704d84/b1/h2/k6/a1 | Ispanak | Birim kodu 1 — adı doğrulanmadı | Sebze | 245 | 0.4898 | 3.0612 | 1.7496 | 0.8954 | hesaplanabilir | %91.43 |
| 0d44f686-a528-4270-9ae8-d96d7040b6b7/b1/h2/k6/a1 | Domates (Salkım) | Birim kodu 1 — adı doğrulanmadı | Sebze | 245 | 0.6122 | 5.3061 | 2.3035 | 0.9562 | hesaplanabilir | %91.84 |
| 10ba026c-cadd-4820-84de-df801435e300/b1/h2/k5/a1 | Kavun | Birim kodu 1 — adı doğrulanmadı | Meyve | 245 | 1.1510 | 49.7878 | 7.0560 | 0.9018 | hesaplanabilir | %91.02 |
| 11d5bdfc-e46e-4519-b1ff-f51f44fa33d6/b1/h2/k5/a1 | Kivi (Yerli) | Birim kodu 1 — adı doğrulanmadı | Meyve | 245 | 0.8571 | 22.8571 | 4.7809 | 0.9715 | hesaplanabilir | %94.69 |
| 11e951cf-20da-4b35-b30c-755a9de4baf2/b3/h2/k7/a1 | Ananas | Birim kodu 3 — adı doğrulanmadı | İthal Ürünler | 245 | 1.1020 | 15.5102 | 3.9383 | 0.9487 | hesaplanabilir | %89.80 |
| 1358df5f-fc3f-409b-8a70-d3f24f345515/b1/h2/k6/a2 | Patlıcan I. | Birim kodu 1 — adı doğrulanmadı | Sebze | 245 | 1.3673 | 17.8571 | 4.2258 | 0.9573 | hesaplanabilir | %85.31 |
| 14a33a31-a74d-4802-a80e-833527afe45b/b1/h2/k5/a1 | Limon | Birim kodu 1 — adı doğrulanmadı | Meyve | 245 | 0.9184 | 18.4694 | 4.2976 | 0.9844 | hesaplanabilir | %92.24 |
| 16758602-6200-4eb9-807a-2a8e2180df0c/b5/h2/k6/a1 | Dereotu | Birim kodu 5 — adı doğrulanmadı | Sebze | 245 | 0.0857 | 0.3551 | 0.5959 | 0.9093 | hesaplanabilir | %97.55 |
| 1a90c148-edb7-4c79-b979-c37d647f4ef0/b1/h2/k6/a1 | Patlıcan Bostan | Birim kodu 1 — adı doğrulanmadı | Sebze | 245 | 0.9796 | 9.5918 | 3.0971 | 0.9659 | hesaplanabilir | %86.94 |
| 1caf76f1-3b20-4e53-b54d-3b05df4a93d3/b1/h2/k6/a1 | Fasülye (Çalı) | Birim kodu 1 — adı doğrulanmadı | Sebze | 245 | 2.3061 | 50.1020 | 7.0783 | 0.8665 | hesaplanabilir | %84.08 |

### Doğrulama Dönemi (`hedef_max`, ilk 10 seri)

| SeriId | UrunAd | BirimAciklamasi | Kategori | OrnekSayisi | MAE | MSE | RMSE | R2 | R2Durumu | DegismediOrani |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 03f24cf5-adfe-4405-8398-a957d5704d84/b1/h2/k6/a1 | Ispanak | Birim kodu 1 — adı doğrulanmadı | Sebze | 245 | 0.7347 | 5.7143 | 2.3905 | 0.8851 | hesaplanabilir | %88.98 |
| 0d44f686-a528-4270-9ae8-d96d7040b6b7/b1/h2/k6/a1 | Domates (Salkım) | Birim kodu 1 — adı doğrulanmadı | Sebze | 245 | 0.7959 | 5.6122 | 2.3690 | 0.9595 | hesaplanabilir | %87.35 |
| 10ba026c-cadd-4820-84de-df801435e300/b1/h2/k5/a1 | Kavun | Birim kodu 1 — adı doğrulanmadı | Meyve | 245 | 1.4041 | 66.8735 | 8.1776 | 0.9100 | hesaplanabilir | %90.61 |
| 11d5bdfc-e46e-4519-b1ff-f51f44fa33d6/b1/h2/k5/a1 | Kivi (Yerli) | Birim kodu 1 — adı doğrulanmadı | Meyve | 245 | 1.0612 | 32.6531 | 5.7143 | 0.9736 | hesaplanabilir | %93.47 |
| 11e951cf-20da-4b35-b30c-755a9de4baf2/b3/h2/k7/a1 | Ananas | Birim kodu 3 — adı doğrulanmadı | İthal Ürünler | 245 | 1.3673 | 20.5102 | 4.5288 | 0.9465 | hesaplanabilir | %87.76 |
| 1358df5f-fc3f-409b-8a70-d3f24f345515/b1/h2/k6/a2 | Patlıcan I. | Birim kodu 1 — adı doğrulanmadı | Sebze | 245 | 1.8571 | 22.3469 | 4.7273 | 0.9449 | hesaplanabilir | %76.73 |
| 14a33a31-a74d-4802-a80e-833527afe45b/b1/h2/k5/a1 | Limon | Birim kodu 1 — adı doğrulanmadı | Meyve | 245 | 1.5306 | 62.3469 | 7.8960 | 0.9756 | hesaplanabilir | %89.80 |
| 16758602-6200-4eb9-807a-2a8e2180df0c/b5/h2/k6/a1 | Dereotu | Birim kodu 5 — adı doğrulanmadı | Sebze | 245 | 0.1469 | 0.7918 | 0.8899 | 0.9222 | hesaplanabilir | %94.29 |
| 1a90c148-edb7-4c79-b979-c37d647f4ef0/b1/h2/k6/a1 | Patlıcan Bostan | Birim kodu 1 — adı doğrulanmadı | Sebze | 245 | 1.6327 | 14.6939 | 3.8333 | 0.9583 | hesaplanabilir | %76.33 |
| 1caf76f1-3b20-4e53-b54d-3b05df4a93d3/b1/h2/k6/a1 | Fasülye (Çalı) | Birim kodu 1 — adı doğrulanmadı | Sebze | 245 | 2.8776 | 82.7551 | 9.0970 | 0.8880 | hesaplanabilir | %82.86 |

### Test Dönemi (`hedef_min`, ilk 10 seri)

| SeriId | UrunAd | BirimAciklamasi | Kategori | OrnekSayisi | MAE | MSE | RMSE | R2 | R2Durumu | DegismediOrani |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 03f24cf5-adfe-4405-8398-a957d5704d84/b1/h2/k6/a1 | Ispanak | Birim kodu 1 — adı doğrulanmadı | Sebze | 244 | 3.0164 | 21.9590 | 4.6860 | 0.8180 | hesaplanabilir | %43.44 |
| 0d44f686-a528-4270-9ae8-d96d7040b6b7/b1/h2/k6/a1 | Domates (Salkım) | Birim kodu 1 — adı doğrulanmadı | Sebze | 244 | 3.9672 | 35.6803 | 5.9733 | 0.9196 | hesaplanabilir | %39.75 |
| 10ba026c-cadd-4820-84de-df801435e300/b1/h2/k5/a1 | Kavun | Birim kodu 1 — adı doğrulanmadı | Meyve | 244 | 4.2336 | 48.0369 | 6.9309 | 0.9722 | hesaplanabilir | %21.31 |
| 11d5bdfc-e46e-4519-b1ff-f51f44fa33d6/b1/h2/k5/a1 | Kivi (Yerli) | Birim kodu 1 — adı doğrulanmadı | Meyve | 244 | 4.9180 | 47.2869 | 6.8765 | 0.8356 | hesaplanabilir | %29.51 |
| 11e951cf-20da-4b35-b30c-755a9de4baf2/b3/h2/k7/a1 | Ananas | Birim kodu 3 — adı doğrulanmadı | İthal Ürünler | 244 | 4.5205 | 41.4221 | 6.4360 | 0.7950 | hesaplanabilir | %30.74 |
| 1358df5f-fc3f-409b-8a70-d3f24f345515/b1/h2/k6/a2 | Patlıcan I. | Birim kodu 1 — adı doğrulanmadı | Sebze | 244 | 2.9508 | 33.4180 | 5.7808 | 0.9347 | hesaplanabilir | %53.28 |
| 14a33a31-a74d-4802-a80e-833527afe45b/b1/h2/k5/a1 | Limon | Birim kodu 1 — adı doğrulanmadı | Meyve | 244 | 2.1025 | 12.4959 | 3.5350 | 0.9415 | hesaplanabilir | %52.46 |
| 16758602-6200-4eb9-807a-2a8e2180df0c/b5/h2/k6/a1 | Dereotu | Birim kodu 5 — adı doğrulanmadı | Sebze | 244 | 0.9385 | 2.1434 | 1.4641 | 0.7878 | hesaplanabilir | %40.57 |
| 1a90c148-edb7-4c79-b979-c37d647f4ef0/b1/h2/k6/a1 | Patlıcan Bostan | Birim kodu 1 — adı doğrulanmadı | Sebze | 244 | 3.4139 | 32.7500 | 5.7228 | 0.9669 | hesaplanabilir | %45.49 |
| 1caf76f1-3b20-4e53-b54d-3b05df4a93d3/b1/h2/k6/a1 | Fasülye (Çalı) | Birim kodu 1 — adı doğrulanmadı | Sebze | 244 | 7.7008 | 123.7992 | 11.1265 | 0.9322 | hesaplanabilir | %23.36 |

### Test Dönemi (`hedef_max`, ilk 10 seri)

| SeriId | UrunAd | BirimAciklamasi | Kategori | OrnekSayisi | MAE | MSE | RMSE | R2 | R2Durumu | DegismediOrani |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 03f24cf5-adfe-4405-8398-a957d5704d84/b1/h2/k6/a1 | Ispanak | Birim kodu 1 — adı doğrulanmadı | Sebze | 244 | 2.6025 | 18.7910 | 4.3349 | 0.8835 | hesaplanabilir | %50.00 |
| 0d44f686-a528-4270-9ae8-d96d7040b6b7/b1/h2/k6/a1 | Domates (Salkım) | Birim kodu 1 — adı doğrulanmadı | Sebze | 244 | 3.8443 | 32.4590 | 5.6973 | 0.9441 | hesaplanabilir | %38.52 |
| 10ba026c-cadd-4820-84de-df801435e300/b1/h2/k5/a1 | Kavun | Birim kodu 1 — adı doğrulanmadı | Meyve | 244 | 4.5410 | 50.5246 | 7.1081 | 0.9741 | hesaplanabilir | %18.03 |
| 11d5bdfc-e46e-4519-b1ff-f51f44fa33d6/b1/h2/k5/a1 | Kivi (Yerli) | Birim kodu 1 — adı doğrulanmadı | Meyve | 244 | 5.6557 | 58.8279 | 7.6699 | 0.7735 | hesaplanabilir | %27.46 |
| 11e951cf-20da-4b35-b30c-755a9de4baf2/b3/h2/k7/a1 | Ananas | Birim kodu 3 — adı doğrulanmadı | İthal Ürünler | 244 | 4.2131 | 42.0410 | 6.4839 | 0.8329 | hesaplanabilir | %36.89 |
| 1358df5f-fc3f-409b-8a70-d3f24f345515/b1/h2/k6/a2 | Patlıcan I. | Birim kodu 1 — adı doğrulanmadı | Sebze | 244 | 4.3484 | 57.6762 | 7.5945 | 0.9571 | hesaplanabilir | %38.93 |
| 14a33a31-a74d-4802-a80e-833527afe45b/b1/h2/k5/a1 | Limon | Birim kodu 1 — adı doğrulanmadı | Meyve | 244 | 3.8730 | 29.4139 | 5.4235 | 0.9237 | hesaplanabilir | %34.43 |
| 16758602-6200-4eb9-807a-2a8e2180df0c/b5/h2/k6/a1 | Dereotu | Birim kodu 5 — adı doğrulanmadı | Sebze | 244 | 1.0615 | 2.4877 | 1.5772 | 0.8983 | hesaplanabilir | %36.89 |
| 1a90c148-edb7-4c79-b979-c37d647f4ef0/b1/h2/k6/a1 | Patlıcan Bostan | Birim kodu 1 — adı doğrulanmadı | Sebze | 244 | 3.8156 | 38.7336 | 6.2236 | 0.9661 | hesaplanabilir | %42.62 |
| 1caf76f1-3b20-4e53-b54d-3b05df4a93d3/b1/h2/k6/a1 | Fasülye (Çalı) | Birim kodu 1 — adı doğrulanmadı | Sebze | 244 | 7.1803 | 101.7787 | 10.0885 | 0.9573 | hesaplanabilir | %27.05 |

## 7. Tutarlılık ve Bütünlük Kontrolleri

- **Tekillik ve eşleşme:** Doğrulama (`30.361` kayıt) ve test (`21.141` kayıt) tahmin dosyalarındaki bütün `KaynakKayitNo` ve `HedefKaynakKayitNo` değerleri Aşama 2 ve Aşama 1 kayıtlarıyla birebir eşleşmektedir.
- **Tahmin eşitliği:** Bütün kayıtlarda `tahmin_min == fiyat_min` ve `tahmin_max == fiyat_max` eşitliği tamdır.
- **Aralık ve işaret geçerliliği:** `tahmin_min > tahmin_max` olan kayıt sayısı `0`, negatif veya sıfır tahmin sayısı `0`, `NaN`/sonsuz değer sayısı `0`'dır.
- **Kaynak bütünlüğü:** Orijinal CSV dosyaları ile Aşama 1 ve Aşama 2 çıktılarının SHA-256 özetleri işlem öncesi ve sonrasında tamamen aynıdır.

Sonraki aşama (Aşama 4: Linear Regression, Random Forest ve XGBoost modellerinin eğitimi) kullanıcı onayına bağlıdır.
