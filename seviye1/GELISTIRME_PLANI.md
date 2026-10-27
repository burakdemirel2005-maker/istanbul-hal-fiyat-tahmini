# İstanbul Hal Fiyat Tahmini Seviye 1 geliştirme planı

Bu plan kullanıcının 27 Ekim 2026 kapsamını esas alır. Araştırma raporundaki kur, motorin, meteoroloji, HKS ve diğer ek veri önerileri Seviye 1 kapsamına dahil değildir. Önceki çalışan indirici ve inceleme kodları korunur. Aşama 1 ve Aşama 2 tamamlandıktan sonra kullanıcının onayıyla Aşama 3 (Naive Baseline) de uygulandı; her sonraki aşama ayrıca kullanıcı onayı bekler.

## Zorunlu teslimat ve aşamalar

| Aşama | Teslimat | Kabul koşulu |
|---|---|---|
| 1 Veri analizi ve temizleme | Temiz CSV, karantina ve inceleme listeleri, kalite raporu, dosya hashleri | Kaynaklar değişmemiş; bütün satırlar mutabık; kurallar tekrarlanabilir |
| 2 Özellikler ve kronoloji | Geçmiş fiyat/takvim özellikleri, ertesi işlem günü alt/üst hedefleri, tarih tabanlı eğitim/doğrulama/test | Yalnızca tahmin anında bilinen veri; hedef tarihine göre ayrım; son 12 ay test |
| 3 Naive Baseline | Bugünün alt/üst fiyatını taşıyan referans | Ortak hedef tarihleri; gerçek değerlerle MAE, MSE, RMSE, R² |
| 4 Makine öğrenmesi | Linear Regression, Random Forest, XGBoost; kaydedilmiş modeller/preprocessing | Seçim yalnızca eğitim/doğrulamada; test ayar için kullanılmaz |
| 5 Değerlendirme | Ürün ve alt/üst hedef bazında ortak koşullarda tablo/grafik | Başlangıçla kıyas; doğrulama seçimi ve bağımsız test raporu ayrı |
| 6 Satın alma hesabı | Miktar × mevcut/tahmini alt, üst veya orta fiyat; TL/yüzde fark | Aynı birim; sıfır bölme kontrolü; açıkça varsayımsal maliyet senaryosu |
| 7 Streamlit uygulaması | Türkçe, responsive, Plotly etkileşimli altı bölüm; tutarlı CSS | Gerçek kaydedilmiş modelle tahmin; sayfa açılışında eğitim yok |
| 8 Son kontroller | Entegrasyon/UI testleri, README, kurulum/çalıştırma açıklamaları | Eksik dosya, bilinmeyen ürün, model tutarsızlığı ve gerçek tahmin yolu doğrulanmış |

## Streamlit uygulamasının zorunlu bölümleri

1. **Ana Panel:** ürün sayısı, veri tarih aralığı, veri güncelleme tarihi, genel fiyat özetleri ve Plotly grafikler. Veri son gözlem tarihi ile dosya işleme zamanı ayrı gösterilir. Farklı birimlerdeki bütün fiyatlar tek kilogram ortalaması gibi sunulmaz.
2. **Ürün Analizi:** aranabilir ürün/çeşit/birim seçimi, kategori, ölçü birimi veya doğrulanmamış kod açıklaması, geçmiş min/max grafikleri, aylık/yıllık değişim ve mevsimsellik. Geçmişte adı farklı olan seriler otomatik birleştirilmez.
3. **Fiyat Tahmini:** son gözlem tarihi ve fiyatları, ertesi işlem günü min/max tahmini, beklenen yüzde değişimi, kullanılan model ve ürün bazında geçmiş test hatası, fiyat geçmişiyle tahmin grafiği. Eğitilmiş modeli olmayan üründe sahte tahmin gösterilmez. Resmî işlem takvimi bilinmediğinde sonraki tarihin kesinliği açıklanır.
4. **Satın Alma Planlama:** ürün/miktar ve minimum/maksimum/aritmetik orta fiyat seçimi; mevcut maliyet, ertesi işlem günü tahmini maliyet ve TL/yüzde fark. Orta fiyat gerçekleşmiş işlem ortalaması değildir; maliyet hesabı varsayımsaldır. Ölçü birimi belirsizken kilogram gibi isimler uydurulmaz.
5. **Model Karşılaştırma:** Naive Baseline, Linear Regression, RF ve XGBoost; min/max ayrı filtreler; MAE/MSE/RMSE/R² tabloları ve grafikler. Seçilen model doğrulamaya dayanır; test kazananını otomatik üretim modeli seçmek yasaktır.
6. **Proje Hakkında:** amaç, İBB kaynağı/kapsamı, temizleme, algoritmalar, tarih ayrımı ve veri/model/karar desteği sınırlılıkları.

## CSS ve kullanıcı deneyimi hedefi

Profesyonel ve sade dashboard: açık arka plan, koyu lacivert metin, petrol yeşili ana vurgu, fiyat değişimlerinde renk yanında açık metin ve işaretler. Tutarlı sistem yazı tipi, okunabilir kartlar, dengeli boşluklar, ince sınırlar ve hafif köşeler. Küçük ekranda kartlar alt alta; grafik ve tablolar taşmadan kullanılabilir. Gereksiz animasyon yok. CSS değişiklikleri uygulamanın kendi alanlarına odaklanır; Streamlit'in kırılgan iç sınıflarına kapsamlı bağımlılık kurulmaz. Bu aşamada CSS veya arayüz kodu yazılmadı.

## Teknik sözleşmeler

- Basit Python düzeni: veri hazırlama/eğitim ayrı komutlar, uygulama yalnızca kayıtlı çıktıları okur. İlk teslim için veritabanı, API sunucusu veya ayrı frontend gerekmiyor.
- Python, Pandas, NumPy, scikit-learn, XGBoost, Streamlit ve Plotly planlanır. Paket sürümleri ilgili aşamada mevcut ortamla doğrulanıp kaydedilir; bu aşamada kurulum yapılmaz.
- Eğitim çıktısı model + preprocessing + özellik sırası + ürün/birim kimliği + eğitim son tarihi + veri/ayar sürümünü birlikte saklar. Yalnızca güvenilir yerel model dosyaları yüklenir.
- Streamlit'te veri/model yükleme önbelleğe alınır; dosya/sürüm değişince yenilenir. Eğitim, sayfa açılışı veya widget değişimiyle tetiklenmez.
- Eksik veri/model/özellik uyumsuzluğunda Türkçe hata gösterilir. Model yoksa tahmin/maliyet alanı boş bırakılır; uydurma sayı gösterilmez.
- Min tahmin > max tahmin ise ham tutarsızlık sayısı ve kullanılan düzeltme raporlanır. Düzeltme yöntemi doğrulamada sabitlenir; testte değiştirilmez. Düzeltilemeyen sonuç kullanıcıya tahmin diye verilmez.
- Eğitim/test yalnızca aynı ürün ve birimdeki doğrulanabilir ardışık hedeflerde yapılır. Kaynak kayıt numarası ve kalite işaretleri model özelliği değildir.
- Mevcut son kaynak tarihi 26.10.2026 olduğundan son 12 ay 27.10.2025–26.10.2026 olarak ayrılır. Gerçek hedef tarihine göre bölme Aşama 2'de yapılacak.
- Önceki incelemede test dönemi başlangıç hataları görülmüştü. Bu gerçek saklanmaz; önceki raporlar yeni model veya ürün seçiminin girdisi yapılmaz.

## Güncel Tamamlanma Durumu

Aşama 1'den Aşama 8'e kadar tüm proje aşamaları, 3 Ekim 2025 fiyat sürekliliği araştırması, `Linear Regression`, `Random Forest` ve `XGBoost` modellerinin eğitimi ve doğrulama kümesinde kilitlenmesi (`ciktilar/asama4/`), karşılaştırma tabloları ve grafikleri (`ciktilar/asama5/`), tekil ve çoklu ürün sepeti satın alma karar destek modülü (`ciktilar/asama6/`), 6 sayfalı Türkçe Streamlit web uygulaması (`app.py`), 20 bölümlük akademik proje raporu (`PROJE_RAPORU.md`) ve 47 birim/entegrasyon testi (`test_asama1.py`, `test_asama2.py`, `test_asama3.py`, `test_asama4_8.py`) eksiksiz tamamlanmış ve bağımsız doğrulayıcılarla (`dogrula_asama1.py`, `dogrula_asama2.py`, `dogrula_asama3.py`, `dogrula_asama4_6.py`) doğrulanmıştır.
