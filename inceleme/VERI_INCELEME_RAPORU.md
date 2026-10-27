# İstanbul hal fiyatları veri incelemesi

İnceleme tarihi: 27 Ekim 2026

Bu veriler, seçilen meyve ve sebzelerin bir sonraki kayıtlı hal veri günündeki alt ve üst fiyatlarını tahmin eden bir ders projesi için yeterli büyüklüktedir. Ana kaynak meyve-sebze CSV dosyası olmalıdır. Ham veriler doğrudan modele verilmeden önce tekrarlar, geçersiz fiyatlar, ürün kimlikleri, birim değişimleri ve zaman aralıkları ele alınmalıdır. Bu çalışma veri denetimidir; makine öğrenmesi modelleri eğitilmemiştir.

## İncelenen kaynaklar ve doğrulamanın kapsamı

- `/Users/burakdemirel/Desktop/IBB_Hal_Meyve_Sebze_2016_2026.csv`
- `/Users/burakdemirel/Desktop/IBB_Hal_Tum_Urunler_2016_2026.csv`
- Çalışma klasöründeki `ibb_hal/indir.py`, `getCategories.json`, `veriler/gunluk_erisim_durumu.csv` ve kaynak CSV kopyaları.

Masaüstü dosyaları ile çalışma klasöründeki kaynak kopyalarının SHA-256 özetleri eşleşmektedir. İndirici kodunda kaynak adresi `https://halfiyatlaripublicdata.ibb.gov.tr/api/HalManager/getProductPricebyDay` olarak yer almaktadır. Bu incelemede servise yeniden istek gönderilmemiştir. Kaynak bütünlüğü yerel kayıtlar üzerinden kontrol edilmiştir; API'nin geçmiş verilerinin gerçeği veya piyasanın tüm işlemlerini eksiksiz temsil ettiği ayrıca doğrulanmamıştır.

## Dosyaların karşılaştırılması

| Ölçüt | Meyve sebze | Tüm ürünler |
|---|---:|---:|
| Satır | 382.897 | 464.540 |
| Sütun | 9 | 9 |
| İlk veri tarihi | 04.01.2016 | 01.01.2016 |
| Son veri tarihi | 26.10.2026 | 26.10.2026 |
| Kayıt bulunan farklı gün | 2.674 | 3.319 |
| Ürün kimliği | 255 | 314 |
| Farklı ürün adı | 260 | 320 |
| Boş hücre | 0 | 0 |
| Tam tekrar fazlalığı | 1.761 | 2.732 |
| Anahtar tekrar fazlalığı | 1.761 | 2.733 |
| Sıfır fiyat içeren satır | 60 | 10.707 |
| Negatif fiyat içeren satır | 0 | 0 |
| Minimumun maksimumu geçtiği satır | 91 | 126 |

Anahtar: tarih + ürün kimliği + birim kodu + hal türü. Tekrar fazlalığı, ilk kopya korunduğunda kalan gereksiz satır sayısıdır. Meyve-sebze dosyasında tekrar gruplarında toplam 3.522 satır vardır. Tam tekrarlar kaldırılınca bu dosyada çelişkili anahtar kalmamıştır. Tüm ürünler dosyasında ise tam tekrar dışında bir anahtar çakışması bulunmaktadır.

Tüm ürünler dosyasında `HalTurId=2` ile filtrelenen kayıtlar, sıraları ve değerleriyle meyve-sebze dosyasına tamamen eşittir. Kalan 81.643 kayıt su ürünleri tarafındadır. İki dosyayı alt alta eklemek meyve-sebze gözlemlerini iki kez sayar.

## Sütunların projedeki anlamı

| Sütun | Kullanım |
|---|---|
| Tarih | Gözlem tarihi, zaman sırası ve takvim özellikleri |
| TabelaGId | Kaynak ürün kimliği; seri oluşturmanın temel alanı |
| UrunAd | Ürünün o kayıttaki adı; tek başına güvenilir seri anahtarı değil |
| BirimTurId | Birim kodu; farklı kodlardaki fiyatlar doğrudan birleştirilmemeli |
| KategoriId | Yerel kategori kataloğunda 5 meyve, 6 sebze, 7 ithal ürünler |
| EnDusukFiyat | O günün alt fiyatı; gelecek gün için ayrı hedef |
| EnYuksekFiyat | O günün üst fiyatı; gelecek gün için ayrı hedef |
| GuneAit | Kaynak zaman bilgisi; tarih kısmı Tarih ile bütün satırlarda aynı |
| HalTurId | Meyve-sebze dosyasında sabit 2; bu dosyada ayırt edici özellik değil |

Meyve-sebzede 116.774 meyve, 226.317 sebze ve 39.806 ithal ürün kaydı vardır. İthal ürünler de bu dosyanın kapsamındadır. Birim kodlarının resmi açıklaması mevcut yerel kategori kataloğunda bulunmamaktadır. Kod 1 dahil hiçbir kodun adı bu denetimde resmen doğrulanmış kabul edilmemelidir. Sonuçların TL/kg gibi birimlerle sunulmasından önce birim sözlüğü doğrulanmalıdır.

Bu alanlarda satış miktarı, gerçekleşmiş ortalama işlem fiyatı, işletmenin alış maliyeti, stok veya talep bulunmamaktadır. Alt ve üst fiyatın aritmetik ortalaması hesaplanabilir ancak gerçekleşmiş ortalama satış fiyatı olarak adlandırılamaz.

## Veri kalitesi ve uygulanacak işlemler

1. **Tekrarlar:** Meyve-sebzede 1.761 tam tekrarın bir kopyası tutulmalı. Ham dosya korunmalı ve bu işlem türetilmiş veri üzerinde yapılmalı.
2. **Geçersiz fiyatlar:** 60 sıfır fiyatlı ve 91 ters aralıklı satır tespit edildi. Tekrarlar çıkarıldıktan sonra bu kayıtlar dışlandığında profil için 380.985 satır kalmaktadır. Sıfır fiyatı ücretsiz ürün olarak yorumlamamak; ters aralığı otomatik yer değiştirmek yerine inceleme listesinde tutmak gerekir.
3. **Eşit alt ve üst:** 276 satırda fiyatlar eşittir. Eşitlik tek başına hata değildir; sıfır olanlar ayrıca geçersiz fiyat kontrolüne tabidir.
4. **Ürün adı ve kimlik:** 13 kimlik birden fazla adla, 9 ad birden fazla kimlikle görülüyor. Örneğin aynı kimlikte `Hurma / Trabzon Hurması`, `Soğan (Beyaz) / Soğan Kuru` ve `Asma Yaprağı / Yaprak (Salamura)` görülmektedir. Bunların hepsi yalnızca yazım düzeltmesi kabul edilemez. Özellikle ürün anlamı değişebilen adlar elle incelenmeli, gerekirse dönemler ayrı tutulmalıdır.
5. **Birim değişimleri:** 31 ürün kimliğinde birden fazla birim kodu vardır. Temel seri anahtarı ürün kimliği + birim kodu + hal türü olmalıdır. Ad değişiklikleri için bu anahtar da tek başına yeterli semantik güvence sağlamaz.
6. **Metin boşlukları:** 5.520 satırın ürün adında başta veya sonda boşluk vardır. Görüntüleme alanında temizlenebilir; özgün ad ayrıca korunmalıdır.
7. **Aykırı değerler:** 05.10.2017 tarihli Biber (Renkli) kaydında alt fiyat 9, üst fiyat 10.200'dür. Otomatik olarak 10,20 kabul edilmemelidir. Temiz profil içinde üst fiyatı bir önceki kayıtlı gözleme göre en az üç katına çıkan veya üçte birine düşen 168 satır ayrı listelenmiştir. Bu liste hata hükmü değildir; mevsim araları da büyük değişim yaratabilir. Alt ve üst fiyat için ek inceleme yapılmalı; eşikler eğitim döneminde belirlenmelidir.

İnceleme listeleri aynı klasörde CSV olarak bulunmaktadır. Henüz nihai temiz eğitim dosyası oluşturulmamıştır; özellikle aykırı değer ve ad değişikliği kararları açıklanabilir bir kurala bağlanmalıdır.

## Zaman kapsamı ve eksik günler

Yerel erişim günlüğü 01.01.2016–27.10.2026 arasındaki 3.953 takvim gününü kapsıyor. Günlükte 3.319 veri bulunan yanıt ve 634 başarılı boş yanıt var; hata durumu kayıtlı değil. Bu, bütün günlerde meyve-sebze bulunduğu anlamına gelmez. Meyve-sebze olmayan 1.279 takvim gününün 1.130'u hafta sonudur; 149'u hafta içine denk gelir. Boş yanıt, işlem yapılmadığını tek başına kanıtlamaz.

| Yıl | Meyve sebze satırı | Kayıtlı gün | Ürün kimliği |
|---|---:|---:|---:|
| 2016 | 40.197 | 248 | 204 |
| 2017 | 39.717 | 249 | 208 |
| 2018 | 39.284 | 248 | 193 |
| 2019 | 37.156 | 247 | 181 |
| 2020 | 35.796 | 250 | 167 |
| 2021 | 35.753 | 244 | 180 |
| 2022 | 36.531 | 248 | 181 |
| 2023 | 36.653 | 251 | 174 |
| 2024 | 32.808 | 246 | 169 |
| 2025 | 29.824 | 246 | 162 |
| 2026 | 19.178 | 197 | 128 |

2026 tamamlanmış bir yıl değildir. Buna ek olarak günlük medyan satır sayısı 2025'te 128, 2026'nın mevcut bölümünde 97'dir. Kaynakta ürün kapsamının değiştiği görülmektedir; nedeni bu dosyalardan kesinleştirilemez. 21.07.2016 gününde yalnızca bir ürün kaydı vardır. Dolayısıyla toplam satır sayısı, her ürünün kesintisiz izlendiği anlamına gelmez.

Eksik günler takvime eklenip gelecekten geriye doldurulmamalıdır. Sezon arası kaydı olmayan üründe bir sonraki kaydı hedef almak, ertesi işlem günü yerine haftalar veya aylar sonrasını tahmin etmeye dönüşebilir.

## Tahmin hedefinin doğru tanımlanması

Bir gözlem tarihi t için özellikler t günü yayımlanan bilgiler ve daha eski kayıtlardan üretilmeli. Hedef, aynı ürün ve birimin bir sonraki ortak meyve-sebze veri günündeki alt ve üst fiyatıdır. Gün içi yayımlanma saati bilinmediğinden t fiyatının mevcut olduğu varsayımı açıkça yazılmalıdır.

Ortak takvim, bu CSV'de en az bir meyve-sebze kaydı bulunan günlerden oluşturulmuştur; resmi işlem takvimi olduğu doğrulanmamıştır. Sonraki ortak gün için ürün kaydı yoksa hedef bilinmiyor kabul edilir. Kaynağın eksik günleri nedeniyle gerçek bir sonraki işlem günüyle fark olabileceği raporlanmalıdır.

Kontrollü profilde 379.192 geçerli ertesi veri günü çifti bulundu. Sonraki ürün kaydı bulunduğu halde sonraki ortak veri günüyle uyuşmayan 1.506 çift hedefe alınmadı. Her serinin son kaydında gelecek hedef bulunmaması normaldir. Hedefler geçersiz fiyatlar çıkarıldıktan sonra üretilirken de bu ortak takvim kontrolü korunmuştur.

## Ürün seçimi

Seri profili ürün kimliği + birim + hal türü düzeyindedir. Aday filtresi: en az 500 eğitim hedefi, 180 doğrulama hedefi, 180 test hedefi; kendi aktif tarih aralığında en az yüzde 85 kayıt kapsamı ve son veri gününde kayıt. Bu ön filtreyle 64 seri bulundu. Bu eşikler ders projesini yönetilebilir tutan inceleme tercihleridir; bilimsel bir zorunluluk değildir.

İlk uygulama için 10 anlaşılır aday: Domates, Domates (Salkım), Patates, Havuç (Beypazarı), Lahana (Beyaz), Brokoli, Elma (Starkin), Armut (Santamaria), Muz (Yerli), Limon. Hepsi mevcut aday tablosunda bulunmaktadır. Her birinin kimlik ve birim kodu tablo üzerinden sabitlenmeli; ürün adının dönem içindeki anlamı son kez kontrol edilmelidir. Farklı domates çeşitleri ayrı seriler olarak kalmalıdır.

Bu liste fiyat tahmini başarısına göre seçilmemiştir. Uygulama öncesinde kapsam ve ürün anlamı üzerinden sabitlenmelidir. Ürün seçiminde yakın dönem kayıt varlığının kullanılması, sonuçların sürekli takip edilen seçili ürünlere ait olduğu anlamına gelir; bütün ürünlere genellenmemelidir.

## Eğitim ve değerlendirme düzeni

- Eğitim: hedef tarihi 26.10.2024 ve öncesi.
- Doğrulama: hedef tarihi 27.10.2024–26.10.2025.
- Test: hedef tarihi 27.10.2025–26.10.2026.

Bunlar mevcut son tarihe göre önerilen takvim sınırlarıdır; her sınır günü veri içermez. Ayrım özellik satırının tarihiyle değil hedef tarihiyle yapılmalıdır. Özellik ve hedefin aynı bölümde olmasını gerektiren bir protokol seçilirse sınır çiftleri ayrıca çıkarılabilir.

Başlangıç modeli bugünün alt ve üst fiyatını bir sonraki gün için aynen kullanmalıdır. Test döneminde incelenebilir 23.332 çiftte alt fiyat yüzde 39,91, üst fiyat yüzde 34,74 oranında değişmemiştir. Bu oranlar tahmin doğruluğu veya model başarısı değildir; başlangıç modelinin neden ciddi bir karşılaştırma olduğunu gösterir. Ürün bazında başlangıç MAE değerleri seri tablosunda yer alır. Farklı birim ve fiyat ölçeklerini tek genel MAE altında karıştırmak uygun değildir.

Test dönemi bu denetimde kalite ve başlangıç davranışı bakımından incelendi. Bundan sonraki model ve parametre seçimleri doğrulama döneminde yapılmalı; test sonucu görüldükçe ayarlama yapılmamalıdır. Modelleme protokolü sabitlenmeli ve bu denetim şeffafça belirtilmelidir. Daha sonra hiç incelenmemiş ileri tarihli veriler toplanırsa ek bağımsız kontrol yapılabilir.

İlk özellikler: mevcut alt ve üst fiyat, önceki 1/2/5/10 gözlemin fiyatları, yalnızca geçmişi kullanan 5/10/20 gözlemlik hareketli özetler, fiyat aralığı ve takvim bilgileri. Gözlem sayısı ile takvim günü farkı karıştırılmamalı; aradaki gün sayısı ayrıca izlenmeli. Kaynakta olmayan gelecek hava durumu veya gelecekteki fiyatlar eklenmemeli. Ölçekleme ve öğrenilen dönüşümler sadece eğitim verisinde fit edilmelidir.

Önce basit başlangıç modeli, ardından doğrusal regresyon, Random Forest ve XGBoost karşılaştırılabilir. Ürün başına ayrı model, birim karışıklığını azaltır ve ders sunumunda açıklamayı kolaylaştırır. Bütün geçmiş ve yakın dönem eğitim pencereleri arasındaki tercih doğrulama verisinde yapılmalıdır; uzun dönem fiyat seviyeleri değiştiğinden eski verinin otomatik olarak faydalı olacağı varsayılmamalıdır.

Her ürün ve her hedef için MAE ve RMSE; ek olarak R² ve başlangıç modeline göre hata değişimi raporlanmalıdır. Yüzde 90 doğruluk gibi regresyonla uyumsuz tek bir başarı yüzdesi kullanılmamalıdır. Yön tahmini ayrıca incelenecekse artış/azalış/değişmedi sınıfları ve örnek sayıları açık olmalıdır. Tahmin edilen alt fiyatın üst fiyatı geçmesi gibi tutarsızlıklar da ölçülmelidir.

## Satın alma planlamasının sınırları

Mevcut veriyle fiyatın beklenen artış veya düşüşü yardımcı gösterge olarak sunulabilir. Stok, satış miktarı, bozulma maliyeti ve işletme talebi bulunmadığından optimum satın alma miktarı veya gerçekleşmiş tasarruf hesaplandığı iddia edilmemelidir. Örnek arayüzde mevcut fiyat, tahmini alt/üst fiyat, değişim ve geçmiş test hatası gösterilebilir. Alt/üst fiyat tahmini bir güven aralığı değildir.

## Üretilen inceleme dosyaları

- `inceleme_ozeti.json`: denetim sayıları ve dosya özetleri.
- `meyve_sebze_yillik_kapsam.csv`, `tum_urunler_yillik_kapsam.csv`: yıllık kapsam.
- `meyve_sebze_supheli_fiyatlar.csv`, `tum_urunler_supheli_fiyatlar.csv`: sıfır, negatif veya ters aralık kayıtları.
- `meyve_sebze_tekrarlanan_anahtarlar.csv`, `tum_urunler_tekrarlanan_anahtarlar.csv`: tekrar gruplarının tüm satırları.
- `urun_kimlik_ad_birim.csv`: ürün kimliği, ad ve birim ilişkileri.
- `urun_serisi_uygunlugu.csv`: seri kapsamı, dönem hedef sayıları ve başlangıç hataları.
- `surekli_urun_adaylari.csv`: ön filtreyi geçen 64 seri.
- `meyve_sebze_olmayan_gunler.csv`, `gunluk_urun_kapsami.csv`: takvim ve günlük kapsam.
- `buyuk_fiyat_degisimleri.csv`: otomatik karar verilmeden incelenecek büyük üst fiyat değişimleri.

Denetim kodu: `ibb_hal/veri_inceleme.py`. Kod başarıyla çalıştırılmıştır. Kaynak CSV dosyaları değiştirilmemiştir. Henüz model karşılaştırması, nihai temiz eğitim seti veya tasarruf testi yapılmamıştır. Hocanın değerlendirme yönergesi sunulmadığından puan garantisi verilemez.
