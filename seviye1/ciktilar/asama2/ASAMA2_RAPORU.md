# Aşama 2 özellik mühendisliği ve kronolojik veri ayrımı

Çalıştırma zamanı: 2026-10-27T18:35:52.180502+03:00

Aşama 1'deki 380.985 gözlemden **354.896 etiketli örnek** üretildi. 54 sayısal özellik var. Hiçbir model eğitilmedi, performans ölçülmedi veya hiperparametre seçilmedi. Yalnızca İBB fiyatları ve gözlem tarihleri kullanıldı.

## Hedef ve tahmin anı

Tahmin anı, t gününün İBB fiyatları yayımlandıktan sonradır. Mevcut t minimum/maksimum fiyatları biliniyor kabul edilir. Hedef, aynı ürün/birim/kategori/ad döneminin bir sonraki ortak meyve-sebze veri günündeki gerçek minimum ve maksimum fiyatıdır. Sabah t fiyatı bilinmeden yapılacak tahmin bu deneyden farklıdır.

Ham gün takvimi kullanılır. Sonraki ortak günde ürün kaydı yoksa haftalar/aylar sonraki ürün kaydına atlanmaz. Ad/birim değişimleri üzerinden hedef eşleştirilmez. Bu takvim resmî işlem takvimi değildir; kaynakta eksik bir gün varsa gerçek ertesi işlem günü bu veriden doğrulanamaz.

## Üretilen özellikler

- Güncel minimum, maksimum, aritmetik orta, fiyat aralığı ve göreli aralık.
- Her iki fiyatın 1, 2, 5, 10 ve 20 önceki gözlem değerleri.
- Her iki fiyat için 5, 10 ve 20 gözlemlik geçmiş ortalama, standart sapma, minimum ve maksimum. Pencereler t gününü içerir, geleceği içermez; standart sapma ddof=0.
- Bir gözlem önceye göre fiyat farkı ve oransal değişim; beş gözlem önceye göre oransal değişim.
- t gününün yıl, ay, hafta günü, yıl günü ve dönemsel sinüs/kosinüs değerleri; önceki gözlemden geçen takvim günü.

Oran değerleri kesirdir; örneğin 0,10 yüzde 10 demektir. Model için hedef günün gerçek takvim bilgisi veya t'den sonra kaç gün geçeceği kullanılmadı. HedefTarih yalnızca etiket eşleştirme ve bölümleme metadatasıdır. Aritmetik orta, gerçekleşmiş ortalama işlem fiyatı değildir.

Tam özellik sırası ozellik_sozlesmesi.json içindedir. Kimlikler, ürün adları, kayıt numaraları, kalite işaretleri, hedef bilgisi ve bölüm etiketleri model girdisi listesinde değildir. Ölçekleme, imputasyon, kırpma, veri genelinden öğrenilen normalizasyon veya özellik seçimi uygulanmadı.

## Kesintili seriler ve başlangıç kaybı

Aynı serinin gözlemleri arasında en az bir ortak veri günü eksikse yeni bağlam başlar. Hafta sonu ortak takvimde bulunmuyorsa cuma–pazartesi ardışık veri günleridir; tek başına kesinti sayılmaz. En büyük lag 20 olduğundan her yeni bağlamda ilk 20 gözlem özellik için yetersizdir. Bu muhafazakâr kural test sonucuna bakılmadan sabitlendi; başarısının en iyi olduğu iddia edilmez.

Birbirini dışlayan dışlama nedenleri:

| Neden | Kayıt |
| --- | --- |
| yetersiz_kesintisiz_gecmis | 24.289 |
| sonraki_veri_gununde_ayni_seri_yok | 1.696 |
| gelecek_veri_gunu_henuz_bilinmiyor | 104 |

354.896 kabul edilen + 26.089 dışlanan = 380.985 temiz gözlem. Dışlama listesi kayıt ve neden düzeyinde saklandı. İnceleme işaretleri dışlama nedeni değildir. Öncelik: bilinmeyen sonraki gün, eksik hedef, yetersiz geçmiş. Aynı kayıtta ikinci bir neden olsa da mutabakat için bir ana neden yazılır.

## Eğitim doğrulama ve test

| Bölüm | Takvim sınırı |
|---|---|
| Eğitim | Hedef tarihi 2024-10-26 ve öncesi |
| Doğrulama | 2024-10-27 – 2025-10-26 |
| Test | 2025-10-27 – 2026-10-26 |

Gerçekte kayıt bulunan hedef tarihleri:

| Bölüm | Örnek | Seri | İlk hedef | Son hedef |
| --- | --- | --- | --- | --- |
| egitim | 303.394 | 285 | 2016-02-02 | 2024-10-25 |
| dogrulama | 30.361 | 152 | 2024-10-28 | 2025-10-24 |
| test | 21.141 | 123 | 2025-10-27 | 2026-10-26 |

Ayrım özellik satırının tarihine değil **hedef tarihine** göre yapıldı; hiçbir hedef tarihi iki bölümde yer almıyor. Bölüm sınırını aşan fakat geçmişi geçerli olan özellik pencereleri korunur. Bu geçmişin kullanılması sızıntı değildir. Eğitim/yeniden eğitim anında yalnızca o anda etiketi bilinen örnekler kullanılmalıdır.

Son 12 ayın sınırı Aşama 1 sözleşmesinden alındı ve değiştirilmedi. Nihai test hiperparametre/özellik/ürün seçimi için kullanılmayacak. Bu aşamada testte yalnızca kapsam ve veri sözleşmesi kontrol edildi. Önceki incelemede bu dönemin başlangıç hatalarına bakılmış olduğu bilgisi korunur; tamamen hiç görülmemiş test iddiası yoktur.

## Kalite işaretleri ve ürün kapsamı

İnceleme işaretli mevcut gözlem veya geçmiş pencere bilgisi yalnızca denetim metadatasında; HedefIncelemeGerekli etiket kalitesi metadatasında tutulur. Bunlar X matrisine girmez. Şüpheli hedefler test hatası yükseltebilir diye silinmedi. Ürünler test başarısına veya son dönemde bulunma şartına göre seçilmedi; bütün uygun seriler veri setlerinde tutuldu.

Eğitimde hiç görülmeyen seri sayısı: doğrulamada 2, testte 4. Bunlar da korunur; ileride ürün başına modeli olmayan seriye sahte tahmin üretilemez. urun_kapsami.csv her serinin bölüm kayıt sayılarını verir; bunlar başarı ölçütü veya ürün seçimi değildir.

## Son gün için hazır girdiler

Son ortak veri tarihi 2026-10-26. Bu tarihteki 104 gözlemden 91 tanesinin bütün özellikleri hazır. Bunlar son_gun_girdileri.csv içinde aynı özellik fonksiyonuyla üretildi; gelecekteki fiyat kullanılmadı ve tahmin üretilmedi. Son gözlemi eski olan veya geçmişi yetersiz seri bilgisi son_gozlem_durumu.csv içinde tutulur.

Bu girdiler, sonradan eğitilecek gerçek modeller içindir. Fiyat tahmini ve Streamlit uygulaması henüz yoktur. Tahmin için ertesi tarihin kesinmiş gibi atanması bu aşamanın çıktısı değildir.

## Dosyalar ve doğrulama

egitim.csv.gz, dogrulama.csv.gz ve test.csv.gz sıkıştırılmış CSV dosyalarıdır; Pandas read_csv ile doğrudan okunabilir. Özellik listesi ve hedef listesi açık sözleşmede saklıdır; bütün sayısal sütunları körlemesine modele vermek hatalıdır. Çıktılar orijinal fiyatları değiştirmez. Aşama 1 dosyaları ve orijinal CSV hashleri işlem öncesi/sonrası karşılaştırılır. Ayrı doğrulayıcı kaydedilmiş örnekleri, gerçek hedefleri, geçmiş sınırlarını ve bölümlemeyi kaynakla denetler. Yapılan testler son kullanıcıya verilen çalışma sonucunda ayrıca raporlanır.

Sonraki aşama Naive Baseline'dır. Kullanıcı onayı olmadan çalıştırılmaz.
