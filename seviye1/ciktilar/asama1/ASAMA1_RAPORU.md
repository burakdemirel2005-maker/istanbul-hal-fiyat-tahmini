# Aşama 1 veri analizi ve temizleme raporu

Çalıştırma zamanı: 2026-10-27T18:29:32.562714+03:00

Yalnızca İBB hal fiyatları işlendi. Ana dosya 382.897 kayıt içeriyor. Kesin geçersiz kayıtlar ve tekrarlar ayrıldıktan sonra **380.985 yapısal olarak geçerli gözlem** kaldı. Bu sayı eğitim örneği sayısı değildir; gecikmeler, hedef eşleştirmesi ve dönem ayrımı sonraki aşamalardadır.

## Gerçek veri kapsamı

| Ölçüt | Değer |
|---|---:|
| Ham meyve sebze kaydı | 382.897 |
| Özgün sütun sayısı | 9 |
| Tarih aralığı | 2016-01-04 – 2026-10-26 |
| Meyve sebze kaydı bulunan gün | 2.674 |
| Ürün kimliği | 255 |
| Ham farklı ürün adı | 260 |
| Birim kodu | 9 |
| Boş hücre | 0 |
| Sıfır fiyat içeren satır | 60 |
| Negatif fiyat içeren satır | 0 |
| Minimumun maksimumu geçtiği satır | 91 |
| Alt ve üst fiyatın eşit olduğu ham satır | 276 |

Tüm ürünler dosyası 464.540 satır içeriyor. Bunun HalTurId=2 bölümü, ana dosyayla kayıt çokluğu da korunarak eşleşiyor: **Evet**. Dosyalar birleştirilmedi; su ürünleri temiz veriye katılmadı.

## Yapılan temizleme ve kayıt mutabakatı

| İşlem | Kayıt |
|---|---:|
| Başlangıç | 382.897 |
| Çıkarılan fazla tam tekrar | 1.761 |
| Normalizasyon sonrası ek tekrar | 0 |
| Karantinaya alınan kesin geçersiz veya çelişkili kayıt | 151 |
| Toplam ayrılan | 1.912 |
| Temiz veri | 380.985 |

Her kaynak kaydı temiz dosyada veya dışlama dosyasında tam bir kez yer alıyor. Birden fazla sorun taşıyan kayıtta nedenler birlikte tutuluyor; neden toplamı satır toplamından farklı olabilir. Kaynak kayıt numarası, başlık hariç 1'den başlayan CSV kayıt sırasıdır.

Boş/geçersiz tarih, tarih uyuşmazlığı, sonsuz/sayısal olmayan fiyat, geçersiz kod/kimlik, kapsam dışı hal/kategori, sıfır-negatif fiyat ve ters fiyat aralığı kontrol edildi. Tam tekrarlar ilk özgün kayıt korunarak çıkarıldı. Aynı tarih/kimlik/birim/hal anahtarında farklı kayıtlar varsa tüm çelişkili grup karantinaya alınır; fiyatların ortalaması alınmaz.

Ürün adlarında baş/son ve tekrarlı boşluklar ile Unicode yazımı düzenlendi: 5.499 tekilleştirilmiş kayıtta değişiklik var. Özgün ad UrunAdHam alanında korunuyor. Ürün çeşitleri birleştirilmedi; benzer adlar otomatik eşleştirilmedi. Fiyatlar yuvarlanmadı, ters aralıklar yer değiştirilmedi, sıfırlar doldurulmadı. Eksik günler oluşturulmadı; ileri/geri doldurma, interpolasyon veya kırpma yapılmadı.

## İnceleme gerektiren değerler ve modellemeye hazırlık

Temiz dosyada **356 kayıt** inceleme işareti taşıyor; **380.629 kayıtta** bu işaret yok. Alt/üst oranı en az 10 olan 37 kayıt ve aynı seri içinde önceki gözleme göre fiyatlardan biri en az üç kat değişen 328 kayıt var. İki neden aynı kayıtta bulunabilir. Bu sabit eşikler veri incelemesi içindir; öğrenilmiş aykırılık eşiği değildir ve fiyatların kesin yanlış olduğunu kanıtlamaz.

İşaretli satırlar temiz veride korunmuştur; inceleme listesi ayrıca verilmiştir. Örneğin 2017-10-05 Biber (Renkli) alt 9 / üst 10200 kaydı inceleme gerektirir. Bunu 10,20 olarak düzeltmek için kaynakta kanıt yok. Mevsim araları ve gerçek şoklar da büyük değişim yaratabilir. İşaretli test hedeflerini modelin hatası yüksek olduğu için elemek yasaktır. Nihai değerlendirme evreni sonraki aşamada önceden tanımlanmalıdır.

Dolayısıyla Aşama 2'ye aktarılabilecek temiz gözlem sayısı 380.985; bunların 380.629 tanesi mevcut inceleme kurallarında işaretsizdir. “Tamamı hatasız” iddiası yoktur. Modelin kullanılabilir hedef/özellik satırı sayısı henüz hesaplanmadı. Şüpheli değerlerin nasıl kullanılacağı sonraki aşamada açık bir politika olmalıdır.

## Ürün kimliği birim ve ad dönemleri

Ham veride 13 ürün kimliğinde birden fazla ad, 31 kimlikte birden fazla birim; 9 adda birden fazla kimlik var. SeriId; ürün kimliği, birim, hal, kategori ve ad dönemi bileşimidir. Aynı kimlik/birim/kategori içinde ad değiştiğinde yeni dönem başlar; ilerideki ad geçmişe yazılmaz. Temiz veride 252 ürün kimliği, 287 temel seri ve 299 ad dönemli seri var. Birim değişip eski birime dönülürse oluşan boşluğu sonraki aşama ayrıca kontrol etmelidir.

Kategori 5=Meyve, 6=Sebze, 7=İthal Ürünler olarak mevcut İBB sözlüğünden açıklanmıştır; ithal ürünler dosyada tutulmuştur. Birim isimleri kaynak CSV'de bulunmuyor. Birim kodu korunur, kilogram/adet/demet gibi doğrulanmamış isimler üretilmez. Bu, daha sonra miktar girişi ve satın alma simülasyonu için açık bir kısıttır.

Temiz veride kaydı kalmayan ürün kimlikleri (bütün kayıtları geçersiz/tekrar olarak ayrılanlar):

| TabelaGId | UrunAd | HamKayit |
| --- | --- | --- |
| 0f9b79f7-07d5-466f-9e26-13150ea93f85 | Sarımsak(Taze) | 2 |
| 81b46534-98d0-415a-a309-f1ae3d108b82 | Soğan Kuru(ikinci) | 1 |
| e9ac07ce-471e-4acf-85ee-aa7e48fe71fc | Biber sivri kıl | 1 |

Kategori kapsamı:

| Kategori | HamKayit | TemizKayit |
| --- | --- | --- |
| Meyve | 116774 | 116151 |
| Sebze | 226317 | 225249 |
| İthal Ürünler | 39806 | 39585 |

Ürün serileri tablosunda her seri için alt/üst fiyatın minimumu, medyanı ve maksimumu da verilir. Bunlar yalnızca açıklayıcı istatistiklerdir, öğrenilmiş dönüşüm değildir. Birimleri ve ürünleri farklı gözlemlerin tek bir genel fiyat ortalaması verilmez.

## Tarih sürekliliği ve yıllık kapsam

| Yil | HamKayit | VeriGunu | HamUrun | TemizKayit |
| --- | --- | --- | --- | --- |
| 2016 | 40197 | 248 | 204 | 38416 |
| 2017 | 39717 | 249 | 208 | 39682 |
| 2018 | 39284 | 248 | 193 | 39283 |
| 2019 | 37156 | 247 | 181 | 37101 |
| 2020 | 35796 | 250 | 167 | 35786 |
| 2021 | 35753 | 244 | 180 | 35735 |
| 2022 | 36531 | 248 | 181 | 36520 |
| 2023 | 36653 | 251 | 174 | 36653 |
| 2024 | 32808 | 246 | 169 | 32808 |
| 2025 | 29824 | 246 | 162 | 29824 |
| 2026 | 19178 | 197 | 128 | 19177 |

2026 tam yıl değildir. Kayıt olmayan günler resmî tatil veya halin kapalı olması diye etiketlenmedi. Ana CSV'nin ilk-son günü arasında 1275 takvim gününde meyve-sebze kaydı yok; bunların 147 tanesi hafta içi. Bu sayım sorgu başlangıcından değil ana dosyanın ilk tarihinden başlar.

En seyrek 5 gün:

| Tarih | HamUrunKimligi | TemizKayit |
| --- | --- | --- |
| 2016-07-21 | 1 | 1 |
| 2016-04-04 | 34 | 34 |
| 2016-08-03 | 35 | 35 |
| 2016-08-12 | 68 | 68 |
| 2026-02-26 | 89 | 89 |

Veri günü takvimi ham ana CSV'den saklandı. Böylece geçersiz fiyatı çıkarılan bir gün takvimden kaybolup iki gün sonrasının “ertesi gün” sayılmasına neden olmaz. Takvim, resmî işlem takvimi değildir; “bir sonraki hal işlem günü” hedefi mevcut veride bir sonraki kayıtlı meyve-sebze günüyle temsil edilecektir. Ürünün o gün kaydı yoksa sonraki uzak kaydı hedef almak doğru değildir. Bu aşamada hedef üretilmedi.

## Veri sızıntısı ve ayrılmış test dönemi

Son kaynak tarihinden geriye 12 takvim ayı ayrılacak şekilde **2025-10-27 – 2026-10-26** test sınırı sözleşmeye yazıldı. Sınırlar temizleme sonrası son tarihe kaydırılmadı. Hedef tarihine göre gerçek eğitim/doğrulama/test ayrımı sonraki aşamada yapılacak. Bu aşamada hedef, lag/rolling model özelliği, scaler, imputer, model, hiperparametre araması veya test performansı üretilmedi.

Temizleme kuralları satırın kendi değerlerine, sabit yapısal kurallara ve inceleme için yalnızca önceki gözleme dayanır. Tüm dönem ortalaması/çeyrekliğiyle aykırı kırpma yapılmadı. Kalite işaretleri ve kaynak kayıt numarası gelecekte model girdisi yapılmamalıdır. Gelecekteki ürün adları geçmişe taşınmadı; ürünler test başarısına göre seçilmedi.

Önceki inceleme klasöründe test dönemi başlangıç hataları bulunuyordu. Yeni Aşama 1 betiği bu hesapları çalıştırmıyor veya kullanmıyor. Daha önce bakılmış dönemi hiç görülmemiş test gibi tanıtmamak gerekir; bundan sonraki seçimler yalnızca eğitim/doğrulama ile yapılmalıdır.

## Dosyalar ve doğrulama

- temiz_veri.csv: özgün fiyatlar, temizlenmiş kimlik/adlar, izlenebilirlik ve kalite işaretleri.
- dislanan_kayitlar.csv: çıkarılan tam kopyalar ve karantinadaki özgün kayıtlar, nedenleriyle.
- inceleme_bekleyen.csv: temiz dosyada korunan, inceleme işaretli kayıtların kopyası.
- urun_serileri.csv: seri, kategori, birim kodu, tarih aralığı ve kalite kapsamı; başarı sıralaması yok.
- ad_birim_iliskileri.csv: özgün ad ve birim ilişkileri.
- veri_gunleri.csv: ham veri günü takvimi ve günlük temiz kayıt sayısı.
- yillik_kapsam.csv: yıllık ham/temiz kapsam.
- ozet.json ve veri_sozlesmesi.json: makine tarafından okunabilen kurallar, sayılar ve test sınırı.
- kaynak_manifestosu.json: işlem öncesi/sonrası kaynak ve çıktı SHA-256 özetleri.

Kaynak dosyalarının işlem öncesi/sonrası SHA-256 karşılaştırması eşit. Kaynaklar, önceki indirici ve araştırma dosyaları değiştirilmedi. Hiçbir harici veri dosyası okunup birleştirilmedi; ağ isteği yapılmadı.

Bu aşama veri hazırlığıdır. Streamlit/CSS uygulaması, model eğitimi, maliyet simülasyonu ve model doğrulaması henüz uygulanmadı. Sonraki aşama kullanıcı onayına bağlıdır.
