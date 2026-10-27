"""Yalnızca Aşama 1: İBB CSV analizi ve denetlenebilir veri temizliği.

Model, hedef, gecikmeli model özelliği veya harici veri üretmez.
Çalıştırma klasöründen bağımsızdır; kaynak CSV'lere yazmaz.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import unicodedata
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
COLUMNS = ['Tarih', 'TabelaGId', 'UrunAd', 'BirimTurId', 'KategoriId',
           'EnDusukFiyat', 'EnYuksekFiyat', 'GuneAit', 'HalTurId']
KEY = ['Tarih', 'TabelaGId', 'BirimTurId', 'HalTurId']
IDENTITY = ['TabelaGId', 'BirimTurId', 'HalTurId', 'KategoriId']
NUMERIC = ['BirimTurId', 'KategoriId', 'EnDusukFiyat', 'EnYuksekFiyat', 'HalTurId']
INTEGER = ['BirimTurId', 'KategoriId', 'HalTurId']
CATEGORY = {5: 'Meyve', 6: 'Sebze', 7: 'İthal Ürünler'}
# Bunlar öğrenilmiş aykırılık sınırları değildir; sadece inceleme işareti koyar.
REVIEW_RULES = {'alt_ust_orani': 10.0, 'onceki_gozleme_gore_kat': 3.0}


def sha256(path: Path) -> str:
    with path.open('rb') as file:
        digest = hashlib.file_digest(file, 'sha256')
    return digest.hexdigest()


def read_source(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    if list(data.columns) != COLUMNS:
        raise ValueError(f'Beklenen 9 sütun/sıra bulunamadı: {path.name}')
    if data.empty:
        raise ValueError(f'Kaynak dosya boş: {path.name}')
    return data


def add_reason(reasons: pd.Series, mask: pd.Series, label: str) -> None:
    reasons.loc[mask] = reasons.loc[mask].map(lambda old: f'{old};{label}' if old else label)


def clean_data(raw: pd.DataFrame, as_of: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Tam tekrarları çıkarır; kesin geçersiz/çelişkili satırları karantinaya alır."""
    source = raw.copy()
    source.insert(0, 'KaynakKayitNo', np.arange(1, len(source) + 1))
    duplicates = raw.duplicated(keep='first')
    work = source.loc[~duplicates].copy()
    raw_work = work.copy()
    reasons = pd.Series('', index=work.index, dtype=str)
    missing = work[COLUMNS].apply(lambda x: x.str.strip().eq('')).any(axis=1)
    add_reason(reasons, missing, 'eksik_zorunlu_alan')
    date = pd.to_datetime(work.Tarih, format='%Y-%m-%d', errors='coerce')
    source_date = pd.to_datetime(work.GuneAit, format='ISO8601', errors='coerce')
    # Kaynak alanı takvim günü içerir; fiyat yayımlanma saati olduğu varsayılmaz.
    source_day = pd.to_datetime(work.GuneAit.str[:10], format='%Y-%m-%d', errors='coerce')
    add_reason(reasons, date.isna() | source_date.isna(), 'gecersiz_tarih')
    add_reason(reasons, date.notna() & source_day.notna() & date.ne(source_day), 'tarih_uyusmazligi')
    add_reason(reasons, date.gt(pd.Timestamp(as_of)), 'gelecek_tarih')
    for column in NUMERIC:
        work[column] = pd.to_numeric(work[column], errors='coerce')
    finite = pd.Series(np.isfinite(work[NUMERIC].to_numpy(dtype=float)).all(axis=1), index=work.index)
    add_reason(reasons, ~finite, 'gecersiz_sayisal_deger')
    for column in INTEGER:
        valid_code = work[column].notna() & work[column].gt(0) & work[column].mod(1).eq(0)
        add_reason(reasons, ~valid_code, f'gecersiz_{column}')
    add_reason(reasons, ~work.KategoriId.isin(CATEGORY), 'kapsam_disi_kategori')
    add_reason(reasons, work.HalTurId.ne(2), 'kapsam_disi_hal')
    add_reason(reasons, work.EnDusukFiyat.le(0) | work.EnYuksekFiyat.le(0), 'sifir_veya_negatif_fiyat')
    add_reason(reasons, work.EnDusukFiyat.gt(work.EnYuksekFiyat), 'ters_fiyat_araligi')
    work['TabelaGId'] = work.TabelaGId.str.strip().str.lower()
    valid_guid = work.TabelaGId.str.fullmatch(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}')
    add_reason(reasons, ~valid_guid, 'gecersiz_urun_kimligi')
    work['UrunAdHam'] = work.UrunAd
    work['UrunAd'] = work.UrunAd.map(lambda x: unicodedata.normalize('NFC', ' '.join(x.split())))
    # Tip/ad normalizasyonundan sonra aynı gözlemin yeni kopyası varsa ayrı kaydet.
    valid = reasons.eq('')
    normalized_duplicates = pd.Series(False, index=work.index)
    normalized_duplicates.loc[valid] = work.loc[valid].duplicated(COLUMNS, keep='first')
    available = valid & ~normalized_duplicates
    conflicts = pd.Series(False, index=work.index)
    conflicts.loc[available] = work.loc[available].duplicated(KEY, keep=False)
    add_reason(reasons, conflicts, 'celiskili_gozlem_anahtari')
    keep = reasons.eq('') & ~normalized_duplicates
    cleaned = work.loc[keep].copy()
    for column in INTEGER:
        cleaned[column] = cleaned[column].astype('int64')
    cleaned['Tarih'] = date.loc[keep].dt.strftime('%Y-%m-%d')
    cleaned['Kategori'] = cleaned.KategoriId.map(CATEGORY)
    # Birim adı kaynak CSV'de yok: kilogram/adet gibi isimler tahmin edilmez.
    cleaned['BirimAciklamasi'] = cleaned.BirimTurId.map(lambda x: f'Birim kodu {x} (adı doğrulanmadı)')
    cleaned = cleaned.sort_values(IDENTITY + ['Tarih', 'KaynakKayitNo'], kind='stable')
    # Sonraki adı geçmişe yaymayız. Her ad değişiminde yeni dönem başlar.
    cleaned['AdDonemi'] = cleaned.groupby(IDENTITY, sort=False).UrunAd.transform(
        lambda x: x.ne(x.shift()).cumsum()).astype('int64')
    cleaned['SeriId'] = (cleaned.TabelaGId + '|b' + cleaned.BirimTurId.astype(str)
                         + '|h' + cleaned.HalTurId.astype(str)
                         + '|k' + cleaned.KategoriId.astype(str)
                         + '|a' + cleaned.AdDonemi.astype(str))
    # Yalnızca tanısal kontroller. Lag/rolling değerleri model verisine eklenmez.
    diagnostic = pd.DataFrame(index=cleaned.index)
    diagnostic['GenisFiyatAraligi'] = cleaned.EnYuksekFiyat.div(cleaned.EnDusukFiyat).ge(REVIEW_RULES['alt_ust_orani'])
    previous = cleaned.groupby('SeriId', sort=False)[['EnDusukFiyat', 'EnYuksekFiyat']].shift(1)
    ratio = cleaned[['EnDusukFiyat', 'EnYuksekFiyat']].div(previous)
    jump = REVIEW_RULES['onceki_gozleme_gore_kat']
    diagnostic['BuyukFiyatDegisimi'] = (ratio.ge(jump) | ratio.le(1 / jump)).any(axis=1)
    review_reasons = pd.Series('', index=cleaned.index, dtype=str)
    add_reason(review_reasons, diagnostic.GenisFiyatAraligi, 'alt_ust_orani_en_az_10')
    add_reason(review_reasons, diagnostic.BuyukFiyatDegisimi, 'onceki_gozleme_gore_en_az_3_kat_degisim')
    cleaned['IncelemeGerekli'] = review_reasons.ne('')
    cleaned['IncelemeNedeni'] = review_reasons
    # Modelde hangi satırın kullanılacağı bu aşamada başarıya bakılarak seçilmez.
    cleaned = cleaned.sort_values(['Tarih', 'SeriId'], kind='stable').reset_index(drop=True)
    rejected_parts = []
    for mask, label, frame in [
        (duplicates, 'tam_tekrar_fazlaligi', source),
        (normalized_duplicates, 'normalizasyon_sonrasi_tekrar', raw_work),
        (reasons.ne(''), '', raw_work),
    ]:
        part = frame.loc[mask].copy()
        part['DislamaNedeni'] = label if label else reasons.loc[mask]
        rejected_parts.append(part)
    rejected = pd.concat(rejected_parts, ignore_index=True).sort_values('KaynakKayitNo')
    stats = {
        'ham_kayit': len(raw), 'tam_tekrar_fazlaligi': int(duplicates.sum()),
        'normalizasyon_sonrasi_tekrar': int(normalized_duplicates.sum()),
        'kesin_gecersiz_veya_celiskili': int(reasons.ne('').sum()),
        'dislanan_toplam': len(rejected), 'temiz_kayit': len(cleaned),
        'inceleme_isaretli_kayit': int(cleaned.IncelemeGerekli.sum()),
        'inceleme_isaretsiz_kayit': int((~cleaned.IncelemeGerekli).sum()),
        'genis_aralik_isaretli': int(diagnostic.GenisFiyatAraligi.sum()),
        'buyuk_degisim_isaretli': int(diagnostic.BuyukFiyatDegisimi.sum()),
        'duzeltilen_ad_kayit': int(work.UrunAd.ne(work.UrunAdHam).sum()),
        'dislama_nedenleri': reasons.loc[reasons.ne('')].str.split(';').explode().value_counts().to_dict(),
    }
    return cleaned, rejected, stats


def raw_profile(raw: pd.DataFrame) -> dict:
    price = raw[['EnDusukFiyat', 'EnYuksekFiyat']].apply(pd.to_numeric, errors='coerce')
    dates = pd.to_datetime(raw.Tarih, format='%Y-%m-%d', errors='coerce')
    return {
        'satir': len(raw), 'sutun': len(raw.columns), 'ilk_tarih': str(dates.min().date()),
        'son_tarih': str(dates.max().date()), 'veri_gunu': int(dates.nunique()),
        'urun_kimligi': int(raw.TabelaGId.nunique()), 'urun_adi': int(raw.UrunAd.nunique()),
        'birim_kodu': int(raw.BirimTurId.nunique()),
        'bos_hucre': raw.apply(lambda s: s.str.strip().eq('')).sum().to_dict(),
        'sifir_fiyatli': int(price.eq(0).any(axis=1).sum()),
        'negatif_fiyatli': int(price.lt(0).any(axis=1).sum()),
        'ters_fiyat_araligi': int(price.EnDusukFiyat.gt(price.EnYuksekFiyat).sum()),
        'esit_fiyat': int(price.EnDusukFiyat.eq(price.EnYuksekFiyat).sum()),
        'tam_tekrar_fazlaligi': int(raw.duplicated().sum()),
        'anahtar_tekrar_fazlaligi': int(raw.duplicated(KEY).sum()),
        'cok_adli_kimlik': int(raw.groupby('TabelaGId').UrunAd.nunique().gt(1).sum()),
        'cok_birimli_kimlik': int(raw.groupby('TabelaGId').BirimTurId.nunique().gt(1).sum()),
        'cok_kimlikli_ad': int(raw.groupby('UrunAd').TabelaGId.nunique().gt(1).sum()),
        'kategori_satirlari': raw.KategoriId.value_counts().sort_index().to_dict(),
        'birim_satirlari': raw.BirimTurId.value_counts().sort_index().to_dict(),
    }


def atomic_csv(data: pd.DataFrame, path: Path) -> None:
    temp = path.with_suffix('.csv.tmp')
    data.to_csv(temp, index=False, encoding='utf-8-sig', lineterminator='\n')
    temp.replace(path)


def json_write(data: dict, path: Path) -> None:
    def convert(value):
        if isinstance(value, np.integer):
            return int(value)
        raise TypeError(type(value).__name__)
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=convert, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def table(frame: pd.DataFrame) -> str:
    header = '| ' + ' | '.join(map(str, frame.columns)) + ' |'
    separator = '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |'
    rows = ['| ' + ' | '.join(str(x).replace('|', '/') for x in row) + ' |'
            for row in frame.itertuples(index=False, name=None)]
    return '\n'.join([header, separator, *rows])


def make_report(summary: dict, yearly: pd.DataFrame, daily: pd.DataFrame, output: Path) -> None:
    raw, cleaned, test = summary['ana_veri'], summary['temizleme'], summary['test_donemi_sozlesmesi']
    n = lambda value: f'{value:,}'.replace(',', '.')
    text = f'''# Aşama 1 veri analizi ve temizleme raporu

Çalıştırma zamanı: {summary['calistirma_zamani']}

Yalnızca İBB hal fiyatları işlendi. Ana dosya {n(raw['satir'])} kayıt içeriyor. Kesin geçersiz kayıtlar ve tekrarlar ayrıldıktan sonra **{n(cleaned['temiz_kayit'])} yapısal olarak geçerli gözlem** kaldı. Bu sayı eğitim örneği sayısı değildir; gecikmeler, hedef eşleştirmesi ve dönem ayrımı sonraki aşamalardadır.

## Gerçek veri kapsamı

| Ölçüt | Değer |
|---|---:|
| Ham meyve sebze kaydı | {n(raw['satir'])} |
| Özgün sütun sayısı | {raw['sutun']} |
| Tarih aralığı | {raw['ilk_tarih']} – {raw['son_tarih']} |
| Meyve sebze kaydı bulunan gün | {n(raw['veri_gunu'])} |
| Ürün kimliği | {raw['urun_kimligi']} |
| Ham farklı ürün adı | {raw['urun_adi']} |
| Birim kodu | {raw['birim_kodu']} |
| Boş hücre | {sum(raw['bos_hucre'].values())} |
| Sıfır fiyat içeren satır | {raw['sifir_fiyatli']} |
| Negatif fiyat içeren satır | {raw['negatif_fiyatli']} |
| Minimumun maksimumu geçtiği satır | {raw['ters_fiyat_araligi']} |
| Alt ve üst fiyatın eşit olduğu ham satır | {raw['esit_fiyat']} |

Tüm ürünler dosyası {n(summary['karsilastirma']['tum_urunler_satir'])} satır içeriyor. Bunun HalTurId=2 bölümü, ana dosyayla kayıt çokluğu da korunarak eşleşiyor: **Evet**. Dosyalar birleştirilmedi; su ürünleri temiz veriye katılmadı.

## Yapılan temizleme ve kayıt mutabakatı

| İşlem | Kayıt |
|---|---:|
| Başlangıç | {n(cleaned['ham_kayit'])} |
| Çıkarılan fazla tam tekrar | {n(cleaned['tam_tekrar_fazlaligi'])} |
| Normalizasyon sonrası ek tekrar | {n(cleaned['normalizasyon_sonrasi_tekrar'])} |
| Karantinaya alınan kesin geçersiz veya çelişkili kayıt | {n(cleaned['kesin_gecersiz_veya_celiskili'])} |
| Toplam ayrılan | {n(cleaned['dislanan_toplam'])} |
| Temiz veri | {n(cleaned['temiz_kayit'])} |

Her kaynak kaydı temiz dosyada veya dışlama dosyasında tam bir kez yer alıyor. Birden fazla sorun taşıyan kayıtta nedenler birlikte tutuluyor; neden toplamı satır toplamından farklı olabilir. Kaynak kayıt numarası, başlık hariç 1'den başlayan CSV kayıt sırasıdır.

Boş/geçersiz tarih, tarih uyuşmazlığı, sonsuz/sayısal olmayan fiyat, geçersiz kod/kimlik, kapsam dışı hal/kategori, sıfır-negatif fiyat ve ters fiyat aralığı kontrol edildi. Tam tekrarlar ilk özgün kayıt korunarak çıkarıldı. Aynı tarih/kimlik/birim/hal anahtarında farklı kayıtlar varsa tüm çelişkili grup karantinaya alınır; fiyatların ortalaması alınmaz.

Ürün adlarında baş/son ve tekrarlı boşluklar ile Unicode yazımı düzenlendi: {n(cleaned['duzeltilen_ad_kayit'])} tekilleştirilmiş kayıtta değişiklik var. Özgün ad UrunAdHam alanında korunuyor. Ürün çeşitleri birleştirilmedi; benzer adlar otomatik eşleştirilmedi. Fiyatlar yuvarlanmadı, ters aralıklar yer değiştirilmedi, sıfırlar doldurulmadı. Eksik günler oluşturulmadı; ileri/geri doldurma, interpolasyon veya kırpma yapılmadı.

## İnceleme gerektiren değerler ve modellemeye hazırlık

Temiz dosyada **{n(cleaned['inceleme_isaretli_kayit'])} kayıt** inceleme işareti taşıyor; **{n(cleaned['inceleme_isaretsiz_kayit'])} kayıtta** bu işaret yok. Alt/üst oranı en az 10 olan {cleaned['genis_aralik_isaretli']} kayıt ve aynı seri içinde önceki gözleme göre fiyatlardan biri en az üç kat değişen {cleaned['buyuk_degisim_isaretli']} kayıt var. İki neden aynı kayıtta bulunabilir. Bu sabit eşikler veri incelemesi içindir; öğrenilmiş aykırılık eşiği değildir ve fiyatların kesin yanlış olduğunu kanıtlamaz.

İşaretli satırlar temiz veride korunmuştur; inceleme listesi ayrıca verilmiştir. Örneğin 2017-10-05 Biber (Renkli) alt 9 / üst 10200 kaydı inceleme gerektirir. Bunu 10,20 olarak düzeltmek için kaynakta kanıt yok. Mevsim araları ve gerçek şoklar da büyük değişim yaratabilir. İşaretli test hedeflerini modelin hatası yüksek olduğu için elemek yasaktır. Nihai değerlendirme evreni sonraki aşamada önceden tanımlanmalıdır.

Dolayısıyla Aşama 2'ye aktarılabilecek temiz gözlem sayısı {n(cleaned['temiz_kayit'])}; bunların {n(cleaned['inceleme_isaretsiz_kayit'])} tanesi mevcut inceleme kurallarında işaretsizdir. “Tamamı hatasız” iddiası yoktur. Modelin kullanılabilir hedef/özellik satırı sayısı henüz hesaplanmadı. Şüpheli değerlerin nasıl kullanılacağı sonraki aşamada açık bir politika olmalıdır.

## Ürün kimliği birim ve ad dönemleri

Ham veride {raw['cok_adli_kimlik']} ürün kimliğinde birden fazla ad, {raw['cok_birimli_kimlik']} kimlikte birden fazla birim; {raw['cok_kimlikli_ad']} adda birden fazla kimlik var. SeriId; ürün kimliği, birim, hal, kategori ve ad dönemi bileşimidir. Aynı kimlik/birim/kategori içinde ad değiştiğinde yeni dönem başlar; ilerideki ad geçmişe yazılmaz. Temiz veride {summary['temiz_kapsam']['urun_kimligi']} ürün kimliği, {summary['temiz_kapsam']['kimlik_birim_hal_kategori']} temel seri ve {summary['temiz_kapsam']['ad_donemli_seri']} ad dönemli seri var. Birim değişip eski birime dönülürse oluşan boşluğu sonraki aşama ayrıca kontrol etmelidir.

Kategori 5=Meyve, 6=Sebze, 7=İthal Ürünler olarak mevcut İBB sözlüğünden açıklanmıştır; ithal ürünler dosyada tutulmuştur. Birim isimleri kaynak CSV'de bulunmuyor. Birim kodu korunur, kilogram/adet/demet gibi doğrulanmamış isimler üretilmez. Bu, daha sonra miktar girişi ve satın alma simülasyonu için açık bir kısıttır.

Temiz veride kaydı kalmayan ürün kimlikleri (bütün kayıtları geçersiz/tekrar olarak ayrılanlar):

{table(pd.DataFrame(summary['temiz_kapsam']['kaydi_kalmayan_urunler']))}

Kategori kapsamı:

{table(pd.DataFrame(summary['temiz_kapsam']['kategori_kapsami']))}

Ürün serileri tablosunda her seri için alt/üst fiyatın minimumu, medyanı ve maksimumu da verilir. Bunlar yalnızca açıklayıcı istatistiklerdir, öğrenilmiş dönüşüm değildir. Birimleri ve ürünleri farklı gözlemlerin tek bir genel fiyat ortalaması verilmez.

## Tarih sürekliliği ve yıllık kapsam

{table(yearly)}

2026 tam yıl değildir. Kayıt olmayan günler resmî tatil veya halin kapalı olması diye etiketlenmedi. Ana CSV'nin ilk-son günü arasında {summary['takvim']['kayitsiz_takvim_gunu']} takvim gününde meyve-sebze kaydı yok; bunların {summary['takvim']['kayitsiz_haftaici_gun']} tanesi hafta içi. Bu sayım sorgu başlangıcından değil ana dosyanın ilk tarihinden başlar.

En seyrek 5 gün:

{table(daily.nsmallest(5, 'HamUrunKimligi')[['Tarih', 'HamUrunKimligi', 'TemizKayit']])}

Veri günü takvimi ham ana CSV'den saklandı. Böylece geçersiz fiyatı çıkarılan bir gün takvimden kaybolup iki gün sonrasının “ertesi gün” sayılmasına neden olmaz. Takvim, resmî işlem takvimi değildir; “bir sonraki hal işlem günü” hedefi mevcut veride bir sonraki kayıtlı meyve-sebze günüyle temsil edilecektir. Ürünün o gün kaydı yoksa sonraki uzak kaydı hedef almak doğru değildir. Bu aşamada hedef üretilmedi.

## Veri sızıntısı ve ayrılmış test dönemi

Son kaynak tarihinden geriye 12 takvim ayı ayrılacak şekilde **{test['baslangic_dahil']} – {test['bitis_dahil']}** test sınırı sözleşmeye yazıldı. Sınırlar temizleme sonrası son tarihe kaydırılmadı. Hedef tarihine göre gerçek eğitim/doğrulama/test ayrımı sonraki aşamada yapılacak. Bu aşamada hedef, lag/rolling model özelliği, scaler, imputer, model, hiperparametre araması veya test performansı üretilmedi.

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
'''
    (output / 'ASAMA1_RAPORU.md').write_text(text, encoding='utf-8')


def run(main_path: Path, compare_path: Path, output: Path, as_of: str) -> dict:
    main_path, compare_path, output = main_path.resolve(), compare_path.resolve(), output.resolve()
    if main_path == compare_path:
        raise ValueError('Ana dosya ve karşılaştırma dosyası farklı olmalıdır.')
    for source in [main_path, compare_path]:
        if not source.is_file():
            raise FileNotFoundError(f'Kaynak CSV bulunamadı: {source}')
        if source.is_relative_to(output):
            raise ValueError('Çıktı klasörü kaynak dosyaları içermemelidir.')
    before = {str(p): sha256(p) for p in [main_path, compare_path]}
    raw, comparison = read_source(main_path), read_source(compare_path)
    subset = comparison.loc[comparison.HalTurId.eq('2')]
    # Değerler ve her satırın tekrar sayısı aynı olmalı; sıralamaya bağımlı değil.
    raw_counts = raw.value_counts(COLUMNS).sort_index()
    subset_counts = subset.value_counts(COLUMNS).sort_index()
    if not raw_counts.equals(subset_counts):
        raise ValueError('İki kaynakta meyve-sebze kayıtları/çoklukları eşleşmiyor; çıktı üretilmedi.')
    cleaned, rejected, stats = clean_data(raw, as_of)
    if cleaned.empty:
        raise ValueError('Temizleme sonrasında geçerli veri kalmadı.')
    assert len(cleaned) + len(rejected) == len(raw)
    assert not cleaned.duplicated(KEY).any()
    assert set(cleaned.KaynakKayitNo).isdisjoint(rejected.KaynakKayitNo)
    profile = raw_profile(raw)
    output.mkdir(parents=True, exist_ok=True)
    atomic_csv(cleaned, output/'temiz_veri.csv')
    atomic_csv(rejected, output/'dislanan_kayitlar.csv')
    atomic_csv(cleaned.loc[cleaned.IncelemeGerekli], output/'inceleme_bekleyen.csv')
    daily = raw.groupby('Tarih').agg(HamKayit=('TabelaGId','size'), HamUrunKimligi=('TabelaGId','nunique'))
    daily['TemizKayit'] = cleaned.groupby('Tarih').size().reindex(daily.index, fill_value=0)
    daily = daily.sort_index().reset_index()
    atomic_csv(daily, output/'veri_gunleri.csv')
    yearly = raw.assign(Yil=raw.Tarih.str[:4]).groupby('Yil').agg(HamKayit=('Tarih','size'), VeriGunu=('Tarih','nunique'), HamUrun=('TabelaGId','nunique'))
    yearly['TemizKayit'] = cleaned.groupby(cleaned.Tarih.str[:4]).size().reindex(yearly.index,fill_value=0)
    yearly = yearly.reset_index()
    atomic_csv(yearly, output/'yillik_kapsam.csv')
    identities = raw.groupby(['TabelaGId','UrunAd','BirimTurId','KategoriId','HalTurId']).agg(IlkTarih=('Tarih','min'), SonTarih=('Tarih','max'), HamKayit=('Tarih','size')).reset_index()
    atomic_csv(identities, output/'ad_birim_iliskileri.csv')
    series = cleaned.groupby(['SeriId',*IDENTITY,'UrunAd','AdDonemi','Kategori','BirimAciklamasi']).agg(
        Kayit=('Tarih','size'), IlkTarih=('Tarih','min'), SonTarih=('Tarih','max'),
        IncelemeKayit=('IncelemeGerekli','sum'),
        DusukFiyatMin=('EnDusukFiyat','min'), DusukFiyatMedyan=('EnDusukFiyat','median'), DusukFiyatMax=('EnDusukFiyat','max'),
        YuksekFiyatMin=('EnYuksekFiyat','min'), YuksekFiyatMedyan=('EnYuksekFiyat','median'), YuksekFiyatMax=('EnYuksekFiyat','max')).reset_index()
    # Aralıklar sadece kalite özetidir; öğrenme özelliği değildir.
    time_sorted = cleaned.sort_values(['SeriId','Tarih'])
    gaps = pd.to_datetime(time_sorted.Tarih).groupby(time_sorted.SeriId).diff().dt.days
    max_gap = gaps.groupby(time_sorted.SeriId).max()
    series['EnUzunGozlemArasiGun'] = series.SeriId.map(max_gap).astype('Int64')
    atomic_csv(series, output/'urun_serileri.csv')
    source_last = pd.Timestamp(profile['son_tarih'])
    test_start = source_last - pd.DateOffset(years=1) + pd.Timedelta(days=1)
    test_contract = {'baslangic_dahil':str(test_start.date()), 'bitis_dahil':str(source_last.date()),
                     'ay_sayisi':12,'ayirma_anahtari':'hedef_tarihi (sonraki aşama)',
                     'hiperparametre_seciminde_kullanilamaz':True, 'henuz_fiziksel_ayrim_yapilmadi':True}
    contract = {
        'surum':'1.0','asama':1,'kaynak_kapsami':'Yalnızca ana İBB meyve-sebze CSV',
        'tahmin_hedefi':'Bir sonraki hal işlem gününün minimum ve maksimum fiyatları',
        'gozlem_anahtari':KEY,'seri_anahtari':[*IDENTITY,'AdDonemi'],
        'veri_gunu_takvimi':'veri_gunleri.csv, ham kaynağa göre; resmi işlem takvimi değildir',
        'birimler':'BirimTurId kodları korunur; birim adı doğrulanmadı',
        'kategori_sozlugu':CATEGORY,'inceleme_kurallari':REVIEW_RULES,
        'inceleme_isaretleri_otomatik_silme_degil':True,
        'test_donemi':test_contract,'orijinal_fiyatlar_degistirildi':False,
        'harici_veri_kullanildi':False,'hedef_ve_model_ozelligi_uretildi':False,
        'model_girdisi_yapilmayacak_alanlar':['KaynakKayitNo','UrunAdHam','IncelemeGerekli','IncelemeNedeni','BirimAciklamasi'],
        'sonraki_asama_kullanici_onayi_gerektirir':True,
    }
    all_days = pd.date_range(profile['ilk_tarih'],profile['son_tarih'],freq='D')
    missing_days = all_days.difference(pd.to_datetime(raw.Tarih.unique()))
    lost = raw.loc[~raw.TabelaGId.str.strip().str.lower().isin(cleaned.TabelaGId)].groupby('TabelaGId').agg(
        UrunAd=('UrunAd',lambda x:' / '.join(sorted(set(x)))), HamKayit=('Tarih','size')).reset_index()
    category_rows = [{'Kategori':label,'HamKayit':int(raw.KategoriId.eq(str(code)).sum()),
                      'TemizKayit':int(cleaned.KategoriId.eq(code).sum())} for code,label in CATEGORY.items()]
    summary = {'calistirma_zamani':datetime.now(ZoneInfo('Europe/Istanbul')).isoformat(),
               'ana_veri':profile,'karsilastirma':{'tum_urunler_satir':len(comparison),'meyve_sebze_ayni_coklukta_esit':True},
               'temizleme':stats,'temiz_kapsam':{'urun_kimligi':int(cleaned.TabelaGId.nunique()),
                  'kimlik_birim_hal_kategori':len(cleaned[IDENTITY].drop_duplicates()),'ad_donemli_seri':len(series),
                  'kaydi_kalmayan_urunler':lost.to_dict('records'),'kategori_kapsami':category_rows},
               'takvim':{'kayitsiz_takvim_gunu':len(missing_days),'kayitsiz_haftaici_gun':int((missing_days.dayofweek<5).sum())},
               'test_donemi_sozlesmesi':test_contract}
    json_write(summary, output/'ozet.json')
    json_write(contract, output/'veri_sozlesmesi.json')
    make_report(summary,yearly,daily,output)
    after = {str(p):sha256(p) for p in [main_path,compare_path]}
    if before != after:
        raise RuntimeError('Kaynak dosyası çalışma sırasında değişmiş; çıktılar geçerli kabul edilmemeli.')
    manifest = {'kaynak_once':before,'kaynak_sonra':after,'kaynaklar_degismedi':True,
                'kod_sha256':sha256(Path(__file__)),
                'cikti_sha256':{p.name:sha256(p) for p in sorted(output.iterdir()) if p.is_file() and p.name!='kaynak_manifestosu.json' and not p.name.endswith('.tmp')}}
    json_write(manifest,output/'kaynak_manifestosu.json')
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ana',type=Path,default=Path.home()/'Desktop/IBB_Hal_Meyve_Sebze_2016_2026.csv')
    parser.add_argument('--karsilastirma',type=Path,default=Path.home()/'Desktop/IBB_Hal_Tum_Urunler_2016_2026.csv')
    parser.add_argument('--cikti',type=Path,default=ROOT/'ciktilar/asama1')
    parser.add_argument('--inceleme-tarihi',default='2026-10-27')
    args=parser.parse_args()
    result=run(args.ana,args.karsilastirma,args.cikti,args.inceleme_tarihi)
    print(json.dumps(result['temizleme'],ensure_ascii=False,indent=2))
    print(f'Rapor: {args.cikti.resolve() / "ASAMA1_RAPORU.md"}')


if __name__=='__main__':
    main()
