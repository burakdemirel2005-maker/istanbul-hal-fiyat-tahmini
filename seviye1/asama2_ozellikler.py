"""Aşama 2: yalnızca geçmiş özellikler, ertesi veri günü hedefi ve tarih ayrımı.

Model eğitmez. Özellik üretimi hem geçmiş deneyler hem son gün girdileri için
aynı build_features fonksiyonundan geçer. Sadece Aşama 1 çıktıları okunur.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from asama1_veri import atomic_csv, json_write, sha256, table

ROOT = Path(__file__).resolve().parent
VERSION = '2.0.0'
LAGS = (1, 2, 5, 10, 20)
WINDOWS = (5, 10, 20)
PRICES = {'min': 'EnDusukFiyat', 'max': 'EnYuksekFiyat'}
FEATURES = ['fiyat_min', 'fiyat_max', 'fiyat_orta', 'fiyat_araligi', 'goreli_fiyat_araligi']
FEATURES += [f'{side}_lag_{lag}' for side in PRICES for lag in LAGS]
FEATURES += [f'{side}_rolling_{window}_{stat}' for side in PRICES for window in WINDOWS
             for stat in ('mean', 'std', 'min', 'max')]
FEATURES += [f'{side}_{name}' for side in PRICES
             for name in ('fark_1', 'degisim_orani_1', 'degisim_orani_5')]
FEATURES += ['tarih_yil', 'tarih_ay', 'tarih_hafta_gunu', 'tarih_yil_gunu',
             'tarih_ay_sin', 'tarih_ay_cos', 'tarih_yil_sin', 'tarih_yil_cos',
             'onceki_gozlemden_gun']
TARGETS = ['hedef_min', 'hedef_max']
METADATA = ['KaynakKayitNo', 'Tarih', 'SeriId', 'TabelaGId', 'UrunAd', 'BirimTurId',
            'KategoriId', 'HalTurId', 'AdDonemi', 'BaglamId', 'OzellikEnEskiTarih',
            'OzellikEnYeniTarih', 'IncelemeGerekli', 'GecmisteIncelemeSayisi']
LABEL_METADATA = ['HedefTarih', 'HedefKaynakKayitNo', 'HedefIncelemeGerekli', 'Bolum']


def build_features(cleaned: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """t fiyatları yayımlandıktan sonraki tahmin anına ait, sadece t ve geçmişi."""
    required = set(METADATA[:9] + ['EnDusukFiyat', 'EnYuksekFiyat', 'IncelemeGerekli'])
    if not required.issubset(cleaned.columns):
        raise ValueError(f'Aşama 1 sütunları eksik: {sorted(required-set(cleaned.columns))}')
    if len(calendar) == 0 or calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ValueError('Ham veri günü takvimi sıralı, benzersiz ve boş olmayan bir takvim olmalı.')
    if cleaned.duplicated(['SeriId', 'Tarih']).any():
        raise ValueError('Aynı seri/tarih için birden fazla kayıt var.')
    d = cleaned.sort_values(['SeriId', 'Tarih'], kind='stable').reset_index(drop=True).copy()
    d['Tarih'] = pd.to_datetime(d.Tarih, format='%Y-%m-%d', errors='raise')
    positions = pd.Series(np.arange(len(calendar)), index=calendar)
    day_index = d.Tarih.map(positions)
    if day_index.isna().any():
        raise ValueError('Temiz kaydın tarihi ham veri günü takviminde bulunamadı.')
    if not np.isfinite(d[list(PRICES.values())].to_numpy(dtype=float)).all():
        raise ValueError('Fiyatlarda sonlu olmayan değer var; Aşama 1 yeniden denetlenmeli.')
    if not (d.EnDusukFiyat.gt(0) & d.EnYuksekFiyat.ge(d.EnDusukFiyat)).all():
        raise ValueError('Geçersiz fiyat aralığı var; Aşama 1 yeniden denetlenmeli.')
    if not pd.api.types.is_bool_dtype(d.IncelemeGerekli):
        raise ValueError('IncelemeGerekli alanı boolean olmalı; metin boola çevrilmemeli.')
    if not d.groupby('SeriId')[['TabelaGId', 'BirimTurId', 'KategoriId', 'HalTurId', 'AdDonemi', 'UrunAd']].nunique().eq(1).all().all():
        raise ValueError('Bir seri içinde ürün, birim, kategori veya ad dönemi değişiyor.')
    # Arada global veri günü eksilirse pencere sıfırlanır; hafta sonu kendi başına kesinti değildir.
    gap = day_index.groupby(d.SeriId, sort=False).diff().ne(1)
    segment = gap.groupby(d.SeriId, sort=False).cumsum().astype('int64')
    d['BaglamId'] = d.SeriId + '|p' + segment.astype(str)
    grouped = d.groupby('BaglamId', sort=False)
    d['fiyat_min'], d['fiyat_max'] = d.EnDusukFiyat, d.EnYuksekFiyat
    d['fiyat_orta'] = (d.EnDusukFiyat + d.EnYuksekFiyat) / 2
    d['fiyat_araligi'] = d.EnYuksekFiyat - d.EnDusukFiyat
    d['goreli_fiyat_araligi'] = d.fiyat_araligi / d.fiyat_orta
    for side, source in PRICES.items():
        for lag in LAGS:
            d[f'{side}_lag_{lag}'] = grouped[source].shift(lag)
        for window in WINDOWS:
            rolling = grouped[source].rolling(window=window, min_periods=window)
            for stat in ('mean', 'std', 'min', 'max'):
                values = rolling.std(ddof=0) if stat == 'std' else getattr(rolling, stat)()
                d[f'{side}_rolling_{window}_{stat}'] = values.reset_index(level=0, drop=True).reindex(d.index)
        d[f'{side}_fark_1'] = d[source] - d[f'{side}_lag_1']
        d[f'{side}_degisim_orani_1'] = d[source] / d[f'{side}_lag_1'] - 1
        d[f'{side}_degisim_orani_5'] = d[source] / d[f'{side}_lag_5'] - 1
    d['tarih_yil'], d['tarih_ay'] = d.Tarih.dt.year, d.Tarih.dt.month
    d['tarih_hafta_gunu'], d['tarih_yil_gunu'] = d.Tarih.dt.dayofweek, d.Tarih.dt.dayofyear
    d['tarih_ay_sin'] = np.sin(2 * np.pi * (d.tarih_ay - 1) / 12)
    d['tarih_ay_cos'] = np.cos(2 * np.pi * (d.tarih_ay - 1) / 12)
    d['tarih_yil_sin'] = np.sin(2 * np.pi * (d.tarih_yil_gunu - 1) / 365.25)
    d['tarih_yil_cos'] = np.cos(2 * np.pi * (d.tarih_yil_gunu - 1) / 365.25)
    d['onceki_gozlemden_gun'] = d.Tarih.groupby(d.SeriId, sort=False).diff().dt.days
    d['OzellikEnEskiTarih'] = grouped.Tarih.shift(max(LAGS))
    d['OzellikEnYeniTarih'] = d.Tarih
    d['GecmisteIncelemeSayisi'] = (grouped.IncelemeGerekli.rolling(21, min_periods=1).sum()
                                  .reset_index(level=0, drop=True).reindex(d.index).astype('int64'))
    d['EksikOzellikSayisi'] = (~np.isfinite(d[FEATURES].to_numpy(dtype=float))).sum(axis=1)
    d['OzellikHazir'] = d.EksikOzellikSayisi.eq(0)
    return d.sort_values(['Tarih', 'SeriId'], kind='stable').reset_index(drop=True)


def attach_targets(features: pd.DataFrame, cleaned: pd.DataFrame,
                   calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Sonraki ortak günde AYNI seri varsa etiketle; uzaktaki kayda atlama."""
    d = features.copy()
    next_day = pd.Series(calendar[1:], index=calendar[:-1])
    d['HedefTarih'] = d.Tarih.map(next_day)
    labels = cleaned[['SeriId', 'Tarih', 'EnDusukFiyat', 'EnYuksekFiyat',
                      'KaynakKayitNo', 'IncelemeGerekli']].copy()
    labels['Tarih'] = pd.to_datetime(labels.Tarih)
    labels = labels.rename(columns={'Tarih':'HedefTarih', 'EnDusukFiyat':'hedef_min',
                                    'EnYuksekFiyat':'hedef_max', 'KaynakKayitNo':'HedefKaynakKayitNo',
                                    'IncelemeGerekli':'HedefIncelemeGerekli'})
    d = d.merge(labels, on=['SeriId', 'HedefTarih'], how='left', validate='many_to_one')
    d['DislamaNedeni'] = np.select(
        [d.HedefTarih.isna(), d.HedefKaynakKayitNo.isna(), ~d.OzellikHazir],
        ['gelecek_veri_gunu_henuz_bilinmiyor', 'sonraki_veri_gununde_ayni_seri_yok',
         'yetersiz_kesintisiz_gecmis'], default='')
    return d


def split_by_target(data: pd.DataFrame, test_start: str, test_end: str) -> tuple[dict, dict]:
    start, end = pd.Timestamp(test_start), pd.Timestamp(test_end)
    validation_start = start - pd.DateOffset(years=1)
    accepted = data.loc[data.DislamaNedeni.eq('')].copy()
    if accepted.HedefTarih.gt(end).any():
        raise ValueError('Test sözleşmesi sonrasına taşan hedef var.')
    accepted['Bolum'] = np.select(
        [accepted.HedefTarih.lt(validation_start), accepted.HedefTarih.lt(start)],
        ['egitim', 'dogrulama'], default='test')
    accepted['HedefKaynakKayitNo'] = accepted.HedefKaynakKayitNo.astype('int64')
    accepted['HedefIncelemeGerekli'] = accepted.HedefIncelemeGerekli.astype(bool)
    parts = {name:accepted.loc[accepted.Bolum.eq(name), METADATA+LABEL_METADATA+FEATURES+TARGETS]
             .sort_values(['HedefTarih','SeriId'],kind='stable').reset_index(drop=True)
             for name in ['egitim','dogrulama','test']}
    bounds = {'egitim_son_dahil':str((validation_start-pd.Timedelta(days=1)).date()),
              'dogrulama_ilk_dahil':str(validation_start.date()),
              'dogrulama_son_dahil':str((start-pd.Timedelta(days=1)).date()),
              'test_ilk_dahil':str(start.date()), 'test_son_dahil':str(end.date())}
    return parts, bounds


def save_frame(frame: pd.DataFrame, path: Path) -> None:
    export = frame.copy()
    for column in export.select_dtypes(include=['datetime64[ns]']).columns:
        export[column] = export[column].dt.strftime('%Y-%m-%d')
    if path.name.endswith('.gz'):
        temp=path.with_name(path.name+'.tmp')
        export.to_csv(temp,index=False,encoding='utf-8-sig',lineterminator='\n',
                      compression={'method':'gzip','compresslevel':6,'mtime':0})
        temp.replace(path)
    else:
        atomic_csv(export,path)


def write_report(summary: dict, coverage: pd.DataFrame, output: Path) -> None:
    n=lambda x:f'{x:,}'.replace(',','.')
    b=summary['sinirlar']
    rows=pd.DataFrame([{'Bölüm':name,'Örnek':n(s['kayit']), 'Seri':s['seri'],
                       'İlk hedef':s['ilk_hedef'],'Son hedef':s['son_hedef']}
                      for name,s in summary['bolumler'].items()])
    report=f'''# Aşama 2 özellik mühendisliği ve kronolojik veri ayrımı

Çalıştırma zamanı: {summary['calistirma_zamani']}

Aşama 1'deki {n(summary['temiz_gozlem'])} gözlemden **{n(summary['kullanilabilir_ornek'])} etiketli örnek** üretildi. {len(FEATURES)} sayısal özellik var. Hiçbir model eğitilmedi, performans ölçülmedi veya hiperparametre seçilmedi. Yalnızca İBB fiyatları ve gözlem tarihleri kullanıldı.

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

{table(pd.DataFrame([{'Neden':k,'Kayıt':n(v)} for k,v in summary['dislama_nedenleri'].items()]))}

{n(summary['kullanilabilir_ornek'])} kabul edilen + {n(summary['dislanan_ornek'])} dışlanan = {n(summary['temiz_gozlem'])} temiz gözlem. Dışlama listesi kayıt ve neden düzeyinde saklandı. İnceleme işaretleri dışlama nedeni değildir. Öncelik: bilinmeyen sonraki gün, eksik hedef, yetersiz geçmiş. Aynı kayıtta ikinci bir neden olsa da mutabakat için bir ana neden yazılır.

## Eğitim doğrulama ve test

| Bölüm | Takvim sınırı |
|---|---|
| Eğitim | Hedef tarihi {b['egitim_son_dahil']} ve öncesi |
| Doğrulama | {b['dogrulama_ilk_dahil']} – {b['dogrulama_son_dahil']} |
| Test | {b['test_ilk_dahil']} – {b['test_son_dahil']} |

Gerçekte kayıt bulunan hedef tarihleri:

{table(rows)}

Ayrım özellik satırının tarihine değil **hedef tarihine** göre yapıldı; hiçbir hedef tarihi iki bölümde yer almıyor. Bölüm sınırını aşan fakat geçmişi geçerli olan özellik pencereleri korunur. Bu geçmişin kullanılması sızıntı değildir. Eğitim/yeniden eğitim anında yalnızca o anda etiketi bilinen örnekler kullanılmalıdır.

Son 12 ayın sınırı Aşama 1 sözleşmesinden alındı ve değiştirilmedi. Nihai test hiperparametre/özellik/ürün seçimi için kullanılmayacak. Bu aşamada testte yalnızca kapsam ve veri sözleşmesi kontrol edildi. Önceki incelemede bu dönemin başlangıç hatalarına bakılmış olduğu bilgisi korunur; tamamen hiç görülmemiş test iddiası yoktur.

## Kalite işaretleri ve ürün kapsamı

İnceleme işaretli mevcut gözlem veya geçmiş pencere bilgisi yalnızca denetim metadatasında; HedefIncelemeGerekli etiket kalitesi metadatasında tutulur. Bunlar X matrisine girmez. Şüpheli hedefler test hatası yükseltebilir diye silinmedi. Ürünler test başarısına veya son dönemde bulunma şartına göre seçilmedi; bütün uygun seriler veri setlerinde tutuldu.

Eğitimde hiç görülmeyen seri sayısı: doğrulamada {summary['egitimde_gorulmeyen']['dogrulama']}, testte {summary['egitimde_gorulmeyen']['test']}. Bunlar da korunur; ileride ürün başına modeli olmayan seriye sahte tahmin üretilemez. urun_kapsami.csv her serinin bölüm kayıt sayılarını verir; bunlar başarı ölçütü veya ürün seçimi değildir.

## Son gün için hazır girdiler

Son ortak veri tarihi {summary['son_veri_tarihi']}. Bu tarihteki {summary['son_gun']['gozlem']} gözlemden {summary['son_gun']['ozellik_hazir']} tanesinin bütün özellikleri hazır. Bunlar son_gun_girdileri.csv içinde aynı özellik fonksiyonuyla üretildi; gelecekteki fiyat kullanılmadı ve tahmin üretilmedi. Son gözlemi eski olan veya geçmişi yetersiz seri bilgisi son_gozlem_durumu.csv içinde tutulur.

Bu girdiler, sonradan eğitilecek gerçek modeller içindir. Fiyat tahmini ve Streamlit uygulaması henüz yoktur. Tahmin için ertesi tarihin kesinmiş gibi atanması bu aşamanın çıktısı değildir.

## Dosyalar ve doğrulama

egitim.csv.gz, dogrulama.csv.gz ve test.csv.gz sıkıştırılmış CSV dosyalarıdır; Pandas read_csv ile doğrudan okunabilir. Özellik listesi ve hedef listesi açık sözleşmede saklıdır; bütün sayısal sütunları körlemesine modele vermek hatalıdır. Çıktılar orijinal fiyatları değiştirmez. Aşama 1 dosyaları ve orijinal CSV hashleri işlem öncesi/sonrası karşılaştırılır. Ayrı doğrulayıcı kaydedilmiş örnekleri, gerçek hedefleri, geçmiş sınırlarını ve bölümlemeyi kaynakla denetler. Yapılan testler son kullanıcıya verilen çalışma sonucunda ayrıca raporlanır.

Sonraki aşama Naive Baseline'dır. Kullanıcı onayı olmadan çalıştırılmaz.
'''
    (output/'ASAMA2_RAPORU.md').write_text(report,encoding='utf-8')


def run(source: Path, output: Path) -> dict:
    source, output=source.resolve(),output.resolve()
    if source==output or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError('Aşama 1 ve Aşama 2 çıktı klasörleri ayrı olmalıdır.')
    names=['temiz_veri.csv','veri_gunleri.csv','veri_sozlesmesi.json','kaynak_manifestosu.json']
    inputs={str(source/name):sha256(source/name) for name in names}
    stage1=json.loads((source/'kaynak_manifestosu.json').read_text())
    for name in ['temiz_veri.csv','veri_gunleri.csv','veri_sozlesmesi.json']:
        if inputs[str(source/name)]!=stage1['cikti_sha256'][name]:
            raise ValueError(f'Aşama 1 manifestosu uyuşmuyor: {name}')
    originals={p:sha256(Path(p)) for p in stage1['kaynak_sonra']}
    if originals!=stage1['kaynak_sonra']:
        raise ValueError('Orijinal CSV hashleri Aşama 1 ile uyuşmuyor.')
    contract1=json.loads((source/'veri_sozlesmesi.json').read_text())
    clean=pd.read_csv(source/'temiz_veri.csv',keep_default_na=False,float_precision='round_trip')
    calendar=pd.DatetimeIndex(pd.to_datetime(pd.read_csv(source/'veri_gunleri.csv').Tarih))
    features=build_features(clean,calendar)
    labelled=attach_targets(features,clean,calendar)
    c=contract1['test_donemi']
    parts,bounds=split_by_target(labelled,c['baslangic_dahil'],c['bitis_dahil'])
    output.mkdir(parents=True,exist_ok=True)
    for name,frame in parts.items():
        save_frame(frame,output/f'{name}.csv.gz')
    excluded=labelled.loc[labelled.DislamaNedeni.ne(''),
                          ['KaynakKayitNo','Tarih','SeriId','HedefTarih','EksikOzellikSayisi','DislamaNedeni']]
    save_frame(excluded,output/'dislanan_ornekler.csv')
    latest=features.loc[features.Tarih.eq(calendar.max())]
    save_frame(latest.loc[latest.OzellikHazir,METADATA+FEATURES],output/'son_gun_girdileri.csv')
    status=features.sort_values('Tarih').groupby('SeriId',sort=False).tail(1)[
        ['SeriId','TabelaGId','UrunAd','BirimTurId','Tarih','OzellikHazir','EksikOzellikSayisi']].copy()
    status['SonVeriGunundeKaydiVar']=status.Tarih.eq(calendar.max())
    save_frame(status,output/'son_gozlem_durumu.csv')
    coverage=clean.groupby(['SeriId','TabelaGId','UrunAd','BirimTurId']).size().rename('TemizGozlem').reset_index()
    for name,frame in parts.items():
        coverage[name+'_ornek']=coverage.SeriId.map(frame.groupby('SeriId').size()).fillna(0).astype('int64')
    coverage['EgitimdeGoruldu']=coverage.egitim_ornek.gt(0)
    save_frame(coverage,output/'urun_kapsami.csv')
    summary={'calistirma_zamani':datetime.now(ZoneInfo('Europe/Istanbul')).isoformat(),
             'temiz_gozlem':len(clean),'ozellik_sayisi':len(FEATURES),
             'kullanilabilir_ornek':sum(len(x) for x in parts.values()),'dislanan_ornek':len(excluded),
             'dislama_nedenleri':excluded.DislamaNedeni.value_counts().to_dict(), 'sinirlar':bounds,
             'son_veri_tarihi':str(calendar.max().date()),
             'son_gun':{'gozlem':len(latest),'ozellik_hazir':int(latest.OzellikHazir.sum())},
             'bolumler':{},'egitimde_gorulmeyen':{}}
    for name,frame in parts.items():
        summary['bolumler'][name]={'kayit':len(frame),'seri':int(frame.SeriId.nunique()),
            'ilk_hedef':str(frame.HedefTarih.min().date()) if len(frame) else None,
            'son_hedef':str(frame.HedefTarih.max().date()) if len(frame) else None,
            'isaretli_mevcut':int(frame.IncelemeGerekli.sum()),
            'isaretli_hedef':int(frame.HedefIncelemeGerekli.sum()),
            'isaretli_gecmis_penceresi':int(frame.GecmisteIncelemeSayisi.gt(0).sum())}
    for name in ['dogrulama','test']:
        summary['egitimde_gorulmeyen'][name]=len(set(parts[name].SeriId)-set(parts['egitim'].SeriId))
    contract={'surum':VERSION,'asama':2,'ozellikler':FEATURES,'hedefler':TARGETS,
              'metadata':METADATA+LABEL_METADATA,'laglar':list(LAGS),'rolling_pencereleri':list(WINDOWS),
              'rolling_t_gununu_icerir':True,'std_ddof':0,'azami_gecmis_lag':20,
              'tahmin_ani':'t günü fiyatları yayımlandıktan sonra',
              'kesinti_politikasi':'Eksik ortak veri gününde aynı seri bağlamı sıfırlanır.',
              'hedef_politikasi':'Bir sonraki ham ortak veri gününde aynı SeriId kaydı zorunlu.',
              'bolumleme_anahtari':'HedefTarih','sinirlar':bounds,
              'harici_veri':False,'model_egitildi':False,'olcekleme_fit_edildi':False,
              'inceleme_isaretli_satirlar_korundu':True,'urun_secimi_yapildi':False,
              'test_hiperparametre_secimine_kapali':True,'sonraki_asama_onay_gerektirir':True}
    json_write(summary,output/'ozet.json')
    json_write(contract,output/'ozellik_sozlesmesi.json')
    write_report(summary,coverage,output)
    after={p:sha256(Path(p)) for p in inputs}
    original_after={p:sha256(Path(p)) for p in originals}
    if after!=inputs or original_after!=originals:
        raise RuntimeError('Girdi çalışma sırasında değişmiş; çıktıları kullanmayın.')
    manifest={'asama1_dizini':str(source),'girdi_once':inputs,'girdi_sonra':after,
              'orijinal_csv_once':originals,'orijinal_csv_sonra':original_after,
              'kod_sha256':sha256(Path(__file__)),
              'cikti_sha256':{p.name:sha256(p) for p in sorted(output.iterdir())
                if p.is_file() and p.name!='kaynak_manifestosu.json' and not p.name.endswith('.tmp')}}
    json_write(manifest,output/'kaynak_manifestosu.json')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asama1',type=Path,default=ROOT/'ciktilar/asama1')
    parser.add_argument('--cikti',type=Path,default=ROOT/'ciktilar/asama2')
    args=parser.parse_args()
    print(json.dumps(run(args.asama1,args.cikti),ensure_ascii=False,indent=2))
