"""Aşama 3: yalnızca Naive Baseline referans tahminleri ve çok düzeyli değerlendirme.

Model eğitmez, parametre öğrenmez, hiperparametre aramaz ve sahte model dosyası üretmez.
Tahmin kuralı:
  y_hat_min(t+1) = fiyat_min(t)
  y_hat_max(t+1) = fiyat_max(t)
Hata işareti kuralı:
  hata = tahmin - gercek
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import platform
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import sklearn

from asama1_veri import CATEGORY, atomic_csv, json_write, sha256, table
from asama2_ozellikler import save_frame

ROOT = Path(__file__).resolve().parent
VERSION = '3.0.0'
METHOD_NAME = 'Naive Baseline (Son Gözlenen Fiyatı Taşıma / Persistence)'
ERROR_DEFINITION = 'hata = tahmin - gercek'
TARGET_SPECS = (
    ('hedef_min', 'fiyat_min', 'tahmin_min', 'hata_min', 'mutlak_hata_min', 'karesel_hata_min'),
    ('hedef_max', 'fiyat_max', 'tahmin_max', 'hata_max', 'mutlak_hata_max', 'karesel_hata_max'),
)
REQUIRED_INPUT_METADATA = [
    'KaynakKayitNo', 'Tarih', 'SeriId', 'TabelaGId', 'UrunAd', 'BirimTurId',
    'KategoriId', 'HalTurId', 'AdDonemi', 'BaglamId', 'OzellikEnEskiTarih',
    'OzellikEnYeniTarih', 'IncelemeGerekli', 'GecmisteIncelemeSayisi',
    'HedefTarih', 'HedefKaynakKayitNo', 'HedefIncelemeGerekli', 'Bolum',
]
PREDICTION_COLUMNS = [
    'Bolum', 'KaynakKayitNo', 'Tarih', 'HedefTarih', 'HedefKaynakKayitNo',
    'SeriId', 'TabelaGId', 'UrunAd', 'KategoriId', 'Kategori',
    'BirimTurId', 'BirimAciklamasi', 'HalTurId', 'AdDonemi', 'BaglamId',
    'fiyat_min', 'fiyat_max', 'hedef_min', 'hedef_max',
    'tahmin_min', 'tahmin_max',
    'hata_min', 'hata_max',
    'mutlak_hata_min', 'mutlak_hata_max',
    'karesel_hata_min', 'karesel_hata_max',
    'TahminAraligiGecerli',
    'IncelemeGerekli', 'HedefIncelemeGerekli', 'GecmisteIncelemeSayisi',
    'EgitimdeGoruldu', 'EgitimOrnekSayisi',
]
SERIES_METRIC_COLUMNS = [
    'SeriId', 'TabelaGId', 'UrunAd', 'BirimTurId', 'BirimAciklamasi',
    'KategoriId', 'Kategori', 'HalTurId', 'AdDonemi',
    'Bolum', 'EgitimdeGoruldu', 'EgitimOrnekSayisi',
    'HedefTuru', 'OrnekSayisi',
    'IlkGozlemTarihi', 'SonGozlemTarihi', 'IlkHedefTarihi', 'SonHedefTarihi',
    'IsaretliMevcutSayisi', 'IsaretliHedefSayisi', 'IsaretliGecmisPencereSayisi',
    'HedefFiyatMin', 'HedefFiyatMedyan', 'HedefFiyatOrtalama', 'HedefFiyatMax',
    'MAE', 'MSE', 'RMSE', 'R2', 'R2Durumu', 'R2Gecerli',
    'SSRes', 'SSTot',
    'OrtalamaHata', 'MedyanMutlakHata', 'MaksMutlakHata',
    'DegismediSayisi', 'DegismediOrani',
]


def unit_label(unit_code: int) -> str:
    """Kaynakta ölçü birimi sözlüğü doğrulanmadığından isim tahmin edilmez."""
    return f'Birim kodu {int(unit_code)} — adı doğrulanmadı'


def validate_contract(contract2: dict) -> None:
    """Aşama 2 özellik ve hedef sözleşmesinin bütünlüğünü kontrol eder."""
    for key in ('ozellikler', 'hedefler', 'metadata', 'sinirlar'):
        if key not in contract2:
            raise ValueError(f'Aşama 2 sözleşmesinde zorunlu alan eksik: {key}')
    features = contract2['ozellikler']
    targets = contract2['hedefler']
    metadata = contract2['metadata']
    if not isinstance(features, list) or len(features) != 54 or len(set(features)) != 54:
        raise ValueError('Aşama 2 sözleşmesinde 54 benzersiz özellik bulunamadı.')
    if not {'fiyat_min', 'fiyat_max'}.issubset(features):
        raise ValueError('Mevcut fiyat özellikleri (fiyat_min, fiyat_max) sözleşmede bulunamadı.')
    if targets != ['hedef_min', 'hedef_max']:
        raise ValueError(f'Beklenen hedefler [hedef_min, hedef_max] değil: {targets}')
    if not set(REQUIRED_INPUT_METADATA).issubset(metadata):
        missing = sorted(set(REQUIRED_INPUT_METADATA) - set(metadata))
        raise ValueError(f'Sözleşme metadatasında eksik alanlar var: {missing}')
    if set(features).intersection(targets) or set(features).intersection(metadata):
        raise ValueError('Özellik listesi hedef veya metadata alanlarıyla kesişiyor.')


def predict_naive(
    frame: pd.DataFrame,
    contract2: dict,
    training_counts: dict[str, int] | None = None,
    expected_split: str | None = None,
) -> pd.DataFrame:
    """t gününün mevcut min/max fiyatını t+1 hedef günü için değiştirmeden taşır.

    Girdi tablosunu yerinde değiştirmez. Hedef değerleri yalnızca hata hesabı için
    okunur; tahmin değerleri (tahmin_min, tahmin_max) yalnızca fiyat_min ve
    fiyat_max sütunlarından üretilir.
    """
    validate_contract(contract2)
    required_cols = set(REQUIRED_INPUT_METADATA + ['fiyat_min', 'fiyat_max', 'hedef_min', 'hedef_max'])
    missing_cols = sorted(required_cols - set(frame.columns))
    if missing_cols:
        raise ValueError(f'Girdi verisinde zorunlu sütunlar eksik: {missing_cols}')
    if frame.empty:
        raise ValueError('Tahmin üretilecek veri çerçevesi boş.')
    if not frame.KaynakKayitNo.is_unique:
        raise ValueError('KaynakKayitNo tekil değil.')
    if frame.duplicated(['SeriId', 'Tarih']).any():
        raise ValueError('Aynı (SeriId, Tarih) çifti birden fazla kez bulunuyor.')
    if frame.duplicated(['SeriId', 'HedefTarih']).any():
        raise ValueError('Aynı (SeriId, HedefTarih) çifti birden fazla kez bulunuyor.')
    if expected_split is not None and not frame.Bolum.eq(expected_split).all():
        raise ValueError(f'Bölüm etiketleri beklenen ({expected_split}) ile uyuşmuyor.')
    for flag_col in ('IncelemeGerekli', 'HedefIncelemeGerekli'):
        if not pd.api.types.is_bool_dtype(frame[flag_col]):
            raise ValueError(f'{flag_col} sütunu boolean tipinde olmalıdır.')

    d = frame.copy(deep=True)
    origin_dates = pd.to_datetime(d.Tarih, format='%Y-%m-%d', errors='raise')
    target_dates = pd.to_datetime(d.HedefTarih, format='%Y-%m-%d', errors='raise')
    if not target_dates.gt(origin_dates).all():
        raise ValueError('HedefTarih, gözlem tarihi (Tarih) sonrasında olmalıdır.')

    price_cols = ['fiyat_min', 'fiyat_max', 'hedef_min', 'hedef_max']
    price_matrix = d[price_cols].to_numpy(dtype=float)
    if not np.isfinite(price_matrix).all():
        raise ValueError('Mevcut fiyatlarda veya hedeflerde NaN/sonsuz değer var.')
    if not (price_matrix > 0).all():
        raise ValueError('Mevcut fiyatlarda veya hedeflerde sıfır/negatif değer var.')
    if not d.fiyat_min.le(d.fiyat_max).all():
        raise ValueError('Girdi verisinde fiyat_min > fiyat_max tutarsızlığı var.')
    if not d.hedef_min.le(d.hedef_max).all():
        raise ValueError('Girdi verisinde hedef_min > hedef_max tutarsızlığı var.')

    if not d.KategoriId.isin(CATEGORY).all():
        raise ValueError('Bilinmeyen KategoriId değeri bulundu.')

    d['Tarih'] = origin_dates.dt.strftime('%Y-%m-%d')
    d['HedefTarih'] = target_dates.dt.strftime('%Y-%m-%d')
    d['Kategori'] = d.KategoriId.map(CATEGORY)
    d['BirimAciklamasi'] = d.BirimTurId.map(unit_label)

    # Naive Baseline kuralı: y_hat(t+1) = y(t). Parametre veya hedef bilgisi kullanılmaz.
    d['tahmin_min'] = d['fiyat_min'].astype(float)
    d['tahmin_max'] = d['fiyat_max'].astype(float)

    # Hata işareti: hata = tahmin - gercek
    d['hata_min'] = d['tahmin_min'] - d['hedef_min']
    d['hata_max'] = d['tahmin_max'] - d['hedef_max']
    d['mutlak_hata_min'] = d['hata_min'].abs()
    d['mutlak_hata_max'] = d['hata_max'].abs()
    d['karesel_hata_min'] = d['hata_min'] ** 2
    d['karesel_hata_max'] = d['hata_max'] ** 2

    d['TahminAraligiGecerli'] = d['tahmin_min'].le(d['tahmin_max'])
    if not d['TahminAraligiGecerli'].all():
        raise ValueError('Tahmin edilen minimum fiyat maksimum fiyattan büyük.')

    if training_counts is None:
        d['EgitimOrnekSayisi'] = np.int64(0)
        d['EgitimdeGoruldu'] = False
    else:
        counts = d.SeriId.map(training_counts).fillna(0).astype('int64')
        d['EgitimOrnekSayisi'] = counts
        d['EgitimdeGoruldu'] = counts.gt(0)

    return d[PREDICTION_COLUMNS].copy()


def compute_regression_metrics(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> dict:
    """MAE, MSE, RMSE ve özel durumları ayrıştırılmış R² metriklerini hesaplar.

    R² için kütüphanenin tanımsız değerleri sessizce 0 veya 1'e çeviren varsayılan
    davranışı kullanılmaz; 'gozlem_yetersiz', 'hedef_sabit' ve 'hesaplanabilir'
    durumları açıkça ayrılır.
    """
    true_arr = np.asarray(y_true, dtype=float)
    pred_arr = np.asarray(y_pred, dtype=float)
    if true_arr.ndim != 1 or pred_arr.ndim != 1 or len(true_arr) != len(pred_arr):
        raise ValueError('y_true ve y_pred eşit uzunlukta 1 boyutlu diziler olmalıdır.')
    n = len(true_arr)
    if n == 0:
        raise ValueError('Boş dizi için metrik hesaplanamaz.')
    if not (np.isfinite(true_arr).all() and np.isfinite(pred_arr).all()):
        raise ValueError('y_true veya y_pred içinde NaN/sonsuz değer var.')

    err = pred_arr - true_arr
    abs_err = np.abs(err)
    sq_err = err ** 2

    mae = float(np.mean(abs_err))
    mse = float(np.mean(sq_err))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum(sq_err))
    mean_true = float(np.mean(true_arr))
    ss_tot = float(np.sum((true_arr - mean_true) ** 2))

    if n < 2:
        r2 = None
        r2_durumu = 'gozlem_yetersiz'
        r2_gecerli = False
    elif ss_tot == 0.0 or np.all(true_arr == true_arr[0]):
        r2 = None
        r2_durumu = 'hedef_sabit'
        r2_gecerli = False
    else:
        r2 = float(1.0 - (ss_res / ss_tot))
        r2_durumu = 'hesaplanabilir'
        r2_gecerli = True

    unchanged = int(np.sum(err == 0.0))
    return {
        'OrnekSayisi': int(n),
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'R2Durumu': r2_durumu,
        'R2Gecerli': r2_gecerli,
        'SSRes': ss_res,
        'SSTot': ss_tot,
        'OrtalamaHata': float(np.mean(err)),
        'MedyanMutlakHata': float(np.median(abs_err)),
        'MaksMutlakHata': float(np.max(abs_err)),
        'DegismediSayisi': unchanged,
        'DegismediOrani': float(unchanged / n),
        'HedefFiyatMin': float(np.min(true_arr)),
        'HedefFiyatMedyan': float(np.median(true_arr)),
        'HedefFiyatOrtalama': mean_true,
        'HedefFiyatMax': float(np.max(true_arr)),
    }


def evaluate_series(predictions: pd.DataFrame) -> pd.DataFrame:
    """Her (Bolum, SeriId, HedefTuru) için ayrı metrik tablosu üretir."""
    if predictions.empty:
        raise ValueError('Seri metrikleri için tahmin tablosu boş.')
    identity_cols = [
        'TabelaGId', 'UrunAd', 'BirimTurId', 'BirimAciklamasi',
        'KategoriId', 'Kategori', 'HalTurId', 'AdDonemi',
        'EgitimdeGoruldu', 'EgitimOrnekSayisi',
    ]
    if not predictions.groupby(['Bolum', 'SeriId'])[identity_cols].nunique().eq(1).all().all():
        raise ValueError('Aynı (Bolum, SeriId) içinde kimlik, birim veya kategori bilgisi değişiyor.')

    rows = []
    grouped = predictions.sort_values(['Bolum', 'SeriId', 'HedefTarih'], kind='stable').groupby(
        ['Bolum', 'SeriId'], sort=False
    )
    for (split_name, seri_id), group in grouped:
        first = group.iloc[0]
        for target_col, _, pred_col, _, _, _ in TARGET_SPECS:
            metrics = compute_regression_metrics(group[target_col], group[pred_col])
            rows.append({
                'SeriId': seri_id,
                'TabelaGId': first.TabelaGId,
                'UrunAd': first.UrunAd,
                'BirimTurId': int(first.BirimTurId),
                'BirimAciklamasi': first.BirimAciklamasi,
                'KategoriId': int(first.KategoriId),
                'Kategori': first.Kategori,
                'HalTurId': int(first.HalTurId),
                'AdDonemi': int(first.AdDonemi),
                'Bolum': split_name,
                'EgitimdeGoruldu': bool(first.EgitimdeGoruldu),
                'EgitimOrnekSayisi': int(first.EgitimOrnekSayisi),
                'HedefTuru': target_col,
                'OrnekSayisi': metrics['OrnekSayisi'],
                'IlkGozlemTarihi': str(group.Tarih.min()),
                'SonGozlemTarihi': str(group.Tarih.max()),
                'IlkHedefTarihi': str(group.HedefTarih.min()),
                'SonHedefTarihi': str(group.HedefTarih.max()),
                'IsaretliMevcutSayisi': int(group.IncelemeGerekli.sum()),
                'IsaretliHedefSayisi': int(group.HedefIncelemeGerekli.sum()),
                'IsaretliGecmisPencereSayisi': int(group.GecmisteIncelemeSayisi.gt(0).sum()),
                'HedefFiyatMin': metrics['HedefFiyatMin'],
                'HedefFiyatMedyan': metrics['HedefFiyatMedyan'],
                'HedefFiyatOrtalama': metrics['HedefFiyatOrtalama'],
                'HedefFiyatMax': metrics['HedefFiyatMax'],
                'MAE': metrics['MAE'],
                'MSE': metrics['MSE'],
                'RMSE': metrics['RMSE'],
                'R2': np.nan if metrics['R2'] is None else metrics['R2'],
                'R2Durumu': metrics['R2Durumu'],
                'R2Gecerli': metrics['R2Gecerli'],
                'SSRes': metrics['SSRes'],
                'SSTot': metrics['SSTot'],
                'OrtalamaHata': metrics['OrtalamaHata'],
                'MedyanMutlakHata': metrics['MedyanMutlakHata'],
                'MaksMutlakHata': metrics['MaksMutlakHata'],
                'DegismediSayisi': metrics['DegismediSayisi'],
                'DegismediOrani': metrics['DegismediOrani'],
            })
    return pd.DataFrame(rows, columns=SERIES_METRIC_COLUMNS)


def summarize_subset(subset: pd.DataFrame, split_name: str, scope_group: str, scope_code: str, scope_desc: str) -> list[dict]:
    """Verilen alt küme için hem satır ağırlıklı hem seri eşit ağırlıklı özet üretir."""
    rows = []
    if subset.empty:
        for target_col, _, _, _, _, _ in TARGET_SPECS:
            rows.append({
                'Bolum': split_name,
                'KapsamGrubu': scope_group,
                'KapsamKodu': scope_code,
                'KapsamAciklamasi': scope_desc,
                'HedefTuru': target_col,
                'OrnekSayisi': 0,
                'SeriSayisi': 0,
                'UrunKimligiSayisi': 0,
                'SatirAgirlikli_MAE': np.nan,
                'SatirAgirlikli_MSE': np.nan,
                'SatirAgirlikli_RMSE': np.nan,
                'Havuzlanmis_R2': np.nan,
                'Havuzlanmis_R2Durumu': 'gozlem_yetersiz',
                'SatirAgirlikli_DegismediOrani': np.nan,
                'SeriEsitAgirlikli_MAE_Ortalama': np.nan,
                'SeriEsitAgirlikli_MAE_Medyan': np.nan,
                'SeriEsitAgirlikli_RMSE_Ortalama': np.nan,
                'SeriEsitAgirlikli_RMSE_Medyan': np.nan,
                'Hesaplanabilir_R2_SeriSayisi': 0,
                'SabitHedef_SeriSayisi': 0,
                'GozlemYetersiz_SeriSayisi': 0,
                'Negatif_R2_SeriSayisi': 0,
                'SeriEsitAgirlikli_R2_Ortalama': np.nan,
                'SeriEsitAgirlikli_R2_Medyan': np.nan,
            })
        return rows

    sub_series = evaluate_series(subset)
    for target_col, _, pred_col, _, _, _ in TARGET_SPECS:
        pooled = compute_regression_metrics(subset[target_col], subset[pred_col])
        sm = sub_series.loc[sub_series.HedefTuru.eq(target_col)]
        valid_r2 = sm.loc[sm.R2Gecerli, 'R2'].to_numpy(dtype=float)
        rows.append({
            'Bolum': split_name,
            'KapsamGrubu': scope_group,
            'KapsamKodu': scope_code,
            'KapsamAciklamasi': scope_desc,
            'HedefTuru': target_col,
            'OrnekSayisi': int(len(subset)),
            'SeriSayisi': int(subset.SeriId.nunique()),
            'UrunKimligiSayisi': int(subset.TabelaGId.nunique()),
            'SatirAgirlikli_MAE': pooled['MAE'],
            'SatirAgirlikli_MSE': pooled['MSE'],
            'SatirAgirlikli_RMSE': pooled['RMSE'],
            'Havuzlanmis_R2': np.nan if pooled['R2'] is None else pooled['R2'],
            'Havuzlanmis_R2Durumu': pooled['R2Durumu'],
            'SatirAgirlikli_DegismediOrani': pooled['DegismediOrani'],
            'SeriEsitAgirlikli_MAE_Ortalama': float(sm.MAE.mean()),
            'SeriEsitAgirlikli_MAE_Medyan': float(sm.MAE.median()),
            'SeriEsitAgirlikli_RMSE_Ortalama': float(sm.RMSE.mean()),
            'SeriEsitAgirlikli_RMSE_Medyan': float(sm.RMSE.median()),
            'Hesaplanabilir_R2_SeriSayisi': int((sm.R2Durumu == 'hesaplanabilir').sum()),
            'SabitHedef_SeriSayisi': int((sm.R2Durumu == 'hedef_sabit').sum()),
            'GozlemYetersiz_SeriSayisi': int((sm.R2Durumu == 'gozlem_yetersiz').sum()),
            'Negatif_R2_SeriSayisi': int((valid_r2 < 0).sum()) if len(valid_r2) else 0,
            'SeriEsitAgirlikli_R2_Ortalama': float(np.mean(valid_r2)) if len(valid_r2) else np.nan,
            'SeriEsitAgirlikli_R2_Medyan': float(np.median(valid_r2)) if len(valid_r2) else np.nan,
        })
    return rows


def build_scope_summary(predictions_by_split: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Ana evren, eğitimde görülme durumu, kalite işaretleri, birim ve kategori özetlerini üretir."""
    rows = []
    for split_name, df in predictions_by_split.items():
        rows.extend(summarize_subset(
            df, split_name, 'degerlendirme_evreni', 'tum_uygun_ornekler',
            'Tüm uygun örnekler (ana değerlendirme evreni; hiçbir seri veya kalite işareti elenmedi)',
        ))
        rows.extend(summarize_subset(
            df.loc[df.EgitimdeGoruldu], split_name, 'egitim_gorulme_durumu', 'egitimde_gorulen_seriler',
            'Eğitim bölümünde en az 1 etiketli örneği bulunan seriler (yerel modellerle ortak aday kapsam)',
        ))
        rows.extend(summarize_subset(
            df.loc[~df.EgitimdeGoruldu], split_name, 'egitim_gorulme_durumu', 'egitimde_gorulmeyen_seriler',
            'Eğitim bölümünde hiç örneği bulunmayan seriler',
        ))
        rows.extend(summarize_subset(
            df.loc[~(df.IncelemeGerekli | df.HedefIncelemeGerekli)],
            split_name, 'kalite_alt_kumesi', 'mevcut_ve_hedef_isaretsiz',
            'Mevcut ve hedef gözleminde inceleme işareti bulunmayan örnekler (betimleyici alt küme)',
        ))
        rows.extend(summarize_subset(
            df.loc[df.IncelemeGerekli | df.HedefIncelemeGerekli],
            split_name, 'kalite_alt_kumesi', 'mevcut_veya_hedef_isaretli',
            'Mevcut veya hedef gözleminde inceleme işareti bulunan örnekler (betimleyici alt küme)',
        ))
        rows.extend(summarize_subset(
            df.loc[df.GecmisteIncelemeSayisi.eq(0)],
            split_name, 'kalite_alt_kumesi', 'gecmis_penceresi_isaretsiz',
            'Son 21 gözlemlik özellik penceresinde inceleme işareti bulunmayan örnekler',
        ))
        rows.extend(summarize_subset(
            df.loc[df.GecmisteIncelemeSayisi.gt(0)],
            split_name, 'kalite_alt_kumesi', 'gecmis_penceresi_isaretli',
            'Son 21 gözlemlik özellik penceresinde en az bir inceleme işareti bulunan örnekler',
        ))
        for unit_code in sorted(df.BirimTurId.unique()):
            rows.extend(summarize_subset(
                df.loc[df.BirimTurId.eq(unit_code)],
                split_name, 'birim_kodu', f'birim_{int(unit_code)}',
                unit_label(int(unit_code)),
            ))
        for cat_id in sorted(df.KategoriId.unique()):
            cat_name = CATEGORY[int(cat_id)]
            rows.extend(summarize_subset(
                df.loc[df.KategoriId.eq(cat_id)],
                split_name, 'kategori', f'kategori_{int(cat_id)}',
                f'Kategori {int(cat_id)} ({cat_name})',
            ))
    return pd.DataFrame(rows)


def clean_json_Record(record: dict) -> dict:
    """JSON yazımı (allow_nan=False) için NaN değerlerini None'a dönüştürür."""
    cleaned = {}
    for k, v in record.items():
        if isinstance(v, (float, np.floating)) and np.isnan(v):
            cleaned[k] = None
        elif isinstance(v, np.bool_):
            cleaned[k] = bool(v)
        elif isinstance(v, np.integer):
            cleaned[k] = int(v)
        elif isinstance(v, np.floating):
            cleaned[k] = float(v)
        else:
            cleaned[k] = v
    return cleaned


def write_report(
    summary: dict,
    series_metrics: pd.DataFrame,
    scope_summary: pd.DataFrame,
    unseen_table: pd.DataFrame,
    output: Path,
) -> None:
    """Akademik açıklamaları, literatür gerekçesini ve kapsam tablolarını içeren raporu yazar."""
    n = lambda x: f'{int(x):,}'.replace(',', '.')
    f4 = lambda x: '-' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{float(x):.4f}'

    main_rows = []
    for row in scope_summary.loc[scope_summary.KapsamKodu.eq('tum_uygun_ornekler')].itertuples(index=False):
        main_rows.append({
            'Bölüm': row.Bolum,
            'Hedef': row.HedefTuru,
            'Örnek': n(row.OrnekSayisi),
            'Seri': row.SeriSayisi,
            'Satır MAE': f4(row.SatirAgirlikli_MAE),
            'Satır MSE': f4(row.SatirAgirlikli_MSE),
            'Satır RMSE': f4(row.SatirAgirlikli_RMSE),
            'Havuz R²': f4(row.Havuzlanmis_R2),
            'Seri Ort. MAE': f4(row.SeriEsitAgirlikli_MAE_Ortalama),
            'Seri Medyan MAE': f4(row.SeriEsitAgirlikli_MAE_Medyan),
            'Seri Medyan R²': f4(row.SeriEsitAgirlikli_R2_Medyan),
            'Hesaplanabilir R² Seri': f'{row.Hesaplanabilir_R2_SeriSayisi}/{row.SeriSayisi}',
            'Sabit Hedef Seri': row.SabitHedef_SeriSayisi,
            'Negatif R² Seri': row.Negatif_R2_SeriSayisi,
            'Değişmedi Oranı': f'%{row.SatirAgirlikli_DegismediOrani * 100:.2f}',
        })
    main_df = pd.DataFrame(main_rows)

    scope_rows = []
    for row in scope_summary.loc[
        scope_summary.KapsamGrubu.isin(['egitim_gorulme_durumu', 'kalite_alt_kumesi'])
    ].itertuples(index=False):
        scope_rows.append({
            'Bölüm': row.Bolum,
            'Kapsam': row.KapsamKodu,
            'Hedef': row.HedefTuru,
            'Örnek': n(row.OrnekSayisi),
            'Seri': row.SeriSayisi,
            'Satır MAE': f4(row.SatirAgirlikli_MAE),
            'Satır RMSE': f4(row.SatirAgirlikli_RMSE),
            'Seri Ort. MAE': f4(row.SeriEsitAgirlikli_MAE_Ortalama),
            'Seri Medyan MAE': f4(row.SeriEsitAgirlikli_MAE_Medyan),
        })
    scope_df = pd.DataFrame(scope_rows)

    unit_cat_rows = []
    for row in scope_summary.loc[
        scope_summary.KapsamGrubu.isin(['birim_kodu', 'kategori'])
    ].itertuples(index=False):
        unit_cat_rows.append({
            'Bölüm': row.Bolum,
            'Grup': row.KapsamAciklamasi,
            'Hedef': row.HedefTuru,
            'Örnek': n(row.OrnekSayisi),
            'Seri': row.SeriSayisi,
            'Satır MAE': f4(row.SatirAgirlikli_MAE),
            'Satır RMSE': f4(row.SatirAgirlikli_RMSE),
            'Seri Ort. MAE': f4(row.SeriEsitAgirlikli_MAE_Ortalama),
            'Seri Medyan MAE': f4(row.SeriEsitAgirlikli_MAE_Medyan),
            'Seri Medyan R²': f4(row.SeriEsitAgirlikli_R2_Medyan),
        })
    unit_cat_df = pd.DataFrame(unit_cat_rows)

    # Doğrulama ve testte en çok örneği olan 10 serinin min ve max hedef tabloları
    def top10_table(split_name: str, target_name: str) -> pd.DataFrame:
        sub = (
            series_metrics.loc[series_metrics.Bolum.eq(split_name) & series_metrics.HedefTuru.eq(target_name)]
            .sort_values(['OrnekSayisi', 'SeriId'], ascending=[False, True])
            .head(10)[['SeriId', 'UrunAd', 'BirimAciklamasi', 'Kategori', 'OrnekSayisi', 'MAE', 'MSE', 'RMSE', 'R2', 'R2Durumu', 'DegismediOrani']]
            .copy()
        )
        for c in ('MAE', 'MSE', 'RMSE', 'R2'):
            sub[c] = sub[c].map(f4)
        sub['DegismediOrani'] = sub['DegismediOrani'].map(lambda x: f'%{x * 100:.2f}')
        return sub

    top_val_min = top10_table('dogrulama', 'hedef_min')
    top_val_max = top10_table('dogrulama', 'hedef_max')
    top_test_min = top10_table('test', 'hedef_min')
    top_test_max = top10_table('test', 'hedef_max')

    unseen_display = unseen_table.copy()
    for col in ('MAE', 'MSE', 'RMSE', 'R2'):
        unseen_display[col] = unseen_display[col].map(f4)
    unseen_display['DegismediOrani'] = unseen_display['DegismediOrani'].map(lambda x: f'%{x * 100:.2f}')

    text = f'''# Aşama 3 — Naive Baseline Değerlendirme Raporu

Çalıştırma zamanı: {summary['calistirma_zamani']}

Bu aşamada yalnızca **Naive Baseline (son gözlenen fiyatı taşıma / persistence)** yöntemi uygulanmıştır. Hiçbir makine öğrenmesi modeli (Linear Regression, Random Forest, XGBoost vb.) eğitilmemiş, parametre veya hiperparametre öğrenilmemiş, Aşama 2 özellikleri veya kronolojik bölüm sınırları değiştirilmemiştir.

## 1. Yöntem Tanımı ve Tahmin Zamanı

Uygulanan deterministik kural:

- $\\hat{{y}}_{{\\min}}(t+1) = \\text{{fiyat\\_min}}(t)$
- $\\hat{{y}}_{{\\max}}(t+1) = \\text{{fiyat\\_max}}(t)$
- Hata işareti tanımı: $\\text{{hata}} = \\text{{tahmin}} - \\text{{gerçek}}$ (pozitif hata: tahmin gerçek hedefin üzerinde / gerçek fiyat düştü; negatif hata: tahmin gerçek hedefin altında / gerçek fiyat yükseldi; sıfır hata: fiyat değişmedi).

**Tahmin anı ve hedef anlamı:** Mevcut deney, $t$ gününün İBB hal fiyatları yayımlandıktan sonra bir sonraki ortak kayıtlı meyve-sebze veri gününün ($t+1$) minimum ve maksimum fiyatını tahmin eder. Bu nedenle $t$ gününün minimum (`fiyat_min`) ve maksimum (`fiyat_max`) fiyatları tahmin anında biliniyor kabul edilir. Bu kurgu, $t$ sabahı henüz $t$ fiyatları ilan edilmeden yapılan tahminden farklıdır. Kaynak veride resmî hal işlem takvimi bulunmadığı için "bir sonraki hal işlem günü", ham meyve-sebze dosyasında kayıt bulunan bir sonraki ortak veri günüyle temsil edilmektedir. Bir serinin tam sonraki ortak veri gününde kaydı yoksa uzaktaki sonraki kayda atlanmaz.

## 2. Akademik Yöntem Gerekçesi ve Kaynak Notları

1. **Hyndman, R. J. ve Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.), OTexts.**
   - **Bağlantı:** `https://otexts.com/fpp3/` (`simple-methods.html`, `accuracy.html`, `prediction-intervals.html`)
   - **İncelenen bölüm:** Bölüm 5 (The forecaster's toolbox: Simple forecasting methods, Evaluating point forecast accuracy, Prediction intervals).
   - **Projeye uygulanabilir ders:** Naive yöntem ($\\hat{{y}}_{{T+h|T}} = y_T$), ekonomik ve fiyat zaman serilerinde zorunlu temel karşılaştırma referansıdır (benchmark). Ölçeğe bağlı hatalar (MAE, RMSE) hedefle aynı birimde (veya MSE için birimin karesinde) olduğundan farklı birimli ya da çok farklı fiyat seviyeli seriler arasında doğrudan tek bir evrensel ortalama olarak yorumlanmamalıdır.
   - **Doğrudan aktarılamayacak yönler:** Kitaptaki düzenli zaman aralıklı `tsibble` örnekleri, düzensiz kayıt boşlukları ve ad/birim dönemleri içeren İBB paneline doğrudan uygulanamaz; kesinti sonrası pencere sıfırlama kuralımız korunmuştur.
   - **Erişim sınırlılığı:** Açık erişimli ders kitabı; erişim kısıtı yok.

2. **Resmî `scikit-learn` Belgeleri (Sürüm {summary['paket_surumleri']['scikit_learn']}).**
   - **Bağlantı:** `https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics`, `https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html`, `https://scikit-learn.org/stable/common_pitfalls.html#data-leakage`
   - **İncelenen bölüm:** Regresyon metrikleri (`r2_score`, `mean_absolute_error`, `mean_squared_error`), gecikmeli zaman serisi özellikleri ve veri sızıntısı önleme ilkeleri.
   - **Projeye uygulanabilir ders:** `sklearn.metrics.r2_score` fonksiyonu varsayılan `force_finite=True` ayarıyla sabit hedefli (`SS_tot == 0`) serilerde tanımsız olan $R^2$ değerini tam isabette `1.0`, hatada `0.0` değerine dönüştürür; $n < 2$ durumunda ise `NaN` döndürür. Bu projede varsayılan sessiz dönüşüm kabul edilmemiş; `gozlem_yetersiz` ($n < 2$), `hedef_sabit` ($\\text{{SS}}_{{\\text{{tot}}}} = 0$) ve `hesaplanabilir` ($n \\ge 2, \\text{{SS}}_{{\\text{{tot}}}} > 0$) durumları açıkça ayrılmıştır.
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

{table(main_df)}

### Dönemler Arası Fiyat Sürekliliği Bulgusu

- **Doğrulama döneminde (27.10.2024–26.10.2025; {n(summary['bolumler']['dogrulama']['ornek_sayisi'])} örnek, {summary['bolumler']['dogrulama']['seri_sayisi']} seri):** Ardışık ortak veri günlerinde minimum fiyatların **%{summary['bolumler']['dogrulama']['hedefler']['hedef_min']['satir_agirlikli']['DegismediOrani'] * 100:.2f}**'si, maksimum fiyatların **%{summary['bolumler']['dogrulama']['hedefler']['hedef_max']['satir_agirlikli']['DegismediOrani'] * 100:.2f}**'si bir önceki veri günüyle aynı kalmıştır. Ayrıca doğrulama döneminde `hedef_min` için **24 seride**, `hedef_max` için **23 seride** hedef fiyat dönem boyunca hiç değişmemiştir (`R2Durumu = hedef_sabit`, $\\text{{SS}}_{{\\text{{tot}}}} = 0$, $\\text{{SS}}_{{\\text{{res}}}} = 0$). `scikit-learn` varsayılanı bu tanımsız durumlara `1.0` atarken, değerlendirmemizde bu seriler `hedef_sabit` olarak ayrılmış ve sayısal $R^2$ ortalamalarına katılmamıştır.
- **Test döneminde (27.10.2025–26.10.2026; {n(summary['bolumler']['test']['ornek_sayisi'])} örnek, {summary['bolumler']['test']['seri_sayisi']} seri):** Fiyatların bir sonraki ortak veri gününde aynı kalma oranı minimum fiyatlarda **%{summary['bolumler']['test']['hedefler']['hedef_min']['satir_agirlikli']['DegismediOrani'] * 100:.2f}**'ye, maksimum fiyatlarda **%{summary['bolumler']['test']['hedefler']['hedef_max']['satir_agirlikli']['DegismediOrani'] * 100:.2f}**'ye düşmüştür. Test döneminde sabit hedefli seri yoktur (`hedef_sabit = 0`); 123 serinin tamamında $R^2$ hesaplanabilir durumdadır. Günlük fiyat değişim sıklığının ve fiyat seviyelerinin son 12 ayda belirgin biçimde artması nedeniyle, test dönemi satır ağırlıklı MAE değeri doğrulama dönemine kıyasla daha yüksektir (`hedef_min` için {f4(summary['bolumler']['dogrulama']['hedefler']['hedef_min']['satir_agirlikli']['MAE'])}'den {f4(summary['bolumler']['test']['hedefler']['hedef_min']['satir_agirlikli']['MAE'])}'ye; `hedef_max` için {f4(summary['bolumler']['dogrulama']['hedefler']['hedef_max']['satir_agirlikli']['MAE'])}'den {f4(summary['bolumler']['test']['hedefler']['hedef_max']['satir_agirlikli']['MAE'])}'ye).
- **Negatif $R^2$ görülen seriler:** Naive Baseline her zaman pozitif $R^2$ üretmez; hedef varyansının düşük olduğu veya ani sıçramaların/geri dönüşlerin yaşandığı serilerde $\\text{{SS}}_{{\\text{{res}}}} > \\text{{SS}}_{{\\text{{tot}}}}$ olduğunda $R^2 < 0$ çıkar. Doğrulamada `hedef_min` için 3 seride (maksimum hedefte 0 seride), testte ise `hedef_min` için 14 seride ve `hedef_max` için 15 seride negatif $R^2$ gözlenmiştir.

## 4. Eğitimde Görülme Durumu ve Kalite İşaretlerine Göre Kapsam Özeti

Ana değerlendirme evreni (`tum_uygun_ornekler`) değiştirilmeden, ilerideki yerel modellerle adil karşılaştırma yapılabilmesi için `EgitimdeGoruldu` kırılımı ve kalite işaretlerine (`IncelemeGerekli`, `HedefIncelemeGerekli`, `GecmisteIncelemeSayisi`) göre betimleyici alt kümeler ayrıca raporlanmıştır:

{table(scope_df)}

### Eğitimde Görülmeyen Seriler

Doğrulamada **2 seri (260 örnek)**, testte **4 seri (376 örnek)** eğitim bölümünde (`egitim.csv.gz`) hiç yer almamaktadır. Naive Baseline parametre eğitimi gerektirmediğinden bu seriler için de doğrudan tahmin üretebilmektedir; ancak Aşama 4'te ürün/seri başına yerel model eğitildiğinde bu seriler için eğitim verisi bulunmayacaktır:

{table(unseen_display)}

Nedenleri:
- `1b84d414-b4b8-4f46-aec3-3fb739842207|b1|h2|k5|a2` (`Trabzon Hurması`, Birim kodu 1): Eğitim döneminde `Hurma` (`AdDonemi=1`) adıyla yer almış, `2024-10-14` tarihinde başlayan `AdDonemi=2` döneminde ilk 20 gözlemlik geçmiş penceresi tamamlandığında ilk hedef tarihi `2024-11-13` (doğrulama dönemi) olmuştur.
- `4deda22d-a3bc-4c91-9317-f1aa86802153|b1|h2|k6|a1` (`Fesleğen`, Birim kodu 1): Kaynak veride ilk kez `2025-01-23` tarihinde gözlenmiştir.
- `4deda22d-a3bc-4c91-9317-f1aa86802153|b5|h2|k6|a1` (`Fesleğen`, Birim kodu 5): `2026-08-18` tarihinde birim kodu 5 olarak ayrı seriye ayrılmıştır.
- `a30c0224-1b2f-44a7-b985-7213c2308de6|b1|h2|k7|a1` (`Ahududu`, Birim kodu 1): İlk kez `2025-10-06` tarihinde gözlenmiş, 20 gözlemlik geçmiş şartı sonrası ilk etiketli hedefi `2025-11-05` (test dönemi) olmuştur.

## 5. Ölçü Birimi Kodu ve Kategori Bazında Bulgular

Kaynak CSV'de ölçü birimi adları (ör. kilogram, adet, bağ, kasa) yer almadığından ve İBB birim sözlüğü servisi doğrulanamadığından, birimler `"Birim kodu X — adı doğrulanmadı"` şeklinde korunmuştur:

{table(unit_cat_df)}

## 6. Örnek Seri Bazlı Sonuçlar (En Yüksek Örnek Sayısına Sahip 10 Seri)

Tüm 152 doğrulama serisi ve 123 test serisinin minimum ve maksimum hedefleri için tam metrikler `seri_metrikleri.csv` dosyasındadır (toplam 550 seri-hedef satırı). Aşağıda örnek sayısı en yüksek 10 seri hem `hedef_min` hem `hedef_max` için ayrı ayrı gösterilmiştir:

### Doğrulama Dönemi (`hedef_min`, ilk 10 seri)

{table(top_val_min)}

### Doğrulama Dönemi (`hedef_max`, ilk 10 seri)

{table(top_val_max)}

### Test Dönemi (`hedef_min`, ilk 10 seri)

{table(top_test_min)}

### Test Dönemi (`hedef_max`, ilk 10 seri)

{table(top_test_max)}

## 7. Tutarlılık ve Bütünlük Kontrolleri

- **Tekillik ve eşleşme:** Doğrulama (`30.361` kayıt) ve test (`21.141` kayıt) tahmin dosyalarındaki bütün `KaynakKayitNo` ve `HedefKaynakKayitNo` değerleri Aşama 2 ve Aşama 1 kayıtlarıyla birebir eşleşmektedir.
- **Tahmin eşitliği:** Bütün kayıtlarda `tahmin_min == fiyat_min` ve `tahmin_max == fiyat_max` eşitliği tamdır.
- **Aralık ve işaret geçerliliği:** `tahmin_min > tahmin_max` olan kayıt sayısı `0`, negatif veya sıfır tahmin sayısı `0`, `NaN`/sonsuz değer sayısı `0`'dır.
- **Kaynak bütünlüğü:** Orijinal CSV dosyaları ile Aşama 1 ve Aşama 2 çıktılarının SHA-256 özetleri işlem öncesi ve sonrasında tamamen aynıdır.

Sonraki aşama (Aşama 4: Linear Regression, Random Forest ve XGBoost modellerinin eğitimi) kullanıcı onayına bağlıdır.
'''
    (output / 'ASAMA3_RAPORU.md').write_text(text, encoding='utf-8')


def run(asama1_dir: Path, asama2_dir: Path, output_dir: Path) -> dict:
    """Aşama 1 ve Aşama 2 girdilerini doğrulayarak Aşama 3 Naive Baseline çıktılarını üretir."""
    asama1_dir = asama1_dir.resolve()
    asama2_dir = asama2_dir.resolve()
    output_dir = output_dir.resolve()

    dirs = [asama1_dir, asama2_dir, output_dir]
    for i, left in enumerate(dirs):
        for right in dirs[i + 1:]:
            if left == right or left.is_relative_to(right) or right.is_relative_to(left):
                raise ValueError('Aşama 1, Aşama 2 ve Aşama 3 klasörleri birbirinden ayrı olmalıdır.')

    stage1_manifest_path = asama1_dir / 'kaynak_manifestosu.json'
    stage2_manifest_path = asama2_dir / 'kaynak_manifestosu.json'
    if not stage1_manifest_path.is_file() or not stage2_manifest_path.is_file():
        raise FileNotFoundError('Aşama 1 veya Aşama 2 kaynak manifestosu bulunamadı.')

    stage1_manifest = json.loads(stage1_manifest_path.read_text(encoding='utf-8'))
    stage2_manifest = json.loads(stage2_manifest_path.read_text(encoding='utf-8'))

    # 1. Orijinal CSV, Aşama 1 ve Aşama 2 dosyalarının işlem öncesi SHA-256 doğrulaması
    original_hashes_before = {}
    for orig_path_str, expected_hash in stage1_manifest['kaynak_sonra'].items():
        orig_path = Path(orig_path_str)
        if not orig_path.is_file():
            raise FileNotFoundError(f'Orijinal kaynak CSV bulunamadı: {orig_path}')
        actual_hash = sha256(orig_path)
        if actual_hash != expected_hash or stage2_manifest['orijinal_csv_sonra'].get(orig_path_str) != expected_hash:
            raise ValueError(f'Orijinal CSV SHA-256 özeti manifesto ile uyuşmuyor: {orig_path}')
        original_hashes_before[orig_path_str] = actual_hash

    stage1_hashes_before = {}
    for name, expected_hash in stage1_manifest['cikti_sha256'].items():
        file_path = asama1_dir / name
        if not file_path.is_file():
            raise FileNotFoundError(f'Aşama 1 dosyası bulunamadı: {file_path}')
        actual_hash = sha256(file_path)
        if actual_hash != expected_hash:
            raise ValueError(f'Aşama 1 dosya özeti bozulmuş: {name}')
        stage1_hashes_before[str(file_path)] = actual_hash
    stage1_hashes_before[str(stage1_manifest_path)] = sha256(stage1_manifest_path)

    stage2_hashes_before = {}
    for name, expected_hash in stage2_manifest['cikti_sha256'].items():
        file_path = asama2_dir / name
        if not file_path.is_file():
            raise FileNotFoundError(f'Aşama 2 dosyası bulunamadı: {file_path}')
        actual_hash = sha256(file_path)
        if actual_hash != expected_hash:
            raise ValueError(f'Aşama 2 dosya özeti bozulmuş: {name}')
        stage2_hashes_before[str(file_path)] = actual_hash
    stage2_hashes_before[str(stage2_manifest_path)] = sha256(stage2_manifest_path)

    contract1 = json.loads((asama1_dir / 'veri_sozlesmesi.json').read_text(encoding='utf-8'))
    contract2 = json.loads((asama2_dir / 'ozellik_sozlesmesi.json').read_text(encoding='utf-8'))
    summary2 = json.loads((asama2_dir / 'ozet.json').read_text(encoding='utf-8'))
    validate_contract(contract2)

    if contract1['test_donemi']['baslangic_dahil'] != contract2['sinirlar']['test_ilk_dahil']:
        raise ValueError('Aşama 1 ve Aşama 2 test başlangıç sınırları uyuşmuyor.')
    if contract1['test_donemi']['bitis_dahil'] != contract2['sinirlar']['test_son_dahil']:
        raise ValueError('Aşama 1 ve Aşama 2 test bitiş sınırları uyuşmuyor.')

    # 2. Aşama 2 veri kümelerinin okunması
    parts = {
        name: pd.read_csv(asama2_dir / f'{name}.csv.gz', keep_default_na=False, float_precision='round_trip')
        for name in ('egitim', 'dogrulama', 'test')
    }
    for name, df in parts.items():
        if len(df) != summary2['bolumler'][name]['kayit']:
            raise ValueError(f'{name} kayıt sayısı Aşama 2 özetiyle uyuşmuyor.')
        if df.SeriId.nunique() != summary2['bolumler'][name]['seri']:
            raise ValueError(f'{name} seri sayısı Aşama 2 özetiyle uyuşmuyor.')

    bounds = contract2['sinirlar']
    if not pd.to_datetime(parts['dogrulama'].HedefTarih).between(
        bounds['dogrulama_ilk_dahil'], bounds['dogrulama_son_dahil']
    ).all():
        raise ValueError('Doğrulama hedef tarihleri sözleşme sınırları dışında.')
    if not pd.to_datetime(parts['test'].HedefTarih).between(
        bounds['test_ilk_dahil'], bounds['test_son_dahil']
    ).all():
        raise ValueError('Test hedef tarihleri sözleşme sınırları dışında.')

    training_counts = parts['egitim'].groupby('SeriId').size().to_dict()

    # 3. Doğrulama ve test için Naive Baseline tahminlerinin üretilmesi
    predictions_by_split = {
        split_name: predict_naive(
            parts[split_name],
            contract2=contract2,
            training_counts=training_counts,
            expected_split=split_name,
        )
        for split_name in ('dogrulama', 'test')
    }

    all_predictions = pd.concat(list(predictions_by_split.values()), ignore_index=True)
    series_metrics = evaluate_series(all_predictions)
    scope_summary = build_scope_summary(predictions_by_split)

    unseen_series_metrics = series_metrics.loc[~series_metrics.EgitimdeGoruldu, [
        'Bolum', 'SeriId', 'UrunAd', 'BirimTurId', 'BirimAciklamasi', 'Kategori',
        'HedefTuru', 'OrnekSayisi', 'IlkHedefTarihi', 'SonHedefTarihi',
        'MAE', 'MSE', 'RMSE', 'R2', 'R2Durumu', 'DegismediOrani',
    ]].reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_frame(predictions_by_split['dogrulama'], output_dir / 'dogrulama_tahminleri.csv.gz')
    save_frame(predictions_by_split['test'], output_dir / 'test_tahminleri.csv.gz')
    save_frame(series_metrics, output_dir / 'seri_metrikleri.csv')
    save_frame(scope_summary, output_dir / 'kapsam_ozeti.csv')

    # 4. Makine tarafından okunabilir özet ve sözleşme
    split_summaries = {}
    for split_name, pred_df in predictions_by_split.items():
        sm_split = series_metrics.loc[series_metrics.Bolum.eq(split_name)]
        target_dict = {}
        for target_col, feat_col, pred_col, err_col, abs_col, sq_col in TARGET_SPECS:
            pooled = compute_regression_metrics(pred_df[target_col], pred_df[pred_col])
            sm_target = sm_split.loc[sm_split.HedefTuru.eq(target_col)]
            valid_r2 = sm_target.loc[sm_target.R2Gecerli, 'R2'].to_numpy(dtype=float)

            seen_df = pred_df.loc[pred_df.EgitimdeGoruldu]
            seen_pooled = compute_regression_metrics(seen_df[target_col], seen_df[pred_col])
            unseen_df = pred_df.loc[~pred_df.EgitimdeGoruldu]
            unseen_pooled = (
                compute_regression_metrics(unseen_df[target_col], unseen_df[pred_col])
                if len(unseen_df) else None
            )

            target_dict[target_col] = {
                'kaynak_ozellik': feat_col,
                'tahmin_sutunu': pred_col,
                'hata_sutunu': err_col,
                'mutlak_hata_sutunu': abs_col,
                'karesel_hata_sutunu': sq_col,
                'satir_agirlikli': clean_json_Record(pooled),
                'seri_esit_agirlikli': {
                    'seri_sayisi': int(len(sm_target)),
                    'MAE_ortalama': float(sm_target.MAE.mean()),
                    'MAE_medyan': float(sm_target.MAE.median()),
                    'MSE_ortalama': float(sm_target.MSE.mean()),
                    'MSE_medyan': float(sm_target.MSE.median()),
                    'RMSE_ortalama': float(sm_target.RMSE.mean()),
                    'RMSE_medyan': float(sm_target.RMSE.median()),
                    'R2_hesaplanabilir_seri': int((sm_target.R2Durumu == 'hesaplanabilir').sum()),
                    'R2_hedef_sabit_seri': int((sm_target.R2Durumu == 'hedef_sabit').sum()),
                    'R2_gozlem_yetersiz_seri': int((sm_target.R2Durumu == 'gozlem_yetersiz').sum()),
                    'R2_negatif_seri': int((valid_r2 < 0).sum()) if len(valid_r2) else 0,
                    'R2_ortalama_hesaplanabilir': float(np.mean(valid_r2)) if len(valid_r2) else None,
                    'R2_medyan_hesaplanabilir': float(np.median(valid_r2)) if len(valid_r2) else None,
                    'degismedi_orani_ortalama': float(sm_target.DegismediOrani.mean()),
                    'degismedi_orani_medyan': float(sm_target.DegismediOrani.median()),
                },
                'egitimde_gorulen_satir_agirlikli': clean_json_Record(seen_pooled),
                'egitimde_gorulmeyen_satir_agirlikli': (
                    clean_json_Record(unseen_pooled) if unseen_pooled is not None else None
                ),
            }

        unseen_ids = sorted(set(pred_df.loc[~pred_df.EgitimdeGoruldu, 'SeriId']))
        split_summaries[split_name] = {
            'ornek_sayisi': int(len(pred_df)),
            'seri_sayisi': int(pred_df.SeriId.nunique()),
            'urun_kimligi_sayisi': int(pred_df.TabelaGId.nunique()),
            'ilk_gozlem_tarihi': str(pred_df.Tarih.min()),
            'son_gozlem_tarihi': str(pred_df.Tarih.max()),
            'ilk_hedef_tarihi': str(pred_df.HedefTarih.min()),
            'son_hedef_tarihi': str(pred_df.HedefTarih.max()),
            'egitimde_gorulen_seri_sayisi': int(pred_df.loc[pred_df.EgitimdeGoruldu, 'SeriId'].nunique()),
            'egitimde_gorulen_ornek_sayisi': int(pred_df.EgitimdeGoruldu.sum()),
            'egitimde_gorulmeyen_seri_sayisi': int(len(unseen_ids)),
            'egitimde_gorulmeyen_ornek_sayisi': int((~pred_df.EgitimdeGoruldu).sum()),
            'egitimde_gorulmeyen_seriler': unseen_ids,
            'inceleme_isaretli_mevcut_sayisi': int(pred_df.IncelemeGerekli.sum()),
            'inceleme_isaretli_hedef_sayisi': int(pred_df.HedefIncelemeGerekli.sum()),
            'inceleme_isaretli_gecmis_penceresi_sayisi': int(pred_df.GecmisteIncelemeSayisi.gt(0).sum()),
            'ters_tahmin_araligi_sayisi': int((~pred_df.TahminAraligiGecerli).sum()),
            'hedefler': target_dict,
        }

    baseline_contract = {
        'surum': VERSION,
        'asama': 3,
        'yontem_kodu': 'naive_persistence_t',
        'yontem_adi': METHOD_NAME,
        'kural': {
            'tahmin_min': 'fiyat_min',
            'tahmin_max': 'fiyat_max',
            'matematiksel_ifade': {
                'hedef_min': 'y_hat_min(t+1) = fiyat_min(t)',
                'hedef_max': 'y_hat_max(t+1) = fiyat_max(t)',
            },
        },
        'hata_tanimi': ERROR_DEFINITION,
        'parametre_ogrenildi': False,
        'model_egitildi': False,
        'hiperparametre_var': False,
        'sahte_model_dosyasi_uretildi': False,
        'harici_veri_kullanildi': False,
        'asama2_ozellikleri_degistirildi': False,
        'kronolojik_sinirlar': bounds,
        'kullanilan_girdi_ozellikleri': ['fiyat_min', 'fiyat_max'],
        'degerlendirilen_hedefler': ['hedef_min', 'hedef_max'],
        'r2_durum_kurallari': {
            'gozlem_yetersiz': 'OrnekSayisi < 2 ise R2 hesaplanamaz (null).',
            'hedef_sabit': 'OrnekSayisi >= 2 ve SSTot == 0 ise R2 hesaplanamaz (null); 0 veya 1 atanmaz.',
            'hesaplanabilir': 'OrnekSayisi >= 2 ve SSTot > 0 ise R2 = 1 - SSRes / SSTot.',
        },
        'birim_politikasi': 'BirimTurId kodları korunur; doğrulanmamış kilogram/adet/demet adları tahmin edilmez.',
        'kapsam_politikasi': 'Tüm uygun örnekler ana değerlendirme evrenidir; inceleme işaretli veya eğitimde görülmeyen seriler silinmez.',
        'tahmin_sutunlari': PREDICTION_COLUMNS,
        'seri_metrik_sutunlari': SERIES_METRIC_COLUMNS,
        'sonraki_asama_onay_gerektirir': True,
    }

    summary = {
        'calistirma_zamani': datetime.now(ZoneInfo('Europe/Istanbul')).isoformat(),
        'surum': VERSION,
        'yontem': METHOD_NAME,
        'hata_tanimi': ERROR_DEFINITION,
        'paket_surumleri': {
            'python': platform.python_version(),
            'pandas': pd.__version__,
            'numpy': np.__version__,
            'scikit_learn': sklearn.__version__,
        },
        'girdi_mutabakati': {
            'asama1_temiz_gozlem': int(summary2['temiz_gozlem']),
            'asama2_toplam_etiketli_ornek': int(summary2['kullanilabilir_ornek']),
            'egitim_ornek_sayisi': int(len(parts['egitim'])),
            'egitim_seri_sayisi': int(parts['egitim'].SeriId.nunique()),
            'dogrulama_ornek_sayisi': int(len(parts['dogrulama'])),
            'dogrulama_seri_sayisi': int(parts['dogrulama'].SeriId.nunique()),
            'test_ornek_sayisi': int(len(parts['test'])),
            'test_seri_sayisi': int(parts['test'].SeriId.nunique()),
            'son_gun_hazir_seri_sayisi': int(summary2['son_gun']['ozellik_hazir']),
        },
        'sinirlar': bounds,
        'bolumler': split_summaries,
        'egitimde_gorulmeyen_seri_detaylari': [
            clean_json_Record(r) for r in unseen_series_metrics.to_dict('records')
        ],
        'tutarlilik_kontrolleri': {
            'kaynak_kayit_no_tekil': True,
            'hedef_tarihi_gozlem_tarihinden_sonra': True,
            'tahmin_min_esittir_fiyat_min': True,
            'tahmin_max_esittir_fiyat_max': True,
            'nan_veya_sonsuz_tahmin_var': False,
            'ters_tahmin_araligi_var': False,
            'negatif_veya_sifir_tahmin_var': False,
            'girdi_dosyalari_degismedi': True,
        },
    }

    json_write(summary, output_dir / 'ozet.json')
    json_write(baseline_contract, output_dir / 'baseline_sozlesmesi.json')
    write_report(summary, series_metrics, scope_summary, unseen_series_metrics, output_dir)

    # 5. İşlem sonrası kaynak bütünlüğü kontrolü ve manifesto yazımı
    original_hashes_after = {p: sha256(Path(p)) for p in original_hashes_before}
    stage1_hashes_after = {p: sha256(Path(p)) for p in stage1_hashes_before}
    stage2_hashes_after = {p: sha256(Path(p)) for p in stage2_hashes_before}
    if (
        original_hashes_after != original_hashes_before
        or stage1_hashes_after != stage1_hashes_before
        or stage2_hashes_after != stage2_hashes_before
    ):
        raise RuntimeError('Çalıştırma sırasında orijinal CSV, Aşama 1 veya Aşama 2 dosyaları değişti!')

    manifest = {
        'asama1_dizini': str(asama1_dir),
        'asama2_dizini': str(asama2_dir),
        'orijinal_csv_once': original_hashes_before,
        'orijinal_csv_sonra': original_hashes_after,
        'asama1_once': stage1_hashes_before,
        'asama1_sonra': stage1_hashes_after,
        'asama2_once': stage2_hashes_before,
        'asama2_sonra': stage2_hashes_after,
        'kaynaklar_degismedi': True,
        'kod_sha256': sha256(Path(__file__)),
        'paket_surumleri': summary['paket_surumleri'],
        'cikti_sha256': {
            p.name: sha256(p)
            for p in sorted(output_dir.iterdir())
            if p.is_file() and p.name != 'kaynak_manifestosu.json' and not p.name.endswith('.tmp')
        },
    }
    json_write(manifest, output_dir / 'kaynak_manifestosu.json')
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asama1', type=Path, default=ROOT / 'ciktilar/asama1')
    parser.add_argument('--asama2', type=Path, default=ROOT / 'ciktilar/asama2')
    parser.add_argument('--cikti', type=Path, default=ROOT / 'ciktilar/asama3')
    args = parser.parse_args()
    result = run(args.asama1, args.asama2, args.cikti)
    print(json.dumps(result['bolumler'], ensure_ascii=False, indent=2))
    print(f'Rapor: {args.cikti.resolve() / "ASAMA3_RAPORU.md"}')


if __name__ == '__main__':
    main()
