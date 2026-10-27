"""Kaydedilen bütün Aşama 3 Naive Baseline tahmin ve metrik dosyalarını bağımsız denetler.

asama3_baseline modülündeki hesap fonksiyonlarını çağırmaz. Kaydedilmiş tahmin
dosyalarını, Aşama 2/Aşama 1 kaynaklarını, hata dizilerini, seri bazlı metrikleri
ve kapsam özetlerini doğrudan NumPy/Pandas ile yeniden hesaplayarak karşılaştırır.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def digest(path: Path | str) -> str:
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def independent_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Bağımsız NumPy dizileri üzerinden MAE, MSE, RMSE ve R² durumunu hesaplar."""
    err = y_pred - y_true
    abs_err = np.abs(err)
    sq_err = err ** 2
    n = len(y_true)
    mae = float(np.mean(abs_err))
    mse = float(np.mean(sq_err))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum(sq_err))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if n < 2:
        r2, status, valid = np.nan, 'gozlem_yetersiz', False
    elif ss_tot == 0.0 or np.all(y_true == y_true[0]):
        r2, status, valid = np.nan, 'hedef_sabit', False
    else:
        r2, status, valid = float(1.0 - ss_res / ss_tot), 'hesaplanabilir', True
    unchanged = int(np.sum(err == 0.0))
    return {
        'OrnekSayisi': n,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'R2Durumu': status,
        'R2Gecerli': valid,
        'SSRes': ss_res,
        'SSTot': ss_tot,
        'OrtalamaHata': float(np.mean(err)),
        'MedyanMutlakHata': float(np.median(abs_err)),
        'MaksMutlakHata': float(np.max(abs_err)),
        'DegismediSayisi': unchanged,
        'DegismediOrani': float(unchanged / n),
    }


def verify(output: Path) -> dict:
    output = Path(output).resolve()
    manifest = json.loads((output / 'kaynak_manifestosu.json').read_text(encoding='utf-8'))
    contract = json.loads((output / 'baseline_sozlesmesi.json').read_text(encoding='utf-8'))
    summary = json.loads((output / 'ozet.json').read_text(encoding='utf-8'))
    checks = []

    # 1. SHA-256 bütünlük kontrolleri
    for before_key, after_key in [
        ('orijinal_csv_once', 'orijinal_csv_sonra'),
        ('asama1_once', 'asama1_sonra'),
        ('asama2_once', 'asama2_sonra'),
    ]:
        for path_str, expected in manifest[before_key].items():
            assert digest(path_str) == expected == manifest[after_key][path_str], path_str
    for name, expected in manifest['cikti_sha256'].items():
        assert digest(output / name) == expected, name
    assert digest(ROOT / 'asama3_baseline.py') == manifest['kod_sha256']
    checks.append('Orijinal CSV, Aşama 1, Aşama 2 ve Aşama 3 dosyalarının SHA-256 özetleri tam eşleşiyor.')

    # 2. Aşama 1 ve Aşama 2 kaynaklarıyla satır/kimlik mutabakatı
    asama1_dir = Path(manifest['asama1_dizini'])
    asama2_dir = Path(manifest['asama2_dizini'])
    clean = pd.read_csv(asama1_dir / 'temiz_veri.csv', keep_default_na=False, float_precision='round_trip')
    calendar = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(asama1_dir / 'veri_gunleri.csv').Tarih))
    clean_by_id = clean.set_index('KaynakKayitNo')

    stage2_parts = {
        name: pd.read_csv(asama2_dir / f'{name}.csv.gz', keep_default_na=False, float_precision='round_trip')
        for name in ('egitim', 'dogrulama', 'test')
    }
    train_counts = stage2_parts['egitim'].groupby('SeriId').size().to_dict()

    pred_parts = {
        'dogrulama': pd.read_csv(output / 'dogrulama_tahminleri.csv.gz', keep_default_na=False, float_precision='round_trip'),
        'test': pd.read_csv(output / 'test_tahminleri.csv.gz', keep_default_na=False, float_precision='round_trip'),
    }

    bounds = contract['kronolojik_sinirlar']
    for split_name, pred in pred_parts.items():
        src = stage2_parts[split_name]
        assert len(pred) == len(src) == summary['bolumler'][split_name]['ornek_sayisi']
        assert pred.SeriId.nunique() == src.SeriId.nunique() == summary['bolumler'][split_name]['seri_sayisi']
        assert pred.KaynakKayitNo.is_unique
        assert not pred.duplicated(['SeriId', 'Tarih']).any()
        assert not pred.duplicated(['SeriId', 'HedefTarih']).any()
        assert pred.Bolum.eq(split_name).all()

        # Kaynak ve hedef kimliklerinin Aşama 2 ve Aşama 1 ile birebir eşleşmesi
        np.testing.assert_array_equal(pred.KaynakKayitNo.to_numpy(), src.KaynakKayitNo.to_numpy())
        np.testing.assert_array_equal(pred.HedefKaynakKayitNo.to_numpy(), src.HedefKaynakKayitNo.to_numpy())
        for col in ('Tarih', 'HedefTarih', 'SeriId', 'TabelaGId', 'UrunAd', 'BirimTurId',
                    'KategoriId', 'HalTurId', 'AdDonemi', 'BaglamId',
                    'IncelemeGerekli', 'HedefIncelemeGerekli', 'GecmisteIncelemeSayisi'):
            np.testing.assert_array_equal(pred[col].to_numpy(), src[col].to_numpy(), err_msg=f'{split_name}.{col}')

        origins = clean_by_id.loc[pred.KaynakKayitNo].reset_index(drop=True)
        targets = clean_by_id.loc[pred.HedefKaynakKayitNo].reset_index(drop=True)
        np.testing.assert_array_equal(pred.fiyat_min.to_numpy(dtype=float), origins.EnDusukFiyat.to_numpy(dtype=float))
        np.testing.assert_array_equal(pred.fiyat_max.to_numpy(dtype=float), origins.EnYuksekFiyat.to_numpy(dtype=float))
        np.testing.assert_array_equal(pred.hedef_min.to_numpy(dtype=float), targets.EnDusukFiyat.to_numpy(dtype=float))
        np.testing.assert_array_equal(pred.hedef_max.to_numpy(dtype=float), targets.EnYuksekFiyat.to_numpy(dtype=float))

        # Kronolojik sıralama ve sınır kontrolü
        t_orig = pd.to_datetime(pred.Tarih)
        t_targ = pd.to_datetime(pred.HedefTarih)
        assert t_targ.gt(t_orig).all()
        cal_pos = calendar.get_indexer(t_orig)
        assert (cal_pos >= 0).all() and (cal_pos < len(calendar) - 1).all()
        np.testing.assert_array_equal(calendar[cal_pos + 1], t_targ)
        if split_name == 'dogrulama':
            assert t_targ.between(bounds['dogrulama_ilk_dahil'], bounds['dogrulama_son_dahil']).all()
        else:
            assert t_targ.between(bounds['test_ilk_dahil'], bounds['test_son_dahil']).all()

        # Naive Baseline tahmin eşitliği ve geçerlilik kontrolleri
        num_cols = [
            'fiyat_min', 'fiyat_max', 'hedef_min', 'hedef_max',
            'tahmin_min', 'tahmin_max', 'hata_min', 'hata_max',
            'mutlak_hata_min', 'mutlak_hata_max', 'karesel_hata_min', 'karesel_hata_max',
        ]
        assert np.isfinite(pred[num_cols].to_numpy(dtype=float)).all()
        np.testing.assert_array_equal(pred.tahmin_min.to_numpy(dtype=float), pred.fiyat_min.to_numpy(dtype=float))
        np.testing.assert_array_equal(pred.tahmin_max.to_numpy(dtype=float), pred.fiyat_max.to_numpy(dtype=float))
        assert (pred.tahmin_min.to_numpy(dtype=float) > 0).all()
        assert (pred.tahmin_max.to_numpy(dtype=float) > 0).all()
        assert (pred.tahmin_min.to_numpy(dtype=float) <= pred.tahmin_max.to_numpy(dtype=float)).all()
        assert pred.TahminAraligiGecerli.all()

        # Hata dizilerinin bağımsız kontrolü (hata = tahmin - gercek)
        err_min = pred.tahmin_min.to_numpy(dtype=float) - pred.hedef_min.to_numpy(dtype=float)
        err_max = pred.tahmin_max.to_numpy(dtype=float) - pred.hedef_max.to_numpy(dtype=float)
        np.testing.assert_allclose(pred.hata_min.to_numpy(dtype=float), err_min, rtol=0, atol=1e-12)
        np.testing.assert_allclose(pred.hata_max.to_numpy(dtype=float), err_max, rtol=0, atol=1e-12)
        np.testing.assert_allclose(pred.mutlak_hata_min.to_numpy(dtype=float), np.abs(err_min), rtol=0, atol=1e-12)
        np.testing.assert_allclose(pred.mutlak_hata_max.to_numpy(dtype=float), np.abs(err_max), rtol=0, atol=1e-12)
        np.testing.assert_allclose(pred.karesel_hata_min.to_numpy(dtype=float), err_min ** 2, rtol=0, atol=1e-12)
        np.testing.assert_allclose(pred.karesel_hata_max.to_numpy(dtype=float), err_max ** 2, rtol=0, atol=1e-12)

        # Eğitimde görülme metadatası kontrolü
        expected_counts = pred.SeriId.map(train_counts).fillna(0).astype('int64')
        np.testing.assert_array_equal(pred.EgitimOrnekSayisi.to_numpy(), expected_counts.to_numpy())
        np.testing.assert_array_equal(pred.EgitimdeGoruldu.to_numpy(), expected_counts.gt(0).to_numpy())

        # Birim açıklamasının doğrulanmamış formatta olması
        assert pred.BirimAciklamasi.str.contains('adı doğrulanmadı').all()

    assert set(pred_parts['dogrulama'].KaynakKayitNo).isdisjoint(pred_parts['test'].KaynakKayitNo)
    checks.append('Doğrulama (30.361) ve test (21.141) tahmin kayıtları, Aşama 1/2 kaynakları ve hata dizileri bağımsız doğrulandı.')

    # 3. Seri bazlı metriklerin (seri_metrikleri.csv) bağımsız yeniden hesaplanması
    series_df = pd.read_csv(output / 'seri_metrikleri.csv', keep_default_na=False, na_values=[''], float_precision='round_trip')
    expected_series_rows = 2 * (pred_parts['dogrulama'].SeriId.nunique() + pred_parts['test'].SeriId.nunique())
    assert len(series_df) == expected_series_rows == 550

    combined = pd.concat([pred_parts['dogrulama'], pred_parts['test']], ignore_index=True)
    recalc_index = 0
    for (split_name, seri_id), group in combined.sort_values(['Bolum', 'SeriId', 'HedefTarih'], kind='stable').groupby(
        ['Bolum', 'SeriId'], sort=False
    ):
        for target_col, pred_col in [('hedef_min', 'tahmin_min'), ('hedef_max', 'tahmin_max')]:
            row = series_df.iloc[recalc_index]
            recalc_index += 1
            assert row.Bolum == split_name and row.SeriId == seri_id and row.HedefTuru == target_col
            m = independent_metrics(group[target_col].to_numpy(dtype=float), group[pred_col].to_numpy(dtype=float))
            assert int(row.OrnekSayisi) == m['OrnekSayisi']
            assert row.R2Durumu == m['R2Durumu']
            assert bool(row.R2Gecerli) == m['R2Gecerli']
            for metric_name in ('MAE', 'MSE', 'RMSE', 'SSRes', 'SSTot', 'OrtalamaHata', 'MedyanMutlakHata', 'MaksMutlakHata', 'DegismediOrani'):
                np.testing.assert_allclose(float(row[metric_name]), m[metric_name], rtol=1e-12, atol=1e-12, err_msg=f'{seri_id}.{target_col}.{metric_name}')
            assert int(row.DegismediSayisi) == m['DegismediSayisi']
            if m['R2Gecerli']:
                np.testing.assert_allclose(float(row.R2), m['R2'], rtol=1e-12, atol=1e-12)
            else:
                assert pd.isna(row.R2)
    checks.append('seri_metrikleri.csv içindeki 550 seri-hedef satırının tamamı bağımsız hesapla birebir eşleşti.')

    # 4. Kapsam özeti (kapsam_ozeti.csv) ve ozet.json mutabakatı
    scope_df = pd.read_csv(output / 'kapsam_ozeti.csv', keep_default_na=False, na_values=[''], float_precision='round_trip')
    for split_name, pred in pred_parts.items():
        for target_col, pred_col in [('hedef_min', 'tahmin_min'), ('hedef_max', 'tahmin_max')]:
            m_all = independent_metrics(pred[target_col].to_numpy(dtype=float), pred[pred_col].to_numpy(dtype=float))
            rep_all = summary['bolumler'][split_name]['hedefler'][target_col]['satir_agirlikli']
            for k in ('OrnekSayisi', 'MAE', 'MSE', 'RMSE', 'R2', 'SSRes', 'SSTot', 'DegismediSayisi', 'DegismediOrani'):
                np.testing.assert_allclose(rep_all[k], m_all[k], rtol=1e-12, atol=1e-12)
            scope_row = scope_df.loc[
                scope_df.Bolum.eq(split_name)
                & scope_df.KapsamKodu.eq('tum_uygun_ornekler')
                & scope_df.HedefTuru.eq(target_col)
            ].iloc[0]
            np.testing.assert_allclose(float(scope_row.SatirAgirlikli_MAE), m_all['MAE'], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(float(scope_row.SatirAgirlikli_RMSE), m_all['RMSE'], rtol=1e-12, atol=1e-12)
    checks.append('kapsam_ozeti.csv ve ozet.json içindeki satır ağırlıklı ve seri eşit ağırlıklı metrikler doğrulandı.')

    assert contract['model_egitildi'] is False
    assert contract['parametre_ogrenildi'] is False
    assert contract['harici_veri_kullanildi'] is False

    result = {
        'durum': 'basarili',
        'kontroller': checks,
        'dogrulanan_dogrulama_ornegi': int(len(pred_parts['dogrulama'])),
        'dogrulanan_test_ornegi': int(len(pred_parts['test'])),
        'dogrulanan_seri_metrik_satiri': int(len(series_df)),
        'dogrulanan_kapsam_ozeti_satiri': int(len(scope_df)),
    }
    (output.parent / 'asama3_dogrulama.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    return result


if __name__ == '__main__':
    print(json.dumps(verify(ROOT / 'ciktilar/asama3'), ensure_ascii=False, indent=2))
