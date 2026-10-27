"""Aşama 3 Naive Baseline metriklerini, uç durumlarını ve sızıntı/şema kurallarını sınar."""
from __future__ import annotations

import math
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from asama2_ozellikler import FEATURES, LABEL_METADATA, METADATA, TARGETS
from asama3_baseline import (
    ROOT,
    build_scope_summary,
    compute_regression_metrics,
    evaluate_series,
    predict_naive,
    validate_contract,
)
from dogrula_asama3 import verify as verify_stage3_outputs


def make_contract() -> dict:
    return {
        'surum': '2.0.0',
        'asama': 2,
        'ozellikler': list(FEATURES),
        'hedefler': list(TARGETS),
        'metadata': list(METADATA + LABEL_METADATA),
        'sinirlar': {
            'egitim_son_dahil': '2024-10-26',
            'dogrulama_ilk_dahil': '2024-10-27',
            'dogrulama_son_dahil': '2025-10-26',
            'test_ilk_dahil': '2025-10-27',
            'test_son_dahil': '2026-10-26',
        },
    }


def make_sample_rows() -> pd.DataFrame:
    rows = []
    specs = [
        # Seri 1 (birim 1): 3 gözlem
        ('2025-01-02', '2025-01-03', 'u1|b1|h2|k6|a1', 'u1', 'Domates', 1, 6, 10.0, 20.0, 12.0, 25.0, False, False),
        ('2025-01-03', '2025-01-06', 'u1|b1|h2|k6|a1', 'u1', 'Domates', 1, 6, 12.0, 25.0, 12.0, 22.0, False, False),
        ('2025-01-06', '2025-01-07', 'u1|b1|h2|k6|a1', 'u1', 'Domates', 1, 6, 12.0, 22.0, 16.0, 30.0, False, True),
        # Seri 2 (aynı ürün adı ve kimliği fakat birim 3): 2 gözlem, sabit hedef_min
        ('2025-01-02', '2025-01-03', 'u1|b3|h2|k6|a1', 'u1', 'Domates', 3, 6, 100.0, 150.0, 100.0, 160.0, False, False),
        ('2025-01-03', '2025-01-06', 'u1|b3|h2|k6|a1', 'u1', 'Domates', 3, 6, 100.0, 160.0, 100.0, 140.0, False, False),
    ]
    for idx, (t, ht, sid, gid, name, unit, cat, fmin, fmax, hmin, hmax, rev, hrev) in enumerate(specs, start=1):
        rec = {col: 1.0 for col in FEATURES}
        rec.update({
            'KaynakKayitNo': idx,
            'Tarih': t,
            'HedefTarih': ht,
            'HedefKaynakKayitNo': idx + 100,
            'SeriId': sid,
            'TabelaGId': gid,
            'UrunAd': name,
            'BirimTurId': unit,
            'KategoriId': cat,
            'HalTurId': 2,
            'AdDonemi': 1,
            'BaglamId': f'{sid}|p1',
            'OzellikEnEskiTarih': '2024-12-01',
            'OzellikEnYeniTarih': t,
            'IncelemeGerekli': rev,
            'HedefIncelemeGerekli': hrev,
            'GecmisteIncelemeSayisi': int(rev),
            'Bolum': 'dogrulama',
            'fiyat_min': fmin,
            'fiyat_max': fmax,
            'hedef_min': hmin,
            'hedef_max': hmax,
        })
        rows.append(rec)
    return pd.DataFrame(rows)


class BaselineUnitTests(unittest.TestCase):
    def test_hand_calculated_mae_mse_rmse_and_positive_r2(self):
        # y_true = [10, 20, 30] (ortalama=20, SSTot = 100 + 0 + 100 = 200)
        # y_pred = [12, 18, 33] -> hata (tahmin - gercek) = [+2, -2, +3]
        # mutlak hata = [2, 2, 3] -> MAE = 7/3
        # karesel hata = [4, 4, 9] -> SSRes = 17, MSE = 17/3, RMSE = sqrt(17/3)
        # R2 = 1 - 17/200 = 0.915
        m = compute_regression_metrics([10.0, 20.0, 30.0], [12.0, 18.0, 33.0])
        self.assertEqual(m['OrnekSayisi'], 3)
        self.assertAlmostEqual(m['MAE'], 7.0 / 3.0, places=12)
        self.assertAlmostEqual(m['MSE'], 17.0 / 3.0, places=12)
        self.assertAlmostEqual(m['RMSE'], math.sqrt(17.0 / 3.0), places=12)
        self.assertAlmostEqual(m['R2'], 1.0 - 17.0 / 200.0, places=12)
        self.assertEqual(m['R2Durumu'], 'hesaplanabilir')
        self.assertTrue(m['R2Gecerli'])
        self.assertAlmostEqual(m['OrtalamaHata'], 1.0, places=12)
        self.assertEqual(m['MedyanMutlakHata'], 2.0)
        self.assertEqual(m['MaksMutlakHata'], 3.0)
        self.assertEqual(m['DegismediSayisi'], 0)

    def test_negative_r2_is_preserved_accurately(self):
        # y_true = [10, 12, 10] -> mean = 32/3, SSTot = 8/3 = 2.666666...
        # y_pred = [14, 8, 14] -> hatalar = [+4, -4, +4], SSRes = 48
        # R2 = 1 - 48 / (8/3) = 1 - 18 = -17.0
        m = compute_regression_metrics([10.0, 12.0, 10.0], [14.0, 8.0, 14.0])
        self.assertEqual(m['R2Durumu'], 'hesaplanabilir')
        self.assertTrue(m['R2Gecerli'])
        self.assertAlmostEqual(m['R2'], -17.0, places=12)

    def test_single_observation_group_r2_is_marked_insufficient(self):
        m = compute_regression_metrics([25.0], [25.0])
        self.assertEqual(m['OrnekSayisi'], 1)
        self.assertEqual(m['MAE'], 0.0)
        self.assertEqual(m['MSE'], 0.0)
        self.assertEqual(m['RMSE'], 0.0)
        self.assertIsNone(m['R2'])
        self.assertEqual(m['R2Durumu'], 'gozlem_yetersiz')
        self.assertFalse(m['R2Gecerli'])

    def test_constant_target_group_r2_is_never_silently_zero_or_one(self):
        # Hem hatasız hem hatalı sabit hedef durumunda R2 None olmalı ve R2Durumu='hedef_sabit' olmalı
        perfect = compute_regression_metrics([50.0, 50.0, 50.0], [50.0, 50.0, 50.0])
        self.assertIsNone(perfect['R2'])
        self.assertEqual(perfect['R2Durumu'], 'hedef_sabit')
        self.assertFalse(perfect['R2Gecerli'])
        self.assertEqual(perfect['MAE'], 0.0)

        imperfect = compute_regression_metrics([50.0, 50.0, 50.0], [45.0, 50.0, 55.0])
        self.assertIsNone(imperfect['R2'])
        self.assertEqual(imperfect['R2Durumu'], 'hedef_sabit')
        self.assertFalse(imperfect['R2Gecerli'])
        self.assertAlmostEqual(imperfect['MAE'], 10.0 / 3.0, places=12)

    def test_min_and_max_targets_and_units_never_mix(self):
        df = make_sample_rows()
        pred = predict_naive(df, make_contract(), training_counts={'u1|b1|h2|k6|a1': 100})
        sm = evaluate_series(pred)
        # 2 seri x 2 hedef = 4 satır
        self.assertEqual(len(sm), 4)
        self.assertEqual(sm.SeriId.nunique(), 2)
        self.assertTrue(sm.BirimAciklamasi.str.contains('adı doğrulanmadı').all())

        s1_min = sm.loc[sm.SeriId.eq('u1|b1|h2|k6|a1') & sm.HedefTuru.eq('hedef_min')].iloc[0]
        s1_max = sm.loc[sm.SeriId.eq('u1|b1|h2|k6|a1') & sm.HedefTuru.eq('hedef_max')].iloc[0]
        # Seri 1 min: tahmin=[10,12,12], hedef=[12,12,16] -> hatalar=[-2,0,-4] -> MAE=2.0, MSE=20/3
        self.assertAlmostEqual(s1_min.MAE, 2.0, places=12)
        self.assertAlmostEqual(s1_min.MSE, 20.0 / 3.0, places=12)
        self.assertTrue(s1_min.EgitimdeGoruldu)
        self.assertEqual(s1_min.EgitimOrnekSayisi, 100)

        # Seri 1 max: tahmin=[20,25,22], hedef=[25,22,30] -> hatalar=[-5,+3,-8] -> MAE=16/3, MSE=98/3
        self.assertAlmostEqual(s1_max.MAE, 16.0 / 3.0, places=12)
        self.assertAlmostEqual(s1_max.MSE, 98.0 / 3.0, places=12)

        # Seri 2 (birim 3): eğitimde görülmedi, hedef_min sabit (100, 100)
        s2_min = sm.loc[sm.SeriId.eq('u1|b3|h2|k6|a1') & sm.HedefTuru.eq('hedef_min')].iloc[0]
        s2_max = sm.loc[sm.SeriId.eq('u1|b3|h2|k6|a1') & sm.HedefTuru.eq('hedef_max')].iloc[0]
        self.assertFalse(s2_min.EgitimdeGoruldu)
        self.assertEqual(s2_min.EgitimOrnekSayisi, 0)
        self.assertEqual(s2_min.R2Durumu, 'hedef_sabit')
        self.assertTrue(pd.isna(s2_min.R2))
        self.assertEqual(s2_max.R2Durumu, 'hesaplanabilir')
        self.assertFalse(pd.isna(s2_max.R2))

    def test_predictions_depend_only_on_current_price_and_ignore_future_target_changes(self):
        df1 = make_sample_rows()
        df2 = df1.copy(deep=True)
        # Gelecekteki gerçek hedefleri 10 katına çıkaralım ve diğer lag özelliklerini değiştirelim
        df2['hedef_min'] = df2['hedef_min'] * 10.0
        df2['hedef_max'] = df2['hedef_max'] * 10.0
        df2['min_lag_1'] = 99999.0
        df2['max_lag_1'] = 99999.0

        p1 = predict_naive(df1, make_contract())
        p2 = predict_naive(df2, make_contract())

        np.testing.assert_array_equal(p1.tahmin_min.to_numpy(), df1.fiyat_min.to_numpy())
        np.testing.assert_array_equal(p1.tahmin_max.to_numpy(), df1.fiyat_max.to_numpy())
        np.testing.assert_array_equal(p1.tahmin_min.to_numpy(), p2.tahmin_min.to_numpy())
        np.testing.assert_array_equal(p1.tahmin_max.to_numpy(), p2.tahmin_max.to_numpy())

    def test_input_dataframe_is_never_mutated(self):
        df = make_sample_rows()
        before = df.copy(deep=True)
        _ = predict_naive(df, make_contract(), training_counts={'u1|b1|h2|k6|a1': 10})
        pd.testing.assert_frame_equal(df, before)

    def test_validation_and_test_splits_do_not_mix_in_summaries(self):
        val_df = make_sample_rows()
        test_df = make_sample_rows()
        test_df['Bolum'] = 'test'
        test_df['KaynakKayitNo'] = test_df['KaynakKayitNo'] + 1000
        test_df['Tarih'] = ['2026-01-02', '2026-01-03', '2026-01-06', '2026-01-02', '2026-01-03']
        test_df['HedefTarih'] = ['2026-01-05', '2026-01-06', '2026-01-07', '2026-01-05', '2026-01-06']
        test_df['hedef_min'] = test_df['hedef_min'] + 50.0
        test_df['hedef_max'] = test_df['hedef_max'] + 50.0

        val_pred = predict_naive(val_df, make_contract(), expected_split='dogrulama')
        test_pred = predict_naive(test_df, make_contract(), expected_split='test')
        scopes = build_scope_summary({'dogrulama': val_pred, 'test': test_pred})

        val_mae = scopes.loc[
            scopes.Bolum.eq('dogrulama') & scopes.KapsamKodu.eq('tum_uygun_ornekler') & scopes.HedefTuru.eq('hedef_min'),
            'SatirAgirlikli_MAE',
        ].iloc[0]
        test_mae = scopes.loc[
            scopes.Bolum.eq('test') & scopes.KapsamKodu.eq('tum_uygun_ornekler') & scopes.HedefTuru.eq('hedef_min'),
            'SatirAgirlikli_MAE',
        ].iloc[0]
        self.assertNotAlmostEqual(val_mae, test_mae)
        with self.assertRaisesRegex(ValueError, 'Bölüm etiketleri'):
            predict_naive(val_df, make_contract(), expected_split='test')

    def test_invalid_schema_prices_or_dates_raise_clear_errors(self):
        contract = make_contract()
        df = make_sample_rows()

        # Eksik sütun
        with self.assertRaisesRegex(ValueError, 'zorunlu sütunlar eksik'):
            predict_naive(df.drop(columns=['fiyat_min']), contract)

        # Ters mevcut fiyat aralığı (fiyat_min > fiyat_max)
        inverted = df.copy()
        inverted.loc[0, 'fiyat_min'] = 50.0
        inverted.loc[0, 'fiyat_max'] = 20.0
        with self.assertRaisesRegex(ValueError, 'fiyat_min > fiyat_max'):
            predict_naive(inverted, contract)

        # Sıfır veya negatif fiyat
        non_pos = df.copy()
        non_pos.loc[0, 'fiyat_min'] = 0.0
        with self.assertRaisesRegex(ValueError, 'sıfır/negatif'):
            predict_naive(non_pos, contract)

        # HedefTarih <= Tarih
        bad_date = df.copy()
        bad_date.loc[0, 'HedefTarih'] = bad_date.loc[0, 'Tarih']
        with self.assertRaisesRegex(ValueError, 'HedefTarih'):
            predict_naive(bad_date, contract)

        # Bozuk sözleşme (hedef özelliği X içinde)
        bad_contract = make_contract()
        bad_contract['ozellikler'] = FEATURES[:-1] + ['hedef_min']
        with self.assertRaisesRegex(ValueError, 'kesişiyor'):
            validate_contract(bad_contract)

    def test_saved_stage3_outputs_pass_independent_verification(self):
        result = verify_stage3_outputs(ROOT / 'ciktilar/asama3')
        self.assertEqual(result['durum'], 'basarili')
        self.assertEqual(result['dogrulanan_dogrulama_ornegi'], 30361)
        self.assertEqual(result['dogrulanan_test_ornegi'], 21141)
        self.assertEqual(result['dogrulanan_seri_metrik_satiri'], 550)


if __name__ == '__main__':
    unittest.main()
