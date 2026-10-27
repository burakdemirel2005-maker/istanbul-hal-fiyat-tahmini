#!/usr/bin/env python3
"""Aşama 4–8 Birim ve Entegrasyon Testleri (Bölüm 14 — 12 Test Alanı).

Kapsanan test alanları:
  1. Veri sızıntısı kontrolü
  2. Eğitim, doğrulama ve test ayrımı kontrolü
  3. Ön işleme tutarlılığı
  4. Model yükleme ve SHA-256 doğrulama kontrolü
  5. Tahmin üretimi kontrolü
  6. Ürün ID (SeriId) eşleşmesi ve kesintili seri geri dönüş kontrolü
  7. Minimum fiyatın maksimum fiyattan büyük olmaması (min <= max tutarlılığı)
  8. Satın alma hesaplama doğruluğu (3 fiyat türü ve bütçe kontrolü)
  9. Çoklu ürün sepeti simülasyonu doğruluğu
  10. Hatalı kullanıcı girişlerinin engellenmesi (miktar <= 0, NaN, inf, negatif bütçe vb.)
  11. Web uygulaması veri bağlantıları ve 6 sayfa yapısı
  12. Eksik model veya veri dosyası durumunda anlamlı hata üretimi
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from app import PAGES, load_dashboard_bundle, verify_required_app_files
from asama4_modeller import build_preprocessor
from asama6_satin_alma import PurchasingDecisionSupport
from tahmin_servisi import (
    CATEGORICAL_FEATURES,
    HalPredictionService,
    SELECTED_NUMERIC_FEATURES,
    SERIES_STAT_FEATURES,
    SeriesHistoryStatsEncoder,
    enforce_min_max_consistency,
)

ROOT = Path(__file__).resolve().parent


class TestAsama4To8(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.service = HalPredictionService(ROOT / 'ciktilar/asama4')
        cls.pds = PurchasingDecisionSupport(ROOT / 'ciktilar/asama4', ROOT / 'ciktilar/asama5')
        cls.latest_features = pd.read_csv(ROOT / 'ciktilar/asama2/son_gun_girdileri.csv', low_memory=False)
        cls.latest_status = pd.read_csv(ROOT / 'ciktilar/asama2/son_gozlem_durumu.csv', low_memory=False)
        cls.clean_df = pd.read_csv(ROOT / 'ciktilar/asama1/temiz_veri.csv', low_memory=False)
        cls.val_preds = pd.read_csv(ROOT / 'ciktilar/asama4/dogrulama_tum_modeller_tahminleri.csv.gz', low_memory=False)
        cls.test_preds = pd.read_csv(ROOT / 'ciktilar/asama4/test_tum_modeller_tahminleri.csv.gz', low_memory=False)

    def test_01_no_data_leakage_in_series_encoder(self) -> None:
        """1. Veri sızıntısı kontrolü: SeriesHistoryStatsEncoder yalnızca eğitim verisinden öğrenir."""
        art = self.service.artifacts[('random_forest', 'hedef_min')]
        enc = art.series_encoder
        # Doğrulamada ilk kez görülen seriler eğitim encoder tablosunda yer almamalıdır
        unseen_val_sid = '1b84d414-b4b8-4f46-aec3-3fb739842207|b1|h2|k5|a2'
        self.assertNotIn(unseen_val_sid, enc.seen_series_)
        # Yeni bir seri geldiğinde global eğitim ortalamasını sızıntısız atamalıdır
        sample_row = self.latest_features.iloc[[0]].copy()
        sample_row['SeriId'] = 'yeni_gorulmemis_seri_id'
        transformed = enc.transform(sample_row)
        for col in SERIES_STAT_FEATURES:
            self.assertAlmostEqual(float(transformed[col].iloc[0]), float(enc.global_defaults_[col]), places=9)

    def test_02_chronological_split_and_counts(self) -> None:
        """2. Eğitim, doğrulama ve test ayrımı kontrolü."""
        s2_summary = json.loads((ROOT / 'ciktilar/asama2/ozet.json').read_text(encoding='utf-8'))
        self.assertEqual(s2_summary['bolumler']['egitim']['kayit'], 303394)
        self.assertEqual(s2_summary['bolumler']['dogrulama']['kayit'], 30361)
        self.assertEqual(s2_summary['bolumler']['test']['kayit'], 21141)
        self.assertEqual(len(self.val_preds), 30361)
        self.assertEqual(len(self.test_preds), 21141)
        self.assertLessEqual(str(self.val_preds['HedefTarih'].max()), '2025-10-26')
        self.assertGreaterEqual(str(self.test_preds['HedefTarih'].min()), '2025-10-27')
        val_ids = set(self.val_preds['KaynakKayitNo'])
        test_ids = set(self.test_preds['KaynakKayitNo'])
        self.assertEqual(len(val_ids.intersection(test_ids)), 0)

    def test_03_preprocessing_consistency_and_unknown_categories(self) -> None:
        """3. Ön işleme tutarlılığı ve bilinmeyen kategorilerde güvenli dönüşüm."""
        sample_df = self.latest_features.head(10).copy()
        art = self.service.artifacts[('linear_regression', 'hedef_min')]
        enc_df = art.series_encoder.transform(sample_df)
        self.assertEqual(
            list(enc_df.columns),
            SELECTED_NUMERIC_FEATURES + CATEGORICAL_FEATURES + SERIES_STAT_FEATURES,
        )
        # Bilinmeyen BirimTurId veya KategoriId geldiğinde OneHotEncoder hata vermemelidir
        unseen_cat_df = enc_df.copy()
        unseen_cat_df['BirimTurId'] = 999
        unseen_cat_df['KategoriId'] = 999
        transformed_matrix = art.pipeline.named_steps['preprocessor'].transform(unseen_cat_df)
        self.assertTrue(np.isfinite(transformed_matrix).all())

    def test_04_model_loading_and_registry_integrity(self) -> None:
        """4. Model yükleme ve kayıt defteri SHA-256 bütünlüğü kontrolü."""
        self.assertEqual(self.service.selected_model_code, 'random_forest')
        self.assertEqual(len(self.service.artifacts), 6)
        for mcode in ('linear_regression', 'random_forest', 'xgboost'):
            for tcol in ('hedef_min', 'hedef_max'):
                self.assertIn((mcode, tcol), self.service.artifacts)
                art = self.service.artifacts[(mcode, tcol)]
                self.assertIn('mae', art.validation_metrics)
                self.assertIn('rmse', art.validation_metrics)
                self.assertIn('mae', art.test_metrics)
                self.assertIn('rmse', art.test_metrics)

    def test_05_prediction_generation_all_models(self) -> None:
        """5. Tahmin üretimi kontrolü: 4 modelin tamamı geçerli ve sonlu tahmin üretir."""
        for mcode in self.service.available_models():
            pred_df = self.service.predict_dataframe(self.latest_features, model_code=mcode)
            self.assertEqual(len(pred_df), 91)
            self.assertTrue(np.isfinite(pred_df['tahmin_min']).all())
            self.assertTrue(np.isfinite(pred_df['tahmin_max']).all())
            self.assertTrue((pred_df['tahmin_min'] > 0).all())
            self.assertTrue((pred_df['tahmin_max'] >= pred_df['tahmin_min']).all())

    def test_06_product_id_matching_and_discontinued_fallback(self) -> None:
        """6. Ürün ID eşleşmesi: aktif seride ML tahmini, kesintili seride dürüst geri dönüş."""
        active_sid = str(self.latest_features.iloc[0]['SeriId'])
        res_active = self.service.predict_for_series_row(
            series_id=active_sid,
            latest_features_df=self.latest_features,
            status_df=self.latest_status,
            clean_df=self.clean_df,
            model_code='random_forest',
        )
        self.assertTrue(res_active['tahmin_uretildi'])
        self.assertFalse(res_active['geri_donus_kullanildi'])
        self.assertEqual(res_active['kullanilan_model_kodu'], 'random_forest')

        inactive_sids = set(self.latest_status['SeriId']) - set(self.latest_features['SeriId'])
        inactive_sid = sorted(inactive_sids)[0]
        res_inactive_no_fb = self.service.predict_for_series_row(
            series_id=inactive_sid,
            latest_features_df=self.latest_features,
            status_df=self.latest_status,
            clean_df=self.clean_df,
            model_code='random_forest',
            allow_naive_fallback=False,
        )
        self.assertFalse(res_inactive_no_fb['tahmin_uretildi'])

        res_inactive_fb = self.service.predict_for_series_row(
            series_id=inactive_sid,
            latest_features_df=self.latest_features,
            status_df=self.latest_status,
            clean_df=self.clean_df,
            model_code='random_forest',
            allow_naive_fallback=True,
        )
        self.assertTrue(res_inactive_fb['tahmin_uretildi'])
        self.assertTrue(res_inactive_fb['geri_donus_kullanildi'])
        self.assertEqual(res_inactive_fb['kullanilan_model_kodu'], 'naive_baseline')

        with self.assertRaises(ValueError):
            self.service.predict_for_series_row(
                series_id='var_olmayan_seri_kimligi',
                latest_features_df=self.latest_features,
                status_df=self.latest_status,
                clean_df=self.clean_df,
            )

    def test_07_min_not_greater_than_max_consistency(self) -> None:
        """7. Minimum fiyatın maksimum fiyattan büyük olmaması kontrolü."""
        raw_min = np.array([15.0, 40.0, -2.0])
        raw_max = np.array([20.0, 32.0, -5.0])
        fix_min, fix_max, inv = enforce_min_max_consistency(raw_min, raw_max)
        self.assertTrue((fix_min <= fix_max).all())
        self.assertTrue((fix_min >= 0.01).all())
        self.assertEqual(inv.tolist(), [False, True, False])
        self.assertAlmostEqual(float(fix_min[1]), 32.0)
        self.assertAlmostEqual(float(fix_max[1]), 40.0)

        for mcode in self.service.available_models():
            self.assertTrue((self.val_preds[f'{mcode}_tahmin_min'] <= self.val_preds[f'{mcode}_tahmin_max']).all())
            self.assertTrue((self.test_preds[f'{mcode}_tahmin_min'] <= self.test_preds[f'{mcode}_tahmin_max']).all())

    def test_08_purchasing_calculation_accuracy(self) -> None:
        """8. Satın alma hesaplama doğruluğu (minimum, maksimum, aritmetik_orta ve bütçe)."""
        active_sid = str(self.latest_features.iloc[0]['SeriId'])
        for ptype in ('minimum', 'maksimum', 'aritmetik_orta'):
            sim = self.pds.simulate_single_purchase(
                series_id=active_sid,
                quantity=125.0,
                price_type=ptype,
                model_code='random_forest',
                budget=5000.0,
            )
            self.assertAlmostEqual(sim['mevcut_toplam_maliyet_tl'], sim['mevcut_birim_fiyat_tl'] * 125.0, places=6)
            self.assertAlmostEqual(sim['tahmini_toplam_maliyet_tl'], sim['tahmini_birim_fiyat_tl'] * 125.0, places=6)
            self.assertAlmostEqual(
                sim['toplam_maliyet_farki_tl'],
                sim['tahmini_toplam_maliyet_tl'] - sim['mevcut_toplam_maliyet_tl'],
                places=6,
            )
            expected_pct = (sim['tahmini_birim_fiyat_tl'] - sim['mevcut_birim_fiyat_tl']) / sim['mevcut_birim_fiyat_tl'] * 100.0
            self.assertAlmostEqual(sim['degisim_orani_yuzde'], expected_pct, places=6)
            self.assertTrue(sim['butce_analizi']['butce_girildi'])
            self.assertAlmostEqual(
                sim['butce_analizi']['tahmini_butce_farki_tl'],
                5000.0 - sim['tahmini_toplam_maliyet_tl'],
                places=6,
            )

    def test_09_multi_product_basket_simulation(self) -> None:
        """9. Çoklu ürün sepeti simülasyonu doğruluğu ve birim uyarısı."""
        sids = self.latest_features['SeriId'].head(4).tolist()
        items = [{'seri_id': str(sids[0]), 'miktar': 50.0}, {'seri_id': str(sids[1]), 'miktar': 80.0}, {'seri_id': str(sids[2]), 'miktar': 120.0}]
        basket = self.pds.simulate_basket_purchase(
            basket_items=items,
            price_type='aritmetik_orta',
            model_code='random_forest',
            total_budget=20000.0,
        )
        self.assertEqual(basket['kalem_sayisi'], 3)
        sum_cur = sum(k['mevcut_toplam_maliyet_tl'] for k in basket['kalemler'])
        sum_pred = sum(k['tahmini_toplam_maliyet_tl'] for k in basket['kalemler'])
        self.assertAlmostEqual(basket['toplam_mevcut_maliyet_tl'], sum_cur, places=6)
        self.assertAlmostEqual(basket['toplam_tahmini_maliyet_tl'], sum_pred, places=6)
        self.assertAlmostEqual(basket['toplam_maliyet_farki_tl'], sum_pred - sum_cur, places=6)

        # Aynı ürün serisi sepette iki kez verilirse hata üretmelidir
        with self.assertRaises(ValueError):
            self.pds.simulate_basket_purchase(
                basket_items=[{'seri_id': str(sids[0]), 'miktar': 10.0}, {'seri_id': str(sids[0]), 'miktar': 20.0}],
            )

    def test_10_invalid_user_inputs_rejected(self) -> None:
        """10. Hatalı kullanıcı girişlerinin engellenmesi (miktar <= 0, NaN, inf, negatif bütçe vb.)."""
        sid = str(self.latest_features.iloc[0]['SeriId'])
        for bad_qty in (0, -10, float('nan'), float('inf'), 'abc', None, True):
            with self.assertRaises(ValueError):
                self.pds.simulate_single_purchase(sid, quantity=bad_qty)  # type: ignore[arg-type]

        for bad_budget in (0, -500, float('nan'), float('inf'), 'xyz'):
            with self.assertRaises(ValueError):
                self.pds.simulate_single_purchase(sid, quantity=10.0, budget=bad_budget)  # type: ignore[arg-type]

        with self.assertRaises(ValueError):
            self.pds.simulate_single_purchase(sid, quantity=10.0, price_type='gecersiz_fiyat_turu')
        with self.assertRaises(ValueError):
            self.pds.simulate_single_purchase(sid, quantity=10.0, model_code='gecersiz_model')
        with self.assertRaises(ValueError):
            self.pds.simulate_basket_purchase([])

    def test_11_web_app_data_bindings_and_pages(self) -> None:
        """11. Web uygulaması veri bağlantıları ve 6 zorunlu sayfa yapısı kontrolü."""
        from streamlit.testing.v1 import AppTest

        self.assertEqual(len(PAGES), 6)
        self.assertEqual(verify_required_app_files(ROOT), [])
        bundle = load_dashboard_bundle()
        self.assertEqual(len(bundle['clean_df']), 380985)
        self.assertEqual(len(bundle['latest_features_df']), 91)
        self.assertEqual(len(bundle['series_meta']), 299)
        self.assertEqual(len(bundle['main_comp_df']), 16)

        at = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30)
        at.run()
        self.assertFalse(at.exception, f'Ana sayfa yükleme hatası: {at.exception}')
        for page in PAGES:
            at.sidebar.radio[0].set_value(page).run()
            self.assertFalse(at.exception, f'{page} sayfasında çalışma hatası: {at.exception}')

    def test_12_missing_or_corrupted_files_raise_clear_errors(self) -> None:
        """12. Eksik model veya veri dosyası durumunda anlamlı hata üretimi."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            missing_list = verify_required_app_files(tmp_path)
            self.assertGreater(len(missing_list), 0)

            with self.assertRaises(FileNotFoundError):
                HalPredictionService(tmp_path)
            with self.assertRaises(FileNotFoundError):
                PurchasingDecisionSupport(tmp_path, tmp_path)

            # Bozuk SHA-256 özeti kontrolü
            fake_asama4 = tmp_path / 'asama4'
            fake_models = fake_asama4 / 'modeller'
            fake_models.mkdir(parents=True, exist_ok=True)
            reg = json.loads((ROOT / 'ciktilar/asama4/model_kayit_defteri.json').read_text(encoding='utf-8'))
            for entry in reg['modeller']:
                shutil.copy2(ROOT / 'ciktilar/asama4/modeller' / entry['dosya_adi'], fake_models / entry['dosya_adi'])
            reg['modeller'][0]['sha256'] = '0' * 64
            (fake_asama4 / 'model_kayit_defteri.json').write_text(json.dumps(reg), encoding='utf-8')
            with self.assertRaises(ValueError):
                HalPredictionService(fake_asama4)


if __name__ == '__main__':
    unittest.main()
