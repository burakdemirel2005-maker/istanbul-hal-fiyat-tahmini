"""Aşama 2'de geleceğe bakma, yanlış hedef ve pencere karışmasını sınar."""
import unittest

import numpy as np
import pandas as pd

from asama2_ozellikler import FEATURES, TARGETS, build_features, attach_targets, split_by_target


def fixture(count=85):
    calendar=pd.bdate_range('2024-01-01',periods=count)
    rows=[]
    for i,date in enumerate(calendar):
        for side,unit in [('urun1',1),('urun2',3)]:
            rows.append({'KaynakKayitNo':len(rows)+1,'Tarih':date.strftime('%Y-%m-%d'),
                         'SeriId':side,'TabelaGId':side,'UrunAd':side,'BirimTurId':unit,
                         'KategoriId':6,'HalTurId':2,'AdDonemi':1,
                         'EnDusukFiyat':10.+i+(1000 if unit==3 else 0),
                         'EnYuksekFiyat':30.+2*i+(1000 if unit==3 else 0),
                         'IncelemeGerekli':False})
    return pd.DataFrame(rows),calendar


class FeatureTests(unittest.TestCase):
    def test_lag_rolling_and_ddof_match_hand_calculation(self):
        raw,calendar=fixture()
        f=build_features(raw,calendar)
        r=f.loc[f.SeriId.eq('urun1') & f.Tarih.eq(calendar[20])].iloc[0]
        self.assertEqual(r.min_lag_20,10)
        self.assertEqual(r.min_lag_1,29)
        self.assertEqual(r.min_rolling_5_mean,np.mean([26,27,28,29,30]))
        self.assertAlmostEqual(r.min_rolling_5_std,np.std([26,27,28,29,30],ddof=0))
        self.assertEqual(r.OzellikEnEskiTarih,calendar[0])
        self.assertEqual(r.OzellikEnYeniTarih,calendar[20])
        self.assertTrue(r.OzellikHazir)

    def test_first_twenty_rows_have_no_complete_history(self):
        raw,calendar=fixture()
        f=build_features(raw,calendar)
        for _,g in f.groupby('SeriId'):
            self.assertFalse(g.iloc[:20].OzellikHazir.any())
            self.assertTrue(g.iloc[20:].OzellikHazir.all())

    def test_target_is_next_global_day_same_series_and_unit(self):
        raw,calendar=fixture()
        d=attach_targets(build_features(raw,calendar),raw,calendar)
        r=d.loc[d.SeriId.eq('urun1') & d.Tarih.eq(calendar[20])].iloc[0]
        self.assertEqual(r.HedefTarih,calendar[21])
        self.assertEqual(r.hedef_min,31)
        self.assertEqual(r.hedef_max,72)
        self.assertEqual(r.min_lag_1,29)
        self.assertFalse(d.loc[d.Tarih.eq(calendar[-1]),'DislamaNedeni'].eq('').any())

    def test_future_price_changes_do_not_affect_past_features(self):
        raw,calendar=fixture()
        first=build_features(raw,calendar)
        future=raw.copy()
        mask=pd.to_datetime(future.Tarih).gt(calendar[40])
        future.loc[mask,['EnDusukFiyat','EnYuksekFiyat']]*=10000
        second=build_features(future,calendar)
        pd.testing.assert_frame_equal(first.loc[first.Tarih.le(calendar[40]),FEATURES],
                                      second.loc[second.Tarih.le(calendar[40]),FEATURES])

    def test_future_calendar_extension_does_not_affect_past_features(self):
        raw,calendar=fixture()
        part=raw.loc[pd.to_datetime(raw.Tarih).le(calendar[50])]
        before=build_features(part,calendar[:51])
        after=build_features(raw,calendar)
        pd.testing.assert_frame_equal(before[FEATURES],after.loc[after.Tarih.le(calendar[50]),FEATURES].reset_index(drop=True))

    def test_missing_product_day_resets_history_and_does_not_jump_target(self):
        raw,calendar=fixture()
        raw=raw.loc[~(raw.SeriId.eq('urun1') & raw.Tarih.eq(str(calendar[25].date())))]
        d=attach_targets(build_features(raw,calendar),raw,calendar)
        g=d.loc[d.SeriId.eq('urun1')].set_index('Tarih')
        self.assertEqual(g.loc[calendar[24],'DislamaNedeni'],'sonraki_veri_gununde_ayni_seri_yok')
        self.assertFalse(g.loc[calendar[26]:calendar[45],'OzellikHazir'].any())
        self.assertTrue(g.loc[calendar[46],'OzellikHazir'])
        self.assertNotEqual(g.loc[calendar[24],'BaglamId'],g.loc[calendar[26],'BaglamId'])

    def test_weekend_does_not_reset_trading_day_history(self):
        raw,calendar=fixture()
        f=build_features(raw,calendar)
        self.assertTrue(f.groupby('SeriId').BaglamId.nunique().eq(1).all())
        monday=f.loc[f.SeriId.eq('urun1') & f.Tarih.eq(calendar[25])].iloc[0]
        self.assertEqual(monday.onceki_gozlemden_gun,3)

    def test_name_period_does_not_share_history_or_target(self):
        raw,calendar=fixture()
        mask=raw.SeriId.eq('urun1') & pd.to_datetime(raw.Tarih).ge(calendar[30])
        raw.loc[mask,['SeriId','UrunAd']]='yeni_ad'
        raw.loc[mask,'AdDonemi']=2
        d=attach_targets(build_features(raw,calendar),raw,calendar)
        self.assertEqual(d.loc[d.SeriId.eq('urun1') & d.Tarih.eq(calendar[29]),'DislamaNedeni'].iloc[0],
                         'sonraki_veri_gununde_ayni_seri_yok')
        self.assertFalse(d.loc[d.SeriId.eq('yeni_ad') & d.Tarih.lt(calendar[50]),'OzellikHazir'].any())

    def test_quality_flags_are_metadata_and_do_not_filter_targets(self):
        raw,calendar=fixture()
        raw.loc[raw.Tarih.eq(str(calendar[21].date())),'IncelemeGerekli']=True
        d=attach_targets(build_features(raw,calendar),raw,calendar)
        r=d.loc[d.SeriId.eq('urun1') & d.Tarih.eq(calendar[20])].iloc[0]
        self.assertTrue(r.HedefIncelemeGerekli)
        self.assertEqual(r.DislamaNedeni,'')
        self.assertFalse(any('inceleme' in c.lower() or 'hedef' in c.lower() for c in FEATURES))
        self.assertTrue(set(TARGETS).isdisjoint(FEATURES))

    def test_split_uses_target_date_not_origin(self):
        raw,calendar=fixture()
        d=attach_targets(build_features(raw,calendar),raw,calendar)
        test_start=str(calendar[60].date())
        parts,_=split_by_target(d,test_start,str(calendar[-1].date()))
        boundary=parts['test'].loc[parts['test'].Tarih.eq(calendar[59])]
        self.assertEqual(len(boundary),2)
        self.assertTrue(boundary.HedefTarih.eq(calendar[60]).all())
        self.assertTrue(parts['dogrulama'].HedefTarih.lt(calendar[60]).all())
        ids=[set(x.KaynakKayitNo) for x in parts.values()]
        self.assertTrue(ids[0].isdisjoint(ids[1]) and ids[0].isdisjoint(ids[2]) and ids[1].isdisjoint(ids[2]))

    def test_latest_features_can_be_built_without_future_prices(self):
        raw,calendar=fixture()
        f=build_features(raw,calendar)
        last=f.loc[f.Tarih.eq(calendar[-1])]
        self.assertEqual(len(last),2)
        self.assertTrue(last.OzellikHazir.all())
        self.assertTrue(np.isfinite(last[FEATURES].to_numpy()).all())

    def test_shuffle_is_harmless_and_input_is_not_mutated(self):
        raw,calendar=fixture()
        before=raw.copy(deep=True)
        a=build_features(raw,calendar)
        b=build_features(raw.sample(frac=1,random_state=42),calendar)
        pd.testing.assert_frame_equal(raw,before)
        pd.testing.assert_frame_equal(a,b)

    def test_duplicate_or_mixed_unit_series_and_bad_price_are_rejected(self):
        raw,calendar=fixture()
        with self.assertRaisesRegex(ValueError,'birden fazla'):
            build_features(pd.concat([raw,raw.iloc[:1]]),calendar)
        mixed=raw.copy()
        mixed.loc[0,'BirimTurId']=99
        with self.assertRaisesRegex(ValueError,'değişiyor'):
            build_features(mixed,calendar)
        invalid=raw.copy()
        invalid.loc[0,'EnDusukFiyat']=0
        with self.assertRaisesRegex(ValueError,'Geçersiz fiyat'):
            build_features(invalid,calendar)


if __name__=='__main__':
    unittest.main()
