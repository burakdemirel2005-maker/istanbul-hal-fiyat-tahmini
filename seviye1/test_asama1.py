"""Temizliğin veri kaybı, yanlış birleştirme ve geleceğe bakma risklerini sınar."""
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from asama1_veri import COLUMNS, clean_data, run


def row(day='2024-01-02', name='Domates', unit='1', low='10', high='20', **changes):
    result = dict(zip(COLUMNS, [day,'d685a9bf-9598-43fc-a5bf-42dafd586a98',name,
                              unit,'6',low,high,f'{day}T00:00:00','2']))
    result.update(changes)
    return result


def clean(rows):
    return clean_data(pd.DataFrame(rows, columns=COLUMNS), '2026-10-27')


class CleaningTests(unittest.TestCase):
    def test_exact_duplicate_accounting_and_input_preserved(self):
        raw=pd.DataFrame([row(),row(),row('2024-01-03')],columns=COLUMNS)
        original=raw.copy(deep=True)
        kept,rejected,stats=clean_data(raw,'2026-10-27')
        pd.testing.assert_frame_equal(raw,original)
        self.assertEqual((len(kept),len(rejected),stats['tam_tekrar_fazlaligi']),(2,1,1))
        self.assertEqual(set(kept.KaynakKayitNo)|set(rejected.KaynakKayitNo),{1,2,3})

    def test_invalid_prices_are_quarantined_without_repair(self):
        data=[row(),row('2024-01-03',low='0'),row('2024-01-04',low='-1'),
              row('2024-01-05',low='30',high='20'),row('2024-01-06',high='inf'),
              row('2024-01-07',low='NaN'),row('2024-01-08',high='abc')]
        kept,rejected,_=clean(data)
        self.assertEqual((len(kept),len(rejected)),(1,6))
        self.assertEqual(rejected.loc[rejected.KaynakKayitNo.eq(4),'EnDusukFiyat'].iloc[0],'30')

    def test_conflicting_key_quarantines_both_prices(self):
        kept,rejected,_=clean([row(),row(low='11'),row('2024-01-03')])
        self.assertEqual((len(kept),len(rejected)),(1,2))
        self.assertTrue(rejected.DislamaNedeni.str.contains('celiskili_gozlem_anahtari').all())

    def test_units_are_never_merged(self):
        kept,rejected,_=clean([row(),row(unit='3')])
        self.assertEqual(len(rejected),0)
        self.assertEqual(kept.SeriId.nunique(),2)
        self.assertTrue(kept.BirimAciklamasi.str.contains('doğrulanmadı').all())

    def test_name_periods_split_even_when_old_name_returns(self):
        kept,_,_=clean([row(name='Hurma'),row('2024-01-03',name='Trabzon Hurması'),
                        row('2024-01-04',name='Hurma')])
        self.assertEqual(kept.AdDonemi.tolist(),[1,2,3])
        self.assertEqual(kept.SeriId.nunique(),3)

    def test_appending_future_does_not_change_past_names_or_flags(self):
        past=[row(),row('2024-01-03',low='12',high='22')]
        first,_,_=clean(past)
        extended,_,_=clean(past+[row('2025-01-01',name='Yeni ürün',low='1000',high='2000')])
        pd.testing.assert_frame_equal(first,extended.iloc[:2].reset_index(drop=True))

    def test_whitespace_normalization_preserves_raw_and_removes_equivalent_copy(self):
        kept,rejected,stats=clean([row(name='  Domates  Salkım '),row(name='Domates Salkım')])
        self.assertEqual(kept.UrunAd.iloc[0],'Domates Salkım')
        self.assertEqual(kept.UrunAdHam.iloc[0],'  Domates  Salkım ')
        self.assertEqual(stats['normalizasyon_sonrasi_tekrar'],1)
        self.assertEqual(len(rejected),1)

    def test_invalid_dates_codes_and_identifiers(self):
        items=[row(),row('2024-02-30'),row('2027-01-01'),
               row('2024-01-04',GuneAit='2024-01-05T00:00:00'),
               row('2024-01-06',unit='1.5'),row('2024-01-07',TabelaGId='hatalı'),
               row('2024-01-08',KategoriId='2'),row('2024-01-09',HalTurId='1'),
               row('2024-01-10',name='  ')]
        kept,rejected,_=clean(items)
        self.assertEqual((len(kept),len(rejected)),(1,8))

    def test_equal_positive_bounds_are_valid(self):
        kept,rejected,_=clean([row(low='10',high='10')])
        self.assertEqual((len(kept),len(rejected)),(1,0))

    def test_suspicious_price_is_flagged_and_not_deleted_or_changed(self):
        kept,rejected,_=clean([row(low='9',high='10200')])
        self.assertEqual(len(rejected),0)
        self.assertTrue(kept.IncelemeGerekli.iloc[0])
        self.assertEqual(kept.EnYuksekFiyat.iloc[0],10200)

    def test_missing_days_are_not_filled(self):
        kept,_,_=clean([row('2024-01-02'),row('2024-01-10')])
        self.assertEqual(kept.Tarih.tolist(),['2024-01-02','2024-01-10'])
        self.assertFalse(any(c.lower().startswith(('hedef_', 'lag_', 'rolling_')) for c in kept.columns))

    def test_mismatched_comparison_aborts_without_outputs_or_source_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            main,other=root/'ana.csv',root/'tum.csv'
            pd.DataFrame([row()],columns=COLUMNS).to_csv(main,index=False)
            pd.DataFrame([row(low='11')],columns=COLUMNS).to_csv(other,index=False)
            before=(main.read_bytes(),other.read_bytes())
            with self.assertRaisesRegex(ValueError,'eşleşmiyor'):
                run(main,other,root/'out','2026-10-27')
            self.assertFalse((root/'out').exists())
            self.assertEqual(before,(main.read_bytes(),other.read_bytes()))


if __name__=='__main__':
    unittest.main()
