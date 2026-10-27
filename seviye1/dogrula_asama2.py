"""Kaydedilen bütün Aşama 2 örneklerini bağımsız NumPy hesaplarıyla denetler.

Özellik üreticisini çağırmaz. Kaynak fiyatlar, geçmiş pencereleri, hedefler,
bölümler, son gün girdileri ve satır mutabakatını gerçek dosyalardan kontrol eder.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def verify(output: Path) -> dict:
    output=Path(output)
    manifest=json.loads((output/'kaynak_manifestosu.json').read_text())
    contract=json.loads((output/'ozellik_sozlesmesi.json').read_text())
    summary=json.loads((output/'ozet.json').read_text())
    checks=[]
    for before,after in [('girdi_once','girdi_sonra'),('orijinal_csv_once','orijinal_csv_sonra')]:
        for path,expected in manifest[before].items():
            assert digest(path)==expected==manifest[after][path],path
    for name,expected in manifest['cikti_sha256'].items():
        assert digest(output/name)==expected,name
    assert digest(ROOT/'asama2_ozellikler.py')==manifest['kod_sha256']
    checks.append('Aşama 1/orijinal kaynaklar değişmemiş; bütün Aşama 2 dosyalarının hashleri eşleşiyor.')
    source=Path(manifest['asama1_dizini'])
    clean=pd.read_csv(source/'temiz_veri.csv',keep_default_na=False,float_precision='round_trip')
    clean['Tarih']=pd.to_datetime(clean.Tarih)
    calendar=pd.DatetimeIndex(pd.to_datetime(pd.read_csv(source/'veri_gunleri.csv').Tarih))
    parts={name:pd.read_csv(output/f'{name}.csv.gz',keep_default_na=False,float_precision='round_trip')
           for name in ['egitim','dogrulama','test']}
    frames=[]
    bounds=contract['sinirlar']
    for name,frame in parts.items():
        assert frame.Bolum.eq(name).all()
        for col in ['Tarih','HedefTarih','OzellikEnEskiTarih','OzellikEnYeniTarih']:
            frame[col]=pd.to_datetime(frame[col])
        if name=='egitim':
            assert frame.HedefTarih.le(bounds['egitim_son_dahil']).all()
        elif name=='dogrulama':
            assert frame.HedefTarih.between(bounds['dogrulama_ilk_dahil'],bounds['dogrulama_son_dahil']).all()
        else:
            assert frame.HedefTarih.between(bounds['test_ilk_dahil'],bounds['test_son_dahil']).all()
        assert len(frame)==summary['bolumler'][name]['kayit']
        frames.append(frame)
    samples=pd.concat(frames,ignore_index=True)
    assert samples.KaynakKayitNo.is_unique
    for i,left in enumerate(parts.values()):
        for right in list(parts.values())[i+1:]:
            assert set(left.HedefTarih).isdisjoint(right.HedefTarih)
    for earlier,later in [('egitim','dogrulama'),('dogrulama','test')]:
        if len(parts[earlier]) and len(parts[later]):
            assert parts[earlier].HedefTarih.max() <= parts[later].Tarih.min()
    assert bounds['test_ilk_dahil']==str((calendar.max()-pd.DateOffset(years=1)+pd.Timedelta(days=1)).date())
    assert bounds['test_son_dahil']==str(calendar.max().date())
    checks.append('Hedef tarihlerine göre ayrım, bölüm ayrıklığı ve öğrenme anındaki etiket erişilebilirliği doğru.')
    source_by_id=clean.set_index('KaynakKayitNo')
    origins=source_by_id.loc[samples.KaynakKayitNo].reset_index(drop=True)
    labels=source_by_id.loc[samples.HedefKaynakKayitNo].reset_index(drop=True)
    for col in ['TabelaGId','SeriId','BirimTurId','KategoriId','HalTurId','UrunAd','AdDonemi']:
        np.testing.assert_array_equal(origins[col],samples[col])
        np.testing.assert_array_equal(labels[col],samples[col])
    np.testing.assert_array_equal(origins.Tarih,samples.Tarih)
    np.testing.assert_array_equal(labels.Tarih,samples.HedefTarih)
    calendar_positions=calendar.get_indexer(samples.Tarih)
    assert (calendar_positions>=0).all() and (calendar_positions<len(calendar)-1).all()
    np.testing.assert_array_equal(calendar[calendar_positions+1],samples.HedefTarih)
    np.testing.assert_array_equal(labels.EnDusukFiyat,samples.hedef_min)
    np.testing.assert_array_equal(labels.EnYuksekFiyat,samples.hedef_max)
    np.testing.assert_array_equal(origins.IncelemeGerekli,samples.IncelemeGerekli)
    np.testing.assert_array_equal(labels.IncelemeGerekli,samples.HedefIncelemeGerekli)
    checks.append('Bütün gerçek hedefler aynı ürün/birim/ad döneminin tam sonraki ortak gününe eşit; uzak kayıt hedef alınmamış.')
    latest=pd.read_csv(output/'son_gun_girdileri.csv',keep_default_na=False,float_precision='round_trip')
    for col in ['Tarih','OzellikEnEskiTarih','OzellikEnYeniTarih']:
        latest[col]=pd.to_datetime(latest[col])
    assert latest.Tarih.eq(calendar.max()).all()
    assert not set(contract['hedefler']).intersection(latest.columns)
    tested=pd.concat([samples,latest],ignore_index=True)
    assert tested.KaynakKayitNo.is_unique
    features=contract['ozellikler']
    assert len(features)==54 and len(set(features))==54
    assert not set(features).intersection(contract['metadata']+contract['hedefler'])
    assert np.isfinite(tested[features].to_numpy(dtype=float)).all()
    assert tested.OzellikEnYeniTarih.eq(tested.Tarih).all()
    assert samples.OzellikEnYeniTarih.lt(samples.HedefTarih).all()
    tested_positions=pd.Series(np.arange(len(tested)),index=tested.KaynakKayitNo)
    history_ready_ids=set()
    max_abs_diff={}
    # Her tam geçmiş penceresini kaynak dizileri üzerinden bağımsız oluştur.
    for _,group in clean.groupby('SeriId',sort=False):
        group=group.sort_values('Tarih').reset_index(drop=True)
        day_position=calendar.get_indexer(group.Tarih)
        breaks=np.flatnonzero(np.diff(day_position)!=1)+1
        for index_block in np.split(np.arange(len(group)),breaks):
            if len(index_block)<=20:
                continue
            block=group.iloc[index_block].reset_index(drop=True)
            valid_pos=np.arange(20,len(block))
            history_ready_ids.update(block.KaynakKayitNo.iloc[valid_pos])
            target_positions=block.KaynakKayitNo.iloc[valid_pos].map(tested_positions)
            present=target_positions.notna().to_numpy()
            pos=valid_pos[present]
            if not len(pos):
                continue
            actual=tested.iloc[target_positions.loc[target_positions.notna()].astype(int).to_numpy()]
            hist_index=pos[:,None]-np.arange(20,-1,-1)[None,:]
            lower=block.EnDusukFiyat.to_numpy(dtype=float)
            upper=block.EnYuksekFiyat.to_numpy(dtype=float)
            expected={'fiyat_min':lower[pos],'fiyat_max':upper[pos],
                      'fiyat_orta':(lower[pos]+upper[pos])/2,'fiyat_araligi':upper[pos]-lower[pos]}
            expected['goreli_fiyat_araligi']=expected['fiyat_araligi']/expected['fiyat_orta']
            for side,array in [('min',lower),('max',upper)]:
                for lag in [1,2,5,10,20]:
                    expected[f'{side}_lag_{lag}']=array[pos-lag]
                for window in [5,10,20]:
                    values=array[hist_index[:,-window:]]
                    for stat in ['mean','std','min','max']:
                        expected[f'{side}_rolling_{window}_{stat}']=getattr(values,stat)(axis=1)
                expected[f'{side}_fark_1']=array[pos]-array[pos-1]
                expected[f'{side}_degisim_orani_1']=array[pos]/array[pos-1]-1
                expected[f'{side}_degisim_orani_5']=array[pos]/array[pos-5]-1
            dates=pd.DatetimeIndex(block.Tarih.iloc[pos])
            expected.update(tarih_yil=dates.year,tarih_ay=dates.month,tarih_hafta_gunu=dates.dayofweek,
                tarih_yil_gunu=dates.dayofyear,tarih_ay_sin=np.sin(2*np.pi*(dates.month-1)/12),
                tarih_ay_cos=np.cos(2*np.pi*(dates.month-1)/12),
                tarih_yil_sin=np.sin(2*np.pi*(dates.dayofyear-1)/365.25),
                tarih_yil_cos=np.cos(2*np.pi*(dates.dayofyear-1)/365.25),
                onceki_gozlemden_gun=(block.Tarih.iloc[pos].to_numpy()-block.Tarih.iloc[pos-1].to_numpy())/np.timedelta64(1,'D'))
            assert set(expected)==set(features)
            for feature,values in expected.items():
                observed=actual[feature].to_numpy(dtype=float)
                # Rolling Welford (Pandas) ve doğrudan NumPy varyansı sayısal olarak az farklı olabilir.
                atol=1e-5 if feature.endswith('_std') else 1e-9
                np.testing.assert_allclose(observed,np.asarray(values),rtol=1e-9,atol=atol,err_msg=feature)
                max_abs_diff[feature]=max(max_abs_diff.get(feature,0),float(np.max(np.abs(observed-values))))
            np.testing.assert_array_equal(actual.OzellikEnEskiTarih,block.Tarih.iloc[pos-20])
            flags=block.IncelemeGerekli.to_numpy()[hist_index].sum(axis=1)
            np.testing.assert_array_equal(actual.GecmisteIncelemeSayisi,flags)
    checks.append(f'{len(tested):,} kayıt için 54 özelliğin tamamı kaynak geçmişinden bağımsız hesaplandı ve karşılaştırıldı.')
    assert set(tested.KaynakKayitNo).issubset(history_ready_ids)
    next_map=pd.Series(calendar[1:],index=calendar[:-1])
    proposed=pd.MultiIndex.from_arrays([clean.SeriId,clean.Tarih.map(next_map)])
    known=pd.MultiIndex.from_frame(clean[['SeriId','Tarih']])
    expected_samples=set(clean.loc[clean.KaynakKayitNo.isin(history_ready_ids)&proposed.isin(known),'KaynakKayitNo'])
    assert set(samples.KaynakKayitNo)==expected_samples
    expected_latest=set(clean.loc[clean.KaynakKayitNo.isin(history_ready_ids)&clean.Tarih.eq(calendar.max()),'KaynakKayitNo'])
    assert set(latest.KaynakKayitNo)==expected_latest
    excluded=pd.read_csv(output/'dislanan_ornekler.csv',keep_default_na=False)
    ids=np.concatenate([samples.KaynakKayitNo.to_numpy(),excluded.KaynakKayitNo.to_numpy()])
    np.testing.assert_array_equal(np.sort(ids),np.sort(clean.KaynakKayitNo))
    assert len(np.unique(ids))==len(clean)
    assert excluded.DislamaNedeni.value_counts().to_dict()==summary['dislama_nedenleri']
    checks.append('Kabul/dışlama kümesi, bütün temiz kayıtların mutabakatı ve son gün uygunluğu bağımsız doğrulandı.')
    assert summary['kullanilabilir_ornek']==len(samples)
    assert contract['model_egitildi'] is False and contract['harici_veri'] is False
    result={'durum':'basarili','kontroller':checks,'kontrol_edilen_etiketli_ornek':len(samples),
            'kontrol_edilen_son_gun_girdisi':len(latest),'ozellik_sayisi':len(features),
            'azami_std_sayisal_farki':max(v for k,v in max_abs_diff.items() if k.endswith('_std'))}
    (output.parent/'asama2_dogrulama.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result


if __name__=='__main__':
    print(json.dumps(verify(ROOT/'ciktilar/asama2'),ensure_ascii=False,indent=2))
