"""Kaydedilen gerçek CSV'leri kaynağa karşı bağımsız doğrular; temizlik çalıştırmaz."""
from pathlib import Path
import hashlib
import json
import unicodedata

import numpy as np
import pandas as pd


def file_hash(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def verify(output):
    output=Path(output)
    manifest=json.loads((output/'kaynak_manifestosu.json').read_text())
    summary=json.loads((output/'ozet.json').read_text())
    contract=json.loads((output/'veri_sozlesmesi.json').read_text())
    checks=[]
    for path,expected in manifest['kaynak_once'].items():
        assert file_hash(path)==expected==manifest['kaynak_sonra'][path], path
    checks.append('İki orijinal kaynağın öncesi/sonrası/güncel SHA-256 özetleri aynı.')
    for filename,expected in manifest['cikti_sha256'].items():
        assert file_hash(output/filename)==expected,filename
    checks.append('Manifestodaki bütün çıktı dosyalarının SHA-256 özetleri aynı.')
    sources=list(manifest['kaynak_once'])
    raw=pd.read_csv(sources[0],dtype=str,keep_default_na=False)
    clean=pd.read_csv(output/'temiz_veri.csv',keep_default_na=False)
    rejected=pd.read_csv(output/'dislanan_kayitlar.csv',dtype=str,keep_default_na=False)
    kept_ids=clean.KaynakKayitNo.to_numpy(dtype=int)
    rejected_ids=rejected.KaynakKayitNo.to_numpy(dtype=int)
    all_ids=np.concatenate([kept_ids,rejected_ids])
    assert np.array_equal(np.sort(all_ids),np.arange(1,len(raw)+1))
    assert len(np.unique(all_ids))==len(raw)
    assert len(clean)==summary['temizleme']['temiz_kayit']
    checks.append('Her ham kayıt temiz veride veya dışlama dosyasında tam bir kez bulunuyor.')
    originals=raw.iloc[kept_ids-1].reset_index(drop=True)
    for name in ['EnDusukFiyat','EnYuksekFiyat']:
        np.testing.assert_array_equal(clean[name].to_numpy(),pd.to_numeric(originals[name]).to_numpy())
    assert originals.Tarih.equals(clean.Tarih)
    assert originals.GuneAit.equals(clean.GuneAit)
    assert originals.UrunAd.equals(clean.UrunAdHam)
    for name in ['BirimTurId','HalTurId','KategoriId']:
        np.testing.assert_array_equal(clean[name],pd.to_numeric(originals[name]))
    normalized=originals.UrunAd.map(lambda x:unicodedata.normalize('NFC',' '.join(x.split())))
    assert normalized.equals(clean.UrunAd)
    assert originals.TabelaGId.str.strip().str.lower().equals(clean.TabelaGId)
    checks.append('Temiz fiyatlar, tarihler, kodlar ve özgün adlar kaynak kayıtlarıyla eşleşiyor.')
    pd.testing.assert_frame_equal(rejected[raw.columns].reset_index(drop=True),raw.iloc[rejected_ids-1].reset_index(drop=True))
    checks.append('Karantina/tekrar dosyasındaki bütün özgün alanlar kaynağa eşit.')
    assert np.isfinite(clean[['EnDusukFiyat','EnYuksekFiyat']].to_numpy()).all()
    assert clean.EnDusukFiyat.gt(0).all()
    assert clean.EnYuksekFiyat.ge(clean.EnDusukFiyat).all()
    assert not clean.duplicated(['Tarih','TabelaGId','BirimTurId','HalTurId']).any()
    assert clean.groupby('SeriId')[['TabelaGId','BirimTurId','HalTurId','KategoriId','UrunAd']].nunique().eq(1).all().all()
    assert not clean.columns.str.lower().str.startswith(('hedef_','lag_','rolling_')).any()
    checks.append('Fiyat, benzersiz anahtar, seri kimliği/birimi/adı ve aşama sınırı kontrolleri geçti.')
    review=pd.read_csv(output/'inceleme_bekleyen.csv',keep_default_na=False)
    assert set(review.KaynakKayitNo)==set(clean.loc[clean.IncelemeGerekli,'KaynakKayitNo'])
    assert len(review)==summary['temizleme']['inceleme_isaretli_kayit']
    assert clean.loc[clean.IncelemeGerekli,'IncelemeNedeni'].ne('').all()
    assert clean.loc[~clean.IncelemeGerekli,'IncelemeNedeni'].eq('').all()
    special=clean.loc[clean.Tarih.eq('2017-10-05') & clean.UrunAd.eq('Biber (Renkli)')]
    assert len(special)==1 and special.EnYuksekFiyat.iloc[0]==10200 and special.IncelemeGerekli.iloc[0]
    checks.append('İnceleme işaretleri/listesi tutarlı; örnek uç fiyat korunmuş.')
    calendar=pd.read_csv(output/'veri_gunleri.csv')
    assert set(calendar.Tarih)==set(raw.Tarih)
    assert calendar.HamKayit.sum()==len(raw)
    assert calendar.TemizKayit.sum()==len(clean)
    annual=pd.read_csv(output/'yillik_kapsam.csv')
    assert annual.HamKayit.sum()==len(raw) and annual.TemizKayit.sum()==len(clean)
    start=pd.Timestamp(raw.Tarih.max())-pd.DateOffset(years=1)+pd.Timedelta(days=1)
    assert contract['test_donemi']['baslangic_dahil']==str(start.date())
    assert contract['test_donemi']['bitis_dahil']==raw.Tarih.max()
    assert contract['harici_veri_kullanildi'] is False
    checks.append('Ham gün takvimi, yıllık toplamlar ve son 12 ay sınırı bağımsız doğrulandı.')
    result={'durum':'basarili','kontroller':checks,'temiz_kayit':len(clean),'inceleme_kayit':len(review)}
    # Sonuç çıktı klasörünün dışında tutulur; veri manifestosu döngüye girmez.
    (output.parent/'asama1_dogrulama.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result


if __name__=='__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent/'ciktilar/asama1'),ensure_ascii=False,indent=2))
