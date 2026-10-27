"""Kaynak CSV dosyalarını değiştirmeden proje uygunluğunu inceler."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'inceleme'
OUT.mkdir(exist_ok=True)
DESKTOP = Path('/Users/burakdemirel/Desktop')
FILES = {'meyve_sebze': DESKTOP / 'IBB_Hal_Meyve_Sebze_2016_2026.csv',
         'tum_urunler': DESKTOP / 'IBB_Hal_Tum_Urunler_2016_2026.csv'}
KEY = ['Tarih', 'TabelaGId', 'BirimTurId', 'HalTurId']
SERIES = ['TabelaGId', 'BirimTurId', 'HalTurId']
summary = {}
frames = {}
for name, path in FILES.items():
    d = pd.read_csv(path)
    frames[name] = d
    date = pd.to_datetime(d.Tarih, errors='coerce')
    bad = (d.EnDusukFiyat <= 0) | (d.EnYuksekFiyat <= 0) | (d.EnDusukFiyat > d.EnYuksekFiyat)
    dup = d.duplicated(KEY, keep=False)
    summary[name] = {
        'satir': len(d), 'sutun': len(d.columns), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'ilk_tarih': str(date.min().date()), 'son_tarih': str(date.max().date()),
        'gun': date.nunique(), 'urun_kimligi': d.TabelaGId.nunique(), 'urun_adi': d.UrunAd.nunique(),
        'bos_hucre': d.isna().sum().to_dict(), 'gecersiz_tarih': int(date.isna().sum()),
        'tarih_uyusmazligi': int((d.Tarih != d.GuneAit.str[:10]).sum()),
        'tam_tekrar_fazlaligi': int(d.duplicated().sum()), 'anahtar_tekrar_fazlaligi': int(d.duplicated(KEY).sum()),
        'tekrar_grubundaki_satir': int(dup.sum()),
        'sifir_fiyat': int(((d.EnDusukFiyat == 0) | (d.EnYuksekFiyat == 0)).sum()),
        'negatif_fiyat': int(((d.EnDusukFiyat < 0) | (d.EnYuksekFiyat < 0)).sum()),
        'ters_aralik': int((d.EnDusukFiyat > d.EnYuksekFiyat).sum()),
        'esit_alt_ust': int((d.EnDusukFiyat == d.EnYuksekFiyat).sum()),
        'kategori': d.KategoriId.value_counts().sort_index().to_dict(),
        'hal_turu': d.HalTurId.value_counts().sort_index().to_dict(),
        'birim': d.BirimTurId.value_counts().sort_index().to_dict(),
        'fiyat_ozeti': d[['EnDusukFiyat', 'EnYuksekFiyat']].describe(percentiles=[.01,.5,.99]).to_dict(),
    }
    d.loc[bad].to_csv(OUT / f'{name}_supheli_fiyatlar.csv', index=False, encoding='utf-8-sig')
    d.loc[dup].to_csv(OUT / f'{name}_tekrarlanan_anahtarlar.csv', index=False, encoding='utf-8-sig')
    annual = d.assign(Yil=date.dt.year).groupby('Yil').agg(Satir=('Tarih','size'), Gun=('Tarih','nunique'), Urun=('TabelaGId','nunique'))
    annual.to_csv(OUT / f'{name}_yillik_kapsam.csv', encoding='utf-8-sig')

d = frames['meyve_sebze']
a = frames['tum_urunler']
subset = a[a.HalTurId == 2].reset_index(drop=True)
summary['dosya_iliskisi'] = {'filtrelenmis_tam_esitlik': d.equals(subset), 'tum_dosya_hal2_satir': len(subset)}
for name, original in [('meyve_sebze', ROOT/'veriler/hal_meyve_sebze_2016_2026.csv'), ('tum_urunler',ROOT/'veriler/hal_tum_urunler_2016_2026.csv')]:
    summary[name]['yerel_kaynakla_hash_esit'] = hashlib.sha256(original.read_bytes()).hexdigest() == summary[name]['sha256']

names = d.groupby('TabelaGId').agg(AdSayisi=('UrunAd','nunique'), Adlar=('UrunAd',lambda x:' / '.join(sorted(set(x)))), Birimler=('BirimTurId',lambda x:','.join(map(str, sorted(set(x))))), Kategoriler=('KategoriId',lambda x:','.join(map(str, sorted(set(x))))))
names.to_csv(OUT/'urun_kimlik_ad_birim.csv', encoding='utf-8-sig')
summary['kimlik_sorunlari'] = {'cok_adli_kimlik': int((names.AdSayisi > 1).sum()), 'cok_birimli_kimlik': int(names.Birimler.str.contains(',').sum()), 'cok_kimlikli_ad': int((d.groupby('UrunAd').TabelaGId.nunique() > 1).sum()), 'cok_kategorili_kimlik': int(names.Kategoriler.str.contains(',').sum())}

days = pd.read_csv(ROOT/'veriler/gunluk_erisim_durumu.csv')
summary['erisim'] = {'durumlar': days.Durum.value_counts().to_dict(), 'ilk_sorgu': days.Tarih.min(), 'son_sorgu': days.Tarih.max()}
missing = days[days.MeyveSebzeSatir.fillna(0) == 0].copy()
missing['HaftaGunu'] = pd.to_datetime(missing.Tarih).dt.day_name()
missing.to_csv(OUT/'meyve_sebze_olmayan_gunler.csv', index=False, encoding='utf-8-sig')
summary['erisim']['meyve_sebze_olmayan_gun_hafta_dagilimi'] = missing.HaftaGunu.value_counts().to_dict()

# Çelişkili anahtarlar ve geçersiz fiyatlar profil hesabından çıkarılır; kaynak korunur.
clean = d.drop_duplicates().copy()
conflict = clean.duplicated(KEY, keep=False)
summary['profil_dislamalari'] = {'tam_tekrar_fazlaligi':len(d)-len(clean), 'celiskili_anahtar_satiri':int(conflict.sum())}
clean = clean.loc[~conflict & (clean.EnDusukFiyat > 0) & (clean.EnYuksekFiyat > 0) & (clean.EnDusukFiyat <= clean.EnYuksekFiyat)].copy()
clean['dt'] = pd.to_datetime(clean.Tarih)
clean = clean.sort_values(SERIES+['dt'])
global_days = pd.DatetimeIndex(sorted(pd.to_datetime(d.Tarih.unique())))
next_day = dict(zip(global_days[:-1],global_days[1:]))
cut = global_days.max() - pd.DateOffset(years=1)
valid_cut = cut - pd.DateOffset(years=1)
clean['sonraki_tarih'] = clean.groupby(SERIES).dt.shift(-1)
clean['hedef_min'] = clean.groupby(SERIES).EnDusukFiyat.shift(-1)
clean['hedef_max'] = clean.groupby(SERIES).EnYuksekFiyat.shift(-1)
clean['beklenen_tarih'] = clean.dt.map(next_day)
clean['uygun_hedef'] = clean.sonraki_tarih.eq(clean.beklenen_tarih) & clean.sonraki_tarih.notna()
clean['gun_araligi'] = (clean.sonraki_tarih-clean.dt).dt.days
rows=[]
for key,g in clean.groupby(SERIES):
    t = g[g.uygun_hedef]
    test=t[t.sonraki_tarih > cut]
    valid=t[(t.sonraki_tarih > valid_cut)&(t.sonraki_tarih <= cut)]
    train=t[t.sonraki_tarih <= valid_cut]
    denom = ((global_days >= g.dt.min()) & (global_days <= g.dt.max())).sum()
    rows.append(dict(zip(SERIES,key)) | {'UrunAd':g.UrunAd.iloc[-1], 'KategoriId':int(g.KategoriId.iloc[-1]), 'Kayit':len(g), 'IlkTarih':g.Tarih.min(), 'SonTarih':g.Tarih.max(), 'KapsamaOrani':len(g)/denom, 'EnUzunAraGun':g.gun_araligi.max(), 'GecerliHedef':len(t), 'EgitimHedef':len(train), 'DogrulamaHedef':len(valid), 'TestHedef':len(test), 'SonDonemDevam':g.dt.max()==global_days.max(), 'TestAyniMinOran':float((test.hedef_min==test.EnDusukFiyat).mean()), 'TestAyniMaxOran':float((test.hedef_max==test.EnYuksekFiyat).mean()), 'BaslangicMinMAE':float((test.hedef_min-test.EnDusukFiyat).abs().mean()), 'BaslangicMaxMAE':float((test.hedef_max-test.EnYuksekFiyat).abs().mean())})
profile=pd.DataFrame(rows).sort_values(['TestHedef','Kayit'], ascending=False)
profile.to_csv(OUT/'urun_serisi_uygunlugu.csv',index=False,encoding='utf-8-sig')
candidates=profile[(profile.EgitimHedef>=500)&(profile.DogrulamaHedef>=180)&(profile.TestHedef>=180)&profile.SonDonemDevam&(profile.KapsamaOrani>=.85)]
candidates.to_csv(OUT/'surekli_urun_adaylari.csv',index=False,encoding='utf-8-sig')
targets=clean[clean.uygun_hedef]
test=targets[targets.sonraki_tarih>cut]
summary['hedef']={'profil_satir':len(clean), 'gecerli_sonraki_islem_gunu_cifti':len(targets), 'sonraki_kayit_ama_sonraki_genel_gun_degil':int((clean.sonraki_tarih.notna()&~clean.uygun_hedef).sum()), 'test_hedef_tarihi':f'({cut.date()}, {global_days.max().date()}]', 'dogrulama_hedef_tarihi':f'({valid_cut.date()}, {cut.date()}]', 'test_cifti':len(test), 'aday_seri':len(candidates), 'test_ayni_min_oran':float((test.hedef_min==test.EnDusukFiyat).mean()), 'test_ayni_max_oran':float((test.hedef_max==test.EnYuksekFiyat).mean())}
summary['ilk_adaylar']=candidates.head(20).replace({np.nan:None}).to_dict('records')
summary['ad_degisenler']=names[names.AdSayisi>1].reset_index().to_dict('records')
# Ani değişimler hata hükmü değildir; inceleme listesi olarak saklanır.
for col in ['EnDusukFiyat','EnYuksekFiyat']:
    clean[col+'_onceki']=clean.groupby(SERIES)[col].shift(1)
ratio=clean.EnYuksekFiyat/clean.EnYuksekFiyat_onceki
jumps=clean[(ratio>=3)|(ratio<=1/3)]
jumps.to_csv(OUT/'buyuk_fiyat_degisimleri.csv',index=False,encoding='utf-8-sig')
summary['uc_kontrol']={'en_az_3_kat_degisim_satiri':len(jumps)}
daily=d.groupby('Tarih').agg(Satir=('TabelaGId','size'),Urun=('TabelaGId','nunique'))
daily.to_csv(OUT/'gunluk_urun_kapsami.csv',encoding='utf-8-sig')
summary['seyrek_gunler']=daily.nsmallest(10,'Urun').reset_index().to_dict('records')
summary['metin']={'bas_son_bosluk_satiri':int(d.UrunAd.ne(d.UrunAd.str.strip()).sum())}
with (OUT/'inceleme_ozeti.json').open('w') as f:
    json.dump(summary,f,ensure_ascii=False,indent=2,default=lambda x:int(x) if isinstance(x,np.integer) else str(x),allow_nan=False)
print(json.dumps(summary,ensure_ascii=False,indent=2,default=str))
