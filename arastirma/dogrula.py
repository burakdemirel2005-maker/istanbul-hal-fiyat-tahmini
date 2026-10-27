"""Araştırma raporunun yerel kanıtlarını kontrol eder; yeni ağ isteği yapmaz."""
from pathlib import Path
import hashlib
import json
import re

ROOT=Path(__file__).resolve().parent
report=ROOT/'WEB_ARASTIRMA_RAPORU.md'
text=report.read_text()
summary=json.loads((ROOT.parent/'inceleme/inceleme_ozeti.json').read_text())
pilot=json.loads((ROOT/'pilot/erisim_denetimi.json').read_text())
assert summary['meyve_sebze']['satir']==382897
assert all(x.get('http')==200 for x in pilot)
assert len(json.loads((ROOT/'pilot/open_meteo.json').read_text())['daily']['time'])==7
nasa=json.loads((ROOT/'pilot/nasa_power.json').read_text())
assert all(len(x)==7 for x in nasa['properties']['parameter'].values())
extensions={'open_meteo':'json','nasa_power':'json','tcmb_xml':'xml','ibb_categories':'json','ibb_catalog':'html','ibb_license':'html'}
for record in pilot:
    raw=(ROOT/'pilot'/f"{record['kaynak']}.{extensions[record['kaynak']]}").read_bytes()
    assert hashlib.sha256(raw).hexdigest()==record['sha256']
assert text.count('```')==2
links=list(dict.fromkeys(re.findall(r'\]\((https?://[^)]+)\)',text)))
assert len(links)>=25
notes={
 'denetim_tarihi':'2026-10-27',
 'swagger_html':{'dosya':'ibb_swagger.html','http':200,'sha256':hashlib.sha256((ROOT/'pilot/ibb_swagger.html').read_bytes()).hexdigest()},
 'swagger_belgesi':{'url':'https://halfiyatlaripublicdata.ibb.gov.tr/swagger/docs/v1','sonuc':'İki istekte RemoteDisconnected; birim sözlüğü doğrulanamadı.'},
 'mdpi_tam_metin':{'url':'https://www.mdpi.com/2071-1050/18/1/503','sonuc':'Web aracı HTTP 429; yayıncı arama dizini metniyle sınırlı inceleme.'},
 'kaynak_baglantilari':links,
 'kapsam':'Bu kontrol dosya bütünlüğünü ve pilot yanıtlarını doğrular; bütün dönem verisinin kalitesini veya tahmin başarısını doğrulamaz.'
}
(ROOT/'pilot/ek_erisim_notlari.json').write_text(json.dumps(notes,ensure_ascii=False,indent=2))
print(f'Rapor: {len(text.split())} kelime; {len(links)} benzersiz kaynak bağlantısı.')
print(f'{len(pilot)} HTTP 200 yanıtının hashleri ve iki hava pilotunun 7 günlük kapsamı doğrulandı.')
