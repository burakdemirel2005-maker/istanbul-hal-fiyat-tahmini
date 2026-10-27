"""Az sayıda kamuya açık kaynak isteğiyle erişim ve alan kontrolü yapar."""
from pathlib import Path
import concurrent.futures
import datetime as dt
import hashlib
import json
import urllib.request
import xml.etree.ElementTree as ET

OUT=Path(__file__).resolve().parent/'pilot'
OUT.mkdir(exist_ok=True)
SOURCES={
 'open_meteo': 'https://archive-api.open-meteo.com/v1/archive?latitude=36.89&longitude=30.70&start_date=2016-01-04&end_date=2016-01-10&daily=temperature_2m_mean,temperature_2m_min,temperature_2m_max,precipitation_sum&models=era5&timezone=Europe%2FIstanbul',
 'nasa_power': 'https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M,T2M_MIN,T2M_MAX,PRECTOTCORR&community=AG&longitude=30.70&latitude=36.89&start=20160104&end=20160110&format=JSON&time-standard=UTC',
 'tcmb_xml': 'https://www.tcmb.gov.tr/kurlar/202610/05102026.xml',
 'ibb_categories': 'https://halfiyatlaripublicdata.ibb.gov.tr/api/HalManager/getCategories',
 'ibb_catalog': 'https://data.ibb.gov.tr/tr/dataset/hal-urunleri-ve-fiyatlari-web-servisi',
 'ibb_license': 'https://data.ibb.gov.tr/license',
}
def fetch(item):
 name,url=item
 result={'kaynak':name,'url':url,'denetim_utc':dt.datetime.now(dt.timezone.utc).isoformat()}
 try:
  req=urllib.request.Request(url,headers={'User-Agent':'CourseProjectResearch/1.0'})
  with urllib.request.urlopen(req,timeout=25) as r:
   raw=r.read()
   result.update(http=r.status,son_url=r.url,byte=len(raw),sha256=hashlib.sha256(raw).hexdigest())
  ext='xml' if name=='tcmb_xml' else 'json' if name in ['open_meteo','nasa_power','ibb_categories'] else 'html'
  (OUT/f'{name}.{ext}').write_bytes(raw)
  if name=='open_meteo':
   d=json.loads(raw); result.update(gun=len(d['daily']['time']),ilk=d['daily']['time'][0],son=d['daily']['time'][-1],birimler=d['daily_units'],ilk_kayit={k:v[0] for k,v in d['daily'].items()})
  elif name=='nasa_power':
   d=json.loads(raw); result.update(parametreler=list(d['properties']['parameter']),ilk_kayit={k:next(iter(v.items())) for k,v in d['properties']['parameter'].items()},baslik=d.get('header',{}))
  elif name=='tcmb_xml':
   d=ET.fromstring(raw); result.update(tarih=d.attrib,kurlar={x.attrib.get('CurrencyCode'):{e.tag:e.text for e in x if e.tag in ['ForexBuying','ForexSelling','Unit']} for x in d if x.attrib.get('CurrencyCode') in ['USD','EUR']})
  elif name=='ibb_categories':
   result['yanit']=json.loads(raw)
 except Exception as e:
  result['hata']=f'{type(e).__name__}: {e}'
 return result
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
 results=list(pool.map(fetch,SOURCES.items()))
(OUT/'erisim_denetimi.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
print(json.dumps(results,ensure_ascii=False,indent=2))
