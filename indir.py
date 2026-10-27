"""İBB günlük hal fiyatlarını indirir; ham yanıtları saklar, CSV üretir."""
import concurrent.futures
import csv
import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'veriler'
OUT.mkdir(exist_ok=True)
DB = ROOT / 'ham_yanitlar.sqlite'
URL = 'https://halfiyatlaripublicdata.ibb.gov.tr/api/HalManager/'
START = dt.date(2016, 1, 1)
END = dt.date(2026, 10, 27)
lock = threading.Lock()
stop = threading.Event()
next_request = 0.0


def request(endpoint, data=None):
    global next_request
    with lock:
        delay = max(0, next_request - time.monotonic())
        next_request = max(next_request, time.monotonic()) + 0.15
    if delay:
        time.sleep(delay)
    if stop.is_set():
        raise RuntimeError('Erişim kısıtı nedeniyle indirme durduruldu.')
    req = urllib.request.Request(
        URL + endpoint,
        data=json.dumps(data).encode() if data is not None else None,
        headers={'Content-Type': 'application/json', 'Accept': 'application/json'},
    )
    try:
        with urllib.request.urlopen(req, timeout=35) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403, 429):
            stop.set()
        raise
    result = json.loads(raw)
    if result.get('ResponseStatus') is not True or not isinstance(result.get('Results'), list):
        raise ValueError(str(result.get('Message') or 'Geçersiz API yanıtı'))
    return raw, result['Results']


def fetch(day):
    for attempt in range(1, 4):
        try:
            raw, rows = request('getProductPricebyDay', {'item': {'Day': day}})
            if any(r.get('GuneAit', '')[:10] != day for r in rows):
                raise ValueError('Yanıttaki tarih sorgu tarihiyle eşleşmiyor.')
            return day, raw, rows, attempt, ''
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            if stop.is_set():
                break
            if attempt < 3:
                time.sleep(attempt * 2)
    return day, None, [], attempt, error


def main():
    con = sqlite3.connect(DB)
    con.execute('CREATE TABLE IF NOT EXISTS days (day TEXT PRIMARY KEY, raw BLOB, fetched TEXT, attempts INTEGER, error TEXT)')
    dates = [(START + dt.timedelta(days=i)).isoformat() for i in range((END-START).days+1)]
    cached = {r[0] for r in con.execute('SELECT day FROM days WHERE raw IS NOT NULL')}
    pending = [d for d in dates if d not in cached]
    print(f'Başlangıç: {len(dates)} gün; kayıtlı: {len(cached)}; sorgulanacak: {len(pending)}', flush=True)
    for endpoint in ['getCategories']:
        p = ROOT / (endpoint + '.json')
        if not p.exists():
            raw, _ = request(endpoint)
            p.write_bytes(raw)
    completed = len(cached)
    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        # En fazla dört aktif sorgu; kuyruk sınırlı tutulur.
        iterator = iter(pending)
        futures = {pool.submit(fetch, d) for d in [next(iterator, None) for _ in range(4)] if d}
        while futures:
            done, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                day, raw, rows, attempts, error = future.result()
                con.execute('INSERT OR REPLACE INTO days VALUES (?,?,?,?,?)',
                            (day, gzip.compress(raw) if raw else None,
                             dt.datetime.now(dt.timezone.utc).isoformat(), attempts, error))
                con.commit()
                completed += 1
                failures += bool(error)
                if completed % 100 == 0 or error or completed == len(dates):
                    print(f'İlerleme: {completed}/{len(dates)} gün; son: {day}; satır: {len(rows)}; hata: {failures}', flush=True)
                if not stop.is_set():
                    upcoming = next(iterator, None)
                    if upcoming:
                        futures.add(pool.submit(fetch, upcoming))
    export(con, dates)
    con.close()


def export(con, dates):
    fields = ['Tarih', 'TabelaGId', 'UrunAd', 'BirimTurId', 'KategoriId', 'EnDusukFiyat', 'EnYuksekFiyat', 'GuneAit', 'HalTurId']
    names = ['hal_tum_urunler_2016_2026.csv', 'hal_meyve_sebze_2016_2026.csv', 'gunluk_erisim_durumu.csv']
    handles = [(OUT / (n + '.tmp')).open('w', encoding='utf-8-sig', newline='') for n in names]
    allwriter, vegwriter = [csv.DictWriter(h, fieldnames=fields) for h in handles[:2]]
    daywriter = csv.writer(handles[2])
    allwriter.writeheader()
    vegwriter.writeheader()
    daywriter.writerow(['Tarih', 'Durum', 'ToplamSatir', 'MeyveSebzeSatir', 'SifirFiyatliSatir', 'YinelenenAnahtar', 'IndirmeZamaniUTC', 'Hata'])
    stats = dict(requested_days=len(dates), nonempty_days=0, empty_days=0, failed_days=0,
                 all_rows=0, fruit_vegetable_rows=0, fruit_vegetable_days=0, zero_price_rows=0,
                 duplicate_key_rows=0, inverted_price_rows=0, first_data_date=None, last_data_date=None,
                 years={})
    for day in dates:
        result = con.execute('SELECT raw,fetched,error FROM days WHERE day=?', (day,)).fetchone()
        if not result or result[0] is None:
            stats['failed_days'] += 1
            daywriter.writerow([day, 'hata', '', '', '', '', result[1] if result else '', result[2] if result else 'Sorgulanmadı'])
            continue
        rows = json.loads(gzip.decompress(result[0]))['Results']
        veg = sum(r['HalTurId'] == 2 for r in rows)
        zero = sum(r['EnDusukFiyat'] == 0 or r['EnYuksekFiyat'] == 0 for r in rows)
        keys = [(r['TabelaGId'], r['GuneAit'], r['BirimTurId'], r['HalTurId']) for r in rows]
        dup = len(keys) - len(set(keys))
        stats['nonempty_days' if rows else 'empty_days'] += 1
        stats['all_rows'] += len(rows)
        stats['fruit_vegetable_rows'] += veg
        stats['fruit_vegetable_days'] += bool(veg)
        stats['zero_price_rows'] += zero
        stats['duplicate_key_rows'] += dup
        stats['inverted_price_rows'] += sum(r['EnDusukFiyat'] > r['EnYuksekFiyat'] for r in rows)
        if rows:
            stats['first_data_date'] = stats['first_data_date'] or day
            stats['last_data_date'] = day
        year = stats['years'].setdefault(day[:4], {'all_rows': 0, 'fruit_vegetable_rows': 0, 'empty_days': 0})
        year['all_rows'] += len(rows)
        year['fruit_vegetable_rows'] += veg
        year['empty_days'] += not bool(rows)
        daywriter.writerow([day, 'veri_var' if rows else 'bos_yanit', len(rows), veg, zero, dup, result[1], ''])
        for row in rows:
            record = {'Tarih': day, **row}
            allwriter.writerow(record)
            if row['HalTurId'] == 2:
                vegwriter.writerow(record)
    for h in handles:
        h.close()
    for name in names:
        (OUT / (name + '.tmp')).replace(OUT / name)
    stats['sha256'] = {name: hashlib.sha256((OUT / name).read_bytes()).hexdigest() for name in names}
    (OUT / 'ozet.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'ACIKLAMA.txt').write_text(
        'İBB HAL FİYATLARI\n'
        f'Sorgulanan dönem: {START} – {END}\n'
        f'Kaynak: {URL}getProductPricebyDay\n'
        'Katalog: https://data.ibb.gov.tr/tr/dataset/hal-urunleri-ve-fiyatlari-web-servisi\n'
        'Lisans: https://data.ibb.gov.tr/license\n\n'
        'hal_tum_urunler: API tarafından dönen tüm hal türleri ve kayıtlar.\n'
        'hal_meyve_sebze: HalTurId=2 olan meyve, sebze ve ithal ürünler.\n'
        'CSV: UTF-8 BOM, virgül ayırıcı, nokta ondalık ayracı.\n'
        'Tarih sütunu GuneAit alanından üretilmiştir; diğer alanlar kaynak değerleridir.\n'
        'BirimTurId birim kodudur; tüm satırları kilogram olarak yorumlamayın.\n'
        'Fiyatlar alt ve üst fiyatlardır; gerçekleşmiş işlem ortalaması değildir.\n'
        'Sıfır fiyatlar, yinelenen kayıtlar ve ters fiyat aralıkları değiştirilmeden korunmuştur.\n'
        'Eksik günler doldurulmamıştır. Başarılı boş yanıtlar ile istek hataları ayrı listelenir.\n'
        '2026 yılı yalnızca sorgu bitiş tarihine kadardır.\n'
        'EPDK verisi ve model eğitimi bu dosyalara dahil değildir.\n'
        'Ham API yanıtları üst klasördeki ham_yanitlar.sqlite içinde gzip olarak saklanır.\n\n'
        + json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
    print('SONUC ' + json.dumps(stats, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
