#!/usr/bin/env python3
"""Aşama 4, 5, 6 ve Web Uygulaması Bağımsız Doğrulama Betiği.

Orijinal CSV'lerin ve Aşama 1–3 çıktılarının değişmediğini, Aşama 4–6 model/rapor/grafik
çıktılarının SHA-256 manifestolarıyla birebir uyuştuğunu, tüm tahminlerde
0 < tahmin_min <= tahmin_max koşulunun sağlandığını ve satın alma simülasyonlarının
matematiksel tutarlılığını bağımsız olarak denetler.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from asama1_veri import sha256
from asama4_modeller import verify_upstream_hashes
from asama6_satin_alma import PurchasingDecisionSupport
from app import PAGES, verify_required_app_files
from tahmin_servisi import HalPredictionService

ROOT = Path(__file__).resolve().parent
OUT_JSON = ROOT / 'ciktilar/asama4_6_dogrulama.json'


def verify_manifest_file(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f'Manifesto bulunamadı: {manifest_path}')
    data = json.loads(manifest_path.read_text(encoding='utf-8'))
    checked_count = 0
    for group in ('girdiler', 'ciktilar'):
        for item in data.get(group, []):
            p = Path(item['yol'])
            if not p.is_file():
                raise FileNotFoundError(f'Manifestoda kayıtlı dosya eksik: {p}')
            actual_hash = sha256(p)
            if actual_hash != item['sha256']:
                raise RuntimeError(f'SHA-256 uyuşmazlığı ({p}): beklenen={item["sha256"]}, gerçek={actual_hash}')
            checked_count += 1
    return {'manifesto': str(manifest_path.relative_to(ROOT)), 'dogrulanan_dosya_sayisi': checked_count}


def main() -> None:
    upstream_hashes = verify_upstream_hashes()
    m4 = verify_manifest_file(ROOT / 'ciktilar/asama4/kaynak_manifestosu.json')
    m5 = verify_manifest_file(ROOT / 'ciktilar/asama5/kaynak_manifestosu.json')
    m6 = verify_manifest_file(ROOT / 'ciktilar/asama6/kaynak_manifestosu.json')

    service = HalPredictionService(ROOT / 'ciktilar/asama4')
    pds = PurchasingDecisionSupport(ROOT / 'ciktilar/asama4', ROOT / 'ciktilar/asama5')

    val_preds = pd.read_csv(ROOT / 'ciktilar/asama4/dogrulama_tum_modeller_tahminleri.csv.gz', low_memory=False)
    test_preds = pd.read_csv(ROOT / 'ciktilar/asama4/test_tum_modeller_tahminleri.csv.gz', low_memory=False)
    latest_preds = pd.read_csv(ROOT / 'ciktilar/asama4/son_gun_tahminleri.csv', low_memory=False)
    main_comp = pd.read_csv(ROOT / 'ciktilar/asama5/model_karsilastirma_ozeti.csv', low_memory=False)

    if len(val_preds) != 30361 or len(test_preds) != 21141:
        raise RuntimeError('Doğrulama veya test tahmin satır sayısı hatalı.')
    if len(latest_preds) != 299 * 4:
        raise RuntimeError(f'Son gün tahmin satır sayısı 1196 değil: {len(latest_preds)}')

    for mcode in service.available_models():
        for df_split in (val_preds, test_preds):
            pmin = df_split[f'{mcode}_tahmin_min'].to_numpy(dtype=float)
            pmax = df_split[f'{mcode}_tahmin_max'].to_numpy(dtype=float)
            if not (np.isfinite(pmin).all() and np.isfinite(pmax).all()):
                raise RuntimeError(f'{mcode} tahminlerinde NaN/inf bulundu.')
            if not ((pmin > 0).all() and (pmin <= pmax).all()):
                raise RuntimeError(f'{mcode} tahminlerinde min <= max veya pozitiflik ihlali var.')

    if not ((latest_preds['TahminMinFiyat'] > 0).all() and (latest_preds['TahminMinFiyat'] <= latest_preds['TahminMaxFiyat']).all()):
        raise RuntimeError('Son gün tahminlerinde min <= max ihlali var.')

    missing_app = verify_required_app_files(ROOT)
    if missing_app:
        raise RuntimeError(f'Web uygulaması dosyaları eksik: {missing_app}')

    sample_sid = str(latest_preds[latest_preds['OzellikHazir']].iloc[0]['SeriId'])
    sim = pds.simulate_single_purchase(sample_sid, quantity=50.0, price_type='aritmetik_orta', budget=5000.0)
    if not np.isclose(sim['mevcut_toplam_maliyet_tl'], sim['mevcut_birim_fiyat_tl'] * 50.0):
        raise RuntimeError('Tekil satın alma maliyet çarpımı hatalı.')

    payload = {
        'dogrulama_zamani': datetime.now(timezone.utc).isoformat(),
        'durum': 'BASARILI',
        'ust_asama_dosya_sayisi': len(upstream_hashes),
        'manifesto_kontrolleri': [m4, m5, m6],
        'yuklenen_model_sayisi': len(service.artifacts),
        'secilen_ana_model_kodu': service.selected_model_code,
        'dogrulama_tahmin_satiri': len(val_preds),
        'test_tahmin_satiri': len(test_preds),
        'son_gun_tahmin_satiri': len(latest_preds),
        'karsilastirma_tablo_satiri': len(main_comp),
        'streamlit_sayfa_sayisi': len(PAGES),
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == '__main__':
    main()
