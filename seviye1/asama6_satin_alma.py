#!/usr/bin/env python3
"""Aşama 6 — Satın Alma Karar Destek ve Çoklu Ürün Sepeti Simülasyon Modülü.

Tekil ürün ve çoklu ürün sepeti için:
  - Mevcut tahmini maliyet (mevcut_fiyat * miktar)
  - Gelecek işlem günü tahmini maliyet (tahmini_fiyat * miktar)
  - TL maliyet farkı ve yüzdesel değişim oranı
  - Modelin geçmiş test MAE / RMSE belirsizlik bağlamı
  - Yön beklentisi sınıflandırması (Fiyat Artışı / Fiyat Düşüşü / Düşük Değişim Beklentisi)
  - İsteğe bağlı bütçe yeterlilik analizi
hesaplamalarını gerçekleştirir. Tahminleri kesin fiyat veya tasarruf garantisi olarak sunmaz.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from asama1_veri import CATEGORY, sha256
from asama3_baseline import unit_label
from tahmin_servisi import HalPredictionService, MODEL_NAMES

ROOT = Path(__file__).resolve().parent
ASAMA1_DIR = ROOT / 'ciktilar/asama1'
ASAMA2_DIR = ROOT / 'ciktilar/asama2'
ASAMA4_DIR = ROOT / 'ciktilar/asama4'
ASAMA5_DIR = ROOT / 'ciktilar/asama5'
OUT_DIR = ROOT / 'ciktilar/asama6'

VALID_PRICE_TYPES = {
    'minimum': ('MevcutMinFiyat', 'TahminMinFiyat', 'hedef_min', 'Minimum Fiyat'),
    'maksimum': ('MevcutMaxFiyat', 'TahminMaxFiyat', 'hedef_max', 'Maksimum Fiyat'),
    'aritmetik_orta': ('MevcutOrtaFiyat', 'TahminOrtaFiyat', 'hedef_orta', 'Aritmetik Ortalama Fiyat'),
}

# Düşük değişim sinyal eşiği: değişim oranı <%0.75 veya modelin geçmiş test MAE'sinin %15'inden küçükse
LOW_CHANGE_PCT_THRESHOLD = 0.75
LOW_CHANGE_MAE_FRACTION = 0.15


def _validate_positive_number(val: Any, field_name: str) -> float:
    if val is None or isinstance(val, bool):
        raise ValueError(f'{field_name} boş veya boolean olamaz.')
    try:
        num = float(val)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{field_name} sayısal bir değer olmalıdır: {val!r}') from exc
    if not math.isfinite(num) or num <= 0.0:
        raise ValueError(f'{field_name} sıfırdan büyük sonlu bir sayı olmalıdır: {num}')
    return num


class PurchasingDecisionSupport:
    """Son gün model tahminleri ve geçmiş hata tabloları üzerinden satın alma simülasyonu yapar."""

    def __init__(
        self,
        asama4_dir: Path | str | None = None,
        asama5_dir: Path | str | None = None,
    ) -> None:
        self.asama4_dir = Path(asama4_dir).resolve() if asama4_dir else ASAMA4_DIR.resolve()
        self.asama5_dir = Path(asama5_dir).resolve() if asama5_dir else ASAMA5_DIR.resolve()

        latest_path = self.asama4_dir / 'son_gun_tahminleri.csv'
        registry_path = self.asama4_dir / 'model_kayit_defteri.json'
        series_metrics_path = self.asama5_dir / 'seri_bazli_model_karsilastirma.csv'
        main_metrics_path = self.asama5_dir / 'model_karsilastirma_ozeti.csv'

        for req_path in (latest_path, registry_path, series_metrics_path, main_metrics_path):
            if not req_path.is_file():
                raise FileNotFoundError(f'Karar destek modülü için gerekli dosya bulunamadı: {req_path}')

        self.latest_preds_df = pd.read_csv(latest_path, low_memory=False)
        self.registry = json.loads(registry_path.read_text(encoding='utf-8'))
        self.selected_model_code: str = self.registry['secilen_ana_model_kodu']
        self.series_metrics_df = pd.read_csv(series_metrics_path, low_memory=False)
        self.main_metrics_df = pd.read_csv(main_metrics_path, low_memory=False)

    def get_historical_error_context(
        self,
        series_id: str,
        model_code: str,
        price_type: str,
    ) -> dict[str, Any]:
        """Ürün serisinin test (yoksa doğrulama/genel) dönemindeki gerçek MAE ve RMSE değerlerini döndürür."""
        _, _, target_key, _ = VALID_PRICE_TYPES[price_type]
        lookup_targets = ['hedef_min', 'hedef_max'] if target_key == 'hedef_orta' else [target_key]

        for split_name, split_label in [('test', 'Test Dönemi (2025–2026)'), ('dogrulama', 'Doğrulama Dönemi (2024–2025)')]:
            sub = self.series_metrics_df[
                (self.series_metrics_df['Bolum'] == split_name)
                & (self.series_metrics_df['SeriId'] == series_id)
                & (self.series_metrics_df['ModelKodu'] == model_code)
                & (self.series_metrics_df['Hedef'].isin(lookup_targets))
            ]
            if not sub.empty:
                return {
                    'hata_kaynagi': f'seri_{split_name}',
                    'hata_kaynagi_aciklamasi': f'Ürünün {split_label} geçmiş performansı ({int(sub["OrnekSayisi"].iloc[0])} işlem günü)',
                    'ornek_sayisi': int(sub['OrnekSayisi'].iloc[0]),
                    'gecmis_mae_tl': float(sub['MAE'].mean()),
                    'gecmis_rmse_tl': float(sub['RMSE'].mean()),
                }

        # Seri test veya doğrulamada yer almadıysa modelin genel test hatasını döndür
        gsub = self.main_metrics_df[
            (self.main_metrics_df['Bolum'] == 'test')
            & (self.main_metrics_df['ModelKodu'] == model_code)
            & (self.main_metrics_df['Hedef'].isin(lookup_targets))
        ]
        return {
            'hata_kaynagi': 'genel_test',
            'hata_kaynagi_aciklamasi': 'Seri son dönemde kesintili olduğu için modelin genel Test Dönemi ortalaması baz alındı',
            'ornek_sayisi': int(gsub['OrnekSayisi'].iloc[0]),
            'gecmis_mae_tl': float(gsub['MAE'].mean()),
            'gecmis_rmse_tl': float(gsub['RMSE'].mean()),
        }

    def simulate_single_purchase(
        self,
        series_id: str,
        quantity: float,
        price_type: str = 'aritmetik_orta',
        model_code: str | None = None,
        budget: float | None = None,
    ) -> dict[str, Any]:
        """Tek bir ürün serisi için satın alma maliyeti ve karar destek analizi üretir."""
        if not isinstance(series_id, str) or not series_id.strip():
            raise ValueError('Geçerli bir ürün serisi kimliği (SeriId) seçilmelidir.')
        qty = _validate_positive_number(quantity, 'miktar')
        if price_type not in VALID_PRICE_TYPES:
            raise ValueError(f'Geçersiz fiyat_turu: {price_type!r}. Beklenen: {sorted(VALID_PRICE_TYPES)}')
        mcode = model_code or self.selected_model_code
        if mcode not in MODEL_NAMES:
            raise ValueError(f'Geçersiz model_kodu: {mcode!r}')
        validated_budget = _validate_positive_number(budget, 'butce') if budget is not None else None

        match = self.latest_preds_df[
            (self.latest_preds_df['SeriId'] == series_id)
            & (self.latest_preds_df['IstenenModelKodu'] == mcode)
        ]
        if match.empty:
            raise ValueError(f'Ürün serisi bulunamadı: {series_id}')

        row = match.iloc[0]
        cur_col, pred_col, _, price_type_label = VALID_PRICE_TYPES[price_type]
        cur_price = float(row[cur_col])
        pred_price = float(row[pred_col])

        cur_cost = cur_price * qty
        pred_cost = pred_price * qty
        diff_unit_tl = pred_price - cur_price
        diff_cost_tl = pred_cost - cur_cost
        pct_change = (diff_unit_tl / cur_price) * 100.0 if cur_price > 0 else 0.0

        err_ctx = self.get_historical_error_context(
            series_id=series_id,
            model_code=str(row['KullanilanModelKodu']),
            price_type=price_type,
        )
        hist_mae = float(err_ctx['gecmis_mae_tl'])
        hist_rmse = float(err_ctx['gecmis_rmse_tl'])
        mae_cost_band = hist_mae * qty

        # Yön beklentisi sınıflandırması (modelin geçmiş MAE belirsizliğini dikkate alır)
        signal_threshold_tl = max(0.05, LOW_CHANGE_MAE_FRACTION * hist_mae)
        if abs(pct_change) < LOW_CHANGE_PCT_THRESHOLD or abs(diff_unit_tl) < signal_threshold_tl:
            beklenti_sinifi = 'Düşük Değişim Beklentisi'
            karar_yorumu = (
                f'Modelin beklediği birim fiyat değişimi ({diff_unit_tl:+.2f} TL, %{pct_change:+.2f}), '
                f'bu ürünün geçmiş ortalama mutlak tahmin hatası (MAE: {hist_mae:.2f} TL) dikkate alındığında '
                f'düşük değişim bandındadır. Satın alma zamanlamasında stok ve tazelik ihtiyacı önceliklendirilmelidir.'
            )
        elif diff_unit_tl > 0:
            beklenti_sinifi = 'Fiyat Artışı Beklentisi'
            karar_yorumu = (
                f'Bir sonraki hal işlem gününde birim fiyatın {cur_price:.2f} TL seviyesinden '
                f'{pred_price:.2f} TL seviyesine (%{pct_change:+.2f}) yükselmesi beklenmektedir. '
                f'Toplam tahmini maliyet artışı {diff_cost_tl:+.2f} TL olup ürünün geçmiş MAE belirsizlik bandı '
                f'±{mae_cost_band:.2f} TL düzeyindedir; mevcut fiyattan alım değerlendirilebilir (kesin tasarruf garantisi değildir).'
            )
        else:
            beklenti_sinifi = 'Fiyat Düşüşü Beklentisi'
            karar_yorumu = (
                f'Bir sonraki hal işlem gününde birim fiyatın {cur_price:.2f} TL seviyesinden '
                f'{pred_price:.2f} TL seviyesine (%{pct_change:+.2f}) gerilemesi beklenmektedir. '
                f'Toplam tahmini maliyet farkı {diff_cost_tl:+.2f} TL olup ürünün geçmiş MAE belirsizlik bandı '
                f'±{mae_cost_band:.2f} TL düzeyindedir; acil olmayan alımlarda bir sonraki işlem günü izlenebilir (kesin fiyat garantisi değildir).'
            )

        budget_info: dict[str, Any] = {
            'butce_girildi': validated_budget is not None,
            'butce_tl': validated_budget,
            'mevcut_maliyet_butceye_uygun': None,
            'tahmini_maliyet_butceye_uygun': None,
            'mevcut_butce_farki_tl': None,
            'tahmini_butce_farki_tl': None,
            'butce_durumu_aciklamasi': 'Bütçe sınırı belirtilmedi.',
        }
        if validated_budget is not None:
            cur_ok = cur_cost <= validated_budget
            pred_ok = pred_cost <= validated_budget
            cur_rem = validated_budget - cur_cost
            pred_rem = validated_budget - pred_cost
            if cur_ok and pred_ok:
                b_desc = f'Hem mevcut maliyet ({cur_cost:,.2f} TL) hem tahmini maliyet ({pred_cost:,.2f} TL) bütçe ({validated_budget:,.2f} TL) sınırları içindedir.'
            elif cur_ok and not pred_ok:
                b_desc = f'Mevcut maliyet bütçeye uygun ({cur_cost:,.2f} TL) ancak beklenen fiyat artışıyla tahmini maliyet bütçeyi {-pred_rem:,.2f} TL aşmaktadır.'
            elif not cur_ok and pred_ok:
                b_desc = f'Mevcut maliyet bütçeyi {-cur_rem:,.2f} TL aşmakta, tahmini fiyat düşüşüyle bütçe sınırına ({pred_cost:,.2f} TL) gerilemektedir.'
            else:
                b_desc = f'Hem mevcut maliyet hem tahmini maliyet belirlenen bütçeyi ({validated_budget:,.2f} TL) aşmaktadır (tahmini aşım: {-pred_rem:,.2f} TL).'
            budget_info = {
                'butce_girildi': True,
                'butce_tl': validated_budget,
                'mevcut_maliyet_butceye_uygun': cur_ok,
                'tahmini_maliyet_butceye_uygun': pred_ok,
                'mevcut_butce_farki_tl': cur_rem,
                'tahmini_butce_farki_tl': pred_rem,
                'butce_durumu_aciklamasi': b_desc,
            }

        warnings_list = []
        if str(row['Uyarilar']).strip() and str(row['Uyarilar']) != 'nan':
            warnings_list.extend([w.strip() for w in str(row['Uyarilar']).split('|') if w.strip()])

        return {
            'seri_id': series_id,
            'urun_ad': str(row['UrunAd']),
            'kategori_id': int(row['KategoriId']),
            'kategori': str(row['Kategori']),
            'birim_kodu': int(row['BirimTurId']),
            'birim_aciklamasi': str(row['BirimAciklamasi']),
            'son_gozlem_tarihi': str(row['SonGozlemTarihi']),
            'ozellik_hazir': bool(row['OzellikHazir']),
            'geri_donus_kullanildi': bool(row['GeriDonusKullanildi']),
            'istenen_model_kodu': mcode,
            'kullanilan_model_kodu': str(row['KullanilanModelKodu']),
            'kullanilan_model_adi': str(row['KullanilanModelAdi']),
            'fiyat_turu': price_type,
            'fiyat_turu_etiketi': price_type_label,
            'miktar': qty,
            'mevcut_birim_fiyat_tl': cur_price,
            'tahmini_birim_fiyat_tl': pred_price,
            'mevcut_min_fiyat_tl': float(row['MevcutMinFiyat']),
            'mevcut_max_fiyat_tl': float(row['MevcutMaxFiyat']),
            'tahmini_min_fiyat_tl': float(row['TahminMinFiyat']),
            'tahmini_max_fiyat_tl': float(row['TahminMaxFiyat']),
            'birim_fiyat_farki_tl': diff_unit_tl,
            'degisim_orani_yuzde': pct_change,
            'mevcut_toplam_maliyet_tl': cur_cost,
            'tahmini_toplam_maliyet_tl': pred_cost,
            'toplam_maliyet_farki_tl': diff_cost_tl,
            'beklenti_sinifi': beklenti_sinifi,
            'karar_yorumu': karar_yorumu,
            'belirsizlik_baglami': {
                **err_ctx,
                'miktar_bazli_mae_bandi_tl': mae_cost_band,
                'miktar_bazli_rmse_bandi_tl': hist_rmse * qty,
            },
            'butce_analizi': budget_info,
            'uyarilar': warnings_list,
            'sorumluluk_notu': (
                'Bu simülasyon geçmiş İBB hal verileriyle eğitilmiş makine öğrenmesi tahminlerine dayanır; '
                'kesin piyasa fiyatı veya tasarruf garantisi niteliği taşımaz.'
            ),
        }

    def simulate_basket_purchase(
        self,
        basket_items: list[dict[str, Any]],
        price_type: str = 'aritmetik_orta',
        model_code: str | None = None,
        total_budget: float | None = None,
    ) -> dict[str, Any]:
        """Çoklu ürün sepeti için kalem bazlı ve toplam satın alma maliyeti simülasyonu yapar."""
        if not isinstance(basket_items, list) or len(basket_items) == 0:
            raise ValueError('Çoklu ürün sepeti en az bir ürün kalemi içermelidir.')
        if price_type not in VALID_PRICE_TYPES:
            raise ValueError(f'Geçersiz fiyat_turu: {price_type!r}')
        mcode = model_code or self.selected_model_code
        if mcode not in MODEL_NAMES:
            raise ValueError(f'Geçersiz model_kodu: {mcode!r}')
        validated_budget = _validate_positive_number(total_budget, 'toplam_butce') if total_budget is not None else None

        seen_series: set[str] = set()
        item_results: list[dict[str, Any]] = []
        for idx, item in enumerate(basket_items, start=1):
            if not isinstance(item, dict) or 'seri_id' not in item or 'miktar' not in item:
                raise ValueError(f'Sepet kalemi #{idx} içinde seri_id ve miktar alanları zorunludur.')
            sid = str(item['seri_id'])
            if sid in seen_series:
                raise ValueError(f'Sepette aynı ürün serisi birden fazla kez tekrarlandı: {sid}')
            seen_series.add(sid)
            res = self.simulate_single_purchase(
                series_id=sid,
                quantity=item['miktar'],
                price_type=price_type,
                model_code=mcode,
                budget=None,
            )
            item_results.append(res)

        total_cur_cost = float(sum(r['mevcut_toplam_maliyet_tl'] for r in item_results))
        total_pred_cost = float(sum(r['tahmini_toplam_maliyet_tl'] for r in item_results))
        total_diff_tl = total_pred_cost - total_cur_cost
        total_pct_change = (total_diff_tl / total_cur_cost * 100.0) if total_cur_cost > 0 else 0.0
        total_mae_band = float(sum(r['belirsizlik_baglami']['miktar_bazli_mae_bandi_tl'] for r in item_results))
        total_rmse_band = float(np.sqrt(sum((r['belirsizlik_baglami']['miktar_bazli_rmse_bandi_tl']) ** 2 for r in item_results)))

        unit_codes = sorted({int(r['birim_kodu']) for r in item_results})
        basket_warnings: list[str] = []
        if len(unit_codes) > 1:
            basket_warnings.append(
                f'Sepette farklı birim kodlarına ({unit_codes}) sahip ürünler bulunmaktadır. '
                f'Birim fiyatlar farklı birim kodları arasında ortalanamaz; yalnızca her kalemin kendi birimiyle çarpılan '
                f'TL tutarları (fiyat × miktar = TL) toplanmıştır.'
            )
        fallback_items = [r['urun_ad'] for r in item_results if r['geri_donus_kullanildi']]
        if fallback_items:
            basket_warnings.append(
                f'Sepetteki şu ürünlerde son gün 20 gözlem penceresi eksik olduğundan Naive Baseline geri dönüşü kullanılmıştır: '
                f'{", ".join(fallback_items)}.'
            )

        increasing_items = [r['urun_ad'] for r in item_results if r['beklenti_sinifi'] == 'Fiyat Artışı Beklentisi']
        decreasing_items = [r['urun_ad'] for r in item_results if r['beklenti_sinifi'] == 'Fiyat Düşüşü Beklentisi']
        stable_items = [r['urun_ad'] for r in item_results if r['beklenti_sinifi'] == 'Düşük Değişim Beklentisi']

        budget_summary: dict[str, Any] = {
            'butce_girildi': validated_budget is not None,
            'toplam_butce_tl': validated_budget,
            'mevcut_sepet_butceye_uygun': None,
            'tahmini_sepet_butceye_uygun': None,
            'mevcut_butce_farki_tl': None,
            'tahmini_butce_farki_tl': None,
            'butce_durumu_aciklamasi': 'Toplam sepet bütçesi belirtilmedi.',
        }
        if validated_budget is not None:
            cur_ok = total_cur_cost <= validated_budget
            pred_ok = total_pred_cost <= validated_budget
            cur_rem = validated_budget - total_cur_cost
            pred_rem = validated_budget - total_pred_cost
            budget_summary = {
                'butce_girildi': True,
                'toplam_butce_tl': validated_budget,
                'mevcut_sepet_butceye_uygun': cur_ok,
                'tahmini_sepet_butceye_uygun': pred_ok,
                'mevcut_butce_farki_tl': cur_rem,
                'tahmini_butce_farki_tl': pred_rem,
                'butce_durumu_aciklamasi': (
                    f'Toplam tahmini sepet maliyeti ({total_pred_cost:,.2f} TL) bütçe ({validated_budget:,.2f} TL) '
                    + (f'içindedir (kalan: {pred_rem:,.2f} TL).' if pred_ok else f'sınırını {-pred_rem:,.2f} TL aşmaktadır.')
                ),
            }

        return {
            'kalem_sayisi': len(item_results),
            'fiyat_turu': price_type,
            'fiyat_turu_etiketi': VALID_PRICE_TYPES[price_type][3],
            'istenen_model_kodu': mcode,
            'istenen_model_adi': MODEL_NAMES[mcode],
            'toplam_mevcut_maliyet_tl': total_cur_cost,
            'toplam_tahmini_maliyet_tl': total_pred_cost,
            'toplam_maliyet_farki_tl': total_diff_tl,
            'toplam_degisim_orani_yuzde': total_pct_change,
            'toplam_mae_belirsizlik_bandi_tl': total_mae_band,
            'bagimsiz_rmse_belirsizlik_bandi_tl': total_rmse_band,
            'fiyat_artisi_beklenen_urunler': increasing_items,
            'fiyat_dususu_beklenen_urunler': decreasing_items,
            'dusuk_degisim_beklenen_urunler': stable_items,
            'birim_kodlari': unit_codes,
            'butce_analizi': budget_summary,
            'sepet_uyarilari': basket_warnings,
            'kalemler': item_results,
            'sorumluluk_notu': (
                'Sepet simülasyonu geçmiş İBB hal verileriyle eğitilmiş model tahminlerine dayanır; '
                'kesin maliyet veya tasarruf garantisi sunmaz.'
            ),
        }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pds = PurchasingDecisionSupport()

    #Son günde aktif 91 seri içinden örnek senaryolar üret
    ready_sids = (
        pds.latest_preds_df[
            (pds.latest_preds_df['IstenenModelKodu'] == pds.selected_model_code)
            & (pds.latest_preds_df['OzellikHazir'])
        ]
        .sort_values('UrunAd')['SeriId']
        .tolist()
    )

    scenario_rows: list[dict[str, Any]] = []
    for sid in ready_sids:
        for ptype in ('minimum', 'maksimum', 'aritmetik_orta'):
            sim = pds.simulate_single_purchase(
                series_id=sid,
                quantity=100.0,
                price_type=ptype,
                model_code=pds.selected_model_code,
                budget=7500.0,
            )
            scenario_rows.append({
                'SeriId': sim['seri_id'],
                'UrunAd': sim['urun_ad'],
                'Kategori': sim['kategori'],
                'BirimKodu': sim['birim_kodu'],
                'BirimAciklamasi': sim['birim_aciklamasi'],
                'SonGozlemTarihi': sim['son_gozlem_tarihi'],
                'ModelKodu': sim['kullanilan_model_kodu'],
                'ModelAdi': sim['kullanilan_model_adi'],
                'FiyatTuru': sim['fiyat_turu'],
                'Miktar': sim['miktar'],
                'MevcutBirimFiyatTL': sim['mevcut_birim_fiyat_tl'],
                'TahminiBirimFiyatTL': sim['tahmini_birim_fiyat_tl'],
                'BirimFiyatFarkiTL': sim['birim_fiyat_farki_tl'],
                'DegisimOraniYuzde': sim['degisim_orani_yuzde'],
                'MevcutToplamMaliyetTL': sim['mevcut_toplam_maliyet_tl'],
                'TahminiToplamMaliyetTL': sim['tahmini_toplam_maliyet_tl'],
                'ToplamMaliyetFarkiTL': sim['toplam_maliyet_farki_tl'],
                'GecmisTestMAETL': sim['belirsizlik_baglami']['gecmis_mae_tl'],
                'GecmisTestRMSETL': sim['belirsizlik_baglami']['gecmis_rmse_tl'],
                'MiktarBazliMAEBandiTL': sim['belirsizlik_baglami']['miktar_bazli_mae_bandi_tl'],
                'BeklentiSinifi': sim['beklenti_sinifi'],
                'ButceTL': sim['butce_analizi']['butce_tl'],
                'TahminiMaliyetButceyeUygun': sim['butce_analizi']['tahmini_maliyet_butceye_uygun'],
                'KararYorumu': sim['karar_yorumu'],
            })

    scenario_df = pd.DataFrame(scenario_rows)
    scenario_df.to_csv(OUT_DIR / 'ornek_satin_alma_senaryolari.csv', index=False)

    # Bölüm 8.4 Örnek Çoklu Ürün Sepeti (Domates, Salatalık, Patates, Soğan Kuru)
    target_names = [('Domates', 100.0), ('Salatalık', 80.0), ('Patates', 150.0), ('Soğan (Kuru)', 120.0)]
    basket_input: list[dict[str, Any]] = []
    ready_sub = pds.latest_preds_df[
        (pds.latest_preds_df['IstenenModelKodu'] == pds.selected_model_code)
        & (pds.latest_preds_df['OzellikHazir'])
    ]
    for uname, qty in target_names:
        m = ready_sub[ready_sub['UrunAd'] == uname]
        if m.empty:
            m = ready_sub[ready_sub['UrunAd'].str.contains(uname.split()[0], regex=False)]
        basket_input.append({'seri_id': str(m.iloc[0]['SeriId']), 'miktar': qty})

    basket_sim = pds.simulate_basket_purchase(
        basket_items=basket_input,
        price_type='aritmetik_orta',
        model_code=pds.selected_model_code,
        total_budget=25000.0,
    )
    (OUT_DIR / 'ornek_sepet_simulasyonu.json').write_text(
        json.dumps(basket_sim, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )

    mid_df = scenario_df[scenario_df['FiyatTuru'] == 'aritmetik_orta']
    ozet = {
        'asama': 'Aşama 6 — Satın Alma Karar Destek Sistemi',
        'varsayilan_model_kodu': pds.selected_model_code,
        'aktif_seri_sayisi': len(ready_sids),
        'toplam_senaryo_satiri': len(scenario_df),
        'aritmetik_orta_beklenti_dagilimi': mid_df['BeklentiSinifi'].value_counts().to_dict(),
        'ornek_sepet_ozeti': {
            'kalem_sayisi': basket_sim['kalem_sayisi'],
            'toplam_mevcut_maliyet_tl': basket_sim['toplam_mevcut_maliyet_tl'],
            'toplam_tahmini_maliyet_tl': basket_sim['toplam_tahmini_maliyet_tl'],
            'toplam_maliyet_farki_tl': basket_sim['toplam_maliyet_farki_tl'],
            'toplam_degisim_orani_yuzde': basket_sim['toplam_degisim_orani_yuzde'],
            'toplam_mae_belirsizlik_bandi_tl': basket_sim['toplam_mae_belirsizlik_bandi_tl'],
        },
    }
    (OUT_DIR / 'ozet.json').write_text(json.dumps(ozet, ensure_ascii=False, indent=2), encoding='utf-8')

    lines = [
        '# Aşama 6 — Satın Alma Karar Destek Sistemi Raporu',
        '',
        '## 1. Karar Destek Mimarisi ve Belirsizlik İlkeleri',
        '- Sistem; ürün serisi (`SeriId`), satın alma miktarı (`miktar > 0`), fiyat türü (`minimum`, `maksimum`, `aritmetik_orta`) ve isteğe bağlı bütçe (`butce`) girdilerini doğrular.',
        '- Model tahminleri kesin fiyat veya tasarruf garantisi olarak sunulmaz; her tahminin yanında ilgili ürün serisinin Test Dönemi (`2025-10-27` – `2026-10-26`) gerçek **MAE** ve **RMSE** değerleri ve miktar bazlı belirsizlik bandı (`± MAE × miktar`) gösterilir.',
        '- Tahmini değişim yüzdesi `%0,75`\'in altında veya ürünün geçmiş MAE değerinin `%15`\'inden küçük olduğunda sistem yapay alarm üretmez ve **Düşük Değişim Beklentisi** sınıfını atar.',
        '',
        '## 2. Örnek Çoklu Ürün Sepeti Simülasyonu (`Domates`, `Salatalık`, `Patates`, `Soğan (Kuru)`)',
        f"- **Kullanılan Model:** `{basket_sim['istenen_model_adi']}` (`{basket_sim['fiyat_turu_etiketi']}`)",
        f"- **Toplam Mevcut Maliyet:** `{basket_sim['toplam_mevcut_maliyet_tl']:,.2f} TL`",
        f"- **Toplam Tahmini Maliyet:** `{basket_sim['toplam_tahmini_maliyet_tl']:,.2f} TL`",
        f"- **Beklenen Toplam Fark:** `{basket_sim['toplam_maliyet_farki_tl']:+,.2f} TL` (`%{basket_sim['toplam_degisim_orani_yuzde']:+.2f}`)",
        f"- **Toplam MAE Belirsizlik Bandı:** `±{basket_sim['toplam_mae_belirsizlik_bandi_tl']:,.2f} TL`",
        f"- **Bütçe Durumu:** {basket_sim['butce_analizi']['butce_durumu_aciklamasi']}",
        '',
        '| Ürün | Birim Açıklaması | Miktar | Mevcut Birim (TL) | Tahmini Birim (TL) | Mevcut Maliyet (TL) | Tahmini Maliyet (TL) | Fark (TL) | Değişim (%) | Geçmiş MAE (TL) | Beklenti Sınıfı |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    for item in basket_sim['kalemler']:
        lines.append(
            f"| **{item['urun_ad']}** | {item['birim_aciklamasi']} | {item['miktar']:,.0f} | "
            f"{item['mevcut_birim_fiyat_tl']:.2f} | {item['tahmini_birim_fiyat_tl']:.2f} | "
            f"{item['mevcut_toplam_maliyet_tl']:,.2f} | {item['tahmini_toplam_maliyet_tl']:,.2f} | "
            f"{item['toplam_maliyet_farki_tl']:+,.2f} | %{item['degisim_orani_yuzde']:+.2f} | "
            f"{item['belirsizlik_baglami']['gecmis_mae_tl']:.2f} | {item['beklenti_sinifi']} |"
        )

    (OUT_DIR / 'ASAMA6_RAPORU.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    inputs = [
        ASAMA4_DIR / 'son_gun_tahminleri.csv',
        ASAMA4_DIR / 'model_kayit_defteri.json',
        ASAMA5_DIR / 'seri_bazli_model_karsilastirma.csv',
        ASAMA5_DIR / 'model_karsilastirma_ozeti.csv',
    ]
    outputs = [
        OUT_DIR / 'ornek_satin_alma_senaryolari.csv',
        OUT_DIR / 'ornek_sepet_simulasyonu.json',
        OUT_DIR / 'ozet.json',
        OUT_DIR / 'ASAMA6_RAPORU.md',
    ]
    manifest = {
        'asama': 'asama6_satin_alma',
        'olusturulma_zamani': datetime.now(timezone.utc).isoformat(),
        'girdiler': [{'yol': str(p), 'boyut_bayt': p.stat().st_size, 'sha256': sha256(p)} for p in inputs],
        'ciktilar': [{'yol': str(p), 'boyut_bayt': p.stat().st_size, 'sha256': sha256(p)} for p in outputs],
    }
    (OUT_DIR / 'kaynak_manifestosu.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )

    print(json.dumps({
        'durum': 'TAMAMLANDI',
        'senaryo_sayisi': len(scenario_df),
        'sepet_kalem_sayisi': basket_sim['kalem_sayisi'],
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
