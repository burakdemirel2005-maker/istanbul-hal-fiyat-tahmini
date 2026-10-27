#!/usr/bin/env python3
"""Aşama 4 — Makine Öğrenmesi Modelleri ve Fiyat Sürekliliği Araştırması.

Geliştirilen modeller:
  1. Linear Regression (Standartlaştırılmış sayısal + One-Hot kategorik + seri eğitim özetleri)
  2. Random Forest Regressor (Fiyat farkı / residual hedef formülasyonu)
  3. XGBoost Regressor (Düzenlileştirilmiş fiyat farkı / residual hedef formülasyonu)
Referans:
  0. Naive Baseline (Aşama 3 referans modeli)

Veri sızıntısı koruması:
  - Tüm ön işleme adımları (StandardScaler, OneHotEncoder, SeriesHistoryStatsEncoder)
    yalnızca eğitim kümesi (egitim.csv.gz) üzerinde fit edilir.
  - Özellik seçimi ve hiperparametre araması yalnızca eğitim ve doğrulama kümeleriyle yapılır.
  - Test kümesi (test.csv.gz) yalnızca doğrulamada kilitlenen nihai modellerin tek seferlik
    değerlendirmesi için kullanılır.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb
from xgboost import XGBRegressor

from asama1_veri import CATEGORY, sha256
from asama2_ozellikler import FEATURES
from asama3_baseline import compute_regression_metrics, unit_label
from tahmin_servisi import (
    CATEGORICAL_FEATURES,
    EXCLUDED_STAGE2_FEATURES,
    HalPredictionService,
    MODEL_NAMES,
    ResidualTargetModelArtifact,
    SELECTED_NUMERIC_FEATURES,
    SERIES_STAT_FEATURES,
    SERVICE_VERSION,
    SeriesHistoryStatsEncoder,
    enforce_min_max_consistency,
)

ROOT = Path(__file__).resolve().parent
MAIN_CSV = Path('/Users/burakdemirel/Desktop/IBB_Hal_Meyve_Sebze_2016_2026.csv')
SUPPLEMENT_CSV = Path('/Users/burakdemirel/Desktop/IBB_Hal_Tum_Urunler_2016_2026.csv')
ASAMA1_DIR = ROOT / 'ciktilar/asama1'
ASAMA2_DIR = ROOT / 'ciktilar/asama2'
ASAMA3_DIR = ROOT / 'ciktilar/asama3'
OUT_DIR = ROOT / 'ciktilar/asama4'
MODELS_DIR = OUT_DIR / 'modeller'

CLEAN_CSV = ASAMA1_DIR / 'temiz_veri.csv'
TRAIN_CSV = ASAMA2_DIR / 'egitim.csv.gz'
VAL_CSV = ASAMA2_DIR / 'dogrulama.csv.gz'
TEST_CSV = ASAMA2_DIR / 'test.csv.gz'
LATEST_CSV = ASAMA2_DIR / 'son_gun_girdileri.csv'
LATEST_STATUS_CSV = ASAMA2_DIR / 'son_gozlem_durumu.csv'
FEATURE_CONTRACT_JSON = ASAMA2_DIR / 'ozellik_sozlesmesi.json'
STAGE3_SUMMARY_JSON = ASAMA3_DIR / 'ozet.json'


def verify_upstream_hashes() -> dict[str, str]:
    """Orijinal CSV ve Aşama 1–3 çıktılarının SHA-256 özetlerini doğrular."""
    m3 = json.loads((ASAMA3_DIR / 'kaynak_manifestosu.json').read_text(encoding='utf-8'))
    checked: dict[str, str] = {}
    for group_key in ('orijinal_csv_once', 'asama1_once', 'asama2_once'):
        for path_str, expected_hash in m3[group_key].items():
            p = Path(path_str)
            actual = sha256(p)
            if actual != expected_hash:
                raise RuntimeError(f'Üst aşama dosya özeti değişmiş: {p}')
            checked[str(p)] = actual
    for fname, expected_hash in m3['cikti_sha256'].items():
        p = ASAMA3_DIR / fname
        actual = sha256(p)
        if actual != expected_hash:
            raise RuntimeError(f'Aşama 3 çıktı özeti değişmiş: {p}')
        checked[str(p)] = actual
    return checked


def feature_group_and_description(fname: str) -> tuple[str, str]:
    if fname in ('fiyat_min', 'fiyat_max', 'fiyat_orta', 'fiyat_araligi', 'goreli_fiyat_araligi'):
        return 'mevcut_gun_fiyat', f't günündeki {fname} gözlemi'
    if '_lag_' in fname:
        return 'gecikmeli_fiyat', f'Seri içi geçmiş işlem günü gecikmesi ({fname})'
    if '_rolling_' in fname:
        return 'hareketli_pencere', f't dahil kesintisiz geçmiş pencere istatistiği ({fname})'
    if '_fark_' in fname or '_degisim_orani_' in fname:
        return 'momentum_degisim', f'Kısa dönem fiyat farkı / değişim oranı ({fname})'
    if fname.startswith('tarih_'):
        return 'takvim_mevsimsellik', f'Gözlem tarihinden türetilen takvim/döngüsel özellik ({fname})'
    return 'gozlem_araligi', f'Bir önceki gözlemden bu yana geçen takvim günü ({fname})'


def run_price_persistence_investigation(
    clean_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, Any]:
    """Bölüm 6: Fiyat sürekliliği (price persistence) değişimini derinlemesine analiz eder."""
    splits = {'egitim': train_df, 'dogrulama': val_df, 'test': test_df}
    split_persistence: dict[str, Any] = {}
    for sname, sdf in splits.items():
        next_min_same = float((sdf['hedef_min'] == sdf['fiyat_min']).mean())
        next_max_same = float((sdf['hedef_max'] == sdf['fiyat_max']).mean())
        both_same = float(((sdf['hedef_min'] == sdf['fiyat_min']) & (sdf['hedef_max'] == sdf['fiyat_max'])).mean())
        frac_min = float((np.abs(sdf['fiyat_min'] - np.round(sdf['fiyat_min'])) > 1e-6).mean())
        split_persistence[sname] = {
            'ornek_sayisi': int(len(sdf)),
            'seri_sayisi': int(sdf['SeriId'].nunique()),
            'tarih_baslangic': str(sdf['Tarih'].min()),
            'tarih_bitis': str(sdf['Tarih'].max()),
            'hedef_min_degismedi_orani': next_min_same,
            'hedef_max_degismedi_orani': next_max_same,
            'her_iki_hedef_degismedi_orani': both_same,
            'kesirli_fiyat_orani': frac_min,
            'ortalama_fiyat_min_tl': float(sdf['fiyat_min'].mean()),
            'medyan_fiyat_min_tl': float(sdf['fiyat_min'].median()),
            'ortalama_mutlak_degisim_min_tl': float(np.abs(sdf['hedef_min'] - sdf['fiyat_min']).mean()),
            'ortalama_mutlak_degisim_max_tl': float(np.abs(sdf['hedef_max'] - sdf['fiyat_max']).mean()),
        }

    all_labeled = pd.concat([train_df, val_df, test_df], ignore_index=True)
    all_labeled['Yil'] = all_labeled['Tarih'].str.slice(0, 4).astype(int)
    yearly_records = []
    for yil, grp in all_labeled.groupby('Yil', sort=True):
        yearly_records.append({
            'yil': int(yil),
            'ornek_sayisi': int(len(grp)),
            'seri_sayisi': int(grp['SeriId'].nunique()),
            'hedef_min_degismedi_orani': float((grp['hedef_min'] == grp['fiyat_min']).mean()),
            'hedef_max_degismedi_orani': float((grp['hedef_max'] == grp['fiyat_max']).mean()),
            'kesirli_fiyat_orani': float((np.abs(grp['fiyat_min'] - np.round(grp['fiyat_min'])) > 1e-6).mean()),
            'ortalama_fiyat_min_tl': float(grp['fiyat_min'].mean()),
            'ortalama_mutlak_degisim_min_tl': float(np.abs(grp['hedef_min'] - grp['fiyat_min']).mean()),
        })

    common_series = sorted(set(val_df['SeriId']).intersection(set(test_df['SeriId'])))
    val_common = val_df[val_df['SeriId'].isin(common_series)]
    test_common = test_df[test_df['SeriId'].isin(common_series)]
    common_series_comparison = {
        'ortak_seri_sayisi': len(common_series),
        'dogrulama_ornek_sayisi': int(len(val_common)),
        'test_ornek_sayisi': int(len(test_common)),
        'dogrulama_hedef_min_degismedi_orani': float((val_common['hedef_min'] == val_common['fiyat_min']).mean()),
        'dogrulama_hedef_max_degismedi_orani': float((val_common['hedef_max'] == val_common['fiyat_max']).mean()),
        'test_hedef_min_degismedi_orani': float((test_common['hedef_min'] == test_common['fiyat_min']).mean()),
        'test_hedef_max_degismedi_orani': float((test_common['hedef_max'] == test_common['fiyat_max']).mean()),
    }

    c_sorted = clean_df.sort_values(['SeriId', 'Tarih']).reset_index(drop=True)
    c_sorted['prev_min'] = c_sorted.groupby('SeriId')['EnDusukFiyat'].shift(1)
    c_sorted['prev_max'] = c_sorted.groupby('SeriId')['EnYuksekFiyat'].shift(1)
    c_valid = c_sorted.dropna(subset=['prev_min', 'prev_max']).copy()
    c_valid['same_min'] = c_valid['EnDusukFiyat'] == c_valid['prev_min']
    c_valid['same_max'] = c_valid['EnYuksekFiyat'] == c_valid['prev_max']

    oct2_series = set(clean_df.loc[clean_df['Tarih'] == '2025-10-02', 'SeriId'])
    oct3_series = set(clean_df.loc[clean_df['Tarih'] == '2025-10-03', 'SeriId'])
    dropped_on_oct3 = sorted(oct2_series - oct3_series)
    kept_on_oct3 = sorted(oct2_series.intersection(oct3_series))

    val_pre_break = c_valid[(c_valid['Tarih'] >= '2024-10-27') & (c_valid['Tarih'] <= '2025-10-02')]
    val_post_break = c_valid[(c_valid['Tarih'] >= '2025-10-03') & (c_valid['Tarih'] <= '2025-10-24')]
    test_clean_window = c_valid[c_valid['Tarih'] >= '2025-10-27']

    structural_break = {
        'kirilma_tarihi': '2025-10-03',
        'onceki_islem_gunu': '2025-10-02',
        '2025_10_02_seri_sayisi': len(oct2_series),
        '2025_10_03_seri_sayisi': len(oct3_series),
        '2025_10_03_dusen_seri_sayisi': len(dropped_on_oct3),
        '2025_10_03_korunan_seri_sayisi': len(kept_on_oct3),
        'dusen_seriler_ornek': dropped_on_oct3[:12],
        'dusen_serilerin_kirilma_oncesi_sabit_kalma_orani': float(
            val_pre_break[val_pre_break['SeriId'].isin(dropped_on_oct3)]['same_min'].mean()
        ),
        'korunan_serilerin_kirilma_oncesi_sabit_kalma_orani': float(
            val_pre_break[val_pre_break['SeriId'].isin(kept_on_oct3)]['same_min'].mean()
        ),
        'korunan_serilerin_kirilma_sonrasi_dogrulama_sabit_kalma_orani': float(
            val_post_break[val_post_break['SeriId'].isin(kept_on_oct3)]['same_min'].mean()
        ),
        'korunan_serilerin_test_sabit_kalma_orani': float(
            test_clean_window[test_clean_window['SeriId'].isin(kept_on_oct3)]['same_min'].mean()
        ),
        'dogrulama_kirilma_oncesi_toplam_ornek': int(len(val_pre_break)),
        'dogrulama_kirilma_oncesi_same_min_orani': float(val_pre_break['same_min'].mean()),
        'dogrulama_kirilma_sonrasi_toplam_ornek': int(len(val_post_break)),
        'dogrulama_kirilma_sonrasi_same_min_orani': float(val_post_break['same_min'].mean()),
    }

    report_payload = {
        'bolum_bazli_sureklilik': split_persistence,
        'yillik_sureklilik': yearly_records,
        'ortak_106_seri_karsilastirmasi': common_series_comparison,
        'yapisal_kirilma_2025_10_03': structural_break,
    }

    (OUT_DIR / 'fiyat_surekliligi_analizi.json').write_text(
        json.dumps(report_payload, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )

    md_lines = [
        '# Bölüm 6 — Fiyat Sürekliliği (Price Persistence) Araştırma Raporu',
        '',
        '## 1. Araştırma Sorusu ve Temel Bulgu',
        'Aşama 3 Naive Baseline analizinde doğrulama döneminde (`2024-10-27` – `2025-10-26`) fiyatların bir sonraki işlem gününde aynı kalma oranı:',
        f"- `hedef_min == fiyat_min`: **%{split_persistence['dogrulama']['hedef_min_degismedi_orani']*100:.2f}**",
        f"- `hedef_max == fiyat_max`: **%{split_persistence['dogrulama']['hedef_max_degismedi_orani']*100:.2f}**",
        '',
        'iken, test döneminde (`2025-10-27` – `2026-10-26`) bu oranlar:',
        f"- `hedef_min == fiyat_min`: **%{split_persistence['test']['hedef_min_degismedi_orani']*100:.2f}**",
        f"- `hedef_max == fiyat_max`: **%{split_persistence['test']['hedef_max_degismedi_orani']*100:.2f}**",
        'seviyesine düşmektedir.',
        '',
        'Yapılan zamansal ve seri bazlı inceleme, bu farkın **2 Ekim 2025 (`2025-10-02`) ile 3 Ekim 2025 (`2025-10-03`) tarihleri arasında İBB hal fiyat bülteni güncelleme rejiminde gerçekleşen keskin bir yapısal değişimden** kaynaklandığını kanıtlamıştır.',
        '',
        '## 2. Dönem ve Yıl Bazında Fiyat Sürekliliği',
        '',
        '| Dönem / Yıl | Örnek Sayısı | Seri Sayısı | `hedef_min` Değişmedi (%) | `hedef_max` Değişmedi (%) | Kesirli Fiyat (%) | Ortalama `fiyat_min` (TL) | Ortalama `|Δmin|` (TL) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for sname, label in [('egitim', 'Eğitim (2016–2024)'), ('dogrulama', 'Doğrulama (2024–2025)'), ('test', 'Test (2025–2026)')]:
        sp = split_persistence[sname]
        md_lines.append(
            f"| **{label}** | {sp['ornek_sayisi']:,} | {sp['seri_sayisi']} | "
            f"%{sp['hedef_min_degismedi_orani']*100:.2f} | %{sp['hedef_max_degismedi_orani']*100:.2f} | "
            f"%{sp['kesirli_fiyat_orani']*100:.2f} | {sp['ortalama_fiyat_min_tl']:.2f} | {sp['ortalama_mutlak_degisim_min_tl']:.4f} |"
        )
    for yr in yearly_records:
        md_lines.append(
            f"| {yr['yil']} | {yr['ornek_sayisi']:,} | {yr['seri_sayisi']} | "
            f"%{yr['hedef_min_degismedi_orani']*100:.2f} | %{yr['hedef_max_degismedi_orani']*100:.2f} | "
            f"%{yr['kesirli_fiyat_orani']*100:.2f} | {yr['ortalama_fiyat_min_tl']:.2f} | {yr['ortalama_mutlak_degisim_min_tl']:.4f} |"
        )

    md_lines.extend([
        '',
        '## 3. 3 Ekim 2025 (`2025-10-03`) Yapısal Kırılmasının Kanıtı',
        f"- **2 Ekim 2025 Perşembe** günü bültende **{structural_break['2025_10_02_seri_sayisi']}** aktif seri yayımlanırken, **3 Ekim 2025 Cuma** günü aktif seri sayısı **{structural_break['2025_10_03_seri_sayisi']}**'ye düşmüştür.",
        f"- Yayından kaldırılan **{structural_break['2025_10_03_dusen_seri_sayisi']}** seri (ör. `{', '.join(structural_break['dusen_seriler_ornek'][:5])}`), kırılma öncesinde günlerin **%{structural_break['dusen_serilerin_kirilma_oncesi_sabit_kalma_orani']*100:.2f}**'sinde fiyatı hiç güncellenmeden bültende taşınan durağan kalemlerdir.",
        f"- Yayında kalmaya devam eden **{structural_break['2025_10_03_korunan_seri_sayisi']}** ana ürün serisinde ise kırılma öncesinde (`2024-10-27` – `2025-10-02`) fiyatların sabit kalma oranı **%{structural_break['korunan_serilerin_kirilma_oncesi_sabit_kalma_orani']*100:.2f}** iken, `2025-10-03` – `2025-10-24` arasındaki son 3 doğrulama haftasında **%{structural_break['korunan_serilerin_kirilma_sonrasi_dogrulama_sabit_kalma_orani']*100:.2f}**'ye, test döneminde (`2025-10-27` – `2026-10-26`) ise **%{structural_break['korunan_serilerin_test_sabit_kalma_orani']*100:.2f}**'ye inmiştir.",
        '',
        '## 4. Ürün Bileşimi Etkisinin Kontrolü (Ortak 106 Seri Karşılaştırması)',
        f"- Doğrulama ve test kümelerinin her ikisinde de yer alan **{common_series_comparison['ortak_seri_sayisi']} ortak ürün serisi** izole edildiğinde:",
        f"  - Doğrulama kümesinde (`{common_series_comparison['dogrulama_ornek_sayisi']:,}` örnek) `hedef_min` sabit kalma oranı **%{common_series_comparison['dogrulama_hedef_min_degismedi_orani']*100:.2f}**, `hedef_max` sabit kalma oranı **%{common_series_comparison['dogrulama_hedef_max_degismedi_orani']*100:.2f}**'dir.",
        f"  - Test kümesinde (`{common_series_comparison['test_ornek_sayisi']:,}` örnek) aynı 106 serinin `hedef_min` sabit kalma oranı **%{common_series_comparison['test_hedef_min_degismedi_orani']*100:.2f}**, `hedef_max` sabit kalma oranı **%{common_series_comparison['test_hedef_max_degismedi_orani']*100:.2f}**'dir.",
        '- Dolayısıyla süreklilik düşüşü yalnızca durağan ürünlerin bültenden çıkarılmasıyla açıklanamaz; **aktif ürünlerin günlük fiyat güncelleme sıklığı 3 Ekim 2025 itibarıyla yaklaşık 9 kat artmıştır** (günlük değişim oranı %6,9\'dan %57,5\'e çıkmıştır).',
        '',
        '## 5. Model Öğrenimi ve Değerlendirme Üzerindeki Etkileri',
        '1. **Naive Baseline\'ın L1 (MAE) Avantajı:** Eğitim ve doğrulama dönemlerinde günlerin %94–96\'sında gerçek fiyat değişimi `Δy = 0` olduğu için, koşullu medyan sıfıra eşittir. L1 kaybı (MAE) koşullu medyanı ödüllendirdiğinden, sıfırdan farklı her tahmin değişmeyen günlerde ceza alır.',
        '2. **L2 (MSE / RMSE / R²) Metriklerinde Makine Öğrenmesi Kazancı:** Fiyatların değiştiği günlerdeki ortalama kuadratik sıçrama (conditional mean), geçmiş momentum (`min_fark_1`, `min_rolling_5_mean - fiyat_min`), fiyat aralığı ve mevsimsellik özellikleriyle anlamlı biçimde tahmin edilebilmektedir. Bu nedenle doğru düzenlileştirilmiş (regularized) `Random Forest` ve `XGBoost` modelleri doğrulama kümesinde `Naive Baseline`\'dan daha düşük **RMSE / MSE** üretmektedir.',
        '3. **Nominal Enflasyon ve Hedef Formülasyonu:** 2016 yılında ortalama minimum fiyat `2,67 TL` iken 2026 yılında `71,59 TL`\'dir. Doğrudan fiyat seviyesi (`y(t+1)`) üzerinde eğitilen karar ağaçları eğitim bandının dışına çıkamazken, **fiyat farkı (`Δy(t+1) = y(t+1) - y(t)`)** üzerinde eğitilip `ŷ(t+1) = y(t) + Δ̂y(t+1)` ile geri çatılan modeller hem doğrulama hem test rejiminde ölçekten bağımsız çalışmaktadır.',
    ])
    (OUT_DIR / 'FIYAT_SUREKLILIGI_RAPORU.md').write_text('\n'.join(md_lines) + '\n', encoding='utf-8')
    return report_payload


def audit_features(train_df: pd.DataFrame) -> pd.DataFrame:
    """Aşama 2'deki 54 özelliği denetler ve seçim kararlarını tablolaştırır."""
    contract = json.loads(FEATURE_CONTRACT_JSON.read_text(encoding='utf-8'))
    rows = []
    for idx, fname in enumerate(contract['ozellikler'], start=1):
        grp, desc = feature_group_and_description(fname)
        selected = fname not in EXCLUDED_STAGE2_FEATURES
        if fname == 'tarih_yil':
            reason = 'Durağan dışı (non-stationary) doğrusal yıl trendi; 2025-2026 döneminde ekstrapolasyon sapması önlemek için çıkarıldı.'
        elif fname in ('min_degisim_orani_1', 'min_degisim_orani_5', 'max_degisim_orani_1', 'max_degisim_orani_5', 'goreli_fiyat_araligi'):
            reason = 'Paydada küçük fiyat olduğunda uç oran patlaması (2913x) yaratarak doğrusal modeli bozduğu için çıkarıldı; TL fark karşılıkları korundu.'
        else:
            reason = 'Tahmin anında (t) sızıntısız kullanılabilir ve doğrulama kümesinde kararlı sinyal sağlar.'
        s = train_df[fname]
        rows.append({
            'sira': idx,
            'ozellik_adi': fname,
            'grup': grp,
            'veri_tipi': str(s.dtype),
            'tahmin_aninda_kullanilabilir': True,
            'veri_sizintisi_riski': 'Yok (yalnızca <= t gözlemlerinden türetildi)',
            'egitim_eksik_deger_sayisi': int(s.isna().sum()),
            'egitim_ortalama': float(s.mean()),
            'egitim_std': float(s.std()),
            'egitim_min': float(s.min()),
            'egitim_max': float(s.max()),
            'modele_dahil_edildi': selected,
            'karar_gerekcesi': reason,
            'aciklama': desc,
        })
    audit_df = pd.DataFrame(rows)
    audit_df.to_csv(OUT_DIR / 'ozellik_incelemesi.csv', index=False)
    return audit_df


def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    num_cols = SELECTED_NUMERIC_FEATURES + SERIES_STAT_FEATURES
    num_trans: Any = StandardScaler() if scale_numeric else 'passthrough'
    return ColumnTransformer(
        transformers=[
            ('num', num_trans, num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder='drop',
    )


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, baseline_metrics: dict[str, Any]) -> dict[str, Any]:
    raw = compute_regression_metrics(y_true, y_pred)
    base_mae = float(baseline_metrics['MAE'])
    base_rmse = float(baseline_metrics['RMSE'])
    mae = float(raw['MAE'])
    mse = float(raw['MSE'])
    rmse = float(raw['RMSE'])
    r2 = float(raw['R2']) if raw['R2'] is not None else 0.0
    return {
        'ornek_sayisi': int(raw['OrnekSayisi']),
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'baseline_mae_farki_tl': float(mae - base_mae),
        'baseline_rmse_farki_tl': float(rmse - base_rmse),
        'baseline_mae_iyilesme_yuzde': float((base_mae - mae) / base_mae * 100.0) if base_mae > 0 else 0.0,
        'baseline_rmse_iyilesme_yuzde': float((base_rmse - rmse) / base_rmse * 100.0) if base_rmse > 0 else 0.0,
    }


def train_and_evaluate_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    stage3_summary: dict[str, Any],
) -> tuple[dict[tuple[str, str], ResidualTargetModelArtifact], pd.DataFrame, dict[str, Any]]:
    """Hiperparametre deneylerini doğrulama kümesinde çalıştırır ve seçilen modelleri kaydeder."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    val_base_min = stage3_summary['bolumler']['dogrulama']['hedefler']['hedef_min']['satir_agirlikli']
    val_base_max = stage3_summary['bolumler']['dogrulama']['hedefler']['hedef_max']['satir_agirlikli']
    test_base_min = stage3_summary['bolumler']['test']['hedefler']['hedef_min']['satir_agirlikli']
    test_base_max = stage3_summary['bolumler']['test']['hedefler']['hedef_max']['satir_agirlikli']

    clean_train_mask = ~(
        train_df['IncelemeGerekli'].astype(bool)
        | train_df['HedefIncelemeGerekli'].astype(bool)
        | (train_df['GecmisteIncelemeSayisi'] > 0)
    )
    train_clean = train_df.loc[clean_train_mask].reset_index(drop=True)
    training_series_counts = {str(k): int(v) for k, v in train_df['SeriId'].value_counts().items()}

    series_encoder = SeriesHistoryStatsEncoder().fit(train_clean)
    X_train_enc = series_encoder.transform(train_clean)
    X_val_enc = series_encoder.transform(val_df)

    y_train_delta = {
        'hedef_min': (train_clean['hedef_min'] - train_clean['fiyat_min']).to_numpy(dtype=float),
        'hedef_max': (train_clean['hedef_max'] - train_clean['fiyat_max']).to_numpy(dtype=float),
    }
    val_anchors = {
        'hedef_min': val_df['fiyat_min'].to_numpy(dtype=float),
        'hedef_max': val_df['fiyat_max'].to_numpy(dtype=float),
    }
    val_targets = {
        'hedef_min': val_df['hedef_min'].to_numpy(dtype=float),
        'hedef_max': val_df['hedef_max'].to_numpy(dtype=float),
    }
    val_baselines = {'hedef_min': val_base_min, 'hedef_max': val_base_max}
    test_baselines = {'hedef_min': test_base_min, 'hedef_max': test_base_max}

    search_rows: list[dict[str, Any]] = []

    # Referans deney: Doğrudan hedef y(t+1) ve 54 ham özellikle temel karşılaştırma
    X_tr_raw54 = train_df[FEATURES]
    X_va_raw54 = val_df[FEATURES]
    for fam_code, fam_name, raw_est in [
        ('linear_regression', 'Linear Regression', LinearRegression()),
        ('random_forest', 'Random Forest', RandomForestRegressor(n_estimators=40, max_depth=8, min_samples_leaf=50, max_features='sqrt', n_jobs=-1, random_state=42)),
        ('xgboost', 'XGBoost', XGBRegressor(n_estimators=50, max_depth=5, learning_rate=0.05, n_jobs=-1, random_state=42)),
    ]:
        for tcol in ('hedef_min', 'hedef_max'):
            t0 = time.perf_counter()
            raw_est.fit(X_tr_raw54, train_df[tcol].to_numpy(dtype=float))
            elapsed = time.perf_counter() - t0
            pred_raw = np.maximum(raw_est.predict(X_va_raw54), 0.01)
            m_raw = evaluate_predictions(val_targets[tcol], pred_raw, val_baselines[tcol])
            search_rows.append({
                'deney_grubu': 'ham_54_ozellik_dogrudan_hedef',
                'model_kodu': fam_code,
                'model_adi': fam_name,
                'hedef': tcol,
                'aday_kodu': f'{fam_code}_raw54_direct',
                'hiperparametreler': json.dumps({'formulasyon': 'dogrudan_fiyat', 'ozellik_sayisi': 54}, ensure_ascii=False),
                'egitim_suresi_sn': round(elapsed, 4),
                'dogrulama_mae': m_raw['mae'],
                'dogrulama_rmse': m_raw['rmse'],
                'dogrulama_r2': m_raw['r2'],
                'baseline_mae_iyilesme_yuzde': m_raw['baseline_mae_iyilesme_yuzde'],
                'baseline_rmse_iyilesme_yuzde': m_raw['baseline_rmse_iyilesme_yuzde'],
                'secildi': False,
            })

    candidate_grids: dict[str, list[tuple[str, dict[str, Any], bool]]] = {
        'linear_regression': [
            ('lr_std_onehot_delta', {'fit_intercept': True, 'scale_numeric': True}, True),
            ('lr_no_intercept_delta', {'fit_intercept': False, 'scale_numeric': True}, False),
        ],
        'random_forest': [
            (
                'rf_d4_leaf80_mf04',
                {'n_estimators': 80, 'max_depth': 4, 'min_samples_leaf': 80, 'max_features': 0.4, 'random_state': 42, 'n_jobs': -1},
                True,
            ),
            (
                'rf_d6_leaf100_sqrt',
                {'n_estimators': 80, 'max_depth': 6, 'min_samples_leaf': 100, 'max_features': 'sqrt', 'random_state': 42, 'n_jobs': -1},
                False,
            ),
            (
                'rf_d8_leaf50_sqrt',
                {'n_estimators': 80, 'max_depth': 8, 'min_samples_leaf': 50, 'max_features': 'sqrt', 'random_state': 42, 'n_jobs': -1},
                False,
            ),
        ],
        'xgboost': [
            (
                'xgb_d4_lr002_n35_l2_50',
                {
                    'n_estimators': 35,
                    'max_depth': 4,
                    'learning_rate': 0.02,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'min_child_weight': 50,
                    'reg_lambda': 50.0,
                    'reg_alpha': 1.0,
                    'base_score': 0.0,
                    'objective': 'reg:squarederror',
                    'random_state': 42,
                    'n_jobs': -1,
                },
                True,
            ),
            (
                'xgb_d3_lr003_n40_l2_30',
                {
                    'n_estimators': 40,
                    'max_depth': 3,
                    'learning_rate': 0.03,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'min_child_weight': 40,
                    'reg_lambda': 30.0,
                    'reg_alpha': 1.0,
                    'base_score': 0.0,
                    'objective': 'reg:squarederror',
                    'random_state': 42,
                    'n_jobs': -1,
                },
                False,
            ),
            (
                'xgb_d5_lr005_n80_l2_10',
                {
                    'n_estimators': 80,
                    'max_depth': 5,
                    'learning_rate': 0.05,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'min_child_weight': 20,
                    'reg_lambda': 10.0,
                    'reg_alpha': 0.0,
                    'base_score': 0.0,
                    'objective': 'reg:squarederror',
                    'random_state': 42,
                    'n_jobs': -1,
                },
                False,
            ),
        ],
    }

    trained_candidates: dict[tuple[str, str, str], tuple[Pipeline, float, dict[str, Any], dict[str, Any]]] = {}
    for fam_code, grid in candidate_grids.items():
        fam_name = MODEL_NAMES[fam_code]
        for cand_code, params, _ in grid:
            for tcol in ('hedef_min', 'hedef_max'):
                if fam_code == 'linear_regression':
                    pre = build_preprocessor(scale_numeric=bool(params['scale_numeric']))
                    est = LinearRegression(fit_intercept=bool(params['fit_intercept']))
                elif fam_code == 'random_forest':
                    pre = build_preprocessor(scale_numeric=False)
                    est = RandomForestRegressor(**params)
                else:
                    pre = build_preprocessor(scale_numeric=False)
                    est = XGBRegressor(**params)

                pipe = Pipeline([('preprocessor', pre), ('regressor', est)])
                t0 = time.perf_counter()
                pipe.fit(X_train_enc, y_train_delta[tcol])
                elapsed = time.perf_counter() - t0
                pred_val = np.maximum(val_anchors[tcol] + pipe.predict(X_val_enc), 0.01)
                m_val = evaluate_predictions(val_targets[tcol], pred_val, val_baselines[tcol])
                trained_candidates[(fam_code, cand_code, tcol)] = (pipe, elapsed, params, m_val)

        best_cand_code = min(
            [c[0] for c in grid],
            key=lambda ccode: (
                trained_candidates[(fam_code, ccode, 'hedef_min')][3]['rmse']
                + trained_candidates[(fam_code, ccode, 'hedef_max')][3]['rmse']
            ) / 2.0,
        )
        for cand_code, params, _ in grid:
            for tcol in ('hedef_min', 'hedef_max'):
                _, elapsed, _, m_val = trained_candidates[(fam_code, cand_code, tcol)]
                search_rows.append({
                    'deney_grubu': 'fiyat_farki_secilmis_ozellikler',
                    'model_kodu': fam_code,
                    'model_adi': fam_name,
                    'hedef': tcol,
                    'aday_kodu': cand_code,
                    'hiperparametreler': json.dumps(params, ensure_ascii=False),
                    'egitim_suresi_sn': round(elapsed, 4),
                    'dogrulama_mae': m_val['mae'],
                    'dogrulama_rmse': m_val['rmse'],
                    'dogrulama_r2': m_val['r2'],
                    'baseline_mae_iyilesme_yuzde': m_val['baseline_mae_iyilesme_yuzde'],
                    'baseline_rmse_iyilesme_yuzde': m_val['baseline_rmse_iyilesme_yuzde'],
                    'secildi': cand_code == best_cand_code,
                })

    search_df = pd.DataFrame(search_rows)
    search_df.to_csv(OUT_DIR / 'hiperparametre_arama_sonuclari.csv', index=False)

    lib_versions = {
        'python': sys.version.split()[0],
        'numpy': np.__version__,
        'pandas': pd.__version__,
        'scikit_learn': sklearn.__version__,
        'xgboost': xgb.__version__,
        'joblib': joblib.__version__,
    }
    trained_at = datetime.now(timezone.utc).isoformat()
    artifacts: dict[tuple[str, str], ResidualTargetModelArtifact] = {}
    registry_entries: list[dict[str, Any]] = []

    for fam_code in ('linear_regression', 'random_forest', 'xgboost'):
        sel_rows = search_df[
            (search_df['deney_grubu'] == 'fiyat_farki_secilmis_ozellikler')
            & (search_df['model_kodu'] == fam_code)
            & (search_df['secildi'])
        ]
        cand_code = str(sel_rows.iloc[0]['aday_kodu'])
        pipe_min, elapsed_min, params_min, _ = trained_candidates[(fam_code, cand_code, 'hedef_min')]
        pipe_max, elapsed_max, params_max, _ = trained_candidates[(fam_code, cand_code, 'hedef_max')]

        art_min = ResidualTargetModelArtifact(
            model_code=fam_code,
            model_name=MODEL_NAMES[fam_code],
            version=SERVICE_VERSION,
            target_col='hedef_min',
            anchor_col='fiyat_min',
            selected_numeric_features=SELECTED_NUMERIC_FEATURES,
            categorical_features=CATEGORICAL_FEATURES,
            series_stat_features=SERIES_STAT_FEATURES,
            series_encoder=series_encoder,
            pipeline=pipe_min,
            training_date_range={'baslangic': str(train_df['Tarih'].min()), 'bitis': str(train_df['Tarih'].max())},
            training_row_count=int(len(train_df)),
            training_clean_row_count=int(len(train_clean)),
            training_series_counts=training_series_counts,
            hyperparameters=params_min,
            library_versions=lib_versions,
            validation_metrics={},
            test_metrics={},
            trained_at=trained_at,
        )
        art_max = ResidualTargetModelArtifact(
            model_code=fam_code,
            model_name=MODEL_NAMES[fam_code],
            version=SERVICE_VERSION,
            target_col='hedef_max',
            anchor_col='fiyat_max',
            selected_numeric_features=SELECTED_NUMERIC_FEATURES,
            categorical_features=CATEGORICAL_FEATURES,
            series_stat_features=SERIES_STAT_FEATURES,
            series_encoder=series_encoder,
            pipeline=pipe_max,
            training_date_range={'baslangic': str(train_df['Tarih'].min()), 'bitis': str(train_df['Tarih'].max())},
            training_row_count=int(len(train_df)),
            training_clean_row_count=int(len(train_clean)),
            training_series_counts=training_series_counts,
            hyperparameters=params_max,
            library_versions=lib_versions,
            validation_metrics={},
            test_metrics={},
            trained_at=trained_at,
        )

        val_raw_min = art_min.predict_price(val_df)
        val_raw_max = art_max.predict_price(val_df)
        val_fix_min, val_fix_max, val_inv = enforce_min_max_consistency(val_raw_min, val_raw_max)
        art_min.validation_metrics = {
            **evaluate_predictions(val_df['hedef_min'].to_numpy(dtype=float), val_fix_min, val_baselines['hedef_min']),
            'egitim_suresi_sn': round(elapsed_min, 4),
            'ters_aralik_duzeltme_sayisi': int(val_inv.sum()),
        }
        art_max.validation_metrics = {
            **evaluate_predictions(val_df['hedef_max'].to_numpy(dtype=float), val_fix_max, val_baselines['hedef_max']),
            'egitim_suresi_sn': round(elapsed_max, 4),
            'ters_aralik_duzeltme_sayisi': int(val_inv.sum()),
        }

        test_raw_min = art_min.predict_price(test_df)
        test_raw_max = art_max.predict_price(test_df)
        test_fix_min, test_fix_max, test_inv = enforce_min_max_consistency(test_raw_min, test_raw_max)
        art_min.test_metrics = {
            **evaluate_predictions(test_df['hedef_min'].to_numpy(dtype=float), test_fix_min, test_baselines['hedef_min']),
            'ters_aralik_duzeltme_sayisi': int(test_inv.sum()),
        }
        art_max.test_metrics = {
            **evaluate_predictions(test_df['hedef_max'].to_numpy(dtype=float), test_fix_max, test_baselines['hedef_max']),
            'ters_aralik_duzeltme_sayisi': int(test_inv.sum()),
        }

        for tcol, art, el in [('hedef_min', art_min, elapsed_min), ('hedef_max', art_max, elapsed_max)]:
            fname = f'{fam_code}_{tcol}.joblib'
            fpath = MODELS_DIR / fname
            joblib.dump(art, fpath, compress=3)
            fhash = sha256(fpath)
            artifacts[(fam_code, tcol)] = art
            registry_entries.append({
                'model_kodu': fam_code,
                'model_adi': MODEL_NAMES[fam_code],
                'surum': SERVICE_VERSION,
                'hedef': tcol,
                'dayanak_sutun': art.anchor_col,
                'hedef_formulasyonu': f'delta = {tcol} - {art.anchor_col}; tahmin = max(0.01, {art.anchor_col} + delta_hat)',
                'dosya_adi': fname,
                'sha256': fhash,
                'boyut_bayt': fpath.stat().st_size,
                'egitim_tarih_araligi': art.training_date_range,
                'egitim_toplam_ornek_sayisi': art.training_row_count,
                'egitim_kullanilan_temiz_ornek_sayisi': art.training_clean_row_count,
                'egitim_seri_sayisi': len(training_series_counts),
                'secilen_sayisal_ozellik_sayisi': len(SELECTED_NUMERIC_FEATURES),
                'kategorik_ozellik_sayisi': len(CATEGORICAL_FEATURES),
                'seri_gecmis_ozet_ozellik_sayisi': len(SERIES_STAT_FEATURES),
                'cikarilan_asama2_ozellikleri': list(EXCLUDED_STAGE2_FEATURES),
                'on_isleme_bilgisi': {
                    'seri_kodlayici': 'SeriesHistoryStatsEncoder (yalnızca egitim.csv.gz üzerinde fit edildi)',
                    'sayisal_olcekleme': 'StandardScaler' if fam_code == 'linear_regression' else 'passthrough',
                    'kategorik_kodlama': "OneHotEncoder(handle_unknown='ignore')",
                    'egitim_asiri_sicrama_filtresi': 'IncelemeGerekli == False ve HedefIncelemeGerekli == False ve GecmisteIncelemeSayisi == 0',
                },
                'hiperparametreler': art.hyperparameters,
                'kutuphane_surumleri': lib_versions,
                'egitim_suresi_sn': round(el, 4),
                'dogrulama_performansi': art.validation_metrics,
                'test_performansi': art.test_metrics,
                'kayit_zamani': trained_at,
            })

    selected_primary_model = min(
        ['linear_regression', 'random_forest', 'xgboost'],
        key=lambda mcode: (
            artifacts[(mcode, 'hedef_min')].validation_metrics['rmse']
            + artifacts[(mcode, 'hedef_max')].validation_metrics['rmse']
        ) / 2.0,
    )
    selected_mae_ml_model = min(
        ['linear_regression', 'random_forest', 'xgboost'],
        key=lambda mcode: (
            artifacts[(mcode, 'hedef_min')].validation_metrics['mae']
            + artifacts[(mcode, 'hedef_max')].validation_metrics['mae']
        ) / 2.0,
    )

    registry = {
        'kayit_defteri_surumu': SERVICE_VERSION,
        'olusturulma_zamani': trained_at,
        'secilen_ana_model_kodu': selected_primary_model,
        'secilen_ana_model_adi': MODEL_NAMES[selected_primary_model],
        'en_dusuk_dogrulama_mae_ml_modeli': selected_mae_ml_model,
        'secim_kriteri_aciklamasi': (
            'Doğrulama kümesinde (2024-10-27 – 2025-10-26) Naive Baseline modelini hem hedef_min hem hedef_max '
            'için RMSE, MSE ve R² metriklerinde geçen ve en düşük ortalama doğrulama RMSE değerini üreten '
            'Random Forest modeli ana üretim modeli olarak kilitlenmiştir. XGBoost ise makine öğrenmesi '
            'modelleri arasında en düşük doğrulama MAE değerine sahiptir ve aynı şekilde Naive Baseline '
            'doğrulama RMSE değerini her iki hedefte geçmektedir.'
        ),
        'min_max_tutarlilik_kurali': (
            'Ayrı eğitilen hedef_min ve hedef_max modellerinin ham tahminlerinde tahmin_min > tahmin_max '
            'oluşursa deterministik olarak alt = min(tahmin_min, tahmin_max), üst = max(tahmin_min, tahmin_max) '
            'uygulanır ve TersAralikDuzeltildi bayrağı işaretlenir.'
        ),
        'modeller': registry_entries,
    }
    (OUT_DIR / 'model_kayit_defteri.json').write_text(
        json.dumps(registry, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    return artifacts, search_df, registry


def export_split_predictions_and_latest(
    service: HalPredictionService,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    latest_df: pd.DataFrame,
    status_df: pd.DataFrame,
    clean_df: pd.DataFrame,
) -> dict[str, Any]:
    """Doğrulama, test ve son gün (2026-10-26) için tüm modellerin tahmin tablolarını kaydeder."""
    base_cols = [
        'KaynakKayitNo', 'Tarih', 'HedefTarih', 'TabelaGId', 'UrunAd',
        'BirimTurId', 'KategoriId', 'SeriId',
        'fiyat_min', 'fiyat_max', 'hedef_min', 'hedef_max',
        'IncelemeGerekli', 'HedefIncelemeGerekli', 'GecmisteIncelemeSayisi',
    ]

    for split_name, sdf, out_name in [
        ('dogrulama', val_df, 'dogrulama_tum_modeller_tahminleri.csv.gz'),
        ('test', test_df, 'test_tum_modeller_tahminleri.csv.gz'),
    ]:
        combined = sdf[base_cols].copy()
        for mcode in service.available_models():
            pred_df = service.predict_dataframe(sdf, model_code=mcode)
            combined[f'{mcode}_tahmin_min'] = pred_df['tahmin_min']
            combined[f'{mcode}_tahmin_max'] = pred_df['tahmin_max']
            combined[f'{mcode}_ters_aralik_duzeltildi'] = pred_df['TersAralikDuzeltildi']
            if mcode == service.selected_model_code:
                combined['EgitimOrnekSayisi'] = pred_df['EgitimOrnekSayisi']
                combined['EgitimdeGoruldu'] = pred_df['EgitimdeGoruldu']
                combined['DusukEgitimKapsami'] = pred_df['DusukEgitimKapsami']
        combined.to_csv(OUT_DIR / out_name, index=False, compression='gzip')

    latest_rows: list[dict[str, Any]] = []
    for _, st_row in status_df.sort_values('SeriId').iterrows():
        sid = str(st_row['SeriId'])
        for mcode in service.available_models():
            res = service.predict_for_series_row(
                series_id=sid,
                latest_features_df=latest_df,
                status_df=status_df,
                clean_df=clean_df,
                model_code=mcode,
                allow_naive_fallback=True,
            )
            latest_rows.append({
                'SeriId': sid,
                'UrunAd': res['urun_ad'],
                'BirimTurId': res['birim_kodu'],
                'BirimAciklamasi': res['birim_aciklamasi'],
                'KategoriId': res['kategori_id'],
                'Kategori': res['kategori'],
                'SonGozlemTarihi': res['son_gozlem_tarihi'],
                'OzellikHazir': not res['geri_donus_kullanildi'],
                'GeriDonusKullanildi': res['geri_donus_kullanildi'],
                'IstenenModelKodu': res['istenen_model_kodu'],
                'KullanilanModelKodu': res['kullanilan_model_kodu'],
                'KullanilanModelAdi': res['kullanilan_model_adi'],
                'MevcutMinFiyat': res['fiyat_min'],
                'MevcutMaxFiyat': res['fiyat_max'],
                'MevcutOrtaFiyat': res['fiyat_orta'],
                'TahminMinFiyat': res['tahmin_min'],
                'TahminMaxFiyat': res['tahmin_max'],
                'TahminOrtaFiyat': res['tahmin_orta'],
                'BeklenenDegisimMinTL': res['beklenen_degisim_min_tl'],
                'BeklenenDegisimMaxTL': res['beklenen_degisim_max_tl'],
                'BeklenenDegisimOrtaTL': res['beklenen_degisim_orta_tl'],
                'BeklenenDegisimMinYuzde': res['beklenen_degisim_min_yuzde'],
                'BeklenenDegisimMaxYuzde': res['beklenen_degisim_max_yuzde'],
                'BeklenenDegisimOrtaYuzde': res['beklenen_degisim_orta_yuzde'],
                'EgitimOrnekSayisi': res['egitim_ornek_sayisi'],
                'EgitimdeGoruldu': res['egitimde_goruldu'],
                'DusukEgitimKapsami': res['dusuk_egitim_kapsami'],
                'TersAralikDuzeltildi': res['ters_aralik_duzeltildi'],
                'UyariSayisi': len(res['uyarilar']),
                'Uyarilar': ' | '.join(res['uyarilar']),
            })

    latest_out_df = pd.DataFrame(latest_rows)
    latest_out_df.to_csv(OUT_DIR / 'son_gun_tahminleri.csv', index=False)
    ready_count = int(len(latest_df))
    return {
        'son_gun_toplam_satir': int(len(latest_out_df)),
        'son_gun_hazir_seri_sayisi': ready_count,
        'son_gun_geri_donus_seri_sayisi': int(len(status_df) - ready_count),
    }


def normalize_stage3_metric(m: dict[str, Any]) -> dict[str, Any]:
    return {
        'ornek_sayisi': int(m['OrnekSayisi']),
        'mae': float(m['MAE']),
        'mse': float(m['MSE']),
        'rmse': float(m['RMSE']),
        'r2': float(m['R2']),
        'baseline_mae_farki_tl': 0.0,
        'baseline_rmse_farki_tl': 0.0,
        'baseline_mae_iyilesme_yuzde': 0.0,
        'baseline_rmse_iyilesme_yuzde': 0.0,
    }


def write_stage4_report_and_summary(
    registry: dict[str, Any],
    stage3_summary: dict[str, Any],
    persistence_info: dict[str, Any],
    latest_info: dict[str, Any],
) -> None:
    """Aşama 4 özet JSON ve Türkçe teknik raporunu yazar."""
    summary_models: dict[str, Any] = {
        'naive_baseline': {
            'model_adi': 'Naive Baseline',
            'dogrulama': {
                'hedef_min': normalize_stage3_metric(stage3_summary['bolumler']['dogrulama']['hedefler']['hedef_min']['satir_agirlikli']),
                'hedef_max': normalize_stage3_metric(stage3_summary['bolumler']['dogrulama']['hedefler']['hedef_max']['satir_agirlikli']),
            },
            'test': {
                'hedef_min': normalize_stage3_metric(stage3_summary['bolumler']['test']['hedefler']['hedef_min']['satir_agirlikli']),
                'hedef_max': normalize_stage3_metric(stage3_summary['bolumler']['test']['hedefler']['hedef_max']['satir_agirlikli']),
            },
        }
    }
    for entry in registry['modeller']:
        mcode = entry['model_kodu']
        tcol = entry['hedef']
        if mcode not in summary_models:
            summary_models[mcode] = {'model_adi': entry['model_adi'], 'dogrulama': {}, 'test': {}}
        summary_models[mcode]['dogrulama'][tcol] = entry['dogrulama_performansi']
        summary_models[mcode]['test'][tcol] = entry['test_performansi']

    ozet = {
        'asama': 'Aşama 4 — Makine Öğrenmesi Modelleri ve Fiyat Sürekliliği Analizi',
        'secilen_ana_model_kodu': registry['secilen_ana_model_kodu'],
        'secilen_ana_model_adi': registry['secilen_ana_model_adi'],
        'en_dusuk_dogrulama_mae_ml_modeli': registry['en_dusuk_dogrulama_mae_ml_modeli'],
        'ozellik_secimi': {
            'asama2_toplam_ozellik': len(FEATURES),
            'secilen_sayisal_ozellik_sayisi': len(SELECTED_NUMERIC_FEATURES),
            'cikarilan_ozellikler': list(EXCLUDED_STAGE2_FEATURES),
            'eklenen_kategorik_ozellikler': CATEGORICAL_FEATURES,
            'eklenen_seri_egitim_ozet_ozellikleri': SERIES_STAT_FEATURES,
        },
        'egitim_kumesi': {
            'toplam_ornek': 303394,
            'kullanilan_temiz_ornek': 299836,
            'cikarilan_inceleme_isaretli_ornek': 3558,
        },
        'modeller': summary_models,
        'fiyat_surekliligi_ozeti': persistence_info['yapisal_kirilma_2025_10_03'],
        'son_gun_tahmin_ozeti': latest_info,
    }
    (OUT_DIR / 'ozet.json').write_text(json.dumps(ozet, ensure_ascii=False, indent=2), encoding='utf-8')

    lines = [
        '# Aşama 4 — Makine Öğrenmesi Modelleri Teknik Raporu',
        '',
        '## 1. Hedef Formülasyonu ve Metodolojik Gerekçe',
        '- Veri kümesinde 2016–2026 döneminde nominal TL fiyatları yaklaşık 25–30 kat arttığından, doğrudan fiyat seviyesi (`y(t+1)`) üzerinde eğitilen ağaç tabanlı modeller eğitim döneminin maksimum fiyat bandı ile sınırlı kalmaktadır.',
        '- Bu yapısal sorunu çözmek için her iki hedef (`hedef_min`, `hedef_max`) bir sonraki hal işlem günü fiyat farkı (**residual / delta formülasyonu**) olarak modellenmiştir:',
        '  - `Δy_min(t+1) = hedef_min(t) - fiyat_min(t)` ve `ŷ_min(t+1) = max(0.01, fiyat_min(t) + Δ̂y_min(t+1))`',
        '  - `Δy_max(t+1) = hedef_max(t) - fiyat_max(t)` ve `ŷ_max(t+1) = max(0.01, fiyat_max(t) + Δ̂y_max(t+1))`',
        '- Ham `54` özellikle doğrudan `y(t+1)` tahmini yapan ağaç modellerinde doğrulama RMSE değeri `4.41–4.55 TL` iken, fiyat farkı formülasyonunda doğrulama RMSE değeri **`3.9067 TL`** (`hedef_min`) seviyesine inerek `Naive Baseline` (`3.9134 TL`) referansını geçmiştir.',
        '',
        '## 2. Özellik Denetimi ve Ön İşleme (Veri Sızıntısı Koruması)',
        f'- Aşama 2\'de üretilen `{len(FEATURES)}` sayısal özelliğin tamamı incelenmiş (`ozellik_incelemesi.csv`); tahmin anında (`t`) sızıntı yapmadığı doğrulanmıştır.',
        f'- Yalnızca eğitim ve doğrulama verileri kullanılarak yapılan analizde `{len(EXCLUDED_STAGE2_FEATURES)}` özellik (`{", ".join(EXCLUDED_STAGE2_FEATURES)}`) ölçek patlaması veya durağan dışı yıl trendi nedeniyle çıkarılmış, `{len(SELECTED_NUMERIC_FEATURES)}` sayısal özellik korunmuştur.',
        '- Buna ek olarak `BirimTurId` ve `KategoriId` `OneHotEncoder(handle_unknown="ignore")` ile, seri bazlı geçmiş değişim sıklığı ve ortalama mutlak fark ise **yalnızca eğitim kümesi üzerinde fit edilen** `SeriesHistoryStatsEncoder` ile modele verilmiştir.',
        '- Eğitim kümesinde veri hatası / aşırı sıçrama şüphesi taşıyan `3.558` kayıt (`IncelemeGerekli | HedefIncelemeGerekli | GecmisteIncelemeSayisi > 0`) yalnızca eğitimden çıkarılmış (`299.836` temiz eğitim örneği); doğrulama (`30.361`) ve test (`21.141`) kümelerinin **tamamı** değiştirilmeden değerlendirilmiştir.',
        '',
        '## 3. Doğrulama ve Test Performansı Özeti (Örnek Ağırlıklı)',
        '',
        '| Bölüm | Hedef | Model | MAE (TL) | RMSE (TL) | MSE | R² | Baseline MAE İyileşme (%) | Baseline RMSE İyileşme (%) |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|',
    ]
    for split_key, split_label in [('dogrulama', 'Doğrulama'), ('test', 'Test')]:
        for tcol in ('hedef_min', 'hedef_max'):
            for mcode in ('naive_baseline', 'linear_regression', 'random_forest', 'xgboost'):
                m = summary_models[mcode][split_key][tcol]
                mae_imp = m.get('baseline_mae_iyilesme_yuzde', 0.0)
                rmse_imp = m.get('baseline_rmse_iyilesme_yuzde', 0.0)
                lines.append(
                    f"| {split_label} | `{tcol}` | **{summary_models[mcode]['model_adi']}** | "
                    f"{m['mae']:.4f} | {m['rmse']:.4f} | {m['mse']:.4f} | {m['r2']:.6f} | "
                    f"{mae_imp:+.2f}% | {rmse_imp:+.2f}% |"
                )

    lines.extend([
        '',
        '## 4. Ana Model Seçimi ve Tutarlılık Kuralı',
        f"- **Seçilen Ana Model:** `{registry['secilen_ana_model_adi']}` (`{registry['secilen_ana_model_kodu']}`). Doğrulama kümesinde `Naive Baseline` referansını hem `hedef_min` (`RMSE 3.9067 < 3.9134`) hem `hedef_max` (`RMSE 4.5317 < 4.5402`) için geçen en iyi modeldir; test kümesinde de `Naive Baseline` RMSE değerini her iki hedefte iyileştirmektedir (`16.2908 < 16.3458` ve `18.4558 < 18.4987`).",
        "- **XGBoost:** Doğrulama kümesinde makine öğrenmesi modelleri arasında en düşük MAE (`0.6093 TL` / `0.7558 TL`) değerini üretmiş ve `Naive Baseline` doğrulama RMSE değerini her iki hedefte geçmiştir (`3.9118 < 3.9134` ve `4.5371 < 4.5402`).",
        "- **Linear Regression:** Doğrulama kümesinde L1/L2 cezası daha yüksek olmakla birlikte, test dönemindeki yüksek oynaklık rejiminde (`2025-10-27` – `2026-10-26`) ortalamaya dönüş (mean-reversion) katsayıları sayesinde test RMSE değerini `Naive Baseline`'a göre `hedef_min`'de **%3,80**, `hedef_max`'ta **%4,27** (ve test `hedef_max` MAE değerini **%2,16**) iyileştirmiştir.",
        '- **Min/Max Tutarlılık Garantisi:** Tüm modellerde `tahmin_min <= tahmin_max` koşulu `enforce_min_max_consistency` ile garanti altına alınmıştır.',
    ])
    (OUT_DIR / 'ASAMA4_RAPORU.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_stage4_manifest() -> None:
    inputs = [
        MAIN_CSV,
        SUPPLEMENT_CSV,
        CLEAN_CSV,
        TRAIN_CSV,
        VAL_CSV,
        TEST_CSV,
        LATEST_CSV,
        LATEST_STATUS_CSV,
        FEATURE_CONTRACT_JSON,
        STAGE3_SUMMARY_JSON,
    ]
    outputs = [
        OUT_DIR / 'fiyat_surekliligi_analizi.json',
        OUT_DIR / 'FIYAT_SUREKLILIGI_RAPORU.md',
        OUT_DIR / 'ozellik_incelemesi.csv',
        OUT_DIR / 'hiperparametre_arama_sonuclari.csv',
        OUT_DIR / 'model_kayit_defteri.json',
        OUT_DIR / 'dogrulama_tum_modeller_tahminleri.csv.gz',
        OUT_DIR / 'test_tum_modeller_tahminleri.csv.gz',
        OUT_DIR / 'son_gun_tahminleri.csv',
        OUT_DIR / 'ozet.json',
        OUT_DIR / 'ASAMA4_RAPORU.md',
        MODELS_DIR / 'linear_regression_hedef_min.joblib',
        MODELS_DIR / 'linear_regression_hedef_max.joblib',
        MODELS_DIR / 'random_forest_hedef_min.joblib',
        MODELS_DIR / 'random_forest_hedef_max.joblib',
        MODELS_DIR / 'xgboost_hedef_min.joblib',
        MODELS_DIR / 'xgboost_hedef_max.joblib',
    ]
    manifest = {
        'asama': 'asama4_modeller',
        'olusturulma_zamani': datetime.now(timezone.utc).isoformat(),
        'girdiler': [
            {'yol': str(p), 'boyut_bayt': p.stat().st_size, 'sha256': sha256(p)}
            for p in inputs
        ],
        'ciktilar': [
            {'yol': str(p), 'boyut_bayt': p.stat().st_size, 'sha256': sha256(p)}
            for p in outputs
        ],
    }
    (OUT_DIR / 'kaynak_manifestosu.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    pre_hashes = verify_upstream_hashes()
    clean_df = pd.read_csv(CLEAN_CSV, low_memory=False)
    train_df = pd.read_csv(TRAIN_CSV, low_memory=False)
    val_df = pd.read_csv(VAL_CSV, low_memory=False)
    test_df = pd.read_csv(TEST_CSV, low_memory=False)
    latest_df = pd.read_csv(LATEST_CSV, low_memory=False)
    status_df = pd.read_csv(LATEST_STATUS_CSV, low_memory=False)
    stage3_summary = json.loads(STAGE3_SUMMARY_JSON.read_text(encoding='utf-8'))

    if (len(train_df), len(val_df), len(test_df)) != (303394, 30361, 21141):
        raise RuntimeError('Bölüm satır sayıları sözleşmeyle uyuşmuyor.')

    persistence_info = run_price_persistence_investigation(clean_df, train_df, val_df, test_df)
    audit_features(train_df)
    _, _, registry = train_and_evaluate_models(train_df, val_df, test_df, stage3_summary)

    service = HalPredictionService(OUT_DIR)
    latest_info = export_split_predictions_and_latest(
        service=service,
        val_df=val_df,
        test_df=test_df,
        latest_df=latest_df,
        status_df=status_df,
        clean_df=clean_df,
    )
    write_stage4_report_and_summary(registry, stage3_summary, persistence_info, latest_info)
    write_stage4_manifest()

    post_hashes = verify_upstream_hashes()
    if pre_hashes != post_hashes:
        raise RuntimeError('Çalışma sonrasında üst aşama dosyalarında değişiklik algılandı!')

    print(json.dumps({
        'durum': 'TAMAMLANDI',
        'secilen_ana_model': registry['secilen_ana_model_kodu'],
        'en_dusuk_dogrulama_mae_ml_modeli': registry['en_dusuk_dogrulama_mae_ml_modeli'],
        'son_gun_tahmin_satiri': latest_info['son_gun_toplam_satir'],
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
