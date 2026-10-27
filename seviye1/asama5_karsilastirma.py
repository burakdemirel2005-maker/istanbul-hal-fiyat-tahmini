#!/usr/bin/env python3
"""Aşama 5 — Model Karşılaştırma, Çok Düzeyli Değerlendirme ve Görsel Raporlama.

Aynı doğrulama (30.361) ve test (21.141) kayıtları üzerinde:
  - Naive Baseline
  - Linear Regression
  - Random Forest
  - XGBoost
modellerini hedef_min ve hedef_max için ayrı ayrı karşılaştırır; örnek ağırlıklı,
seri eşit ağırlıklı, eğitimde görülme durumu, kategori, birim kodu, seri ve ay
bazında metrik tabloları ile gerçek veri grafiklerini üretir.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from asama1_veri import CATEGORY, sha256
from asama3_baseline import compute_regression_metrics, unit_label
from tahmin_servisi import HalPredictionService, MODEL_NAMES

ROOT = Path(__file__).resolve().parent
ASAMA4_DIR = ROOT / 'ciktilar/asama4'
OUT_DIR = ROOT / 'ciktilar/asama5'
CHARTS_DIR = OUT_DIR / 'grafikler'

MODEL_ORDER = ['naive_baseline', 'linear_regression', 'random_forest', 'xgboost']


def compute_series_level_table(
    val_preds: pd.DataFrame,
    test_preds: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split_name, df in [('dogrulama', val_preds), ('test', test_preds)]:
        for sid, grp in df.groupby('SeriId', sort=True):
            first = grp.iloc[0]
            for tcol, tshort in [('hedef_min', 'min'), ('hedef_max', 'max')]:
                y_true = grp[tcol].to_numpy(dtype=float)
                base_pred = grp[f'naive_baseline_tahmin_{tshort}'].to_numpy(dtype=float)
                base_m = compute_regression_metrics(y_true, base_pred)
                base_mae = float(base_m['MAE'])
                base_rmse = float(base_m['RMSE'])

                for mcode in MODEL_ORDER:
                    y_pred = grp[f'{mcode}_tahmin_{tshort}'].to_numpy(dtype=float)
                    m = compute_regression_metrics(y_true, y_pred)
                    mae = float(m['MAE'])
                    rmse = float(m['RMSE'])
                    rows.append({
                        'Bolum': split_name,
                        'SeriId': str(sid),
                        'UrunAd': str(first['UrunAd']),
                        'BirimTurId': int(first['BirimTurId']),
                        'BirimAciklamasi': unit_label(int(first['BirimTurId'])),
                        'KategoriId': int(first['KategoriId']),
                        'Kategori': CATEGORY[int(first['KategoriId'])],
                        'EgitimdeGoruldu': bool(first['EgitimdeGoruldu']),
                        'EgitimOrnekSayisi': int(first['EgitimOrnekSayisi']),
                        'Hedef': tcol,
                        'ModelKodu': mcode,
                        'ModelAdi': MODEL_NAMES[mcode],
                        'OrnekSayisi': int(m['OrnekSayisi']),
                        'MAE': mae,
                        'MSE': float(m['MSE']),
                        'RMSE': rmse,
                        'R2': float(m['R2']) if m['R2'] is not None else np.nan,
                        'R2Durumu': str(m['R2Durumu']),
                        'BaselineMAE': base_mae,
                        'BaselineRMSE': base_rmse,
                        'BaselineMAEFarkiTL': mae - base_mae,
                        'BaselineRMSEFarkiTL': rmse - base_rmse,
                        'BaselineMAEIyilesmeYuzde': ((base_mae - mae) / base_mae * 100.0) if base_mae > 1e-12 else 0.0,
                        'BaselineRMSEIyilesmeYuzde': ((base_rmse - rmse) / base_rmse * 100.0) if base_rmse > 1e-12 else 0.0,
                    })
    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_DIR / 'seri_bazli_model_karsilastirma.csv', index=False)
    return out_df


def compute_main_comparison_tables(
    val_preds: pd.DataFrame,
    test_preds: pd.DataFrame,
    series_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Ana karşılaştırma, alt grup (kategori/birim/eğitimde görülme) ve aylık hata tablolarını üretir."""
    main_rows: list[dict[str, Any]] = []
    subgroup_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []

    for split_name, df in [('dogrulama', val_preds), ('test', test_preds)]:
        for tcol, tshort in [('hedef_min', 'min'), ('hedef_max', 'max')]:
            y_true = df[tcol].to_numpy(dtype=float)
            base_pred = df[f'naive_baseline_tahmin_{tshort}'].to_numpy(dtype=float)
            base_m = compute_regression_metrics(y_true, base_pred)
            base_mae = float(base_m['MAE'])
            base_rmse = float(base_m['RMSE'])

            for mcode in MODEL_ORDER:
                y_pred = df[f'{mcode}_tahmin_{tshort}'].to_numpy(dtype=float)
                m = compute_regression_metrics(y_true, y_pred)
                mae = float(m['MAE'])
                rmse = float(m['RMSE'])
                s_sub = series_df[
                    (series_df['Bolum'] == split_name)
                    & (series_df['Hedef'] == tcol)
                    & (series_df['ModelKodu'] == mcode)
                ]
                better_rmse_series = int((s_sub['RMSE'] < s_sub['BaselineRMSE'] - 1e-9).sum())
                equal_rmse_series = int((np.abs(s_sub['RMSE'] - s_sub['BaselineRMSE']) <= 1e-9).sum())

                main_rows.append({
                    'Bolum': split_name,
                    'Hedef': tcol,
                    'ModelKodu': mcode,
                    'ModelAdi': MODEL_NAMES[mcode],
                    'OrnekSayisi': int(m['OrnekSayisi']),
                    'SeriSayisi': int(len(s_sub)),
                    'MAE': mae,
                    'MSE': float(m['MSE']),
                    'RMSE': rmse,
                    'R2': float(m['R2']),
                    'BaselineMAEFarkiTL': mae - base_mae,
                    'BaselineRMSEFarkiTL': rmse - base_rmse,
                    'BaselineMAEIyilesmeYuzde': (base_mae - mae) / base_mae * 100.0,
                    'BaselineRMSEIyilesmeYuzde': (base_rmse - rmse) / base_rmse * 100.0,
                    'SeriEsitAgirlikliMAEOrtalama': float(s_sub['MAE'].mean()),
                    'SeriEsitAgirlikliMAEMedyan': float(s_sub['MAE'].median()),
                    'SeriEsitAgirlikliRMSEOrtalama': float(s_sub['RMSE'].mean()),
                    'SeriEsitAgirlikliRMSEMedyan': float(s_sub['RMSE'].median()),
                    'SeriEsitAgirlikliR2Ortalama': float(s_sub['R2'].dropna().mean()),
                    'BaselineRMSEDahaIyiSeriSayisi': better_rmse_series,
                    'BaselineRMSEEsitSeriSayisi': equal_rmse_series,
                    'TersAralikDuzeltmeSayisi': int(df[f'{mcode}_ters_aralik_duzeltildi'].sum()),
                })

            # Alt gruplar: EgitimdeGoruldu, KategoriId, BirimTurId
            for group_type, col_name in [
                ('EgitimdeGoruldu', 'EgitimdeGoruldu'),
                ('Kategori', 'KategoriId'),
                ('BirimKodu', 'BirimTurId'),
            ]:
                for gval, gdf in df.groupby(col_name, sort=True):
                    gy_true = gdf[tcol].to_numpy(dtype=float)
                    gbase_m = compute_regression_metrics(gy_true, gdf[f'naive_baseline_tahmin_{tshort}'].to_numpy(dtype=float))
                    gbase_mae = float(gbase_m['MAE'])
                    gbase_rmse = float(gbase_m['RMSE'])
                    if group_type == 'Kategori':
                        glabel = CATEGORY[int(gval)]
                    elif group_type == 'BirimKodu':
                        glabel = unit_label(int(gval))
                    else:
                        glabel = 'Eğitimde Görüldü' if bool(gval) else 'Eğitimde Görülmedi'

                    for mcode in MODEL_ORDER:
                        gm = compute_regression_metrics(gy_true, gdf[f'{mcode}_tahmin_{tshort}'].to_numpy(dtype=float))
                        gmae = float(gm['MAE'])
                        grmse = float(gm['RMSE'])
                        subgroup_rows.append({
                            'Bolum': split_name,
                            'Hedef': tcol,
                            'AltGrupTuru': group_type,
                            'AltGrupDegeri': str(gval),
                            'AltGrupAciklamasi': glabel,
                            'ModelKodu': mcode,
                            'ModelAdi': MODEL_NAMES[mcode],
                            'OrnekSayisi': int(gm['OrnekSayisi']),
                            'SeriSayisi': int(gdf['SeriId'].nunique()),
                            'MAE': gmae,
                            'MSE': float(gm['MSE']),
                            'RMSE': grmse,
                            'R2': float(gm['R2']) if gm['R2'] is not None else np.nan,
                            'BaselineMAEIyilesmeYuzde': ((gbase_mae - gmae) / gbase_mae * 100.0) if gbase_mae > 1e-12 else 0.0,
                            'BaselineRMSEIyilesmeYuzde': ((gbase_rmse - grmse) / gbase_rmse * 100.0) if gbase_rmse > 1e-12 else 0.0,
                        })

            # Aylık hata analizi
            df_month = df.copy()
            df_month['YilAy'] = df_month['HedefTarih'].str.slice(0, 7)
            for ym, mdf in df_month.groupby('YilAy', sort=True):
                my_true = mdf[tcol].to_numpy(dtype=float)
                for mcode in MODEL_ORDER:
                    mm = compute_regression_metrics(my_true, mdf[f'{mcode}_tahmin_{tshort}'].to_numpy(dtype=float))
                    monthly_rows.append({
                        'Bolum': split_name,
                        'YilAy': str(ym),
                        'Hedef': tcol,
                        'ModelKodu': mcode,
                        'ModelAdi': MODEL_NAMES[mcode],
                        'OrnekSayisi': int(mm['OrnekSayisi']),
                        'SeriSayisi': int(mdf['SeriId'].nunique()),
                        'MAE': float(mm['MAE']),
                        'RMSE': float(mm['RMSE']),
                        'MSE': float(mm['MSE']),
                    })

    main_df = pd.DataFrame(main_rows)
    subgroup_df = pd.DataFrame(subgroup_rows)
    monthly_df = pd.DataFrame(monthly_rows)

    main_df.to_csv(OUT_DIR / 'model_karsilastirma_ozeti.csv', index=False)
    subgroup_df.to_csv(OUT_DIR / 'alt_grup_karsilastirma.csv', index=False)
    monthly_df.to_csv(OUT_DIR / 'aylik_hata_analizi.csv', index=False)
    return main_df, subgroup_df, monthly_df


def extract_feature_importances(service: HalPredictionService) -> pd.DataFrame:
    """Eğitilmiş modellerden özellik önemlerini / standartlaştırılmış katsayı büyüklüklerini çıkarır."""
    rows: list[dict[str, Any]] = []
    for mcode in ('linear_regression', 'random_forest', 'xgboost'):
        for tcol in ('hedef_min', 'hedef_max'):
            art = service.artifacts[(mcode, tcol)]
            pipe = art.pipeline
            pre = pipe.named_steps['preprocessor']
            reg = pipe.named_steps['regressor']
            feat_names = [name.split('__', 1)[-1] for name in pre.get_feature_names_out()]

            if mcode == 'linear_regression':
                raw_vals = np.abs(np.asarray(reg.coef_, dtype=float))
                metric_type = 'mutlak_standart_katsayi'
            else:
                raw_vals = np.asarray(reg.feature_importances_, dtype=float)
                metric_type = 'agac_ozellik_onemi'

            total = float(raw_vals.sum())
            norm_vals = raw_vals / total if total > 0 else raw_vals
            order = np.argsort(-norm_vals)
            for rank_idx, col_idx in enumerate(order, start=1):
                rows.append({
                    'ModelKodu': mcode,
                    'ModelAdi': MODEL_NAMES[mcode],
                    'Hedef': tcol,
                    'Sira': rank_idx,
                    'OzellikAdi': feat_names[col_idx],
                    'OnemOlcutu': metric_type,
                    'HamDeger': float(raw_vals[col_idx]),
                    'NormalizeOnem': float(norm_vals[col_idx]),
                })
    imp_df = pd.DataFrame(rows)
    imp_df.to_csv(OUT_DIR / 'ozellik_onemleri.csv', index=False)
    return imp_df


def generate_comparison_charts(
    main_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    series_df: pd.DataFrame,
    imp_df: pd.DataFrame,
    test_preds: pd.DataFrame,
) -> list[Path]:
    """Bölüm 7.5 kapsamındaki tüm karşılaştırma ve hata grafiklerini üretir."""
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['axes.unicode_minus'] = False
    saved_paths: list[Path] = []

    colors = {
        'Naive Baseline': '#64748b',
        'Linear Regression': '#2563eb',
        'Random Forest': '#059669',
        'XGBoost': '#d97706',
    }

    # 1. MAE Karşılaştırma Grafiği
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=150)
    for ax, split_name, title_prefix in zip(axes, ['dogrulama', 'test'], ['Doğrulama (2024–2025)', 'Test (2025–2026)']):
        sub = main_df[main_df['Bolum'] == split_name]
        x = np.arange(2)
        width = 0.19
        for i, mcode in enumerate(MODEL_ORDER):
            mname = MODEL_NAMES[mcode]
            vals = [
                float(sub[(sub['Hedef'] == 'hedef_min') & (sub['ModelKodu'] == mcode)]['MAE'].iloc[0]),
                float(sub[(sub['Hedef'] == 'hedef_max') & (sub['ModelKodu'] == mcode)]['MAE'].iloc[0]),
            ]
            bars = ax.bar(x + (i - 1.5) * width, vals, width, label=mname, color=colors[mname])
            for b in bars:
                ax.text(
                    b.get_x() + b.get_width() / 2.0,
                    b.get_height() + 0.015 * max(vals),
                    f'{b.get_height():.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=7.5,
                )
        ax.set_xticks(x)
        ax.set_xticklabels(['Minimum Fiyat (hedef_min)', 'Maksimum Fiyat (hedef_max)'])
        ax.set_ylabel('MAE (TL)')
        ax.set_title(f'{title_prefix} — Modele Göre MAE')
        ax.grid(axis='y', linestyle='--', alpha=0.35)
        ax.legend(fontsize=8.5)
    fig.tight_layout()
    p1 = CHARTS_DIR / 'model_mae_karsilastirma.png'
    fig.savefig(p1)
    plt.close(fig)
    saved_paths.append(p1)

    # 2. RMSE Karşılaştırma Grafiği
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=150)
    for ax, split_name, title_prefix in zip(axes, ['dogrulama', 'test'], ['Doğrulama (2024–2025)', 'Test (2025–2026)']):
        sub = main_df[main_df['Bolum'] == split_name]
        x = np.arange(2)
        width = 0.19
        for i, mcode in enumerate(MODEL_ORDER):
            mname = MODEL_NAMES[mcode]
            vals = [
                float(sub[(sub['Hedef'] == 'hedef_min') & (sub['ModelKodu'] == mcode)]['RMSE'].iloc[0]),
                float(sub[(sub['Hedef'] == 'hedef_max') & (sub['ModelKodu'] == mcode)]['RMSE'].iloc[0]),
            ]
            bars = ax.bar(x + (i - 1.5) * width, vals, width, label=mname, color=colors[mname])
            for b in bars:
                ax.text(
                    b.get_x() + b.get_width() / 2.0,
                    b.get_height() + 0.01 * max(vals),
                    f'{b.get_height():.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=7.5,
                )
        ax.set_xticks(x)
        ax.set_xticklabels(['Minimum Fiyat (hedef_min)', 'Maksimum Fiyat (hedef_max)'])
        ax.set_ylabel('RMSE (TL)')
        ax.set_title(f'{title_prefix} — Modele Göre RMSE')
        ax.grid(axis='y', linestyle='--', alpha=0.35)
        ax.legend(fontsize=8.5)
    fig.tight_layout()
    p2 = CHARTS_DIR / 'model_rmse_karsilastirma.png'
    fig.savefig(p2)
    plt.close(fig)
    saved_paths.append(p2)

    # 3. Gerçek ve Tahmin Edilen Fiyat Karşılaştırması (Test Kümesi Örnek Ürünler)
    sample_names = ['Domates', 'Salatalık', 'Patates', 'Biber (Sivri)']
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), dpi=150)
    for ax, uname in zip(axes.flatten(), sample_names):
        sub_p = test_preds[test_preds['UrunAd'] == uname].sort_values('HedefTarih').tail(60)
        if sub_p.empty:
            sub_p = test_preds[test_preds['UrunAd'].str.contains(uname.split()[0], regex=False)].sort_values('HedefTarih').tail(60)
        dates = pd.to_datetime(sub_p['HedefTarih'])
        ax.plot(dates, sub_p['hedef_min'], label='Gerçek Min Fiyat', color='#111827', linewidth=2.0)
        ax.plot(dates, sub_p['random_forest_tahmin_min'], label='Random Forest (Ana Model)', color='#059669', linestyle='--', linewidth=1.6)
        ax.plot(dates, sub_p['linear_regression_tahmin_min'], label='Linear Regression', color='#2563eb', linestyle=':', linewidth=1.5)
        ax.set_title(f"{sub_p['UrunAd'].iloc[0]} — Son 60 İşlem Günü (hedef_min)")
        ax.set_ylabel('Fiyat (TL)')
        ax.grid(True, linestyle='--', alpha=0.35)
        ax.legend(fontsize=8)
        ax.tick_params(axis='x', rotation=30)
    fig.tight_layout()
    p3 = CHARTS_DIR / 'gercek_vs_tahmin_ornek_urunler.png'
    fig.savefig(p3)
    plt.close(fig)
    saved_paths.append(p3)

    # 4. Zamana Göre Hata Değişimi (Aylık RMSE — Doğrulama ve Test)
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 7.8), dpi=150, sharex=True)
    for ax, tcol, tlabel in zip(axes, ['hedef_min', 'hedef_max'], ['Minimum Fiyat (hedef_min)', 'Maksimum Fiyat (hedef_max)']):
        sub_m = monthly_df[monthly_df['Hedef'] == tcol].copy()
        sub_m['DonemEtiketi'] = sub_m['YilAy'] + ' (' + sub_m['Bolum'].map({'dogrulama': 'Doğr.', 'test': 'Test'}) + ')'
        sub_m = sub_m.sort_values(['Bolum', 'YilAy'])
        months = list(dict.fromkeys(sub_m['DonemEtiketi']))
        for mcode in MODEL_ORDER:
            mname = MODEL_NAMES[mcode]
            m_series = sub_m[sub_m['ModelKodu'] == mcode].set_index('DonemEtiketi').reindex(months)
            ax.plot(months, m_series['RMSE'], marker='o', markersize=4, linewidth=1.7, label=mname, color=colors[mname])
        if '2025-10 (Doğr.)' in months:
            idx_oct = months.index('2025-10 (Doğr.)')
            ax.axvline(idx_oct, color='#dc2626', linestyle='--', linewidth=1.3, label='2025-10-03 Rejim Kırılması')
        ax.set_ylabel('Aylık RMSE (TL)')
        ax.set_title(f'Zamana Göre Aylık RMSE Değişimi — {tlabel}')
        ax.grid(True, linestyle='--', alpha=0.35)
        ax.legend(fontsize=8.5, ncol=3)
        ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    p4 = CHARTS_DIR / 'zamana_gore_hata_degisimi.png'
    fig.savefig(p4)
    plt.close(fig)
    saved_paths.append(p4)

    # 5. Ürün Bazında Tahmin Hataları (Test Kümesi En Çok Örnekli 15 Ürün — hedef_min)
    top_series = (
        series_df[(series_df['Bolum'] == 'test') & (series_df['Hedef'] == 'hedef_min') & (series_df['ModelKodu'] == 'random_forest')]
        .sort_values(['OrnekSayisi', 'RMSE'], ascending=[False, False])
        .head(15)
    )
    top_sids = list(top_series['SeriId'])
    sub_top = series_df[
        (series_df['Bolum'] == 'test')
        & (series_df['Hedef'] == 'hedef_min')
        & (series_df['SeriId'].isin(top_sids))
    ]
    fig, ax = plt.subplots(figsize=(13.5, 6.5), dpi=150)
    y_pos = np.arange(len(top_sids))
    height = 0.2
    labels = [str(top_series.loc[top_series['SeriId'] == s, 'UrunAd'].iloc[0]) for s in top_sids]
    for i, mcode in enumerate(MODEL_ORDER):
        mname = MODEL_NAMES[mcode]
        vals = [
            float(sub_top[(sub_top['SeriId'] == s) & (sub_top['ModelKodu'] == mcode)]['MAE'].iloc[0])
            for s in top_sids
        ]
        ax.barh(y_pos + (i - 1.5) * height, vals, height, label=mname, color=colors[mname])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Test MAE (TL)')
    ax.set_title('Ürün Bazında Test Dönemi Tahmin Hataları (hedef_min — En Yüksek Kapsamlı 15 Ürün)')
    ax.grid(axis='x', linestyle='--', alpha=0.35)
    ax.legend(fontsize=9)
    fig.tight_layout()
    p5 = CHARTS_DIR / 'urun_bazli_hata_dagilimi.png'
    fig.savefig(p5)
    plt.close(fig)
    saved_paths.append(p5)

    # 6. Özellik Önem Düzeyleri (Random Forest & XGBoost İlk 12 Özellik)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), dpi=150)
    for ax, mcode in zip(axes, ['random_forest', 'xgboost']):
        mname = MODEL_NAMES[mcode]
        sub_i = imp_df[(imp_df['ModelKodu'] == mcode) & (imp_df['Hedef'] == 'hedef_min')].sort_values('Sira').head(12)
        ax.barh(sub_i['OzellikAdi'][::-1], sub_i['NormalizeOnem'][::-1] * 100.0, color=colors[mname])
        ax.set_xlabel('Göreli Önem (%)')
        ax.set_title(f'{mname} — En Önemli 12 Özellik (hedef_min)')
        ax.grid(axis='x', linestyle='--', alpha=0.35)
    fig.tight_layout()
    p6 = CHARTS_DIR / 'ozellik_onem_duzeyleri.png'
    fig.savefig(p6)
    plt.close(fig)
    saved_paths.append(p6)

    return saved_paths


def write_stage5_summary_and_report(
    main_df: pd.DataFrame,
    subgroup_df: pd.DataFrame,
    series_df: pd.DataFrame,
    imp_df: pd.DataFrame,
    registry: dict[str, Any],
    chart_paths: list[Path],
) -> None:
    """Aşama 5 özet JSON, Markdown raporu ve kaynak manifestosunu yazar."""
    ozet = {
        'asama': 'Aşama 5 — Model Karşılaştırma ve Çok Düzeyli Değerlendirme',
        'secilen_ana_model_kodu': registry['secilen_ana_model_kodu'],
        'secilen_ana_model_adi': registry['secilen_ana_model_adi'],
        'en_dusuk_dogrulama_mae_ml_modeli': registry['en_dusuk_dogrulama_mae_ml_modeli'],
        'karsilastirilan_modeller': [MODEL_NAMES[m] for m in MODEL_ORDER],
        'ana_tablo': main_df.to_dict(orient='records'),
        'uretilen_grafikler': [p.name for p in chart_paths],
    }
    (OUT_DIR / 'ozet.json').write_text(json.dumps(ozet, ensure_ascii=False, indent=2), encoding='utf-8')

    lines = [
        '# Aşama 5 — Model Karşılaştırma ve Değerlendirme Raporu',
        '',
        '## 1. Karşılaştırma Protokolü',
        '- Tüm modeller (`Naive Baseline`, `Linear Regression`, `Random Forest`, `XGBoost`) **birebir aynı** doğrulama (`30.361` kayıt, `152` seri) ve test (`21.141` kayıt, `123` seri) örnekleri üzerinde değerlendirilmiştir.',
        '- Model seçimi ve hiperparametre kilitlemesi yalnızca doğrulama kümesi üzerinde tamamlanmış, test kümesi tek seferlik nihai performans raporlaması için kullanılmıştır.',
        '',
        '## 2. Ana Karşılaştırma Tablosu (Örnek Ağırlıklı ve Seri Eşit Ağırlıklı)',
        '',
        '| Bölüm | Hedef | Model | MAE (TL) | RMSE (TL) | MSE | R² | Baseline MAE İyileşme (%) | Baseline RMSE İyileşme (%) | Seri Ort. MAE | Seri Ort. RMSE | Baseline\'dan İyi Seri Sayısı |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for _, r in main_df.iterrows():
        lines.append(
            f"| {r['Bolum']} | `{r['Hedef']}` | **{r['ModelAdi']}** | "
            f"{r['MAE']:.4f} | {r['RMSE']:.4f} | {r['MSE']:.4f} | {r['R2']:.6f} | "
            f"{r['BaselineMAEIyilesmeYuzde']:+.2f}% | {r['BaselineRMSEIyilesmeYuzde']:+.2f}% | "
            f"{r['SeriEsitAgirlikliMAEOrtalama']:.4f} | {r['SeriEsitAgirlikliRMSEOrtalama']:.4f} | "
            f"{int(r['BaselineRMSEDahaIyiSeriSayisi'])}/{int(r['SeriSayisi'])} |"
        )

    lines.extend([
        '',
        '## 3. Eğitimde Görülen ve Görülmeyen Seriler Bazında Performans',
        '',
        '| Bölüm | Hedef | Alt Grup | Model | Örnek | Seri | MAE (TL) | RMSE (TL) | Baseline RMSE İyileşme (%) |',
        '|---|---|---|---|---:|---:|---:|---:|---:|',
    ])
    sub_seen = subgroup_df[subgroup_df['AltGrupTuru'] == 'EgitimdeGoruldu']
    for _, r in sub_seen.iterrows():
        lines.append(
            f"| {r['Bolum']} | `{r['Hedef']}` | {r['AltGrupAciklamasi']} | **{r['ModelAdi']}** | "
            f"{int(r['OrnekSayisi']):,} | {int(r['SeriSayisi'])} | {r['MAE']:.4f} | {r['RMSE']:.4f} | "
            f"{r['BaselineRMSEIyilesmeYuzde']:+.2f}% |"
        )

    lines.extend([
        '',
        '## 4. En Önemli Özellikler (İlk 10 — `hedef_min`)',
        '',
        '| Sıra | Random Forest (Göreli Önem %) | XGBoost (Göreli Önem %) | Linear Regression (|Standart Katsayı| %) |',
        '|---:|---|---|---|',
    ])
    for rank in range(1, 11):
        rf_r = imp_df[(imp_df['ModelKodu'] == 'random_forest') & (imp_df['Hedef'] == 'hedef_min') & (imp_df['Sira'] == rank)].iloc[0]
        xgb_r = imp_df[(imp_df['ModelKodu'] == 'xgboost') & (imp_df['Hedef'] == 'hedef_min') & (imp_df['Sira'] == rank)].iloc[0]
        lr_r = imp_df[(imp_df['ModelKodu'] == 'linear_regression') & (imp_df['Hedef'] == 'hedef_min') & (imp_df['Sira'] == rank)].iloc[0]
        lines.append(
            f"| {rank} | `{rf_r['OzellikAdi']}` (%{rf_r['NormalizeOnem']*100:.2f}) | "
            f"`{xgb_r['OzellikAdi']}` (%{xgb_r['NormalizeOnem']*100:.2f}) | "
            f"`{lr_r['OzellikAdi']}` (%{lr_r['NormalizeOnem']*100:.2f}) |"
        )

    lines.extend([
        '',
        '## 5. Üretilen Görseller (`ciktilar/asama5/grafikler/`)',
    ])
    for p in chart_paths:
        lines.append(f'- `{p.name}`')

    (OUT_DIR / 'ASAMA5_RAPORU.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    inputs = [
        ASAMA4_DIR / 'dogrulama_tum_modeller_tahminleri.csv.gz',
        ASAMA4_DIR / 'test_tum_modeller_tahminleri.csv.gz',
        ASAMA4_DIR / 'model_kayit_defteri.json',
        ASAMA4_DIR / 'ozet.json',
    ]
    outputs = [
        OUT_DIR / 'model_karsilastirma_ozeti.csv',
        OUT_DIR / 'alt_grup_karsilastirma.csv',
        OUT_DIR / 'seri_bazli_model_karsilastirma.csv',
        OUT_DIR / 'aylik_hata_analizi.csv',
        OUT_DIR / 'ozellik_onemleri.csv',
        OUT_DIR / 'ozet.json',
        OUT_DIR / 'ASAMA5_RAPORU.md',
        *chart_paths,
    ]
    manifest = {
        'asama': 'asama5_karsilastirma',
        'olusturulma_zamani': datetime.now(timezone.utc).isoformat(),
        'girdiler': [{'yol': str(p), 'boyut_bayt': p.stat().st_size, 'sha256': sha256(p)} for p in inputs],
        'ciktilar': [{'yol': str(p), 'boyut_bayt': p.stat().st_size, 'sha256': sha256(p)} for p in outputs],
    }
    (OUT_DIR / 'kaynak_manifestosu.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    val_preds = pd.read_csv(ASAMA4_DIR / 'dogrulama_tum_modeller_tahminleri.csv.gz', low_memory=False)
    test_preds = pd.read_csv(ASAMA4_DIR / 'test_tum_modeller_tahminleri.csv.gz', low_memory=False)
    registry = json.loads((ASAMA4_DIR / 'model_kayit_defteri.json').read_text(encoding='utf-8'))
    service = HalPredictionService(ASAMA4_DIR)

    series_df = compute_series_level_table(val_preds, test_preds)
    main_df, subgroup_df, monthly_df = compute_main_comparison_tables(val_preds, test_preds, series_df)
    imp_df = extract_feature_importances(service)
    chart_paths = generate_comparison_charts(main_df, monthly_df, series_df, imp_df, test_preds)
    write_stage5_summary_and_report(main_df, subgroup_df, series_df, imp_df, registry, chart_paths)

    print(json.dumps({
        'durum': 'TAMAMLANDI',
        'ana_tablo_satir': len(main_df),
        'seri_tablo_satir': len(series_df),
        'grafik_sayisi': len(chart_paths),
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
