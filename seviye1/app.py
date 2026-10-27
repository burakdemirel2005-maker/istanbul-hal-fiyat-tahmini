#!/usr/bin/env python3
"""İstanbul Hal Fiyat Tahmini ve Satın Alma Karar Destek Sistemi — Streamlit Web Uygulaması.

Kahramanmaraş Sütçü İmam Üniversitesi — Bilgisayar Mühendisliği Bölümü
Makine Öğrenmesi Dersi Projesi
Geliştiriciler:
  - Burak Demirel
  - Ebubekir Samet Gözübenli
  - Resul Kılınç
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from asama1_veri import CATEGORY
from asama3_baseline import unit_label
from asama6_satin_alma import PurchasingDecisionSupport, VALID_PRICE_TYPES
from tahmin_servisi import HalPredictionService, MODEL_NAMES

ROOT = Path(__file__).resolve().parent
ASAMA1_DIR = ROOT / 'ciktilar/asama1'
ASAMA2_DIR = ROOT / 'ciktilar/asama2'
ASAMA3_DIR = ROOT / 'ciktilar/asama3'
ASAMA4_DIR = ROOT / 'ciktilar/asama4'
ASAMA5_DIR = ROOT / 'ciktilar/asama5'
ASAMA6_DIR = ROOT / 'ciktilar/asama6'

PAGES = [
    '1. Ana Panel',
    '2. Ürün Analizi',
    '3. Fiyat Tahmini',
    '4. Satın Alma Planlama',
    '5. Model Karşılaştırma',
    '6. Proje Hakkında',
]

MODEL_COLORS = {
    'Naive Baseline': '#64748b',
    'Linear Regression': '#2563eb',
    'Random Forest': '#059669',
    'XGBoost': '#d97706',
}


def resolve_clean_data_path(base_dir: Path | None = None) -> Path:
    """Yerelde temiz_veri.csv, GitHub/bulut ortamında temiz_veri.csv.gz dosyasını çözer."""
    bdir = Path(base_dir).resolve() if base_dir else ROOT
    raw_csv = bdir / 'ciktilar/asama1/temiz_veri.csv'
    gz_csv = bdir / 'ciktilar/asama1/temiz_veri.csv.gz'
    if raw_csv.is_file():
        return raw_csv
    if gz_csv.is_file():
        return gz_csv
    return raw_csv


def verify_required_app_files(base_dir: Path | None = None) -> list[Path]:
    """Web uygulamasının ihtiyaç duyduğu veri ve model dosyalarının varlığını kontrol eder."""
    bdir = Path(base_dir).resolve() if base_dir else ROOT
    clean_path = resolve_clean_data_path(bdir)
    required = [
        clean_path,
        bdir / 'ciktilar/asama1/ozet.json',
        bdir / 'ciktilar/asama2/ozet.json',
        bdir / 'ciktilar/asama2/son_gun_girdileri.csv',
        bdir / 'ciktilar/asama2/son_gozlem_durumu.csv',
        bdir / 'ciktilar/asama4/model_kayit_defteri.json',
        bdir / 'ciktilar/asama4/son_gun_tahminleri.csv',
        bdir / 'ciktilar/asama4/fiyat_surekliligi_analizi.json',
        bdir / 'ciktilar/asama5/model_karsilastirma_ozeti.csv',
        bdir / 'ciktilar/asama5/alt_grup_karsilastirma.csv',
        bdir / 'ciktilar/asama5/seri_bazli_model_karsilastirma.csv',
        bdir / 'ciktilar/asama5/aylik_hata_analizi.csv',
        bdir / 'ciktilar/asama5/ozellik_onemleri.csv',
        bdir / 'ciktilar/asama6/ornek_satin_alma_senaryolari.csv',
        bdir / 'ciktilar/asama6/ornek_sepet_simulasyonu.json',
    ]
    missing = [p for p in required if not p.is_file()]
    return missing


@st.cache_resource(show_spinner=False)
def get_prediction_service() -> HalPredictionService:
    return HalPredictionService(ASAMA4_DIR)


@st.cache_resource(show_spinner=False)
def get_decision_support() -> PurchasingDecisionSupport:
    return PurchasingDecisionSupport(ASAMA4_DIR, ASAMA5_DIR)


@st.cache_data(show_spinner=False)
def load_dashboard_bundle() -> dict[str, Any]:
    """Uygulama verilerini tek seferde önbelleğe yükler."""
    missing = verify_required_app_files(ROOT)
    if missing:
        raise FileNotFoundError(
            'Uygulama için zorunlu çıktı dosyaları eksik: ' + ', '.join(str(p.relative_to(ROOT)) for p in missing)
        )

    clean_df = pd.read_csv(resolve_clean_data_path(ROOT), low_memory=False)
    clean_df['Kategori'] = clean_df['KategoriId'].map(CATEGORY)
    clean_df['OrtaFiyat'] = (clean_df['EnDusukFiyat'] + clean_df['EnYuksekFiyat']) / 2.0
    clean_df['FiyatAraligi'] = clean_df['EnYuksekFiyat'] - clean_df['EnDusukFiyat']

    latest_features_df = pd.read_csv(ASAMA2_DIR / 'son_gun_girdileri.csv', low_memory=False)
    latest_status_df = pd.read_csv(ASAMA2_DIR / 'son_gozlem_durumu.csv', low_memory=False)
    latest_preds_df = pd.read_csv(ASAMA4_DIR / 'son_gun_tahminleri.csv', low_memory=False)

    main_comp_df = pd.read_csv(ASAMA5_DIR / 'model_karsilastirma_ozeti.csv', low_memory=False)
    subgroup_comp_df = pd.read_csv(ASAMA5_DIR / 'alt_grup_karsilastirma.csv', low_memory=False)
    series_comp_df = pd.read_csv(ASAMA5_DIR / 'seri_bazli_model_karsilastirma.csv', low_memory=False)
    monthly_err_df = pd.read_csv(ASAMA5_DIR / 'aylik_hata_analizi.csv', low_memory=False)
    feature_imp_df = pd.read_csv(ASAMA5_DIR / 'ozellik_onemleri.csv', low_memory=False)
    test_preds_df = pd.read_csv(ASAMA4_DIR / 'test_tum_modeller_tahminleri.csv.gz', low_memory=False)

    s1_summary = json.loads((ASAMA1_DIR / 'ozet.json').read_text(encoding='utf-8'))
    s2_summary = json.loads((ASAMA2_DIR / 'ozet.json').read_text(encoding='utf-8'))
    s4_registry = json.loads((ASAMA4_DIR / 'model_kayit_defteri.json').read_text(encoding='utf-8'))
    persistence_json = json.loads((ASAMA4_DIR / 'fiyat_surekliligi_analizi.json').read_text(encoding='utf-8'))
    sample_basket_json = json.loads((ASAMA6_DIR / 'ornek_sepet_simulasyonu.json').read_text(encoding='utf-8'))

    # Seri seçim sözlüğü (okunabilir Türkçe etiket -> SeriId)
    series_meta = (
        latest_preds_df[latest_preds_df['IstenenModelKodu'] == s4_registry['secilen_ana_model_kodu']]
        .sort_values(['OzellikHazir', 'UrunAd', 'SeriId'], ascending=[False, True, True])
        .reset_index(drop=True)
    )
    series_meta['GorunenEtiket'] = series_meta.apply(
        lambda r: (
            f"{r['UrunAd']} | {r['Kategori']} | Birim Kodu {int(r['BirimTurId'])} | "
            f"{'Aktif (2026-10-26)' if bool(r['OzellikHazir']) else 'Kesintili (Son: ' + str(r['SonGozlemTarihi']) + ')'} "
            f"[{str(r['SeriId']).split('|')[0][:8]}]"
        ),
        axis=1,
    )

    return {
        'clean_df': clean_df,
        'latest_features_df': latest_features_df,
        'latest_status_df': latest_status_df,
        'latest_preds_df': latest_preds_df,
        'main_comp_df': main_comp_df,
        'subgroup_comp_df': subgroup_comp_df,
        'series_comp_df': series_comp_df,
        'monthly_err_df': monthly_err_df,
        'feature_imp_df': feature_imp_df,
        'test_preds_df': test_preds_df,
        's1_summary': s1_summary,
        's2_summary': s2_summary,
        's4_registry': s4_registry,
        'persistence_json': persistence_json,
        'sample_basket_json': sample_basket_json,
        'series_meta': series_meta,
    }


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        .main-header {
            background: linear-gradient(135deg, #064e3b 0%, #047857 60%, #059669 100%);
            color: #ffffff;
            padding: 1.4rem 1.8rem;
            border-radius: 14px;
            margin-bottom: 1.2rem;
            box-shadow: 0 4px 14px rgba(5, 150, 105, 0.16);
        }
        .main-header h1 {
            color: #ffffff !important;
            font-size: 1.65rem !important;
            margin: 0 0 0.35rem 0 !important;
            font-weight: 700 !important;
        }
        .main-header p {
            color: #ecfdf5 !important;
            margin: 0 !important;
            font-size: 0.96rem !important;
        }
        .info-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-left: 5px solid #059669;
            border-radius: 10px;
            padding: 1rem 1.2rem;
            margin-bottom: 0.9rem;
            box-shadow: 0 1px 4px rgba(15, 23, 42, 0.04);
        }
        .warn-card {
            background: #fffbeb;
            border: 1px solid #fde68a;
            border-left: 5px solid #d97706;
            border-radius: 10px;
            padding: 0.95rem 1.15rem;
            margin-bottom: 0.9rem;
            color: #78350f;
        }
        .badge-up {
            display: inline-block;
            background: #fee2e2;
            color: #991b1b;
            padding: 0.25rem 0.65rem;
            border-radius: 999px;
            font-weight: 600;
            font-size: 0.85rem;
        }
        .badge-down {
            display: inline-block;
            background: #dcfce7;
            color: #166534;
            padding: 0.25rem 0.65rem;
            border-radius: 999px;
            font-weight: 600;
            font-size: 0.85rem;
        }
        .badge-neutral {
            display: inline-block;
            background: #f1f5f9;
            color: #334155;
            padding: 0.25rem 0.65rem;
            border-radius: 999px;
            font-weight: 600;
            font-size: 0.85rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="main-header">
            <h1>İstanbul Hal Ürünlerinin Makine Öğrenmesi ile Fiyat Tahmini ve Satın Alma Planlaması</h1>
            <p>İBB Hal Fiyat Verileri (2016–2026) • Veri Sızıntısız Zaman Serisi Modellemesi • Çoklu Ürün Satın Alma Karar Destek Sistemi</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page_ana_panel(bundle: dict[str, Any]) -> None:
    st.subheader('Genel Veri ve Model Özeti')
    s4_reg = bundle['s4_registry']
    series_meta = bundle['series_meta']
    ready_df = series_meta[series_meta['OzellikHazir']].copy()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric('Temiz Gözlem', '380.985', 'Ham: 382.897')
    c2.metric('Etiketli Örnek', '354.896', '54 Aşama-2 Özelliği')
    c3.metric('Tekil Ürün / Seri', '252 / 299', 'Ad & Birim Ayrımı')
    c4.metric('Son Gün Aktif Seri', '91', '2026-10-26 İşlem Günü')
    c5.metric('Veri Tarih Aralığı', '2016 – 2026', '2.734 Ortak İşlem Günü')
    c6.metric('Seçilen Ana Model', s4_reg['secilen_ana_model_adi'], 'Doğrulama RMSE Lideri')

    st.markdown(
        """
        <div class="info-card">
            <strong>Metodolojik Özet:</strong> Veri kümesi kronolojik olarak <strong>Eğitim (303.394 örnek, 2016–2024)</strong>,
            <strong>Doğrulama (30.361 örnek, 2024–2025)</strong> ve <strong>Test (21.141 örnek, 2025–2026)</strong> olarak bölünmüştür.
            Ölçekleme ve seri geçmişi özetleri yalnızca eğitim kümesinde öğrenilmiş; hiperparametre ve model seçimi yalnızca doğrulama kümesinde kilitlenmiştir.
            Birim kodları kaynak CSV'de metin sözlüğü içermediğinden <em>"Birim kodu X — adı doğrulanmadı"</em> olarak korunmakta ve farklı birim kodları birbirine karıştırılmamaktadır.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns([1.05, 1.25])
    with col_left:
        st.markdown('#### Kategori Bazında Seri ve Gözlem Dağılımı')
        cat_summary = (
            bundle['clean_df']
            .groupby('Kategori', as_index=False)
            .agg(
                GozlemSayisi=('EnDusukFiyat', 'size'),
                SeriSayisi=('SeriId', 'nunique'),
                UrunSayisi=('TabelaGId', 'nunique'),
            )
            .sort_values('GozlemSayisi', ascending=False)
        )
        fig_cat = px.bar(
            cat_summary,
            x='Kategori',
            y='GozlemSayisi',
            color='Kategori',
            text='GozlemSayisi',
            color_discrete_sequence=['#059669', '#2563eb', '#d97706'],
            labels={'GozlemSayisi': 'Temiz Gözlem Sayısı', 'Kategori': 'Kategori'},
            title='Kategorilere Göre Temiz Gözlem Sayısı (2016–2026)',
        )
        fig_cat.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig_cat.update_layout(height=360, showlegend=False, margin=dict(t=45, b=20, l=20, r=20))
        st.plotly_chart(fig_cat, width='stretch')
        st.dataframe(
            cat_summary.rename(columns={
                'Kategori': 'Kategori',
                'GozlemSayisi': 'Temiz Gözlem Sayısı',
                'SeriSayisi': 'Seri Sayısı',
                'UrunSayisi': 'Tekil Ürün (TabelaGId)',
            }),
            width='stretch',
            hide_index=True,
        )

    with col_right:
        st.markdown('#### Son İşlem Günü (2026-10-26) Fiyat Özeti — Birim Koduna Göre')
        unit_options = sorted(ready_df['BirimTurId'].unique().tolist())
        sel_unit = st.selectbox(
            'Karşılaştırma için birim kodu seçiniz (farklı birim kodları birleştirilmez):',
            options=unit_options,
            format_func=lambda u: f"{unit_label(int(u))} ({int((ready_df['BirimTurId'] == u).sum())} aktif seri)",
        )
        unit_sub = ready_df[ready_df['BirimTurId'] == sel_unit].sort_values('MevcutOrtaFiyat', ascending=False)
        top_exp = unit_sub.head(7)
        top_cheap = unit_sub.tail(7).sort_values('MevcutOrtaFiyat', ascending=True)

        tab_high, tab_low = st.tabs(['En Yüksek Fiyatlı 7 Ürün (Son Gün)', 'En Düşük Fiyatlı 7 Ürün (Son Gün)'])
        with tab_high:
            st.dataframe(
                top_exp[['UrunAd', 'Kategori', 'MevcutMinFiyat', 'MevcutMaxFiyat', 'MevcutOrtaFiyat', 'TahminMinFiyat', 'TahminMaxFiyat']].rename(columns={
                    'UrunAd': 'Ürün Adı',
                    'MevcutMinFiyat': 'Mevcut Min (TL)',
                    'MevcutMaxFiyat': 'Mevcut Max (TL)',
                    'MevcutOrtaFiyat': 'Mevcut Orta (TL)',
                    'TahminMinFiyat': 'Tahmin Min (TL)',
                    'TahminMaxFiyat': 'Tahmin Max (TL)',
                }),
                width='stretch',
                hide_index=True,
            )
        with tab_low:
            st.dataframe(
                top_cheap[['UrunAd', 'Kategori', 'MevcutMinFiyat', 'MevcutMaxFiyat', 'MevcutOrtaFiyat', 'TahminMinFiyat', 'TahminMaxFiyat']].rename(columns={
                    'UrunAd': 'Ürün Adı',
                    'MevcutMinFiyat': 'Mevcut Min (TL)',
                    'MevcutMaxFiyat': 'Mevcut Max (TL)',
                    'MevcutOrtaFiyat': 'Mevcut Orta (TL)',
                    'TahminMinFiyat': 'Tahmin Min (TL)',
                    'TahminMaxFiyat': 'Tahmin Max (TL)',
                }),
                width='stretch',
                hide_index=True,
            )

    st.markdown('---')
    st.markdown('#### Son İşlem Gününde (2026-10-26) Beklenen Değişimi En Belirgin Aktif Ürünler (Ana Model: Random Forest)')
    movers = ready_df.reindex(ready_df['BeklenenDegisimOrtaYuzde'].abs().sort_values(ascending=False).index).head(12)
    st.dataframe(
        movers[[
            'UrunAd', 'Kategori', 'BirimAciklamasi', 'MevcutMinFiyat', 'MevcutMaxFiyat',
            'TahminMinFiyat', 'TahminMaxFiyat', 'BeklenenDegisimMinTL', 'BeklenenDegisimMaxTL', 'BeklenenDegisimOrtaYuzde',
        ]].rename(columns={
            'UrunAd': 'Ürün Adı',
            'BirimAciklamasi': 'Birim Kodu',
            'MevcutMinFiyat': 'Mevcut Min (TL)',
            'MevcutMaxFiyat': 'Mevcut Max (TL)',
            'TahminMinFiyat': 'Tahmin Min (TL)',
            'TahminMaxFiyat': 'Tahmin Max (TL)',
            'BeklenenDegisimMinTL': 'Δ Min (TL)',
            'BeklenenDegisimMaxTL': 'Δ Max (TL)',
            'BeklenenDegisimOrtaYuzde': 'Beklenen Orta Değişim (%)',
        }),
        width='stretch',
        hide_index=True,
    )


def render_page_urun_analizi(bundle: dict[str, Any]) -> None:
    st.subheader('Ürün Bazlı Geçmiş Fiyat ve Mevsimsellik Analizi')
    clean_df: pd.DataFrame = bundle['clean_df']
    series_meta: pd.DataFrame = bundle['series_meta']

    fcol1, fcol2, fcol3 = st.columns([1.0, 1.1, 1.9])
    with fcol1:
        cat_opts = ['Tümü'] + sorted(series_meta['Kategori'].unique().tolist())
        sel_cat = st.selectbox('Kategori Filtresi:', options=cat_opts)
    with fcol2:
        search_q = st.text_input('Ürün Adı Ara:', value='', placeholder='Örn: Domates, Biber, Elma...')

    filtered_meta = series_meta.copy()
    if sel_cat != 'Tümü':
        filtered_meta = filtered_meta[filtered_meta['Kategori'] == sel_cat]
    if search_q.strip():
        filtered_meta = filtered_meta[filtered_meta['UrunAd'].str.contains(search_q.strip(), case=False, na=False)]
    if filtered_meta.empty:
        st.warning('Arama kriterlerine uygun ürün serisi bulunamadı. Filtreleri genişletiniz.')
        return

    with fcol3:
        default_idx = 0
        dom_matches = filtered_meta.index[filtered_meta['UrunAd'] == 'Domates'].tolist()
        if dom_matches:
            default_idx = list(filtered_meta.index).index(dom_matches[0])
        sel_label = st.selectbox(
            'Ürün Serisi Seçiniz:',
            options=filtered_meta['GorunenEtiket'].tolist(),
            index=default_idx,
        )

    sel_row = filtered_meta[filtered_meta['GorunenEtiket'] == sel_label].iloc[0]
    sid = str(sel_row['SeriId'])
    prod_df = clean_df[clean_df['SeriId'] == sid].sort_values('Tarih').reset_index(drop=True)
    prod_df['TarihDT'] = pd.to_datetime(prod_df['Tarih'])

    min_d = prod_df['TarihDT'].min().date()
    max_d = prod_df['TarihDT'].max().date()
    opt_col1, opt_col2, opt_col3 = st.columns([1.4, 0.8, 0.8])
    with opt_col1:
        date_range = st.date_input(
            'Tarih Aralığı Seçiniz:',
            value=(max(min_d, pd.Timestamp('2023-01-01').date()), max_d),
            min_value=min_d,
            max_value=max_d,
        )
    with opt_col2:
        show_ma7 = st.checkbox('7 Gözlemlik Hareketli Ortalama', value=True)
    with opt_col3:
        show_ma30 = st.checkbox('30 Gözlemlik Hareketli Ortalama', value=True)

    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start_d, end_d = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    else:
        start_d, end_d = pd.Timestamp(min_d), pd.Timestamp(max_d)

    prod_df['MinMA7'] = prod_df['EnDusukFiyat'].rolling(7, min_periods=1).mean()
    prod_df['MaxMA7'] = prod_df['EnYuksekFiyat'].rolling(7, min_periods=1).mean()
    prod_df['OrtaMA30'] = prod_df['OrtaFiyat'].rolling(30, min_periods=1).mean()

    view_df = prod_df[(prod_df['TarihDT'] >= start_d) & (prod_df['TarihDT'] <= end_d)].copy()
    if view_df.empty:
        st.warning('Seçilen tarih aralığında gözlem bulunmamaktadır.')
        return

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric('Seçili Dönem Gözlem', f'{len(view_df):,}', f'Toplam: {len(prod_df):,}')
    k2.metric('Son Minimum Fiyat', f"{view_df['EnDusukFiyat'].iloc[-1]:.2f} TL", f"Tarih: {view_df['Tarih'].iloc[-1]}")
    k3.metric('Son Maksimum Fiyat', f"{view_df['EnYuksekFiyat'].iloc[-1]:.2f} TL", f"Aralık: {view_df['FiyatAraligi'].iloc[-1]:.2f} TL")
    k4.metric('Dönem Ort. Min / Max', f"{view_df['EnDusukFiyat'].mean():.2f} / {view_df['EnYuksekFiyat'].mean():.2f} TL")
    k5.metric('Dönem En Düşük / En Yüksek', f"{view_df['EnDusukFiyat'].min():.2f} / {view_df['EnYuksekFiyat'].max():.2f} TL")
    k6.metric('Birim Kodu', f"Kod {int(sel_row['BirimTurId'])}", 'Adı doğrulanmadı')

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=view_df['TarihDT'],
        y=view_df['EnYuksekFiyat'],
        mode='lines',
        name='Maksimum Fiyat (TL)',
        line=dict(color='#dc2626', width=1.8),
    ))
    fig_ts.add_trace(go.Scatter(
        x=view_df['TarihDT'],
        y=view_df['EnDusukFiyat'],
        mode='lines',
        name='Minimum Fiyat (TL)',
        fill='tonexty',
        fillcolor='rgba(220, 38, 38, 0.08)',
        line=dict(color='#059669', width=1.8),
    ))
    if show_ma7:
        fig_ts.add_trace(go.Scatter(
            x=view_df['TarihDT'],
            y=view_df['MinMA7'],
            mode='lines',
            name='Min Fiyat 7-Gözlem Ort.',
            line=dict(color='#047857', width=1.4, dash='dot'),
        ))
    if show_ma30:
        fig_ts.add_trace(go.Scatter(
            x=view_df['TarihDT'],
            y=view_df['OrtaMA30'],
            mode='lines',
            name='Orta Fiyat 30-Gözlem Ort.',
            line=dict(color='#1d4ed8', width=2.0, dash='dash'),
        ))
    fig_ts.update_layout(
        title=f"{sel_row['UrunAd']} ({sel_row['BirimAciklamasi']}) — Minimum ve Maksimum Fiyat Zaman Serisi",
        xaxis_title='Tarih',
        yaxis_title='Fiyat (TL)',
        height=440,
        hovermode='x unified',
        margin=dict(t=50, b=30, l=30, r=20),
    )
    st.plotly_chart(fig_ts, width='stretch')

    c_sp, c_sea = st.columns(2)
    with c_sp:
        fig_spread = px.area(
            view_df,
            x='TarihDT',
            y='FiyatAraligi',
            title='Gün İçi Fiyat Aralığı (En Yüksek − En Düşük Fiyat, TL)',
            labels={'TarihDT': 'Tarih', 'FiyatAraligi': 'Fiyat Aralığı (TL)'},
            color_discrete_sequence=['#d97706'],
        )
        fig_spread.update_layout(height=340, margin=dict(t=45, b=25, l=25, r=20))
        st.plotly_chart(fig_spread, width='stretch')

    with c_sea:
        view_df['Ay'] = view_df['TarihDT'].dt.month
        month_names = {
            1: 'Oca', 2: 'Şub', 3: 'Mar', 4: 'Nis', 5: 'May', 6: 'Haz',
            7: 'Tem', 8: 'Ağu', 9: 'Eyl', 10: 'Eki', 11: 'Kas', 12: 'Ara',
        }
        sea_df = (
            view_df.groupby('Ay', as_index=False)
            .agg(OrtMin=('EnDusukFiyat', 'mean'), OrtMax=('EnYuksekFiyat', 'mean'))
            .sort_values('Ay')
        )
        sea_df['AyAdi'] = sea_df['Ay'].map(month_names)
        fig_sea = go.Figure()
        fig_sea.add_trace(go.Bar(x=sea_df['AyAdi'], y=sea_df['OrtMin'], name='Aylık Ort. Min (TL)', marker_color='#059669'))
        fig_sea.add_trace(go.Bar(x=sea_df['AyAdi'], y=sea_df['OrtMax'], name='Aylık Ort. Max (TL)', marker_color='#2563eb'))
        fig_sea.update_layout(
            barmode='group',
            title='Seçili Dönemde Aylara Göre Ortalama Fiyat Dağılımı',
            xaxis_title='Ay',
            yaxis_title='Ortalama Fiyat (TL)',
            height=340,
            margin=dict(t=45, b=25, l=25, r=20),
        )
        st.plotly_chart(fig_sea, width='stretch')


def render_page_fiyat_tahmini(bundle: dict[str, Any]) -> None:
    st.subheader('Gelecek Hal İşlem Günü Fiyat Tahmini')
    service = get_prediction_service()
    pds = get_decision_support()
    series_meta: pd.DataFrame = bundle['series_meta']
    clean_df: pd.DataFrame = bundle['clean_df']
    latest_features_df: pd.DataFrame = bundle['latest_features_df']
    latest_status_df: pd.DataFrame = bundle['latest_status_df']

    col_filter, col_prod, col_model = st.columns([0.9, 1.7, 1.0])
    with col_filter:
        only_active = st.checkbox('Yalnızca son günde (2026-10-26) aktif 91 seriyi listele', value=True)
    sub_meta = series_meta[series_meta['OzellikHazir']] if only_active else series_meta

    with col_prod:
        default_idx = 0
        dom_idx = sub_meta.index[sub_meta['UrunAd'] == 'Domates'].tolist()
        if dom_idx:
            default_idx = list(sub_meta.index).index(dom_idx[0])
        sel_label = st.selectbox('Tahmin Edilecek Ürün Serisi:', options=sub_meta['GorunenEtiket'].tolist(), index=default_idx)

    with col_model:
        model_opts = ['random_forest', 'xgboost', 'linear_regression', 'naive_baseline']
        sel_model = st.selectbox(
            'Tahmin Modeli Seçiniz:',
            options=model_opts,
            index=0,
            format_func=lambda m: f"{MODEL_NAMES[m]}" + (' (Seçilen Ana Model)' if m == service.selected_model_code else ''),
        )

    sel_row = sub_meta[sub_meta['GorunenEtiket'] == sel_label].iloc[0]
    sid = str(sel_row['SeriId'])

    pred_res = service.predict_for_series_row(
        series_id=sid,
        latest_features_df=latest_features_df,
        status_df=latest_status_df,
        clean_df=clean_df,
        model_code=sel_model,
        allow_naive_fallback=True,
    )
    err_min = pds.get_historical_error_context(sid, pred_res['kullanilan_model_kodu'], 'minimum')
    err_max = pds.get_historical_error_context(sid, pred_res['kullanilan_model_kodu'], 'maksimum')

    if pred_res['geri_donus_kullanildi']:
        st.markdown(
            f"""
            <div class="warn-card">
                <strong>Kesintili Seri Bildirimi:</strong> Seçilen ürün serisinin son gözlem tarihi
                <strong>{pred_res['son_gozlem_tarihi']}</strong> olup son veri gününde (2026-10-26) 20 gözlemlik kesintisiz
                geçmiş penceresi bulunmamaktadır. Sahte özellik üretmemek adına açıkça
                <strong>Naive Baseline (son gözlenen fiyatın korunması)</strong> geri dönüşü gösterilmektedir.
            </div>
            """,
            unsafe_allow_html=True,
        )
    for w in pred_res['uyarilar']:
        if not pred_res['geri_donus_kullanildi']:
            st.info(w)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric('Mevcut Minimum Fiyat', f"{pred_res['fiyat_min']:.2f} TL", f"Son Gözlem: {pred_res['son_gozlem_tarihi']}")
    m2.metric(
        'Tahmini Minimum Fiyat',
        f"{pred_res['tahmin_min']:.2f} TL",
        f"{pred_res['beklenen_degisim_min_tl']:+.2f} TL (%{pred_res['beklenen_degisim_min_yuzde']:+.2f})",
    )
    m3.metric('Mevcut Maksimum Fiyat', f"{pred_res['fiyat_max']:.2f} TL", pred_res['birim_aciklamasi'])
    m4.metric(
        'Tahmini Maksimum Fiyat',
        f"{pred_res['tahmin_max']:.2f} TL",
        f"{pred_res['beklenen_degisim_max_tl']:+.2f} TL (%{pred_res['beklenen_degisim_max_yuzde']:+.2f})",
    )
    m5.metric('Geçmiş Test MAE (Min / Max)', f"{err_min['gecmis_mae_tl']:.2f} / {err_max['gecmis_mae_tl']:.2f} TL", 'Belirsizlik Bandı')
    m6.metric('Geçmiş Test RMSE (Min / Max)', f"{err_min['gecmis_rmse_tl']:.2f} / {err_max['gecmis_rmse_tl']:.2f} TL", f"Eğitim Örneği: {pred_res['egitim_ornek_sayisi']:,}")

    # Geçmiş son 60 gözlem + bir sonraki işlem günü tahmini grafiği
    hist_sub = clean_df[clean_df['SeriId'] == sid].sort_values('Tarih').tail(60).copy()
    hist_sub['TarihDT'] = pd.to_datetime(hist_sub['Tarih'])
    next_dt = hist_sub['TarihDT'].iloc[-1] + pd.Timedelta(days=1)

    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=hist_sub['TarihDT'],
        y=hist_sub['EnYuksekFiyat'],
        mode='lines+markers',
        name='Geçmiş Maksimum Fiyat',
        line=dict(color='#dc2626', width=2),
        marker=dict(size=5),
    ))
    fig_pred.add_trace(go.Scatter(
        x=hist_sub['TarihDT'],
        y=hist_sub['EnDusukFiyat'],
        mode='lines+markers',
        name='Geçmiş Minimum Fiyat',
        line=dict(color='#059669', width=2),
        marker=dict(size=5),
    ))
    fig_pred.add_trace(go.Scatter(
        x=[hist_sub['TarihDT'].iloc[-1], next_dt],
        y=[pred_res['fiyat_max'], pred_res['tahmin_max']],
        mode='lines+markers',
        name=f"Tahmini Maksimum ({pred_res['kullanilan_model_adi']})",
        line=dict(color='#f97316', width=2.5, dash='dash'),
        marker=dict(size=10, symbol='diamond'),
    ))
    fig_pred.add_trace(go.Scatter(
        x=[hist_sub['TarihDT'].iloc[-1], next_dt],
        y=[pred_res['fiyat_min'], pred_res['tahmin_min']],
        mode='lines+markers',
        name=f"Tahmini Minimum ({pred_res['kullanilan_model_adi']})",
        line=dict(color='#10b981', width=2.5, dash='dash'),
        marker=dict(size=10, symbol='diamond'),
    ))
    fig_pred.update_layout(
        title=f"{pred_res['urun_ad']} — Son 60 Gözlem ve Bir Sonraki İşlem Günü Fiyat Tahmini ({pred_res['kullanilan_model_adi']})",
        xaxis_title='Tarih',
        yaxis_title='Fiyat (TL)',
        height=430,
        hovermode='x unified',
        margin=dict(t=50, b=30, l=30, r=20),
    )
    st.plotly_chart(fig_pred, width='stretch')

    # Tüm modellerin bu ürün serisi için yan yana tahmin tablosu
    st.markdown('#### Bu Ürün Serisi İçin Tüm Modellerin Tahmin Karşılaştırması')
    all_models_sub = bundle['latest_preds_df'][bundle['latest_preds_df']['SeriId'] == sid].copy()
    st.dataframe(
        all_models_sub[[
            'IstenenModelKodu', 'KullanilanModelAdi', 'MevcutMinFiyat', 'TahminMinFiyat',
            'BeklenenDegisimMinTL', 'BeklenenDegisimMinYuzde',
            'MevcutMaxFiyat', 'TahminMaxFiyat', 'BeklenenDegisimMaxTL', 'BeklenenDegisimMaxYuzde',
        ]].rename(columns={
            'IstenenModelKodu': 'İstenen Model Kodu',
            'KullanilanModelAdi': 'Uygulanan Model',
            'MevcutMinFiyat': 'Mevcut Min (TL)',
            'TahminMinFiyat': 'Tahmin Min (TL)',
            'BeklenenDegisimMinTL': 'Δ Min (TL)',
            'BeklenenDegisimMinYuzde': 'Δ Min (%)',
            'MevcutMaxFiyat': 'Mevcut Max (TL)',
            'TahminMaxFiyat': 'Tahmin Max (TL)',
            'BeklenenDegisimMaxTL': 'Δ Max (TL)',
            'BeklenenDegisimMaxYuzde': 'Δ Max (%)',
        }),
        width='stretch',
        hide_index=True,
    )


def _render_expectation_badge(label: str) -> str:
    if label == 'Fiyat Artışı Beklentisi':
        return f'<span class="badge-up">▲ {label}</span>'
    if label == 'Fiyat Düşüşü Beklentisi':
        return f'<span class="badge-down">▼ {label}</span>'
    return f'<span class="badge-neutral">● {label}</span>'


def render_page_satin_alma(bundle: dict[str, Any]) -> None:
    st.subheader('Satın Alma Planlama ve Maliyet Simülasyonu')
    pds = get_decision_support()
    series_meta: pd.DataFrame = bundle['series_meta']
    active_meta = series_meta[series_meta['OzellikHazir']].copy()

    tab_single, tab_basket = st.tabs(['1. Tekil Ürün Maliyet Simülasyonu', '2. Çoklu Ürün Sepeti Simülasyonu'])

    with tab_single:
        col_a, col_b, col_c, col_d = st.columns([1.8, 0.9, 1.0, 1.0])
        with col_a:
            sel_label = st.selectbox('Satın Alınacak Ürün Serisi:', options=active_meta['GorunenEtiket'].tolist(), key='single_prod')
        with col_b:
            qty = st.number_input('Satın Alınacak Miktar:', min_value=0.5, max_value=100000.0, value=100.0, step=10.0, key='single_qty')
        with col_c:
            ptype = st.selectbox(
                'Fiyat Türü:',
                options=['minimum', 'maksimum', 'aritmetik_orta'],
                index=2,
                format_func=lambda k: VALID_PRICE_TYPES[k][3],
                key='single_ptype',
            )
        with col_d:
            use_budget = st.checkbox('Bütçe Sınırı Belirt', value=True, key='single_use_budget')
            budget_val = st.number_input('Bütçe (TL):', min_value=1.0, max_value=10000000.0, value=10000.0, step=500.0, disabled=not use_budget, key='single_budget')

        sid = str(active_meta[active_meta['GorunenEtiket'] == sel_label].iloc[0]['SeriId'])
        sim = pds.simulate_single_purchase(
            series_id=sid,
            quantity=qty,
            price_type=ptype,
            model_code=pds.selected_model_code,
            budget=budget_val if use_budget else None,
        )

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric('Mevcut Toplam Maliyet', f"{sim['mevcut_toplam_maliyet_tl']:,.2f} TL", f"Birim: {sim['mevcut_birim_fiyat_tl']:.2f} TL")
        k2.metric(
            'Tahmini Toplam Maliyet',
            f"{sim['tahmini_toplam_maliyet_tl']:,.2f} TL",
            f"Birim: {sim['tahmini_birim_fiyat_tl']:.2f} TL",
        )
        k3.metric(
            'Beklenen Maliyet Farkı',
            f"{sim['toplam_maliyet_farki_tl']:+,.2f} TL",
            f"%{sim['degisim_orani_yuzde']:+.2f}",
        )
        k4.metric(
            'Miktar Bazlı MAE Belirsizliği',
            f"±{sim['belirsizlik_baglami']['miktar_bazli_mae_bandi_tl']:,.2f} TL",
            f"Birim MAE: {sim['belirsizlik_baglami']['gecmis_mae_tl']:.2f} TL",
        )
        k5.metric('Birim Kodu', f"Kod {sim['birim_kodu']}", 'Miktar bu birimde çarpılır')

        st.markdown(
            f"""
            <div class="info-card">
                <div style="margin-bottom:0.4rem;"><strong>Yön Beklentisi:</strong> {_render_expectation_badge(sim['beklenti_sinifi'])}</div>
                <div><strong>Karar Destek Yorumu:</strong> {sim['karar_yorumu']}</div>
                <div style="margin-top:0.4rem;"><strong>Bütçe Değerlendirmesi:</strong> {sim['butce_analizi']['butce_durumu_aciklamasi']}</div>
                <div style="margin-top:0.35rem; font-size:0.86rem; color:#475569;"><em>{sim['sorumluluk_notu']}</em></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Üç fiyat türünde (min, orta, max) maliyet karşılaştırma grafiği
        multi_p = [
            pds.simulate_single_purchase(sid, qty, pt, pds.selected_model_code, None)
            for pt in ('minimum', 'aritmetik_orta', 'maksimum')
        ]
        fig_cost = go.Figure()
        fig_cost.add_trace(go.Bar(
            x=[r['fiyat_turu_etiketi'] for r in multi_p],
            y=[r['mevcut_toplam_maliyet_tl'] for r in multi_p],
            name='Mevcut Toplam Maliyet (TL)',
            marker_color='#64748b',
        ))
        fig_cost.add_trace(go.Bar(
            x=[r['fiyat_turu_etiketi'] for r in multi_p],
            y=[r['tahmini_toplam_maliyet_tl'] for r in multi_p],
            name='Tahmini Toplam Maliyet (TL)',
            marker_color='#059669',
            error_y=dict(type='data', array=[r['belirsizlik_baglami']['miktar_bazli_mae_bandi_tl'] for r in multi_p], visible=True),
        ))
        fig_cost.update_layout(
            barmode='group',
            title=f"{sim['urun_ad']} ({qty:,.0f} Birim) — Fiyat Türlerine Göre Mevcut ve Tahmini Maliyet (±MAE Bandı)",
            yaxis_title='Toplam Maliyet (TL)',
            height=360,
            margin=dict(t=45, b=25, l=25, r=20),
        )
        st.plotly_chart(fig_cost, width='stretch')

    with tab_basket:
        st.markdown('#### Çoklu Ürün Sepeti Maliyet Simülasyonu')
        default_names = ['Domates', 'Salatalık', 'Patates', 'Soğan (Kuru)']
        default_labels = []
        for dn in default_names:
            m = active_meta[active_meta['UrunAd'] == dn]
            if not m.empty:
                default_labels.append(str(m.iloc[0]['GorunenEtiket']))

        sel_basket_labels = st.multiselect(
            'Sepete Eklenecek Ürünleri Seçiniz:',
            options=active_meta['GorunenEtiket'].tolist(),
            default=default_labels,
        )
        bcol1, bcol2, bcol3 = st.columns(3)
        with bcol1:
            b_ptype = st.selectbox(
                'Sepet Fiyat Türü:',
                options=['minimum', 'maksimum', 'aritmetik_orta'],
                index=2,
                format_func=lambda k: VALID_PRICE_TYPES[k][3],
                key='basket_ptype',
            )
        with bcol2:
            b_use_budget = st.checkbox('Toplam Sepet Bütçesi Belirt', value=True, key='basket_use_budget')
        with bcol3:
            b_budget = st.number_input('Toplam Sepet Bütçesi (TL):', min_value=10.0, max_value=50000000.0, value=25000.0, step=1000.0, disabled=not b_use_budget, key='basket_budget')

        if not sel_basket_labels:
            st.warning('Sepet simülasyonu için en az bir ürün seçiniz.')
        else:
            qty_cols = st.columns(min(4, len(sel_basket_labels)))
            default_qtys = {'Domates': 100.0, 'Salatalık': 80.0, 'Patates': 150.0, 'Soğan (Kuru)': 120.0}
            basket_items_input: list[dict[str, Any]] = []
            for idx, lbl in enumerate(sel_basket_labels):
                r_meta = active_meta[active_meta['GorunenEtiket'] == lbl].iloc[0]
                uname = str(r_meta['UrunAd'])
                sid_item = str(r_meta['SeriId'])
                with qty_cols[idx % len(qty_cols)]:
                    q_val = st.number_input(
                        f"{uname} Miktarı (Birim {int(r_meta['BirimTurId'])}):",
                        min_value=1.0,
                        max_value=100000.0,
                        value=float(default_qtys.get(uname, 50.0)),
                        step=10.0,
                        key=f'bqty_{sid_item}',
                    )
                basket_items_input.append({'seri_id': sid_item, 'miktar': q_val})

            basket_res = pds.simulate_basket_purchase(
                basket_items=basket_items_input,
                price_type=b_ptype,
                model_code=pds.selected_model_code,
                total_budget=b_budget if b_use_budget else None,
            )

            bk1, bk2, bk3, bk4, bk5 = st.columns(5)
            bk1.metric('Sepet Kalem Sayısı', f"{basket_res['kalem_sayisi']} Ürün")
            bk2.metric('Toplam Mevcut Maliyet', f"{basket_res['toplam_mevcut_maliyet_tl']:,.2f} TL")
            bk3.metric('Toplam Tahmini Maliyet', f"{basket_res['toplam_tahmini_maliyet_tl']:,.2f} TL")
            bk4.metric(
                'Toplam Beklenen Fark',
                f"{basket_res['toplam_maliyet_farki_tl']:+,.2f} TL",
                f"%{basket_res['toplam_degisim_orani_yuzde']:+.2f}",
            )
            bk5.metric('Toplam MAE Belirsizliği', f"±{basket_res['toplam_mae_belirsizlik_bandi_tl']:,.2f} TL")

            for bw in basket_res['sepet_uyarilari']:
                st.info(bw)
            st.markdown(
                f"""
                <div class="info-card">
                    <strong>Sepet Bütçe Durumu:</strong> {basket_res['butce_analizi']['butce_durumu_aciklamasi']}<br/>
                    <strong>Fiyat Artışı Beklenenler:</strong> {', '.join(basket_res['fiyat_artisi_beklenen_urunler']) or 'Yok'} |
                    <strong>Fiyat Düşüşü Beklenenler:</strong> {', '.join(basket_res['fiyat_dususu_beklenen_urunler']) or 'Yok'} |
                    <strong>Düşük Değişim Beklenenler:</strong> {', '.join(basket_res['dusuk_degisim_beklenen_urunler']) or 'Yok'}
                </div>
                """,
                unsafe_allow_html=True,
            )

            items_table = pd.DataFrame([
                {
                    'Ürün Adı': it['urun_ad'],
                    'Birim Kodu': it['birim_aciklamasi'],
                    'Miktar': it['miktar'],
                    'Mevcut Birim (TL)': it['mevcut_birim_fiyat_tl'],
                    'Tahmini Birim (TL)': it['tahmini_birim_fiyat_tl'],
                    'Mevcut Maliyet (TL)': it['mevcut_toplam_maliyet_tl'],
                    'Tahmini Maliyet (TL)': it['tahmini_toplam_maliyet_tl'],
                    'Maliyet Farkı (TL)': it['toplam_maliyet_farki_tl'],
                    'Değişim (%)': it['degisim_orani_yuzde'],
                    'Geçmiş MAE (TL)': it['belirsizlik_baglami']['gecmis_mae_tl'],
                    'Beklenti Sınıfı': it['beklenti_sinifi'],
                }
                for it in basket_res['kalemler']
            ])
            st.dataframe(items_table, width='stretch', hide_index=True)

            fig_b = go.Figure()
            fig_b.add_trace(go.Bar(
                x=items_table['Ürün Adı'],
                y=items_table['Mevcut Maliyet (TL)'],
                name='Mevcut Maliyet (TL)',
                marker_color='#64748b',
            ))
            fig_b.add_trace(go.Bar(
                x=items_table['Ürün Adı'],
                y=items_table['Tahmini Maliyet (TL)'],
                name='Tahmini Maliyet (TL)',
                marker_color='#059669',
            ))
            fig_b.update_layout(
                barmode='group',
                title='Sepet Kalemleri Bazında Mevcut ve Tahmini Maliyet Karşılaştırması',
                yaxis_title='Maliyet (TL)',
                height=360,
                margin=dict(t=45, b=25, l=25, r=20),
            )
            st.plotly_chart(fig_b, width='stretch')


def render_page_model_karsilastirma(bundle: dict[str, Any]) -> None:
    st.subheader('Makine Öğrenmesi Modellerinin Akademik Karşılaştırması')
    main_comp_df: pd.DataFrame = bundle['main_comp_df']
    subgroup_comp_df: pd.DataFrame = bundle['subgroup_comp_df']
    monthly_err_df: pd.DataFrame = bundle['monthly_err_df']
    feature_imp_df: pd.DataFrame = bundle['feature_imp_df']
    s4_reg = bundle['s4_registry']

    st.markdown(
        f"""
        <div class="info-card">
            <strong>Model Seçim Kararı:</strong> {s4_reg['secim_kriteri_aciklamasi']}
        </div>
        """,
        unsafe_allow_html=True,
    )

    split_sel = st.radio(
        'Değerlendirme Bölümü Seçiniz:',
        options=['dogrulama', 'test'],
        horizontal=True,
        format_func=lambda s: 'Doğrulama Dönemi (2024-10-27 – 2025-10-26 | N=30.361)' if s == 'dogrulama' else 'Test Dönemi (2025-10-27 – 2026-10-26 | N=21.141)',
    )
    sub_main = main_comp_df[main_comp_df['Bolum'] == split_sel].copy()

    st.markdown('#### 1. Ana Performans Tablosu (`hedef_min` ve `hedef_max`)')
    st.dataframe(
        sub_main[[
            'Hedef', 'ModelAdi', 'OrnekSayisi', 'SeriSayisi', 'MAE', 'RMSE', 'MSE', 'R2',
            'BaselineMAEIyilesmeYuzde', 'BaselineRMSEIyilesmeYuzde',
            'SeriEsitAgirlikliMAEOrtalama', 'SeriEsitAgirlikliRMSEOrtalama', 'BaselineRMSEDahaIyiSeriSayisi',
        ]].rename(columns={
            'Hedef': 'Hedef Değişken',
            'ModelAdi': 'Model',
            'OrnekSayisi': 'Örnek Sayısı',
            'SeriSayisi': 'Seri Sayısı',
            'BaselineMAEIyilesmeYuzde': 'Baseline MAE İyileşme (%)',
            'BaselineRMSEIyilesmeYuzde': 'Baseline RMSE İyileşme (%)',
            'SeriEsitAgirlikliMAEOrtalama': 'Seri Eşit Ort. MAE',
            'SeriEsitAgirlikliRMSEOrtalama': 'Seri Eşit Ort. RMSE',
            'BaselineRMSEDahaIyiSeriSayisi': "Baseline'dan Düşük RMSE Seri Sayısı",
        }),
        width='stretch',
        hide_index=True,
    )

    c_mae, c_rmse = st.columns(2)
    with c_mae:
        fig_mae = px.bar(
            sub_main,
            x='Hedef',
            y='MAE',
            color='ModelAdi',
            barmode='group',
            text='MAE',
            color_discrete_map=MODEL_COLORS,
            title='Modele Göre Ortalama Mutlak Hata (MAE — TL)',
        )
        fig_mae.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig_mae.update_layout(height=370, margin=dict(t=45, b=25, l=25, r=20))
        st.plotly_chart(fig_mae, width='stretch')

    with c_rmse:
        fig_rmse = px.bar(
            sub_main,
            x='Hedef',
            y='RMSE',
            color='ModelAdi',
            barmode='group',
            text='RMSE',
            color_discrete_map=MODEL_COLORS,
            title='Modele Göre Kök Ortalama Karesel Hata (RMSE — TL)',
        )
        fig_rmse.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig_rmse.update_layout(height=370, margin=dict(t=45, b=25, l=25, r=20))
        st.plotly_chart(fig_rmse, width='stretch')

    st.markdown('#### 2. Zamana Göre Aylık Hata Değişimi ve 2025-10-03 Rejim Kırılması')
    t_sel = st.selectbox('Zaman Serisi Hata Grafiği İçin Hedef:', options=['hedef_min', 'hedef_max'], format_func=lambda x: 'Minimum Fiyat (hedef_min)' if x == 'hedef_min' else 'Maksimum Fiyat (hedef_max)')
    m_sub = monthly_err_df[monthly_err_df['Hedef'] == t_sel].copy()
    m_sub['DonemAy'] = m_sub['YilAy'] + ' (' + m_sub['Bolum'].map({'dogrulama': 'Doğr.', 'test': 'Test'}) + ')'
    fig_time = px.line(
        m_sub,
        x='DonemAy',
        y='RMSE',
        color='ModelAdi',
        markers=True,
        color_discrete_map=MODEL_COLORS,
        title=f'Aylık RMSE Değişimi ({t_sel}) — Doğrulama ve Test Dönemleri',
        labels={'DonemAy': 'Yıl-Ay (Bölüm)', 'RMSE': 'Aylık RMSE (TL)', 'ModelAdi': 'Model'},
    )
    fig_time.update_layout(height=400, margin=dict(t=45, b=45, l=25, r=20))
    st.plotly_chart(fig_time, width='stretch')

    st.markdown('#### 3. Özellik Önem Düzeyleri (İlk 15 Özellik)')
    imp_col1, imp_col2 = st.columns(2)
    for col_ui, mcode in zip([imp_col1, imp_col2], ['random_forest', 'xgboost']):
        with col_ui:
            imp_sub = (
                feature_imp_df[
                    (feature_imp_df['ModelKodu'] == mcode)
                    & (feature_imp_df['Hedef'] == t_sel)
                ]
                .sort_values('Sira')
                .head(15)
                .iloc[::-1]
            )
            fig_imp = px.bar(
                imp_sub,
                x='NormalizeOnem',
                y='OzellikAdi',
                orientation='h',
                title=f"{MODEL_NAMES[mcode]} — En Önemli 15 Özellik ({t_sel})",
                labels={'NormalizeOnem': 'Göreli Önem Oranı', 'OzellikAdi': 'Özellik'},
                color_discrete_sequence=[MODEL_COLORS[MODEL_NAMES[mcode]]],
            )
            fig_imp.update_layout(height=420, margin=dict(t=45, b=25, l=25, r=20))
            st.plotly_chart(fig_imp, width='stretch')

    st.markdown('#### 4. Alt Gruplara Göre Performans (Eğitimde Görülme Durumu, Kategori ve Birim Kodu)')
    sub_view = subgroup_comp_df[(subgroup_comp_df['Bolum'] == split_sel) & (subgroup_comp_df['Hedef'] == t_sel)]
    st.dataframe(
        sub_view[[
            'AltGrupTuru', 'AltGrupAciklamasi', 'ModelAdi', 'OrnekSayisi', 'SeriSayisi',
            'MAE', 'RMSE', 'R2', 'BaselineMAEIyilesmeYuzde', 'BaselineRMSEIyilesmeYuzde',
        ]].rename(columns={
            'AltGrupTuru': 'Alt Grup Türü',
            'AltGrupAciklamasi': 'Alt Grup',
            'ModelAdi': 'Model',
            'OrnekSayisi': 'Örnek Sayısı',
            'SeriSayisi': 'Seri Sayısı',
            'BaselineMAEIyilesmeYuzde': 'Baseline MAE İyileşme (%)',
            'BaselineRMSEIyilesmeYuzde': 'Baseline RMSE İyileşme (%)',
        }),
        width='stretch',
        hide_index=True,
    )


def render_page_proje_hakkinda(bundle: dict[str, Any]) -> None:
    st.subheader('Proje Hakkında, Metodoloji ve Akademik Künye')
    p_info = bundle['persistence_json']['yapisal_kirilma_2025_10_03']

    col_team, col_uni = st.columns([1.1, 1.3])
    with col_team:
        st.markdown(
            """
            <div class="info-card">
                <h4 style="margin-top:0;">Proje Ekibi</h4>
                <ul>
                    <li><strong>Burak Demirel</strong></li>
                    <li><strong>Ebubekir Samet Gözübenli</strong></li>
                    <li><strong>Resul Kılınç</strong></li>
                </ul>
                <p style="margin-bottom:0;">
                    <strong>Üniversite:</strong> Kahramanmaraş Sütçü İmam Üniversitesi<br/>
                    <strong>Bölüm:</strong> Bilgisayar Mühendisliği Bölümü<br/>
                    <strong>Ders:</strong> Makine Öğrenmesi
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_uni:
        st.markdown(
            """
            <div class="info-card">
                <h4 style="margin-top:0;">Projenin Amacı ve Kapsamı</h4>
                <p>
                    İstanbul Büyükşehir Belediyesi (İBB) Hal Müdürlüğü tarafından yayımlanan 2016–2026 dönemi
                    sebze, meyve ve ithal ürün fiyat bültenlerini kullanarak bir sonraki hal işlem gününün
                    <strong>minimum</strong> ve <strong>maksimum</strong> fiyatlarını veri sızıntısı olmadan tahmin etmek
                    ve manav, restoran ve gıda işletmeleri için belirsizlik duyarlı bir <strong>satın alma karar destek sistemi</strong> sunmaktır.
                </p>
                <p style="margin-bottom:0;">
                    <strong>Veri Kapsamı Kısıtı:</strong> Bu sürümde yalnızca İBB hal fiyat verileri kullanılmış;
                    döviz kuru, motorin fiyatı, meteoroloji, TÜİK endeksleri veya HKS hacim verileri modele dahil edilmemiştir.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('#### Metodolojik Mimari ve Veri Sızıntısı Önlemleri')
    st.markdown(
        f"""
        1. **Aşama 1 — Denetlenebilir Veri Temizliği:** `382.897` ham kayıttan `1.761` tam tekrar, `60` sıfır fiyatlı kayıt ve `91` `EnDusukFiyat > EnYuksekFiyat` kaydı ayrıştırılarak `380.985` yapısal geçerli gözlem ve `299` homojen ürün/birim/ad serisi (`SeriId`) elde edilmiştir.
        2. **Aşama 2 — Kesinti Duyarlı Özellik Mühendisliği:** Yalnızca $t$ ve öncesindeki 20 kesintisiz ortak işlem gününden `54` sayısal özellik üretilmiş (`354.896` etiketli örnek) ve hedef tarih (`HedefTarih`) üzerinden kronolojik olarak **Eğitim (`303.394`)**, **Doğrulama (`30.361`)** ve **Test (`21.141`)** kümelerine ayrılmıştır.
        3. **Aşama 3 — Naive Baseline:** $\\hat{{y}}(t+1) = y(t)$ kuralıyla çok düzeyli referans metrikleri oluşturulmuştur.
        4. **Bölüm 6 — 3 Ekim 2025 Fiyat Sürekliliği Kırılması Bulgusu:** `2025-10-02` tarihinde `{p_info['2025_10_02_seri_sayisi']}` olan günlük seri sayısı `2025-10-03` tarihinde `{p_info['2025_10_03_seri_sayisi']}` seriye düşmüş; yayında kalan `{p_info['2025_10_03_korunan_seri_sayisi']}` ana seride fiyatların aynı kalma oranı **%{p_info['korunan_serilerin_kirilma_oncesi_sabit_kalma_orani']*100:.2f}** seviyesinden doğrulama son 3 haftasında **%{p_info['korunan_serilerin_kirilma_sonrasi_dogrulama_sabit_kalma_orani']*100:.2f}** ve test döneminde **%{p_info['korunan_serilerin_test_sabit_kalma_orani']*100:.2f}** seviyesine inmiştir.
        5. **Aşama 4–5 — Fiyat Farkı ($\\Delta y$) Modellemesi:** 2016–2026 arasındaki 25–30 katlık nominal fiyat artışı nedeniyle modeller $\\Delta y(t+1) = y(t+1) - y(t)$ farkını tahmin edecek şekilde eğitilmiş; **Random Forest** ve **XGBoost** doğrulama kümesinde Naive Baseline RMSE değerini her iki hedefte geçmiştir.
        6. **Kısıtlar ve Gelecek Çalışmalar:** Kaynak veride işlem hacmi (kg/ton), ürün kalitesi/boyutu alt kırılımı, resmi birim adı sözlüğü ve harici makroekonomik/meteorolojik göstergeler yer almamaktadır. İkinci aşamada HKS işlem miktarları, meteorolojik don/yağış verileri ve nakliye maliyet endekslerinin gecikmeli olarak eklenmesi planlanmaktadır.
        """
    )


def main() -> None:
    st.set_page_config(
        page_title='İBB Hal Fiyat Tahmini ve Karar Destek Sistemi',
        page_icon='🥬',
        layout='wide',
        initial_sidebar_state='expanded',
    )
    inject_custom_css()
    render_header()

    try:
        bundle = load_dashboard_bundle()
    except Exception as exc:
        st.error(f'Veri veya model dosyaları yüklenirken hata oluştu: {exc}')
        st.stop()

    with st.sidebar:
        st.markdown('### Gezinme Menüsü')
        selected_page = st.radio('Sayfa Seçiniz:', options=PAGES, index=0)
        st.markdown('---')
        st.markdown(
            f"""
            **Sistem Durumu**
            - **Ana Model:** `{bundle['s4_registry']['secilen_ana_model_adi']}`
            - **Temiz Gözlem:** `380.985`
            - **Etiketli Örnek:** `354.896`
            - **Son İşlem Günü:** `2026-10-26`
            - **Aktif Seri:** `91 / 299`
            """
        )
        st.caption('KSÜ Bilgisayar Mühendisliği • Makine Öğrenmesi Projesi')

    if selected_page == '1. Ana Panel':
        render_page_ana_panel(bundle)
    elif selected_page == '2. Ürün Analizi':
        render_page_urun_analizi(bundle)
    elif selected_page == '3. Fiyat Tahmini':
        render_page_fiyat_tahmini(bundle)
    elif selected_page == '4. Satın Alma Planlama':
        render_page_satin_alma(bundle)
    elif selected_page == '5. Model Karşılaştırma':
        render_page_model_karsilastirma(bundle)
    elif selected_page == '6. Proje Hakkında':
        render_page_proje_hakkinda(bundle)


if __name__ == '__main__':
    main()
