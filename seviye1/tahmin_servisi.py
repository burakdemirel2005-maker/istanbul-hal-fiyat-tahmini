"""Aşama 4–7 ortak ön işleme, model paketleme ve tahmin servisi modülü.

Eğitim sırasında yalnızca eğitim verisine (egitim.csv.gz) fit edilen ön işleme
adımlarını ve modelleri şema/sürüm/performans metadatasıyla birlikte kaydeder,
yükler ve web/karar destek katmanlarına güvenli tahmin üretir.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from asama1_veri import CATEGORY, sha256
from asama2_ozellikler import FEATURES, TARGETS
from asama3_baseline import unit_label

ROOT = Path(__file__).resolve().parent
SERVICE_VERSION = '4.0.0'

# Doğrulama analizinde belirlenen, ölçek patlaması veya durağan dışı yıl trendi yaratan 6 özellik
EXCLUDED_STAGE2_FEATURES = (
    'tarih_yil',
    'min_degisim_orani_1',
    'min_degisim_orani_5',
    'max_degisim_orani_1',
    'max_degisim_orani_5',
    'goreli_fiyat_araligi',
)
SELECTED_NUMERIC_FEATURES = [f for f in FEATURES if f not in EXCLUDED_STAGE2_FEATURES]
CATEGORICAL_FEATURES = ['BirimTurId', 'KategoriId']
SERIES_STAT_FEATURES = [
    'seri_egitim_degismedi_min',
    'seri_egitim_degismedi_max',
    'seri_egitim_mutlak_fark_min',
    'seri_egitim_mutlak_fark_max',
]
MODEL_NAMES = {
    'naive_baseline': 'Naive Baseline',
    'linear_regression': 'Linear Regression',
    'random_forest': 'Random Forest',
    'xgboost': 'XGBoost',
}
MIN_PRICE_FLOOR = 0.01
LOW_TRAINING_SAMPLE_THRESHOLD = 30


class SeriesHistoryStatsEncoder(BaseEstimator, TransformerMixin):
    """SeriId bazında yalnızca eğitim verisinden geçmiş fiyat değişim özetlerini öğrenir.

    Eğitimde görülmeyen yeni bir SeriId geldiğinde sızıntı yapmadan eğitim kümesinin
    genel önsel (prior) ortalamasını atar.
    """

    def __init__(self) -> None:
        self.series_table_: dict[str, dict[str, float]] = {}
        self.global_defaults_: dict[str, float] = {}
        self.seen_series_: set[str] = set()

    def fit(self, X: pd.DataFrame, y: Any = None) -> 'SeriesHistoryStatsEncoder':
        required = {'SeriId', 'min_fark_1', 'max_fark_1'}
        if not required.issubset(X.columns):
            raise ValueError(f'SeriesHistoryStatsEncoder için eksik sütunlar: {sorted(required - set(X.columns))}')
        grouped = X.groupby('SeriId', sort=False)
        stats_df = pd.DataFrame({
            'seri_egitim_degismedi_min': grouped['min_fark_1'].apply(lambda s: float(np.mean(s == 0.0))),
            'seri_egitim_degismedi_max': grouped['max_fark_1'].apply(lambda s: float(np.mean(s == 0.0))),
            'seri_egitim_mutlak_fark_min': grouped['min_fark_1'].apply(lambda s: float(np.mean(np.abs(s)))),
            'seri_egitim_mutlak_fark_max': grouped['max_fark_1'].apply(lambda s: float(np.mean(np.abs(s)))),
        })
        self.series_table_ = stats_df.to_dict(orient='index')
        self.global_defaults_ = {
            col: float(stats_df[col].mean()) if len(stats_df) else 0.0
            for col in SERIES_STAT_FEATURES
        }
        self.seen_series_ = set(stats_df.index)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.global_defaults_:
            raise RuntimeError('SeriesHistoryStatsEncoder henüz fit edilmedi.')
        required = set(['SeriId'] + SELECTED_NUMERIC_FEATURES + CATEGORICAL_FEATURES)
        missing = sorted(required - set(X.columns))
        if missing:
            raise ValueError(f'Ön işleme girdisinde zorunlu sütunlar eksik: {missing}')
        out = X[SELECTED_NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
        for col in SERIES_STAT_FEATURES:
            mapping = {sid: vals[col] for sid, vals in self.series_table_.items()}
            out[col] = X['SeriId'].map(mapping).fillna(self.global_defaults_[col]).astype(float)
        return out


@dataclass
class ResidualTargetModelArtifact:
    """Fiyat farkı (delta = hedef - mevcut_fiyat) üzerinden tahmin üreten model paketi."""

    model_code: str
    model_name: str
    version: str
    target_col: str
    anchor_col: str
    selected_numeric_features: list[str]
    categorical_features: list[str]
    series_stat_features: list[str]
    series_encoder: SeriesHistoryStatsEncoder
    pipeline: Any
    training_date_range: dict[str, str]
    training_row_count: int
    training_clean_row_count: int
    training_series_counts: dict[str, int]
    hyperparameters: dict[str, Any]
    library_versions: dict[str, str]
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    trained_at: str

    def predict_price(self, df: pd.DataFrame) -> np.ndarray:
        """Verilen özellik tablosu için özgün TL ölçeğinde hedef fiyat tahmini üretir."""
        if self.anchor_col not in df.columns:
            raise ValueError(f'Dayanak fiyat sütunu ({self.anchor_col}) bulunamadı.')
        anchor = df[self.anchor_col].to_numpy(dtype=float)
        if not np.isfinite(anchor).all() or not (anchor > 0).all():
            raise ValueError(f'{self.anchor_col} sütununda geçersiz (NaN/<=0) fiyat var.')
        X_enriched = self.series_encoder.transform(df)
        delta_hat = np.asarray(self.pipeline.predict(X_enriched), dtype=float)
        raw_price = anchor + delta_hat
        return np.maximum(raw_price, MIN_PRICE_FLOOR)


def enforce_min_max_consistency(
    pred_min: np.ndarray | float,
    pred_max: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tahmin edilen alt fiyatın üst fiyatı geçmesini denetler ve deterministik düzeltir.

    Kural (doğrulamada sabitlenmiştir):
      tahmin_min > tahmin_max olan kayıtlarda alt = min(alt, üst), üst = max(alt, üst)
      uygulanır ve hangi kayıtlarda düzeltme yapıldığı boolean maske olarak döndürülür.
    """
    p_min = np.maximum(np.asarray(pred_min, dtype=float), MIN_PRICE_FLOOR)
    p_max = np.maximum(np.asarray(pred_max, dtype=float), MIN_PRICE_FLOOR)
    if p_min.shape != p_max.shape:
        raise ValueError('pred_min ve pred_max boyutları uyuşmuyor.')
    inverted_mask = p_min > p_max
    fixed_min = np.minimum(p_min, p_max)
    fixed_max = np.maximum(p_min, p_max)
    return fixed_min, fixed_max, inverted_mask


class HalPredictionService:
    """Kaydedilmiş model dosyalarını yükleyen ve yeniden eğitim yapmadan tahmin sunan servis."""

    def __init__(self, asama4_dir: Path | str | None = None) -> None:
        self.asama4_dir = Path(asama4_dir).resolve() if asama4_dir else (ROOT / 'ciktilar/asama4').resolve()
        self.models_dir = self.asama4_dir / 'modeller'
        self.registry_path = self.asama4_dir / 'model_kayit_defteri.json'
        if not self.registry_path.is_file():
            raise FileNotFoundError(f'Model kayıt defteri bulunamadı: {self.registry_path}')
        self.registry = json.loads(self.registry_path.read_text(encoding='utf-8'))
        self.selected_model_code: str = self.registry['secilen_ana_model_kodu']
        self.artifacts: dict[tuple[str, str], ResidualTargetModelArtifact] = {}
        self._load_and_verify_models()

    def _load_and_verify_models(self) -> None:
        for entry in self.registry.get('modeller', []):
            model_code = entry['model_kodu']
            target_col = entry['hedef']
            file_path = self.models_dir / entry['dosya_adi']
            if not file_path.is_file():
                raise FileNotFoundError(f'Kayıtlı model dosyası eksik: {file_path}')
            actual_hash = sha256(file_path)
            if actual_hash != entry['sha256']:
                raise ValueError(f'Model dosyası SHA-256 özeti uyuşmuyor: {file_path.name}')
            artifact = joblib.load(file_path)
            if not isinstance(artifact, ResidualTargetModelArtifact):
                raise TypeError(f'Geçersiz model paketi tipi: {type(artifact).__name__}')
            if artifact.model_code != model_code or artifact.target_col != target_col:
                raise ValueError(f'Model paketi metadatası kayıt defteriyle uyuşmuyor: {file_path.name}')
            self.artifacts[(model_code, target_col)] = artifact

    def available_models(self) -> list[str]:
        return ['naive_baseline', 'linear_regression', 'random_forest', 'xgboost']

    def predict_dataframe(self, df: pd.DataFrame, model_code: str | None = None) -> pd.DataFrame:
        """Özellikleri hazır (OzellikHazir=True) veri çerçevesi için min/max tahmini üretir."""
        code = model_code or self.selected_model_code
        if code not in MODEL_NAMES:
            raise ValueError(f'Bilinmeyen model kodu: {code}')
        if df.empty:
            raise ValueError('Tahmin için boş veri çerçevesi verildi.')
        for req in ('SeriId', 'fiyat_min', 'fiyat_max'):
            if req not in df.columns:
                raise ValueError(f'Tahmin girdisinde zorunlu sütun eksik: {req}')

        out = df.copy()
        f_min = out['fiyat_min'].to_numpy(dtype=float)
        f_max = out['fiyat_max'].to_numpy(dtype=float)
        if not (np.isfinite(f_min).all() and np.isfinite(f_max).all()):
            raise ValueError('Mevcut fiyatlarda NaN veya sonsuz değer var.')
        if not ((f_min > 0).all() and (f_max >= f_min).all()):
            raise ValueError('Mevcut fiyat aralığı geçersiz (fiyat_min <= 0 veya fiyat_min > fiyat_max).')

        if code == 'naive_baseline':
            raw_min = f_min.copy()
            raw_max = f_max.copy()
            inverted = np.zeros(len(out), dtype=bool)
            fixed_min, fixed_max = raw_min, raw_max
            train_counts = self.artifacts[(self.selected_model_code, 'hedef_min')].training_series_counts
        else:
            art_min = self.artifacts[(code, 'hedef_min')]
            art_max = self.artifacts[(code, 'hedef_max')]
            raw_min = art_min.predict_price(out)
            raw_max = art_max.predict_price(out)
            fixed_min, fixed_max, inverted = enforce_min_max_consistency(raw_min, raw_max)
            train_counts = art_min.training_series_counts

        out['ModelKodu'] = code
        out['ModelAdi'] = MODEL_NAMES[code]
        out['ham_tahmin_min'] = raw_min
        out['ham_tahmin_max'] = raw_max
        out['tahmin_min'] = fixed_min
        out['tahmin_max'] = fixed_max
        out['tahmin_orta'] = (fixed_min + fixed_max) / 2.0
        out['mevcut_orta'] = (f_min + f_max) / 2.0
        out['TersAralikDuzeltildi'] = inverted
        out['TahminAraligiGecerli'] = fixed_min <= fixed_max
        out['beklenen_degisim_min_tl'] = fixed_min - f_min
        out['beklenen_degisim_max_tl'] = fixed_max - f_max
        out['beklenen_degisim_orta_tl'] = out['tahmin_orta'] - out['mevcut_orta']
        out['beklenen_degisim_min_yuzde'] = (fixed_min - f_min) / f_min * 100.0
        out['beklenen_degisim_max_yuzde'] = (fixed_max - f_max) / f_max * 100.0
        out['beklenen_degisim_orta_yuzde'] = (out['tahmin_orta'] - out['mevcut_orta']) / out['mevcut_orta'] * 100.0

        counts = out['SeriId'].map(train_counts).fillna(0).astype('int64')
        out['EgitimOrnekSayisi'] = counts
        out['EgitimdeGoruldu'] = counts.gt(0)
        out['DusukEgitimKapsami'] = counts.lt(LOW_TRAINING_SAMPLE_THRESHOLD)
        return out

    def predict_for_series_row(
        self,
        series_id: str,
        latest_features_df: pd.DataFrame,
        status_df: pd.DataFrame,
        clean_df: pd.DataFrame | None = None,
        model_code: str | None = None,
        allow_naive_fallback: bool = True,
    ) -> dict[str, Any]:
        """Tek bir ürün serisi için son gözlem durumunu denetleyerek tahmin veya geri dönüş üretir."""
        code = model_code or self.selected_model_code
        if code not in MODEL_NAMES:
            raise ValueError(f'Bilinmeyen model kodu: {code}')

        ready_match = latest_features_df.loc[latest_features_df['SeriId'].eq(series_id)]
        if not ready_match.empty:
            pred_df = self.predict_dataframe(ready_match.iloc[[0]], model_code=code)
            row = pred_df.iloc[0]
            warnings_list = []
            if not bool(row.EgitimdeGoruldu):
                warnings_list.append(
                    'Bu seri eğitim döneminde (2016–2024) yer almamıştır; tahmin ortak modelin genel parametreleriyle üretilmiştir.'
                )
            elif bool(row.DusukEgitimKapsami):
                warnings_list.append(
                    f'Bu serinin eğitim döneminde yalnızca {int(row.EgitimOrnekSayisi)} örneği bulunmaktadır (düşük eğitim kapsamı).'
                )
            if bool(row.TersAralikDuzeltildi):
                warnings_list.append(
                    'Ham model çıktısında tahmini minimum fiyat maksimum fiyatı geçtiği için sıralama düzeltmesi uygulandı.'
                )
            return {
                'tahmin_uretildi': True,
                'geri_donus_kullanildi': False,
                'seri_id': series_id,
                'urun_ad': str(row.UrunAd),
                'birim_kodu': int(row.BirimTurId),
                'birim_aciklamasi': unit_label(int(row.BirimTurId)),
                'kategori_id': int(row.KategoriId),
                'kategori': CATEGORY[int(row.KategoriId)],
                'son_gozlem_tarihi': str(row.Tarih),
                'hedef_tarih_aciklamasi': 'Bir sonraki hal işlem günü (kesin takvim tarihi kaynakta yayımlanmamıştır)',
                'kullanilan_model_kodu': code,
                'kullanilan_model_adi': MODEL_NAMES[code],
                'istenen_model_kodu': code,
                'fiyat_min': float(row.fiyat_min),
                'fiyat_max': float(row.fiyat_max),
                'fiyat_orta': float(row.mevcut_orta),
                'tahmin_min': float(row.tahmin_min),
                'tahmin_max': float(row.tahmin_max),
                'tahmin_orta': float(row.tahmin_orta),
                'beklenen_degisim_min_tl': float(row.beklenen_degisim_min_tl),
                'beklenen_degisim_max_tl': float(row.beklenen_degisim_max_tl),
                'beklenen_degisim_orta_tl': float(row.beklenen_degisim_orta_tl),
                'beklenen_degisim_min_yuzde': float(row.beklenen_degisim_min_yuzde),
                'beklenen_degisim_max_yuzde': float(row.beklenen_degisim_max_yuzde),
                'beklenen_degisim_orta_yuzde': float(row.beklenen_degisim_orta_yuzde),
                'egitim_ornek_sayisi': int(row.EgitimOrnekSayisi),
                'egitimde_goruldu': bool(row.EgitimdeGoruldu),
                'dusuk_egitim_kapsami': bool(row.DusukEgitimKapsami),
                'ters_aralik_duzeltildi': bool(row.TersAralikDuzeltildi),
                'uyarilar': warnings_list,
            }

        # Son günde 20 kesintisiz gözlem özelliği hazır olmayan seriler
        status_match = status_df.loc[status_df['SeriId'].eq(series_id)]
        if status_match.empty:
            raise ValueError(f'SeriId veri kümesinde bulunamadı: {series_id}')
        st = status_match.iloc[0]
        if not allow_naive_fallback or clean_df is None:
            return {
                'tahmin_uretildi': False,
                'geri_donus_kullanildi': False,
                'seri_id': series_id,
                'urun_ad': str(st.UrunAd),
                'birim_kodu': int(st.BirimTurId),
                'birim_aciklamasi': unit_label(int(st.BirimTurId)),
                'son_gozlem_tarihi': str(st.Tarih),
                'neden': (
                    'Son veri gününde 20 gözlemlik kesintisiz geçmiş penceresi tamamlanmadığı için '
                    'makine öğrenmesi özellikleri üretilemedi; sahte tahmin gösterilmez.'
                ),
            }

        last_obs = clean_df.loc[clean_df['SeriId'].eq(series_id)].sort_values('Tarih').iloc[-1]
        f_min = float(last_obs.EnDusukFiyat)
        f_max = float(last_obs.EnYuksekFiyat)
        f_mid = (f_min + f_max) / 2.0
        train_counts = self.artifacts[(self.selected_model_code, 'hedef_min')].training_series_counts
        t_cnt = int(train_counts.get(series_id, 0))
        return {
            'tahmin_uretildi': True,
            'geri_donus_kullanildi': True,
            'seri_id': series_id,
            'urun_ad': str(last_obs.UrunAd),
            'birim_kodu': int(last_obs.BirimTurId),
            'birim_aciklamasi': unit_label(int(last_obs.BirimTurId)),
            'kategori_id': int(last_obs.KategoriId),
            'kategori': CATEGORY[int(last_obs.KategoriId)],
            'son_gozlem_tarihi': str(last_obs.Tarih),
            'hedef_tarih_aciklamasi': 'Son gözlemden sonraki hal işlem günü (kesintili seri — Naive Baseline geri dönüşü)',
            'kullanilan_model_kodu': 'naive_baseline',
            'kullanilan_model_adi': 'Naive Baseline (Kesintili Seri Geri Dönüşü)',
            'istenen_model_kodu': code,
            'fiyat_min': f_min,
            'fiyat_max': f_max,
            'fiyat_orta': f_mid,
            'tahmin_min': f_min,
            'tahmin_max': f_max,
            'tahmin_orta': f_mid,
            'beklenen_degisim_min_tl': 0.0,
            'beklenen_degisim_max_tl': 0.0,
            'beklenen_degisim_orta_tl': 0.0,
            'beklenen_degisim_min_yuzde': 0.0,
            'beklenen_degisim_max_yuzde': 0.0,
            'beklenen_degisim_orta_yuzde': 0.0,
            'egitim_ornek_sayisi': t_cnt,
            'egitimde_goruldu': t_cnt > 0,
            'dusuk_egitim_kapsami': t_cnt < LOW_TRAINING_SAMPLE_THRESHOLD,
            'ters_aralik_duzeltildi': False,
            'uyarilar': [
                f'Son veri gününde 20 kesintisiz gözlem penceresi bulunmadığından (son gözlem: {last_obs.Tarih}) '
                f'makine öğrenmesi modeli yerine açıkça Naive Baseline (son gözlenen fiyatın korunması) geri dönüşü uygulanmıştır.'
            ],
        }
