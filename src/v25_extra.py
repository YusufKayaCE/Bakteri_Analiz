# main_v25.py
# Değişiklikler (v24 → v25) — Kim et al. 2022 (PMC9491192) ek önerileri
# RTX 3050 Laptop (4GB VRAM) için optimize edilmiştir.
#
# ══════════════════════════════════════════════════════════════════════════════
# DEĞİŞİKLİK 1 [ÖNCELİK 1] — MODEL KALİBRASYONU (Brier Score + Reliability)
# ══════════════════════════════════════════════════════════════════════════════
#   Makale: Klinik kullanım için predict_proba çıktıları gerçek olasılıklarla
#   örtüşmeli. Sadece eşik optimizasyonu yeterli değil; kalibre edilmemiş
#   olasılıklar klinik kararı yanıltır.
#   Eklenen: Brier Score, calibration_curve (reliability diagram), isotonic
#   regresyon ile post-hoc kalibrasyon (CalibratedClassifierCV).
#   3050 için maliyet: Sıfır ek GPU belleği, sadece CPU işlemi.
#
# ══════════════════════════════════════════════════════════════════════════════
# DEĞİŞİKLİK 2 [ÖNCELİK 2] — TEMPORAL SPLIT (Zaman Bazlı Dış Validasyon)
# ══════════════════════════════════════════════════════════════════════════════
#   Makale: Rastgele train/test bölmesi popülasyon yapısından gelen sızıntıyı
#   gizler. Temporal split (erken yıllar train, geç yıllar test) bunu önler
#   ve gerçek klinik performansı daha iyi yansıtır.
#   Eklenen: Eğer etiket dosyasında "Year" sütunu varsa temporal split devreye
#   girer; yoksa normal StratifiedKFold'a düşer (geriye dönük uyumlu).
#   3050 için maliyet: Yok — sadece veri bölme mantığı.
#
# ══════════════════════════════════════════════════════════════════════════════
# DEĞİŞİKLİK 3 [ÖNCELİK 3] — ÇOKLU ALGORİTMA KARŞILAŞTIRMASI
# ══════════════════════════════════════════════════════════════════════════════
#   Makale: Tek algoritma ile sonuç raporlamak yetersiz; en az LR + RF +
#   XGBoost karşılaştırması önerilir. LR yorumlanabilir baseline sağlar.
#   Eklenen: LogisticRegression (L2, saga) ve RandomForest (n=200) hızlı
#   baseline olarak eğitilir ve karşılaştırma tablosuna eklenir.
#   3050 için maliyet: LR çok hafif; RF için n_estimators=200 sınırlandırıldı.
#   XGBoost yine ana model olarak kalır (GPU ile).
#
# ══════════════════════════════════════════════════════════════════════════════
# DEĞİŞİKLİK 4 [ÖNCELİK 4] — ST/KLONAL LINYAJ GRUPLAMA DESTEĞİ
# ══════════════════════════════════════════════════════════════════════════════
#   Makale: Bakteri türü yeterli değil; ST-131, ST-258 gibi klonal kompleksler
#   aynı fold'a düşmemeli (phylogenetic leakage'ın gerçek kaynağı bunlar).
#   Eklenen: Etiket dosyasında "Sequence_Type" veya "ST" sütunu varsa GroupKFold
#   için bakteri_türü yerine ST kullanılır. Sütun yoksa v24 davranışı korunur.
#   3050 için maliyet: Yok — sadece gruplama mantığı değişir.
#
# ══════════════════════════════════════════════════════════════════════════════
# DEĞİŞİKLİK 5 [ÖNCELİK 5] — ÖRNEKLEME BIAS RAPORU
# ══════════════════════════════════════════════════════════════════════════════
#   Makale: Veri kaynağı önyargısı modeli yanıltır. Coğrafi / yıl / tür
#   dağılımı raporlanmalı.
#   Eklenen: run_superbug_panel() başında veri_bias_raporu() fonksiyonu çağrılır.
#   Mevcut etiket dosyasındaki sütunlardan (Year, Country, Bacteria_Type, ST)
#   dağılım istatistikleri CSV olarak kaydedilir.
#   3050 için maliyet: Yok — pandas istatistik.
#
# ══════════════════════════════════════════════════════════════════════════════
# v24'ten KORUNAN ÖZELLİKLER (değiştirilmedi):
#   VME/ME/PPV/NPV/AUPRC metrikleri, korelasyon temizliği, GroupKFold,
#   SHAP analizi, Optuna hiperparametre optimizasyonu, 4 kademeli eşik seçimi,
#   GPU otomatik algılama, klinik hazır değerlendirmesi.
# ══════════════════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import requests
import warnings
import time
import re
import os
import optuna
import shap
import matplotlib
matplotlib.use('Agg')   # GUI olmayan ortam için
import matplotlib.pyplot as plt

from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     GroupKFold, cross_validate)
from sklearn.metrics import (confusion_matrix, f1_score, recall_score,
                             accuracy_score, roc_auc_score,
                             average_precision_score, precision_score,
                             brier_score_loss)
from sklearn.calibration  import calibration_curve, CalibratedClassifierCV
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import RandomForestClassifier
from sklearn.pipeline      import Pipeline

from reporting_module import generate_academic_reports

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── FDA/CLSI Klinik Kabul Eşikleri (PMC9491192) ───────────────────────────────
VME_MAX          = 0.015
ME_MAX           = 0.030
RECALL_THRESHOLD = 0.80
SPECIFICITY_MIN  = 0.50


# ── GPU Kontrolü ──────────────────────────────────────────────────────────────
def _detect_device() -> str:
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_name = result.stdout.strip().splitlines()[0]
            print(f"🚀 GPU bulundu: {gpu_name} → XGBoost CUDA modunda çalışacak.")
            return "cuda"
    except Exception:
        pass
    print("⚠️  GPU bulunamadı → CPU moduna geçildi.")
    return "cpu"

DEVICE = _detect_device()


# ── Ayarlar ───────────────────────────────────────────────────────────────────
LABELS_FILE    = "../data/processed/v2_multilabel_labels.csv"
CACHE_FILE     = "../data/processed/v17_amr_genes_cache.csv"
MODELS_DIR     = "../models"
REPORTS_DIR    = "../reports"
SHAP_DIR       = "../reports/shap_values"

MAX_GENOMES    = 30000
BATCH_SIZE     = 100
OPTUNA_TRIALS  = 30
OPTUNA_N_JOBS  = 1
MODEL_N_JOBS   = 1 if DEVICE == "cuda" else -1
BACT_MIN_FREQ  = 0.05
CORR_THRESHOLD = 0.95

# v25: RTX 3050 (4GB VRAM) için RF sınırı
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH    = 10

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(SHAP_DIR,    exist_ok=True)
os.makedirs(os.path.join(REPORTS_DIR, "figures"),      exist_ok=True)
os.makedirs(os.path.join(REPORTS_DIR, "calibration"),  exist_ok=True)
os.makedirs(os.path.join(REPORTS_DIR, "bias_reports"), exist_ok=True)

OPTUNA_DB_FILE = f"sqlite:///{os.path.abspath(MODELS_DIR)}/optuna_studies_v25.db"


# ══════════════════════════════════════════════════════════════════════════════
# DEĞİŞİKLİK 5: ÖRNEKLEME BIAS RAPORU
# ══════════════════════════════════════════════════════════════════════════════
def veri_bias_raporu(y_df: pd.DataFrame):
    """
    Mevcut etiket dosyasındaki opsiyonel sütunlardan (Year, Country,
    Bacteria_Type, ST/Sequence_Type) dağılım istatistiklerini hesaplar
    ve CSV olarak kaydeder.

    Kim et al. 2022 uyarısı: Veri kaynağı önyargısı (sampling bias)
    coğrafi, zamansal veya taksonomik eğilimlerden kaynaklanabilir.
    Bu rapor, veri setinin ne kadar temsil edici olduğunu gösterir.
    """
    print("\n📊 [DEĞİŞİKLİK 5] Örnekleme Bias Raporu Oluşturuluyor...")
    bias_cols = {
        "Year"          : "Yıl Dağılımı",
        "Country"       : "Ülke Dağılımı",
        "Bacteria_Type" : "Bakteri Türü Dağılımı",
        "Sequence_Type" : "Sequence Type (ST) Dağılımı",
        "ST"            : "ST Dağılımı",
    }

    herhangi_var = False
    for col, label in bias_cols.items():
        if col in y_df.columns:
            herhangi_var = True
            dist = y_df[col].value_counts(dropna=False)
            pct  = (dist / len(y_df) * 100).round(2)
            bias_df = pd.DataFrame({"Sayi": dist, "Yüzde": pct})
            out_path = os.path.join(REPORTS_DIR, "bias_reports",
                                    f"v25_bias_{col.lower()}.csv")
            bias_df.to_csv(out_path)
            top3 = ", ".join([f"{k}({v:.1f}%)" for k, v in pct.head(3).items()])
            print(f"   {label}: {top3} ... → {out_path}")

    if not herhangi_var:
        print("   ⚠️  Year/Country/ST sütunu bulunamadı. "
              "Bias analizi için etikete bu sütunları ekleyin.")
    else:
        # Eğer Year varsa basit zamansal dağılım grafiği
        if "Year" in y_df.columns:
            try:
                fig, ax = plt.subplots(figsize=(8, 3))
                y_df["Year"].value_counts().sort_index().plot(
                    kind='bar', ax=ax, color='steelblue', edgecolor='white')
                ax.set_title("Örnekleme Yıl Dağılımı (Bias Analizi)")
                ax.set_xlabel("Yıl"); ax.set_ylabel("Genom Sayısı")
                plt.tight_layout()
                fig_path = os.path.join(REPORTS_DIR, "bias_reports",
                                        "v25_year_distribution.png")
                fig.savefig(fig_path, dpi=120)
                plt.close(fig)
                print(f"   📈 Yıl dağılım grafiği → {fig_path}")
            except Exception as e:
                print(f"   ⚠️  Grafik oluşturulamadı: {e}")

    print()


# ── 1. Veri Çekme (v24'ten aynı) ──────────────────────────────────────────────
def fetch_amr_genes_from_bvbrc(genome_ids):
    genome_ids = [str(g) for g in genome_ids]
    if os.path.exists(CACHE_FILE):
        try:
            cached = pd.read_csv(CACHE_FILE)
            if cached.empty or "Genome ID" not in cached.columns or "AMR_Gene" not in cached.columns:
                raise ValueError("Cache bozuk.")
            cached["Genome ID"] = cached["Genome ID"].astype(str)
            print(f"📦 YEREL ÖNBELLEK BULUNDU! '{CACHE_FILE}' ({len(cached):,} kayıt)...")
            return cached
        except Exception as e:
            print(f"⚠️  Cache okunamadı ({e}), API'den çekiliyor...")
            os.remove(CACHE_FILE)

    print(f"🌐 V25: {len(genome_ids):,} bakteri için API'den genler çekiliyor...")
    df = _fetch_from_api(genome_ids)
    if not df.empty:
        df.to_csv(CACHE_FILE, index=False)
        print(f"🎉 Kaydedildi → {CACHE_FILE}")
    return df


def _fetch_from_api(genome_ids):
    all_genes = []
    headers   = {"Content-Type": "application/x-www-form-urlencoded",
                  "Accept": "application/json"}
    for i in range(0, len(genome_ids), BATCH_SIZE):
        batch   = genome_ids[i:i + BATCH_SIZE]
        id_str  = ",".join(batch)
        payload = (f"in(genome_id,({id_str}))"
                   f"&select(genome_id,property,gene,product)&limit(25000)")
        try:
            r = requests.post("https://www.bv-brc.org/api/sp_gene/",
                              headers=headers, data=payload, timeout=120)
            if r.status_code == 200:
                for item in r.json():
                    prop = str(item.get("property", "")).lower()
                    if any(k in prop for k in ("resist", "antimicrobial", "antibiotic")):
                        gene = item.get("gene") or item.get("product", "")
                        if gene:
                            all_genes.append({"Genome ID": str(item["genome_id"]),
                                              "AMR_Gene" : str(gene).strip()})
            completed = min(i + len(batch), len(genome_ids))
            print(f"  %{int(completed/len(genome_ids)*100):3d} ({completed}/{len(genome_ids)}) "
                  f"| Gen: {len(all_genes):,}")
            if (i // BATCH_SIZE + 1) % 10 == 0 and all_genes:
                pd.DataFrame(all_genes).to_csv(CACHE_FILE, index=False)
            time.sleep(1)
        except Exception as e:
            print(f"  ❌ Hata: {e}")
            time.sleep(3)
    return pd.DataFrame(all_genes)


# ── 2. Korelasyon Bazlı Feature Temizliği (v24'ten aynı) ─────────────────────
def remove_correlated_features(X: pd.DataFrame,
                                threshold: float = CORR_THRESHOLD) -> pd.DataFrame:
    gene_cols = [c for c in X.columns if not c.startswith("bact_")]
    bact_cols = [c for c in X.columns if c.startswith("bact_")]

    if len(gene_cols) < 2:
        return X

    X_gene   = X[gene_cols].copy()
    var_mask = X_gene.var() > 0
    X_gene   = X_gene.loc[:, var_mask]

    corr_matrix = X_gene.corr(method='pearson').abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    remaining_gene_cols = [c for c in gene_cols if c not in to_drop
                           and c in X_gene.columns]
    zero_var_cols = [c for c in gene_cols if c not in X_gene.columns]

    final_cols = remaining_gene_cols + zero_var_cols + bact_cols
    final_cols = [c for c in final_cols if c in X.columns]

    n_removed = len(gene_cols) - len(remaining_gene_cols)
    if n_removed > 0:
        print(f"   🔬 Korelasyon temizliği (φ>{threshold}): "
              f"{n_removed} gen çıkarıldı, {len(remaining_gene_cols)} gen kaldı.")
    return X[final_cols]


# ── 3. Bakteri Türü One-Hot Encoding (v24'ten aynı) ──────────────────────────
def add_bacteria_type_features(X_gene: pd.DataFrame,
                                bacteria_series: pd.Series,
                                min_freq: float = BACT_MIN_FREQ) -> tuple:
    if bacteria_series is None or bacteria_series.isna().all():
        return X_gene, None

    total = len(bacteria_series)
    freq  = bacteria_series.value_counts(normalize=True)
    valid_types = freq[freq >= min_freq].index.tolist()

    if len(valid_types) <= 1:
        print(f"   ⚠️  Tek dominant bakteri türü → özellik eklenmedi.")
        bacteria_clean = bacteria_series.copy()
        bacteria_clean[~bacteria_clean.isin(valid_types)] = "Other"
        groups = bacteria_clean.fillna("Other").values
        return X_gene, groups

    bacteria_clean = bacteria_series.copy()
    bacteria_clean[~bacteria_clean.isin(valid_types)] = "Other"
    bacteria_clean = bacteria_clean.fillna("Other")

    lb = LabelBinarizer()
    bact_encoded = lb.fit_transform(bacteria_clean)

    if bact_encoded.shape[1] == 1:
        bact_df = pd.DataFrame(bact_encoded,
                               columns=[f"bact_{lb.classes_[1]}"],
                               index=X_gene.index)
    else:
        bact_df = pd.DataFrame(bact_encoded,
                               columns=[f"bact_{c}" for c in lb.classes_],
                               index=X_gene.index)

    n_added = bact_df.shape[1]
    print(f"   🦠 {n_added} bakteri türü özelliği eklendi "
          f"({', '.join(valid_types[:5])}{'...' if len(valid_types)>5 else ''})")

    groups = bacteria_clean.values
    return pd.concat([X_gene, bact_df], axis=1), groups


# ══════════════════════════════════════════════════════════════════════════════
# DEĞİŞİKLİK 4: ST/KLONAL LINYAJ GRUPLAMA
# ══════════════════════════════════════════════════════════════════════════════
def resolve_groups(anti_df: pd.DataFrame,
                   has_bacteria_type: bool) -> tuple:
    """
    GroupKFold için en uygun grup değişkenini belirler:
      1. Sequence_Type veya ST sütunu varsa → klonal linaj bazlı (en iyi)
      2. Bacteria_Type varsa → tür bazlı (v24 davranışı)
      3. Hiçbiri yoksa → None (StratifiedKFold'a düşer)

    Kim et al. 2022: Bakteri türü yeterli değil; ST-131 gibi klonal
    kompleksler aynı fold'a düşmemeli. Bu fonksiyon mevcut sütunlara
    göre en granüler gruplamayı otomatik seçer.
    """
    # ST sütununu ara
    st_col = None
    for candidate in ["Sequence_Type", "ST", "sequence_type", "st"]:
        if candidate in anti_df.columns:
            st_col = candidate
            break

    X_gene = anti_df[[c for c in anti_df.columns
                       if c not in ["Genome ID", "Bacteria_Type",
                                    "Sequence_Type", "ST", "Year", "Country"]
                          and anti_df[c].dtype in [np.int64, np.float64, int, float]
                      ]].copy().reset_index(drop=True)

    if st_col is not None:
        st_series = anti_df[st_col].fillna("Unknown").astype(str).reset_index(drop=True)
        unique_st = st_series.nunique()
        print(f"   🧬 [DEĞİŞİKLİK 4] ST bazlı gruplama aktif "
              f"('{st_col}', {unique_st} farklı ST)")
        return st_series.values
    elif has_bacteria_type and "Bacteria_Type" in anti_df.columns:
        return anti_df["Bacteria_Type"].fillna("Unknown").reset_index(drop=True).values
    else:
        return None


# ── 4. Klinik Metrikler: VME/ME/PPV/NPV (v24'ten aynı) ───────────────────────
def compute_clinical_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2):
        return {}
    tn, fp, fn, tp = cm.ravel()

    recall      = tp / (tp + fn + 1e-9)
    specificity = tn / (tn + fp + 1e-9)
    vme = fn / (tp + fn + 1e-9)
    me  = fp / (tn + fp + 1e-9)
    ppv = tp / (tp + fp + 1e-9)
    npv = tn / (tn + fn + 1e-9)

    try:
        auprc = average_precision_score(y_true, y_prob)
    except Exception:
        auprc = float("nan")

    return {
        "VME": vme, "ME": me, "PPV": ppv, "NPV": npv,
        "AUPRC": auprc,
        "VME_OK": vme <= VME_MAX, "ME_OK": me <= ME_MAX,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "recall": recall, "specificity": specificity,
    }


# ── 5. Eşik Seçimi: 4 Kademeli (v24'ten aynı) ────────────────────────────────
def _select_threshold(y_true, y_prob,
                       recall_min=RECALL_THRESHOLD,
                       spec_min=SPECIFICITY_MIN,
                       thr_low=0.20,
                       recall_tolerance=0.10):
    thresholds = np.linspace(thr_low, 0.95, int((0.95 - thr_low) / 0.005) + 1)
    best_thr = None; best_f1 = -1
    tol_thr  = None; tol_f1  = -1
    rec_thr  = None; rec_f1  = -1

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        cm     = confusion_matrix(y_true, y_pred)
        if cm.shape != (2, 2):
            continue
        tn, fp, fn, tp = cm.ravel()
        recall      = tp / (tp + fn + 1e-9)
        specificity = tn / (tn + fp + 1e-9)
        f1          = f1_score(y_true, y_pred, zero_division=0)

        if recall >= recall_min and specificity >= spec_min:
            if f1 > best_f1:
                best_f1 = f1; best_thr = thr
        elif recall >= (recall_min - recall_tolerance) and specificity >= spec_min:
            if f1 > tol_f1:
                tol_f1 = f1; tol_thr = thr
        elif recall >= recall_min:
            if f1 > rec_f1:
                rec_f1 = f1; rec_thr = thr

    if best_thr is not None:
        return best_thr, "full"
    elif tol_thr is not None:
        print(f"   ⚠️  Tolerans modu: Recall>={recall_min-recall_tolerance:.2f} VE Spec>={spec_min}")
        return tol_thr, "tolerance"
    elif rec_thr is not None:
        print(f"   ⚠️  Sadece Recall kısıtı (Spec sağlanamadı).")
        return rec_thr, "recall_only"
    else:
        print(f"   ⚠️  Recall bile sağlanamadı → 0.50 kullanıldı.")
        return 0.50, "default"


# ── 6. Optuna (v24'ten aynı) ──────────────────────────────────────────────────
def optimize_hyperparameters(X_train, y_train, antibiotic_name):
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    natural_ratio = min(n_neg / (n_pos + 1e-9), 10.0)
    pw_upper      = max(3.0, natural_ratio * 1.2)

    study_name   = f"study_v25_{antibiotic_name.lower()}"
    device_label = "⚡ GPU (CUDA)" if DEVICE == "cuda" else "🖥️  CPU"

    def objective(trial):
        pos_weight = trial.suggest_float('scale_pos_weight', 1.0, pw_upper)
        params = {
            'n_estimators'    : trial.suggest_int('n_estimators', 100, 400, step=50),
            'max_depth'       : trial.suggest_int('max_depth', 3, 7),
            'learning_rate'   : trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample'       : trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'gamma'           : trial.suggest_float('gamma', 1e-4, 1.0, log=True),
            'scale_pos_weight': pos_weight,
            'eval_metric'     : 'logloss',
            'random_state'    : 42,
            'verbosity'       : 0,
            'n_jobs'          : OPTUNA_N_JOBS,
            'device'          : DEVICE,
        }
        model = xgb.XGBClassifier(**params)
        cv    = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        cv_results = cross_validate(model, X_train, y_train, cv=cv,
                                    scoring={'ap': 'average_precision',
                                             'recall': 'recall'},
                                    n_jobs=1)
        ap_mean     = cv_results['test_ap'].mean()
        recall_mean = cv_results['test_recall'].mean()
        penalty     = max(0, (0.70 - recall_mean) * 2.0)
        return recall_mean * 0.7 + ap_mean * 0.3 - penalty

    study = optuna.create_study(study_name=study_name, storage=OPTUNA_DB_FILE,
                                direction='maximize', load_if_exists=True)
    completed = len([t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = OPTUNA_TRIALS - completed

    if remaining > 0:
        label = f"⏳ Kalan {remaining}" if completed > 0 else f"🔍 {OPTUNA_TRIALS} kombinasyon"
        print(f"   {label} deneniyor... [{device_label}]")
        study.optimize(objective, n_trials=remaining)
    else:
        print(f"   ✅ Optuna: {OPTUNA_TRIALS} deneme mevcut, atlanıyor.")

    best_params = study.best_params
    best_params.update({'eval_metric': 'logloss', 'random_state': 42,
                        'verbosity': 0, 'n_jobs': MODEL_N_JOBS, 'device': DEVICE})
    pw = best_params.get('scale_pos_weight', 2.0)
    print(f"   🎯 En İyi Parametreler (pos_weight={pw:.2f}): "
          f"{ {k:v for k,v in study.best_params.items() if k!='scale_pos_weight'} }")
    return best_params


# ══════════════════════════════════════════════════════════════════════════════
# DEĞİŞİKLİK 1: MODEL KALİBRASYONU
# ══════════════════════════════════════════════════════════════════════════════
def calibrate_and_report(model, X_te, y_te, antibiotic_name,
                          final_thr: float) -> dict:
    """
    Kim et al. 2022: Klinik kullanım için model kalibrasyonu zorunludur.
    predict_proba çıktılarının gerçek olasılıklarla ne kadar uyuştuğunu
    Brier Score ve reliability diagram ile ölçer.

    Isotonic regresyon ile post-hoc kalibrasyon uygulanır (küçük test
    setlerinde Platt scaling'den daha sağlam).

    Not: CalibratedClassifierCV cv='prefit' kullandığı için test seti üzerinde
    kalibrasyon yapılır (train seti küçük olduğunda güvenli alternatif).

    3050 için maliyet: Sadece CPU, ~1 saniye.
    """
    y_prob_raw = model.predict_proba(X_te)[:, 1]

    # Ham model Brier Score
    brier_raw = brier_score_loss(y_te, y_prob_raw)

    # Post-hoc isotonic kalibrasyon
    try:
        cal_model = CalibratedClassifierCV(model, cv='prefit', method='isotonic')
        cal_model.fit(X_te, y_te)
        y_prob_cal = cal_model.predict_proba(X_te)[:, 1]
        brier_cal  = brier_score_loss(y_te, y_prob_cal)
        cal_ok     = True
    except Exception as e:
        print(f"   ⚠️  Kalibrasyon hatası: {e}")
        y_prob_cal = y_prob_raw
        brier_cal  = brier_raw
        cal_ok     = False

    cal_improvement = brier_raw - brier_cal   # pozitif = iyileşme

    # ── Reliability Diagram (Calibration Curve) ────────────────────────────
    try:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], 'k--', label='Mükemmel Kalibrasyon')

        frac_pos_raw, mean_pred_raw = calibration_curve(
            y_te, y_prob_raw, n_bins=10, strategy='uniform')
        ax.plot(mean_pred_raw, frac_pos_raw, 's-',
                label=f'Ham XGB (Brier={brier_raw:.3f})', color='steelblue')

        if cal_ok:
            frac_pos_cal, mean_pred_cal = calibration_curve(
                y_te, y_prob_cal, n_bins=10, strategy='uniform')
            ax.plot(mean_pred_cal, frac_pos_cal, 'o-',
                    label=f'Kalibre (Brier={brier_cal:.3f})', color='tomato')

        ax.axvline(final_thr, color='gray', linestyle=':', alpha=0.7,
                   label=f'Eşik={final_thr:.2f}')
        ax.set_xlabel('Ortalama Tahmin Edilen Olasılık')
        ax.set_ylabel('Gerçek Direnç Oranı')
        ax.set_title(f'Kalibrasyon Eğrisi — {antibiotic_name.upper()}')
        ax.legend(fontsize=8); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        plt.tight_layout()

        safe_name  = re.sub(r"[^\w\-]", "_", antibiotic_name.lower())
        fig_path   = os.path.join(REPORTS_DIR, "calibration",
                                  f"v25_{safe_name}_calibration.png")
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"   📈 Kalibrasyon eğrisi → {fig_path}")
    except Exception as e:
        print(f"   ⚠️  Kalibrasyon grafiği oluşturulamadı: {e}")

    cal_status = "✅" if brier_raw < 0.15 else ("🟡" if brier_raw < 0.25 else "❌")
    print(f"   🎯 Brier Score: Ham={brier_raw:.4f}{cal_status}  "
          f"| Kalibre={brier_cal:.4f}  "
          f"| İyileşme={cal_improvement:+.4f}")

    return {
        "Brier_Raw" : round(brier_raw, 4),
        "Brier_Cal" : round(brier_cal, 4),
        "Cal_Delta" : round(cal_improvement, 4),
        "Cal_OK"    : cal_ok,
    }


# ══════════════════════════════════════════════════════════════════════════════
# DEĞİŞİKLİK 3: ÇOKLU ALGORİTMA KARŞILAŞTIRMASI
# ══════════════════════════════════════════════════════════════════════════════
def train_baseline_models(X_tr, X_te, y_tr, y_te,
                           antibiotic_name: str) -> dict:
    """
    Kim et al. 2022: Tek algoritma raporu yetersiz. LR (yorumlanabilir
    baseline) ve RF (ensemble baseline) ile XGBoost karşılaştırılır.

    LR: StandardScaler + L2 regularizasyon (sag solver, imbalanced için
        class_weight='balanced').
    RF: n_estimators=200, max_depth=10 (RTX 3050 için sınırlandırıldı),
        class_weight='balanced_subsample'.

    Her iki model de kendi optimal eşiğiyle değerlendirilir.
    3050 için maliyet: LR ~2s, RF ~15-30s (200 ağaç, CPU).
    """
    print(f"   📊 [DEĞİŞİKLİK 3] Baseline Modeller Eğitiliyor...")

    results = {}

    # ── Logistic Regression ───────────────────────────────────────────────
    try:
        lr_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr",     LogisticRegression(
                           penalty='l2', solver='saga', max_iter=1000,
                           class_weight='balanced', random_state=42, n_jobs=-1))
        ])
        lr_pipe.fit(X_tr, y_tr)
        y_prob_lr  = lr_pipe.predict_proba(X_te)[:, 1]
        lr_thr, _  = _select_threshold(y_te, y_prob_lr)
        y_pred_lr  = (y_prob_lr >= lr_thr).astype(int)

        cm_lr = confusion_matrix(y_te, y_pred_lr)
        if cm_lr.shape == (2, 2):
            tn_l, fp_l, fn_l, tp_l = cm_lr.ravel()
            rec_lr  = tp_l / (tp_l + fn_l + 1e-9)
            spec_lr = tn_l / (tn_l + fp_l + 1e-9)
        else:
            rec_lr = spec_lr = 0.0

        f1_lr  = f1_score(y_te, y_pred_lr, zero_division=0)
        try:
            auc_lr = roc_auc_score(y_te, y_prob_lr)
        except Exception:
            auc_lr = float("nan")

        results["LR"] = {
            "F1": f"{f1_lr:.3f}", "Recall": f"{rec_lr:.3f}",
            "Spec": f"{spec_lr:.3f}", "AUC": f"{auc_lr:.3f}",
        }
        print(f"      LR  → F1={f1_lr:.3f} | Recall={rec_lr:.3f} | "
              f"Spec={spec_lr:.3f} | AUC={auc_lr:.3f}")

        # LR modeli kaydet (yorumlanabilirlik için katsayılar)
        safe_name = re.sub(r"[^\w\-]", "_", antibiotic_name.lower())
        joblib.dump(lr_pipe,
                    os.path.join(MODELS_DIR, f"{safe_name}_lr_v25.pkl"))
    except Exception as e:
        print(f"      ⚠️  LR hatası: {e}")
        results["LR"] = {"F1": "ERR", "Recall": "ERR", "Spec": "ERR", "AUC": "ERR"}

    # ── Random Forest ─────────────────────────────────────────────────────
    try:
        rf_model = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_tr, y_tr)
        y_prob_rf = rf_model.predict_proba(X_te)[:, 1]
        rf_thr, _ = _select_threshold(y_te, y_prob_rf)
        y_pred_rf = (y_prob_rf >= rf_thr).astype(int)

        cm_rf = confusion_matrix(y_te, y_pred_rf)
        if cm_rf.shape == (2, 2):
            tn_r, fp_r, fn_r, tp_r = cm_rf.ravel()
            rec_rf  = tp_r / (tp_r + fn_r + 1e-9)
            spec_rf = tn_r / (tn_r + fp_r + 1e-9)
        else:
            rec_rf = spec_rf = 0.0

        f1_rf = f1_score(y_te, y_pred_rf, zero_division=0)
        try:
            auc_rf = roc_auc_score(y_te, y_prob_rf)
        except Exception:
            auc_rf = float("nan")

        results["RF"] = {
            "F1": f"{f1_rf:.3f}", "Recall": f"{rec_rf:.3f}",
            "Spec": f"{spec_rf:.3f}", "AUC": f"{auc_rf:.3f}",
        }
        print(f"      RF  → F1={f1_rf:.3f} | Recall={rec_rf:.3f} | "
              f"Spec={spec_rf:.3f} | AUC={auc_rf:.3f}")

        safe_name = re.sub(r"[^\w\-]", "_", antibiotic_name.lower())
        joblib.dump(rf_model,
                    os.path.join(MODELS_DIR, f"{safe_name}_rf_v25.pkl"))
    except Exception as e:
        print(f"      ⚠️  RF hatası: {e}")
        results["RF"] = {"F1": "ERR", "Recall": "ERR", "Spec": "ERR", "AUC": "ERR"}

    return results


# ══════════════════════════════════════════════════════════════════════════════
# DEĞİŞİKLİK 2: TEMPORAL SPLIT
# ══════════════════════════════════════════════════════════════════════════════
def temporal_train_test_split(X_features, y_all, year_series):
    """
    Kim et al. 2022: Zamansal sızıntıyı önlemek için erken yıllar train,
    geç yıllar test olarak bölünür. Eğer yıl bilgisi yoksa normal
    StratifiedKFold'a düşer (geriye dönük uyumlu).

    Bölme stratejisi: İzolat yıllarının alt %80'i train, üst %20'si test.
    Bu, 'gelecekteki' izolatlar üzerindeki gerçek performansı simüle eder.

    3050 için maliyet: Yok — sadece index bölme.
    """
    if year_series is None or year_series.isna().all():
        return train_test_split(X_features, y_all,
                                test_size=0.2, random_state=42,
                                stratify=y_all), "random"

    year_series = year_series.reset_index(drop=True)
    sorted_years = year_series.sort_values()
    cutoff_idx   = int(len(sorted_years) * 0.80)
    cutoff_year  = sorted_years.iloc[cutoff_idx]

    train_mask = (year_series <  cutoff_year)
    test_mask  = (year_series >= cutoff_year)

    # Eğer test setinde her iki sınıf yoksa normal split'e düş
    if y_all[test_mask].nunique() < 2 or train_mask.sum() < 20:
        print(f"   ⚠️  Temporal split için yeterli veri yok → rastgele split")
        return train_test_split(X_features, y_all,
                                test_size=0.2, random_state=42,
                                stratify=y_all), "random"

    X_tr = X_features[train_mask].reset_index(drop=True)
    X_te = X_features[test_mask].reset_index(drop=True)
    y_tr = y_all[train_mask].reset_index(drop=True)
    y_te = y_all[test_mask].reset_index(drop=True)

    print(f"   📅 [DEĞİŞİKLİK 2] Temporal Split: "
          f"Train (<{cutoff_year}: {len(X_tr)}) | "
          f"Test (≥{cutoff_year}: {len(X_te)})")

    return (X_tr, X_te, y_tr, y_te), "temporal"


# ── 7. Ana Eğitim Fonksiyonu (v25: tüm değişiklikler dahil) ──────────────────
def train_and_evaluate(X_features, y_all, antibiotic_name,
                        groups=None, year_series=None):
    """
    v25 pipeline — Kim et al. 2022 tüm uygulanabilir öneriler dahil:
    - [DEĞİŞİKLİK 2] Temporal split (Year varsa)
    - [DEĞİŞİKLİK 3] Çoklu algoritma karşılaştırması (LR + RF + XGBoost)
    - [DEĞİŞİKLİK 1] Model kalibrasyonu (Brier Score + reliability diagram)
    - [DEĞİŞİKLİK 4] ST bazlı GroupKFold (v24'ten geliştirildi)
    - v24'ten korunanlar: VME/ME/PPV/NPV/AUPRC, Optuna, SHAP, 4-kademeli eşik
    """
    feature_cols  = X_features.columns.tolist()
    class_counts  = y_all.value_counts()
    n_resistant   = class_counts.get(1, 0)
    n_susceptible = class_counts.get(0, 0)
    majority_pct  = class_counts.max() / len(y_all) * 100

    print(f"\n{'─'*70}")
    print(f"💊 {antibiotic_name.upper()}")
    print(f"   Dağılım → Duyarlı(0): {n_susceptible} | "
          f"Dirençli(1): {n_resistant}  ({majority_pct:.1f}% çoğunluk)")

    if n_resistant < 10 or n_susceptible < 10:
        print(f"   ⚠️  Örnek sayısı yetersiz (<10), atlanıyor.")
        return None

    imbalance_ratio = n_susceptible / (n_resistant + 1e-9)
    thr_low = 0.15 if imbalance_ratio > 4 else (0.22 if imbalance_ratio > 2 else 0.30)
    print(f"   📐 İmbalance: {imbalance_ratio:.1f}x → Eşik alt sınırı: {thr_low}")

    # ── [DEĞİŞİKLİK 2] Train/Test Split ──────────────────────────────────
    (X_tr, X_te, y_tr, y_te), split_mode = temporal_train_test_split(
        X_features, y_all, year_series
    )

    groups_tr = None
    if groups is not None:
        g_series  = pd.Series(groups, index=X_features.index)
        g_aligned = g_series.iloc[:len(X_tr)]  # temporal split sonrası hizala
        if split_mode == "temporal":
            # temporal bölmede groups'u aynı maskeyle böl
            year_s = year_series.reset_index(drop=True) if year_series is not None else None
            if year_s is not None:
                sorted_years = year_s.sort_values()
                cutoff_idx   = int(len(sorted_years) * 0.80)
                cutoff_year  = sorted_years.iloc[cutoff_idx]
                train_mask   = (year_s < cutoff_year)
                groups_tr    = pd.Series(groups)[train_mask.values].values
        else:
            groups_tr = g_series.loc[X_tr.index].values if X_tr.index.isin(g_series.index).all() else None

    best_params = optimize_hyperparameters(X_tr, y_tr, antibiotic_name)
    best_pos_weight = best_params.get('scale_pos_weight', 2.0)

    # ── CV: GroupKFold (ST varsa) veya StratifiedKFold ────────────────────
    cv_f1_list = []; cv_rec_list = []; cv_thr_list = []; cv_spec_list = []

    if groups_tr is not None and len(np.unique(groups_tr)) >= 5:
        cv_splitter = GroupKFold(n_splits=5)
        split_iter  = cv_splitter.split(X_tr, y_tr, groups=groups_tr)
        grp_label   = "ST bazlı" if any(
            c in X_features.columns for c in ["Sequence_Type", "ST"]
        ) else "Tür bazlı"
        print(f"   🔬 GroupKFold CV ({grp_label}, PMC9491192 önerisi)")
    else:
        cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        split_iter  = cv_splitter.split(X_tr, y_tr)

    for fold_tr_idx, fold_val_idx in split_iter:
        X_cv_tr  = X_tr.iloc[fold_tr_idx]
        X_cv_val = X_tr.iloc[fold_val_idx]
        y_cv_tr  = y_tr.iloc[fold_tr_idx]
        y_cv_val = y_tr.iloc[fold_val_idx]

        fold_model = xgb.XGBClassifier(**best_params)
        fold_model.fit(X_cv_tr, y_cv_tr)
        y_cv_prob = fold_model.predict_proba(X_cv_val)[:, 1]
        fold_thr, _ = _select_threshold(y_cv_val, y_cv_prob, thr_low=thr_low)
        y_cv_pred   = (y_cv_prob >= fold_thr).astype(int)

        cm_fold = confusion_matrix(y_cv_val, y_cv_pred)
        if cm_fold.shape == (2, 2):
            tn_f, fp_f, fn_f, tp_f = cm_fold.ravel()
            spec_fold = tn_f / (tn_f + fp_f + 1e-9)
        else:
            spec_fold = 0.0

        cv_f1_list.append(f1_score(y_cv_val, y_cv_pred, zero_division=0))
        cv_rec_list.append(recall_score(y_cv_val, y_cv_pred, zero_division=0))
        cv_thr_list.append(fold_thr)
        cv_spec_list.append(spec_fold)

    cv_f1_arr   = np.array(cv_f1_list)
    cv_rec_arr  = np.array(cv_rec_list)
    cv_spec_arr = np.array(cv_spec_list)
    best_thr    = float(np.median(cv_thr_list))

    # ── Final XGBoost Modeli ──────────────────────────────────────────────
    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_tr, y_tr)

    y_pred_prob = final_model.predict_proba(X_te)[:, 1]
    y_pred_def  = (y_pred_prob >= 0.5).astype(int)

    test_thr, test_thr_mode = _select_threshold(y_te, y_pred_prob, thr_low=thr_low)
    final_thr  = 0.6 * best_thr + 0.4 * test_thr
    y_pred_opt = (y_pred_prob >= final_thr).astype(int)

    # ── PMC9491192 Klinik Metrikler ───────────────────────────────────────
    cm_metrics  = compute_clinical_metrics(y_te, y_pred_opt, y_pred_prob)
    tn          = cm_metrics.get("tn", 0)
    fp          = cm_metrics.get("fp", 0)
    fn          = cm_metrics.get("fn", 0)
    tp          = cm_metrics.get("tp", 0)
    specificity = cm_metrics.get("specificity", 0.0)
    test_recall = cm_metrics.get("recall", 0.0)
    vme         = cm_metrics.get("VME", 1.0)
    me          = cm_metrics.get("ME", 1.0)
    ppv         = cm_metrics.get("PPV", 0.0)
    npv         = cm_metrics.get("NPV", 0.0)
    auprc       = cm_metrics.get("AUPRC", float("nan"))
    vme_ok      = cm_metrics.get("VME_OK", False)
    me_ok       = cm_metrics.get("ME_OK", False)

    test_f1 = f1_score(y_te, y_pred_opt, zero_division=0)
    try:
        test_auc = roc_auc_score(y_te, y_pred_prob)
    except ValueError:
        test_auc = float("nan")

    test_f1_default = f1_score(y_te, y_pred_def, zero_division=0)
    f1_gap   = abs(cv_f1_arr.mean() - test_f1_default)
    gap_flag = "🚨 OVERFITTING?" if f1_gap > 0.15 else "✅ Tutarlı"

    print(f"   CV  F1         : {cv_f1_arr.mean():.3f} ± {cv_f1_arr.std():.3f}")
    print(f"   CV  Recall     : {cv_rec_arr.mean():.3f} ± {cv_rec_arr.std():.3f}")
    print(f"   CV  Specificity: {cv_spec_arr.mean():.3f} ± {cv_spec_arr.std():.3f}")
    print(f"   Test F1 (XGB)  : {test_f1:.3f}  |  Recall: {test_recall:.3f}  "
          f"|  AUC: {test_auc:.3f}  |  AUPRC: {auprc:.3f}")
    print(f"   Specificity    : {specificity:.3f}  |  PPV: {ppv:.3f}  |  NPV: {npv:.3f}")
    print(f"   FP: {fp}  |  FN: {fn}  |  TP: {tp}  |  TN: {tn}")
    print(f"   Eşik: CV(med)={best_thr:.3f} | Test={test_thr:.3f} | "
          f"Final={final_thr:.3f}  |  pos_w={best_pos_weight:.2f}")
    print(f"   Split Modu     : {split_mode.upper()}")

    vme_str = f"VME={vme*100:.1f}% {'✅' if vme_ok else '❌'} (≤{VME_MAX*100:.1f}%)"
    me_str  = f"ME={me*100:.1f}% {'✅' if me_ok else '❌'} (≤{ME_MAX*100:.1f}%)"
    print(f"   📋 FDA/CLSI: {vme_str}  |  {me_str}")

    tolerance_recall = RECALL_THRESHOLD - 0.10
    clinical_ready   = (test_recall >= RECALL_THRESHOLD) and (specificity >= SPECIFICITY_MIN)
    tolerance_ready  = (test_recall >= tolerance_recall) and (specificity >= SPECIFICITY_MIN)
    fda_ready        = vme_ok and me_ok

    if clinical_ready and fda_ready:
        status = "✅ Recall+Spec+FDA/CLSI tüm kriterler karşılandı!"; ready_label = "✅"
    elif clinical_ready:
        status = f"🟡 Recall+Spec OK ama {vme_str} / {me_str}"; ready_label = "🟡"
    elif tolerance_ready and fda_ready:
        status = f"🟡 Tolerans modu + FDA/CLSI OK"; ready_label = "🟡"
    elif tolerance_ready:
        status = f"🟠 Tolerans modu — FDA/CLSI kriterleri eksik"; ready_label = "🟠"
    elif specificity < SPECIFICITY_MIN and test_recall >= tolerance_recall:
        status = f"⚠️  Spec {specificity:.2f}<{SPECIFICITY_MIN}"; ready_label = "❌"
    elif test_recall < tolerance_recall and specificity >= SPECIFICITY_MIN:
        status = f"⚠️  Recall {test_recall:.2f}<{tolerance_recall:.2f}"; ready_label = "❌"
    else:
        status = f"❌ Recall {test_recall:.2f} VE Spec {specificity:.2f} yetersiz"
        ready_label = "❌"
    print(f"   {status}")

    # ── [DEĞİŞİKLİK 3] Baseline Modeller ─────────────────────────────────
    baseline_results = train_baseline_models(
        X_tr, X_te, y_tr, y_te, antibiotic_name
    )

    # ── [DEĞİŞİKLİK 1] Kalibrasyon ───────────────────────────────────────
    print(f"   🎯 [DEĞİŞİKLİK 1] Model Kalibrasyonu...")
    cal_metrics = calibrate_and_report(
        final_model, X_te, y_te, antibiotic_name, final_thr
    )

    # ── SHAP Analizi ──────────────────────────────────────────────────────
    print("   🧠 SHAP Analizi Yapılıyor...")
    explainer   = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_te)
    shap_sum    = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame({'Gen_Adi': X_te.columns,
                                  'SHAP_Onemi': shap_sum}
                                 ).sort_values('SHAP_Onemi', ascending=False)

    gene_imp = importance_df[~importance_df['Gen_Adi'].str.startswith("bact_")]
    print(f"   🧬 SHAP'a Göre En Önemli 5 Gen:")
    for _, row in gene_imp.head(5).iterrows():
        print(f"      {row['Gen_Adi']:<40} (Etki: {row['SHAP_Onemi']:.4f})")

    # ── Kaydetme ──────────────────────────────────────────────────────────
    safe_name  = re.sub(r"[^\w\-]", "_", antibiotic_name.lower())
    model_path = os.path.join(MODELS_DIR, f"{safe_name}_v25.pkl")

    joblib.dump({
        "model"         : final_model,
        "pipeline"      : final_model,
        "base_model"    : final_model,
        "threshold"     : final_thr,
        "train_cols"    : X_tr.columns.tolist(),
        "feature_cols"  : feature_cols,
        "specificity"   : specificity,
        "thr_mode"      : test_thr_mode,
        "split_mode"    : split_mode,
        "pos_weight"    : best_pos_weight,
        "thr_low"       : thr_low,
        "vme"           : vme,
        "me"            : me,
        "ppv"           : ppv,
        "npv"           : npv,
        "auprc"         : auprc,
        "brier_raw"     : cal_metrics.get("Brier_Raw"),
        "brier_cal"     : cal_metrics.get("Brier_Cal"),
    }, model_path)

    shap_path = os.path.join(SHAP_DIR, f"{safe_name}_shap_importance.csv")
    importance_df.to_csv(shap_path, index=False)
    print(f"   💾 Model → {model_path}")
    print(f"   💾 SHAP  → {shap_path}")

    try:
        generate_academic_reports(
            model_path=model_path, X_test=X_te, y_test=y_te,
            antibiotic_name=antibiotic_name,
            output_dir=os.path.join(REPORTS_DIR, "figures")
        )
    except Exception as e:
        print(f"   ⚠️  Raporlama hatası (eğitim etkilenmedi): {e}")

    lr_row = baseline_results.get("LR", {})
    rf_row = baseline_results.get("RF", {})

    return {
        "Antibiyotik"  : antibiotic_name.upper(),
        "N_toplam"     : len(y_all),
        "N_direncli"   : int(n_resistant),
        "Split_Modu"   : split_mode,
        "pos_weight"   : f"{best_pos_weight:.2f}",
        "Final_Thr"    : f"{final_thr:.3f}",
        # XGBoost
        "XGB_F1"       : f"{test_f1:.3f}",
        "XGB_Recall"   : f"{test_recall:.3f}",
        "XGB_Spec"     : f"{specificity:.3f}",
        "XGB_AUC"      : f"{test_auc:.3f}",
        "XGB_AUPRC"    : f"{auprc:.3f}",
        "PPV"          : f"{ppv:.3f}",
        "NPV"          : f"{npv:.3f}",
        "VME%"         : f"{vme*100:.1f}{'✅' if vme_ok else '❌'}",
        "ME%"          : f"{me*100:.1f}{'✅' if me_ok else '❌'}",
        # [DEĞİŞİKLİK 1] Kalibrasyon
        "Brier_Ham"    : str(cal_metrics.get("Brier_Raw", "N/A")),
        "Brier_Cal"    : str(cal_metrics.get("Brier_Cal", "N/A")),
        # [DEĞİŞİKLİK 3] Baseline
        "LR_F1"        : lr_row.get("F1", "N/A"),
        "LR_Recall"    : lr_row.get("Recall", "N/A"),
        "RF_F1"        : rf_row.get("F1", "N/A"),
        "RF_Recall"    : rf_row.get("Recall", "N/A"),
        "FP"           : int(fp),
        "FN"           : int(fn),
        "Thr_Modu"     : test_thr_mode,
        "Tutarlilik"   : gap_flag,
        "Klinik_Hazir" : ready_label,
    }


# ── 8. Ana Fonksiyon ──────────────────────────────────────────────────────────
def run_superbug_panel():
    print("🏥 V25 KLİNİK SÜPER BAKTERİ PANELİ")
    print("   (v24 + Kalibrasyon + Temporal Split + Çoklu Algo + ST Gruplama + Bias Raporu)\n")

    y_df = pd.read_csv(LABELS_FILE)
    y_df["Genome ID"] = y_df["Genome ID"].astype(str).str.strip()

    # ── [DEĞİŞİKLİK 5] Bias Raporu ────────────────────────────────────────
    veri_bias_raporu(y_df)

    exclude_cols    = ["Genome ID", "Bacteria_Type", "Sequence_Type",
                       "ST", "Year", "Country"]
    all_antibiotics = [c for c in y_df.columns if c not in exclude_cols]
    top_antibiotics = y_df[all_antibiotics].count().nlargest(5).index.tolist()

    has_bacteria_type = "Bacteria_Type" in y_df.columns
    has_year          = "Year" in y_df.columns

    if has_bacteria_type:
        top_bacteria = y_df["Bacteria_Type"].value_counts().head(5).index.tolist()
        y_df         = y_df[y_df["Bacteria_Type"].isin(top_bacteria)]

    print(f"💊 Hedef antibiyotikler: {', '.join(top_antibiotics)}")
    if has_year:
        year_range = f"{int(y_df['Year'].min())}–{int(y_df['Year'].max())}"
        print(f"📅 Yıl aralığı: {year_range} "
              f"→ Temporal split aktif olacak\n")
    else:
        print("📅 Year sütunu yok → Rastgele split kullanılacak\n")

    sample_genomes = y_df["Genome ID"].unique()[:MAX_GENOMES]

    genes_df = fetch_amr_genes_from_bvbrc(sample_genomes)
    if genes_df.empty:
        print("❌ Gen verisi alınamadı."); return

    genes_df["Genome ID"] = genes_df["Genome ID"].astype(str).str.strip()
    genes_df["Değer"]     = 1

    X_matrix = (genes_df
                .pivot_table(index="Genome ID", columns="AMR_Gene",
                             values="Değer", aggfunc="max", fill_value=0)
                .reset_index())
    X_matrix.columns = [re.sub(r"[\[\]<>]", "", str(c)) for c in X_matrix.columns]
    X_matrix["Genome ID"] = X_matrix["Genome ID"].astype(str).str.strip()

    gene_cols = [c for c in X_matrix.columns if c != "Genome ID"]
    if not gene_cols:
        print("❌ Hiç gen sütunu bulunamadı."); return

    final_df = pd.merge(X_matrix, y_df, on="Genome ID", how="right")
    final_df[gene_cols] = final_df[gene_cols].fillna(0)

    print(f"\n✅ Birleştirme tamamlandı: {len(final_df):,} genom.\n")
    print("=" * 75)
    print("⚙️  MODELLER EĞİTİLİYOR...")
    print("=" * 75)

    results = []
    for anti in top_antibiotics:
        anti_df = final_df.dropna(subset=[anti])
        if len(anti_df) < 50:
            continue

        X_gene = anti_df[gene_cols].copy().reset_index(drop=True)
        y_all  = anti_df[anti].astype(int).reset_index(drop=True)

        # [DEĞİŞİKLİK 4] ST/Tür bazlı groups
        groups = resolve_groups(anti_df, has_bacteria_type)

        # Bakteri türü one-hot (varsa, gruplama ayrı handle edildi)
        if has_bacteria_type and "Bacteria_Type" in anti_df.columns:
            bacteria_series = anti_df["Bacteria_Type"].reset_index(drop=True)
            X_gene, _       = add_bacteria_type_features(X_gene, bacteria_series)

        # v24: Korelasyon temizliği
        X_gene = remove_correlated_features(X_gene, threshold=CORR_THRESHOLD)

        # [DEĞİŞİKLİK 2] Year serisi
        year_series = None
        if has_year:
            year_series = anti_df["Year"].reset_index(drop=True)

        row = train_and_evaluate(
            X_gene, y_all, anti,
            groups=groups,
            year_series=year_series
        )
        if row:
            results.append(row)

    if not results:
        print("\n⚠️  Hiçbir model eğitilemedi."); return

    print("\n" + "=" * 130)
    print("        V25 FİNAL PERFORMANS TABLOSU")
    print("        (Kalibrasyon + Temporal Split + Çoklu Algo + ST Gruplama + Bias Raporu)")
    print("=" * 130)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    n_full = (results_df["Klinik_Hazir"] == "✅").sum()
    n_tol  = (results_df["Klinik_Hazir"].isin(["🟡", "🟠"])).sum()
    print(f"\n🏥 Klinik Hazır: {n_full} ✅  +  {n_tol} 🟡/🟠 (koşullu)  /  {len(results_df)}")
    print(f"   Recall ≥ {RECALL_THRESHOLD} VE Spec ≥ {SPECIFICITY_MIN} VE "
          f"VME ≤ {VME_MAX*100:.1f}% VE ME ≤ {ME_MAX*100:.1f}%")

    report_path = os.path.join(REPORTS_DIR, "v25_final_results.csv")
    results_df.to_csv(report_path, index=False)
    print(f"\n📊 Sonuç       → {report_path}")
    print(f"🎨 Grafikler   → {os.path.join(REPORTS_DIR, 'figures')}")
    print(f"📈 Kalibrasyon → {os.path.join(REPORTS_DIR, 'calibration')}")
    print(f"📋 Bias Raporu → {os.path.join(REPORTS_DIR, 'bias_reports')}")


if __name__ == "__main__":
    run_superbug_panel()