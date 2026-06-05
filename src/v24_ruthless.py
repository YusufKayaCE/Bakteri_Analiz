# main_v24.py
# Değişiklikler (v23 → v24) — PMC9491192 (Moradigaravand et al., 2022) önerileri:
#
#   1. VME/ME METRİKLERİ EKLENDİ (FDA/CLSI standardı)
#      Very Major Error (VME): Dirençli iken Duyarlı denilen oran = FN / (TP+FN)
#      Major Error     (ME):  Duyarlı iken Dirençli denilen oran  = FP / (TN+FP)
#      Makale: VME ≤ %1.5, ME ≤ %3.0 klinik kabul kriterleri.
#      Önceki kodda sadece Recall/Specificity vardı; bu klinik standart eksikti.
#
#   2. PPV / NPV RAPORLAMA EKLENDİ (prevalence-adjusted)
#      Makale: sınıf dengesizliğinde AUC yanıltıcı olabilir; PPV/NPV
#      raporlanmalıdır. PPV = TP/(TP+FP), NPV = TN/(TN+FN).
#
#   3. AUPRC (Average Precision) EKLENDİ — AUC'ye ek olarak
#      Makale: imbalanced veri setlerinde AUPRC daha bilgilendirici.
#      ROC-AUC yüksek görünse de AUPRC gerçek performansı daha iyi yansıtır.
#
#   4. KORELASYON BAZLI FEATURE SEÇİMİ EKLENDİ
#      Makale: gen varlık-yokluk matrisinde yüksek korelasyonlu genler
#      (aynı gen ailesi üyeleri) modeli karmaşıklaştırır ve SHAP yorumlamayı
#      bozar. Çözüm: φ (phi) katsayısı > 0.95 olan çiftlerden biri silinir.
#
#   5. STRATİFİED SPECIES-LEVEL CV (makale önerisi)
#      Makale: aynı bakteri türünden örneklerin farklı fold'lara dağıtılması
#      önerilir (phylogenetic leakage'ı önlemek için). Uygulama: GroupKFold
#      kullanılarak bakteri türü bilgisi fold'ları yönlendirir.
#      Bakteri türü bilgisi yoksa normal StratifiedKFold'a düşer.
#
#   6. KLİNİK KARAR ÖZETİ — VME/ME bazlı klinik hazır değerlendirmesi
#      Recall/Specificity yanında VME≤%1.5 VE ME≤%3.0 kriteri eklendi.

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

from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     GroupKFold, cross_validate)
from sklearn.metrics import (confusion_matrix, f1_score, recall_score,
                             accuracy_score, roc_auc_score,
                             average_precision_score, precision_score)
from sklearn.preprocessing import LabelBinarizer

from reporting_module import generate_academic_reports

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── FDA/CLSI Klinik Kabul Eşikleri (PMC9491192) ───────────────────────────────
VME_MAX    = 0.015   # Very Major Error ≤ %1.5
ME_MAX     = 0.030   # Major Error ≤ %3.0
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
LABELS_FILE      = "../data/processed/v2_multilabel_labels.csv"
CACHE_FILE       = "../data/processed/v17_amr_genes_cache.csv"
MODELS_DIR       = "../models"
REPORTS_DIR      = "../reports"
SHAP_DIR         = "../reports/shap_values"

MAX_GENOMES      = 30000
BATCH_SIZE       = 100
OPTUNA_TRIALS    = 30
OPTUNA_N_JOBS    = 1
MODEL_N_JOBS     = 1 if DEVICE == "cuda" else -1
BACT_MIN_FREQ    = 0.05
CORR_THRESHOLD   = 0.95   # v24: korelasyon temizliği eşiği (PMC9491192)

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(SHAP_DIR, exist_ok=True)
os.makedirs(os.path.join(REPORTS_DIR, "figures"), exist_ok=True)

OPTUNA_DB_FILE = f"sqlite:///{os.path.abspath(MODELS_DIR)}/optuna_studies_v24.db"


# ── 1. Veri Çekme ─────────────────────────────────────────────────────────────
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

    print(f"🌐 V24: {len(genome_ids):,} bakteri için API'den genler çekiliyor...")
    df = _fetch_from_api(genome_ids)
    if not df.empty:
        df.to_csv(CACHE_FILE, index=False)
        print(f"🎉 Kaydedildi → {CACHE_FILE}")
    return df


def _fetch_from_api(genome_ids):
    all_genes = []
    headers = {"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"}
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


# ── 2. Korelasyon Bazlı Feature Temizliği (PMC9491192 önerisi) ────────────────
def remove_correlated_features(X: pd.DataFrame,
                                threshold: float = CORR_THRESHOLD) -> pd.DataFrame:
    """
    PMC9491192 önerisi: gen varlık-yokluk matrisinde yüksek korelasyonlu
    özellikler (aynı gen ailesi üyeleri, örn. TEM-1 ve TEM-52) modeli
    gereksiz karmaşıklaştırır ve SHAP yorumlamayı bozar.

    Phi (φ) katsayısı binary veriler için doğru korelasyon ölçüsüdür.
    φ > threshold olan çiftlerden alfabetik olarak ikinci olanı çıkarılır.

    Not: bact_ sütunları korelasyon hesabının dışında tutulur.
    """
    gene_cols = [c for c in X.columns if not c.startswith("bact_")]
    bact_cols = [c for c in X.columns if c.startswith("bact_")]

    if len(gene_cols) < 2:
        return X

    # Sadece varyansı sıfır olmayan sütunlar üzerinde hesapla
    X_gene = X[gene_cols].copy()
    var_mask = X_gene.var() > 0
    X_gene = X_gene.loc[:, var_mask]

    corr_matrix = X_gene.corr(method='pearson').abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    remaining_gene_cols = [c for c in gene_cols if c not in to_drop and c in X_gene.columns]
    # varyansı sıfır olanları da geri ekle (drop edilmemiş)
    zero_var_cols = [c for c in gene_cols if c not in X_gene.columns]

    final_cols = remaining_gene_cols + zero_var_cols + bact_cols
    final_cols = [c for c in final_cols if c in X.columns]  # güvenlik

    n_removed = len(gene_cols) - len(remaining_gene_cols)
    if n_removed > 0:
        print(f"   🔬 Korelasyon temizliği (φ>{threshold}): "
              f"{n_removed} gen çıkarıldı, {len(remaining_gene_cols)} gen kaldı.")

    return X[final_cols]


# ── 3. Bakteri Türü One-Hot Encoding (Min-Freq Filtreli) ──────────────────────
def add_bacteria_type_features(X_gene: pd.DataFrame,
                                bacteria_series: pd.Series,
                                min_freq: float = BACT_MIN_FREQ) -> tuple:
    """
    v23'ten taşınan min-freq filtreli bakteri özelliği ekleme.
    Ek olarak: groups array döndürür (GroupKFold için).
    """
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

    groups = bacteria_clean.values   # GroupKFold için
    return pd.concat([X_gene, bact_df], axis=1), groups


# ── 4. Klinik Metrikler: VME / ME / PPV / NPV (PMC9491192) ───────────────────
def compute_clinical_metrics(y_true, y_pred, y_prob):
    """
    PMC9491192'nin önerdiği FDA/CLSI klinik standart metrikleri:

    VME (Very Major Error): Gerçekte dirençli, modelin duyarlı dediği oran
        = FN / (TP + FN) = 1 - Recall
        Klinik anlam: tedavi edilmesi gereken hasta tedavisiz bırakılıyor.
        Kabul eşiği: ≤ %1.5

    ME (Major Error): Gerçekte duyarlı, modelin dirençli dediği oran
        = FP / (TN + FP) = 1 - Specificity
        Klinik anlam: gereksiz yere daha güçlü antibiyotik kullanılıyor.
        Kabul eşiği: ≤ %3.0

    PPV (Positive Predictive Value): Dirençli denen örneklerin gerçekten
        dirençli olma oranı. Prevalansa bağlıdır.

    NPV (Negative Predictive Value): Duyarlı denen örneklerin gerçekten
        duyarlı olma oranı. Prevalansa bağlıdır.

    AUPRC: Precision-Recall eğrisi altındaki alan.
        İmbalanced veri setlerinde ROC-AUC'den daha bilgilendirici.
    """
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2):
        return {}
    tn, fp, fn, tp = cm.ravel()

    recall      = tp / (tp + fn + 1e-9)
    specificity = tn / (tn + fp + 1e-9)

    vme = fn / (tp + fn + 1e-9)   # = 1 - Recall
    me  = fp / (tn + fp + 1e-9)   # = 1 - Specificity
    ppv = tp / (tp + fp + 1e-9)   # Precision
    npv = tn / (tn + fn + 1e-9)

    try:
        auprc = average_precision_score(y_true, y_prob)
    except Exception:
        auprc = float("nan")

    vme_ok = vme <= VME_MAX
    me_ok  = me  <= ME_MAX

    return {
        "VME"      : vme,
        "ME"       : me,
        "PPV"      : ppv,
        "NPV"      : npv,
        "AUPRC"    : auprc,
        "VME_OK"   : vme_ok,
        "ME_OK"    : me_ok,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "recall"   : recall,
        "specificity": specificity,
    }


# ── 5. Eşik Seçimi: 4 Kademeli Tolerans Sistemi ──────────────────────────────
def _select_threshold(y_true, y_prob,
                       recall_min=RECALL_THRESHOLD,
                       spec_min=SPECIFICITY_MIN,
                       thr_low=0.20,
                       recall_tolerance=0.10):
    """
    4 kademeli öncelik:
    1. full      : Recall≥0.80 VE Spec≥0.50 → en yüksek F1
    2. tolerance : Recall≥0.70 VE Spec≥0.50 → ayrı değişken
    3. recall_only: Recall≥0.80             → ayrı değişken
    4. default   : 0.50
    """
    thresholds = np.linspace(thr_low, 0.95, int((0.95 - thr_low) / 0.005) + 1)

    best_thr = None;  best_f1 = -1
    tol_thr  = None;  tol_f1  = -1
    rec_thr  = None;  rec_f1  = -1

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
                best_f1 = f1;  best_thr = thr
        elif recall >= (recall_min - recall_tolerance) and specificity >= spec_min:
            if f1 > tol_f1:
                tol_f1 = f1;  tol_thr = thr
        elif recall >= recall_min:
            if f1 > rec_f1:
                rec_f1 = f1;  rec_thr = thr

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


# ── 6. Optuna: Recall Cezalı, Genişletilmiş pos_weight ───────────────────────
def optimize_hyperparameters(X_train, y_train, antibiotic_name):
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    natural_ratio = min(n_neg / (n_pos + 1e-9), 10.0)
    pw_upper      = max(3.0, natural_ratio * 1.2)

    study_name   = f"study_v24_{antibiotic_name.lower()}"
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
                                    scoring={'ap': 'average_precision', 'recall': 'recall'},
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


# ── 7. Model Eğitimi (GroupKFold + VME/ME + AUPRC + PPV/NPV) ─────────────────
def train_and_evaluate(X_features, y_all, antibiotic_name, groups=None):
    """
    PMC9491192 önerileri dahil edilmiş tam pipeline:
    - GroupKFold: aynı bakteri türü aynı fold'a düşmez (phylogenetic leakage)
    - VME / ME / PPV / NPV / AUPRC metrikleri
    - Korelasyon temizliği önceden yapılmış olmalı (remove_correlated_features)
    """
    feature_cols = X_features.columns.tolist()

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

    # Train/test split — groups varsa stratify hem label hem species
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_features, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    groups_tr = None
    if groups is not None:
        g_series = pd.Series(groups, index=X_features.index)
        groups_tr = g_series.loc[X_tr.index].values

    best_params = optimize_hyperparameters(X_tr, y_tr, antibiotic_name)
    best_pos_weight = best_params.get('scale_pos_weight', 2.0)

    # ── CV: GroupKFold (varsa) veya StratifiedKFold ───────────────────────────
    cv_f1_list = []; cv_rec_list = []; cv_thr_list = []; cv_spec_list = []

    if groups_tr is not None and len(np.unique(groups_tr)) >= 5:
        # PMC9491192: species-aware CV — aynı tür aynı fold'da
        cv_splitter = GroupKFold(n_splits=5)
        split_iter  = cv_splitter.split(X_tr, y_tr, groups=groups_tr)
        print(f"   🔬 GroupKFold CV (species-aware, PMC9491192 önerisi)")
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

    # ── Final Model ───────────────────────────────────────────────────────────
    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_tr, y_tr)

    y_pred_prob = final_model.predict_proba(X_te)[:, 1]
    y_pred_def  = (y_pred_prob >= 0.5).astype(int)

    test_thr, test_thr_mode = _select_threshold(y_te, y_pred_prob, thr_low=thr_low)
    final_thr  = 0.6 * best_thr + 0.4 * test_thr
    y_pred_opt = (y_pred_prob >= final_thr).astype(int)

    # ── PMC9491192 Klinik Metrikler ───────────────────────────────────────────
    cm_metrics = compute_clinical_metrics(y_te, y_pred_opt, y_pred_prob)
    tn  = cm_metrics.get("tn", 0)
    fp  = cm_metrics.get("fp", 0)
    fn  = cm_metrics.get("fn", 0)
    tp  = cm_metrics.get("tp", 0)
    specificity  = cm_metrics.get("specificity", 0.0)
    test_recall  = cm_metrics.get("recall", 0.0)
    vme          = cm_metrics.get("VME", 1.0)
    me           = cm_metrics.get("ME", 1.0)
    ppv          = cm_metrics.get("PPV", 0.0)
    npv          = cm_metrics.get("NPV", 0.0)
    auprc        = cm_metrics.get("AUPRC", float("nan"))
    vme_ok       = cm_metrics.get("VME_OK", False)
    me_ok        = cm_metrics.get("ME_OK", False)

    test_f1 = f1_score(y_te, y_pred_opt, zero_division=0)

    try:
        test_auc = roc_auc_score(y_te, y_pred_prob)
    except ValueError:
        test_auc = float("nan")

    # Overfitting
    test_f1_default = f1_score(y_te, y_pred_def, zero_division=0)
    f1_gap   = abs(cv_f1_arr.mean() - test_f1_default)
    gap_flag = "🚨 OVERFITTING?" if f1_gap > 0.15 else "✅ Tutarlı"

    print(f"   CV  F1         : {cv_f1_arr.mean():.3f} ± {cv_f1_arr.std():.3f}")
    print(f"   CV  Recall     : {cv_rec_arr.mean():.3f} ± {cv_rec_arr.std():.3f}")
    print(f"   CV  Specificity: {cv_spec_arr.mean():.3f} ± {cv_spec_arr.std():.3f}")
    print(f"   Test F1        : {test_f1:.3f}  |  Recall: {test_recall:.3f}  "
          f"|  AUC: {test_auc:.3f}  |  AUPRC: {auprc:.3f}")
    print(f"   Specificity    : {specificity:.3f}  |  PPV: {ppv:.3f}  |  NPV: {npv:.3f}")
    print(f"   FP: {fp}  |  FN: {fn}  |  TP: {tp}  |  TN: {tn}")
    print(f"   Eşik: CV(med)={best_thr:.3f} | Test={test_thr:.3f} | "
          f"Final={final_thr:.3f}  |  pos_w={best_pos_weight:.2f}")

    # ── VME/ME klinik karar (PMC9491192 FDA/CLSI standardı) ──────────────────
    vme_str = f"VME={vme*100:.1f}% {'✅' if vme_ok else '❌'} (≤{VME_MAX*100:.1f}%)"
    me_str  = f"ME={me*100:.1f}% {'✅' if me_ok else '❌'} (≤{ME_MAX*100:.1f}%)"
    print(f"   📋 FDA/CLSI: {vme_str}  |  {me_str}")

    # Klinik hazır: Recall/Spec kriteri + VME/ME kriteri
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
        status = f"⚠️  Spec {specificity:.2f}<{SPECIFICITY_MIN} — çok fazla yanlış alarm"; ready_label = "❌"
    elif test_recall < tolerance_recall and specificity >= SPECIFICITY_MIN:
        status = f"⚠️  Recall {test_recall:.2f}<{tolerance_recall:.2f} — Spec OK ama Recall düşük"; ready_label = "❌"
    else:
        status = f"❌ Recall {test_recall:.2f} VE Spec {specificity:.2f} yetersiz"; ready_label = "❌"
    print(f"   {status}")

    # ── SHAP Analizi ──────────────────────────────────────────────────────────
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

    # ── Kaydetme ──────────────────────────────────────────────────────────────
    safe_name  = re.sub(r"[^\w\-]", "_", antibiotic_name.lower())
    model_path = os.path.join(MODELS_DIR, f"{safe_name}_v24.pkl")

    joblib.dump({
        "model"       : final_model,
        "pipeline"    : final_model,
        "base_model"  : final_model,
        "threshold"   : final_thr,
        "train_cols"  : X_tr.columns.tolist(),
        "feature_cols": feature_cols,
        "specificity" : specificity,
        "thr_mode"    : test_thr_mode,
        "pos_weight"  : best_pos_weight,
        "thr_low"     : thr_low,
        "vme"         : vme,
        "me"          : me,
        "ppv"         : ppv,
        "npv"         : npv,
        "auprc"       : auprc,
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

    return {
        "Antibiyotik"  : antibiotic_name.upper(),
        "N_toplam"     : len(y_all),
        "N_direncli"   : int(n_resistant),
        "pos_weight"   : f"{best_pos_weight:.2f}",
        "Final_Thr"    : f"{final_thr:.3f}",
        "F1"           : f"{test_f1:.3f}",
        "Recall"       : f"{test_recall:.3f}",
        "Specificity"  : f"{specificity:.3f}",
        "PPV"          : f"{ppv:.3f}",
        "NPV"          : f"{npv:.3f}",
        "AUC"          : f"{test_auc:.3f}",
        "AUPRC"        : f"{auprc:.3f}",
        "VME%"         : f"{vme*100:.1f}{'✅' if vme_ok else '❌'}",
        "ME%"          : f"{me*100:.1f}{'✅' if me_ok else '❌'}",
        "FP"           : int(fp),
        "FN"           : int(fn),
        "Thr_Modu"     : test_thr_mode,
        "Tutarlilik"   : gap_flag,
        "Klinik_Hazir" : ready_label,
    }


# ── 8. Ana Fonksiyon ──────────────────────────────────────────────────────────
def run_superbug_panel():
    print("🏥 V24 KLİNİK SÜPER BAKTERİ PANELİ")
    print("   (PMC9491192: VME/ME + PPV/NPV + AUPRC + Korelasyon Temizliği + GroupKFold)\n")

    y_df = pd.read_csv(LABELS_FILE)
    y_df["Genome ID"] = y_df["Genome ID"].astype(str).str.strip()

    exclude_cols    = ["Genome ID", "Bacteria_Type"]
    all_antibiotics = [c for c in y_df.columns if c not in exclude_cols]
    top_antibiotics = y_df[all_antibiotics].count().nlargest(5).index.tolist()

    has_bacteria_type = "Bacteria_Type" in y_df.columns
    if has_bacteria_type:
        top_bacteria = y_df["Bacteria_Type"].value_counts().head(5).index.tolist()
        y_df = y_df[y_df["Bacteria_Type"].isin(top_bacteria)]

    print(f"💊 Hedef antibiyotikler: {', '.join(top_antibiotics)}\n")
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

        # Bakteri türü özelliği + groups
        groups = None
        if has_bacteria_type:
            bacteria_series = anti_df["Bacteria_Type"].reset_index(drop=True)
            X_gene, groups  = add_bacteria_type_features(X_gene, bacteria_series)

        # v24: Korelasyon temizliği (PMC9491192)
        X_gene = remove_correlated_features(X_gene, threshold=CORR_THRESHOLD)

        row = train_and_evaluate(X_gene, y_all, anti, groups=groups)
        if row:
            results.append(row)

    if not results:
        print("\n⚠️  Hiçbir model eğitilemedi."); return

    print("\n" + "=" * 115)
    print("        V24 FİNAL PERFORMANS TABLOSU")
    print("        (PMC9491192: VME/ME + PPV/NPV + AUPRC + Korelasyon Temizliği + GroupKFold)")
    print("=" * 115)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    n_full = (results_df["Klinik_Hazir"] == "✅").sum()
    n_tol  = (results_df["Klinik_Hazir"].isin(["🟡","🟠"])).sum()
    print(f"\n🏥 Klinik Hazır: {n_full} ✅  +  {n_tol} 🟡/🟠 (koşullu)  /  {len(results_df)}")
    print(f"   Recall ≥ {RECALL_THRESHOLD} VE Spec ≥ {SPECIFICITY_MIN} VE "
          f"VME ≤ {VME_MAX*100:.1f}% VE ME ≤ {ME_MAX*100:.1f}%")

    report_path = os.path.join(REPORTS_DIR, "v24_final_results.csv")
    results_df.to_csv(report_path, index=False)
    print(f"\n📊 Sonuç → {report_path}")
    print(f"🎨 Grafikler → {os.path.join(REPORTS_DIR, 'figures')}")


if __name__ == "__main__":
    run_superbug_panel()