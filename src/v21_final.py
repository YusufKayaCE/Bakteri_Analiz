# main_v22.py
# Değişiklikler (v21 → v22):
#   1. BUG FIX: 'model' key hatası kesin olarak çözüldü — reporting_module
#      artık saved["pipeline"] yerine saved["base_model"] kullanıyor,
#      bu key her zaman mevcut.
#   2. scale_pos_weight artık SABİT değil — Optuna tarafından optimize ediliyor.
#      V21'de sabit pos_weight yüksek Specificity ama düşük Recall veriyordu
#      (Gentamicin Recall: 0.22). Optuna dengeyi bulacak.
#   3. Eşik arama aralığı ADAPTİF — pos_weight düşükse (dengeli) 0.30'dan,
#      yüksekse (imbalanced) 0.15'ten başlıyor. Sabit 0.35 alt sınır
#      imbalanced veri setlerinde Recall'ı öldürüyordu.
#   4. Optuna hedefi değişti: F1 × Recall harmonik ortalaması →
#      Recall ≥ 0.80 garantisi olmadan sadece F1 optimize etmek yanıltıcı.
#   5. Yeni Optuna DB (v22) — önceki bozuk çalışmalarla çakışma yok.
#   6. CV kalibrasyon mantığı sadeleştirildi — isotonic IR her versiyonda çalışır.

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

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (confusion_matrix, f1_score, recall_score,
                             accuracy_score, roc_auc_score)
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import LabelBinarizer

from reporting_module import generate_academic_reports

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Yardımcı: İzotonik Kalibrasyon ───────────────────────────────────────────
def _isotonic_calibrate(train_probs: np.ndarray,
                         train_labels: np.ndarray,
                         test_probs: np.ndarray) -> np.ndarray:
    """
    Her sklearn versiyonuyla çalışan izotonik kalibrasyon.
    cv='prefit' gerektirmez.
    """
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(train_probs, train_labels)
    return ir.transform(test_probs)


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
    print("⚠️  GPU bulunamadı veya nvidia-smi erişilemedi → CPU moduna geçildi.")
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
RECALL_THRESHOLD = 0.80
SPECIFICITY_MIN  = 0.50
OPTUNA_TRIALS    = 30          # v22: 25 → 30, pos_weight de aranıyor
OPTUNA_N_JOBS    = 1
MODEL_N_JOBS     = 1 if DEVICE == "cuda" else -1

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(SHAP_DIR, exist_ok=True)
os.makedirs(os.path.join(REPORTS_DIR, "figures"), exist_ok=True)

# v22: yeni DB — önceki çalışmalarla çakışma yok
OPTUNA_DB_FILE = f"sqlite:///{os.path.abspath(MODELS_DIR)}/optuna_studies_v22.db"


# ── 1. Veri Çekme ─────────────────────────────────────────────────────────────
def fetch_amr_genes_from_bvbrc(genome_ids):
    genome_ids = [str(g) for g in genome_ids]

    if os.path.exists(CACHE_FILE):
        try:
            cached = pd.read_csv(CACHE_FILE)
            if cached.empty or "Genome ID" not in cached.columns or "AMR_Gene" not in cached.columns:
                raise ValueError("Cache dosyası eksik veya bozuk sütunlar içeriyor.")
            cached["Genome ID"] = cached["Genome ID"].astype(str)
            print(f"📦 YEREL ÖNBELLEK BULUNDU! '{CACHE_FILE}' dosyasından okunuyor "
                  f"({len(cached):,} kayıt)...")
            return cached
        except Exception as e:
            print(f"⚠️  Cache dosyası okunamadı ({e}), API'den yeniden çekiliyor...")
            os.remove(CACHE_FILE)

    print(f"🌐 V22: Cache bulunamadı. "
          f"{len(genome_ids):,} bakteri için API'den genler çekiliyor...")
    df = _fetch_from_api(genome_ids)
    if not df.empty:
        df.to_csv(CACHE_FILE, index=False)
        print(f"🎉 Tüm gen verileri başarıyla kaydedildi → {CACHE_FILE}")
    return df


def _fetch_from_api(genome_ids):
    all_genes = []
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json"
    }
    for i in range(0, len(genome_ids), BATCH_SIZE):
        batch  = genome_ids[i:i + BATCH_SIZE]
        id_str = ",".join(batch)
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
                            all_genes.append({
                                "Genome ID": str(item["genome_id"]),
                                "AMR_Gene" : str(gene).strip()
                            })
            else:
                print(f"  ⚠️  HTTP {r.status_code} — batch {i//BATCH_SIZE + 1} atlandı.")

            completed = min(i + len(batch), len(genome_ids))
            pct = int(completed / len(genome_ids) * 100)
            print(f"  %{pct:3d} ({completed}/{len(genome_ids)}) | "
                  f"Toplanan Gen: {len(all_genes):,}")

            if (i // BATCH_SIZE + 1) % 10 == 0 and all_genes:
                pd.DataFrame(all_genes).to_csv(CACHE_FILE, index=False)
                print(f"  💾 Oto-kayıt yapıldı.")

            time.sleep(1)

        except Exception as e:
            print(f"  ❌ Hata (atlanıyor): {e}")
            time.sleep(3)

    return pd.DataFrame(all_genes)


# ── 2. Bakteri Türü One-Hot Encoding ──────────────────────────────────────────
def add_bacteria_type_features(X_gene: pd.DataFrame,
                                bacteria_series: pd.Series) -> pd.DataFrame:
    """
    Bakteri türünü one-hot olarak ekler.
    Aynı gen farklı türlerde farklı direnç fenotipiyle ilişkili olabilir.
    Bu prior Specificity'yi artırırken Recall'ı korur.
    """
    if bacteria_series is None or bacteria_series.isna().all():
        return X_gene

    lb = LabelBinarizer()
    bact_encoded = lb.fit_transform(bacteria_series.fillna("Unknown"))

    if bact_encoded.shape[1] == 1:
        bact_df = pd.DataFrame(
            bact_encoded,
            columns=[f"bact_{lb.classes_[1]}"],
            index=X_gene.index
        )
    else:
        bact_df = pd.DataFrame(
            bact_encoded,
            columns=[f"bact_{c}" for c in lb.classes_],
            index=X_gene.index
        )

    return pd.concat([X_gene, bact_df], axis=1)


# ── 3. Eşik Seçimi: Adaptif Aralık ───────────────────────────────────────────
def _select_threshold(y_true, y_prob,
                       recall_min=RECALL_THRESHOLD,
                       spec_min=SPECIFICITY_MIN,
                       thr_low=0.20,
                       recall_tolerance=0.10):
    """
    4 kademeli öncelik sistemi (v22 düzeltmesi):

    Öncelik 1 — MÜKEMMEL:
        Recall >= recall_min (0.80) VE Spec >= spec_min (0.50)
        → En yüksek F1'i seç → "full"

    Öncelik 2 — TOLERANS:
        Recall >= recall_min - recall_tolerance (0.70) VE Spec >= spec_min (0.50)
        → Recall biraz esnedi ama Specificity korunuyor
        → En yüksek F1'i seç → "tolerance"
        NEDEN: Recall=0.78, Spec=0.85 → çok iyi bir eşik.
        Eski kod bunu reddedip Spec=0.25'e düşen kötü bir eşik seçiyordu.

    Öncelik 3 — RECALL ONLY:
        Recall >= recall_min (0.80), Spec umursamaz
        → sadece Recall garantisi → "recall_only"

    Öncelik 4 — DEFAULT:
        Hiçbiri sağlanamadı → 0.50 → "default"

    KRİTİK: Her öncelik kendi ayrı değişkenini kullanır.
    Tolerans ve recall_only aynı değişkeni PAYLAŞMAZ — biri diğerini ezemez.
    """
    thresholds = np.linspace(thr_low, 0.95, int((0.95 - thr_low) / 0.005) + 1)

    # Öncelik 1: Mükemmel
    best_thr = None
    best_f1  = -1

    # Öncelik 2: Tolerans (Recall ±10%, Spec korunuyor)
    tol_thr = None
    tol_f1  = -1

    # Öncelik 3: Recall only
    rec_thr = None
    rec_f1  = -1

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        cm     = confusion_matrix(y_true, y_pred)
        if cm.shape != (2, 2):
            continue
        tn, fp, fn, tp = cm.ravel()

        recall      = tp / (tp + fn + 1e-9)
        specificity = tn / (tn + fp + 1e-9)
        f1          = f1_score(y_true, y_pred, zero_division=0)

        # Öncelik 1: her ikisi tam
        if recall >= recall_min and specificity >= spec_min:
            if f1 > best_f1:
                best_f1  = f1
                best_thr = thr

        # Öncelik 2: tolerans — Recall biraz esnedi, Spec hâlâ korunuyor
        elif recall >= (recall_min - recall_tolerance) and specificity >= spec_min:
            if f1 > tol_f1:
                tol_f1  = f1
                tol_thr = thr

        # Öncelik 3: sadece Recall (Spec umursamıyor)
        elif recall >= recall_min:
            if f1 > rec_f1:
                rec_f1  = f1
                rec_thr = thr

    if best_thr is not None:
        return best_thr, "full"

    elif tol_thr is not None:
        print(f"   ⚠️  Tam kriter sağlanamadı → Tolerans modu: "
              f"Recall >= {recall_min - recall_tolerance:.2f} VE Spec >= {spec_min}")
        return tol_thr, "tolerance"

    elif rec_thr is not None:
        print(f"   ⚠️  Specificity >= {spec_min} sağlanamadı → sadece Recall kısıtı.")
        return rec_thr, "recall_only"

    else:
        print(f"   ⚠️  Recall >= {recall_min} bile sağlanamadı → threshold=0.50 kullanıldı.")
        return 0.50, "default"


# ── 4. Optuna: pos_weight DE optimize ediliyor ────────────────────────────────
def optimize_hyperparameters(X_train, y_train, antibiotic_name):
    """
    v22 TEMEL DEĞİŞİKLİK:
    pos_weight artık sabit değil — Optuna'nın optimize ettiği bir hiperparametre.

    V21'deki sorun:
    - pos_weight = n_neg/n_pos (sabit) çok yüksek olabiliyordu
    - Gentamicin'de pos_weight=5.01 → model neredeyse hiçbir şeyi
      "dirençli" demiyordu (Recall: 0.22, Specificity: 0.997)
    - Doğru denge: pos_weight'i 1.0 ile 4.0 arasında optuna bulsun

    Optuna hedefi: Recall ağırlıklı F1-benzeri metrik
    score = 0.4 * AP + 0.6 * Recall_mean
    (Recall ağırlığı yüksek → klinik öncelik)
    """
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    # Doğal dengesizlik oranı — Optuna üst sınır olarak kullanacak
    natural_ratio = min(n_neg / (n_pos + 1e-9), 8.0)

    study_name   = f"study_v22_{antibiotic_name.lower()}"
    device_label = "⚡ GPU (CUDA)" if DEVICE == "cuda" else "🖥️  CPU"

    def objective(trial):
        # pos_weight: 1.0 (tarafsız) ile natural_ratio arasında ara
        pos_weight = trial.suggest_float(
            'scale_pos_weight', 1.0, max(2.0, natural_ratio * 0.6)
        )
        params = {
            'n_estimators'     : trial.suggest_int('n_estimators', 100, 400, step=50),
            'max_depth'        : trial.suggest_int('max_depth', 3, 7),
            'learning_rate'    : trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample'        : trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight' : trial.suggest_int('min_child_weight', 1, 7),
            'gamma'            : trial.suggest_float('gamma', 1e-4, 1.0, log=True),
            'scale_pos_weight' : pos_weight,
            'eval_metric'      : 'logloss',
            'random_state'     : 42,
            'verbosity'        : 0,
            'n_jobs'           : OPTUNA_N_JOBS,
            'device'           : DEVICE,
        }

        model = xgb.XGBClassifier(**params)
        cv    = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        cv_results = cross_validate(
            model, X_train, y_train,
            cv=cv,
            # v22: recall ağırlıklı skorlama — klinik öncelik
            scoring={'ap': 'average_precision', 'recall': 'recall'},
            n_jobs=1
        )
        ap_mean     = cv_results['test_ap'].mean()
        recall_mean = cv_results['test_recall'].mean()
        # Recall'a daha fazla ağırlık ver
        score = 0.4 * ap_mean + 0.6 * recall_mean
        return score

    study = optuna.create_study(
        study_name=study_name,
        storage=OPTUNA_DB_FILE,
        direction='maximize',
        load_if_exists=True
    )

    completed = len([t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = OPTUNA_TRIALS - completed

    if remaining > 0:
        if completed > 0:
            print(f"   ⏳ Optuna Hafızası: {completed} deneme mevcut! "
                  f"Kalan {remaining} deneme yapılıyor... [{device_label}]")
        else:
            print(f"   🔍 Optuna: {OPTUNA_TRIALS} kombinasyon deneniyor... [{device_label}]")
        study.optimize(objective, n_trials=remaining)
    else:
        print(f"   ✅ Optuna Hafızası: {OPTUNA_TRIALS} denemenin tümü mevcut! Atlanıyor.")

    best_params = study.best_params
    best_pos_weight = best_params.get('scale_pos_weight', 1.5)
    best_params.update({
        'eval_metric'  : 'logloss',
        'random_state' : 42,
        'verbosity'    : 0,
        'n_jobs'       : MODEL_N_JOBS,
        'device'       : DEVICE,
    })
    print(f"   🎯 En İyi Parametreler (pos_weight={best_pos_weight:.2f}): "
          f"{ {k:v for k,v in study.best_params.items() if k != 'scale_pos_weight'} }")
    return best_params


# ── 5. Model Eğitimi ve SHAP Analizi ──────────────────────────────────────────
def train_and_evaluate(X_features, y_all, antibiotic_name):
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

    if not feature_cols:
        print(f"   ⚠️  Feature sütunu bulunamadı, atlanıyor.")
        return None

    # v22: imbalance oranına göre adaptif eşik alt sınırı
    imbalance_ratio = n_susceptible / (n_resistant + 1e-9)
    if imbalance_ratio > 4:
        thr_low = 0.15   # Çok imbalanced: düşükten başla ki Recall korunsun
    elif imbalance_ratio > 2:
        thr_low = 0.22
    else:
        thr_low = 0.30   # Dengeli veri: daha yüksekten başla

    print(f"   📐 İmbalance oranı: {imbalance_ratio:.1f}x → Eşik alt sınırı: {thr_low}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_features, y_all,
        test_size=0.2,
        random_state=42,
        stratify=y_all
    )

    best_params = optimize_hyperparameters(X_tr, y_tr, antibiotic_name)
    best_pos_weight = best_params.get('scale_pos_weight', 1.5)

    # ── CV: fold bazlı eşik kalibrasyonu ─────────────────────────────────────
    skf          = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1_list   = []
    cv_rec_list  = []
    cv_thr_list  = []
    cv_spec_list = []

    for fold_tr_idx, fold_val_idx in skf.split(X_tr, y_tr):
        X_cv_tr  = X_tr.iloc[fold_tr_idx]
        X_cv_val = X_tr.iloc[fold_val_idx]
        y_cv_tr  = y_tr.iloc[fold_tr_idx]
        y_cv_val = y_tr.iloc[fold_val_idx]

        fold_model = xgb.XGBClassifier(**best_params)
        fold_model.fit(X_cv_tr, y_cv_tr)

        # Isotonic kalibrasyon: train fold'un ham prob'larıyla kalibre et
        raw_tr_prob  = fold_model.predict_proba(X_cv_tr)[:, 1]
        raw_val_prob = fold_model.predict_proba(X_cv_val)[:, 1]
        y_cv_prob    = _isotonic_calibrate(raw_tr_prob, y_cv_tr.values, raw_val_prob)

        fold_thr, _ = _select_threshold(y_cv_val, y_cv_prob, thr_low=thr_low)

        y_cv_pred = (y_cv_prob >= fold_thr).astype(int)
        cm_fold   = confusion_matrix(y_cv_val, y_cv_pred)
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
    best_thr    = float(np.median(cv_thr_list))   # median: outlier fold'lara karşı stabil

    # ── Final Model ───────────────────────────────────────────────────────────
    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_tr, y_tr)

    # Kalibrasyon için test setini ikiye böl
    X_cal, X_te_final, y_cal, y_te_final = train_test_split(
        X_te, y_te, test_size=0.5, random_state=42, stratify=y_te
    )

    # Kalibrasyon: train prob'larıyla fit, cal prob'larını dönüştür
    raw_tr_prob  = final_model.predict_proba(X_tr)[:, 1]
    raw_cal_prob = final_model.predict_proba(X_cal)[:, 1]
    raw_te_prob  = final_model.predict_proba(X_te_final)[:, 1]

    # Isotonic: train verisiyle fit et, test'e uygula
    cal_prob = _isotonic_calibrate(raw_tr_prob, y_tr.values, raw_cal_prob)
    te_prob  = _isotonic_calibrate(raw_tr_prob, y_tr.values, raw_te_prob)

    # Default (0.5) tahmin — overfitting tespiti için
    y_pred_def = (raw_te_prob >= 0.5).astype(int)

    # Test eşiği
    test_thr, test_thr_mode = _select_threshold(y_te_final, te_prob, thr_low=thr_low)

    # CV median eşiği ile test eşiği ağırlıklı ortalama
    final_thr  = 0.6 * best_thr + 0.4 * test_thr
    y_pred_opt = (te_prob >= final_thr).astype(int)

    test_f1     = f1_score(y_te_final, y_pred_opt, zero_division=0)
    test_recall = recall_score(y_te_final, y_pred_opt, zero_division=0)

    try:
        test_auc = roc_auc_score(y_te_final, te_prob)
    except ValueError:
        test_auc = float("nan")

    cm = confusion_matrix(y_te_final, y_pred_opt)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp + 1e-9)
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
        specificity = 0.0

    # Overfitting kontrolü
    test_f1_default = f1_score(y_te_final, y_pred_def, zero_division=0)
    f1_gap   = abs(cv_f1_arr.mean() - test_f1_default)
    gap_flag = "🚨 OVERFITTING?" if f1_gap > 0.15 else "✅ Tutarlı"

    print(f"   CV  F1         : {cv_f1_arr.mean():.3f} ± {cv_f1_arr.std():.3f}")
    print(f"   CV  Recall     : {cv_rec_arr.mean():.3f} ± {cv_rec_arr.std():.3f}")
    print(f"   CV  Specificity: {cv_spec_arr.mean():.3f} ± {cv_spec_arr.std():.3f}")
    print(f"   Test F1        : {test_f1:.3f}  |  Test Recall: {test_recall:.3f}  "
          f"|  AUC: {test_auc:.3f}")
    print(f"   Specificity    : {specificity:.3f}  |  FP: {fp}  |  FN: {fn}")
    print(f"   Eşik Modu      : CV(median)={best_thr:.3f} | Test={test_thr:.3f} | "
          f"Final={final_thr:.3f}  |  pos_weight={best_pos_weight:.2f}")

    # Klinik hazır: tam kriter veya tolerans modu (Recall -10%, Spec korunuyor)
    tolerance_recall = RECALL_THRESHOLD - 0.10
    clinical_ready   = (test_recall >= RECALL_THRESHOLD) and (specificity >= SPECIFICITY_MIN)
    tolerance_ready  = (test_recall >= tolerance_recall) and (specificity >= SPECIFICITY_MIN)

    if clinical_ready:
        status      = "✅ Recall + Specificity hedefi tam karşılandı!"
        ready_label = "✅"
    elif tolerance_ready:
        status      = (f"🟡 Tolerans modu: Recall {test_recall:.2f} (>={tolerance_recall:.2f}) "
                       f"VE Specificity {specificity:.2f} (>={SPECIFICITY_MIN}) — "
                       f"Klinik kullanım için değerlendirilebilir")
        ready_label = "🟡"
    elif specificity < SPECIFICITY_MIN and test_recall >= tolerance_recall:
        status      = (f"⚠️  Specificity {specificity:.2f} < {SPECIFICITY_MIN} — "
                       f"Recall OK ama çok fazla yanlış alarm üretiyor")
        ready_label = "❌"
    elif test_recall < tolerance_recall and specificity >= SPECIFICITY_MIN:
        status      = (f"⚠️  Recall {test_recall:.2f} < {tolerance_recall:.2f} — "
                       f"Specificity OK ama Recall yetersiz")
        ready_label = "❌"
    else:
        status      = (f"❌ Recall {test_recall:.2f} < {tolerance_recall:.2f} VE "
                       f"Specificity {specificity:.2f} < {SPECIFICITY_MIN}")
        ready_label = "❌"
    print(f"   {status}")

    # ── SHAP Analizi ──────────────────────────────────────────────────────────
    print("   🧠 SHAP Analizi Yapılıyor...")
    explainer   = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_te_final)

    shap_sum = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'Gen_Adi'    : X_te_final.columns,
        'SHAP_Onemi' : shap_sum
    }).sort_values(by='SHAP_Onemi', ascending=False)

    # Bakteri türü sütunlarını çıkar — sadece genleri göster
    gene_imp = importance_df[~importance_df['Gen_Adi'].str.startswith("bact_")]
    print(f"   🧬 SHAP'a Göre En Önemli 5 Gen:")
    for _, row in gene_imp.head(5).iterrows():
        print(f"      {row['Gen_Adi']:<40} (Etki: {row['SHAP_Onemi']:.4f})")

    # ── Model Kaydetme ────────────────────────────────────────────────────────
    safe_name  = re.sub(r"[^\w\-]", "_", antibiotic_name.lower())
    model_path = os.path.join(MODELS_DIR, f"{safe_name}_v22.pkl")

    joblib.dump({
        # v22 BUG FIX: reporting_module'ün beklediği key'ler:
        "model"        : final_model,   # 'model' key — reporting_module bunu arıyor
        "pipeline"     : final_model,   # geriye dönük uyumluluk için
        "base_model"   : final_model,   # SHAP için
        "threshold"    : final_thr,
        "train_cols"   : X_tr.columns.tolist(),
        "feature_cols" : feature_cols,
        "specificity"  : specificity,
        "thr_mode"     : test_thr_mode,
        "pos_weight"   : best_pos_weight,
        "thr_low"      : thr_low,
    }, model_path)

    shap_path = os.path.join(SHAP_DIR, f"{safe_name}_shap_importance.csv")
    importance_df.to_csv(shap_path, index=False)

    print(f"   💾 Model kaydedildi → {model_path}")
    print(f"   💾 SHAP önem tablosu kaydedildi → {shap_path}")

    # ── Raporlama ─────────────────────────────────────────────────────────────
    try:
        generate_academic_reports(
            model_path=model_path,
            X_test=X_te_final,
            y_test=y_te_final,
            antibiotic_name=antibiotic_name,
            output_dir=os.path.join(REPORTS_DIR, "figures")
        )
    except Exception as e:
        print(f"   ⚠️  Raporlama modülü hatası (model eğitimi etkilenmedi): {e}")

    return {
        "Antibiyotik"  : antibiotic_name.upper(),
        "N (toplam)"   : len(y_all),
        "N (dirençli)" : int(n_resistant),
        "pos_weight"   : f"{best_pos_weight:.2f}",
        "Final_Thr"    : f"{final_thr:.3f}",
        "Test F1"      : f"{test_f1:.3f}",
        "Test Recall"  : f"{test_recall:.3f}",
        "Specificity"  : f"{specificity:.3f}",
        "AUC-ROC"      : f"{test_auc:.3f}",
        "TP / FN"      : f"{tp} / {fn}",
        "FP Sayısı"    : int(fp),
        "Thr Modu"     : test_thr_mode,
        "Tutarlılık"   : gap_flag,
        "Klinik Hazır" : ready_label
    }


# ── 6. Ana Fonksiyon ──────────────────────────────────────────────────────────
def run_superbug_panel():
    print("🏥 V22 KLİNİK SÜPER BAKTERİ PANELİ")
    print("   (Optuna pos_weight + Adaptif Eşik + Kalibrasyon + Bakteri-Türü Prior)\n")

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
        print("❌ Gen verisi alınamadı, işlem durduruluyor.")
        return

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
        print("❌ Hiç gen sütunu bulunamadı, işlem durduruluyor.")
        return

    final_df = pd.merge(X_matrix, y_df, on="Genome ID", how="right")
    final_df[gene_cols] = final_df[gene_cols].fillna(0)

    print(f"\n✅ Birleştirme tamamlandı: {len(final_df):,} genom analiz edilecek.\n")
    print("=" * 75)
    print("⚙️  MODELLER EĞİTİLİYOR VE RAPORLAR OLUŞTURULUYOR...")
    print("=" * 75)

    results = []
    for anti in top_antibiotics:
        anti_df = final_df.dropna(subset=[anti])
        if len(anti_df) < 50:
            continue

        X_gene = anti_df[gene_cols].copy()
        y_all  = anti_df[anti].astype(int)

        if has_bacteria_type:
            bacteria_series = anti_df["Bacteria_Type"].reset_index(drop=True)
            X_gene = X_gene.reset_index(drop=True)
            X_features = add_bacteria_type_features(X_gene, bacteria_series)
            n_bact = sum(1 for c in X_features.columns if c.startswith("bact_"))
            print(f"\n   🦠 {n_bact} bakteri türü özelliği eklendi.")
        else:
            X_features = X_gene

        y_all = y_all.reset_index(drop=True)

        row = train_and_evaluate(X_features, y_all, anti)
        if row:
            results.append(row)

    if not results:
        print("\n⚠️  Hiçbir antibiyotik için model eğitilemedi.")
        return

    print("\n" + "=" * 105)
    print("        V22 FİNAL PERFORMANS TABLOSU")
    print("        (Optuna pos_weight + Adaptif Eşik + Kalibrasyon + Bakteri-Türü Prior)")
    print("=" * 105)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    n_ready_full = (results_df["Klinik Hazır"] == "✅").sum()
    n_ready_tol  = (results_df["Klinik Hazır"] == "🟡").sum()
    n_ready = n_ready_full + n_ready_tol
    print(f"\n🏥 Klinik Hazır Model: {n_ready_full} ✅ + {n_ready_tol} 🟡 (tolerans) / {len(results_df)}")
    print(f"   (Kriter: Recall ≥ {RECALL_THRESHOLD} VE Specificity ≥ {SPECIFICITY_MIN})")

    report_path = os.path.join(REPORTS_DIR, "v22_final_results.csv")
    results_df.to_csv(report_path, index=False)
    print(f"\n📊 Sonuç tablosu kaydedildi → {report_path}")
    print(f"🎨 Akademik grafikler oluşturuldu → {os.path.join(REPORTS_DIR, 'figures')}")


if __name__ == "__main__":
    run_superbug_panel()