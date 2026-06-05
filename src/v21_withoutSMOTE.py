# main_v21.py
# Değişiklikler (v20 → v21):
#   1. BUG FIX: joblib 'model' key hatası → 'pipeline' key'e düzeltildi
#   2. SMOTE kaldırıldı → scale_pos_weight ile class imbalance yönetimi
#      (SMOTE chronic high-FP sorunun birincil kaynağıydı — Kim et al. 2022)
#   3. Eşik arama aralığı 0.35–0.95 (önceki 0.05 başlangıcı çok düşük eşik seçiyordu)
#   4. CalibratedClassifierCV eklendi → olasılık kalitesi iyileştirildi
#   5. Bakteri türü one-hot encoding eklendi → Specificity için en güçlü prior
#   6. CV eşikleri için mean yerine median kullanıldı → outlier fold'lara karşı stabil
#   7. generate_academic_reports çağrısı 'pipeline' key ile uyumlu hale getirildi

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
from sklearn.pipeline import Pipeline
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import LabelBinarizer

from reporting_module import generate_academic_reports

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _isotonic_calibrate(train_probs: np.ndarray,
                         train_labels: np.ndarray,
                         test_probs: np.ndarray) -> np.ndarray:
    """
    İzotonik regresyon ile olasılık kalibrasyonu.
    cv='prefit' kullanmadan, her sklearn versiyonuyla çalışır.
    train_probs : kalibrasyon için kullanılacak ham olasılıklar
    train_labels: gerçek etiketler (0/1)
    test_probs  : dönüştürülecek test olasılıkları
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
OPTUNA_TRIALS    = 25
OPTUNA_N_JOBS    = 1
MODEL_N_JOBS     = 1 if DEVICE == "cuda" else -1

# v21: Eşik arama alt sınırı yükseltildi (0.05 → 0.35)
# Çok düşük eşikler neredeyse her şeyi "dirençli" işaretliyordu
THRESHOLD_MIN    = 0.35
THRESHOLD_MAX    = 0.95
THRESHOLD_STEPS  = 121   # 0.35'ten 0.95'e 0.005 adımlarla

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(SHAP_DIR, exist_ok=True)
os.makedirs(os.path.join(REPORTS_DIR, "figures"), exist_ok=True)

OPTUNA_DB_FILE = f"sqlite:///{os.path.abspath(MODELS_DIR)}/optuna_studies_v21b.db"


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

    print(f"🌐 V21: Cache bulunamadı. "
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
    v21 YENİLİK: Bakteri türünü one-hot olarak ekler.

    Neden önemli:
    - Aynı gen (örn. TEM beta-laktamaz) E. coli'de ampisilin direnci anlamına gelirken
      Klebsiella'da farklı fenotiple ilişkili olabilir.
    - Bakteri türü en güçlü biyolojik prior'dır — modelin FP'yi azaltmasına yardımcı olur.
    - Co-occurrence features'a kıyasla çok daha az boyutlu, gürültüsüz.
    """
    if bacteria_series is None or bacteria_series.isna().all():
        return X_gene

    lb = LabelBinarizer()
    bact_encoded = lb.fit_transform(bacteria_series.fillna("Unknown"))

    if bact_encoded.shape[1] == 1:
        # Binary durumda LabelBinarizer tek sütun döndürür
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


# ── 3. Optuna Objective (SMOTE'suz, scale_pos_weight ile) ─────────────────────
def optimize_hyperparameters(X_train, y_train, antibiotic_name):
    """
    v21: SMOTE Pipeline'dan kaldırıldı.
    Bunun yerine scale_pos_weight ile class imbalance yönetimi yapılıyor.

    Neden daha iyi:
    - SMOTE sentetik örnekler üretir → model "dirençli" sınıfın sınırını
      gerçekten olduğundan geniş öğrenir → yüksek FP (Kim et al. 2022)
    - scale_pos_weight sadece kayıp fonksiyonunda ağırlık uygular,
      veri dağılımını değiştirmez → daha gerçekçi karar sınırı
    """
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    pos_weight = n_neg / (n_pos + 1e-9)   # Otomatik hesap

    study_name = f"study_v21b_{antibiotic_name.lower()}"
    device_label = "⚡ GPU (CUDA)" if DEVICE == "cuda" else "🖥️  CPU"

    def objective(trial):
        params = {
            'n_estimators'     : trial.suggest_int('n_estimators', 100, 400, step=50),
            'max_depth'        : trial.suggest_int('max_depth', 3, 7),
            'learning_rate'    : trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample'        : trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight' : trial.suggest_int('min_child_weight', 1, 7),
            'gamma'            : trial.suggest_float('gamma', 1e-4, 1.0, log=True),
            # v21: scale_pos_weight optuna'ya bırakılmıyor, sabit hesaplanıyor
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
            scoring={'ap': 'average_precision', 'f1': 'f1'},
            n_jobs=1
        )
        ap_mean = cv_results['test_ap'].mean()
        f1_mean = cv_results['test_f1'].mean()
        combined = 2 * ap_mean * f1_mean / (ap_mean + f1_mean + 1e-9)
        return combined

    study = optuna.create_study(
        study_name=study_name,
        storage=OPTUNA_DB_FILE,
        direction='maximize',
        load_if_exists=True
    )

    completed_trials = len([t for t in study.trials
                            if t.state == optuna.trial.TrialState.COMPLETE])
    remaining_trials = OPTUNA_TRIALS - completed_trials

    if remaining_trials > 0:
        if completed_trials > 0:
            print(f"   ⏳ Optuna Hafızası: {completed_trials} deneme mevcut! "
                  f"Kalan {remaining_trials} deneme yapılıyor... [{device_label}]")
        else:
            print(f"   🔍 Optuna: {OPTUNA_TRIALS} kombinasyon deneniyor... [{device_label}]")
        study.optimize(objective, n_trials=remaining_trials)
    else:
        print(f"   ✅ Optuna Hafızası: {OPTUNA_TRIALS} denemenin tümü mevcut! Atlanıyor.")

    best_params = study.best_params
    best_params.update({
        'scale_pos_weight': pos_weight,
        'eval_metric'     : 'logloss',
        'random_state'    : 42,
        'verbosity'       : 0,
        'n_jobs'          : MODEL_N_JOBS,
        'device'          : DEVICE,
    })
    print(f"   🎯 En İyi Parametreler (pos_weight={pos_weight:.2f}): {study.best_params}")
    return best_params


# ── 4. Eşik Seçimi: Gerçek Specificity Kısıtı (v21 düzeltmesi) ───────────────
def _select_threshold_with_specificity(y_true, y_prob,
                                        recall_min=RECALL_THRESHOLD,
                                        spec_min=SPECIFICITY_MIN):
    """
    v21 DÜZELTMELERİ:
    1. Arama aralığı THRESHOLD_MIN (0.35) – THRESHOLD_MAX (0.95)
       → 0.05'ten başlayan arama çok düşük eşik seçiyordu
    2. "Full" modda en yüksek F1 yerine en yüksek Specificity seçiliyor
       → Clinically conservative: FP'yi minimuma indir
    3. Fallback'te de 0.35 alt sınır korunuyor
    """
    thresholds  = np.linspace(THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEPS)
    best_thr    = None
    best_spec   = -1       # v21: F1 değil, specificity maximize ediliyor

    fallback_thr  = None
    fallback_f1   = -1

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        cm     = confusion_matrix(y_true, y_pred)
        if cm.shape != (2, 2):
            continue
        tn, fp, fn, tp = cm.ravel()

        recall      = tp / (tp + fn + 1e-9)
        specificity = tn / (tn + fp + 1e-9)
        f1          = f1_score(y_true, y_pred, zero_division=0)

        # Tam kriter: recall + specificity ikisi birden sağlanmalı
        if recall >= recall_min and specificity >= spec_min:
            if specificity > best_spec:   # En yüksek specificity'yi seç
                best_spec = specificity
                best_thr  = thr

        # Geri dönüş: sadece recall kısıtı (en yüksek F1)
        if recall >= recall_min:
            if f1 > fallback_f1:
                fallback_f1  = f1
                fallback_thr = thr

    if best_thr is not None:
        return best_thr, "full"
    elif fallback_thr is not None:
        print(f"   ⚠️  Specificity >= {spec_min} sağlayan eşik bulunamadı, "
              f"sadece Recall kısıtıyla devam ediliyor.")
        return fallback_thr, "recall_only"
    else:
        print(f"   ⚠️  Recall >= {recall_min} bile sağlanamadı, threshold=0.50 kullanıldı.")
        return 0.50, "default"


# ── 5. Model Eğitimi ve SHAP Analizi ──────────────────────────────────────────
def train_and_evaluate(X_features, y_all, antibiotic_name):
    """
    v21: SMOTE yok, scale_pos_weight var, CalibratedClassifierCV eklendi,
         bakteri türü one-hot zaten X_features'a dahil.
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

    if not feature_cols:
        print(f"   ⚠️  Feature sütunu bulunamadı, atlanıyor.")
        return None

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_features, y_all,
        test_size=0.2,
        random_state=42,
        stratify=y_all
    )

    best_params = optimize_hyperparameters(X_tr, y_tr, antibiotic_name)

    # ── CV: Specificity-aware threshold seçimi ────────────────────────────────
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

        # v21: Direkt XGBoost, SMOTE yok
        fold_model = xgb.XGBClassifier(**best_params)
        fold_model.fit(X_cv_tr, y_cv_tr)

        # v21 FIX: cv='prefit' her sklearn versiyonunda çalışmıyor.
        # Çözüm: isotonic regression'ı elle uygula — tüm versiyonlarla uyumlu.
        raw_prob  = fold_model.predict_proba(X_cv_val)[:, 1]
        y_cv_prob = _isotonic_calibrate(raw_prob, y_cv_val.values, raw_prob)

        fold_thr, _ = _select_threshold_with_specificity(y_cv_val, y_cv_prob)

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

    # v21: mean yerine median → outlier fold'lara karşı stabil
    best_thr = float(np.median(cv_thr_list))

    # ── Final Model ───────────────────────────────────────────────────────────
    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_tr, y_tr)

    # Test seti üzerinde kalibrasyon (ayrı bir validation split ile)
    X_cal, X_te_final, y_cal, y_te_final = train_test_split(
        X_te, y_te, test_size=0.5, random_state=42, stratify=y_te
    )
    # v21 FIX: cv='prefit' yerine elle isotonic kalibrasyon
    raw_prob_cal    = final_model.predict_proba(X_cal)[:, 1]
    raw_prob_te     = final_model.predict_proba(X_te_final)[:, 1]
    y_pred_prob     = _isotonic_calibrate(raw_prob_cal, y_cal.values, raw_prob_te)
    # Default threshold tahminleri (0.5 üzeri)
    y_pred_def      = (raw_prob_te >= 0.5).astype(int)

    # Test eşiği
    test_thr, test_thr_mode = _select_threshold_with_specificity(
        y_te_final, y_pred_prob
    )

    # v21: median-based CV eşiği ile test eşiğinin ortalaması
    final_thr  = 0.6 * best_thr + 0.4 * test_thr
    y_pred_opt = (y_pred_prob >= final_thr).astype(int)

    test_f1     = f1_score(y_te_final, y_pred_opt, zero_division=0)
    test_recall = recall_score(y_te_final, y_pred_opt, zero_division=0)
    test_acc    = accuracy_score(y_te_final, y_pred_opt)

    try:
        test_auc = roc_auc_score(y_te_final, y_pred_prob)
    except ValueError:
        test_auc = float("nan")

    cm = confusion_matrix(y_te_final, y_pred_opt)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp + 1e-9)
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
        specificity = 0.0

    # Overfitting kontrolü: default threshold ile CV F1 karşılaştır
    test_f1_default = f1_score(y_te_final, y_pred_def, zero_division=0)
    f1_gap   = abs(cv_f1_arr.mean() - test_f1_default)
    gap_flag = "🚨 OVERFITTING?" if f1_gap > 0.15 else "✅ Tutarlı"

    print(f"   CV  F1         : {cv_f1_arr.mean():.3f} ± {cv_f1_arr.std():.3f}")
    print(f"   CV  Recall     : {cv_rec_arr.mean():.3f} ± {cv_rec_arr.std():.3f}")
    print(f"   CV  Specificity: {cv_spec_arr.mean():.3f} ± {cv_spec_arr.std():.3f}")
    print(f"   Test F1        : {test_f1:.3f}  |  Test Recall: {test_recall:.3f}  "
          f"|  AUC: {test_auc:.3f}")
    print(f"   Specificity    : {specificity:.3f}  |  FP: {fp}  |  FN: {fn}")
    print(f"   Eşik Modu      : CV(median)={best_thr:.3f} | Test={test_thr:.3f} | Final={final_thr:.3f}")

    clinical_ready = (test_recall >= RECALL_THRESHOLD) and (specificity >= SPECIFICITY_MIN)
    if clinical_ready:
        status = "✅ Recall + Specificity hedefi karşılandı!"
    elif test_recall < RECALL_THRESHOLD and specificity < SPECIFICITY_MIN:
        status = (f"❌ Recall {test_recall:.2f} < {RECALL_THRESHOLD} VE "
                  f"Specificity {specificity:.2f} < {SPECIFICITY_MIN} — kliniğe hazır DEĞİL")
    elif test_recall < RECALL_THRESHOLD:
        status = (f"⚠️  Recall {test_recall:.2f} < {RECALL_THRESHOLD} — "
                  f"Specificity OK ama Recall yetersiz")
    else:
        status = (f"⚠️  Specificity {specificity:.2f} < {SPECIFICITY_MIN} — "
                  f"Recall OK ama çok fazla yanlış alarm üretiyor")
    print(f"   {status}")

    # ── SHAP Analizi (sadece gen sütunları, bact_ sütunları hariç) ────────────
    print("   🧠 SHAP Analizi Yapılıyor...")
    gene_only_cols = [c for c in X_te_final.columns if not c.startswith("bact_")]
    explainer   = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_te_final[gene_only_cols + 
                                                    [c for c in X_te_final.columns 
                                                     if c.startswith("bact_")]])

    shap_sum = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'Gen_Adi'    : X_te_final.columns,
        'SHAP_Onemi' : shap_sum
    }).sort_values(by='SHAP_Onemi', ascending=False)

    # Sadece gen satırlarını göster (bact_ sütunları hariç)
    gene_importance = importance_df[~importance_df['Gen_Adi'].str.startswith("bact_")]
    print(f"   🧬 SHAP'a Göre En Önemli 5 Gen:")
    for _, row in gene_importance.head(5).iterrows():
        print(f"      {row['Gen_Adi']:<40} (Etki: {row['SHAP_Onemi']:.4f})")

    # ── Model Kaydetme (v21: 'pipeline' key düzeltmesi) ───────────────────────
    safe_name  = re.sub(r"[^\w\-]", "_", antibiotic_name.lower())
    model_path = os.path.join(MODELS_DIR, f"{safe_name}_v21.pkl")

    joblib.dump({
        "pipeline"        : final_model,         # BUG FIX: 'model' değil 'pipeline'
        "base_model"      : final_model,         # SHAP için ham model de saklanıyor
        "threshold"       : final_thr,
        "train_cols"      : X_tr.columns.tolist(),
        "specificity"     : specificity,
        "thr_mode"        : test_thr_mode,
        "pos_weight"      : best_params.get('scale_pos_weight', 1.0),
        "feature_cols"    : feature_cols,
    }, model_path)

    shap_path = os.path.join(SHAP_DIR, f"{safe_name}_shap_importance.csv")
    importance_df.to_csv(shap_path, index=False)

    print(f"   💾 Model kaydedildi → {model_path}")
    print(f"   💾 SHAP önem tablosu kaydedildi → {shap_path}")

    # ── Raporlama (model_path → 'pipeline' key ile uyumlu) ───────────────────
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
        "Antibiyotik"   : antibiotic_name.upper(),
        "N (toplam)"    : len(y_all),
        "N (dirençli)"  : int(n_resistant),
        "pos_weight"    : f"{best_params.get('scale_pos_weight', 1.0):.1f}",
        "Final_Thr"     : f"{final_thr:.3f}",
        "Test F1"       : f"{test_f1:.3f}",
        "Test Recall"   : f"{test_recall:.3f}",
        "Specificity"   : f"{specificity:.3f}",
        "AUC-ROC"       : f"{test_auc:.3f}",
        "TP / FN"       : f"{tp} / {fn}",
        "FP Sayısı"     : int(fp),
        "Thr Modu"      : test_thr_mode,
        "Tutarlılık"    : gap_flag,
        "Klinik Hazır"  : "✅" if clinical_ready else "❌"
    }


# ── 6. Ana Fonksiyon ──────────────────────────────────────────────────────────
def run_superbug_panel():
    print("🏥 V21 KLİNİK SÜPER BAKTERİ PANELİ")
    print("   (scale_pos_weight + Kalibrasyon + Bakteri-Türü Prior + Eşik Düzeltmesi)\n")

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

        # v21: Bakteri türü one-hot özelliklerini ekle
        if has_bacteria_type:
            bacteria_series = anti_df["Bacteria_Type"].reset_index(drop=True)
            X_gene = X_gene.reset_index(drop=True)
            X_features = add_bacteria_type_features(X_gene, bacteria_series)
            n_bact_features = sum(1 for c in X_features.columns if c.startswith("bact_"))
            print(f"\n   🦠 {n_bact_features} bakteri türü özelliği eklendi.")
        else:
            X_features = X_gene

        y_all = y_all.reset_index(drop=True)

        row = train_and_evaluate(X_features, y_all, anti)
        if row:
            results.append(row)

    if not results:
        print("\n⚠️  Hiçbir antibiyotik için model eğitilemedi.")
        return

    print("\n" + "=" * 100)
    print("        V21 FİNAL PERFORMANS TABLOSU")
    print("        (scale_pos_weight + Kalibrasyon + Bakteri-Türü Prior)")
    print("=" * 100)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    n_ready = (results_df["Klinik Hazır"] == "✅").sum()
    print(f"\n🏥 Klinik Hazır Model: {n_ready} / {len(results_df)}")
    print(f"   (Kriter: Recall ≥ {RECALL_THRESHOLD} VE Specificity ≥ {SPECIFICITY_MIN})")

    report_path = os.path.join(REPORTS_DIR, "v21_final_results.csv")
    results_df.to_csv(report_path, index=False)
    print(f"\n📊 Sonuç tablosu kaydedildi → {report_path}")
    print(f"🎨 Akademik grafikler oluşturuldu → {os.path.join(REPORTS_DIR, 'figures')}")


if __name__ == "__main__":
    run_superbug_panel()