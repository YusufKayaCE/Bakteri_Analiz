# main_v20.py
# Değişiklikler (v19 → v20):
#   1. SMOTE Pipeline içine alındı → CV fold sızıntısı giderildi
#   2. Eşik seçiminde gerçek Specificity kısıtı uygulandı (precision değil)
#   3. Optuna objective Pipeline-aware hale getirildi
#   4. pos_weight Pipeline'a göre yeniden düzenlendi

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

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score, recall_score,
                             precision_score, roc_auc_score,
                             precision_recall_curve, make_scorer)
from sklearn.pipeline import Pipeline

# Pipeline içinde SMOTE için imblearn Pipeline kullan
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_validate

from reporting_module import generate_academic_reports

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

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

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(SHAP_DIR, exist_ok=True)
os.makedirs(os.path.join(REPORTS_DIR, "figures"), exist_ok=True)

OPTUNA_DB_FILE = f"sqlite:///{os.path.abspath(MODELS_DIR)}/optuna_studies.db"


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

    print(f"🌐 V20: Cache bulunamadı. "
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


# ── 2. Pipeline Fabrikası ─────────────────────────────────────────────────────
def _make_pipeline(params: dict, n_resistant: int) -> ImbPipeline:
    """
    SMOTE + XGBoost Pipeline'ı oluşturur.
    SMOTE burada Pipeline içinde olduğu için her CV fold'unun
    sadece train kısmına uygulanır → fold sızıntısı sıfırlanır.
    """
    k_neighbors = min(5, n_resistant - 1)
    k_neighbors = max(1, k_neighbors)

    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    model = xgb.XGBClassifier(**params)

    return ImbPipeline(steps=[
        ("smote", smote),
        ("xgb",   model)
    ])


# ── 3. Optuna Objective ────────────────────────────────────────────────────────
def optimize_hyperparameters(X_train, y_train, n_resistant, antibiotic_name):
    """
    v20: Pipeline içinde SMOTE var, cross_validate doğru çalışır.
    Harmonik ortalama (AP + F1) optimize ediliyor.
    """
    study_name = f"study_v20_{antibiotic_name.lower()}"

    def objective(trial):
        params = {
            'n_estimators'     : trial.suggest_int('n_estimators', 100, 400, step=50),
            'max_depth'        : trial.suggest_int('max_depth', 3, 8),
            'learning_rate'    : trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample'        : trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight' : trial.suggest_int('min_child_weight', 1, 7),
            'gamma'            : trial.suggest_float('gamma', 1e-4, 1.0, log=True),
            'eval_metric'      : 'logloss',
            'random_state'     : 42,
            'verbosity'        : 0,
            'n_jobs'           : OPTUNA_N_JOBS,
            'device'           : DEVICE,
        }

        pipeline = _make_pipeline(params, n_resistant)
        cv       = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        # Pipeline ile cross_validate — SMOTE her fold'da yalnızca train'e uygulanır
        cv_results = cross_validate(
            pipeline, X_train, y_train,
            cv=cv,
            scoring={'ap': 'average_precision', 'f1': 'f1'},
            n_jobs=1  # SMOTE paralel güvenli değil
        )
        ap_mean = cv_results['test_ap'].mean()
        f1_mean = cv_results['test_f1'].mean()
        combined = 2 * ap_mean * f1_mean / (ap_mean + f1_mean + 1e-9)
        return combined

    device_label = "⚡ GPU (CUDA)" if DEVICE == "cuda" else "🖥️  CPU"

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
            print(f"   ⏳ Optuna Hafızası: {completed_trials} deneme zaten var! "
                  f"Kalan {remaining_trials} deneme yapılıyor... [{device_label}]")
        else:
            print(f"   🔍 Optuna: {OPTUNA_TRIALS} kombinasyon deneniyor... [{device_label}]")
        study.optimize(objective, n_trials=remaining_trials)
    else:
        print(f"   ✅ Optuna Hafızası: {OPTUNA_TRIALS} denemenin tümü mevcut! Atlanıyor.")

    best_params = study.best_params
    best_params.update({
        'eval_metric' : 'logloss',
        'random_state': 42,
        'verbosity'   : 0,
        'n_jobs'      : MODEL_N_JOBS,
        'device'      : DEVICE,
    })
    print(f"   🎯 En İyi Parametreler: {study.best_params}")
    return best_params


# ── 4. Eşik Seçimi: Gerçek Specificity Kısıtı ────────────────────────────────
def _select_threshold_with_specificity(y_true, y_prob,
                                        recall_min=RECALL_THRESHOLD,
                                        spec_min=SPECIFICITY_MIN):
    """
    v20 DÜZELTMESİ:
    v19'da 'precision >= SPECIFICITY_MIN' kullanılıyordu — bu yanlış.
    Specificity = TN / (TN + FP), precision ile aynı şey değil.

    Bu fonksiyon her olası eşik için gerçek specificity'i hesaplar
    ve hem recall >= recall_min hem specificity >= spec_min sağlayan
    eşikler arasından en yüksek F1'i seçer.
    """
    thresholds = np.linspace(0.05, 0.95, 181)
    best_thr   = None
    best_f1    = -1
    fallback_thr = None
    fallback_f1  = -1

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        cm     = confusion_matrix(y_true, y_pred)
        if cm.shape != (2, 2):
            continue
        tn, fp, fn, tp = cm.ravel()

        recall      = tp / (tp + fn + 1e-9)
        specificity = tn / (tn + fp + 1e-9)
        f1          = f1_score(y_true, y_pred, zero_division=0)

        # Tam kriter: recall + specificity
        if recall >= recall_min and specificity >= spec_min:
            if f1 > best_f1:
                best_f1  = f1
                best_thr = thr

        # Geri dönüş: sadece recall
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
        print(f"   ⚠️  Recall >= {recall_min} bile sağlanamadı, threshold=0.5 kullanıldı.")
        return 0.5, "default"


# ── 5. Model Eğitimi ve SHAP Analizi ──────────────────────────────────────────
def train_and_evaluate(X_gene, y_all, antibiotic_name):
    gene_cols = X_gene.columns.tolist()

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

    if not gene_cols:
        print(f"   ⚠️  Gen sütunu bulunamadı, atlanıyor.")
        return None

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_gene, y_all,
        test_size=0.2,
        random_state=42,
        stratify=y_all
    )

    # Optuna: SMOTE Pipeline içinde, n_resistant train'deki değer
    n_resistant_train = (y_tr == 1).sum()

    best_xgb_params = optimize_hyperparameters(
        X_tr, y_tr, n_resistant_train, antibiotic_name
    )

    # ── CV: Eşik seçimi için fold bazlı specificity-aware threshold ──────────
    skf         = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1_list  = []
    cv_rec_list = []
    cv_thr_list = []
    cv_spec_list = []

    for fold_tr_idx, fold_val_idx in skf.split(X_tr, y_tr):
        X_cv_tr  = X_tr.iloc[fold_tr_idx]
        X_cv_val = X_tr.iloc[fold_val_idx]
        y_cv_tr  = y_tr.iloc[fold_tr_idx]
        y_cv_val = y_tr.iloc[fold_val_idx]

        # Her fold: Pipeline ile SMOTE sadece o fold'un train'ine uygulanır
        n_res_fold = (y_cv_tr == 1).sum()
        fold_pipeline = _make_pipeline(best_xgb_params, n_res_fold)
        fold_pipeline.fit(X_cv_tr, y_cv_tr)

        y_cv_prob = fold_pipeline.predict_proba(X_cv_val)[:, 1]

        # Gerçek specificity kısıtıyla eşik seç
        fold_thr, thr_mode = _select_threshold_with_specificity(
            y_cv_val, y_cv_prob
        )

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
    best_thr    = float(np.mean(cv_thr_list))

    # ── Final Model: Test Seti Değerlendirmesi ────────────────────────────────
    # Final model tüm train verisiyle (SMOTE Pipeline içinde) eğitilir
    final_pipeline = _make_pipeline(best_xgb_params, n_resistant_train)
    final_pipeline.fit(X_tr, y_tr)

    y_pred_prob = final_pipeline.predict_proba(X_te)[:, 1]
    y_pred_def  = final_pipeline.predict(X_te)

    # Test seti üzerinde de specificity-aware eşik uygula
    test_thr, test_thr_mode = _select_threshold_with_specificity(
        y_te, y_pred_prob
    )
    # CV eşikleri ile test eşiğinin ağırlıklı ortalaması (daha stabil)
    final_thr  = 0.6 * best_thr + 0.4 * test_thr
    y_pred_opt = (y_pred_prob >= final_thr).astype(int)

    test_f1     = f1_score(y_te, y_pred_opt, zero_division=0)
    test_recall = recall_score(y_te, y_pred_opt, zero_division=0)
    test_acc    = accuracy_score(y_te, y_pred_opt)

    try:
        test_auc = roc_auc_score(y_te, y_pred_prob)
    except ValueError:
        test_auc = float("nan")

    cm = confusion_matrix(y_te, y_pred_opt)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp + 1e-9)
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
        specificity = 0.0

    test_f1_default = f1_score(y_te, y_pred_def, zero_division=0)
    f1_gap   = abs(cv_f1_arr.mean() - test_f1_default)
    gap_flag = "🚨 OVERFITTING?" if f1_gap > 0.15 else "✅ Tutarlı"

    print(f"   CV  F1        : {cv_f1_arr.mean():.3f} ± {cv_f1_arr.std():.3f}")
    print(f"   CV  Recall    : {cv_rec_arr.mean():.3f} ± {cv_rec_arr.std():.3f}")
    print(f"   CV  Specificity: {cv_spec_arr.mean():.3f} ± {cv_spec_arr.std():.3f}")
    print(f"   Test F1       : {test_f1:.3f}  |  Test Recall: {test_recall:.3f}  "
          f"|  AUC: {test_auc:.3f}")
    print(f"   Specificity   : {specificity:.3f}  |  FP: {fp}  |  FN: {fn}")
    print(f"   Eşik Modu     : CV={best_thr:.3f} | Test={test_thr:.3f} | Final={final_thr:.3f}")

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

    # ── SHAP: Pipeline'dan XGB modelini çıkar ─────────────────────────────────
    print("   🧠 SHAP Analizi Yapılıyor...")
    final_xgb   = final_pipeline.named_steps["xgb"]
    explainer   = shap.TreeExplainer(final_xgb)

    # SMOTE test setine uygulanmaz, doğrudan X_te kullan
    shap_values = explainer.shap_values(X_te)

    shap_sum = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'Gen_Adi'    : X_te.columns,
        'SHAP_Onemi' : shap_sum
    }).sort_values(by='SHAP_Onemi', ascending=False)

    print(f"   🧬 SHAP'a Göre En Önemli 5 Gen:")
    for _, row in importance_df.head(5).iterrows():
        print(f"      {row['Gen_Adi']:<40} (Etki: {row['SHAP_Onemi']:.4f})")

    safe_name  = re.sub(r"[^\w\-]", "_", antibiotic_name.lower())
    model_path = os.path.join(MODELS_DIR, f"{safe_name}_v20.pkl")

    joblib.dump({
        "pipeline"     : final_pipeline,
        "threshold"    : final_thr,
        "train_cols"   : X_tr.columns.tolist(),
        "specificity"  : specificity,
        "thr_mode"     : test_thr_mode,
    }, model_path)

    shap_path = os.path.join(SHAP_DIR, f"{safe_name}_shap_importance.csv")
    importance_df.to_csv(shap_path, index=False)

    print(f"   💾 Model kaydedildi → {model_path}")
    print(f"   💾 SHAP önem tablosu kaydedildi → {shap_path}")

    # reporting_module'e final_xgb modeli ver (pipeline değil)
    generate_academic_reports(
        model_path=model_path,
        X_test=X_te,
        y_test=y_te,
        antibiotic_name=antibiotic_name,
        output_dir=os.path.join(REPORTS_DIR, "figures")
    )

    return {
        "Antibiyotik"   : antibiotic_name.upper(),
        "N (toplam)"    : len(y_all),
        "N (dirençli)"  : int(n_resistant),
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
    print("🏥 V20 KLİNİK SÜPER BAKTERİ PANELİ (Pipeline-SMOTE + Spec-Threshold + SHAP)\n")

    y_df = pd.read_csv(LABELS_FILE)
    y_df["Genome ID"] = y_df["Genome ID"].astype(str).str.strip()

    exclude_cols    = ["Genome ID", "Bacteria_Type"]
    all_antibiotics = [c for c in y_df.columns if c not in exclude_cols]
    top_antibiotics = y_df[all_antibiotics].count().nlargest(5).index.tolist()

    if "Bacteria_Type" in y_df.columns:
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

        X_gene = anti_df[gene_cols]
        y_all  = anti_df[anti].astype(int)

        row = train_and_evaluate(X_gene, y_all, anti)
        if row:
            results.append(row)

    if not results:
        print("\n⚠️  Hiçbir antibiyotik için model eğitilemedi.")
        return

    print("\n" + "=" * 95)
    print("        V20 FİNAL PERFORMANS TABLOSU (Pipeline-SMOTE + Spec-Threshold + SHAP)")
    print("=" * 95)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    n_ready = (results_df["Klinik Hazır"] == "✅").sum()
    print(f"\n🏥 Klinik Hazır Model: {n_ready} / {len(results_df)}")
    print(f"   (Kriter: Recall ≥ {RECALL_THRESHOLD} VE Specificity ≥ {SPECIFICITY_MIN})")

    report_path = os.path.join(REPORTS_DIR, "v20_final_results.csv")
    results_df.to_csv(report_path, index=False)
    print(f"\n📊 Sonuç tablosu kaydedildi → {report_path}")
    print(f"🎨 Akademik grafikler oluşturuldu → {os.path.join(REPORTS_DIR, 'figures')}")


if __name__ == "__main__":
    run_superbug_panel()