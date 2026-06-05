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

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score, recall_score,
                             precision_score, roc_auc_score,
                             precision_recall_curve)

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
LABELS_FILE    = "../data/processed/v2_multilabel_labels.csv"
CACHE_FILE     = "../data/processed/v17_amr_genes_cache.csv"
MODELS_DIR     = "../models"
REPORTS_DIR    = "../reports"
SHAP_DIR       = "../reports/shap_values"

MAX_GENOMES    = 30000
BATCH_SIZE     = 100
RECALL_THRESHOLD = 0.80
OPTUNA_TRIALS  = 25
OPTUNA_N_JOBS  = 1
MODEL_N_JOBS   = 1 if DEVICE == "cuda" else -1

# DÜZELTME 1: Dizinleri önce oluştur, sonra OPTUNA_DB_FILE'ı tanımla
# (SQLite dosyası henüz mevcut olmayan bir dizine yazılamaz)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(SHAP_DIR, exist_ok=True)

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

    print(f"🌐 V18.1: Cache bulunamadı. "
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

            # DÜZELTME 2: Oto-kayıt koşulu her 10 batch'te bir tetiklenecek şekilde güncellendi
            # (eskisi: % 1000 == 0 → BATCH_SIZE=100 ile nadiren tetikleniyordu)
            if (i // BATCH_SIZE + 1) % 10 == 0 and all_genes:
                pd.DataFrame(all_genes).to_csv(CACHE_FILE, index=False)
                print(f"  💾 Oto-kayıt yapıldı.")

            time.sleep(1)

        except Exception as e:
            print(f"  ❌ Hata (atlanıyor): {e}")
            time.sleep(3)

    return pd.DataFrame(all_genes)


# ── 2. Optuna Objective ────────────────────────────────────────────────────────
def optimize_hyperparameters(X_train, y_train, pos_weight, antibiotic_name):
    study_name = f"study_{antibiotic_name.lower()}"

    def objective(trial):
        param = {
            'n_estimators'     : trial.suggest_int('n_estimators', 100, 400, step=50),
            'max_depth'        : trial.suggest_int('max_depth', 3, 8),
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
        cv     = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        model  = xgb.XGBClassifier(**param)
        scores = cross_val_score(model, X_train, y_train,
                                 cv=cv, scoring='average_precision',
                                 n_jobs=OPTUNA_N_JOBS)
        return scores.mean()

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
        'scale_pos_weight': pos_weight,
        'eval_metric'     : 'logloss',
        'random_state'    : 42,
        'verbosity'       : 0,
        'n_jobs'          : MODEL_N_JOBS,
        'device'          : DEVICE,
    })
    print(f"   🎯 En İyi Parametreler: {study.best_params}")
    return best_params


# ── 3. Model Eğitimi ve SHAP Analizi ──────────────────────────────────────────
def train_and_evaluate(X_gene, y_all, antibiotic_name):
    # DÜZELTME 3: gene_cols artık parametre olarak alınmıyor,
    # X_gene'den doğrudan türetiliyor (gereksiz bağımlılık kaldırıldı)
    gene_cols = X_gene.columns.tolist()

    class_counts  = y_all.value_counts()
    n_resistant   = class_counts.get(1, 0)
    n_susceptible = class_counts.get(0, 0)
    majority_pct  = class_counts.max() / len(y_all) * 100

    print(f"\n{'─'*70}")
    print(f"💊 {antibiotic_name.upper()}")
    print(f"   Dağılım → Duyarlı(0): {n_susceptible} | "
          f"Dirençli(1): {n_resistant}  ({majority_pct:.1f}% çoğunluk)")

    if n_resistant < 10:
        print(f"   ⚠️  Dirençli örnek çok az (<10), atlanıyor.")
        return None

    # DÜZELTME 4: Duyarlı örnek kontrolü eklendi
    if n_susceptible < 10:
        print(f"   ⚠️  Duyarlı örnek çok az (<10), atlanıyor.")
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

    # DÜZELTME 5: reindex kaldırıldı. Aynı DataFrame'den bölündüğü için
    # X_tr ve X_te zaten aynı sütunlara sahip. reindex gereksiz ve yanıltıcıydı.
    pos_weight = n_susceptible / max(n_resistant, 1)

    best_xgb_params = optimize_hyperparameters(X_tr, y_tr, pos_weight, antibiotic_name)

    skf         = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1_list  = []
    cv_rec_list = []
    cv_thr_list = []

    for fold_tr_idx, fold_val_idx in skf.split(X_tr, y_tr):
        X_cv_tr  = X_tr.iloc[fold_tr_idx]
        X_cv_val = X_tr.iloc[fold_val_idx]
        y_cv_tr  = y_tr.iloc[fold_tr_idx]
        y_cv_val = y_tr.iloc[fold_val_idx]

        fold_model = xgb.XGBClassifier(**best_xgb_params)
        fold_model.fit(X_cv_tr, y_cv_tr)
        y_cv_prob = fold_model.predict_proba(X_cv_val)[:, 1]

        p_fold, r_fold, t_fold = precision_recall_curve(y_cv_val, y_cv_prob)
        valid_fold = np.where(r_fold[:-1] >= RECALL_THRESHOLD)[0]

        if len(valid_fold) > 0:
            best_fold_idx = valid_fold[np.argmax([
                2 * p_fold[i] * r_fold[i] / (p_fold[i] + r_fold[i] + 1e-9)
                for i in valid_fold
            ])]
            fold_thr = float(t_fold[best_fold_idx])
        else:
            # DÜZELTME 6: Recall hedefi hiç karşılanamadığında loglama eklendi
            fold_thr = 0.5
            print(f"   ⚠️  Bu fold'da recall >= {RECALL_THRESHOLD} sağlanamadı, "
                  f"threshold=0.5 kullanıldı.")

        y_cv_pred = (y_cv_prob >= fold_thr).astype(int)
        cv_f1_list.append(f1_score(y_cv_val, y_cv_pred, zero_division=0))
        cv_rec_list.append(recall_score(y_cv_val, y_cv_pred, zero_division=0))
        cv_thr_list.append(fold_thr)

    cv_f1_arr  = np.array(cv_f1_list)
    cv_rec_arr = np.array(cv_rec_list)

    final_model = xgb.XGBClassifier(**best_xgb_params)
    final_model.fit(X_tr, y_tr)
    y_pred_prob = final_model.predict_proba(X_te)[:, 1]
    y_pred_def  = final_model.predict(X_te)

    best_thr   = float(np.mean(cv_thr_list))
    y_pred_opt = (y_pred_prob >= best_thr).astype(int)

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
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    test_f1_default = f1_score(y_te, y_pred_def, zero_division=0)
    f1_gap   = abs(cv_f1_arr.mean() - test_f1_default)
    gap_flag = "🚨 OVERFITTING?" if f1_gap > 0.15 else "✅ Tutarlı"

    print(f"   CV  F1     : {cv_f1_arr.mean():.3f} ± {cv_f1_arr.std():.3f}")
    print(f"   CV  Recall : {cv_rec_arr.mean():.3f} ± {cv_rec_arr.std():.3f}")
    print(f"   Test F1    : {test_f1:.3f}  |  Test Recall: {test_recall:.3f}  "
          f"|  AUC: {test_auc:.3f}")

    clinical_ready = test_recall >= RECALL_THRESHOLD
    status = "✅ Recall hedefi karşılandı!" if clinical_ready else \
             f"⚠️  Recall {test_recall:.2f} < {RECALL_THRESHOLD} — kliniğe hazır DEĞİL"
    print(f"   {status}")

    # SHAP ANALİZİ
    print("   🧠 SHAP Analizi Yapılıyor...")
    explainer   = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_te)

    shap_sum = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'Gen_Adi'    : X_te.columns,
        'SHAP_Onemi' : shap_sum
    }).sort_values(by='SHAP_Onemi', ascending=False)

    print(f"   🧬 SHAP'a Göre En Önemli 5 Gen:")
    for _, row in importance_df.head(5).iterrows():
        print(f"      {row['Gen_Adi']:<40} (Etki: {row['SHAP_Onemi']:.4f})")

    safe_name = re.sub(r"[^\w\-]", "_", antibiotic_name.lower())

    # DÜZELTME 7: explainer model dosyasına kaydedilmiyor.
    # TreeExplainer büyük modellerde joblib ile güvenilir serialize edilemiyor.
    # Gerekirse explainer çalışma zamanında modelden yeniden oluşturulmalı.
    model_path = os.path.join(MODELS_DIR, f"{safe_name}_v18.pkl")
    joblib.dump({
        "model"      : final_model,
        "threshold"  : best_thr,
        "train_cols" : X_tr.columns.tolist(),
        # "explainer" : explainer  ← KALDIRILDI (serileştirme güvenilirliği)
    }, model_path)

    shap_path = os.path.join(SHAP_DIR, f"{safe_name}_shap_importance.csv")
    importance_df.to_csv(shap_path, index=False)

    print(f"   💾 Model kaydedildi → {model_path}")
    print(f"   💾 SHAP önem tablosu kaydedildi → {shap_path}")

    return {
        "Antibiyotik"  : antibiotic_name.upper(),
        "N (toplam)"   : len(y_all),
        "N (dirençli)" : int(n_resistant),
        "Optimum_Thr"  : f"{best_thr:.3f}",
        "Test F1"      : f"{test_f1:.3f}",
        "Test Recall"  : f"{test_recall:.3f}",
        "AUC-ROC"      : f"{test_auc:.3f}",
        "TP / FN"      : f"{tp} / {fn}",
        "Tutarlılık"   : gap_flag,
        "Klinik Hazır" : "✅" if clinical_ready else "❌"
    }


# ── 4. Ana Fonksiyon ──────────────────────────────────────────────────────────
def run_superbug_panel():
    print("🏥 V18.1 KLİNİK SÜPER BAKTERİ PANELİ (SHAP + OPTUNA HAFIZASI)\n")

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
    print("⚙️  MODELLER EĞİTİLİYOR (YAPAY ZEKA + SHAP AÇIKLAYICILARI YÜKLENİYOR)...")
    print("=" * 75)

    results = []
    for anti in top_antibiotics:
        anti_df = final_df.dropna(subset=[anti])
        if len(anti_df) < 50:
            continue

        X_gene = anti_df[gene_cols]
        y_all  = anti_df[anti].astype(int)

        # DÜZELTME 3 ile uyumlu: gene_cols artık fonksiyona geçilmiyor
        row = train_and_evaluate(X_gene, y_all, anti)
        if row:
            results.append(row)

    if not results:
        print("\n⚠️  Hiçbir antibiyotik için model eğitilemedi.")
        return

    print("\n" + "=" * 80)
    print("         V18.1 FİNAL PERFORMANS TABLOSU (OPTUNA + SHAP)")
    print("=" * 80)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    report_path = os.path.join(REPORTS_DIR, "v18_final_results.csv")
    results_df.to_csv(report_path, index=False)
    print(f"\n📊 Sonuç tablosu kaydedildi → {report_path}")


if __name__ == "__main__":
    run_superbug_panel()