import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import requests
import warnings
import time
import re
import os
import optuna  # 🌟 V16 YENİLİĞİ: Hiperparametre Avcısı

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score, recall_score,
                             precision_score, roc_auc_score,
                             precision_recall_curve)

warnings.filterwarnings('ignore')
# Optuna'nın terminali gereksiz mesajlarla boğmasını engelliyoruz
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Ayarlar ───────────────────────────────────────────────────────────────────
LABELS_FILE       = "../data/processed/v2_multilabel_labels.csv"
CACHE_FILE        = "../data/processed/v15_amr_genes_cache.csv" # V16 Cache
MODELS_DIR        = "../models"
REPORTS_DIR       = "../reports"
MAX_GENOMES       = 30000
BATCH_SIZE        = 100
RECALL_THRESHOLD  = 0.80  # Klinik minimum hedef
OPTUNA_TRIALS     = 25    # Optuna'nın deneyeceği farklı model sayısı

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


# ── 1. Veri Çekme (Cache Bütünlük Kontrollü + Artımlı) ───────────────────────
def fetch_amr_genes_from_bvbrc(genome_ids):
    genome_ids = [str(g) for g in genome_ids]

    if os.path.exists(CACHE_FILE):
        cached = pd.read_csv(CACHE_FILE)
        cached["Genome ID"] = cached["Genome ID"].astype(str)
        cached_genomes = set(cached["Genome ID"].unique())
        missing = [g for g in genome_ids if g not in cached_genomes]

        if len(missing) == 0:
            print(f"📦 Cache TAM ({len(cached_genomes)} genom) → doğrudan okunuyor.")
            return cached
        else:
            print(f"⚠️  Cache eksik! {len(missing)} genom eksik → API'den tamamlanıyor...")
            new_df = _fetch_from_api(missing)
            if not new_df.empty:
                combined = pd.concat([cached, new_df], ignore_index=True)
                combined.to_csv(CACHE_FILE, index=False)
                print(f"💾 Cache güncellendi → {CACHE_FILE}")
                return combined
            return cached

    print(f"🌐 Cache bulunamadı. {len(genome_ids)} bakteri için API'den çekiliyor...")
    df = _fetch_from_api(genome_ids)
    if not df.empty:
        df.to_csv(CACHE_FILE, index=False)
        print(f"🎉 Tüm gen verileri kaydedildi → {CACHE_FILE}")
    return df


def _fetch_from_api(genome_ids):
    all_genes = []
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json"
    }

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
                            all_genes.append({
                                "Genome ID": str(item["genome_id"]),
                                "AMR_Gene" : gene
                            })

            pct = int(min(i + len(batch), len(genome_ids)) / len(genome_ids) * 100)
            print(f"  %{pct:3d} ({min(i+len(batch), len(genome_ids))}"
                  f"/{len(genome_ids)}) | Gen: {len(all_genes)}")

            if (i + len(batch)) % 1000 == 0 and all_genes:
                pd.DataFrame(all_genes).to_csv(CACHE_FILE, index=False)
                print(f"  💾 Oto-kayıt yapıldı.")

            time.sleep(1)

        except Exception as e:
            print(f"  ❌ Hata (atlanıyor): {e}")
            time.sleep(3)

    return pd.DataFrame(all_genes)


# ── 2. Optuna Objective Fonksiyonu (Parametre Arama) ──────────────────────────
def optimize_hyperparameters(X_train, y_train, pos_weight):
    """Optuna için özel arama uzayı."""
    
    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 400, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'gamma': trial.suggest_float('gamma', 1e-4, 1.0, log=True),
            'scale_pos_weight': pos_weight,
            'eval_metric': 'logloss',
            'random_state': 42,
            'verbosity': 0,
            'n_jobs': -1 # Tüm işlemci çekirdeklerini kullan (Hızlandırır)
        }
        
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        model = xgb.XGBClassifier(**param)
        
        # PR-AUC (Average Precision) hedefleniyor. Dengesiz veride Recall'u artırmanın en iyi yoludur.
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='average_precision', n_jobs=-1)
        return scores.mean()

    print(f"   🔍 Optuna: {OPTUNA_TRIALS} farklı hiperparametre kombinasyonu deneniyor...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=OPTUNA_TRIALS)
    
    best_params = study.best_params
    best_params['scale_pos_weight'] = pos_weight
    best_params['eval_metric'] = 'logloss'
    best_params['random_state'] = 42
    best_params['verbosity'] = 0
    best_params['n_jobs'] = -1
    
    print(f"   🎯 En İyi Parametreler Bulundu: {study.best_params}")
    return best_params


# ── 3. Model Eğitimi (Optuna + Sızıntısız Threshold Optimizasyonu) ────────────
def train_and_evaluate(X_gene, y_all, antibiotic_name):
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

    # ÖNCE BÖL (Sıfır Sızıntı Kuralı)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_gene, y_all,
        test_size=0.2,
        random_state=42,
        stratify=y_all
    )

    X_te = X_te.reindex(columns=X_tr.columns, fill_value=0)
    pos_weight = n_susceptible / n_resistant if n_resistant > 0 else 1

    # 🌟 V16: Sabit ayarlar yerine Optuna ile en iyi ayarları buluyoruz!
    best_xgb_params = optimize_hyperparameters(X_tr, y_tr, pos_weight)
    
    # ── Cross-Validation: Fold-Bazlı Threshold Optimizasyonu ──
    skf         = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1_list  = []
    cv_rec_list = []
    cv_thr_list = []

    for fold_tr_idx, fold_val_idx in skf.split(X_tr, y_tr):
        X_cv_tr  = X_tr.iloc[fold_tr_idx]
        X_cv_val = X_tr.iloc[fold_val_idx]
        y_cv_tr  = y_tr.iloc[fold_tr_idx]
        y_cv_val = y_tr.iloc[fold_val_idx]

        # Optuna'nın bulduğu zırhlı parametrelerle fold modelini kur
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
            fold_thr = t_fold[best_fold_idx]
        else:
            fold_thr = 0.5

        y_cv_pred = (y_cv_prob >= fold_thr).astype(int)
        cv_f1_list.append(f1_score(y_cv_val, y_cv_pred, zero_division=0))
        cv_rec_list.append(recall_score(y_cv_val, y_cv_pred, zero_division=0))
        cv_thr_list.append(fold_thr)

    cv_f1_arr  = np.array(cv_f1_list)
    cv_rec_arr = np.array(cv_rec_list)

    # ── Final Eğitim (En iyi parametreler + Tüm Train) ───────────────────────
    final_model = xgb.XGBClassifier(**best_xgb_params)
    final_model.fit(X_tr, y_tr)
    y_pred_prob = final_model.predict_proba(X_te)[:, 1]
    y_pred_def  = final_model.predict(X_te) 

    # Threshold Ortalaması
    best_thr   = float(np.mean(cv_thr_list))
    y_pred_opt = (y_pred_prob >= best_thr).astype(int)
    thr_note   = f"{best_thr:.3f} (CV fold ortalaması)"

    # ── Metrikler ve Overfitting Kontrolü ────────────────────────────────────
    test_f1     = f1_score(y_te, y_pred_opt, zero_division=0)
    test_recall = recall_score(y_te, y_pred_opt, zero_division=0)
    test_acc    = accuracy_score(y_te, y_pred_opt)
    try:
        test_auc = roc_auc_score(y_te, y_pred_prob)
    except ValueError:
        test_auc = float("nan")

    cm = confusion_matrix(y_te, y_pred_opt)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    # Adil Overfitting Kontrolü
    test_f1_default = f1_score(y_te, y_pred_def, zero_division=0)
    f1_gap   = abs(cv_f1_arr.mean() - test_f1_default)
    gap_flag = "🚨 OVERFITTING?" if f1_gap > 0.15 else "✅ Tutarlı"

    # ── Çıktılar ─────────────────────────────────────────────────────────
    print(f"   CV  F1     : {cv_f1_arr.mean():.3f} ± {cv_f1_arr.std():.3f}  (optimize threshold ile)")
    print(f"   CV  Recall : {cv_rec_arr.mean():.3f} ± {cv_rec_arr.std():.3f}")
    print(f"   Test F1    : {test_f1:.3f}  |  "
          f"Test Recall: {test_recall:.3f}  |  AUC: {test_auc:.3f}")
    print(f"   CV↔Test F1 farkı  : {f1_gap:.3f}  → {gap_flag}")

    clinical_ready = test_recall >= RECALL_THRESHOLD
    print(f"   {'✅ Recall hedefi karşılandı!' if clinical_ready else f'⚠️  Recall {test_recall:.2f} < {RECALL_THRESHOLD} — kliniğe hazır DEĞİL'}")

    imp   = pd.Series(final_model.feature_importances_, index=X_tr.columns)
    top10 = imp.nlargest(10)
    print(f"   🧬 En Önemli 10 Gen:")
    for gene, score in top10.items():
        print(f"      {gene:<40} {score:.4f}")

    # ── Model Kaydet ──────────────────────────────────────────────────────
    model_path = os.path.join(MODELS_DIR, f"{antibiotic_name.lower()}_v16.pkl")
    joblib.dump({"model": final_model, "threshold": best_thr, "train_cols": X_tr.columns.tolist()},
                model_path)
    print(f"\n   💾 Model kaydedildi → {model_path}")

    return {
        "Antibiyotik"   : antibiotic_name.upper(),
        "N (toplam)"    : len(y_all),
        "N (dirençli)"  : int(n_resistant),
        "Optimum_Thr"   : f"{best_thr:.3f}",
        "Test F1"       : f"{test_f1:.3f}",
        "Test Recall"   : f"{test_recall:.3f}",
        "AUC-ROC"       : f"{test_auc:.3f}",
        "TP / FN"       : f"{tp} / {fn}",
        "Tutarlılık"    : gap_flag,
        "Klinik Hazır"  : "✅" if clinical_ready else "❌"
    }


# ── 4. Ana Fonksiyon ──────────────────────────────────────────────────────────
def run_superbug_panel():
    print("🏥 V16.0 KLİNİK SÜPER BAKTERİ PANELİ (OPTUNA HİPERPARAMETRE AVCILIĞI)\n")

    y_df = pd.read_csv(LABELS_FILE)
    y_df["Genome ID"] = y_df["Genome ID"].astype(str).str.strip()

    exclude_cols    = ["Genome ID", "Bacteria_Type"]
    all_antibiotics = [c for c in y_df.columns if c not in exclude_cols]
    top_antibiotics = y_df[all_antibiotics].count().nlargest(5).index.tolist()

    if "Bacteria_Type" in y_df.columns:
        top_bacteria = y_df["Bacteria_Type"].value_counts().head(5).index.tolist()
        y_df = y_df[y_df["Bacteria_Type"].isin(top_bacteria)]
        print(f"🦠 Bakteri türleri : {', '.join(top_bacteria)}")

    print(f"💊 Hedef antibiyotikler: {', '.join(top_antibiotics)}\n")

    sample_genomes = y_df["Genome ID"].unique()[:MAX_GENOMES]

    genes_df = fetch_amr_genes_from_bvbrc(sample_genomes)
    if genes_df.empty: return

    genes_df["Genome ID"] = genes_df["Genome ID"].astype(str).str.strip()
    genes_df["Değer"]     = 1

    X_matrix = (genes_df
                .pivot_table(index="Genome ID", columns="AMR_Gene",
                             values="Değer", aggfunc="max", fill_value=0)
                .reset_index())

    X_matrix.columns = [re.sub(r"[\[\]<>]", "", str(c)) for c in X_matrix.columns]
    X_matrix["Genome ID"] = X_matrix["Genome ID"].astype(str).str.strip()

    gene_cols = [c for c in X_matrix.columns if c != "Genome ID"]

    final_df = pd.merge(X_matrix, y_df, on="Genome ID", how="inner")
    print(f"\n✅ Birleştirme tamamlandı: {len(final_df)} ortak genom bulundu.\n")

    print("=" * 75)
    print("⚙️  MODELLER EĞİTİLİYOR (YAPAY ZEKA, YAPAY ZEKAYI EĞİTİYOR)...")
    print("=" * 75)

    results = []
    for anti in top_antibiotics:
        anti_df = final_df.dropna(subset=[anti])
        if len(anti_df) < 50: continue

        X_gene = anti_df[gene_cols]
        y_all  = anti_df[anti].astype(int)

        row = train_and_evaluate(X_gene, y_all, anti)
        if row:
            results.append(row)

    print("\n" + "=" * 80)
    print("         V16.0 FİNAL PERFORMANS TABLOSU (OPTUNA OPTİMİZE)")
    print("=" * 80)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    report_path = os.path.join(REPORTS_DIR, "v16_final_results.csv")
    results_df.to_csv(report_path, index=False)
    print(f"\n📊 Sonuç tablosu kaydedildi → {report_path}")

if __name__ == "__main__":
    run_superbug_panel()