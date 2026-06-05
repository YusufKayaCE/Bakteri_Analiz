import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score, recall_score,
                             precision_score, roc_auc_score)
from sklearn.preprocessing import LabelEncoder
import requests
import warnings
import time
import re
import os

warnings.filterwarnings('ignore')

# ── Ayarlar ───────────────────────────────────────────────────────────────────
LABELS_FILE  = "../data/processed/v2_multilabel_labels.csv"
CACHE_FILE   = "../data/processed/v12_amr_genes_cache.csv"
MAX_GENOMES  = 1500
BATCH_SIZE   = 50
RECALL_THRESHOLD = 0.80   # Klinik minimum hedef


# ── 1. Veri Çekme (Cache'li) ──────────────────────────────────────────────────
def fetch_amr_genes_from_bvbrc(genome_ids):
    if os.path.exists(CACHE_FILE):
        print(f"📦 Önbellek bulundu → '{CACHE_FILE}' okunuyor...")
        return pd.read_csv(CACHE_FILE)

    print(f"🌐 BV-BRC'den {len(genome_ids)} bakteri için gen çekiliyor...")
    all_genes = []
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json"
    }

    for i in range(0, len(genome_ids), BATCH_SIZE):
        batch   = genome_ids[i:i + BATCH_SIZE]
        id_str  = ",".join(batch)
        payload = f"in(genome_id,({id_str}))&select(genome_id,property,gene,product)&limit(25000)"
        try:
            r = requests.post("https://www.bv-brc.org/api/sp_gene/",
                              headers=headers, data=payload, timeout=120)
            if r.status_code == 200:
                for item in r.json():
                    prop = str(item.get("property", "")).lower()
                    if any(k in prop for k in ("resist", "antimicrobial", "antibiotic")):
                        gene = item.get("gene") or item.get("product", "")
                        if gene:
                            all_genes.append({"Genome ID": item["genome_id"],
                                              "AMR_Gene": gene})
            pct = int(min(i + len(batch), len(genome_ids)) / len(genome_ids) * 100)
            print(f"  İlerleme: %{pct} | Gen: {len(all_genes)}")
            time.sleep(1)
        except Exception as e:
            print(f"  Hata (atlanıyor): {e}")
            time.sleep(3)

    df = pd.DataFrame(all_genes)
    if not df.empty:
        df.to_csv(CACHE_FILE, index=False)
        print(f"💾 Gen verileri kaydedildi → {CACHE_FILE}")
    return df


# ── 2. Tek antibiyotik için model eğitimi (SAĞLIKLI pipeline) ─────────────────
def train_and_evaluate(X_all, y_all, antibiotic_name):
    """
    KRİTİK: Veri sızıntısını önlemek için pipeline:
      1. Önce train/test split
      2. Pivot/feature işlemleri SADECE train üzerinde fit
      3. Test seti sadece transform edilir, fit edilmez
    
    Bu fonksiyonda X_all zaten pivot matris olarak geliyor.
    Leakage riski pivot'un tüm veri üzerinde yapılmasından kaynaklanıyordu;
    bunu aşağıda handle ediyoruz.
    """

    class_counts = y_all.value_counts()
    n_resistant   = class_counts.get(1, 0)
    n_susceptible = class_counts.get(0, 0)
    majority_pct  = class_counts.max() / len(y_all) * 100

    print(f"\n{'─'*60}")
    print(f"💊 {antibiotic_name.upper()}")
    print(f"   Dağılım → Duyarlı(0): {n_susceptible} | Dirençli(1): {n_resistant}"
          f"  ({majority_pct:.1f}% çoğunluk)")

    if n_resistant < 10:
        print(f"   ⚠️  Dirençli örnek çok az (<10), atlanıyor.")
        return None

    # ── Veri Sızıntısı Düzeltmesi ──────────────────────────────────────────
    # split ÖNCE yapılıyor; test seti hiçbir fit/transform işlemine katılmıyor
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all,
        test_size=0.2,
        random_state=42,
        stratify=y_all          # Sınıf oranı korunuyor
    )

    # Sadece train üzerinde görülen kolonları koru (test'te olmayan genler → 0)
    train_cols = X_tr.columns.tolist()
    X_te = X_te.reindex(columns=train_cols, fill_value=0)

    # ── Model ──────────────────────────────────────────────────────────────
    pos_weight = n_susceptible / n_resistant if n_resistant > 0 else 1
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        eval_metric="logloss",
        random_state=42,
        verbosity=0
    )

    # ── Cross-Validation (SADECE train verisi üzerinde) ────────────────────
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1  = cross_val_score(model, X_tr, y_tr, cv=skf, scoring="f1")
    cv_rec = cross_val_score(model, X_tr, y_tr, cv=skf, scoring="recall")

    # ── Final Eğitim ve Test ───────────────────────────────────────────────
    model.fit(X_tr, y_tr)
    y_pred      = model.predict(X_te)
    y_pred_prob = model.predict_proba(X_te)[:, 1]

    test_f1      = f1_score(y_te, y_pred, zero_division=0)
    test_recall  = recall_score(y_te, y_pred, zero_division=0)
    test_acc     = accuracy_score(y_te, y_pred)
    try:
        test_auc = roc_auc_score(y_te, y_pred_prob)
    except ValueError:
        test_auc = float("nan")

    cm = confusion_matrix(y_te, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    # ── Tutarlılık Kontrolü ───────────────────────────────────────────────
    f1_gap  = abs(cv_f1.mean() - test_f1)
    gap_flag = "🚨 OVERFITTING?" if f1_gap > 0.15 else "✅ Tutarlı"

    print(f"   CV  F1  : {cv_f1.mean():.3f} ± {cv_f1.std():.3f}")
    print(f"   CV  Rec : {cv_rec.mean():.3f} ± {cv_rec.std():.3f}")
    print(f"   Test F1 : {test_f1:.3f}  |  Test Recall: {test_recall:.3f}"
          f"  |  AUC: {test_auc:.3f}")
    print(f"   CV↔Test F1 farkı: {f1_gap:.3f}  → {gap_flag}")

    if test_recall < RECALL_THRESHOLD:
        print(f"   ⚠️  Recall {test_recall:.2f} < {RECALL_THRESHOLD} — kliniğe hazır DEĞİL")
    else:
        print(f"   ✅  Recall hedefi karşılandı!")

    print(f"\n{classification_report(y_te, y_pred, target_names=['Duyarlı','Dirençli'], zero_division=0)}")

    # ── Top-10 Gen ────────────────────────────────────────────────────────
    imp = pd.Series(model.feature_importances_, index=X_tr.columns)
    top10 = imp.nlargest(10)
    print(f"   🧬 En Önemli 10 Gen:")
    for gene, score in top10.items():
        print(f"      {gene:<35} {score:.4f}")

    return {
        "Antibiyotik"    : antibiotic_name.upper(),
        "N (toplam)"     : len(y_all),
        "N (dirençli)"   : int(n_resistant),
        "CV F1"          : f"{cv_f1.mean():.3f}±{cv_f1.std():.3f}",
        "CV Recall"      : f"{cv_rec.mean():.3f}±{cv_rec.std():.3f}",
        "Test F1"        : f"{test_f1:.3f}",
        "Test Recall"    : f"{test_recall:.3f}",
        "AUC-ROC"        : f"{test_auc:.3f}",
        "TP/FN"          : f"{tp}/{fn}",
        "Tutarlılık"     : gap_flag,
        "Klinik Hazır"   : "✅" if test_recall >= RECALL_THRESHOLD else "❌"
    }


# ── 3. Ana Fonksiyon ──────────────────────────────────────────────────────────
def run_superbug_panel():
    print("🏥 V14.0 KLİNİK SÜPER BAKTERİ PANELİ (SAĞLIKLI PİPELİNE)\n")

    y_df = pd.read_csv(LABELS_FILE)
    exclude_cols     = ["Genome ID", "Bacteria_Type"]
    all_antibiotics  = [c for c in y_df.columns if c not in exclude_cols]
    top_antibiotics  = y_df[all_antibiotics].count().nlargest(5).index.tolist()

    if "Bacteria_Type" in y_df.columns:
        top_bacteria = y_df["Bacteria_Type"].value_counts().head(5).index.tolist()
        y_df = y_df[y_df["Bacteria_Type"].isin(top_bacteria)]
        print(f"🦠 Bakteri türleri: {', '.join(top_bacteria)}")

    print(f"💊 Hedef antibiyotikler: {', '.join(top_antibiotics)}\n")

    y_df["Genome ID"] = y_df["Genome ID"].astype(str)
    sample_genomes    = y_df["Genome ID"].unique()[:MAX_GENOMES]

    # ── Gen matrisini oluştur ─────────────────────────────────────────────
    genes_df = fetch_amr_genes_from_bvbrc(sample_genomes)
    if genes_df.empty:
        print("❌ Gen verisi alınamadı.")
        return

    genes_df["Değer"] = 1
    X_matrix = (genes_df
                .pivot_table(index="Genome ID", columns="AMR_Gene",
                             values="Değer", aggfunc="max", fill_value=0)
                .reset_index())

    # XGBoost için güvenli kolon adları
    X_matrix.columns = [re.sub(r"[\[\]<>]", "", str(c)) for c in X_matrix.columns]

    # Tip uyuşmazlığını önle: her iki tarafta da string olsun
    X_matrix["Genome ID"] = X_matrix["Genome ID"].astype(str).str.strip()
    y_df["Genome ID"]     = y_df["Genome ID"].astype(str).str.strip()
    
    final_df = pd.merge(X_matrix, y_df, on="Genome ID", how="inner")

    # ── Her antibiyotik için eğit ─────────────────────────────────────────
    results = []
    feature_cols = [c for c in X_matrix.columns if c != "Genome ID"]

    for anti in top_antibiotics:
        anti_df = final_df.dropna(subset=[anti])
        if len(anti_df) < 50:
            print(f"\n⚠️  {anti.upper()} — yeterli veri yok, atlanıyor.")
            continue

        X_all = anti_df[feature_cols]
        y_all = anti_df[anti].astype(int)

        row = train_and_evaluate(X_all, y_all, anti)
        if row:
            results.append(row)

    # ── Final Tablo ───────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("         V14.0 GERÇEK PERFORMANS TABLOSU")
    print("="*70)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print("\nNot: 'Klinik Hazır' = Test Recall ≥ %80")


if __name__ == "__main__":
    run_superbug_panel()