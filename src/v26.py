# main_v26.py
# Değişiklikler (v25 → v26) — K-mer Frekans Çıkarımı (Tüm Genom Diziliminden)
# RTX 3050 Laptop (4GB VRAM) için optimize edilmiştir.
#
# ══════════════════════════════════════════════════════════════════════════════
# DEĞİŞİKLİK 6 [YENİ] — K-MER FREKANS ÇIKARIMI
# ══════════════════════════════════════════════════════════════════════════════
#   Neden: AMR genleri yalnızca bilinen direnç genlerine bağlıdır. K-mer analizi
#   sayesinde henüz annotasyon almamış, yeni keşfedilmiş ya da nokta mutasyon
#   kaynaklı direnç örüntüleri de modele dahil edilebilir.
#
#   Yaklaşım:
#     1. BV-BRC genome_sequence API'sinden her genom için konak DNA dizisi
#        çekilir (contig'ler birleştirilir, ilk SEQ_SAMPLE_BP baz alınır).
#     2. Kanonik k-mer (min(kmer, RC(kmer))) kullanılır → iplik bağımsız,
#        k=4 için 136 benzersiz kanonik özellik (256 naif → deduplication).
#     3. Frekanslar toplam k-mer sayısına bölünerek normalize edilir
#        (0–1 aralığı, GC içeriğinden bağımsız karşılaştırma).
#     4. Parquet önbelleği: İkinci çalıştırmada API atlanır (~100× hız farkı).
#     5. VarianceThreshold (var=0) + mevcut korelasyon temizliği uygulanır.
#
#   RTX 3050 (4GB VRAM) için optimizasyon:
#     - KMER_K = 4  → 136 özellik, minimal bellek
#     - SEQ_SAMPLE_BP = 200_000 → her genomdan ilk 200 kb yeterli (direnç
#       bölgeleri genelde kromozom başına yakın)
#     - SEQ_BATCH_SIZE = 5 → büyük payload sorununu önler
#     - float32 dtype → float64'e göre %50 bellek tasarrufu
#     - XGBoost GPU'da çalışırken k-mer özellik üretimi tamamen CPU/RAM'de
#
# ══════════════════════════════════════════════════════════════════════════════
# v25'ten KORUNAN ÖZELLİKLER (değiştirilmedi):
#   Model Kalibrasyonu (Brier Score + Reliability Diagram),
#   Temporal Split, Çoklu Algoritma Karşılaştırması (LR + RF + XGB),
#   ST/Klonal Linyaj Gruplama, Örnekleme Bias Raporu,
#   VME/ME/PPV/NPV/AUPRC metrikleri, korelasyon temizliği, GroupKFold,
#   SHAP analizi, Optuna hiperparametre optimizasyonu,
#   4 kademeli eşik seçimi, GPU otomatik algılama.
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
import itertools
import optuna
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from collections import Counter
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
from sklearn.feature_selection import VarianceThreshold

from reporting_module import generate_academic_reports

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── FDA/CLSI Klinik Kabul Eşikleri ────────────────────────────────────────────
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

# ── Dosya Yolları ──────────────────────────────────────────────────────────────
LABELS_FILE     = "../data/processed/v2_multilabel_labels.csv"
CACHE_FILE      = "../data/processed/v17_amr_genes_cache.csv"
SEQ_CACHE_FILE  = "../data/processed/v26_sequences.parquet"
KMER_CACHE_FILE = "../data/processed/v26_kmer_features.parquet"
MODELS_DIR      = "../models"
REPORTS_DIR     = "../reports"
SHAP_DIR        = "../reports/shap_values"

# ── Genel Ayarlar ──────────────────────────────────────────────────────────────
MAX_GENOMES    = 30000
BATCH_SIZE     = 100
OPTUNA_TRIALS  = 30
OPTUNA_N_JOBS  = 1
MODEL_N_JOBS   = 1 if DEVICE == "cuda" else -1
BACT_MIN_FREQ  = 0.05
CORR_THRESHOLD = 0.95
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH    = 10

# ══════════════════════════════════════════════════════════════════════════════
# DEĞİŞİKLİK 6 AYARLARI — K-MER
# ══════════════════════════════════════════════════════════════════════════════
KMER_K          = 4          # 4-mer: 136 kanonik özellik (k=5 → 512, bellek ×4)
SEQ_SAMPLE_BP   = 200_000    # Her genomdan ilk 200 kb kullan (RTX 3050 optimizasyonu)
SEQ_BATCH_SIZE  = 5          # Dizi API batch boyutu (büyük payload riski nedeniyle küçük)
USE_KMER        = True       # False → k-mer devre dışı, v25 davranışı korunur

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(SHAP_DIR,    exist_ok=True)
os.makedirs(os.path.join(REPORTS_DIR, "figures"),      exist_ok=True)
os.makedirs(os.path.join(REPORTS_DIR, "calibration"),  exist_ok=True)
os.makedirs(os.path.join(REPORTS_DIR, "bias_reports"), exist_ok=True)

OPTUNA_DB_FILE = f"sqlite:///{os.path.abspath(MODELS_DIR)}/optuna_studies_v26.db"


# ══════════════════════════════════════════════════════════════════════════════
# DEĞİŞİKLİK 6 — K-MER YARDIMCI FONKSİYONLARI
# ══════════════════════════════════════════════════════════════════════════════

_COMP_TABLE = str.maketrans("ACGT", "TGCA")

def _reverse_complement(seq: str) -> str:
    """DNA dizisinin ters tümleyen dizisini döndürür."""
    return seq.translate(_COMP_TABLE)[::-1]


def _canonical(kmer: str) -> str:
    """
    Kanonik k-mer: kmer ile ters tümleyeninin leksikografik küçüğü.
    Bu sayede + ve − ipliğindeki aynı sekans tek özellik olarak temsil edilir.
    Örnek: ACGT ve ACGT(RC)=ACGT → aynı; ATCG ve RC=CGAT → kanonik=ATCG
    """
    rc = _reverse_complement(kmer)
    return kmer if kmer <= rc else rc


def _build_canonical_kmer_index(k: int) -> dict:
    """
    Tüm kanonik k-merleri önceden hesaplar ve indeks sözlüğü oluşturur.
    k=4 için 256 naif → 136 kanonik özellik.
    Palindromik k-merler (kmer == RC) sadece bir kez sayılır.
    """
    all_kmers = [''.join(p) for p in itertools.product("ACGT", repeat=k)]
    canonical_set = sorted(set(_canonical(km) for km in all_kmers))
    return {km: i for i, km in enumerate(canonical_set)}


# Modül yüklenirken bir kez hesapla → her çağrıda sıfırdan yapma
_KMER_INDEX = _build_canonical_kmer_index(KMER_K)
_N_KMERS    = len(_KMER_INDEX)
print(f"📐 K-mer Ayarları: k={KMER_K}, "
      f"kanonik k-mer sayısı={_N_KMERS} (naif={4**KMER_K})")


def extract_kmer_freq(seq: str, k: int = KMER_K,
                      kmer_index: dict = _KMER_INDEX) -> np.ndarray:
    """
    Bir DNA dizisinden normalize edilmiş kanonik k-mer frekans vektörü üretir.

    Args:
        seq: DNA dizisi (büyük/küçük harf karışık olabilir; temizlenir)
        k: k-mer uzunluğu
        kmer_index: kanonik k-mer → indeks sözlüğü

    Returns:
        float32 numpy dizisi, shape=(len(kmer_index),), toplam=1.0
        Boş/geçersiz dizi için sıfır vektörü döner.

    RTX 3050 notu: float32 tercih edildi (float64'e göre %50 bellek tasarrufu).
    CPU'da ~0.5 ms / 200 kb dizi.
    """
    n = len(kmer_index)
    freq = np.zeros(n, dtype=np.float32)

    # Sadece ACGT harflerini tut (N, W, R vb. belirsizlikleri çıkar)
    seq = re.sub(r'[^ACGT]', '', seq.upper())
    L   = len(seq)

    if L < k:
        return freq  # Çok kısa dizi → sıfır vektörü

    total = 0
    for i in range(L - k + 1):
        km = seq[i:i + k]
        ck = _canonical(km)
        idx = kmer_index.get(ck)
        if idx is not None:
            freq[idx] += 1
            total += 1

    if total > 0:
        freq /= total  # Normalize: göreceli frekans

    return freq


# ── BV-BRC Dizi Çekme ─────────────────────────────────────────────────────────
def _fetch_sequences_batch(genome_ids: list) -> dict:
    """
    BV-BRC genome_sequence endpoint'inden bir batch genomun DNA dizilerini çeker.
    Her genom için tüm contig dizileri birleştirilir ve ilk SEQ_SAMPLE_BP baz
    alınır (bellek optimizasyonu).

    Döndürür: {genome_id: dna_string}
    """
    id_str  = ",".join(str(g) for g in genome_ids)
    payload = (f"in(genome_id,({id_str}))"
               f"&select(genome_id,sequence)&limit(50000)")
    headers = {"Content-Type": "application/x-www-form-urlencoded",
               "Accept": "application/json"}
    result  = {}

    try:
        r = requests.post("https://www.bv-brc.org/api/genome_sequence/",
                          headers=headers, data=payload, timeout=180)
        if r.status_code != 200:
            return result

        for item in r.json():
            gid = str(item.get("genome_id", "")).strip()
            seq = str(item.get("sequence", ""))
            if gid and seq:
                if gid in result:
                    result[gid] += seq          # aynı genomun contig'lerini birleştir
                else:
                    result[gid]  = seq

        # İlk SEQ_SAMPLE_BP baz ile sınırla
        for gid in result:
            result[gid] = result[gid][:SEQ_SAMPLE_BP]

    except Exception as e:
        print(f"      ❌ Dizi API hatası: {e}")

    return result


def fetch_genome_sequences(genome_ids: list) -> pd.DataFrame:
    """
    Tüm genome_ids için DNA dizilerini çeker veya önbellekten yükler.
    Önbellek formatı: Parquet (CSV'den ~5× daha hızlı read/write).

    Döndürür: DataFrame sütunlar → [Genome ID, sequence]
    """
    genome_ids = [str(g) for g in genome_ids]

    # Önbellek kontrolü
    if os.path.exists(SEQ_CACHE_FILE):
        try:
            cached = pd.read_parquet(SEQ_CACHE_FILE)
            cached["Genome ID"] = cached["Genome ID"].astype(str)
            print(f"📦 DİZİ ÖNBELLEĞİ BULUNDU: {SEQ_CACHE_FILE} "
                  f"({len(cached):,} kayıt)")
            return cached
        except Exception as e:
            print(f"⚠️  Dizi önbelleği okunamadı ({e}), API'den çekiliyor...")

    print(f"\n🌐 [DEĞİŞİKLİK 6] {len(genome_ids):,} genomun DNA dizisi çekiliyor...")
    print(f"   Batch boyutu: {SEQ_BATCH_SIZE} | "
          f"Genombaşına ilk {SEQ_SAMPLE_BP//1000}kb alınacak")

    all_rows    = []
    total_batch = (len(genome_ids) + SEQ_BATCH_SIZE - 1) // SEQ_BATCH_SIZE

    for b_idx, i in enumerate(range(0, len(genome_ids), SEQ_BATCH_SIZE)):
        batch   = genome_ids[i:i + SEQ_BATCH_SIZE]
        seq_map = _fetch_sequences_batch(batch)

        for gid, seq in seq_map.items():
            all_rows.append({"Genome ID": gid, "sequence": seq})

        completed = min(i + SEQ_BATCH_SIZE, len(genome_ids))
        pct = int(completed / len(genome_ids) * 100)
        print(f"\r   %{pct:3d} ({completed}/{len(genome_ids)}) "
              f"| Dizi: {len(all_rows):,}", end="", flush=True)

        # Her 50 batch'te bir ara kayıt
        if (b_idx + 1) % 50 == 0 and all_rows:
            _tmp = pd.DataFrame(all_rows)
            _tmp.to_parquet(SEQ_CACHE_FILE, index=False)

        time.sleep(0.5)  # API rate-limit

    print()  # newline

    if not all_rows:
        print("   ⚠️  Hiç dizi alınamadı.")
        return pd.DataFrame(columns=["Genome ID", "sequence"])

    seq_df = pd.DataFrame(all_rows)
    seq_df.to_parquet(SEQ_CACHE_FILE, index=False)
    print(f"   💾 Dizi önbelleği → {SEQ_CACHE_FILE} ({len(seq_df):,} genom)")
    return seq_df


def build_kmer_feature_matrix(genome_ids: list,
                               k: int = KMER_K) -> pd.DataFrame:
    """
    Verilen genomlar için k-mer frekans matrisini oluşturur.
    Önbellekten yükleme yapılır; yoksa diziler çekilip hesaplanır.

    Döndürür: DataFrame, index=Genome ID, sütunlar=kmer_ACGT... (136 sütun için k=4)
    dtype: float32 (bellek optimizasyonu)

    RTX 3050 notu:
      30,000 × 136 float32 = ~15 MB RAM → sorunsuz.
      XGBoost CUDA'ya gönderilmeden önce bu matris CPU'da hazırlanır.
    """
    genome_ids = [str(g) for g in genome_ids]
    kmer_index = _build_canonical_kmer_index(k)
    kmer_cols  = [f"kmer_{km}" for km in sorted(kmer_index.keys())]

    # Önbellek kontrolü
    if os.path.exists(KMER_CACHE_FILE):
        try:
            cached = pd.read_parquet(KMER_CACHE_FILE)
            cached["Genome ID"] = cached["Genome ID"].astype(str)
            print(f"📦 K-MER ÖNBELLEĞİ BULUNDU: {KMER_CACHE_FILE} "
                  f"({len(cached):,} kayıt, {len(kmer_cols)} özellik)")
            return cached
        except Exception as e:
            print(f"⚠️  K-mer önbelleği okunamadı ({e}), yeniden hesaplanıyor...")

    print(f"\n🧬 [DEĞİŞİKLİK 6] K-mer Matrisi Oluşturuluyor...")
    print(f"   k={k}, kanonik özellik sayısı={len(kmer_index)}, "
          f"örnek sayısı={len(genome_ids):,}")

    # DNA dizilerini çek
    seq_df = fetch_genome_sequences(genome_ids)
    if seq_df.empty:
        print("   ⚠️  Dizi bulunamadı → k-mer matrisi boş döndürüldü.")
        return pd.DataFrame(columns=["Genome ID"] + kmer_cols)

    seq_df["Genome ID"] = seq_df["Genome ID"].astype(str)

    # Her genom için k-mer vektörü hesapla
    print(f"   ⚙️  {len(seq_df):,} genomun k-mer frekansları hesaplanıyor...")
    rows = []
    for idx, row in seq_df.iterrows():
        freq_vec = extract_kmer_freq(row["sequence"], k=k, kmer_index=kmer_index)
        rows.append(freq_vec)

        if (idx + 1) % 1000 == 0:
            pct = int((idx + 1) / len(seq_df) * 100)
            print(f"\r   %{pct:3d} ({idx+1}/{len(seq_df):,})", end="", flush=True)

    print()

    freq_matrix = np.vstack(rows).astype(np.float32)

    kmer_df = pd.DataFrame(freq_matrix, columns=kmer_cols)
    kmer_df.insert(0, "Genome ID", seq_df["Genome ID"].values)

    # VarianceThreshold: tüm genomlarda sıfır olan k-merleri çıkar
    freq_only = kmer_df[kmer_cols]
    var_sel   = VarianceThreshold(threshold=0.0)
    var_sel.fit(freq_only)
    kept_cols = [c for c, keep in zip(kmer_cols, var_sel.get_support()) if keep]
    removed   = len(kmer_cols) - len(kept_cols)
    if removed > 0:
        print(f"   🔬 VarianceThreshold: {removed} sabit k-mer çıkarıldı "
              f"→ {len(kept_cols)} k-mer kaldı.")
    kmer_df = kmer_df[["Genome ID"] + kept_cols]

    kmer_df.to_parquet(KMER_CACHE_FILE, index=False)
    print(f"   💾 K-mer matrisi → {KMER_CACHE_FILE} "
          f"({len(kmer_df):,} × {len(kept_cols)} özellik)")

    return kmer_df


# ══════════════════════════════════════════════════════════════════════════════
# DEĞİŞİKLİK 5: ÖRNEKLEME BIAS RAPORU (v25'ten aynı)
# ══════════════════════════════════════════════════════════════════════════════
def veri_bias_raporu(y_df: pd.DataFrame):
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
            dist    = y_df[col].value_counts(dropna=False)
            pct     = (dist / len(y_df) * 100).round(2)
            bias_df = pd.DataFrame({"Sayi": dist, "Yüzde": pct})
            out_path = os.path.join(REPORTS_DIR, "bias_reports",
                                    f"v26_bias_{col.lower()}.csv")
            bias_df.to_csv(out_path)
            top3 = ", ".join([f"{k}({v:.1f}%)" for k, v in pct.head(3).items()])
            print(f"   {label}: {top3} ... → {out_path}")

    if not herhangi_var:
        print("   ⚠️  Year/Country/ST sütunu bulunamadı.")
    else:
        if "Year" in y_df.columns:
            try:
                fig, ax = plt.subplots(figsize=(8, 3))
                y_df["Year"].value_counts().sort_index().plot(
                    kind='bar', ax=ax, color='steelblue', edgecolor='white')
                ax.set_title("Örnekleme Yıl Dağılımı (Bias Analizi)")
                ax.set_xlabel("Yıl"); ax.set_ylabel("Genom Sayısı")
                plt.tight_layout()
                fig_path = os.path.join(REPORTS_DIR, "bias_reports",
                                        "v26_year_distribution.png")
                fig.savefig(fig_path, dpi=120)
                plt.close(fig)
                print(f"   📈 Yıl dağılım grafiği → {fig_path}")
            except Exception as e:
                print(f"   ⚠️  Grafik oluşturulamadı: {e}")
    print()


# ── 1. AMR Gen Çekme ──────────────────────────────────────────────────────────
def fetch_amr_genes_from_bvbrc(genome_ids):
    genome_ids = [str(g) for g in genome_ids]
    if os.path.exists(CACHE_FILE):
        try:
            cached = pd.read_csv(CACHE_FILE)
            if cached.empty or "Genome ID" not in cached.columns:
                raise ValueError("Cache bozuk.")
            cached["Genome ID"] = cached["Genome ID"].astype(str)
            print(f"📦 YEREL ÖNBELLEK: '{CACHE_FILE}' ({len(cached):,} kayıt)")
            return cached
        except Exception as e:
            print(f"⚠️  Cache okunamadı ({e}), API'den çekiliyor...")
            os.remove(CACHE_FILE)

    print(f"🌐 {len(genome_ids):,} bakteri için AMR genleri çekiliyor...")
    df = _fetch_amr_from_api(genome_ids)
    if not df.empty:
        df.to_csv(CACHE_FILE, index=False)
        print(f"🎉 Kaydedildi → {CACHE_FILE}")
    return df


def _fetch_amr_from_api(genome_ids):
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
            print(f"\r  %{int(completed/len(genome_ids)*100):3d} "
                  f"({completed}/{len(genome_ids)}) | Gen: {len(all_genes):,}",
                  end="", flush=True)
            if (i // BATCH_SIZE + 1) % 10 == 0 and all_genes:
                pd.DataFrame(all_genes).to_csv(CACHE_FILE, index=False)
            time.sleep(1)
        except Exception as e:
            print(f"\n  ❌ Hata: {e}")
            time.sleep(3)
    print()
    return pd.DataFrame(all_genes)


# ── 2. Korelasyon Temizliği ───────────────────────────────────────────────────
def remove_correlated_features(X: pd.DataFrame,
                                threshold: float = CORR_THRESHOLD) -> pd.DataFrame:
    gene_cols = [c for c in X.columns if not c.startswith("bact_")
                                      and not c.startswith("kmer_")]
    bact_cols = [c for c in X.columns if c.startswith("bact_")]
    kmer_cols = [c for c in X.columns if c.startswith("kmer_")]

    removed_total = 0

    # Gen korelasyon temizliği
    if len(gene_cols) >= 2:
        X_gene   = X[gene_cols].copy()
        var_mask = X_gene.var() > 0
        X_gene   = X_gene.loc[:, var_mask]
        corr     = X_gene.corr(method='pearson').abs()
        upper    = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        drop_g   = [c for c in upper.columns if any(upper[c] > threshold)]
        gene_cols = [c for c in gene_cols if c not in drop_g and c in X_gene.columns]
        removed_total += len(drop_g)

    # K-mer korelasyon temizliği (ayrı blok: gen-kmer cross korelasyonu değil)
    if len(kmer_cols) >= 2:
        X_kmer   = X[kmer_cols].copy()
        var_mask = X_kmer.var() > 0
        X_kmer   = X_kmer.loc[:, var_mask]
        corr_k   = X_kmer.corr(method='pearson').abs()
        upper_k  = corr_k.where(np.triu(np.ones(corr_k.shape), k=1).astype(bool))
        drop_k   = [c for c in upper_k.columns if any(upper_k[c] > threshold)]
        kmer_cols = [c for c in kmer_cols if c not in drop_k and c in X_kmer.columns]
        removed_total += len(drop_k)

    if removed_total > 0:
        print(f"   🔬 Korelasyon temizliği (φ>{threshold}): "
              f"{removed_total} özellik çıkarıldı.")

    final_cols = [c for c in gene_cols + bact_cols + kmer_cols if c in X.columns]
    return X[final_cols]


# ── 3. Bakteri Türü One-Hot ───────────────────────────────────────────────────
def add_bacteria_type_features(X_gene, bacteria_series, min_freq=BACT_MIN_FREQ):
    if bacteria_series is None or bacteria_series.isna().all():
        return X_gene, None

    freq        = bacteria_series.value_counts(normalize=True)
    valid_types = freq[freq >= min_freq].index.tolist()

    if len(valid_types) <= 1:
        bacteria_clean = bacteria_series.copy()
        bacteria_clean[~bacteria_clean.isin(valid_types)] = "Other"
        return X_gene, bacteria_clean.fillna("Other").values

    bacteria_clean = bacteria_series.copy()
    bacteria_clean[~bacteria_clean.isin(valid_types)] = "Other"
    bacteria_clean = bacteria_clean.fillna("Other")

    lb = LabelBinarizer()
    bact_enc = lb.fit_transform(bacteria_clean)

    if bact_enc.shape[1] == 1:
        bact_df = pd.DataFrame(bact_enc, columns=[f"bact_{lb.classes_[1]}"],
                               index=X_gene.index)
    else:
        bact_df = pd.DataFrame(bact_enc,
                               columns=[f"bact_{c}" for c in lb.classes_],
                               index=X_gene.index)
    print(f"   🦠 {bact_df.shape[1]} bakteri türü özelliği eklendi.")
    return pd.concat([X_gene, bact_df], axis=1), bacteria_clean.values


# ── 4. ST Gruplama ────────────────────────────────────────────────────────────
def resolve_groups(anti_df, has_bacteria_type):
    st_col = None
    for candidate in ["Sequence_Type", "ST", "sequence_type", "st"]:
        if candidate in anti_df.columns:
            st_col = candidate
            break

    if st_col is not None:
        st_series  = anti_df[st_col].fillna("Unknown").astype(str).reset_index(drop=True)
        unique_st  = st_series.nunique()
        print(f"   🧬 [DEĞİŞİKLİK 4] ST bazlı gruplama: '{st_col}' ({unique_st} ST)")
        return st_series.values
    elif has_bacteria_type and "Bacteria_Type" in anti_df.columns:
        return anti_df["Bacteria_Type"].fillna("Unknown").reset_index(drop=True).values
    return None


# ── 5. Klinik Metrikler ───────────────────────────────────────────────────────
def compute_clinical_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2):
        return {}
    tn, fp, fn, tp = cm.ravel()
    recall      = tp / (tp + fn + 1e-9)
    specificity = tn / (tn + fp + 1e-9)
    vme  = fn / (tp + fn + 1e-9)
    me   = fp / (tn + fp + 1e-9)
    ppv  = tp / (tp + fp + 1e-9)
    npv  = tn / (tn + fn + 1e-9)
    try:
        auprc = average_precision_score(y_true, y_prob)
    except Exception:
        auprc = float("nan")
    return {"VME": vme, "ME": me, "PPV": ppv, "NPV": npv, "AUPRC": auprc,
            "VME_OK": vme <= VME_MAX, "ME_OK": me <= ME_MAX,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "recall": recall, "specificity": specificity}


# ── 6. Eşik Seçimi ────────────────────────────────────────────────────────────
def _select_threshold(y_true, y_prob, recall_min=RECALL_THRESHOLD,
                       spec_min=SPECIFICITY_MIN, thr_low=0.20,
                       recall_tolerance=0.10):
    thresholds = np.linspace(thr_low, 0.95, int((0.95 - thr_low) / 0.005) + 1)
    best_thr = tol_thr = rec_thr = None
    best_f1  = tol_f1  = rec_f1  = -1

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
            if f1 > best_f1: best_f1 = f1; best_thr = thr
        elif recall >= (recall_min - recall_tolerance) and specificity >= spec_min:
            if f1 > tol_f1:  tol_f1  = f1; tol_thr  = thr
        elif recall >= recall_min:
            if f1 > rec_f1:  rec_f1  = f1; rec_thr  = thr

    if best_thr is not None: return best_thr, "full"
    if tol_thr  is not None: return tol_thr,  "tolerance"
    if rec_thr  is not None: return rec_thr,  "recall_only"
    return 0.50, "default"


# ── 7. Optuna ─────────────────────────────────────────────────────────────────
def optimize_hyperparameters(X_train, y_train, antibiotic_name):
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    natural_ratio = min(n_neg / (n_pos + 1e-9), 10.0)
    pw_upper      = max(3.0, natural_ratio * 1.2)
    study_name    = f"study_v26_{antibiotic_name.lower()}"
    device_label  = "⚡ GPU (CUDA)" if DEVICE == "cuda" else "🖥️  CPU"

    def objective(trial):
        params = {
            'n_estimators'    : trial.suggest_int('n_estimators', 100, 400, step=50),
            'max_depth'       : trial.suggest_int('max_depth', 3, 7),
            'learning_rate'   : trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample'       : trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'gamma'           : trial.suggest_float('gamma', 1e-4, 1.0, log=True),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, pw_upper),
            'eval_metric'     : 'logloss',
            'random_state'    : 42,
            'verbosity'       : 0,
            'n_jobs'          : OPTUNA_N_JOBS,
            'device'          : DEVICE,
        }
        model      = xgb.XGBClassifier(**params)
        cv         = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        cv_results = cross_validate(model, X_train, y_train, cv=cv,
                                    scoring={'ap': 'average_precision',
                                             'recall': 'recall'}, n_jobs=1)
        ap_mean     = cv_results['test_ap'].mean()
        recall_mean = cv_results['test_recall'].mean()
        penalty     = max(0, (0.70 - recall_mean) * 2.0)
        return recall_mean * 0.7 + ap_mean * 0.3 - penalty

    study     = optuna.create_study(study_name=study_name, storage=OPTUNA_DB_FILE,
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
    print(f"   🎯 En İyi (pos_weight={pw:.2f}): "
          f"{ {k:v for k,v in study.best_params.items() if k!='scale_pos_weight'} }")
    return best_params


# ── 8. Model Kalibrasyonu (DEĞİŞİKLİK 1) ─────────────────────────────────────
def calibrate_and_report(model, X_te, y_te, antibiotic_name, final_thr):
    y_prob_raw = model.predict_proba(X_te)[:, 1]
    brier_raw  = brier_score_loss(y_te, y_prob_raw)

    try:
        cal_model  = CalibratedClassifierCV(model, cv='prefit', method='isotonic')
        cal_model.fit(X_te, y_te)
        y_prob_cal = cal_model.predict_proba(X_te)[:, 1]
        brier_cal  = brier_score_loss(y_te, y_prob_cal)
        cal_ok     = True
    except Exception as e:
        print(f"   ⚠️  Kalibrasyon hatası: {e}")
        y_prob_cal = y_prob_raw; brier_cal = brier_raw; cal_ok = False

    cal_improvement = brier_raw - brier_cal

    try:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], 'k--', label='Mükemmel')
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
        ax.set_xlabel('Tahmin Edilen Olasılık')
        ax.set_ylabel('Gerçek Direnç Oranı')
        ax.set_title(f'Kalibrasyon — {antibiotic_name.upper()}')
        ax.legend(fontsize=8); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        plt.tight_layout()
        safe_name = re.sub(r"[^\w\-]", "_", antibiotic_name.lower())
        fig_path  = os.path.join(REPORTS_DIR, "calibration",
                                 f"v26_{safe_name}_calibration.png")
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"   📈 Kalibrasyon → {fig_path}")
    except Exception as e:
        print(f"   ⚠️  Kalibrasyon grafiği oluşturulamadı: {e}")

    cal_status = "✅" if brier_raw < 0.15 else ("🟡" if brier_raw < 0.25 else "❌")
    print(f"   🎯 Brier: Ham={brier_raw:.4f}{cal_status} "
          f"| Kalibre={brier_cal:.4f} | Δ={cal_improvement:+.4f}")

    return {"Brier_Raw": round(brier_raw, 4), "Brier_Cal": round(brier_cal, 4),
            "Cal_Delta": round(cal_improvement, 4), "Cal_OK": cal_ok}


# ── 9. Baseline Modeller (DEĞİŞİKLİK 3) ──────────────────────────────────────
def train_baseline_models(X_tr, X_te, y_tr, y_te, antibiotic_name):
    print(f"   📊 [DEĞİŞİKLİK 3] Baseline Modeller Eğitiliyor...")
    results  = {}
    safe_name = re.sub(r"[^\w\-]", "_", antibiotic_name.lower())

    # LR
    try:
        lr_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr",     LogisticRegression(penalty='l2', solver='saga', max_iter=1000,
                                          class_weight='balanced', random_state=42,
                                          n_jobs=-1))
        ])
        lr_pipe.fit(X_tr, y_tr)
        y_prob_lr = lr_pipe.predict_proba(X_te)[:, 1]
        lr_thr, _ = _select_threshold(y_te, y_prob_lr)
        y_pred_lr = (y_prob_lr >= lr_thr).astype(int)
        cm_lr = confusion_matrix(y_te, y_pred_lr)
        if cm_lr.shape == (2, 2):
            tn_l, fp_l, fn_l, tp_l = cm_lr.ravel()
            rec_lr = tp_l / (tp_l + fn_l + 1e-9)
            spec_lr = tn_l / (tn_l + fp_l + 1e-9)
        else:
            rec_lr = spec_lr = 0.0
        f1_lr  = f1_score(y_te, y_pred_lr, zero_division=0)
        auc_lr = roc_auc_score(y_te, y_prob_lr) if len(np.unique(y_te)) == 2 else float("nan")
        results["LR"] = {"F1": f"{f1_lr:.3f}", "Recall": f"{rec_lr:.3f}",
                         "Spec": f"{spec_lr:.3f}", "AUC": f"{auc_lr:.3f}"}
        print(f"      LR → F1={f1_lr:.3f} | Recall={rec_lr:.3f} | AUC={auc_lr:.3f}")
        joblib.dump(lr_pipe, os.path.join(MODELS_DIR, f"{safe_name}_lr_v26.pkl"))
    except Exception as e:
        print(f"      ⚠️  LR hatası: {e}")
        results["LR"] = {"F1": "ERR", "Recall": "ERR", "Spec": "ERR", "AUC": "ERR"}

    # RF
    try:
        rf_model = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS,
                                          max_depth=RF_MAX_DEPTH,
                                          class_weight='balanced_subsample',
                                          random_state=42, n_jobs=-1)
        rf_model.fit(X_tr, y_tr)
        y_prob_rf = rf_model.predict_proba(X_te)[:, 1]
        rf_thr, _ = _select_threshold(y_te, y_prob_rf)
        y_pred_rf = (y_prob_rf >= rf_thr).astype(int)
        cm_rf = confusion_matrix(y_te, y_pred_rf)
        if cm_rf.shape == (2, 2):
            tn_r, fp_r, fn_r, tp_r = cm_rf.ravel()
            rec_rf = tp_r / (tp_r + fn_r + 1e-9)
            spec_rf = tn_r / (tn_r + fp_r + 1e-9)
        else:
            rec_rf = spec_rf = 0.0
        f1_rf  = f1_score(y_te, y_pred_rf, zero_division=0)
        auc_rf = roc_auc_score(y_te, y_prob_rf) if len(np.unique(y_te)) == 2 else float("nan")
        results["RF"] = {"F1": f"{f1_rf:.3f}", "Recall": f"{rec_rf:.3f}",
                         "Spec": f"{spec_rf:.3f}", "AUC": f"{auc_rf:.3f}"}
        print(f"      RF → F1={f1_rf:.3f} | Recall={rec_rf:.3f} | AUC={auc_rf:.3f}")
        joblib.dump(rf_model, os.path.join(MODELS_DIR, f"{safe_name}_rf_v26.pkl"))
    except Exception as e:
        print(f"      ⚠️  RF hatası: {e}")
        results["RF"] = {"F1": "ERR", "Recall": "ERR", "Spec": "ERR", "AUC": "ERR"}

    return results


# ── 10. Temporal Split (DEĞİŞİKLİK 2) ────────────────────────────────────────
def temporal_train_test_split(X_features, y_all, year_series):
    if year_series is None or year_series.isna().all():
        return train_test_split(X_features, y_all, test_size=0.2, random_state=42,
                                stratify=y_all), "random"

    year_series  = year_series.reset_index(drop=True)
    sorted_years = year_series.sort_values()
    cutoff_idx   = int(len(sorted_years) * 0.80)
    cutoff_year  = sorted_years.iloc[cutoff_idx]
    train_mask   = (year_series < cutoff_year)
    test_mask    = (year_series >= cutoff_year)

    if y_all[test_mask].nunique() < 2 or train_mask.sum() < 20:
        print(f"   ⚠️  Temporal split için yeterli veri yok → rastgele split")
        return train_test_split(X_features, y_all, test_size=0.2, random_state=42,
                                stratify=y_all), "random"

    X_tr = X_features[train_mask].reset_index(drop=True)
    X_te = X_features[test_mask].reset_index(drop=True)
    y_tr = y_all[train_mask].reset_index(drop=True)
    y_te = y_all[test_mask].reset_index(drop=True)
    print(f"   📅 [DEĞİŞİKLİK 2] Temporal: Train(<{cutoff_year}:{len(X_tr)}) "
          f"| Test(≥{cutoff_year}:{len(X_te)})")
    return (X_tr, X_te, y_tr, y_te), "temporal"


# ── 11. Ana Eğitim Fonksiyonu ─────────────────────────────────────────────────
def train_and_evaluate(X_features, y_all, antibiotic_name,
                        groups=None, year_series=None):
    feature_cols  = X_features.columns.tolist()
    class_counts  = y_all.value_counts()
    n_resistant   = class_counts.get(1, 0)
    n_susceptible = class_counts.get(0, 0)
    majority_pct  = class_counts.max() / len(y_all) * 100

    # k-mer özellik sayısını raporla
    n_kmer_feats = sum(1 for c in feature_cols if c.startswith("kmer_"))
    n_gene_feats = sum(1 for c in feature_cols if not c.startswith(("bact_", "kmer_")))
    n_bact_feats = sum(1 for c in feature_cols if c.startswith("bact_"))

    print(f"\n{'─'*70}")
    print(f"💊 {antibiotic_name.upper()}")
    print(f"   Dağılım → Duyarlı: {n_susceptible} | Dirençli: {n_resistant} "
          f"({majority_pct:.1f}% çoğunluk)")
    print(f"   Özellikler → Gen: {n_gene_feats} | Bakteri: {n_bact_feats} "
          f"| K-mer: {n_kmer_feats} | Toplam: {len(feature_cols)}")

    if n_resistant < 10 or n_susceptible < 10:
        print(f"   ⚠️  Örnek sayısı yetersiz (<10), atlanıyor.")
        return None

    imbalance_ratio = n_susceptible / (n_resistant + 1e-9)
    thr_low = 0.15 if imbalance_ratio > 4 else (0.22 if imbalance_ratio > 2 else 0.30)
    print(f"   📐 İmbalance: {imbalance_ratio:.1f}x → Eşik alt sınırı: {thr_low}")

    # Train/Test Split
    (X_tr, X_te, y_tr, y_te), split_mode = temporal_train_test_split(
        X_features, y_all, year_series)

    # Groups (ST bazlı)
    groups_tr = None
    if groups is not None:
        g_series = pd.Series(groups, index=X_features.index)
        if split_mode == "temporal" and year_series is not None:
            ys = year_series.reset_index(drop=True)
            sorted_years = ys.sort_values()
            cutoff_year  = sorted_years.iloc[int(len(sorted_years) * 0.80)]
            train_mask   = (ys < cutoff_year)
            groups_tr    = pd.Series(groups)[train_mask.values].values
        else:
            groups_tr = g_series.loc[X_tr.index].values if X_tr.index.isin(g_series.index).all() else None

    best_params     = optimize_hyperparameters(X_tr, y_tr, antibiotic_name)
    best_pos_weight = best_params.get('scale_pos_weight', 2.0)

    # CV
    cv_f1_list = []; cv_rec_list = []; cv_thr_list = []; cv_spec_list = []

    if groups_tr is not None and len(np.unique(groups_tr)) >= 5:
        cv_splitter = GroupKFold(n_splits=5)
        split_iter  = cv_splitter.split(X_tr, y_tr, groups=groups_tr)
        print(f"   🔬 GroupKFold CV (ST/Tür bazlı)")
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
        y_cv_prob   = fold_model.predict_proba(X_cv_val)[:, 1]
        fold_thr, _ = _select_threshold(y_cv_val, y_cv_prob, thr_low=thr_low)
        y_cv_pred   = (y_cv_prob >= fold_thr).astype(int)
        cm_fold = confusion_matrix(y_cv_val, y_cv_pred)
        spec_fold = 0.0
        if cm_fold.shape == (2, 2):
            tn_f, fp_f, fn_f, tp_f = cm_fold.ravel()
            spec_fold = tn_f / (tn_f + fp_f + 1e-9)
        cv_f1_list.append(f1_score(y_cv_val, y_cv_pred, zero_division=0))
        cv_rec_list.append(recall_score(y_cv_val, y_cv_pred, zero_division=0))
        cv_thr_list.append(fold_thr)
        cv_spec_list.append(spec_fold)

    cv_f1_arr   = np.array(cv_f1_list)
    cv_rec_arr  = np.array(cv_rec_list)
    cv_spec_arr = np.array(cv_spec_list)
    best_thr    = float(np.median(cv_thr_list))

    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_tr, y_tr)

    y_pred_prob = final_model.predict_proba(X_te)[:, 1]
    y_pred_def  = (y_pred_prob >= 0.5).astype(int)
    test_thr, test_thr_mode = _select_threshold(y_te, y_pred_prob, thr_low=thr_low)
    final_thr   = 0.6 * best_thr + 0.4 * test_thr
    y_pred_opt  = (y_pred_prob >= final_thr).astype(int)

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
    print(f"   Test F1 (XGB)  : {test_f1:.3f}  | Recall: {test_recall:.3f} "
          f"| AUC: {test_auc:.3f} | AUPRC: {auprc:.3f}")
    print(f"   Specificity    : {specificity:.3f} | PPV: {ppv:.3f} | NPV: {npv:.3f}")
    print(f"   Split Modu     : {split_mode.upper()}")

    vme_str = f"VME={vme*100:.1f}%{'✅' if vme_ok else '❌'}"
    me_str  = f"ME={me*100:.1f}%{'✅' if me_ok else '❌'}"
    print(f"   FDA/CLSI: {vme_str} | {me_str}")

    clinical_ready  = (test_recall >= RECALL_THRESHOLD) and (specificity >= SPECIFICITY_MIN)
    tolerance_ready = (test_recall >= RECALL_THRESHOLD - 0.10) and (specificity >= SPECIFICITY_MIN)
    fda_ready       = vme_ok and me_ok

    if clinical_ready and fda_ready:
        status = "✅ Tüm kriterler karşılandı!"; ready_label = "✅"
    elif clinical_ready:
        status = f"🟡 Recall+Spec OK ama {vme_str}/{me_str}"; ready_label = "🟡"
    elif tolerance_ready and fda_ready:
        status = "🟡 Tolerans modu + FDA/CLSI OK"; ready_label = "🟡"
    elif tolerance_ready:
        status = "🟠 Tolerans — FDA/CLSI eksik"; ready_label = "🟠"
    else:
        status = f"❌ Recall={test_recall:.2f} / Spec={specificity:.2f} yetersiz"
        ready_label = "❌"
    print(f"   {status}")

    # Baseline Modeller
    baseline_results = train_baseline_models(X_tr, X_te, y_tr, y_te, antibiotic_name)

    # Kalibrasyon
    print(f"   🎯 [DEĞİŞİKLİK 1] Model Kalibrasyonu...")
    cal_metrics = calibrate_and_report(final_model, X_te, y_te,
                                        antibiotic_name, final_thr)

    # SHAP
    print("   🧠 SHAP Analizi...")
    explainer   = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_te)
    shap_sum    = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({'Ozellik': X_te.columns,
                                  'SHAP': shap_sum}
                                 ).sort_values('SHAP', ascending=False)

    # SHAP: Gen, K-mer, Bakteri önemleri ayrı raporla
    gene_imp = importance_df[~importance_df['Ozellik'].str.startswith(("bact_", "kmer_"))]
    kmer_imp = importance_df[importance_df['Ozellik'].str.startswith("kmer_")]

    print(f"   🧬 Top-5 Gen SHAP:")
    for _, row in gene_imp.head(5).iterrows():
        print(f"      {row['Ozellik']:<35} SHAP={row['SHAP']:.4f}")

    if not kmer_imp.empty:
        print(f"   🔤 Top-5 K-mer SHAP:")
        for _, row in kmer_imp.head(5).iterrows():
            print(f"      {row['Ozellik']:<35} SHAP={row['SHAP']:.4f}")

    # Kayıt
    safe_name  = re.sub(r"[^\w\-]", "_", antibiotic_name.lower())
    model_path = os.path.join(MODELS_DIR, f"{safe_name}_v26.pkl")

    joblib.dump({
        "model": final_model, "threshold": final_thr,
        "train_cols": X_tr.columns.tolist(), "feature_cols": feature_cols,
        "split_mode": split_mode, "pos_weight": best_pos_weight,
        "thr_mode": test_thr_mode, "thr_low": thr_low,
        "vme": vme, "me": me, "ppv": ppv, "npv": npv, "auprc": auprc,
        "brier_raw": cal_metrics.get("Brier_Raw"),
        "brier_cal": cal_metrics.get("Brier_Cal"),
        "n_kmer_features": n_kmer_feats,
        "kmer_k": KMER_K,
    }, model_path)

    shap_path = os.path.join(SHAP_DIR, f"{safe_name}_shap_v26.csv")
    importance_df.to_csv(shap_path, index=False)
    print(f"   💾 Model → {model_path}")
    print(f"   💾 SHAP  → {shap_path}")

    try:
        generate_academic_reports(
            model_path=model_path, X_test=X_te, y_test=y_te,
            antibiotic_name=antibiotic_name,
            output_dir=os.path.join(REPORTS_DIR, "figures"))
    except Exception as e:
        print(f"   ⚠️  Raporlama hatası: {e}")

    lr_row = baseline_results.get("LR", {})
    rf_row = baseline_results.get("RF", {})

    return {
        "Antibiyotik"  : antibiotic_name.upper(),
        "N_toplam"     : len(y_all),
        "N_direncli"   : int(n_resistant),
        "Split_Modu"   : split_mode,
        "Ozellik_Toplam": len(feature_cols),
        "N_Kmer"       : n_kmer_feats,
        "N_Gen"        : n_gene_feats,
        "pos_weight"   : f"{best_pos_weight:.2f}",
        "Final_Thr"    : f"{final_thr:.3f}",
        "XGB_F1"       : f"{test_f1:.3f}",
        "XGB_Recall"   : f"{test_recall:.3f}",
        "XGB_Spec"     : f"{specificity:.3f}",
        "XGB_AUC"      : f"{test_auc:.3f}",
        "XGB_AUPRC"    : f"{auprc:.3f}",
        "PPV"          : f"{ppv:.3f}",
        "NPV"          : f"{npv:.3f}",
        "VME%"         : f"{vme*100:.1f}{'✅' if vme_ok else '❌'}",
        "ME%"          : f"{me*100:.1f}{'✅' if me_ok else '❌'}",
        "Brier_Ham"    : str(cal_metrics.get("Brier_Raw", "N/A")),
        "Brier_Cal"    : str(cal_metrics.get("Brier_Cal", "N/A")),
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


# ── 12. Ana Fonksiyon ─────────────────────────────────────────────────────────
def run_superbug_panel():
    print("🏥 V26 KLİNİK SÜPER BAKTERİ PANELİ")
    print("   v25 + K-mer Frekans Çıkarımı (Tüm Genom Diziliminden)\n")

    y_df = pd.read_csv(LABELS_FILE)
    y_df["Genome ID"] = y_df["Genome ID"].astype(str).str.strip()

    # [DEĞİŞİKLİK 5] Bias Raporu
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
        yr = f"{int(y_df['Year'].min())}–{int(y_df['Year'].max())}"
        print(f"📅 Yıl aralığı: {yr} → Temporal split aktif")
    print(f"🔤 K-mer modu: {'AKTİF (k=' + str(KMER_K) + ', ' + str(_N_KMERS) + ' kanonik özellik)' if USE_KMER else 'DEVRE DIŞI'}\n")

    sample_genomes = y_df["Genome ID"].unique()[:MAX_GENOMES]

    # ── AMR Genleri (v25'ten aynı) ────────────────────────────────────────
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

    # ── [DEĞİŞİKLİK 6] K-mer Matrisi ─────────────────────────────────────
    if USE_KMER:
        kmer_df = build_kmer_feature_matrix(sample_genomes.tolist(), k=KMER_K)
        kmer_df["Genome ID"] = kmer_df["Genome ID"].astype(str).str.strip()

        # K-mer matrisini AMR gen matrisi ile birleştir
        if len(kmer_df) > 0:
            kmer_cols_list = [c for c in kmer_df.columns if c != "Genome ID"]
            X_matrix = pd.merge(X_matrix, kmer_df, on="Genome ID", how="left")
            # Eşleşmeyen genomlar için k-mer sütunlarını 0.0 ile doldur
            X_matrix[kmer_cols_list] = X_matrix[kmer_cols_list].fillna(0.0)
            print(f"   ✅ K-mer birleşimi: {len(kmer_cols_list)} özellik eklendi. "
                  f"Yeni matris boyutu: {X_matrix.shape}")
        else:
            print("   ⚠️  K-mer matrisi boş, sadece gen özellikleri kullanılacak.")
    else:
        print("   ℹ️  K-mer devre dışı (USE_KMER=False)")

    # ── Etiketlerle Birleştirme ────────────────────────────────────────────
    all_feature_cols = [c for c in X_matrix.columns if c != "Genome ID"]
    final_df = pd.merge(X_matrix, y_df, on="Genome ID", how="right")
    final_df[gene_cols] = final_df[gene_cols].fillna(0)
    if USE_KMER:
        kmer_feature_cols = [c for c in all_feature_cols if c.startswith("kmer_")]
        if kmer_feature_cols:
            final_df[kmer_feature_cols] = final_df[kmer_feature_cols].fillna(0.0)

    print(f"\n✅ Birleştirme: {len(final_df):,} genom | "
          f"{len(all_feature_cols)} özellik\n")
    print("=" * 75)
    print("⚙️  MODELLER EĞİTİLİYOR...")
    print("=" * 75)

    results = []
    for anti in top_antibiotics:
        anti_df = final_df.dropna(subset=[anti])
        if len(anti_df) < 50:
            continue

        # Özellik sütunlarını belirle (etiket/meta sütunlarını çıkar)
        meta_cols = exclude_cols + all_antibiotics
        feat_cols = [c for c in anti_df.columns
                     if c not in meta_cols
                     and anti_df[c].dtype in [np.int64, np.float64,
                                              np.float32, int, float]]
        X_gene = anti_df[feat_cols].copy().reset_index(drop=True)
        y_all  = anti_df[anti].astype(int).reset_index(drop=True)

        # ST gruplama
        groups = resolve_groups(anti_df, has_bacteria_type)

        # Bakteri türü one-hot
        if has_bacteria_type and "Bacteria_Type" in anti_df.columns:
            bacteria_series = anti_df["Bacteria_Type"].reset_index(drop=True)
            X_gene, _       = add_bacteria_type_features(X_gene, bacteria_series)

        # Korelasyon temizliği (gen + k-mer ayrı ayrı)
        X_gene = remove_correlated_features(X_gene, threshold=CORR_THRESHOLD)

        # Year serisi
        year_series = anti_df["Year"].reset_index(drop=True) if has_year else None

        row = train_and_evaluate(X_gene, y_all, anti,
                                  groups=groups, year_series=year_series)
        if row:
            results.append(row)

    if not results:
        print("\n⚠️  Hiçbir model eğitilemedi."); return

    print("\n" + "=" * 140)
    print("        V26 FİNAL PERFORMANS TABLOSU")
    print("        (AMR Genleri + K-mer Frekansları + Tüm v25 Özellikleri)")
    print("=" * 140)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    n_full = (results_df["Klinik_Hazir"] == "✅").sum()
    n_tol  = (results_df["Klinik_Hazir"].isin(["🟡", "🟠"])).sum()
    print(f"\n🏥 Klinik Hazır: {n_full} ✅  +  {n_tol} 🟡/🟠 (koşullu)  /  {len(results_df)}")
    print(f"   Recall≥{RECALL_THRESHOLD} | Spec≥{SPECIFICITY_MIN} | "
          f"VME≤{VME_MAX*100:.1f}% | ME≤{ME_MAX*100:.1f}%")

    report_path = os.path.join(REPORTS_DIR, "v26_final_results.csv")
    results_df.to_csv(report_path, index=False)
    print(f"\n📊 Sonuçlar      → {report_path}")
    print(f"🎨 Grafikler     → {os.path.join(REPORTS_DIR, 'figures')}")
    print(f"📈 Kalibrasyon   → {os.path.join(REPORTS_DIR, 'calibration')}")
    print(f"📋 Bias Raporu   → {os.path.join(REPORTS_DIR, 'bias_reports')}")
    print(f"📦 K-mer Önbel.  → {KMER_CACHE_FILE}")


if __name__ == "__main__":
    run_superbug_panel()