# main_v27.py
# Kim et al. 2022 (Clin Microbiol Rev 35(3):e00179-21) uyumlu AMR Tahmin Sistemi
# RTX 3050 Laptop (4 GB VRAM / 16 GB RAM) optimize edilmiştir.
#
# ══════════════════════════════════════════════════════════════════════════════
# v26 → v27 DEĞİŞİKLİKLERİ
# ══════════════════════════════════════════════════════════════════════════════
# [v27-C1] PER-SPECIES AYRI MODEL
#   Kim 2022 eleştirisi: türler arası karıştırma filogenetik gürültü yaratır.
#   Her (bakteri_türü × antibiyotik) çifti için bağımsız XGBoost modeli.
#   RAM avantajı: alt-matrisler küçük, sıfır-varyans sütunlar otomatik düşer.
#
# [v27-C2] COĞRAFYA ÖNCELIKLI GroupKFold CV
#   Kim 2022: "Site stratification required for generalizability."
#   Öncelik: Country > Sequence_Type/ST > Bacteria_Type
#   Aynı ülkeden genomlar aynı fold'a → gerçek coğrafi genellenebilirlik.
#
# [v27-C3] AMİNO ASİT K-MER ÖZELLİKLERİ
#   Kim 2022: allel-düzeyinde temsil; nokta mutasyonlarını yakalar.
#   BV-BRC sp_gene API → AMR gen protein dizileri → k=4, hash=8192.
#   DNA k-mer'in 1/4'ü kadar bellek kullanımı.
#
# [v27-C4] BorderlineSMOTE / RandomOverSampler
#   Kim 2022 referansı: imbalanced-learn.
#   imbalance_ratio > 2 → SMOTE (sadece eğitim setine).
#   Başarısızlık → RandomOverSampler fallback.
#   Oversampling sonrası scale_pos_weight dinamik güncellenir.
#
# [v27-C5] MIC REGRESYON MODELİ
#   Kim 2022: breakpoint değişikliklerine karşı sayısal tahmin daha dayanıklı.
#   Etiket CSV'sinde MIC_{antibiyotik} sütunu varsa XGBRegressor eğitilir.
#
# [v27-C6] GEN-GEN ETKİLEŞİM ÖZELLİKLERİ
#   Kim 2022: "Genes treated as independent predictors — functional linkages
#   are not considered."
#   Katman-1: Biyolojik olarak bilinen çiftler (sabit liste).
#   Katman-2: En yüksek varyansa sahip top-12 gen × top-12 gen = 66 çift.
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
import hashlib
import optuna
import shap
import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import mmh3
    def _murmurhash(s: str, size: int) -> int:
        return mmh3.hash(s, signed=False) % size
except ImportError:
    def _murmurhash(s: str, size: int) -> int:
        return int.from_bytes(
            hashlib.md5(s.encode()).digest()[:4], byteorder='little') % size

# [v27-C4] imbalanced-learn
try:
    from imblearn.over_sampling import BorderlineSMOTE, RandomOverSampler
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("⚠️  imbalanced-learn kurulu değil → "
          "pip install imbalanced-learn --break-system-packages")

from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     GroupKFold, cross_validate)
from sklearn.metrics import (confusion_matrix, f1_score, recall_score,
                             roc_auc_score, average_precision_score,
                             brier_score_loss, mean_absolute_error)
from sklearn.calibration  import calibration_curve, CalibratedClassifierCV
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import RandomForestClassifier
from sklearn.pipeline      import Pipeline
from sklearn.feature_selection import VarianceThreshold

try:
    from reporting_module import generate_academic_reports
    REPORTING_AVAILABLE = True
except ImportError:
    REPORTING_AVAILABLE = False

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── FDA/CLSI Klinik Kabul Eşikleri ───────────────────────────────────────────
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
            capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            gpu_name = result.stdout.strip().splitlines()[0]
            print(f"🚀 GPU bulundu: {gpu_name} → XGBoost CUDA+hist")
            return "cuda"
    except Exception:
        pass
    print("⚠️  GPU bulunamadı → CPU")
    return "cpu"

DEVICE = _detect_device()
FALLBACK_TO_CPU: bool = False
_OOM_KEYWORDS = ["out of memory", "cudaerrormemoryallocation",
                 "memory allocation error", "free memory: 0b"]

def _is_oom_error(exc: Exception) -> bool:
    chain = " ".join([str(exc).lower(),
                      str(getattr(exc, '__cause__', '')).lower(),
                      str(getattr(exc, '__context__', '')).lower()])
    return any(kw in chain for kw in _OOM_KEYWORDS)

# ── Dosya Yolları ─────────────────────────────────────────────────────────────
LABELS_FILE     = "../data/processed/v2_multilabel_labels.csv"
CACHE_FILE      = "../data/processed/v17_amr_genes_cache.csv"
SEQ_CACHE_FILE  = "../data/processed/v26_sequences.parquet"
KMER_CACHE_FILE = "../data/processed/v26_kmer_hashed_features.parquet"
AA_KMER_CACHE   = "../data/processed/v27_aa_kmer_features.parquet"
AA_SEQ_CACHE    = "../data/processed/v27_amr_aa_sequences.parquet"
MODELS_DIR      = "../models"
REPORTS_DIR     = "../reports"
SHAP_DIR        = "../reports/shap_values"

# ── Genel Ayarlar ─────────────────────────────────────────────────────────────
MAX_GENOMES       = 30000
BATCH_SIZE        = 100
OPTUNA_TRIALS     = 30
OPTUNA_N_JOBS     = 1
MODEL_N_JOBS      = 1 if DEVICE == "cuda" else -1
BACT_MIN_FREQ     = 0.05
CORR_THRESHOLD    = 0.95
RF_N_ESTIMATORS   = 200
RF_MAX_DEPTH      = 10

# ── DNA K-mer ─────────────────────────────────────────────────────────────────
KMER_K        = 21
HASH_BITS     = 15
HASH_SIZE     = 2 ** HASH_BITS   # 32 768
SEQ_SAMPLE_BP = 200_000
SEQ_BATCH_SIZE= 50
USE_KMER      = True

# ── [v27-C3] Amino Asit K-mer ─────────────────────────────────────────────────
USE_AA_KMER   = True
AA_KMER_K     = 4
AA_HASH_BITS  = 13
AA_HASH_SIZE  = 2 ** AA_HASH_BITS   # 8 192

# ── [v27-C4] SMOTE ───────────────────────────────────────────────────────────
USE_SMOTE         = True
SMOTE_MIN_RATIO   = 2.0    # sadece imbalance > 2x ise uygula

# ── [v27-C1] Per-Species ─────────────────────────────────────────────────────
PER_SPECIES_MIN_SAMPLES   = 50
PER_SPECIES_MIN_RESISTANT = 15

# ── [v27-C6] Gen Etkileşim ───────────────────────────────────────────────────
USE_GENE_INTERACTIONS = True
INTERACTION_TOP_N     = 12   # top-12 gen → 66 çift interaksiyon

# Biyolojik olarak bilinen AMR gen çiftleri (kısmi isim eşleştirmesi)
AMR_KNOWN_INTERACTIONS = {
    'ciprofloxacin': [
        ('DNA gyrase subunit A', 'DNA topoisomerase IV'),
        ('gyrase subunit A',     'NorA'),
        ('gyrase',               'MexAB'),
    ],
    'ceftazidime': [
        ('CTX-M', 'SHV'),
        ('CTX-M', 'KPC'),
        ('CTX-M', 'OXA'),
    ],
    'gentamicin': [
        ("AAC(3)",  "AAC(6')"),
        ("AAC(3)",  "APH(2'')"),
        ("APH",     "ANT"),
    ],
    'ampicillin': [
        ('TEM', 'SHV'),
        ('TEM', 'CTX-M'),
        ('TEM', 'OXA'),
    ],
    'tetracycline': [
        ('Tet(',    'NorA'),
        ('Tet(',    'MFS'),
        ('Tet(',    'efflux'),
    ],
}

for _d in [MODELS_DIR, REPORTS_DIR, SHAP_DIR,
           os.path.join(REPORTS_DIR, "figures"),
           os.path.join(REPORTS_DIR, "calibration"),
           os.path.join(REPORTS_DIR, "bias_reports"),
           os.path.join(REPORTS_DIR, "mic_reports")]:
    os.makedirs(_d, exist_ok=True)

OPTUNA_DB_FILE = "sqlite:///" + os.path.abspath(MODELS_DIR) + "/optuna_studies_v27.db"

print(f"📐 DNA K-mer: k={KMER_K}, hash={HASH_SIZE}")
print(f"📐 AA  K-mer: k={AA_KMER_K}, hash={AA_HASH_SIZE} [v27-C3]")

# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 1 — DNA K-MER YARDIMCI FONKSİYONLARI  (v26'dan değişmedi)
# ══════════════════════════════════════════════════════════════════════════════
_COMP_TABLE = str.maketrans("ACGT", "TGCA")

def _reverse_complement(seq: str) -> str:
    return seq.translate(_COMP_TABLE)[::-1]

def _canonical(kmer: str) -> str:
    rc = _reverse_complement(kmer)
    return kmer if kmer <= rc else rc

def extract_kmer_freq(seq: str, k: int = KMER_K,
                      hash_size: int = HASH_SIZE) -> np.ndarray:
    freq = np.zeros(hash_size, dtype=np.float32)
    seq  = re.sub(r'[^ACGT]', '', seq.upper())
    L    = len(seq)
    if L < k:
        return freq
    total = 0
    for i in range(L - k + 1):
        ck  = _canonical(seq[i:i + k])
        idx = _murmurhash(ck, hash_size)
        freq[idx] += 1
        total += 1
    if total > 0:
        freq /= total
    return freq

def _kmer_worker(args: tuple) -> np.ndarray:
    seq, k, hash_size = args
    return extract_kmer_freq(seq, k=k, hash_size=hash_size)

# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 2 — [v27-C3] AMİNO ASİT K-MER FONKSİYONLARI
# ══════════════════════════════════════════════════════════════════════════════
_AA_ALPHABET = set("ACDEFGHIKLMNPQRSTVWY")

def _aa_kmer_worker(args: tuple) -> np.ndarray:
    """Amino asit k-mer frekansı: 20 standart AA filtresi + MurmurHash."""
    seq, k, hash_size = args
    freq = np.zeros(hash_size, dtype=np.float32)
    seq  = "".join(c for c in seq.upper() if c in _AA_ALPHABET)
    L    = len(seq)
    if L < k:
        return freq
    total = 0
    for i in range(L - k + 1):
        idx = _murmurhash(seq[i:i + k], hash_size)
        freq[idx] += 1
        total += 1
    if total > 0:
        freq /= total
    return freq

def fetch_amr_protein_sequences(genome_ids: list) -> dict:
    """
    BV-BRC sp_gene API'den AMR gen amino asit dizilerini çeker.
    Dönüş: {genome_id: concatenated_aa_string}
    Not: API aa_sequence döndürmezse boş sözlük döner (USE_AA_KMER sessizce devre dışı).
    """
    genome_ids = [str(g) for g in genome_ids]
    result: dict = {}

    # Önbellek kontrolü
    if os.path.exists(AA_SEQ_CACHE):
        try:
            cached = pd.read_parquet(AA_SEQ_CACHE)
            for _, row in cached.iterrows():
                result[str(row["genome_id"])] = str(row["aa_seq"])
            missing = [g for g in genome_ids if g not in result]
            if not missing:
                print(f"   📦 AA dizi önbelleği: {len(result):,} genom")
                return result
            genome_ids = missing
        except Exception:
            pass

    print(f"   🌐 {len(genome_ids):,} genom için AMR protein dizisi çekiliyor...")
    headers = {"Content-Type": "application/x-www-form-urlencoded",
               "Accept": "application/json"}
    new_rows = []
    batch_seq_map: dict = {}

    for i in range(0, len(genome_ids), 50):
        batch  = genome_ids[i:i + 50]
        id_str = ",".join(batch)
        payload = (f"in(genome_id,({id_str}))"
                   f"&select(genome_id,aa_sequence)&limit(10000)")
        try:
            r = requests.post("https://www.bv-brc.org/api/sp_gene/",
                              headers=headers, data=payload, timeout=120)
            if r.status_code == 200:
                data = r.json()
                for item in data:
                    gid = str(item.get("genome_id", "")).strip()
                    aa  = str(item.get("aa_sequence", "")).strip()
                    if gid and aa and aa.lower() not in ("", "none", "null"):
                        if gid in batch_seq_map:
                            batch_seq_map[gid] += aa
                        else:
                            batch_seq_map[gid] = aa
        except Exception:
            pass
        pct = min(100, int((i + 50) / len(genome_ids) * 100))
        print(f"\r   AA protein %{pct:3d}", end="", flush=True)
        time.sleep(0.3)

    print()
    result.update(batch_seq_map)
    for gid, seq in batch_seq_map.items():
        new_rows.append({"genome_id": gid, "aa_seq": seq})

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        if os.path.exists(AA_SEQ_CACHE):
            try:
                old = pd.read_parquet(AA_SEQ_CACHE)
                new_df = pd.concat([old, new_df], ignore_index=True)\
                           .drop_duplicates("genome_id", keep="last")
            except Exception:
                pass
        _write_parquet_chunked(new_df, AA_SEQ_CACHE)

    if not result:
        print("   ⚠️  API aa_sequence döndürmedi → AA k-mer bu oturumda devre dışı.")
    return result

def build_aa_kmer_feature_matrix(genome_ids: list) -> pd.DataFrame:
    """Amino asit k-mer özellik matrisi: k=4, hash_size=8192."""
    k         = AA_KMER_K
    hash_size = AA_HASH_SIZE
    aa_cols   = [f"aa_kmer_h{i}" for i in range(hash_size)]

    if os.path.exists(AA_KMER_CACHE):
        try:
            cached = pd.read_parquet(AA_KMER_CACHE)
            cached["Genome ID"] = cached["Genome ID"].astype(str)
            n_feat = len([c for c in cached.columns if c != "Genome ID"])
            if n_feat == hash_size:
                print(f"   📦 AA K-mer önbelleği: {len(cached):,} × {n_feat}")
                return cached
        except Exception:
            pass

    print(f"\n🔬 [v27-C3] AA K-mer Matrisi (k={k}, hash={hash_size})...")
    prot_seqs    = fetch_amr_protein_sequences(genome_ids)
    genome_ids_s = [str(g) for g in genome_ids]
    valid_gids   = [g for g in genome_ids_s if g in prot_seqs]

    if not valid_gids:
        print("   ⚠️  AA k-mer matrisi boş döndü.")
        return pd.DataFrame(columns=["Genome ID"] + aa_cols)

    args_list = [(prot_seqs[g], k, hash_size) for g in valid_gids]
    results   = joblib.Parallel(n_jobs=-1, backend="loky")(
        joblib.delayed(_aa_kmer_worker)(a) for a in args_list)

    freq_matrix  = np.vstack(results).astype(np.float32)
    aa_kmer_df   = pd.DataFrame(freq_matrix, columns=aa_cols)
    aa_kmer_df.insert(0, "Genome ID", valid_gids)

    # VarianceThreshold — sıfır-varyans binleri çıkar
    vt       = VarianceThreshold(threshold=0.0)
    vt.fit(aa_kmer_df[aa_cols])
    kept     = [c for c, k_ in zip(aa_cols, vt.get_support()) if k_]
    removed  = len(aa_cols) - len(kept)
    if removed:
        print(f"   🔬 AA VarianceThreshold: {removed} bin çıkarıldı")
    aa_kmer_df = aa_kmer_df[["Genome ID"] + kept]

    _write_parquet_chunked(aa_kmer_df, AA_KMER_CACHE)
    print(f"   ✅ AA K-mer: {len(aa_kmer_df):,} × {len(kept)} özellik kaydedildi")
    return aa_kmer_df

# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 3 — YARDIMCI FONKSİYONLAR  (v26'dan değişmedi)
# ══════════════════════════════════════════════════════════════════════════════
def _write_parquet_chunked(df: pd.DataFrame, path: str,
                            row_group_size: int = 500) -> None:
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path, row_group_size=row_group_size,
                   compression='snappy')

def _fetch_sequences_batch(genome_ids: list, max_retries=3) -> dict:
    id_str  = ",".join(str(g) for g in genome_ids)
    payload = (f"in(genome_id,({id_str}))"
               f"&select(genome_id,sequence)&limit(50000)")
    headers = {"Content-Type": "application/x-www-form-urlencoded",
               "Accept": "application/json"}
    result  = {}
    for attempt in range(max_retries):
        try:
            r = requests.post("https://www.bv-brc.org/api/genome_sequence/",
                              headers=headers, data=payload, timeout=180)
            if r.status_code == 200:
                for item in r.json():
                    gid = str(item.get("genome_id", "")).strip()
                    seq = str(item.get("sequence", ""))
                    if gid and seq:
                        result[gid] = result.get(gid, "") + seq
                for gid in result:
                    result[gid] = result[gid][:SEQ_SAMPLE_BP]
                return result
            time.sleep(5)
        except requests.exceptions.RequestException as e:
            print(f"      ❌ Ağ Hatası ({attempt+1}): {e}")
            time.sleep(3 * (attempt + 1))
    return result

def fetch_genome_sequences(genome_ids: list) -> pd.DataFrame:
    genome_ids = [str(g) for g in genome_ids]
    cached_df  = pd.DataFrame(columns=["Genome ID", "sequence"])
    tmp_path   = SEQ_CACHE_FILE + ".tmp"
    if os.path.exists(SEQ_CACHE_FILE):
        try:
            cached_df = pd.read_parquet(SEQ_CACHE_FILE)
            cached_df["Genome ID"] = cached_df["Genome ID"].astype(str)
        except Exception:
            pass
    cached_ids = set(cached_df["Genome ID"].tolist())
    missing    = [g for g in genome_ids if g not in cached_ids]
    if not missing:
        return cached_df
    print(f"\n🌐 {len(missing):,} genomun DNA dizisi çekiliyor...")
    all_rows = []
    for b_idx, i in enumerate(range(0, len(missing), SEQ_BATCH_SIZE)):
        batch   = missing[i:i + SEQ_BATCH_SIZE]
        seq_map = _fetch_sequences_batch(batch)
        for gid, seq in seq_map.items():
            all_rows.append({"Genome ID": gid, "sequence": seq})
        if (b_idx + 1) % 50 == 0 and all_rows:
            try:
                _write_parquet_chunked(pd.DataFrame(all_rows), tmp_path)
            except Exception:
                pass
        pct = min(100, int((i + SEQ_BATCH_SIZE) / len(missing) * 100))
        print(f"\r   %{pct:3d} ({min(i+SEQ_BATCH_SIZE,len(missing))}/{len(missing)})",
              end="", flush=True)
    print()
    if all_rows:
        new_df    = pd.DataFrame(all_rows)
        final_df  = (pd.concat([cached_df, new_df], ignore_index=True)
                     if not cached_df.empty else new_df)
        final_df  = final_df[final_df["Genome ID"].isin(genome_ids)].reset_index(drop=True)
        try:
            _write_parquet_chunked(final_df, SEQ_CACHE_FILE)
        except Exception as e:
            print(f"   ⚠️  Parquet yazılamadı: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return final_df
    return cached_df

def build_kmer_feature_matrix(genome_ids: list) -> pd.DataFrame:
    genome_ids = [str(g) for g in genome_ids]
    kmer_cols  = [f"kmer_h{i}" for i in range(HASH_SIZE)]
    if os.path.exists(KMER_CACHE_FILE):
        try:
            cached = pd.read_parquet(KMER_CACHE_FILE)
            cached["Genome ID"] = cached["Genome ID"].astype(str)
            n_feat = len([c for c in cached.columns if c != "Genome ID"])
            if n_feat == HASH_SIZE:
                print(f"📦 DNA K-MER önbelleği: {len(cached):,} × {n_feat}")
                return cached
            os.remove(KMER_CACHE_FILE)
        except Exception:
            pass
    print(f"\n🧬 DNA K-mer Matrisi hesaplanıyor (k={KMER_K}, hash={HASH_SIZE})...")
    parquet_file      = pq.ParquetFile(SEQ_CACHE_FILE)
    all_gids, rows    = [], []
    for batch in parquet_file.iter_batches(batch_size=1000):
        chunk_df   = batch.to_pandas()
        args_list  = [(row["sequence"], KMER_K, HASH_SIZE)
                      for _, row in chunk_df.iterrows()]
        chunk_res  = joblib.Parallel(n_jobs=-1, backend="loky")(
            joblib.delayed(_kmer_worker)(a) for a in args_list)
        all_gids.extend(chunk_df["Genome ID"].values)
        rows.extend(chunk_res)
    freq_matrix = np.vstack(rows).astype(np.float32)
    kmer_df     = pd.DataFrame(freq_matrix, columns=kmer_cols)
    kmer_df.insert(0, "Genome ID", all_gids)
    vt          = VarianceThreshold(threshold=0.0)
    vt.fit(kmer_df[kmer_cols])
    kept_cols   = [c for c, k_ in zip(kmer_cols, vt.get_support()) if k_]
    kmer_df     = kmer_df[["Genome ID"] + kept_cols]
    _write_parquet_chunked(kmer_df, KMER_CACHE_FILE)
    print(f"   ✅ DNA K-mer: {len(kmer_df):,} × {len(kept_cols)}")
    return kmer_df

# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 4 — VERİ HAZIRLIK FONKSİYONLARI
# ══════════════════════════════════════════════════════════════════════════════
def veri_bias_raporu(y_df: pd.DataFrame):
    print("\n📊 Örnekleme Bias Raporu...")
    bias_cols = {"Year": "Yıl", "Country": "Ülke",
                 "Bacteria_Type": "Bakteri Türü", "Sequence_Type": "ST"}
    for col, label in bias_cols.items():
        if col not in y_df.columns:
            continue
        dist    = y_df[col].value_counts(dropna=False)
        pct     = (dist / len(y_df) * 100).round(2)
        out     = os.path.join(REPORTS_DIR, "bias_reports",
                               f"v27_bias_{col.lower()}.csv")
        pd.DataFrame({"Sayi": dist, "Yüzde": pct}).to_csv(out)
        top3 = ", ".join(f"{k}({v:.1f}%)" for k, v in pct.head(3).items())
        print(f"   {label}: {top3} → {out}")
    print()

def fetch_amr_genes_from_bvbrc(genome_ids):
    genome_ids = [str(g) for g in genome_ids]
    if os.path.exists(CACHE_FILE):
        try:
            cached = pd.read_csv(CACHE_FILE)
            if not cached.empty and "Genome ID" in cached.columns:
                cached["Genome ID"] = cached["Genome ID"].astype(str)
                print(f"📦 YEREL ÖNBELLEK: {CACHE_FILE} ({len(cached):,} kayıt)")
                return cached
        except Exception:
            pass
    print(f"🌐 {len(genome_ids):,} bakteri için AMR genleri çekiliyor...")
    df = _fetch_amr_from_api(genome_ids)
    if not df.empty:
        df.to_csv(CACHE_FILE, index=False)
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

def remove_correlated_features(X: pd.DataFrame,
                                threshold: float = CORR_THRESHOLD) -> pd.DataFrame:
    gene_cols = [c for c in X.columns
                 if not c.startswith(("bact_", "kmer_", "aa_kmer_", "interact_"))]
    bact_cols = [c for c in X.columns if c.startswith("bact_")]
    kmer_cols = [c for c in X.columns
                 if c.startswith("kmer_") or c.startswith("aa_kmer_")]
    inter_cols= [c for c in X.columns if c.startswith("interact_")]

    removed_total = 0
    if len(gene_cols) >= 2:
        X_gene   = X[gene_cols].copy()
        var_mask = X_gene.var() > 0
        X_gene   = X_gene.loc[:, var_mask]
        corr     = X_gene.corr(method='pearson').abs()
        upper    = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        drop_g   = [c for c in upper.columns if any(upper[c] > threshold)]
        gene_cols     = [c for c in gene_cols if c not in drop_g and c in X_gene.columns]
        removed_total += len(drop_g)

    if removed_total:
        print(f"   🔬 Korelasyon temizliği: {removed_total} gen çıkarıldı "
              f"(k-mer / interaksiyon sütunları atlandı)")

    final_cols = [c for c in (gene_cols + bact_cols + kmer_cols + inter_cols)
                  if c in X.columns]
    return X[final_cols]

def add_bacteria_type_features(X_gene, bacteria_series, min_freq=BACT_MIN_FREQ):
    if bacteria_series is None or bacteria_series.isna().all():
        return X_gene, None
    freq        = bacteria_series.value_counts(normalize=True)
    valid_types = freq[freq >= min_freq].index.tolist()
    bacteria_clean = bacteria_series.copy()
    bacteria_clean[~bacteria_clean.isin(valid_types)] = "Other"
    bacteria_clean = bacteria_clean.fillna("Other")
    if len(valid_types) <= 1:
        return X_gene, bacteria_clean.values
    lb       = LabelBinarizer()
    bact_enc = lb.fit_transform(bacteria_clean)
    if bact_enc.shape[1] == 1:
        bact_df = pd.DataFrame(bact_enc,
                               columns=[f"bact_{lb.classes_[1]}"],
                               index=X_gene.index)
    else:
        bact_df = pd.DataFrame(bact_enc,
                               columns=[f"bact_{c}" for c in lb.classes_],
                               index=X_gene.index)
    print(f"   🦠 {bact_df.shape[1]} bakteri türü özelliği eklendi.")
    return pd.concat([X_gene, bact_df], axis=1), bacteria_clean.values

# ── [v27-C2] Coğrafya Öncelikli Gruplama ─────────────────────────────────────
def resolve_groups_v27(anti_df: pd.DataFrame):
    """
    [v27-C2] Kim 2022: site/geographic stratification.
    Öncelik: Country > Sequence_Type/ST > Bacteria_Type
    Gruplar ≥ n_splits gerekli; aksi hâlde bir üst seviyeye düşülür.
    """
    n_splits = 5
    for col in ["Country", "Sequence_Type", "ST", "Bacteria_Type"]:
        if col in anti_df.columns:
            s = anti_df[col].fillna("Unknown").astype(str).reset_index(drop=True)
            if s.nunique() >= n_splits:
                print(f"   🌍 [v27-C2] Gruplama: '{col}' ({s.nunique()} grup)")
                return s.values, col
    return None, None

# ── Klinik Metrikler ──────────────────────────────────────────────────────────
def compute_clinical_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2):
        return {}
    tn, fp, fn, tp = cm.ravel()
    recall      = tp / (tp + fn + 1e-9)
    specificity = tn / (tn + fp + 1e-9)
    return {
        "VME": fn / (tp + fn + 1e-9), "ME": fp / (tn + fp + 1e-9),
        "PPV": tp / (tp + fp + 1e-9), "NPV": tn / (tn + fn + 1e-9),
        "AUPRC": average_precision_score(y_true, y_prob),
        "VME_OK": fn / (tp + fn + 1e-9) <= VME_MAX,
        "ME_OK":  fp / (tn + fp + 1e-9) <= ME_MAX,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "recall": recall, "specificity": specificity,
    }

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

# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 5 — OPTUNA HİPERPARAMETRE OPTİMİZASYONU
# ══════════════════════════════════════════════════════════════════════════════
def optimize_hyperparameters(X_train, y_train, study_key: str,
                              already_oversampled: bool = False):
    global FALLBACK_TO_CPU
    FALLBACK_TO_CPU = False

    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    natural_ratio = min(n_neg / (n_pos + 1e-9), 10.0)
    pw_upper      = (2.0 if already_oversampled else max(3.0, natural_ratio * 1.2))

    # ── DÜZELTME [v27-FIX-1]: f-string içinde ters eğik çizgi kullanılamaz ──
    # Orijinal (hatalı): f"study_v27_{re.sub(r'[^\w]','_', study_key.lower())}"
    _safe_study_key = re.sub(r'[^\w]', '_', study_key.lower())
    study_name      = f"study_v27_{_safe_study_key}"
    # ─────────────────────────────────────────────────────────────────────────

    device_label = "⚡ GPU (CUDA+hist)" if DEVICE == "cuda" else "🖥️  CPU"

    def objective(trial):
        global FALLBACK_TO_CPU
        current_device = 'cpu' if FALLBACK_TO_CPU else DEVICE
        current_njobs  = -1    if FALLBACK_TO_CPU else OPTUNA_N_JOBS
        params = {
            'n_estimators'    : trial.suggest_int('n_estimators', 100, 200, step=50),
            'max_depth'       : trial.suggest_int('max_depth', 3, 7),
            'learning_rate'   : trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample'       : trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.15, 0.25),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'gamma'           : trial.suggest_float('gamma', 1e-4, 1.0, log=True),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, pw_upper),
            'tree_method': 'hist', 'max_bin': 256,
            'eval_metric': 'logloss', 'random_state': 42,
            'verbosity': 0, 'n_jobs': current_njobs, 'device': current_device,
        }
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        try:
            model      = xgb.XGBClassifier(**params)
            cv_results = cross_validate(model, X_train, y_train, cv=cv,
                                        scoring={'ap': 'average_precision',
                                                 'recall': 'recall'}, n_jobs=1)
        except Exception as e:
            if _is_oom_error(e):
                if not FALLBACK_TO_CPU:
                    print(f"\n   ⚠️  GPU OOM → Trial {trial.number}'dan CPU'ya geçiliyor!")
                    FALLBACK_TO_CPU = True
                params.update({'device': 'cpu', 'n_jobs': -1})
                try:
                    model      = xgb.XGBClassifier(**params)
                    cv_results = cross_validate(model, X_train, y_train, cv=cv,
                                                scoring={'ap': 'average_precision',
                                                         'recall': 'recall'}, n_jobs=1)
                except Exception as cpu_exc:
                    if _is_oom_error(cpu_exc):
                        raise optuna.exceptions.TrialPruned()
                    raise
            else:
                raise
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
    final_device = 'cpu' if FALLBACK_TO_CPU else DEVICE
    best_params.update({
        'tree_method': 'hist', 'max_bin': 256, 'eval_metric': 'logloss',
        'random_state': 42, 'verbosity': 0,
        'n_jobs': -1 if FALLBACK_TO_CPU else MODEL_N_JOBS,
        'device': final_device,
    })
    pw       = best_params.get('scale_pos_weight', 2.0)
    mode_str = "CPU (OOM fallback)" if FALLBACK_TO_CPU else device_label
    print(f"   🎯 En İyi (pos_weight={pw:.2f}) [{mode_str}]")
    return best_params

# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 6 — [v27-C4] OVERSAMPLING
# ══════════════════════════════════════════════════════════════════════════════
def apply_oversampling(X_tr: pd.DataFrame, y_tr: pd.Series,
                       imbalance_ratio: float):
    """
    [v27-C4] Kim 2022 referansı: imbalanced-learn.
    Strateji:
      - Gen/bakteri sütunlarına BorderlineSMOTE (gerçek sentez).
      - k-mer sütunlarının sentetik örnekleri en yakın komşudan kopyalanır
        (bellek dostu; k-mer uzayında doğrusal interpolasyon yetersiz kalır).
      - Başarısız → RandomOverSampler fallback.
    """
    if not USE_SMOTE or not IMBLEARN_AVAILABLE or imbalance_ratio < SMOTE_MIN_RATIO:
        return X_tr, y_tr, "none"

    gene_bact_cols = [c for c in X_tr.columns
                      if not c.startswith(("kmer_", "aa_kmer_"))]
    kmer_cols      = [c for c in X_tr.columns
                      if c.startswith("kmer_") or c.startswith("aa_kmer_")]

    X_gb   = X_tr[gene_bact_cols].values.astype(np.float32)
    X_km   = X_tr[kmer_cols].values.astype(np.float32) if kmer_cols else None
    y_arr  = y_tr.values

    method_used = "none"
    try:
        k_nn = min(3, (y_arr == 1).sum() - 1)
        if k_nn < 1:
            raise ValueError("Azınlık sınıfı çok küçük.")
        sampler = BorderlineSMOTE(k_neighbors=k_nn, random_state=42, n_jobs=-1)
        X_gb_res, y_res = sampler.fit_resample(X_gb, y_arr)
        method_used = "BorderlineSMOTE"
    except Exception as e:
        print(f"   ⚠️  BorderlineSMOTE başarısız ({e}) → RandomOverSampler")
        try:
            ros = RandomOverSampler(random_state=42)
            X_gb_res, y_res = ros.fit_resample(X_gb, y_arr)
            method_used = "RandomOverSampler"
        except Exception as e2:
            print(f"   ⚠️  RandomOverSampler de başarısız ({e2}) → oversampling atlandı")
            return X_tr, y_tr, "none"

    n_orig      = len(X_tr)
    n_synthetic = len(X_gb_res) - n_orig

    if X_km is not None and n_synthetic > 0:
        minority_kmer = X_km[y_arr == 1]
        synth_kmer    = minority_kmer[
            np.random.default_rng(42).integers(len(minority_kmer), size=n_synthetic)
        ]
        X_km_res = np.vstack([X_km, synth_kmer])
        X_res    = np.hstack([X_gb_res, X_km_res])
        cols     = gene_bact_cols + kmer_cols
    else:
        X_res = X_gb_res
        cols  = gene_bact_cols

    n_pos_res = (y_res == 1).sum()
    n_neg_res = (y_res == 0).sum()
    print(f"   ♻️  [{method_used}] {n_orig} → {len(y_res)} örnek "
          f"(+{n_synthetic} sentetik, denge: {n_neg_res}/{n_pos_res})")

    X_df = pd.DataFrame(X_res.astype(np.float32), columns=cols)
    return X_df, pd.Series(y_res), method_used

# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 7 — [v27-C6] GEN-GEN ETKİLEŞİM ÖZELLİKLERİ
# ══════════════════════════════════════════════════════════════════════════════
def build_interaction_features(X: pd.DataFrame,
                                antibiotic_name: str) -> pd.DataFrame:
    """
    [v27-C6] Kim 2022: "functional linkages not considered by ML models."
    Katman-1: Bilinen biyolojik gen çiftleri (AMR_KNOWN_INTERACTIONS).
    Katman-2: En yüksek varyansa sahip top-N gen → tüm çiftler (N*(N-1)/2).
    Dönüş: X + interaksiyon sütunları (interact_ öneki).
    """
    if not USE_GENE_INTERACTIONS:
        return X

    gene_cols = [c for c in X.columns
                 if not c.startswith(("bact_", "kmer_", "aa_kmer_", "interact_"))]
    interact_df = pd.DataFrame(index=X.index)

    # Katman-1: Bilinen biyolojik çiftler
    anti_key    = antibiotic_name.lower()
    known_pairs = AMR_KNOWN_INTERACTIONS.get(anti_key, [])
    n_bio = 0
    for pat1, pat2 in known_pairs:
        cols1 = [c for c in gene_cols if pat1.lower() in c.lower()]
        cols2 = [c for c in gene_cols if pat2.lower() in c.lower()]
        if cols1 and cols2:
            c1 = cols1[0]; c2 = cols2[0]
            # ── DÜZELTME [v27-FIX-2]: f-string içinde ters eğik çizgi kullanılamaz ──
            # Orijinal (hatalı):
            #   key = f"interact_{re.sub(r'[^\w]','_', c1[:20])}__X__{re.sub(r'[^\w]','_', c2[:20])}"
            _safe_c1 = re.sub(r'[^\w]', '_', c1[:20])
            _safe_c2 = re.sub(r'[^\w]', '_', c2[:20])
            key = f"interact_{_safe_c1}__X__{_safe_c2}"
            # ─────────────────────────────────────────────────────────────────
            interact_df[key] = (X[c1].values * X[c2].values).astype(np.float32)
            n_bio += 1

    # Katman-2: Varyans-bazlı top-N gen çiftleri
    if gene_cols:
        variances    = X[gene_cols].var()
        top_genes    = variances.nlargest(INTERACTION_TOP_N).index.tolist()
        n_data_pairs = 0
        for i, g1 in enumerate(top_genes):
            for g2 in top_genes[i + 1:]:
                # ── DÜZELTME [v27-FIX-3]: f-string içinde ters eğik çizgi kullanılamaz ──
                # Orijinal (hatalı):
                #   key = f"interact_{re.sub(r'[^\w]','_', g1[:15])}__{re.sub(r'[^\w]','_', g2[:15])}"
                _safe_g1 = re.sub(r'[^\w]', '_', g1[:15])
                _safe_g2 = re.sub(r'[^\w]', '_', g2[:15])
                key = f"interact_{_safe_g1}__{_safe_g2}"
                # ─────────────────────────────────────────────────────────────
                if key not in interact_df.columns:
                    interact_df[key] = (X[g1].values * X[g2].values).astype(np.float32)
                    n_data_pairs += 1

        # Sıfır-varyans interaksiyon sütunlarını temizle
        nonzero = interact_df.var() > 0
        interact_df = interact_df.loc[:, nonzero]
        print(f"   🔗 [v27-C6] İnteraksiyon: {n_bio} biyolojik + "
              f"{nonzero.sum() - n_bio} varyans-bazlı = {nonzero.sum()} çift")

    if interact_df.empty:
        return X
    return pd.concat([X, interact_df], axis=1)

# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 8 — [v27-C5] MIC REGRESYON MODELİ
# ══════════════════════════════════════════════════════════════════════════════
def train_mic_regression_model(X_tr: pd.DataFrame, X_te: pd.DataFrame,
                                y_mic_tr: pd.Series, y_mic_te: pd.Series,
                                antibiotic_name: str, species_key: str) -> dict:
    """
    [v27-C5] Kim 2022: quantitative MIC prediction — breakpoint bağımsız.
    Log2 dönüşümüyle MAPE, log2 MAE, ±1 dilim doğruluk raporlanır.
    """
    print(f"   📏 [v27-C5] MIC Regresyon: {len(y_mic_tr)} eğitim / {len(y_mic_te)} test")
    try:
        y_log_tr = np.log2(y_mic_tr.clip(lower=0.001))
        y_log_te = np.log2(y_mic_te.clip(lower=0.001))

        reg_params = {
            'n_estimators': 200, 'max_depth': 4,
            'learning_rate': 0.05, 'subsample': 0.8,
            'colsample_bytree': 0.2, 'tree_method': 'hist',
            'max_bin': 256, 'random_state': 42, 'verbosity': 0,
            'device': 'cpu',
            'n_jobs': -1,
        }
        reg_model = xgb.XGBRegressor(**reg_params)
        reg_model.fit(X_tr, y_log_tr)

        y_pred_log = reg_model.predict(X_te)
        mae_log2   = mean_absolute_error(y_log_te, y_pred_log)
        within_1d  = np.mean(np.abs(y_log_te - y_pred_log) <= 1.0)
        within_2d  = np.mean(np.abs(y_log_te - y_pred_log) <= 2.0)

        _safe_label = re.sub(r'[^\w\-]', '_',
                             f"{species_key}_{antibiotic_name}".lower())
        model_path  = os.path.join(MODELS_DIR, f"{_safe_label}_mic_v27.pkl")
        joblib.dump({"model": reg_model, "log2_transform": True}, model_path)

        report_path = os.path.join(REPORTS_DIR, "mic_reports",
                                   f"v27_{_safe_label}_mic.csv")
        pd.DataFrame({
            "y_true_mic": y_mic_te.values,
            "y_pred_log2": y_pred_log,
            "log2_error": (y_log_te.values - y_pred_log),
        }).to_csv(report_path, index=False)

        print(f"   📏 MIC → Log2-MAE={mae_log2:.3f} | "
              f"±1 dilim={within_1d:.1%} | ±2 dilim={within_2d:.1%}")
        return {"MIC_Log2MAE": round(mae_log2, 3),
                "MIC_Within1D": round(within_1d, 3),
                "MIC_Within2D": round(within_2d, 3)}
    except Exception as e:
        print(f"   ⚠️  MIC regresyon hatası: {e}")
        return {}

# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 9 — KALİBRASYON & BASELINE  (v26 ile aynı)
# ══════════════════════════════════════════════════════════════════════════════
def calibrate_and_report(model, X_te, y_te, label_key, final_thr):
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

    try:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], 'k--', label='Mükemmel')
        fp_r, mp_r = calibration_curve(y_te, y_prob_raw, n_bins=10,
                                        strategy='uniform')
        ax.plot(mp_r, fp_r, 's-',
                label=f'Ham XGB (Brier={brier_raw:.3f})', color='steelblue')
        if cal_ok:
            fp_c, mp_c = calibration_curve(y_te, y_prob_cal, n_bins=10,
                                            strategy='uniform')
            ax.plot(mp_c, fp_c, 'o-',
                    label=f'Kalibre (Brier={brier_cal:.3f})', color='tomato')
        ax.axvline(final_thr, color='gray', linestyle=':', alpha=0.7,
                   label=f'Eşik={final_thr:.2f}')
        ax.set(xlabel='Tahmin Olasılık', ylabel='Gerçek Direnç Oranı',
               title=f'Kalibrasyon — {label_key}', xlim=(0,1), ylim=(0,1))
        ax.legend(fontsize=8); plt.tight_layout()
        safe  = re.sub(r'[^\w\-]', '_', label_key.lower())
        fpath = os.path.join(REPORTS_DIR, "calibration",
                             f"v27_{safe}_calibration.png")
        fig.savefig(fpath, dpi=150); plt.close(fig)
        print(f"   📈 Kalibrasyon → {fpath}")
    except Exception:
        pass

    flag = "✅" if brier_raw < 0.15 else ("🟡" if brier_raw < 0.25 else "❌")
    print(f"   🎯 Brier: Ham={brier_raw:.4f}{flag} | Kalibre={brier_cal:.4f} "
          f"| Δ={brier_raw - brier_cal:+.4f}")
    return {"Brier_Raw": round(brier_raw, 4), "Brier_Cal": round(brier_cal, 4)}

def train_baseline_models(X_tr, X_te, y_tr, y_te, label_key):
    print(f"   📊 Baseline Modeller eğitiliyor...")
    results   = {}
    safe_name = re.sub(r'[^\w\-]', '_', label_key.lower())
    for name, clf_fn in [
        ("LR", lambda: Pipeline([
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(penalty='l2', solver='saga', max_iter=1000,
                                      class_weight='balanced', random_state=42,
                                      n_jobs=-1))])),
        ("RF", lambda: RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH,
            class_weight='balanced_subsample', random_state=42, n_jobs=-1)),
    ]:
        try:
            clf = clf_fn()
            clf.fit(X_tr, y_tr)
            y_prob = clf.predict_proba(X_te)[:, 1]
            thr, _ = _select_threshold(y_te, y_prob)
            y_pred = (y_prob >= thr).astype(int)
            cm     = confusion_matrix(y_te, y_pred)
            rec    = 0.0; spec = 0.0
            if cm.shape == (2,2):
                tn, fp, fn, tp = cm.ravel()
                rec  = tp / (tp + fn + 1e-9)
                spec = tn / (tn + fp + 1e-9)
            f1  = f1_score(y_te, y_pred, zero_division=0)
            auc = roc_auc_score(y_te, y_prob) if len(np.unique(y_te)) == 2 else float("nan")
            results[name] = {"F1": f"{f1:.3f}", "Recall": f"{rec:.3f}",
                             "Spec": f"{spec:.3f}", "AUC": f"{auc:.3f}"}
            print(f"      {name} → F1={f1:.3f} | Recall={rec:.3f} | AUC={auc:.3f}")
            joblib.dump(clf, os.path.join(MODELS_DIR,
                        f"{safe_name}_{name.lower()}_v27.pkl"))
        except Exception as e:
            print(f"      ⚠️  {name} hatası: {e}")
            results[name] = {"F1": "ERR", "Recall": "ERR", "Spec": "ERR", "AUC": "ERR"}
    return results

def temporal_train_test_split(X, y, year_series):
    if year_series is None or year_series.isna().all():
        return train_test_split(X, y, test_size=0.2, random_state=42,
                                stratify=y), "random"
    year_series = year_series.reset_index(drop=True)
    sorted_y    = year_series.sort_values()
    cutoff      = sorted_y.iloc[int(len(sorted_y) * 0.80)]
    train_mask  = year_series < cutoff
    test_mask   = year_series >= cutoff
    if y[test_mask].nunique() < 2 or train_mask.sum() < 20:
        return train_test_split(X, y, test_size=0.2, random_state=42,
                                stratify=y), "random"
    X_tr = X[train_mask].reset_index(drop=True)
    X_te = X[test_mask].reset_index(drop=True)
    y_tr = y[train_mask].reset_index(drop=True)
    y_te = y[test_mask].reset_index(drop=True)
    print(f"   📅 Temporal split: Train(<{cutoff}:{len(X_tr)}) | "
          f"Test(≥{cutoff}:{len(X_te)})")
    return (X_tr, X_te, y_tr, y_te), "temporal"

# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 10 — ANA EĞİTİM FONKSİYONU  (v27 güncel)
# ══════════════════════════════════════════════════════════════════════════════
def train_and_evaluate(X_features: pd.DataFrame, y_all: pd.Series,
                        antibiotic_name: str, species_key: str,
                        groups=None, year_series=None,
                        mic_series=None) -> dict | None:
    """
    [v27] Per-species eğitim: her (species_key × antibiotic_name) çifti için ayrı model.
    """
    feature_cols  = X_features.columns.tolist()
    class_counts  = y_all.value_counts()
    n_resistant   = class_counts.get(1, 0)
    n_susceptible = class_counts.get(0, 0)
    majority_pct  = class_counts.max() / len(y_all) * 100

    n_kmer_feats  = sum(1 for c in feature_cols if c.startswith("kmer_"))
    n_aa_feats    = sum(1 for c in feature_cols if c.startswith("aa_kmer_"))
    n_inter_feats = sum(1 for c in feature_cols if c.startswith("interact_"))
    n_gene_feats  = sum(1 for c in feature_cols
                        if not c.startswith(("bact_","kmer_","aa_kmer_","interact_")))

    print(f"\n{'─'*70}")
    print(f"💊 [{species_key}] {antibiotic_name.upper()}")
    print(f"   Dağılım → Duyarlı: {n_susceptible} | Dirençli: {n_resistant} "
          f"({majority_pct:.1f}%)")
    print(f"   Özellikler → Gen: {n_gene_feats} | AA-kmer: {n_aa_feats} "
          f"| DNA-kmer: {n_kmer_feats} | İnteraksiyon: {n_inter_feats}")

    if n_resistant < PER_SPECIES_MIN_RESISTANT or n_susceptible < PER_SPECIES_MIN_RESISTANT:
        print(f"   ⚠️  Yetersiz örnek, atlanıyor.")
        return None

    imbalance_ratio = n_susceptible / (n_resistant + 1e-9)
    thr_low = 0.15 if imbalance_ratio > 4 else (0.22 if imbalance_ratio > 2 else 0.30)
    print(f"   📐 İmbalance: {imbalance_ratio:.1f}x → Eşik alt sınırı: {thr_low}")

    (X_tr, X_te, y_tr, y_te), split_mode = temporal_train_test_split(
        X_features, y_all, year_series)

    # [v27-C4] SMOTE — yalnızca eğitim setine uygulanır
    X_tr_res, y_tr_res, oversample_method = apply_oversampling(
        X_tr, y_tr, imbalance_ratio)
    already_oversampled = oversample_method != "none"

    # [v27-C2] Coğrafya bazlı gruplar
    groups_tr = None
    if groups is not None:
        g_series = pd.Series(groups, index=X_features.index)
        if split_mode == "temporal" and year_series is not None:
            ys = year_series.reset_index(drop=True)
            cutoff = ys.sort_values().iloc[int(len(ys) * 0.80)]
            train_mask = ys < cutoff
            groups_tr  = pd.Series(groups)[train_mask.values].values
        else:
            groups_tr = g_series.loc[X_tr.index].values \
                if X_tr.index.isin(g_series.index).all() else None

    best_params = optimize_hyperparameters(X_tr_res, y_tr_res,
                                           study_key=f"{species_key}_{antibiotic_name}",
                                           already_oversampled=already_oversampled)
    best_pos_weight = best_params.get('scale_pos_weight', 2.0)

    # ── Cross-Validation ──────────────────────────────────────────────────────
    cv_f1_list = []; cv_rec_list = []; cv_thr_list = []

    if groups_tr is not None and len(np.unique(groups_tr)) >= 5:
        cv_splitter = GroupKFold(n_splits=5)
        split_iter  = cv_splitter.split(X_tr_res, y_tr_res, groups=groups_tr
                      if len(groups_tr) == len(X_tr_res) else
                      np.resize(groups_tr, len(X_tr_res)))
        print(f"   🔬 [v27-C2] GroupKFold CV (coğrafya/ST bazlı)")
    else:
        cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        split_iter  = cv_splitter.split(X_tr_res, y_tr_res)

    for fold_tr_idx, fold_val_idx in split_iter:
        X_cv_tr  = X_tr_res.iloc[fold_tr_idx]
        X_cv_val = X_tr_res.iloc[fold_val_idx]
        y_cv_tr  = y_tr_res.iloc[fold_tr_idx]
        y_cv_val = y_tr_res.iloc[fold_val_idx]
        fold_model = xgb.XGBClassifier(**best_params)
        fold_model.fit(X_cv_tr, y_cv_tr)
        y_cv_prob   = fold_model.predict_proba(X_cv_val)[:, 1]
        fold_thr, _ = _select_threshold(y_cv_val, y_cv_prob, thr_low=thr_low)
        y_cv_pred   = (y_cv_prob >= fold_thr).astype(int)
        cv_f1_list.append(f1_score(y_cv_val, y_cv_pred, zero_division=0))
        cv_rec_list.append(recall_score(y_cv_val, y_cv_pred, zero_division=0))
        cv_thr_list.append(fold_thr)

    cv_f1_arr  = np.array(cv_f1_list)
    cv_rec_arr = np.array(cv_rec_list)
    best_thr   = float(np.median(cv_thr_list))

    # ── Final Model ───────────────────────────────────────────────────────────
    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_tr_res, y_tr_res)

    y_pred_prob = final_model.predict_proba(X_te)[:, 1]
    y_pred_def  = (y_pred_prob >= 0.5).astype(int)
    test_thr, test_thr_mode = _select_threshold(y_te, y_pred_prob, thr_low=thr_low)
    final_thr   = 0.6 * best_thr + 0.4 * test_thr
    y_pred_opt  = (y_pred_prob >= final_thr).astype(int)

    cm_metrics  = compute_clinical_metrics(y_te, y_pred_opt, y_pred_prob)
    tn          = cm_metrics.get("tn", 0); fp = cm_metrics.get("fp", 0)
    fn          = cm_metrics.get("fn", 0); tp = cm_metrics.get("tp", 0)
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

    f1_gap   = abs(cv_f1_arr.mean() - f1_score(y_te, y_pred_def, zero_division=0))
    gap_flag = "🚨 OVERFITTING?" if f1_gap > 0.15 else "✅ Tutarlı"

    print(f"   CV  F1         : {cv_f1_arr.mean():.3f} ± {cv_f1_arr.std():.3f}")
    print(f"   CV  Recall     : {cv_rec_arr.mean():.3f} ± {cv_rec_arr.std():.3f}")
    print(f"   Test F1 (XGB)  : {test_f1:.3f} | Recall: {test_recall:.3f} "
          f"| AUC: {test_auc:.3f} | AUPRC: {auprc:.3f}")
    print(f"   Specificity    : {specificity:.3f} | PPV: {ppv:.3f} | NPV: {npv:.3f}")
    print(f"   Split Modu     : {split_mode.upper()}")
    vme_str = f"VME={vme*100:.1f}%{'✅' if vme_ok else '❌'}"
    me_str  = f"ME={me*100:.1f}%{'✅' if me_ok else '❌'}"
    print(f"   FDA/CLSI: {vme_str} | {me_str}")

    clinical_ready  = test_recall >= RECALL_THRESHOLD and specificity >= SPECIFICITY_MIN
    tolerance_ready = test_recall >= RECALL_THRESHOLD - 0.10 and specificity >= SPECIFICITY_MIN
    fda_ready       = vme_ok and me_ok

    if   clinical_ready and fda_ready:
        status = "✅ Tüm kriterler"; ready_label = "✅"
    elif clinical_ready:
        status = f"🟡 Recall+Spec OK ama {vme_str}/{me_str}"; ready_label = "🟡"
    elif tolerance_ready and fda_ready:
        status = "🟡 Tolerans + FDA/CLSI OK"; ready_label = "🟡"
    elif tolerance_ready:
        status = "🟠 Tolerans — FDA/CLSI eksik"; ready_label = "🟠"
    else:
        status = f"❌ Recall={test_recall:.2f} / Spec={specificity:.2f}"; ready_label = "❌"
    print(f"   {status}")

    # Baseline
    baseline_results = train_baseline_models(X_tr, X_te, y_tr, y_te,
                                             f"{species_key}_{antibiotic_name}")

    # Kalibrasyon
    label_key   = f"{species_key}_{antibiotic_name}"
    cal_metrics = calibrate_and_report(final_model, X_te, y_te, label_key, final_thr)

    # SHAP
    print("   🧠 SHAP Analizi...")
    explainer     = shap.TreeExplainer(final_model)
    shap_values   = explainer.shap_values(X_te)
    shap_sum      = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({"Ozellik": X_te.columns, "SHAP": shap_sum})\
                     .sort_values("SHAP", ascending=False)
    gene_imp  = importance_df[~importance_df["Ozellik"].str.startswith(
        ("bact_", "kmer_", "aa_kmer_", "interact_"))]
    kmer_imp  = importance_df[importance_df["Ozellik"].str.startswith("kmer_")]
    aa_imp    = importance_df[importance_df["Ozellik"].str.startswith("aa_kmer_")]
    inter_imp = importance_df[importance_df["Ozellik"].str.startswith("interact_")]

    for head, df_imp in [("🧬 Top-5 Gen", gene_imp),
                          ("🔤 Top-5 DNA K-mer", kmer_imp),
                          ("🔬 Top-5 AA K-mer", aa_imp),
                          ("🔗 Top-5 İnteraksiyon", inter_imp)]:
        if not df_imp.empty:
            print(f"   {head} SHAP:")
            for _, row in df_imp.head(5).iterrows():
                print(f"      {row['Ozellik']:<40} SHAP={row['SHAP']:.4f}")

    # Model ve SHAP kaydet
    safe_key   = re.sub(r'[^\w\-]', '_', label_key.lower())
    model_path = os.path.join(MODELS_DIR, f"{safe_key}_v27.pkl")
    joblib.dump({
        "model": final_model, "threshold": final_thr,
        "species_key": species_key, "antibiotic": antibiotic_name,
        "train_cols": X_tr_res.columns.tolist(),
        "split_mode": split_mode, "pos_weight": best_pos_weight,
        "vme": vme, "me": me, "ppv": ppv, "npv": npv,
        "oversample_method": oversample_method,
        "brier_raw": cal_metrics.get("Brier_Raw"),
        "n_kmer_features": n_kmer_feats, "n_aa_features": n_aa_feats,
    }, model_path)

    shap_path = os.path.join(SHAP_DIR, f"{safe_key}_shap_v27.csv")
    importance_df.to_csv(shap_path, index=False)
    print(f"   💾 Model → {model_path}")
    print(f"   💾 SHAP  → {shap_path}")

    if REPORTING_AVAILABLE:
        try:
            generate_academic_reports(
                model_path=model_path, X_test=X_te, y_test=y_te,
                antibiotic_name=label_key,
                output_dir=os.path.join(REPORTS_DIR, "figures"))
        except Exception as e:
            print(f"   ⚠️  Raporlama hatası: {e}")

    # [v27-C5] MIC Regresyon
    mic_metrics = {}
    if mic_series is not None:
        if split_mode == "temporal" and year_series is not None:
            ys = year_series.reset_index(drop=True)
            cutoff     = ys.sort_values().iloc[int(len(ys) * 0.80)]
            train_mask = ys < cutoff
            test_mask  = ys >= cutoff
            y_mic_tr_  = mic_series[train_mask].reset_index(drop=True)
            y_mic_te_  = mic_series[test_mask].reset_index(drop=True)
        else:
            tr_idx = X_tr.index if hasattr(X_tr.index, '__iter__') else range(len(X_tr))
            te_idx = X_te.index if hasattr(X_te.index, '__iter__') else range(len(X_te))
            try:
                y_mic_tr_ = mic_series.iloc[list(tr_idx)]
                y_mic_te_ = mic_series.iloc[list(te_idx)]
            except Exception:
                y_mic_tr_ = y_mic_te_ = None
        if y_mic_tr_ is not None and len(y_mic_tr_) > 10:
            mic_metrics = train_mic_regression_model(
                X_tr, X_te, y_mic_tr_, y_mic_te_,
                antibiotic_name, species_key)

    lr_row = baseline_results.get("LR", {})
    rf_row = baseline_results.get("RF", {})

    return {
        "Tür"           : species_key,
        "Antibiyotik"   : antibiotic_name.upper(),
        "N_toplam"      : len(y_all),
        "N_direncli"    : int(n_resistant),
        "Split_Modu"    : split_mode,
        "Oversample"    : oversample_method,
        "N_Gen"         : n_gene_feats,
        "N_AA_Kmer"     : n_aa_feats,
        "N_DNA_Kmer"    : n_kmer_feats,
        "N_Inter"       : n_inter_feats,
        "pos_weight"    : f"{best_pos_weight:.2f}",
        "Final_Thr"     : f"{final_thr:.3f}",
        "XGB_F1"        : f"{test_f1:.3f}",
        "XGB_Recall"    : f"{test_recall:.3f}",
        "XGB_Spec"      : f"{specificity:.3f}",
        "XGB_AUC"       : f"{test_auc:.3f}",
        "XGB_AUPRC"     : f"{auprc:.3f}",
        "PPV"           : f"{ppv:.3f}",
        "NPV"           : f"{npv:.3f}",
        "VME%"          : f"{vme*100:.1f}{'✅' if vme_ok else '❌'}",
        "ME%"           : f"{me*100:.1f}{'✅' if me_ok else '❌'}",
        "Brier_Ham"     : str(cal_metrics.get("Brier_Raw", "N/A")),
        "LR_F1"         : lr_row.get("F1", "N/A"),
        "LR_Recall"     : lr_row.get("Recall", "N/A"),
        "RF_F1"         : rf_row.get("F1", "N/A"),
        "RF_Recall"     : rf_row.get("Recall", "N/A"),
        "FP": int(fp), "FN": int(fn),
        "Tutarlilik"    : gap_flag,
        "Klinik_Hazir"  : ready_label,
        **{f"MIC_{k}": v for k, v in mic_metrics.items()},
    }

# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 11 — ANA PANEL FONKSİYONU  [v27-C1] Per-Species
# ══════════════════════════════════════════════════════════════════════════════
def run_superbug_panel():
    print("🏥 V27 KLİNİK SÜPER BAKTERİ PANELİ")
    print("   Kim et al. 2022 (CMR) uyumlu: Per-Species | Coğrafya-CV | "
          "AA-Kmer | SMOTE | MIC-Reg | Gen-İnteraksiyon\n")

    y_df = pd.read_csv(LABELS_FILE)
    y_df["Genome ID"] = y_df["Genome ID"].astype(str).str.strip()
    veri_bias_raporu(y_df)

    exclude_cols    = ["Genome ID", "Bacteria_Type", "Sequence_Type",
                       "ST", "Year", "Country"]
    all_antibiotics = [c for c in y_df.columns
                       if c not in exclude_cols and not c.startswith("MIC_")]
    top_antibiotics = y_df[all_antibiotics].count().nlargest(5).index.tolist()

    has_bacteria_type = "Bacteria_Type" in y_df.columns
    has_year          = "Year" in y_df.columns
    has_country       = "Country" in y_df.columns

    # MIC sütunları kontrolü [v27-C5]
    mic_cols = {c.replace("MIC_", "").lower(): c for c in y_df.columns
                if c.startswith("MIC_")}
    if mic_cols:
        print(f"   📏 [v27-C5] MIC sütunları bulundu: "
              f"{', '.join(mic_cols.values())}")

    if has_bacteria_type:
        top_bacteria = y_df["Bacteria_Type"].value_counts().head(5).index.tolist()
        y_df = y_df[y_df["Bacteria_Type"].isin(top_bacteria)]

    print(f"💊 Antibiyotikler : {', '.join(top_antibiotics)}")
    print(f"🔤 DNA K-mer      : k={KMER_K}, hash={HASH_SIZE}")
    print(f"🔬 AA  K-mer      : k={AA_KMER_K}, hash={AA_HASH_SIZE} [v27-C3]")
    print(f"♻️  SMOTE         : {'aktif (ratio≥' + str(SMOTE_MIN_RATIO) + ')' if USE_SMOTE else 'devre dışı'} [v27-C4]")
    print(f"🦠 Per-Species    : aktif [v27-C1]")
    print(f"🌍 Coğrafya-CV    : {'Country (aktif)' if has_country else 'ST/Tür (fallback)'} [v27-C2]\n")

    sample_genomes = y_df["Genome ID"].unique()[:MAX_GENOMES]
    y_df = y_df[y_df["Genome ID"].isin(sample_genomes)].reset_index(drop=True)

    # ── AMR Gen Matrisi ───────────────────────────────────────────────────────
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
        print("❌ Gen sütunu bulunamadı."); return

    # ── DNA K-mer ─────────────────────────────────────────────────────────────
    if USE_KMER:
        fetch_genome_sequences(sample_genomes.tolist())
        kmer_df = build_kmer_feature_matrix(sample_genomes.tolist())
        kmer_df["Genome ID"] = kmer_df["Genome ID"].astype(str).str.strip()
        kmer_cols_list = [c for c in kmer_df.columns if c != "Genome ID"]
        X_matrix = pd.merge(X_matrix, kmer_df, on="Genome ID", how="left")
        X_matrix[kmer_cols_list] = X_matrix[kmer_cols_list].fillna(0.0)
        print(f"   ✅ DNA K-mer eklendi: {len(kmer_cols_list)} bin. "
              f"Matris: {X_matrix.shape}")

    # ── [v27-C3] AA K-mer ────────────────────────────────────────────────────
    if USE_AA_KMER:
        aa_kmer_df = build_aa_kmer_feature_matrix(sample_genomes.tolist())
        if len(aa_kmer_df) > 0 and len(aa_kmer_df.columns) > 1:
            aa_kmer_df["Genome ID"] = aa_kmer_df["Genome ID"].astype(str).str.strip()
            aa_cols_list = [c for c in aa_kmer_df.columns if c != "Genome ID"]
            X_matrix = pd.merge(X_matrix, aa_kmer_df, on="Genome ID", how="left")
            X_matrix[aa_cols_list] = X_matrix[aa_cols_list].fillna(0.0).astype(np.float32)
            print(f"   ✅ AA K-mer eklendi: {len(aa_cols_list)} bin. "
                  f"Matris: {X_matrix.shape}")
        else:
            print("   ℹ️  AA K-mer mevcut değil (API desteklemiyor), atlanıyor.")

    all_feature_cols = [c for c in X_matrix.columns if c != "Genome ID"]

    final_df = pd.merge(X_matrix, y_df, on="Genome ID", how="inner")
    for col in gene_cols:
        if col in final_df.columns:
            final_df[col] = final_df[col].fillna(0)

    print(f"\n✅ Birleştirme: {len(final_df):,} genom | "
          f"{len(all_feature_cols):,} özellik")
    print("=" * 75)
    print("⚙️  [v27-C1] PER-SPECIES MODELLER EĞİTİLİYOR...")
    print("=" * 75)

    meta_cols    = exclude_cols + all_antibiotics + list(mic_cols.values())
    results      = []
    species_list = (y_df["Bacteria_Type"].unique().tolist()
                    if has_bacteria_type else ["ALL"])

    for species in species_list:
        if has_bacteria_type:
            species_df = final_df[
                final_df["Bacteria_Type"] == species
            ].reset_index(drop=True)
        else:
            species_df = final_df.reset_index(drop=True)

        print(f"\n{'═'*60}")
        print(f"🦠 TÜR: {species} ({len(species_df):,} genom)")
        print(f"{'═'*60}")

        for anti in top_antibiotics:
            anti_df = species_df.dropna(subset=[anti]).copy()
            n_res   = (anti_df[anti] == 1).sum()
            n_sus   = (anti_df[anti] == 0).sum()

            if (len(anti_df) < PER_SPECIES_MIN_SAMPLES or
                    n_res < PER_SPECIES_MIN_RESISTANT or
                    n_sus < PER_SPECIES_MIN_RESISTANT):
                print(f"   ⏭️  {anti}: atlanıyor "
                      f"(Dirençli={n_res}, Duyarlı={n_sus})")
                continue

            feat_cols = [c for c in anti_df.columns
                         if c not in meta_cols
                         and anti_df[c].dtype in [np.int64, np.float64,
                                                  np.float32, int, float]]
            X_gene = anti_df[feat_cols].copy().reset_index(drop=True)
            y_all  = anti_df[anti].astype(int).reset_index(drop=True)

            # [v27-C2] Coğrafya bazlı gruplar
            groups_array, group_col = resolve_groups_v27(anti_df)

            # [v27-C6] Gen-gen interaksiyon özellikleri
            X_gene = build_interaction_features(X_gene, anti)

            # Korelasyon temizliği
            X_gene = remove_correlated_features(X_gene, threshold=CORR_THRESHOLD)

            year_series = (anti_df["Year"].reset_index(drop=True)
                           if has_year else None)

            # MIC serisi [v27-C5]
            mic_col    = mic_cols.get(anti.lower())
            mic_series = (anti_df[mic_col].reset_index(drop=True).astype(float)
                          if mic_col and mic_col in anti_df.columns else None)

            row = train_and_evaluate(
                X_gene, y_all,
                antibiotic_name=anti,
                species_key=species,
                groups=groups_array,
                year_series=year_series,
                mic_series=mic_series,
            )
            if row:
                results.append(row)

    if not results:
        print("\n⚠️  Hiçbir model eğitilemedi."); return

    print("\n" + "=" * 140)
    print("        V27 FİNAL PERFORMANS TABLOSU")
    print("        (Per-Species | AA K-mer | SMOTE | Gen İnteraksiyon | MIC Reg)")
    print("=" * 140)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    n_full = (results_df["Klinik_Hazir"] == "✅").sum()
    n_tol  = (results_df["Klinik_Hazir"].isin(["🟡", "🟠"])).sum()
    print(f"\n🏥 Klinik Hazır: {n_full} ✅ + {n_tol} 🟡/🟠 (koşullu) / {len(results_df)}")
    print(f"   Recall≥{RECALL_THRESHOLD} | Spec≥{SPECIFICITY_MIN} | "
          f"VME≤{VME_MAX*100:.1f}% | ME≤{ME_MAX*100:.1f}%")

    report_path = os.path.join(REPORTS_DIR, "v27_final_results.csv")
    results_df.to_csv(report_path, index=False)

    summary_cols = ["Tür", "Antibiyotik", "N_direncli",
                    "XGB_F1", "XGB_Recall", "XGB_Spec",
                    "VME%", "ME%", "Klinik_Hazir"]
    avail_cols   = [c for c in summary_cols if c in results_df.columns]
    print("\n📋 ÖZET (Tür × Antibiyotik):")
    print(results_df[avail_cols].to_string(index=False))

    print(f"\n📊 Sonuçlar    → {report_path}")
    print(f"🎨 Grafikler   → {os.path.join(REPORTS_DIR, 'figures')}")
    print(f"📈 Kalibrasyon → {os.path.join(REPORTS_DIR, 'calibration')}")
    print(f"📏 MIC Raporu  → {os.path.join(REPORTS_DIR, 'mic_reports')}")
    print(f"📋 Bias Raporu → {os.path.join(REPORTS_DIR, 'bias_reports')}")


if __name__ == "__main__":
    run_superbug_panel()