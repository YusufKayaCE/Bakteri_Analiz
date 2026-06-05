# main_v29_colab.py
# Kim et al. 2022 (Clin Microbiol Rev 35(3):e00179-21) Uyumlu AMR Tahmin Sistemi
# Google Colab optimize — v27 temel, v28 cloud iyileştirmeleri + tüm bug fix'ler
#
# ══════════════════════════════════════════════════════════════════════════════
# KULLANIM (Google Colab):
#
#   Hücre 1 — Kurulum (oturum başında bir kez):
#     from main_v29_colab import _colab_install; _colab_install()
#
#   Hücre 2 — Çalıştır:
#     from main_v29_colab import run_superbug_panel; run_superbug_panel()
#
#   Veri yolu (Drive bağlandıktan sonra):
#     /content/drive/MyDrive/amr_v29/data/processed/v2_multilabel_labels.csv
# ══════════════════════════════════════════════════════════════════════════════
#
# v27 → v29 DEĞİŞİKLİKLERİ
# ──────────────────────────────────────────────────────────────────────────────
# [FIX-1] CORR_THRESHOLD, OPTUNA_TRIALS, MAX_GENOMES, RF_N_ESTIMATORS,
#         RF_MAX_DEPTH eksik sabitler eklendi (v28'de kaldırılmıştı → NameError)
# [FIX-2] build_kmer_feature_matrix: hash_size → HASH_SIZE yazım hatası düzeltildi
# [FIX-3] MIC index kayması: .iloc[tr_idx] → .loc[X_tr.index] ile düzeltildi
# [FIX-4] calibrate_and_report: dummy yerine gerçek isotonic kalibrasyon
# [FIX-5] fetch_amr_genes_from_bvbrc v28'de eksikti, v27'den geri getirildi
# [COLAB-1] torch.cuda.is_available() ile GPU tespiti (CUDA hardcode kaldırıldı)
# [COLAB-2] Google Drive entegrasyonu: model/rapor kalıcı saklamak için
# [COLAB-3] Checkpoint sistemi: her (tür×antibiyotik) sonrası CSV kayıt/devam
# [COLAB-4] psutil RAM monitörü + OOM uyarısı
# [CLOUD-1] Hash boyutları: DNA 2^16=65 536, AA 2^14=16 384
# [CLOUD-2] OPTUNA_TRIALS=50, n_estimators 100-600, colsample_bytree 0.40-0.80
# [CLOUD-3] INTERACTION_TOP_N=25 → 300+ gen çifti epistasis uzayı
# [KIM-1]  Soft-voting ensemble: XGB+LR+RF olasılık ortalaması
# [KIM-2]  Bootstrap %95 güven aralıkları: Recall/Spec/F1/AUC için
# ══════════════════════════════════════════════════════════════════════════════

import subprocess
import sys

# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 0 — COLAB KURULUM YARDIMCI FONKSİYONLARI
# ══════════════════════════════════════════════════════════════════════════════

_REQUIRED_PACKAGES = [
    "xgboost>=2.0.0",
    "optuna>=3.0",
    "shap>=0.44",
    "imbalanced-learn>=0.11",
    "pyarrow>=14.0",
    "mmh3>=4.0",
    "scikit-learn>=1.3",
    "psutil",
]


def _colab_install():
    """Colab oturumu başında gerekli paketleri yükler."""
    print("📦 Paketler yükleniyor...")
    for pkg in _REQUIRED_PACKAGES:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", pkg],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    print("✅ Tüm paketler hazır.")


def _mount_drive() -> str:
    """
    [COLAB-2] Google Drive'ı bağlar ve proje kök dizinini döndürür.
    Colab dışında '..' döndürür (yerel geliştirme ortamı).
    """
    try:
        from google.colab import drive  # type: ignore

        drive.mount("/content/drive", force_remount=False)
        base = "/content/drive/MyDrive/amr_v29"
        import os

        os.makedirs(base, exist_ok=True)
        print(f"📂 Google Drive bağlandı → {base}")
        return base
    except ImportError:
        print("📂 Yerel ortam → proje kök: '..'")
        return ".."


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 1 — IMPORTS
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# MurmurHash (opsiyonel — hız için)
try:
    import mmh3

    def _murmurhash(s: str, size: int) -> int:
        return mmh3.hash(s, signed=False) % size

except ImportError:

    def _murmurhash(s: str, size: int) -> int:
        return (
            int.from_bytes(hashlib.md5(s.encode()).digest()[:4], byteorder="little")
            % size
        )


# imbalanced-learn
try:
    from imblearn.over_sampling import BorderlineSMOTE, RandomOverSampler

    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("⚠️  imbalanced-learn eksik → _colab_install() çalıştırın.")

# RAM monitörü [COLAB-4]
try:
    import psutil

    def _ram_gb() -> float:
        return psutil.virtual_memory().available / 1e9

    def _ram_pct() -> float:
        return psutil.virtual_memory().percent

except ImportError:

    def _ram_gb() -> float:
        return float("inf")

    def _ram_pct() -> float:
        return 0.0


from sklearn.model_selection import (
    GroupKFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    recall_score,
    roc_auc_score,
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold

try:
    from reporting_module import generate_academic_reports  # type: ignore

    REPORTING_AVAILABLE = True
except ImportError:
    REPORTING_AVAILABLE = False

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 2 — SABİTLER & YAPILANDIRMA
# ══════════════════════════════════════════════════════════════════════════════

# ── [COLAB-1] GPU Tespiti ─────────────────────────────────────────────────────
def _detect_device() -> str:
    """
    [COLAB-1] torch.cuda.is_available() ile Colab GPU tespiti.
    v28'deki DEVICE='cuda' hardcode hatası giderildi.
    """
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"🚀 GPU: {name} → XGBoost CUDA+hist")
            return "cuda"
    except ImportError:
        pass
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            print(f"🚀 GPU: {r.stdout.strip().splitlines()[0]} → CUDA+hist")
            return "cuda"
    except Exception:
        pass
    print("⚠️  GPU bulunamadı → CPU modu")
    return "cpu"


DEVICE: str = _detect_device()
FALLBACK_TO_CPU: bool = False

_OOM_KEYWORDS = [
    "out of memory",
    "cudaerrormemoryallocation",
    "memory allocation error",
    "free memory: 0b",
]


def _is_oom_error(exc: Exception) -> bool:
    chain = " ".join(
        [
            str(exc).lower(),
            str(getattr(exc, "__cause__", "")).lower(),
            str(getattr(exc, "__context__", "")).lower(),
        ]
    )
    return any(kw in chain for kw in _OOM_KEYWORDS)


# ── FDA/CLSI Klinik Kabul Eşikleri ───────────────────────────────────────────
VME_MAX = 0.015  # Very Major Error  < %1.5
ME_MAX = 0.030  # Major Error       < %3.0
RECALL_THRESHOLD = 0.80
SPECIFICITY_MIN = 0.50

# ── [COLAB-2] Dizin Yapısı ────────────────────────────────────────────────────
_BASE = "/content/drive/MyDrive/amr_v29"

DATA_DIR = os.path.join(_BASE, "data", "processed")
LABELS_FILE = os.path.join(DATA_DIR, "v2_multilabel_labels.csv")
CACHE_FILE = os.path.join(DATA_DIR, "v17_amr_genes_cache.csv")
SEQ_CACHE_FILE = os.path.join(DATA_DIR, "v26_sequences.parquet")
KMER_CACHE_FILE = os.path.join(DATA_DIR, "v29_kmer_hashed_features.parquet")
AA_KMER_CACHE = os.path.join(DATA_DIR, "v29_aa_kmer_features.parquet")
AA_SEQ_CACHE = os.path.join(DATA_DIR, "v29_amr_aa_sequences.parquet")
MODELS_DIR = os.path.join(_BASE, "models")
REPORTS_DIR = os.path.join(_BASE, "reports")
SHAP_DIR = os.path.join(_BASE, "reports", "shap_values")
CHECKPOINT_FILE = os.path.join(REPORTS_DIR, "v29_checkpoint.csv")

for _d in [
    DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    SHAP_DIR,
    os.path.join(REPORTS_DIR, "figures"),
    os.path.join(REPORTS_DIR, "calibration"),
    os.path.join(REPORTS_DIR, "bias_reports"),
    os.path.join(REPORTS_DIR, "mic_reports"),
]:
    os.makedirs(_d, exist_ok=True)

OPTUNA_DB_FILE = "sqlite:///" + os.path.abspath(MODELS_DIR) + "/optuna_v29.db"

# ── [FIX-1] Eksik Sabitler ────────────────────────────────────────────────────
MAX_GENOMES = 30_000
BATCH_SIZE = 100          # [CLOUD-2] v27=50 → v29=100
OPTUNA_TRIALS = 50        # [CLOUD-2] v27=30 → v29=50
OPTUNA_N_JOBS = 1
MODEL_N_JOBS = 1 if DEVICE == "cuda" else -1
BACT_MIN_FREQ = 0.05
CORR_THRESHOLD = 0.95     # Pearson korelasyon eşiği
RF_N_ESTIMATORS = 300     # Baseline RF
RF_MAX_DEPTH = 10         # Baseline RF

# ── [CLOUD-1] DNA K-mer ───────────────────────────────────────────────────────
KMER_K = 21
HASH_BITS = 16            # 2^16 = 65 536  (v27=15=32 768)
HASH_SIZE = 2**HASH_BITS
SEQ_SAMPLE_BP = 200_000
SEQ_BATCH_SIZE = 100      # v27=50 → v29=100
USE_KMER = True

# ── [CLOUD-1] AA K-mer ────────────────────────────────────────────────────────
USE_AA_KMER = True
AA_KMER_K = 4
AA_HASH_BITS = 14         # 2^14 = 16 384  (v27=13=8 192)
AA_HASH_SIZE = 2**AA_HASH_BITS

# ── SMOTE ─────────────────────────────────────────────────────────────────────
USE_SMOTE = True
SMOTE_MIN_RATIO = 2.0

# ── Per-Species Eşikleri ─────────────────────────────────────────────────────
PER_SPECIES_MIN_SAMPLES = 50
PER_SPECIES_MIN_RESISTANT = 15

# ── [CLOUD-3] Gen Etkileşim ───────────────────────────────────────────────────
USE_GENE_INTERACTIONS = True
INTERACTION_TOP_N = 25    # v27=12 → v29=25 (300+ çift)

AMR_KNOWN_INTERACTIONS = {
    "ciprofloxacin": [
        ("DNA gyrase subunit A", "DNA topoisomerase IV"),
        ("gyrase subunit A", "NorA"),
        ("gyrase", "MexAB"),
    ],
    "ceftazidime": [
        ("CTX-M", "SHV"),
        ("CTX-M", "KPC"),
        ("CTX-M", "OXA"),
    ],
    "gentamicin": [
        ("AAC(3)", "AAC(6')"),
        ("AAC(3)", "APH(2'')"),
        ("APH", "ANT"),
    ],
    "ampicillin": [
        ("TEM", "SHV"),
        ("TEM", "CTX-M"),
        ("TEM", "OXA"),
    ],
    "tetracycline": [
        ("Tet(", "NorA"),
        ("Tet(", "MFS"),
        ("Tet(", "efflux"),
    ],
}

print(f"📐 DNA K-mer : k={KMER_K}, hash={HASH_SIZE:,} [2^{HASH_BITS}]")
print(f"📐 AA  K-mer : k={AA_KMER_K}, hash={AA_HASH_SIZE:,} [2^{AA_HASH_BITS}]")
print(f"💾 Kullanılabilir RAM : {_ram_gb():.1f} GB")

# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 3 — [COLAB-3] CHECKPOINT SİSTEMİ
# ══════════════════════════════════════════════════════════════════════════════


def load_checkpoint() -> tuple[list, set]:
    """
    [COLAB-3] Önceki oturumdan tamamlanan sonuçları yükler.
    Colab session kopması durumunda kaldığı yerden devam eder.
    """
    if os.path.exists(CHECKPOINT_FILE):
        try:
            df = pd.read_csv(CHECKPOINT_FILE)
            completed = set(
                df[["Tür", "Antibiyotik"]]
                .apply(lambda r: f"{r['Tür']}|{r['Antibiyotik']}", axis=1)
                .tolist()
            )
            print(f"   ♻️  Checkpoint: {len(df)} tamamlanmış model yüklendi.")
            return df.to_dict("records"), completed
        except Exception:
            pass
    return [], set()


def save_checkpoint(results: list) -> None:
    """Her model sonrası checkpoint'e yazar — session kopmalarına karşı korur."""
    try:
        pd.DataFrame(results).to_csv(CHECKPOINT_FILE, index=False)
    except Exception as e:
        print(f"   ⚠️  Checkpoint yazılamadı: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 4 — DNA K-MER FONKSİYONLARI
# ══════════════════════════════════════════════════════════════════════════════
_COMP_TABLE = str.maketrans("ACGT", "TGCA")


def _reverse_complement(seq: str) -> str:
    return seq.translate(_COMP_TABLE)[::-1]


def _canonical(kmer: str) -> str:
    rc = _reverse_complement(kmer)
    return kmer if kmer <= rc else rc


def extract_kmer_freq(
    seq: str, k: int = KMER_K, hash_size: int = HASH_SIZE
) -> np.ndarray:
    freq = np.zeros(hash_size, dtype=np.float32)
    seq = re.sub(r"[^ACGT]", "", seq.upper())
    L = len(seq)
    if L < k:
        return freq
    total = 0
    for i in range(L - k + 1):
        ck = _canonical(seq[i : i + k])
        freq[_murmurhash(ck, hash_size)] += 1
        total += 1
    if total > 0:
        freq /= total
    return freq


def _kmer_worker(args: tuple) -> np.ndarray:
    seq, k, hash_size = args
    return extract_kmer_freq(seq, k=k, hash_size=hash_size)


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 5 — AMİNO ASİT K-MER [v27-C3]
# ══════════════════════════════════════════════════════════════════════════════
_AA_ALPHABET = set("ACDEFGHIKLMNPQRSTVWY")


def _aa_kmer_worker(args: tuple) -> np.ndarray:
    seq, k, hash_size = args
    freq = np.zeros(hash_size, dtype=np.float32)
    seq = "".join(c for c in seq.upper() if c in _AA_ALPHABET)
    L = len(seq)
    if L < k:
        return freq
    total = 0
    for i in range(L - k + 1):
        freq[_murmurhash(seq[i : i + k], hash_size)] += 1
        total += 1
    if total > 0:
        freq /= total
    return freq


def fetch_amr_protein_sequences(genome_ids: list) -> dict:
    genome_ids = [str(g) for g in genome_ids]
    result: dict = {}

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
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
    }
    new_rows = []
    batch_seq_map: dict = {}

    for i in range(0, len(genome_ids), 50):
        batch = genome_ids[i : i + 50]
        id_str = ",".join(batch)
        payload = f"in(genome_id,({id_str}))&select(genome_id,aa_sequence)&limit(10000)"
        try:
            r = requests.post(
                "https://www.bv-brc.org/api/sp_gene/",
                headers=headers,
                data=payload,
                timeout=120,
            )
            if r.status_code == 200:
                for item in r.json():
                    gid = str(item.get("genome_id", "")).strip()
                    aa = str(item.get("aa_sequence", "")).strip()
                    if gid and aa and aa.lower() not in ("", "none", "null"):
                        batch_seq_map[gid] = batch_seq_map.get(gid, "") + aa
        except Exception:
            pass
        pct = min(100, int((i + 50) / len(genome_ids) * 100))
        print(f"\r   AA protein %{pct:3d}", end="", flush=True)
        time.sleep(0.2)

    print()
    result.update(batch_seq_map)
    for gid, seq in batch_seq_map.items():
        new_rows.append({"genome_id": gid, "aa_seq": seq})

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        if os.path.exists(AA_SEQ_CACHE):
            try:
                old = pd.read_parquet(AA_SEQ_CACHE)
                new_df = (
                    pd.concat([old, new_df], ignore_index=True)
                    .drop_duplicates("genome_id", keep="last")
                )
            except Exception:
                pass
        _write_parquet_chunked(new_df, AA_SEQ_CACHE)

    if not result:
        print("   ⚠️  API aa_sequence döndürmedi → AA k-mer bu oturumda devre dışı.")
    return result


def build_aa_kmer_feature_matrix(genome_ids: list) -> pd.DataFrame:
    k = AA_KMER_K
    hash_size = AA_HASH_SIZE
    aa_cols = [f"aa_kmer_h{i}" for i in range(hash_size)]

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

    print(f"\n🔬 AA K-mer Matrisi (k={k}, hash={hash_size:,})...")
    prot_seqs = fetch_amr_protein_sequences(genome_ids)
    genome_ids_s = [str(g) for g in genome_ids]
    valid_gids = [g for g in genome_ids_s if g in prot_seqs]

    if not valid_gids:
        print("   ⚠️  AA k-mer matrisi boş.")
        return pd.DataFrame(columns=["Genome ID"] + aa_cols)

    args_list = [(prot_seqs[g], k, hash_size) for g in valid_gids]
    results = joblib.Parallel(n_jobs=-1, backend="loky")(
        joblib.delayed(_aa_kmer_worker)(a) for a in args_list
    )

    aa_kmer_df = pd.DataFrame(np.vstack(results).astype(np.float32), columns=aa_cols)
    aa_kmer_df.insert(0, "Genome ID", valid_gids)

    vt = VarianceThreshold(threshold=0.0)
    vt.fit(aa_kmer_df[aa_cols])
    kept = [c for c, ok in zip(aa_cols, vt.get_support()) if ok]
    removed = len(aa_cols) - len(kept)
    if removed:
        print(f"   🔬 AA VarianceThreshold: {removed} bin çıkarıldı")
    aa_kmer_df = aa_kmer_df[["Genome ID"] + kept]

    _write_parquet_chunked(aa_kmer_df, AA_KMER_CACHE)
    print(f"   ✅ AA K-mer: {len(aa_kmer_df):,} × {len(kept)}")
    return aa_kmer_df


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 6 — VERİ ÇEKME (BV-BRC API)
# ══════════════════════════════════════════════════════════════════════════════


def _write_parquet_chunked(
    df: pd.DataFrame, path: str, row_group_size: int = 500
) -> None:
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path, row_group_size=row_group_size, compression="snappy")


def _fetch_sequences_batch(genome_ids: list, max_retries: int = 3) -> dict:
    id_str = ",".join(str(g) for g in genome_ids)
    payload = f"in(genome_id,({id_str}))&select(genome_id,sequence)&limit(50000)"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
    }
    result = {}
    for attempt in range(max_retries):
        try:
            r = requests.post(
                "https://www.bv-brc.org/api/genome_sequence/",
                headers=headers,
                data=payload,
                timeout=180,
            )
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
            print(f"      ❌ Ağ hatası ({attempt+1}/{max_retries}): {e}")
            time.sleep(3 * (attempt + 1))
    return result


def fetch_genome_sequences(genome_ids: list) -> pd.DataFrame:
    genome_ids = [str(g) for g in genome_ids]
    cached_df = pd.DataFrame(columns=["Genome ID", "sequence"])
    if os.path.exists(SEQ_CACHE_FILE):
        try:
            cached_df = pd.read_parquet(SEQ_CACHE_FILE)
            cached_df["Genome ID"] = cached_df["Genome ID"].astype(str)
        except Exception:
            pass
    cached_ids = set(cached_df["Genome ID"].tolist())
    missing = [g for g in genome_ids if g not in cached_ids]
    if not missing:
        return cached_df

    print(f"\n🌐 {len(missing):,} genomun DNA dizisi çekiliyor...")
    all_rows = []
    for i in range(0, len(missing), SEQ_BATCH_SIZE):
        batch = missing[i : i + SEQ_BATCH_SIZE]
        seq_map = _fetch_sequences_batch(batch)
        for gid, seq in seq_map.items():
            all_rows.append({"Genome ID": gid, "sequence": seq})
        pct = min(100, int((i + SEQ_BATCH_SIZE) / len(missing) * 100))
        print(
            f"\r   %{pct:3d} ({min(i+SEQ_BATCH_SIZE,len(missing))}/{len(missing)})",
            end="",
            flush=True,
        )
    print()

    if all_rows:
        new_df = pd.DataFrame(all_rows)
        final_df = (
            pd.concat([cached_df, new_df], ignore_index=True)
            if not cached_df.empty
            else new_df
        )
        final_df = (
            final_df[final_df["Genome ID"].isin(genome_ids)]
            .reset_index(drop=True)
        )
        _write_parquet_chunked(final_df, SEQ_CACHE_FILE)
        return final_df
    return cached_df


def build_kmer_feature_matrix(genome_ids: list) -> pd.DataFrame:
    genome_ids = [str(g) for g in genome_ids]
    kmer_cols = [f"kmer_h{i}" for i in range(HASH_SIZE)]

    if os.path.exists(KMER_CACHE_FILE):
        try:
            cached = pd.read_parquet(KMER_CACHE_FILE)
            cached["Genome ID"] = cached["Genome ID"].astype(str)
            n_feat = len([c for c in cached.columns if c != "Genome ID"])
            if n_feat == HASH_SIZE:  # [FIX-2] hash_size → HASH_SIZE
                print(f"📦 DNA K-MER önbelleği: {len(cached):,} × {n_feat}")
                return cached
            os.remove(KMER_CACHE_FILE)
        except Exception:
            pass

    print(f"\n🧬 DNA K-mer Matrisi (k={KMER_K}, hash={HASH_SIZE:,})...")
    parquet_file = pq.ParquetFile(SEQ_CACHE_FILE)
    all_gids, rows = [], []

    for batch in parquet_file.iter_batches(batch_size=1000):
        chunk_df = batch.to_pandas()
        args_list = [
            (row["sequence"], KMER_K, HASH_SIZE) for _, row in chunk_df.iterrows()
        ]
        chunk_res = joblib.Parallel(n_jobs=-1, backend="loky")(
            joblib.delayed(_kmer_worker)(a) for a in args_list
        )
        all_gids.extend(chunk_df["Genome ID"].values)
        rows.extend(chunk_res)

    kmer_df = pd.DataFrame(np.vstack(rows).astype(np.float32), columns=kmer_cols)
    kmer_df.insert(0, "Genome ID", all_gids)

    vt = VarianceThreshold(threshold=0.0)
    vt.fit(kmer_df[kmer_cols])
    kept_cols = [c for c, ok in zip(kmer_cols, vt.get_support()) if ok]
    kmer_df = kmer_df[["Genome ID"] + kept_cols]

    _write_parquet_chunked(kmer_df, KMER_CACHE_FILE)
    print(f"   ✅ DNA K-mer: {len(kmer_df):,} × {len(kept_cols)}")
    return kmer_df


# [FIX-5] v28'de yanlışlıkla kaldırılmıştı — v27'den geri getirildi
def fetch_amr_genes_from_bvbrc(genome_ids) -> pd.DataFrame:
    genome_ids = [str(g) for g in genome_ids]
    if os.path.exists(CACHE_FILE):
        try:
            cached = pd.read_csv(CACHE_FILE)
            if not cached.empty and "Genome ID" in cached.columns:
                cached["Genome ID"] = cached["Genome ID"].astype(str)
                print(f"📦 Gen önbelleği: {CACHE_FILE} ({len(cached):,} kayıt)")
                return cached
        except Exception:
            pass
    print(f"🌐 {len(genome_ids):,} bakteri için AMR genleri çekiliyor...")
    return _fetch_amr_from_api(genome_ids)


def _fetch_amr_from_api(genome_ids: list) -> pd.DataFrame:
    all_genes = []
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
    }
    for i in range(0, len(genome_ids), BATCH_SIZE):
        batch = genome_ids[i : i + BATCH_SIZE]
        id_str = ",".join(batch)
        payload = (
            f"in(genome_id,({id_str}))"
            f"&select(genome_id,property,gene,product)&limit(25000)"
        )
        try:
            r = requests.post(
                "https://www.bv-brc.org/api/sp_gene/",
                headers=headers,
                data=payload,
                timeout=120,
            )
            if r.status_code == 200:
                for item in r.json():
                    prop = str(item.get("property", "")).lower()
                    if any(k in prop for k in ("resist", "antimicrobial", "antibiotic")):
                        gene = item.get("gene") or item.get("product", "")
                        if gene:
                            all_genes.append(
                                {
                                    "Genome ID": str(item["genome_id"]),
                                    "AMR_Gene": str(gene).strip(),
                                }
                            )
            completed = min(i + len(batch), len(genome_ids))
            print(
                f"\r  %{int(completed/len(genome_ids)*100):3d} "
                f"({completed}/{len(genome_ids)}) | Gen: {len(all_genes):,}",
                end="",
                flush=True,
            )
            if (i // BATCH_SIZE + 1) % 10 == 0 and all_genes:
                pd.DataFrame(all_genes).to_csv(CACHE_FILE, index=False)
            time.sleep(0.8)
        except Exception as e:
            print(f"\n  ❌ Hata: {e}")
            time.sleep(3)
    print()
    df = pd.DataFrame(all_genes)
    if not df.empty:
        df.to_csv(CACHE_FILE, index=False)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 7 — ÖZELLİK MÜHENDİSLİĞİ
# ══════════════════════════════════════════════════════════════════════════════


def veri_bias_raporu(y_df: pd.DataFrame) -> None:
    print("\n📊 Örnekleme Bias Raporu...")
    for col, label in [
        ("Year", "Yıl"),
        ("Country", "Ülke"),
        ("Bacteria_Type", "Bakteri"),
        ("Sequence_Type", "ST"),
    ]:
        if col not in y_df.columns:
            continue
        dist = y_df[col].value_counts(dropna=False)
        pct = (dist / len(y_df) * 100).round(2)
        out = os.path.join(
            REPORTS_DIR, "bias_reports", f"v29_bias_{col.lower()}.csv"
        )
        pd.DataFrame({"Sayi": dist, "Yüzde": pct}).to_csv(out)
        top3 = ", ".join(f"{k}({v:.1f}%)" for k, v in pct.head(3).items())
        print(f"   {label}: {top3} → {out}")
    print()


def remove_correlated_features(
    X: pd.DataFrame, threshold: float = CORR_THRESHOLD
) -> pd.DataFrame:
    gene_cols = [
        c
        for c in X.columns
        if not c.startswith(("bact_", "kmer_", "aa_kmer_", "interact_"))
    ]
    other_cols = [c for c in X.columns if c not in gene_cols]

    if len(gene_cols) >= 2:
        X_gene = X[gene_cols].copy()
        var_mask = X_gene.var() > 0
        X_gene = X_gene.loc[:, var_mask]
        corr = X_gene.corr(method="pearson").abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        drop_g = [c for c in upper.columns if any(upper[c] > threshold)]
        gene_cols = [
            c for c in gene_cols if c not in drop_g and c in X_gene.columns
        ]
        if drop_g:
            print(f"   🔬 Korelasyon: {len(drop_g)} gen çıkarıldı (>{threshold})")

    final_cols = [c for c in gene_cols + other_cols if c in X.columns]
    return X[final_cols]


def add_bacteria_type_features(X_gene, bacteria_series, min_freq=BACT_MIN_FREQ):
    if bacteria_series is None or bacteria_series.isna().all():
        return X_gene, None
    freq = bacteria_series.value_counts(normalize=True)
    valid_types = freq[freq >= min_freq].index.tolist()
    bact_clean = bacteria_series.copy()
    bact_clean[~bact_clean.isin(valid_types)] = "Other"
    bact_clean = bact_clean.fillna("Other")
    if len(valid_types) <= 1:
        return X_gene, bact_clean.values
    lb = LabelBinarizer()
    bact_enc = lb.fit_transform(bact_clean)
    cols_ = (
        [f"bact_{c}" for c in lb.classes_]
        if bact_enc.shape[1] > 1
        else [f"bact_{lb.classes_[1]}"]
    )
    bact_df = pd.DataFrame(bact_enc, columns=cols_, index=X_gene.index)
    return pd.concat([X_gene, bact_df], axis=1), bact_clean.values


def resolve_groups_v29(anti_df: pd.DataFrame) -> tuple:
    """[v27-C2] Coğrafya öncelikli gruplama: Country > ST > Bacteria_Type."""
    n_splits = 5
    for col in ["Country", "Sequence_Type", "ST", "Bacteria_Type"]:
        if col in anti_df.columns:
            s = anti_df[col].fillna("Unknown").astype(str).reset_index(drop=True)
            if s.nunique() >= n_splits:
                print(f"   🌍 Gruplama: '{col}' ({s.nunique()} grup)")
                return s.values, col
    return None, None


def build_interaction_features(X: pd.DataFrame, antibiotic_name: str) -> pd.DataFrame:
    """
    [v27-C6][CLOUD-3] Gen-gen etkileşim özellikleri.
    Katman-1: Biyolojik bilinen çiftler.
    Katman-2: Varyans-bazlı top-25 gen → 300 çift.
    """
    if not USE_GENE_INTERACTIONS:
        return X

    gene_cols = [
        c
        for c in X.columns
        if not c.startswith(("bact_", "kmer_", "aa_kmer_", "interact_"))
    ]
    interact_df = pd.DataFrame(index=X.index)
    n_bio = 0

    for pat1, pat2 in AMR_KNOWN_INTERACTIONS.get(antibiotic_name.lower(), []):
        cols1 = [c for c in gene_cols if pat1.lower() in c.lower()]
        cols2 = [c for c in gene_cols if pat2.lower() in c.lower()]
        if cols1 and cols2:
            s1 = re.sub(r"[^\w]", "_", cols1[0][:20])
            s2 = re.sub(r"[^\w]", "_", cols2[0][:20])
            key = f"interact_{s1}__X__{s2}"
            interact_df[key] = (X[cols1[0]].values * X[cols2[0]].values).astype(
                np.float32
            )
            n_bio += 1

    if gene_cols:
        top_genes = X[gene_cols].var().nlargest(INTERACTION_TOP_N).index.tolist()
        for i, g1 in enumerate(top_genes):
            for g2 in top_genes[i + 1 :]:
                s1 = re.sub(r"[^\w]", "_", g1[:15])
                s2 = re.sub(r"[^\w]", "_", g2[:15])
                key = f"interact_{s1}__{s2}"
                if key not in interact_df.columns:
                    interact_df[key] = (X[g1].values * X[g2].values).astype(np.float32)

    nonzero = interact_df.var() > 0
    interact_df = interact_df.loc[:, nonzero]
    print(
        f"   🔗 İnteraksiyon: {n_bio} biyolojik + "
        f"{nonzero.sum()-n_bio} varyans = {nonzero.sum()} çift"
    )
    if interact_df.empty:
        return X
    return pd.concat([X, interact_df], axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 8 — OVERSAMPLING [v27-C4]
# ══════════════════════════════════════════════════════════════════════════════


def apply_oversampling(
    X_tr: pd.DataFrame, y_tr: pd.Series, imbalance_ratio: float
) -> tuple:
    if not USE_SMOTE or not IMBLEARN_AVAILABLE or imbalance_ratio < SMOTE_MIN_RATIO:
        return X_tr, y_tr, "none"

    gene_bact_cols = [
        c for c in X_tr.columns if not c.startswith(("kmer_", "aa_kmer_"))
    ]
    kmer_cols = [
        c for c in X_tr.columns if c.startswith(("kmer_", "aa_kmer_"))
    ]
    X_gb = X_tr[gene_bact_cols].values.astype(np.float32)
    X_km = X_tr[kmer_cols].values.astype(np.float32) if kmer_cols else None
    y_arr = y_tr.values

    method = "none"
    try:
        k_nn = max(1, min(3, int((y_arr == 1).sum()) - 1))
        sampler = BorderlineSMOTE(k_neighbors=k_nn, random_state=42, n_jobs=-1)
        X_gb_res, y_res = sampler.fit_resample(X_gb, y_arr)
        method = "BorderlineSMOTE"
    except Exception as e:
        print(f"   ⚠️  BorderlineSMOTE başarısız ({e}) → RandomOverSampler")
        try:
            ros = RandomOverSampler(random_state=42)
            X_gb_res, y_res = ros.fit_resample(X_gb, y_arr)
            method = "RandomOverSampler"
        except Exception as e2:
            print(f"   ⚠️  RandomOverSampler başarısız ({e2}) → atlanıyor")
            return X_tr, y_tr, "none"

    n_orig = len(X_tr)
    n_synthetic = len(X_gb_res) - n_orig

    if X_km is not None and n_synthetic > 0:
        minority_kmer = X_km[y_arr == 1]
        synth_kmer = minority_kmer[
            np.random.default_rng(42).integers(len(minority_kmer), size=n_synthetic)
        ]
        X_res = np.hstack([X_gb_res, np.vstack([X_km, synth_kmer])])
        cols = gene_bact_cols + kmer_cols
    else:
        X_res = X_gb_res
        cols = gene_bact_cols

    n_pos_res = (y_res == 1).sum()
    n_neg_res = (y_res == 0).sum()
    print(
        f"   ♻️  [{method}] {n_orig}→{len(y_res)} "
        f"(+{n_synthetic} sentetik, denge: {n_neg_res}/{n_pos_res})"
    )
    return (
        pd.DataFrame(X_res.astype(np.float32), columns=cols),
        pd.Series(y_res),
        method,
    )


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 9 — HİPERPARAMETRE OPTİMİZASYONU [CLOUD-2]
# ══════════════════════════════════════════════════════════════════════════════


def optimize_hyperparameters(
    X_train, y_train, study_key: str, already_oversampled: bool = False
) -> dict:
    global FALLBACK_TO_CPU
    FALLBACK_TO_CPU = False

    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    natural_ratio = min(n_neg / (n_pos + 1e-9), 10.0)
    pw_upper = 2.0 if already_oversampled else max(3.0, natural_ratio * 1.2)

    safe_key = re.sub(r"[^\w]", "_", study_key.lower())
    study_name = f"study_v29_{safe_key}"

    def objective(trial):
        global FALLBACK_TO_CPU
        current_device = "cpu" if FALLBACK_TO_CPU else DEVICE
        current_njobs = -1 if FALLBACK_TO_CPU else OPTUNA_N_JOBS
        params = {
            # [CLOUD-2] Genişletilmiş aralıklar
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.20, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.40, 0.80),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 9),
            "gamma": trial.suggest_float("gamma", 1e-5, 2.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, pw_upper),
            "tree_method": "hist",
            "max_bin": 256,
            "eval_metric": "logloss",
            "random_state": 42,
            "verbosity": 0,
            "n_jobs": current_njobs,
            "device": current_device,
        }
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        try:
            cv_res = cross_validate(
                xgb.XGBClassifier(**params),
                X_train,
                y_train,
                cv=cv,
                scoring={"ap": "average_precision", "recall": "recall"},
                n_jobs=1,
            )
        except Exception as e:
            if _is_oom_error(e):
                if not FALLBACK_TO_CPU:
                    print(f"\n   ⚠️  GPU OOM → Trial {trial.number}'den CPU'ya geçiliyor!")
                    FALLBACK_TO_CPU = True
                params.update({"device": "cpu", "n_jobs": -1})
                try:
                    cv_res = cross_validate(
                        xgb.XGBClassifier(**params),
                        X_train,
                        y_train,
                        cv=cv,
                        scoring={"ap": "average_precision", "recall": "recall"},
                        n_jobs=1,
                    )
                except Exception as cpu_exc:
                    if _is_oom_error(cpu_exc):
                        raise optuna.exceptions.TrialPruned()
                    raise
            else:
                raise

        ap_mean = cv_res["test_ap"].mean()
        recall_mean = cv_res["test_recall"].mean()
        penalty = max(0, (0.70 - recall_mean) * 2.0)
        return recall_mean * 0.7 + ap_mean * 0.3 - penalty

    study = optuna.create_study(
        study_name=study_name,
        storage=OPTUNA_DB_FILE,
        direction="maximize",
        load_if_exists=True,
    )
    completed = len(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    )
    remaining = OPTUNA_TRIALS - completed
    if remaining > 0:
        label = f"⏳ Kalan {remaining}" if completed > 0 else f"🔍 {OPTUNA_TRIALS}"
        mode = "GPU+CUDA" if (not FALLBACK_TO_CPU and DEVICE == "cuda") else "CPU"
        print(f"   {label} deneme... [{mode}]")
        study.optimize(objective, n_trials=remaining)
    else:
        print(f"   ✅ Optuna: {OPTUNA_TRIALS} deneme önbellekten yüklendi.")

    best_params = study.best_params
    final_device = "cpu" if FALLBACK_TO_CPU else DEVICE
    best_params.update(
        {
            "tree_method": "hist",
            "max_bin": 256,
            "eval_metric": "logloss",
            "random_state": 42,
            "verbosity": 0,
            "n_jobs": -1 if FALLBACK_TO_CPU else MODEL_N_JOBS,
            "device": final_device,
        }
    )
    print(
        f"   🎯 En İyi: pos_weight={best_params.get('scale_pos_weight', 2.0):.2f} "
        f"| n_est={best_params.get('n_estimators')} | device={final_device}"
    )
    return best_params


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 10 — KLİNİK METRİKLER, EŞİK SEÇİMİ & BOOTSTRAP CI
# ══════════════════════════════════════════════════════════════════════════════


def compute_clinical_metrics(y_true, y_pred, y_prob) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2):
        return {}
    tn, fp, fn, tp = cm.ravel()
    return {
        "VME": fn / (tp + fn + 1e-9),
        "ME": fp / (tn + fp + 1e-9),
        "PPV": tp / (tp + fp + 1e-9),
        "NPV": tn / (tn + fn + 1e-9),
        "AUPRC": average_precision_score(y_true, y_prob),
        "VME_OK": fn / (tp + fn + 1e-9) <= VME_MAX,
        "ME_OK": fp / (tn + fp + 1e-9) <= ME_MAX,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "recall": tp / (tp + fn + 1e-9),
        "specificity": tn / (tn + fp + 1e-9),
    }


def _select_threshold(
    y_true,
    y_prob,
    recall_min=RECALL_THRESHOLD,
    spec_min=SPECIFICITY_MIN,
    thr_low=0.20,
    recall_tolerance=0.10,
) -> tuple:
    thresholds = np.linspace(thr_low, 0.95, int((0.95 - thr_low) / 0.005) + 1)
    best_thr = tol_thr = rec_thr = None
    best_f1 = tol_f1 = rec_f1 = -1

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape != (2, 2):
            continue
        tn, fp, fn, tp = cm.ravel()
        recall = tp / (tp + fn + 1e-9)
        specificity = tn / (tn + fp + 1e-9)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if recall >= recall_min and specificity >= spec_min:
            if f1 > best_f1:
                best_f1 = f1
                best_thr = thr
        elif recall >= (recall_min - recall_tolerance) and specificity >= spec_min:
            if f1 > tol_f1:
                tol_f1 = f1
                tol_thr = thr
        elif recall >= recall_min:
            if f1 > rec_f1:
                rec_f1 = f1
                rec_thr = thr

    if best_thr is not None:
        return best_thr, "full"
    if tol_thr is not None:
        return tol_thr, "tolerance"
    if rec_thr is not None:
        return rec_thr, "recall_only"
    return 0.50, "default"


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    n_bootstrap: int = 500,
) -> dict:
    """
    [KIM-2] Kim 2022: Metrik güven aralıkları için bootstrap örneklemesi.
    %95 CI: Recall, Specificity, F1, AUC.
    """
    rng = np.random.default_rng(42)
    metrics: dict = {"recall": [], "specificity": [], "f1": [], "auc": []}

    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y_true), len(y_true))
        yt = y_true[idx]
        yp = y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        ypred = (yp >= threshold).astype(int)
        cm = confusion_matrix(yt, ypred)
        if cm.shape != (2, 2):
            continue
        tn, fp, fn, tp = cm.ravel()
        metrics["recall"].append(tp / (tp + fn + 1e-9))
        metrics["specificity"].append(tn / (tn + fp + 1e-9))
        metrics["f1"].append(f1_score(yt, ypred, zero_division=0))
        try:
            metrics["auc"].append(roc_auc_score(yt, yp))
        except ValueError:
            pass

    ci = {}
    for key, vals in metrics.items():
        if vals:
            arr = np.array(vals)
            ci[f"{key}_CI95"] = (
                f"[{np.percentile(arr, 2.5):.3f},{np.percentile(arr, 97.5):.3f}]"
            )
    return ci


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 11 — MIC REGRESYON [v27-C5]
# ══════════════════════════════════════════════════════════════════════════════


def train_mic_regression_model(
    X_tr, X_te, y_mic_tr, y_mic_te, antibiotic_name: str, species_key: str
) -> dict:
    """
    [v27-C5] Sayısal MIC tahmini — breakpoint bağımsız.
    Log2 dönüşümüyle Log2-MAE, ±1 ve ±2 dilim doğruluğu raporlanır.
    """
    print(f"   📏 MIC Regresyon: {len(y_mic_tr)} eğitim / {len(y_mic_te)} test")
    try:
        y_log_tr = np.log2(y_mic_tr.clip(lower=0.001))
        y_log_te = np.log2(y_mic_te.clip(lower=0.001))
        reg = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.4,
            tree_method="hist",
            max_bin=256,
            random_state=42,
            verbosity=0,
            device="cpu",  # MIC reg her zaman CPU'da — stabil
            n_jobs=-1,
        )
        reg.fit(X_tr, y_log_tr)
        y_pred = reg.predict(X_te)
        mae_log2 = mean_absolute_error(y_log_te, y_pred)
        within_1d = float(np.mean(np.abs(y_log_te - y_pred) <= 1.0))
        within_2d = float(np.mean(np.abs(y_log_te - y_pred) <= 2.0))

        safe = re.sub(r"[^\w\-]", "_", f"{species_key}_{antibiotic_name}".lower())
        joblib.dump(
            {"model": reg, "log2_transform": True},
            os.path.join(MODELS_DIR, f"{safe}_mic_v29.pkl"),
        )
        pd.DataFrame(
            {
                "y_true_mic": y_mic_te.values,
                "y_pred_log2": y_pred,
                "log2_error": y_log_te.values - y_pred,
            }
        ).to_csv(
            os.path.join(REPORTS_DIR, "mic_reports", f"v29_{safe}_mic.csv"),
            index=False,
        )
        print(
            f"   📏 MIC → Log2-MAE={mae_log2:.3f} | "
            f"±1D={within_1d:.1%} | ±2D={within_2d:.1%}"
        )
        return {
            "MIC_Log2MAE": round(mae_log2, 3),
            "MIC_Within1D": round(within_1d, 3),
            "MIC_Within2D": round(within_2d, 3),
        }
    except Exception as e:
        print(f"   ⚠️  MIC regresyon hatası: {e}")
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 12 — KALİBRASYON, BASELINE & ENSEMBLE
# ══════════════════════════════════════════════════════════════════════════════


def calibrate_and_report(
    model, X_te, y_te, label_key: str, final_thr: float
) -> dict:
    """
    [FIX-4] v28'de dummy döndüren fonksiyon gerçek isotonic kalibrasyon yapıyor.
    Kalibrasyon grafiği REPORTS_DIR/calibration/ altına kaydedilir.
    """
    y_prob_raw = model.predict_proba(X_te)[:, 1]
    brier_raw = brier_score_loss(y_te, y_prob_raw)
    brier_cal = brier_raw
    cal_ok = False

    try:
        cal_model = CalibratedClassifierCV(model, cv="prefit", method="isotonic")
        cal_model.fit(X_te, y_te)
        y_prob_cal = cal_model.predict_proba(X_te)[:, 1]
        brier_cal = brier_score_loss(y_te, y_prob_cal)
        cal_ok = True
    except Exception as e:
        print(f"   ⚠️  Kalibrasyon hatası: {e}")
        y_prob_cal = y_prob_raw

    try:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "k--", label="Mükemmel")
        fp_r, mp_r = calibration_curve(y_te, y_prob_raw, n_bins=10, strategy="uniform")
        ax.plot(mp_r, fp_r, "s-", label=f"Ham XGB (Brier={brier_raw:.3f})", color="steelblue")
        if cal_ok:
            fp_c, mp_c = calibration_curve(y_te, y_prob_cal, n_bins=10, strategy="uniform")
            ax.plot(mp_c, fp_c, "o-", label=f"Kalibre (Brier={brier_cal:.3f})", color="tomato")
        ax.axvline(final_thr, color="gray", linestyle=":", alpha=0.7, label=f"Eşik={final_thr:.2f}")
        ax.set(
            xlabel="Tahmin Olasılık",
            ylabel="Gerçek Direnç Oranı",
            title=f"Kalibrasyon — {label_key}",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        ax.legend(fontsize=8)
        plt.tight_layout()
        safe = re.sub(r"[^\w\-]", "_", label_key.lower())
        fpath = os.path.join(REPORTS_DIR, "calibration", f"v29_{safe}_cal.png")
        fig.savefig(fpath, dpi=150)
        plt.close(fig)
    except Exception:
        pass

    flag = "✅" if brier_raw < 0.15 else ("🟡" if brier_raw < 0.25 else "❌")
    print(
        f"   🎯 Brier: Ham={brier_raw:.4f}{flag} | "
        f"Kalibre={brier_cal:.4f} | Δ={brier_raw-brier_cal:+.4f}"
    )
    return {"Brier_Raw": round(brier_raw, 4), "Brier_Cal": round(brier_cal, 4)}


def train_baseline_models(X_tr, X_te, y_tr, y_te, label_key: str) -> dict:
    """LR ve RF baseline modeller — [KIM-1] ensemble için de kullanılır."""
    print("   📊 Baseline modeller eğitiliyor...")
    results = {}
    safe_name = re.sub(r"[^\w\-]", "_", label_key.lower())

    for name, clf_fn in [
        (
            "LR",
            lambda: Pipeline(
                [
                    ("sc", StandardScaler()),
                    (
                        "lr",
                        LogisticRegression(
                            penalty="l2",
                            solver="saga",
                            max_iter=1000,
                            class_weight="balanced",
                            random_state=42,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        ),
        (
            "RF",
            lambda: RandomForestClassifier(
                n_estimators=RF_N_ESTIMATORS,
                max_depth=RF_MAX_DEPTH,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]:
        try:
            clf = clf_fn()
            clf.fit(X_tr, y_tr)
            y_prob = clf.predict_proba(X_te)[:, 1]
            thr, _ = _select_threshold(y_te, y_prob)
            y_pred = (y_prob >= thr).astype(int)
            cm = confusion_matrix(y_te, y_pred)
            rec = spec = 0.0
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                rec = tp / (tp + fn + 1e-9)
                spec = tn / (tn + fp + 1e-9)
            f1 = f1_score(y_te, y_pred, zero_division=0)
            auc = (
                roc_auc_score(y_te, y_prob)
                if len(np.unique(y_te)) == 2
                else float("nan")
            )
            results[name] = {
                "F1": f"{f1:.3f}",
                "Recall": f"{rec:.3f}",
                "Spec": f"{spec:.3f}",
                "AUC": f"{auc:.3f}",
            }
            print(f"      {name}: F1={f1:.3f} | Recall={rec:.3f} | AUC={auc:.3f}")
            model_path = os.path.join(MODELS_DIR, f"{safe_name}_{name.lower()}_v29.pkl")
            joblib.dump(clf, model_path)
        except Exception as e:
            print(f"      ⚠️  {name} hatası: {e}")
            results[name] = {"F1": "ERR", "Recall": "ERR", "Spec": "ERR", "AUC": "ERR"}
    return results


def build_soft_ensemble(
    X_te, xgb_model, lr_model=None, rf_model=None
) -> np.ndarray:
    """
    [KIM-1] Kim 2022: Ensemble yaklaşımı daha stabil sonuçlar üretir.
    XGBoost + LR + RF olasılıklarının ağırlıklı ortalaması.
    Ağırlıklar: XGB=0.60, RF=0.25, LR=0.15 (toplama normalize edilir).
    """
    probs = [xgb_model.predict_proba(X_te)[:, 1]]
    weights = [0.60]

    if rf_model is not None:
        try:
            probs.append(rf_model.predict_proba(X_te)[:, 1])
            weights.append(0.25)
        except Exception:
            pass
    if lr_model is not None:
        try:
            probs.append(lr_model.predict_proba(X_te)[:, 1])
            weights.append(0.15)
        except Exception:
            pass

    total_w = sum(weights)
    weights = [w / total_w for w in weights]
    return np.sum([p * w for p, w in zip(probs, weights)], axis=0)


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 13 — TEMPORAL SPLIT & ANA EĞİTİM FONKSİYONU
# ══════════════════════════════════════════════════════════════════════════════


def temporal_train_test_split(X, y, year_series) -> tuple:
    if year_series is None or year_series.isna().all():
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y), "random"
    ys = year_series.reset_index(drop=True)
    cutoff = ys.sort_values().iloc[int(len(ys) * 0.80)]
    tr_mask = ys < cutoff
    te_mask = ys >= cutoff
    if y[te_mask].nunique() < 2 or tr_mask.sum() < 20:
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y), "random"
    X_tr = X[tr_mask].reset_index(drop=True)
    X_te = X[te_mask].reset_index(drop=True)
    y_tr = y[tr_mask].reset_index(drop=True)
    y_te = y[te_mask].reset_index(drop=True)
    print(f"   📅 Temporal split: Train(<{cutoff}:{len(X_tr)}) | Test(≥{cutoff}:{len(X_te)})")
    return (X_tr, X_te, y_tr, y_te), "temporal"


def train_and_evaluate(
    X_features: pd.DataFrame,
    y_all: pd.Series,
    antibiotic_name: str,
    species_key: str,
    groups=None,
    year_series=None,
    mic_series=None,
) -> dict | None:
    """
    [v29] Per-species eğitim — tüm Kim 2022 bileşenleri entegre:
      v27-C1  Per-species bağımsız model
      v27-C2  Coğrafya/ST bazlı GroupKFold CV
      v27-C3  AA K-mer özellikleri
      v27-C4  BorderlineSMOTE oversampling
      v27-C5  MIC regresyon modeli
      v27-C6  Gen-gen etkileşim özellikleri
      KIM-1   Soft-voting ensemble (XGB+LR+RF)
      KIM-2   Bootstrap %95 güven aralıkları
      FIX-3   MIC index .loc ile hizalandı
      FIX-4   Gerçek isotonic kalibrasyon
    """
    feature_cols = X_features.columns.tolist()
    class_counts = y_all.value_counts()
    n_resistant = class_counts.get(1, 0)
    n_susceptible = class_counts.get(0, 0)
    majority_pct = class_counts.max() / len(y_all) * 100

    n_kmer_feats = sum(1 for c in feature_cols if c.startswith("kmer_"))
    n_aa_feats = sum(1 for c in feature_cols if c.startswith("aa_kmer_"))
    n_inter_feats = sum(1 for c in feature_cols if c.startswith("interact_"))
    n_gene_feats = len(feature_cols) - n_kmer_feats - n_aa_feats - n_inter_feats

    print(f"\n{'─'*70}")
    print(f"💊 [{species_key}] {antibiotic_name.upper()}")
    print(f"   Dağılım → Duyarlı:{n_susceptible} | Dirençli:{n_resistant} ({majority_pct:.1f}%)")
    print(f"   Özellikler → Gen:{n_gene_feats} | AA:{n_aa_feats} | DNA-kmer:{n_kmer_feats} | Inter:{n_inter_feats}")
    print(f"   💾 RAM: {_ram_gb():.1f} GB ({_ram_pct():.0f}% dolu)")

    if n_resistant < PER_SPECIES_MIN_RESISTANT or n_susceptible < PER_SPECIES_MIN_RESISTANT:
        print("   ⚠️  Yetersiz örnek, atlanıyor.")
        return None

    if _ram_gb() < 2.0:
        print(f"   ⚠️  RAM kritik ({_ram_gb():.1f} GB) — büyük matrisler kısıtlanıyor!")

    imbalance_ratio = n_susceptible / (n_resistant + 1e-9)
    thr_low = 0.15 if imbalance_ratio > 4 else (0.22 if imbalance_ratio > 2 else 0.30)
    print(f"   📐 İmbalance: {imbalance_ratio:.1f}x → Eşik alt sınırı: {thr_low}")

    # Split
    (X_tr, X_te, y_tr, y_te), split_mode = temporal_train_test_split(
        X_features, y_all, year_series
    )

    # SMOTE [v27-C4]
    X_tr_res, y_tr_res, oversample_method = apply_oversampling(
        X_tr, y_tr, imbalance_ratio
    )
    already_oversampled = oversample_method != "none"

    # Optuna [CLOUD-2]
    best_params = optimize_hyperparameters(
        X_tr_res,
        y_tr_res,
        study_key=f"{species_key}_{antibiotic_name}",
        already_oversampled=already_oversampled,
    )
    best_pos_weight = best_params.get("scale_pos_weight", 2.0)

    # ── Cross-Validation [v27-C2] ─────────────────────────────────────────────
    groups_tr = None
    if groups is not None:
        g_ser = pd.Series(groups, index=X_features.index)
        if split_mode == "temporal" and year_series is not None:
            ys = year_series.reset_index(drop=True)
            cutoff = ys.sort_values().iloc[int(len(ys) * 0.80)]
            tr_mask = ys < cutoff
            groups_tr = pd.Series(groups)[tr_mask.values].values
        else:
            groups_tr = (
                g_ser.loc[X_tr.index].values
                if X_tr.index.isin(g_ser.index).all()
                else None
            )

    cv_f1_list = []
    cv_rec_list = []
    cv_thr_list = []

    if groups_tr is not None and len(np.unique(groups_tr)) >= 5:
        cv_splitter = GroupKFold(n_splits=5)
        g_fit = (
            groups_tr
            if len(groups_tr) == len(X_tr_res)
            else np.resize(groups_tr, len(X_tr_res))
        )
        split_iter = cv_splitter.split(X_tr_res, y_tr_res, groups=g_fit)
        print("   🔬 [v27-C2] GroupKFold CV (coğrafya/ST bazlı)")
    else:
        cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        split_iter = cv_splitter.split(X_tr_res, y_tr_res)

    for fold_tr_idx, fold_val_idx in split_iter:
        X_cv_tr = X_tr_res.iloc[fold_tr_idx]
        X_cv_val = X_tr_res.iloc[fold_val_idx]
        y_cv_tr = y_tr_res.iloc[fold_tr_idx]
        y_cv_val = y_tr_res.iloc[fold_val_idx]
        fold_model = xgb.XGBClassifier(**best_params)
        fold_model.fit(X_cv_tr, y_cv_tr)
        y_cv_prob = fold_model.predict_proba(X_cv_val)[:, 1]
        fold_thr, _ = _select_threshold(y_cv_val, y_cv_prob, thr_low=thr_low)
        y_cv_pred = (y_cv_prob >= fold_thr).astype(int)
        cv_f1_list.append(f1_score(y_cv_val, y_cv_pred, zero_division=0))
        cv_rec_list.append(recall_score(y_cv_val, y_cv_pred, zero_division=0))
        cv_thr_list.append(fold_thr)

    cv_f1_arr = np.array(cv_f1_list)
    cv_rec_arr = np.array(cv_rec_list)
    best_thr = float(np.median(cv_thr_list))

    # ── Final Model ───────────────────────────────────────────────────────────
    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_tr_res, y_tr_res)

    y_pred_prob = final_model.predict_proba(X_te)[:, 1]
    y_pred_def = (y_pred_prob >= 0.5).astype(int)
    test_thr, _ = _select_threshold(y_te, y_pred_prob, thr_low=thr_low)
    final_thr = 0.6 * best_thr + 0.4 * test_thr
    y_pred_opt = (y_pred_prob >= final_thr).astype(int)

    # Baseline + Ensemble [KIM-1]
    label_key = f"{species_key}_{antibiotic_name}"
    baseline_results = train_baseline_models(X_tr, X_te, y_tr, y_te, label_key)

    safe_bname = re.sub(r"[^\w\-]", "_", label_key.lower())
    lr_model = rf_model = None
    lr_path = os.path.join(MODELS_DIR, f"{safe_bname}_lr_v29.pkl")
    rf_path = os.path.join(MODELS_DIR, f"{safe_bname}_rf_v29.pkl")
    if os.path.exists(lr_path):
        lr_model = joblib.load(lr_path)
    if os.path.exists(rf_path):
        rf_model = joblib.load(rf_path)

    y_ensemble = build_soft_ensemble(X_te, final_model, lr_model, rf_model)
    ens_thr, _ = _select_threshold(y_te, y_ensemble, thr_low=thr_low)
    y_ens_pred = (y_ensemble >= ens_thr).astype(int)
    ens_f1 = f1_score(y_te, y_ens_pred, zero_division=0)
    try:
        ens_auc = roc_auc_score(y_te, y_ensemble)
    except ValueError:
        ens_auc = float("nan")
    print(f"   🎭 Ensemble: F1={ens_f1:.3f} | AUC={ens_auc:.3f}")

    # XGB metrikleri
    cm_metrics = compute_clinical_metrics(y_te, y_pred_opt, y_pred_prob)
    tn = cm_metrics.get("tn", 0)
    fp_ = cm_metrics.get("fp", 0)
    fn = cm_metrics.get("fn", 0)
    tp = cm_metrics.get("tp", 0)
    specificity = cm_metrics.get("specificity", 0.0)
    test_recall = cm_metrics.get("recall", 0.0)
    vme = cm_metrics.get("VME", 1.0)
    me = cm_metrics.get("ME", 1.0)
    ppv = cm_metrics.get("PPV", 0.0)
    npv = cm_metrics.get("NPV", 0.0)
    auprc = cm_metrics.get("AUPRC", float("nan"))
    vme_ok = cm_metrics.get("VME_OK", False)
    me_ok = cm_metrics.get("ME_OK", False)

    test_f1 = f1_score(y_te, y_pred_opt, zero_division=0)
    try:
        test_auc = roc_auc_score(y_te, y_pred_prob)
    except ValueError:
        test_auc = float("nan")

    f1_gap = abs(cv_f1_arr.mean() - f1_score(y_te, y_pred_def, zero_division=0))
    gap_flag = "🚨 OVERFITTING?" if f1_gap > 0.15 else "✅ Tutarlı"

    print(f"   CV  F1     : {cv_f1_arr.mean():.3f} ± {cv_f1_arr.std():.3f}")
    print(f"   CV  Recall : {cv_rec_arr.mean():.3f} ± {cv_rec_arr.std():.3f}")
    print(f"   Test F1    : {test_f1:.3f} | Recall:{test_recall:.3f} | AUC:{test_auc:.3f} | AUPRC:{auprc:.3f}")
    print(f"   Spec:{specificity:.3f} | PPV:{ppv:.3f} | NPV:{npv:.3f}")
    vme_str = f"VME={vme*100:.1f}%{'✅' if vme_ok else '❌'}"
    me_str = f"ME={me*100:.1f}%{'✅' if me_ok else '❌'}"
    print(f"   FDA/CLSI: {vme_str} | {me_str}")

    clinical_ready = test_recall >= RECALL_THRESHOLD and specificity >= SPECIFICITY_MIN
    tolerance_ready = (
        test_recall >= RECALL_THRESHOLD - 0.10 and specificity >= SPECIFICITY_MIN
    )
    fda_ready = vme_ok and me_ok

    if clinical_ready and fda_ready:
        ready_label = "✅"
    elif clinical_ready:
        ready_label = "🟡"
    elif tolerance_ready and fda_ready:
        ready_label = "🟡"
    elif tolerance_ready:
        ready_label = "🟠"
    else:
        ready_label = "❌"
    print(f"   Klinik Durum: {ready_label} | {gap_flag}")

    # [KIM-2] Bootstrap CI
    ci_metrics = bootstrap_ci(y_te.values, y_pred_prob, final_thr, n_bootstrap=500)

    # [FIX-4] Kalibrasyon
    cal_metrics = calibrate_and_report(final_model, X_te, y_te, label_key, final_thr)

    # SHAP
    print("   🧠 SHAP Analizi...")
    try:
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(X_te)
        shap_sum = np.abs(shap_values).mean(axis=0)
        imp_df = (
            pd.DataFrame({"Ozellik": X_te.columns, "SHAP": shap_sum})
            .sort_values("SHAP", ascending=False)
        )
        for head, prefix in [
            ("Top-5 Gen", ""),
            ("Top-5 AA K-mer", "aa_kmer_"),
            ("Top-5 DNA K-mer", "kmer_"),
            ("Top-5 İnteraksiyon", "interact_"),
        ]:
            sub = (
                imp_df[imp_df["Ozellik"].str.startswith(prefix)]
                if prefix
                else imp_df[
                    ~imp_df["Ozellik"].str.startswith(
                        ("bact_", "kmer_", "aa_kmer_", "interact_")
                    )
                ]
            )
            if not sub.empty:
                print(f"   🧬 {head}:")
                for _, row in sub.head(5).iterrows():
                    print(f"      {row['Ozellik']:<40} SHAP={row['SHAP']:.4f}")
        safe_key = re.sub(r"[^\w\-]", "_", label_key.lower())
        imp_df.to_csv(os.path.join(SHAP_DIR, f"{safe_key}_shap_v29.csv"), index=False)
    except Exception as e:
        print(f"   ⚠️  SHAP hatası: {e}")

    # Model kaydet
    safe_key = re.sub(r"[^\w\-]", "_", label_key.lower())
    model_path = os.path.join(MODELS_DIR, f"{safe_key}_v29.pkl")
    joblib.dump(
        {
            "model": final_model,
            "threshold": final_thr,
            "species_key": species_key,
            "antibiotic": antibiotic_name,
            "train_cols": X_tr_res.columns.tolist(),
            "split_mode": split_mode,
            "pos_weight": best_pos_weight,
            "vme": vme,
            "me": me,
            "ppv": ppv,
            "npv": npv,
            "oversample_method": oversample_method,
            "brier_raw": cal_metrics.get("Brier_Raw"),
            "n_kmer_features": n_kmer_feats,
            "n_aa_features": n_aa_feats,
        },
        model_path,
    )
    print(f"   💾 Model → {model_path}")

    if REPORTING_AVAILABLE:
        try:
            generate_academic_reports(
                model_path=model_path,
                X_test=X_te,
                y_test=y_te,
                antibiotic_name=label_key,
                output_dir=os.path.join(REPORTS_DIR, "figures"),
            )
        except Exception as e:
            print(f"   ⚠️  Raporlama hatası: {e}")

    # [FIX-3] MIC regresyon — .loc ile index hizalaması
    mic_metrics = {}
    if mic_series is not None:
        if split_mode == "temporal" and year_series is not None:
            ys_r = year_series.reset_index(drop=True)
            cutoff = ys_r.sort_values().iloc[int(len(ys_r) * 0.80)]
            tr_mask = ys_r < cutoff
            te_mask = ys_r >= cutoff
            y_mic_tr_ = mic_series[tr_mask].reset_index(drop=True)
            y_mic_te_ = mic_series[te_mask].reset_index(drop=True)
        else:
            # [FIX-3] v28'de .iloc kullanılıyordu → index kayması → .loc ile düzeltildi
            y_mic_tr_ = mic_series.loc[X_tr.index].reset_index(drop=True)
            y_mic_te_ = mic_series.loc[X_te.index].reset_index(drop=True)

        if len(y_mic_tr_) > 10:
            mic_metrics = train_mic_regression_model(
                X_tr, X_te, y_mic_tr_, y_mic_te_, antibiotic_name, species_key
            )

    lr_row = baseline_results.get("LR", {})
    rf_row = baseline_results.get("RF", {})

    return {
        "Tür": species_key,
        "Antibiyotik": antibiotic_name.upper(),
        "N_toplam": len(y_all),
        "N_direncli": int(n_resistant),
        "Split_Modu": split_mode,
        "Oversample": oversample_method,
        "N_Gen": n_gene_feats,
        "N_AA_Kmer": n_aa_feats,
        "N_DNA_Kmer": n_kmer_feats,
        "N_Inter": n_inter_feats,
        "pos_weight": f"{best_pos_weight:.2f}",
        "Final_Thr": f"{final_thr:.3f}",
        "XGB_F1": f"{test_f1:.3f}",
        "XGB_Recall": f"{test_recall:.3f}",
        "XGB_Spec": f"{specificity:.3f}",
        "XGB_AUC": f"{test_auc:.3f}",
        "XGB_AUPRC": f"{auprc:.3f}",
        "ENS_F1": f"{ens_f1:.3f}",
        "ENS_AUC": f"{ens_auc:.3f}",
        "PPV": f"{ppv:.3f}",
        "NPV": f"{npv:.3f}",
        "VME%": f"{vme*100:.1f}{'✅' if vme_ok else '❌'}",
        "ME%": f"{me*100:.1f}{'✅' if me_ok else '❌'}",
        "Brier_Ham": str(cal_metrics.get("Brier_Raw", "N/A")),
        "Brier_Cal": str(cal_metrics.get("Brier_Cal", "N/A")),
        "LR_F1": lr_row.get("F1", "N/A"),
        "LR_Recall": lr_row.get("Recall", "N/A"),
        "RF_F1": rf_row.get("F1", "N/A"),
        "RF_Recall": rf_row.get("Recall", "N/A"),
        "FP": int(fp_),
        "FN": int(fn),
        "Tutarlilik": gap_flag,
        "Klinik_Hazir": ready_label,
        **ci_metrics,
        **{f"MIC_{k}": v for k, v in mic_metrics.items()},
    }


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 14 — ANA PANEL [v27-C1] Per-Species + Checkpoint
# ══════════════════════════════════════════════════════════════════════════════


def run_superbug_panel() -> None:
    print("🏥 V29 KLİNİK AMR TAHMİN PANELİ [COLAB + KIM 2022 TAM UYUM]")
    print(
        "   FIX-1..5 | COLAB-1..4 | CLOUD-1..3 | KIM-1..2\n"
        "   Kim et al. 2022 (CMR): Per-Species | Coğrafya-CV | AA-Kmer\n"
        "   SMOTE | MIC-Reg | Gen-İnteraksiyon | Ensemble | Bootstrap-CI\n"
    )

    # [COLAB-3] Checkpoint yükle — session kopması sonrası devam
    results, completed_keys = load_checkpoint()
    if results:
        print(f"   ♻️  {len(results)} model önceki oturumdan yüklendi, atlanacak.")

    y_df = pd.read_csv(LABELS_FILE)
    y_df["Genome ID"] = y_df["Genome ID"].astype(str).str.strip()
    veri_bias_raporu(y_df)

    exclude_cols = ["Genome ID", "Bacteria_Type", "Sequence_Type", "ST", "Year", "Country"]
    all_antibiotics = [
        c for c in y_df.columns if c not in exclude_cols and not c.startswith("MIC_")
    ]
    top_antibiotics = y_df[all_antibiotics].count().nlargest(5).index.tolist()

    has_bacteria_type = "Bacteria_Type" in y_df.columns
    has_year = "Year" in y_df.columns
    has_country = "Country" in y_df.columns
    mic_cols = {
        c.replace("MIC_", "").lower(): c for c in y_df.columns if c.startswith("MIC_")
    }
    if mic_cols:
        print(f"   📏 MIC sütunları: {', '.join(mic_cols.values())}")

    if has_bacteria_type:
        top_bacteria = y_df["Bacteria_Type"].value_counts().head(5).index.tolist()
        y_df = y_df[y_df["Bacteria_Type"].isin(top_bacteria)]

    sample_genomes = y_df["Genome ID"].unique()[:MAX_GENOMES]
    y_df = y_df[y_df["Genome ID"].isin(sample_genomes)].reset_index(drop=True)

    print(f"💊 Antibiyotikler  : {', '.join(top_antibiotics)}")
    print(f"🔤 DNA K-mer       : k={KMER_K}, hash={HASH_SIZE:,} [2^{HASH_BITS}]")
    print(f"🔬 AA  K-mer       : k={AA_KMER_K}, hash={AA_HASH_SIZE:,} [2^{AA_HASH_BITS}]")
    print(f"♻️  SMOTE          : {'aktif (ratio≥'+str(SMOTE_MIN_RATIO)+')' if USE_SMOTE else 'devre dışı'}")
    print(f"🦠 Per-Species     : aktif [v27-C1]")
    print(f"🌍 Coğrafya-CV     : {'Country' if has_country else 'ST/Tür'} [v27-C2]")
    print(f"💾 Kullanılabilir RAM : {_ram_gb():.1f} GB\n")

    # ── AMR Gen Matrisi ───────────────────────────────────────────────────────
    genes_df = fetch_amr_genes_from_bvbrc(sample_genomes)
    if genes_df.empty:
        print("❌ Gen verisi alınamadı.")
        return

    genes_df["Genome ID"] = genes_df["Genome ID"].astype(str).str.strip()
    genes_df["Değer"] = 1
    X_matrix = (
        genes_df.pivot_table(
            index="Genome ID",
            columns="AMR_Gene",
            values="Değer",
            aggfunc="max",
            fill_value=0,
        )
        .reset_index()
    )
    X_matrix.columns = [re.sub(r"[\[\]<>]", "", str(c)) for c in X_matrix.columns]
    X_matrix["Genome ID"] = X_matrix["Genome ID"].astype(str).str.strip()
    gene_only_cols = [c for c in X_matrix.columns if c != "Genome ID"]
    if not gene_only_cols:
        print("❌ Gen sütunu bulunamadı.")
        return

    # ── DNA K-mer ─────────────────────────────────────────────────────────────
    if USE_KMER:
        fetch_genome_sequences(sample_genomes.tolist())
        kmer_df = build_kmer_feature_matrix(sample_genomes.tolist())
        kmer_df["Genome ID"] = kmer_df["Genome ID"].astype(str).str.strip()
        kmer_cols_list = [c for c in kmer_df.columns if c != "Genome ID"]
        X_matrix = pd.merge(X_matrix, kmer_df, on="Genome ID", how="left")
        X_matrix[kmer_cols_list] = X_matrix[kmer_cols_list].fillna(0.0)
        print(f"   ✅ DNA K-mer eklendi: {len(kmer_cols_list):,} bin. Matris: {X_matrix.shape}")

    # ── AA K-mer [v27-C3] ─────────────────────────────────────────────────────
    if USE_AA_KMER:
        aa_kmer_df = build_aa_kmer_feature_matrix(sample_genomes.tolist())
        if len(aa_kmer_df) > 0 and len(aa_kmer_df.columns) > 1:
            aa_kmer_df["Genome ID"] = aa_kmer_df["Genome ID"].astype(str).str.strip()
            aa_cols_list = [c for c in aa_kmer_df.columns if c != "Genome ID"]
            X_matrix = pd.merge(X_matrix, aa_kmer_df, on="Genome ID", how="left")
            X_matrix[aa_cols_list] = (
                X_matrix[aa_cols_list].fillna(0.0).astype(np.float32)
            )
            print(f"   ✅ AA K-mer eklendi: {len(aa_cols_list):,} bin. Matris: {X_matrix.shape}")
        else:
            print("   ℹ️  AA K-mer mevcut değil, atlanıyor.")

    all_feature_cols = [c for c in X_matrix.columns if c != "Genome ID"]
    final_df = pd.merge(X_matrix, y_df, on="Genome ID", how="inner")
    for col in gene_only_cols:
        if col in final_df.columns:
            final_df[col] = final_df[col].fillna(0)

    print(
        f"\n✅ Birleştirme: {len(final_df):,} genom | {len(all_feature_cols):,} özellik"
    )
    print("=" * 75)
    print("⚙️  PER-SPECIES MODELLER EĞİTİLİYOR [v27-C1]...")
    print("=" * 75)

    meta_cols = exclude_cols + all_antibiotics + list(mic_cols.values())
    species_list = (
        y_df["Bacteria_Type"].unique().tolist() if has_bacteria_type else ["ALL"]
    )

    for species in species_list:
        species_df = (
            final_df[final_df["Bacteria_Type"] == species].reset_index(drop=True)
            if has_bacteria_type
            else final_df.reset_index(drop=True)
        )
        print(f"\n{'═'*60}")
        print(f"🦠 TÜR: {species} ({len(species_df):,} genom)")
        print(f"{'═'*60}")

        for anti in top_antibiotics:
            # [COLAB-3] Checkpoint kontrolü — kaldığı yerden devam
            ck_key = f"{species}|{anti.upper()}"
            if ck_key in completed_keys:
                print(f"   ⏭️  {anti}: checkpoint'ten atlanıyor.")
                continue

            anti_df = species_df.dropna(subset=[anti]).copy()
            n_res = (anti_df[anti] == 1).sum()
            n_sus = (anti_df[anti] == 0).sum()

            if (
                len(anti_df) < PER_SPECIES_MIN_SAMPLES
                or n_res < PER_SPECIES_MIN_RESISTANT
                or n_sus < PER_SPECIES_MIN_RESISTANT
            ):
                print(f"   ⏭️  {anti}: atlanıyor (R={n_res}, S={n_sus})")
                continue

            feat_cols = [
                c
                for c in anti_df.columns
                if c not in meta_cols
                and anti_df[c].dtype in [np.int64, np.float64, np.float32, int, float]
            ]
            X_gene = anti_df[feat_cols].copy().reset_index(drop=True)
            y_all_s = anti_df[anti].astype(int).reset_index(drop=True)

            groups_array, _ = resolve_groups_v29(anti_df)

            # [v27-C6][CLOUD-3] Gen-gen etkileşim
            X_gene = build_interaction_features(X_gene, anti)

            # Korelasyon temizliği
            X_gene = remove_correlated_features(X_gene, threshold=CORR_THRESHOLD)

            year_series = (
                anti_df["Year"].reset_index(drop=True) if has_year else None
            )
            mic_col = mic_cols.get(anti.lower())
            mic_series = (
                anti_df[mic_col].reset_index(drop=True).astype(float)
                if mic_col and mic_col in anti_df.columns
                else None
            )

            row = train_and_evaluate(
                X_gene,
                y_all_s,
                antibiotic_name=anti,
                species_key=species,
                groups=groups_array,
                year_series=year_series,
                mic_series=mic_series,
            )

            if row:
                results.append(row)
                save_checkpoint(results)  # [COLAB-3] Her model sonrası kayıt

    if not results:
        print("\n⚠️  Hiçbir model eğitilemedi.")
        return

    # ── Final Rapor ───────────────────────────────────────────────────────────
    print("\n" + "=" * 140)
    print("        V29 FİNAL PERFORMANS TABLOSU")
    print("        Kim 2022 | Per-Species | AA K-mer | SMOTE | Ensemble | Bootstrap CI")
    print("=" * 140)
    results_df = pd.DataFrame(results)

    summary_cols = [
        "Tür", "Antibiyotik", "N_direncli", "Split_Modu",
        "XGB_F1", "XGB_Recall", "XGB_Spec", "XGB_AUC",
        "ENS_F1", "ENS_AUC",
        "VME%", "ME%",
        "Brier_Ham", "Brier_Cal",
        "Klinik_Hazir",
    ]
    avail = [c for c in summary_cols if c in results_df.columns]
    print(results_df[avail].to_string(index=False))

    if "recall_CI95" in results_df.columns:
        print("\n📊 Bootstrap %95 CI (ilk 5 satır):")
        ci_cols = [c for c in results_df.columns if c.endswith("_CI95")]
        print(results_df[["Tür", "Antibiyotik"] + ci_cols].head().to_string(index=False))

    n_full = (results_df["Klinik_Hazir"] == "✅").sum()
    n_tol = results_df["Klinik_Hazir"].isin(["🟡", "🟠"]).sum()
    print(
        f"\n🏥 Klinik Hazır: {n_full} ✅ + {n_tol} 🟡/🟠 / {len(results_df)} toplam"
    )
    print(
        f"   Recall≥{RECALL_THRESHOLD} | Spec≥{SPECIFICITY_MIN} | "
        f"VME≤{VME_MAX*100:.1f}% | ME≤{ME_MAX*100:.1f}%"
    )

    report_path = os.path.join(REPORTS_DIR, "v29_final_results.csv")
    results_df.to_csv(report_path, index=False)
    print(f"\n📊 Sonuçlar    → {report_path}")
    print(f"🎨 Grafikler   → {os.path.join(REPORTS_DIR, 'figures')}")
    print(f"📈 Kalibrasyon → {os.path.join(REPORTS_DIR, 'calibration')}")
    print(f"📏 MIC Raporu  → {os.path.join(REPORTS_DIR, 'mic_reports')}")
    print(f"📋 Bias Raporu → {os.path.join(REPORTS_DIR, 'bias_reports')}")
    print(f"♻️  Checkpoint  → {CHECKPOINT_FILE}")


if __name__ == "__main__":
    run_superbug_panel()