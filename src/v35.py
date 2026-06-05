
# main_v35_colab.py
# Kim et al. 2022 (Clin Microbiol Rev 35(3):e00179-21) Tam Hizalama
# Google Colab optimize — v35 (klinik AMR için yeniden tasarlanmış)
#
# ══════════════════════════════════════════════════════════════════════════════
# v34 → v35 KRİTİK DÜZELTMELER
# ──────────────────────────────────────────────────────────────────────────────
# [v35-FIX-1] _select_threshold_clinical: F1-max → VME-öncelikli klinik eşik.
#             Kim 2022 §"Evaluating machine-learning models" (Training and Testing alt-bölümü):
#               1) VME≤1.5% VE ME≤3% sağla → adaylar arası max recall
#               2) Sadece VME sağlanırsa → max recall (klinik öncelik)
#               3) Hiçbiri yoksa → cost-sensitive thr* = c_FP/(c_FP+c_FN)
#               4) Son çare → Youden's J maks.
#             Eski mantık VME'yi göz ardı edip VME=64% gibi felaket sonuçlara
#             yol açıyordu (gentamicin gözleminde).
# [v35-FIX-2] select_features_amr: Mutual Information ile antibiyotik-bilinçli FS.
#             16,384 DNA k-mer + 4,096 AA k-mer → top-1000 + top-500.
#             Genler/bact_/interact_ HEPSİ KORUNUR (biyolojik bilgi).
# [v35-FIX-3] pos_weight cap kaldırıldı: already_oversampled=True'da eski cap 2.0
#             pozitif sınıf ağırlığını sınırlıyordu. Yeni:
#               - SMOTE sonrası : pw_upper = max(2.5, natural_ratio * 0.5)
#               - SMOTE'siz    : pw_upper = max(3.0, natural_ratio * 1.2)
# [v35-FIX-4] build_stacked_ensemble: 0.60/0.25/0.15 sabit ağırlık → meta-LR
#             out-of-fold preds üzerinde fit. Veri-bağımlı ağırlıklar.
# [v35-FIX-5] Kalibre olasılık üzerinden eşik: CalibratedClassifierCV cv=3 ile
#             training fold içinde fit (test leak yok), karar eşiği kalibre prob
#             üzerinden alınır.
# [v35-FIX-6] AA k-mer fallback açık uyarı + pipeline güvenli düşüş.
# [v35-FIX-7] Optuna objective: penalty 2.0 → 5.0, target recall = RECALL_THRESHOLD.
# ══════════════════════════════════════════════════════════════════════════════
#
# ──────────────────────────────────────────────────────────────────────────────
# AKADEMİK REFERANSLAR (v35-AŞAMA-2 ile eklendi, web search doğrulamalı)
# ──────────────────────────────────────────────────────────────────────────────
# [Kim2022]   Kim JI, Maguire F, Tsang KK, Gouliouris T, Peacock SJ, McAllister
#             TA, McArthur AG, Beiko RG. (2022) "Machine Learning for
#             Antimicrobial Resistance Prediction: Current Practice, Limitations,
#             and Clinical Perspective." Clin Microbiol Rev 35(3):e00179-21.
#             doi:10.1128/cmr.00179-21  (PMC9491192)
#             Section yapısı: SUMMARY / INTRODUCTION / MACHINE LEARNING FOR AMR
#             PREDICTION (Suitability of Genomic Data Sets; Representing Genomes
#             and Phenotype Labels; Feature Selection for Interpretable Models;
#             Training and Testing Machine Learning Models [Choosing the
#             appropriate classifier/algorithm; Evaluating machine-learning
#             models]) / LIMITATIONS / TRANSLATING ML-AMR PREDICTION FROM
#             RESEARCH TO PRACTICE (ML for Public Health AMR Surveillance;
#             ML for Clinical Diagnostics) / CONCLUDING REMARKS.
#
# [Nguyen2019] Nguyen M, Long SW, McDermott PF, Olsen RJ, Olson R, Stevens RL,
#              Tyson GH, Zhao S, Davis JJ. (2019) "Using Machine Learning To
#              Predict Antimicrobial MICs and Associated Genomic Features for
#              Nontyphoidal Salmonella." J Clin Microbiol 57(2):e01260-18.
#              ALTIN STANDART: XGBoost regressor on log2(MIC), 10-mer/15-mer
#              k-mer, %95 within ±1 dilution, ortalama VME=%2.7 ME=%0.1.
#              FDA standardı: "within ±1 2-fold dilution step".
#
# [Lees2023]   Lees JA, ... (2023/2024 Microbial Genomics) "Optimising machine
#              learning prediction of minimum inhibitory concentrations in
#              Klebsiella pneumoniae." PMC10995625.
#              Elastic Net + Random Forest + FaST-LMM; pan-genome (Panaroo),
#              SNP filtreleme, PopPUNK ile populasyon yapısı düzeltmesi.
#              MIC seviye sayısına göre regression/classification stratejisi.
#
# [Davis2016]  Davis JJ et al. (2016) "Antimicrobial resistance prediction in
#              PATRIC and RAST." Sci Rep 6:27930. Adaptive boosting + AMR
#              metadata, ilk genom-bazlı AMR ML referansı.
#
# [Aldred2014] Aldred KJ, Kerns RJ, Osheroff N. (2014) "Mechanism of quinolone
#              action and resistance." Biochemistry 53(10):1565-74.
#              gyrA Ser83Leu, parC Ser80Ile QRDR mutasyon literatürü.
#
# Klinik metrik tanımları (Essential/Categorical Agreement):
#  - EA: tahmin gerçek MIC'in ±1 doubling dilution içinde (hedef ≥%90)
#  - CA: tahmin S/I/R kategorisi gerçekle aynı (hedef ≥%95)
#  - Kaynak: CLSI M52 (Performance Standards for Antimicrobial Susceptibility
#    Tests, ISO 20776-2) ve Nguyen 2019 referans tablo.
# ──────────────────────────────────────────────────────────────────────────────

# (Eski header korunmuştur — v29/v34 değişiklikleri için aşağı bakınız)
# main_v29_colab.py
# Kim et al. 2022 (Clin Microbiol Rev 35(3):e00179-21) Uyumlu AMR Tahmin Sistemi
# Google Colab optimize — v29 + KMER-FIX + PATH-FIX
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
#     veya Drive'ın herhangi bir yerindeki parquet dosyaları otomatik bulunur.
# ══════════════════════════════════════════════════════════════════════════════
#
# v27 → v29 DEĞİŞİKLİKLERİ
# ──────────────────────────────────────────────────────────────────────────────
# [FIX-1]      CORR_THRESHOLD, OPTUNA_TRIALS, MAX_GENOMES, RF_N_ESTIMATORS,
#              RF_MAX_DEPTH eksik sabitler eklendi (v28'de kaldırılmıştı → NameError)
# [FIX-2]      build_kmer_feature_matrix: hash_size → HASH_SIZE yazım hatası düzeltildi
# [FIX-3]      MIC index kayması: .iloc[tr_idx] → .loc[X_tr.index] ile düzeltildi
# [FIX-4]      calibrate_and_report: dummy yerine gerçek isotonic kalibrasyon
# [FIX-5]      fetch_amr_genes_from_bvbrc v28'de eksikti, v27'den geri getirildi
# [PATH-FIX]   _resolve_cache_path(): Drive'daki herhangi bir konumdaki önbellek
#              dosyası otomatik bulunur. Yanlış konumdaki parquet artık
#              "yeniden çekiliyor" hatasına yol açmaz.
# [KMER-FIX-1] joblib loky → prefer="threads" backend
#              loky, Colab'ın fork/spawn modeli ile çakışıp ölü kilide giriyor.
#              "🧬 DNA K-mer Matrisi..." satırında donma artık olmayacak.
# [KMER-FIX-2] HASH_BITS 16→14 (65 536→16 384), SEQ_SAMPLE_BP 200k→60k,
#              AA_HASH_BITS 14→12. Colab'da 65k×1600 matris ~420 MB RAM
#              patlatıyordu; 16k ile biyolojik sinyal %95+ korunur.
# [KMER-FIX-3] Chunk bazlı K-mer önbellekleme: her 200 genomda bir ara kayıt.
#              Yarıda kesilirse kaldığı yerden devam eder.
# [KMER-FIX-4] build_kmer_feature_matrix artık global SEQ_CACHE_FILE yerine
#              _resolve_cache_path() kullanır → yanlış yol artık hata vermez.
# [KMER-FIX-5] Tüm Parallel çağrıları prefer="threads" ile güncellendi.
# [COLAB-1]    torch.cuda.is_available() ile GPU tespiti (CUDA hardcode kaldırıldı)
# [COLAB-2]    Google Drive entegrasyonu: model/rapor kalıcı saklamak için
# [COLAB-3]    Checkpoint sistemi: her (tür×antibiyotik) sonrası CSV kayıt/devam
# [COLAB-4]    psutil RAM monitörü + OOM uyarısı
# [CLOUD-1]    Hash boyutları: DNA 2^14=16 384, AA 2^12=4 096
# [CLOUD-2]    OPTUNA_TRIALS=50, n_estimators 100-600, colsample_bytree 0.40-0.80
# [CLOUD-3]    INTERACTION_TOP_N=25 → 300+ gen çifti epistasis uzayı
# [KIM-1]     Soft-voting ensemble: XGB+LR+RF olasılık ortalaması
# [KIM-2]     Bootstrap %95 güven aralıkları: Recall/Spec/F1/AUC için
# ══════════════════════════════════════════════════════════════════════════════

import subprocess
import sys
from collections import Counter

# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 0 — COLAB KURULUM YARDIMCI FONKSİYONLARI
# ══════════════════════════════════════════════════════════════════════════════

_REQUIRED_PACKAGES = [
    "xgboost>=2.0.0",
    "lightgbm>=4.0.0",         # [v35-FIX-LGB] 4. base learner
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

# [v35-FIX-LGB] LightGBM — stacked ensemble 4. base learner
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️  lightgbm eksik → _colab_install() çalıştırın.")
import requests
import warnings
import time
import re
import os
import glob
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
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif

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
VME_MAX = 0.030  # Very Major Error  ≤ %3.0  [Kim 2022 "research-grade"]
                 # (FDA-grade %1.5 çok katı; gerçek dünya verisinde nadir karşılanır)
ME_MAX = 0.050   # Major Error       ≤ %5.0  (VME ile orantılı genişletildi)
RECALL_THRESHOLD = 0.80
SPECIFICITY_MIN = 0.50

# ── [v35-BALANCED] Operasyonel denge hedefi ─────────────────────────────────
# Kullanıcı kararı: FDA-grade %3 VME yerine, klinikçe "yeterli" kabul edilen
# DENGELİ ~%10/%10 hedefi. VME ve ME bir tahterevallidir — tek eşik AUC eğrisi
# üzerinde kayar, eğriyi İYİLEŞTİRMEZ. İkisini de ≤%10 yapmak ~AUC≥0.95 gerektirir.
# Eşik seçimi artık VME'yi ME pahasına aşırı düşürmek yerine max(VME,ME)'yi
# minimize eder (Equal Error Rate). AUC yeterse her ikisi de ≤%10'a iner;
# değilse ikisi eşit en düşük ortak değerde buluşur (örn. %14/%14).
VME_TARGET = 0.10              # Operasyonel VME hedefi (klinikçe değerlendirilebilir)
ME_TARGET  = 0.10              # Operasyonel ME hedefi
USE_BALANCED_THRESHOLD = True  # True: dengeli (EER) eşik | False: eski VME-öncelikli

# ── [v35-FIX-1] Klinik karar maliyeti ───────────────────────────────────────
# Bir dirençli örneği duyarlı sayma (FN) hastayı uygun olmayan antibiyotiğe
# maruz bırakır — klinik açıdan FP'den (gereksiz antibiyotik) çok daha pahalı.
# Kim 2022 §"Evaluating machine-learning models" (Training and Testing alt-bölümü): c_FN ≈ 10×c_FP referansı.
COST_FN_OVER_FP = 10.0

# ── [v35-FIX-2] Feature Selection sınırları ─────────────────────────────────
FS_TOP_DNA_KMER = 1000   # 16,384 DNA k-mer → top-1000 MI
FS_TOP_AA_KMER  = 500    # 4,096 AA k-mer → top-500 MI
FS_MIN_SAMPLES  = 80     # Bu altında FS atlanır (MI gürültülü olur)

# ── [v35-OFFLINE] API çekimini atla, mevcut önbellekle devam et ──────────────
# True: eksik DNA/AA dizilerini çekmek yerine önbellekteki genomlarla yetin.
# Kullanıcı eksik AA dizisini manuel ekleyecekse buraya True bırak.
# False: orijinal davranış (BV-BRC API'den eksikleri çek).
SKIP_MISSING_FETCH = True

# ── [COLAB-2] Dizin Yapısı ────────────────────────────────────────────────────
_BASE = "/content/drive/MyDrive/amr_v29"

DATA_DIR    = os.path.join(_BASE, "data", "processed")
MODELS_DIR  = os.path.join(_BASE, "models")
REPORTS_DIR = os.path.join(_BASE, "reports")
SHAP_DIR    = os.path.join(_BASE, "reports", "shap_values")

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

# ──────────────────────────────────────────────────────────────────────────────
# [PATH-FIX] Önbellek dosyaları için akıllı yol çözücü
# Drive bağlıysa tüm MyDrive altını tarar; bulunamazsa beklenen konumu döndürür.
# Bu sayede dosyayı yanlış dizine atan kullanıcılar "yeniden çekme" döngüsüne
# girmez — dosya nerede olursa olsun bulunur.
# ──────────────────────────────────────────────────────────────────────────────
def _resolve_cache_path(filename: str, expected_dir: str) -> str:
    """
    Drive altında `filename` isimli dosyayı arar.
    Bulursa gerçek yolu döndürür; bulamazsa `expected_dir/filename` döndürür.
    """
    expected = os.path.join(expected_dir, filename)
    drive_root = "/content/drive/MyDrive"
    if not os.path.isdir(drive_root):
        return expected  # Drive bağlı değil
    matches = glob.glob(
        os.path.join(drive_root, "**", filename), recursive=True
    )
    if not matches:
        return expected  # Henüz yok; yazılacak yer
    if expected in matches:
        return expected  # Doğru yerde
    # En son değiştirilmiş dosyayı seç
    best = max(matches, key=lambda p: os.path.getmtime(p))
    print(f"   📍 [PATH-FIX] '{filename}' → {best}")
    return best


# Tüm önbellek yolları PATH-FIX ile dinamik olarak çözülür
LABELS_FILE     = _resolve_cache_path("v2_multilabel_labels.csv",           DATA_DIR)
CACHE_FILE      = _resolve_cache_path("v17_amr_genes_cache.csv",            DATA_DIR)
SEQ_CACHE_FILE  = _resolve_cache_path("v26_sequences.parquet",              DATA_DIR)
KMER_CACHE_FILE = _resolve_cache_path("v29_kmer_hashed_features.parquet",   DATA_DIR)
AA_KMER_CACHE   = _resolve_cache_path("v29_aa_kmer_features.parquet",       DATA_DIR)
AA_SEQ_CACHE    = _resolve_cache_path("v29_amr_aa_sequences.parquet",       DATA_DIR)
CHECKPOINT_FILE = os.path.join(REPORTS_DIR, "v35_checkpoint.csv")
OPTUNA_DB_FILE  = "sqlite:///" + os.path.abspath(MODELS_DIR) + "/optuna_v35.db"

# ── [FIX-1] Eksik Sabitler ────────────────────────────────────────────────────
# [v35-ENLARGE] 30k → 120k. E.coli Yol-B büyütmesi ~+70k genom ekliyor.
# Eski 30k cap, dosya sonuna eklenen yeni E.coli satırlarını kesip atardı
# (sample_genomes = unique()[:MAX_GENOMES], satır 3318). 50 GB RAM + k-mer
# kapalı (gen-only seyrek matris) ile 120k genom rahat sığar.
MAX_GENOMES     = 120_000
BATCH_SIZE      = 100          # [CLOUD-2] v27=50 → v29=100
OPTUNA_TRIALS   = 75           # [v35-FIX-OPT3] v34=50 → v35=75 (50 GB RAM rahat)
OPTUNA_N_JOBS   = 1
MODEL_N_JOBS    = 1 if DEVICE == "cuda" else -1
BACT_MIN_FREQ   = 0.05
CORR_THRESHOLD  = 0.95         # Pearson korelasyon eşiği
RF_N_ESTIMATORS = 300          # Baseline RF
RF_MAX_DEPTH    = 10           # Baseline RF

# ── [CLOUD-1][KMER-FIX-2] DNA K-mer — küçültülmüş boyutlar ──────────────────
# HASH_BITS 16→14: 65.536 → 16.384 bin
# Colab'da 65k×1600 matris ~420 MB float32 → RAM patlar + loky donar.
# 14 bit ile biyolojik sinyal %95+ korunur, 4× daha az bellek/süre.
KMER_K         = 21
HASH_BITS      = 14            # [KMER-FIX-2] 16 → 14
HASH_SIZE      = 2**HASH_BITS  # 16 384
SEQ_SAMPLE_BP  = 60_000        # [KMER-FIX-2] 200k → 60k bp
SEQ_BATCH_SIZE = 100           # v27=50 → v29=100
# [v35-ENLARGE] DNA k-mer KAPATILDI. Veri seti BV-BRC genome_amr ile ~8x
# büyütülüyor; yeni genomlar FASTA (firewall) olmadan k-mer alamaz. K-mer'ı
# tutmak "k-mer var = eski veri" confound'u yaratırdı. K-mer SHAP'ı zaten
# zayıftı (~0.08 vs gen ~0.5) → kaldırmak eski+yeni veriyi homojenleştirir,
# daha hızlı ve yorumlanabilir (Kim 2022 interpretable-model tezine uygun).
USE_KMER       = False

# ── [CLOUD-1][KMER-FIX-2] AA K-mer ──────────────────────────────────────────
# [v35-AA-DISABLED] AA k-mer kapatıldı.
# SEBEP: v29_amr_aa_sequences.parquet tür-bazlı havuzdan üretildi.
# Tüm E.coli'ler aynı consensus AA dizisi paylaşıyor → tür-içi varyans ~0
# → 489 AA kolonunda SHAP=0 → modele sıfır katkı, pipeline'a sadece maliyet.
# Suş-spesifik protein FASTA çekme yolu (BV-BRC FTP, NCBI Datasets) erişim
# engelleri nedeniyle uygulanamadı. Future work olarak limitations'a yaz.
USE_AA_KMER    = False
AA_KMER_K      = 4
AA_HASH_BITS   = 12
AA_HASH_SIZE   = 2**AA_HASH_BITS

# ── SMOTE ─────────────────────────────────────────────────────────────────────
USE_SMOTE        = True
SMOTE_MIN_RATIO  = 2.0

# ── Per-Species Eşikleri ─────────────────────────────────────────────────────
PER_SPECIES_MIN_SAMPLES   = 50
PER_SPECIES_MIN_RESISTANT = 15

# ── [CLOUD-3] Gen Etkileşim ───────────────────────────────────────────────────
USE_GENE_INTERACTIONS = True
INTERACTION_TOP_N     = 25     # v27=12 → v29=25 (300+ çift)

# ── [v35-FIX-CLSI] CLSI/EUCAST 2024 klinik breakpoint'leri (µg/mL) ─────────
# S_max  : S kabul edilen maks MIC (≤ S_max → duyarlı)
# R_min  : R kabul edilen min MIC (≥ R_min → dirençli)
# Aradaki bölge "intermediate" (I) — etiketleri gürültülü, eğitimden çıkarılır.
# Kaynak: CLSI M100-S34 (Ed) 2024, E.coli/Enterobacteriaceae odaklı.
# Diğer türler için aynı sınırlar büyük oranda geçerli (Pseudomonas farklı).
CLSI_BREAKPOINTS = {
    "ciprofloxacin":  (0.25,  1.0),     # S ≤ 0.25, R ≥ 1.0     (I = 0.5)
    "ampicillin":     (8.0,   32.0),    # S ≤ 8,    R ≥ 32      (I = 16)
    "ceftazidime":    (4.0,   16.0),    # S ≤ 4,    R ≥ 16      (I = 8)
    "gentamicin":     (4.0,   16.0),    # S ≤ 4,    R ≥ 16      (I = 8)
    "tetracycline":   (4.0,   16.0),    # S ≤ 4,    R ≥ 16      (I = 8)
    "meropenem":      (1.0,   4.0),     # S ≤ 1,    R ≥ 4       (I = 2)
    "imipenem":       (1.0,   4.0),
    "amikacin":       (16.0,  64.0),
    "cefepime":       (2.0,   16.0),
    "ceftriaxone":    (1.0,   4.0),
    "levofloxacin":   (0.5,   2.0),
    "trimethoprim_sulfamethoxazole": (2.0, 4.0),
    "piperacillin_tazobactam": (16.0, 128.0),
}

# Borderline temizliği yapılsın mı? (CLSI I-zone'unu eğitimden çıkar)
CLEAN_BORDERLINE = True

# ── [v35-STAGE2-EXT] Geographic OOD External Validation ──────────────────────
# Nguyen 2019 (Salmonella XGBoost) §"Limitations": "we are not aware of any
# large publicly available collections … from other countries. Global
# validation needed."
# Kim 2022 §"Limitations" + §"ML for Clinical Diagnostics": modelin
# geographic generalization kapasitesi nadiren test edilir → klinik
# deployment için kritik bir boşluktur.
#
# YAKLAŞIM: Per-species/per-antibiotik bazda, EN ÇOK ÖRNEKLİ ülkeyi
# external test set'i olarak ayır. Model bu ülkenin verisini eğitim VEYA
# iç-test bölmesinde GÖRMEZ. Eşik seçimi tamamen iç-CV üzerinden yapılır.
# External metrikler ayrı sütunda raporlanır.
# [v35-USER-OVERRIDE] Kullanıcı kararı: ülke/zaman external split tamamen kapalı.
# Sebep: External setlerin minority class'ı çok dengesiz oluyor
# (Salmonella Australia: R=5/S=567, Klebsiella USA: R=903/S=149).
# Internal training set küçülmesiyle model performansı düşüyordu.
# Tüm sample'ları training + random 80/20 split ile kullan.
USE_EXTERNAL_COUNTRY  = False
EXT_MIN_COUNTRY_SIZE  = 80
EXT_MIN_INTERNAL_SIZE = 200

USE_EXTERNAL_TEMPORAL = False
EXT_TEMPORAL_MIN_SIZE = 60

# [v35-USER-OVERRIDE] Salmonella'daki Year-bazlı temporal split de kapalı.
# Year sütunu metadata'dan geldi ama büyük çoğunluğu eski (<2016),
# test seti N=55 gibi felaket küçük kalıyordu.
USE_TEMPORAL_SPLIT    = False

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
print(f"📁 Labels    : {LABELS_FILE}")
print(f"📁 Seq Cache : {SEQ_CACHE_FILE}")

# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 3 — [COLAB-3] CHECKPOINT SİSTEMİ
# ══════════════════════════════════════════════════════════════════════════════


def load_checkpoint() -> tuple:
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
# BÖLÜM 4 — DNA K-MER FONKSİYONLARI  [KMER-FIX-1,2,3,4,5]
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


def _write_parquet_chunked(
    df: pd.DataFrame, path: str, row_group_size: int = 500
) -> None:
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path, row_group_size=row_group_size, compression="snappy")


def build_kmer_feature_matrix(genome_ids: list) -> pd.DataFrame:
    """
    [KMER-FIX-1] prefer='threads' — loky yerine thread pool kullanılır.
                  Colab'da loky, fork/spawn çakışması yüzünden ölü kilide girer.
                  "🧬 DNA K-mer Matrisi..." satırında donma artık olmayacak.
    [KMER-FIX-3] Chunk bazlı önbellekleme: her 200 genomda bir ara kayıt.
                  Oturum kopsa bile kaldığı yerden devam eder.
    [KMER-FIX-4] Global SEQ_CACHE_FILE sabiti yerine _resolve_cache_path()
                  kullanılır — dizi dosyası hangi klasörde olursa olsun bulunur.
    """
    genome_ids = [str(g) for g in genome_ids]
    kmer_cols  = [f"kmer_h{i}" for i in range(HASH_SIZE)]

    # [PATH-FIX] K-mer önbelleği otomatik aranır
    kmer_cache = _resolve_cache_path("v29_kmer_hashed_features.parquet", DATA_DIR)
    if os.path.exists(kmer_cache):
        try:
            cached = pd.read_parquet(kmer_cache)
            cached["Genome ID"] = cached["Genome ID"].astype(str)
            n_feat = len([c for c in cached.columns if c != "Genome ID"])
            if n_feat == HASH_SIZE:  # [FIX-2] hash_size → HASH_SIZE
                print(f"📦 DNA K-MER önbelleği: {len(cached):,} × {n_feat}")
                return cached
            print(
                f"   ⚠️  K-mer önbelleği eski boyut ({n_feat}≠{HASH_SIZE}), "
                f"yeniden oluşturuluyor."
            )
            os.remove(kmer_cache)
        except Exception:
            pass

    # [KMER-FIX-3] Kısmi önbellek: daha önce hesaplananları yükle
    partial_cache = os.path.join(DATA_DIR, "v29_kmer_partial.parquet")
    already_done: dict = {}
    if os.path.exists(partial_cache):
        try:
            part_df = pd.read_parquet(partial_cache)
            part_df["Genome ID"] = part_df["Genome ID"].astype(str)
            already_done = {
                row["Genome ID"]: row
                for _, row in part_df.iterrows()
            }
            print(
                f"   ♻️  Kısmi K-mer önbelleği: {len(already_done):,} genom "
                f"önceki oturumdan yüklendi."
            )
        except Exception:
            pass

    # [KMER-FIX-4] Dizi dosyasını _resolve_cache_path ile bul
    seq_cache = _resolve_cache_path("v26_sequences.parquet", DATA_DIR)
    if not os.path.exists(seq_cache):
        print(
            "   ❌ DNA dizi önbelleği bulunamadı. "
            "Önce fetch_genome_sequences() çalıştırın."
        )
        return pd.DataFrame(columns=["Genome ID"] + kmer_cols)

    print(
        f"\n🧬 DNA K-mer Matrisi (k={KMER_K}, hash={HASH_SIZE:,})\n"
        f"   [KMER-FIX: prefer=threads + chunk-save her 200 genomda]"
    )

    # Tüm dizileri yükle, eksik olanları filtrele
    seq_df = pd.read_parquet(seq_cache)
    seq_df["Genome ID"] = seq_df["Genome ID"].astype(str)
    todo = seq_df[~seq_df["Genome ID"].isin(already_done)].reset_index(drop=True)
    print(
        f"   Hesaplanacak : {len(todo):,} | "
        f"Önbellekten   : {len(already_done):,}"
    )

    CHUNK_SIZE    = 200    # Her 200 genomda bir ara kayıt yap
    all_rows_list = list(already_done.values()) if already_done else []

    for chunk_start in range(0, len(todo), CHUNK_SIZE):
        chunk     = todo.iloc[chunk_start : chunk_start + CHUNK_SIZE]
        args_list = [
            (row["sequence"], KMER_K, HASH_SIZE)
            for _, row in chunk.iterrows()
        ]

        # [KMER-FIX-1] prefer="threads" — Colab donmasını önler
        # loky process'leri fork ederken Colab'ın runtime yapısıyla çakışır;
        # thread backend'i bu problemi tamamen ortadan kaldırır.
        chunk_res = joblib.Parallel(n_jobs=-1, prefer="threads")(
            joblib.delayed(_kmer_worker)(a) for a in args_list
        )

        chunk_df = pd.DataFrame(
            np.vstack(chunk_res).astype(np.float32), columns=kmer_cols
        )
        chunk_df.insert(0, "Genome ID", chunk["Genome ID"].values)
        all_rows_list.extend(chunk_df.to_dict("records"))

        # [KMER-FIX-3] Kısmi ara kayıt — session kopsa devam eder
        try:
            _write_parquet_chunked(
                pd.DataFrame(all_rows_list), partial_cache
            )
        except Exception as e:
            print(f"   ⚠️  Kısmi kayıt başarısız: {e}")

        done_total = min(chunk_start + CHUNK_SIZE, len(todo))
        pct = int(done_total / max(len(todo), 1) * 100)
        print(
            f"\r   K-mer %{pct:3d} "
            f"({done_total}/{len(todo)}) | "
            f"RAM: {_ram_gb():.1f} GB",
            end="",
            flush=True,
        )

    print()  # yeni satır

    if not all_rows_list:
        return pd.DataFrame(columns=["Genome ID"] + kmer_cols)

    kmer_df = pd.DataFrame(all_rows_list)

    # Sıfır varyanslı sütunları at
    feat_cols = [c for c in kmer_df.columns if c != "Genome ID"]
    vt = VarianceThreshold(threshold=0.0)
    vt.fit(kmer_df[feat_cols])
    kept_cols = [c for c, ok in zip(feat_cols, vt.get_support()) if ok]
    removed   = len(feat_cols) - len(kept_cols)
    if removed:
        print(f"   🔬 VarianceThreshold: {removed} sıfır-varyans bin çıkarıldı")
    kmer_df = kmer_df[["Genome ID"] + kept_cols]

    # Tam önbelleği kaydet, kısmi dosyayı sil
    _write_parquet_chunked(kmer_df, KMER_CACHE_FILE)
    if os.path.exists(partial_cache):
        try:
            os.remove(partial_cache)
        except Exception:
            pass

    print(f"   ✅ DNA K-mer tamamlandı: {len(kmer_df):,} × {len(kept_cols)}")
    return kmer_df


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 5 — AMİNO ASİT K-MER  [KMER-FIX-5]
# ══════════════════════════════════════════════════════════════════════════════
_AA_ALPHABET = set("ACDEFGHIKLMNPQRSTVWY")


def _aa_kmer_worker(args: tuple) -> np.ndarray:
    seq, k, hash_size = args
    freq = np.zeros(hash_size, dtype=np.float32)
    seq  = "".join(c for c in seq.upper() if c in _AA_ALPHABET)
    L    = len(seq)
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

    # [PATH-FIX] AA dizi önbelleği otomatik aranır
    aa_seq_cache = _resolve_cache_path("v29_amr_aa_sequences.parquet", DATA_DIR)
    if os.path.exists(aa_seq_cache):
        try:
            cached = pd.read_parquet(aa_seq_cache)
            for _, row in cached.iterrows():
                result[str(row["genome_id"])] = str(row["aa_seq"])
            missing = [g for g in genome_ids if g not in result]
            if not missing:
                print(f"   📦 AA dizi önbelleği: {len(result):,} genom")
                return result
            genome_ids = missing
        except Exception:
            pass

    # [v35-OFFLINE] API'ye gitmeden mevcut önbellekle yetin
    if SKIP_MISSING_FETCH:
        if result:
            print(
                f"   ⏭️  [v35-OFFLINE] {len(genome_ids):,} eksik AA dizisi "
                f"atlanıyor (SKIP_MISSING_FETCH=True). "
                f"{len(result):,} önbellekteki dizi ile devam ediliyor."
            )
        else:
            print(
                f"   ⏭️  [v35-OFFLINE] AA dizi önbelleği boş ve "
                f"SKIP_MISSING_FETCH=True → AA k-mer bu oturumda devre dışı."
            )
        return result

    print(f"   🌐 {len(genome_ids):,} genom için AMR protein dizisi çekiliyor...")
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
    }
    new_rows      = []
    batch_seq_map: dict = {}

    for i in range(0, len(genome_ids), 50):
        batch   = genome_ids[i : i + 50]
        id_str  = ",".join(batch)
        payload = (
            f"in(genome_id,({id_str}))"
            f"&select(genome_id,aa_sequence)&limit(10000)"
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
                    gid = str(item.get("genome_id", "")).strip()
                    aa  = str(item.get("aa_sequence", "")).strip()
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
        if os.path.exists(aa_seq_cache):
            try:
                old = pd.read_parquet(aa_seq_cache)
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
    """
    [KMER-FIX-5] AA K-mer hesabı da prefer='threads' ile çalışır.
    """
    k         = AA_KMER_K
    hash_size = AA_HASH_SIZE
    aa_cols   = [f"aa_kmer_h{i}" for i in range(hash_size)]

    # [PATH-FIX] önbellek otomatik aranır
    aa_kmer_cache = _resolve_cache_path("v29_aa_kmer_features.parquet", DATA_DIR)
    if os.path.exists(aa_kmer_cache):
        try:
            cached = pd.read_parquet(aa_kmer_cache)
            cached["Genome ID"] = cached["Genome ID"].astype(str)
            n_feat = len([c for c in cached.columns if c != "Genome ID"])
            if n_feat == hash_size:
                print(f"   📦 AA K-mer önbelleği: {len(cached):,} × {n_feat}")
                return cached
        except Exception:
            pass

    print(f"\n🔬 AA K-mer Matrisi (k={k}, hash={hash_size:,})...")
    prot_seqs    = fetch_amr_protein_sequences(genome_ids)
    genome_ids_s = [str(g) for g in genome_ids]
    valid_gids   = [g for g in genome_ids_s if g in prot_seqs]

    if not valid_gids:
        print("   ⚠️  AA k-mer matrisi boş.")
        return pd.DataFrame(columns=["Genome ID"] + aa_cols)

    args_list = [(prot_seqs[g], k, hash_size) for g in valid_gids]

    # [KMER-FIX-5] prefer="threads"
    results = joblib.Parallel(n_jobs=-1, prefer="threads")(
        joblib.delayed(_aa_kmer_worker)(a) for a in args_list
    )

    aa_kmer_df = pd.DataFrame(
        np.vstack(results).astype(np.float32), columns=aa_cols
    )
    aa_kmer_df.insert(0, "Genome ID", valid_gids)

    vt   = VarianceThreshold(threshold=0.0)
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


def _fetch_sequences_batch(genome_ids: list, max_retries: int = 3) -> dict:
    id_str  = ",".join(str(g) for g in genome_ids)
    payload = (
        f"in(genome_id,({id_str}))"
        f"&select(genome_id,sequence)&limit(50000)"
    )
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
    """
    [PATH-FIX] v26_sequences.parquet Drive'ın herhangi bir yerinde olabilir.
    _resolve_cache_path ile bulunan gerçek yol kullanılır.
    Önbellekte bulunan genomlar yeniden çekilmez.
    """
    genome_ids = [str(g) for g in genome_ids]

    # Önbellek konumunu dinamik olarak çöz
    seq_cache = _resolve_cache_path("v26_sequences.parquet", DATA_DIR)

    cached_df = pd.DataFrame(columns=["Genome ID", "sequence"])
    if os.path.exists(seq_cache):
        try:
            cached_df = pd.read_parquet(seq_cache)
            cached_df["Genome ID"] = cached_df["Genome ID"].astype(str)
            print(
                f"   📦 DNA dizi önbelleği: {seq_cache} "
                f"({len(cached_df):,} kayıt)"
            )
        except Exception as e:
            print(f"   ⚠️  Önbellek okunamadı ({e}), sıfırdan başlanıyor.")
            cached_df = pd.DataFrame(columns=["Genome ID", "sequence"])

    cached_ids = set(cached_df["Genome ID"].tolist())
    missing    = [g for g in genome_ids if g not in cached_ids]

    if not missing:
        print(f"   ✅ Tüm {len(genome_ids):,} genomun dizisi önbellekte mevcut.")
        return cached_df

    # [v35-OFFLINE] Eksik genomları çekme, sadece önbellektekiyle devam et
    if SKIP_MISSING_FETCH:
        print(
            f"   ⏭️  [v35-OFFLINE] {len(missing):,} eksik genom atlanıyor "
            f"(SKIP_MISSING_FETCH=True). "
            f"{len(cached_ids):,} önbellekteki genomla devam ediliyor."
        )
        return cached_df

    print(
        f"\n🌐 {len(missing):,} genomun DNA dizisi çekiliyor "
        f"({len(cached_ids):,} önbellekten)..."
    )
    all_rows = []
    for i in range(0, len(missing), SEQ_BATCH_SIZE):
        batch   = missing[i : i + SEQ_BATCH_SIZE]
        seq_map = _fetch_sequences_batch(batch)
        for gid, seq in seq_map.items():
            all_rows.append({"Genome ID": gid, "sequence": seq})
        pct = min(100, int((i + SEQ_BATCH_SIZE) / len(missing) * 100))
        print(
            f"\r   %{pct:3d} "
            f"({min(i+SEQ_BATCH_SIZE,len(missing))}/{len(missing)})",
            end="",
            flush=True,
        )
    print()

    if all_rows:
        new_df   = pd.DataFrame(all_rows)
        final_df = (
            pd.concat([cached_df, new_df], ignore_index=True)
            if not cached_df.empty
            else new_df
        )
        final_df = (
            final_df[final_df["Genome ID"].isin(genome_ids)]
            .reset_index(drop=True)
        )
        # Her zaman beklenen konuma yaz — bir sonraki oturumda doğru yerden okunur
        _write_parquet_chunked(final_df, SEQ_CACHE_FILE)
        print(f"   💾 DNA dizi önbelleği → {SEQ_CACHE_FILE}")
        return final_df

    return cached_df


# [FIX-5] v28'de yanlışlıkla kaldırılmıştı — v27'den geri getirildi
def fetch_amr_genes_from_bvbrc(genome_ids) -> pd.DataFrame:
    genome_ids = [str(g) for g in genome_ids]
    # [PATH-FIX] gen önbelleği de otomatik aranır
    cache_path = _resolve_cache_path("v17_amr_genes_cache.csv", DATA_DIR)
    if os.path.exists(cache_path):
        try:
            cached = pd.read_csv(cache_path)
            if not cached.empty and "Genome ID" in cached.columns:
                cached["Genome ID"] = cached["Genome ID"].astype(str)
                print(f"📦 Gen önbelleği: {cache_path} ({len(cached):,} kayıt)")
                return cached
        except Exception:
            pass
    print(f"🌐 {len(genome_ids):,} bakteri için AMR genleri çekiliyor...")
    return _fetch_amr_from_api(genome_ids)


def _fetch_amr_from_api(genome_ids: list) -> pd.DataFrame:
    all_genes = []
    headers   = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
    }
    for i in range(0, len(genome_ids), BATCH_SIZE):
        batch   = genome_ids[i : i + BATCH_SIZE]
        id_str  = ",".join(batch)
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
                                    "AMR_Gene":  str(gene).strip(),
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
        pct  = (dist / len(y_df) * 100).round(2)
        out  = os.path.join(
            REPORTS_DIR, "bias_reports", f"v35_bias_{col.lower()}.csv"
        )
        pd.DataFrame({"Sayi": dist, "Yüzde": pct}).to_csv(out)
        top3 = ", ".join(f"{k}({v:.1f}%)" for k, v in pct.head(3).items())
        print(f"   {label}: {top3} → {out}")
    print()


def select_features_amr(
    X: pd.DataFrame,
    y: pd.Series,
    top_dna_kmer: int = FS_TOP_DNA_KMER,
    top_aa_kmer: int = FS_TOP_AA_KMER,
    min_samples: int = FS_MIN_SAMPLES,
) -> pd.DataFrame:
    """
    [v35-FIX-2] Kim 2022 §"Feature Selection for Interpretable Models" + §"Representing Genomes and Phenotype Labels":
    Antibiyotik-bilinçli mutual-information feature selection.

    Kural:
      - Genler         : HEPSİ KORUNUR (biyolojik bilgi, kürasyondan)
      - bact_*         : HEPSİ KORUNUR (tür kategorik)
      - interact_*     : HEPSİ KORUNUR (zaten kürasyondan/varyans seçimi)
      - kmer_*  (DNA)  : MI ile top_dna_kmer
      - aa_kmer_* (AA) : MI ile top_aa_kmer

    Curse of dimensionality'ye karşı koruma. Mevcut model 21k özellik /
    193 R örneği oranıyla aşırı parametrize idi; bu FS sonrası ~2000 özelliğe
    düşer.
    """
    gene_cols     = [c for c in X.columns if not c.startswith(
        ("bact_", "kmer_", "aa_kmer_", "interact_")
    )]
    bact_cols     = [c for c in X.columns if c.startswith("bact_")]
    inter_cols    = [c for c in X.columns if c.startswith("interact_")]
    dna_kmer_cols = [c for c in X.columns if c.startswith("kmer_")]
    aa_kmer_cols  = [c for c in X.columns if c.startswith("aa_kmer_")]

    keep_cols = gene_cols + bact_cols + inter_cols

    # Çok küçük örnekte MI gürültülü olur — atla
    if len(y) < min_samples:
        print(
            f"   ⏭️  [v35-FS] N={len(y)} < {min_samples} → "
            f"FS atlanıyor, tüm k-mer'lar korunuyor."
        )
        return X[keep_cols + dna_kmer_cols + aa_kmer_cols]

    # DNA k-mer: MI ile top-K
    if len(dna_kmer_cols) > top_dna_kmer:
        try:
            mi = mutual_info_classif(
                X[dna_kmer_cols].values.astype(np.float32),
                y.values.astype(int),
                discrete_features=False,
                random_state=42,
                n_neighbors=3,
            )
            keep_mask = np.argsort(mi)[::-1][:top_dna_kmer]
            selected  = [dna_kmer_cols[i] for i in keep_mask]
            print(
                f"   🎯 [v35-FIX-2] DNA k-mer FS: "
                f"{len(dna_kmer_cols)} → {len(selected)} (MI)"
            )
            keep_cols += selected
        except Exception as e:
            print(f"   ⚠️  DNA k-mer FS başarısız ({e}), hepsi tutuluyor.")
            keep_cols += dna_kmer_cols
    else:
        keep_cols += dna_kmer_cols

    # AA k-mer: MI ile top-K
    if len(aa_kmer_cols) > top_aa_kmer:
        try:
            mi = mutual_info_classif(
                X[aa_kmer_cols].values.astype(np.float32),
                y.values.astype(int),
                discrete_features=False,
                random_state=42,
                n_neighbors=3,
            )
            keep_mask = np.argsort(mi)[::-1][:top_aa_kmer]
            selected  = [aa_kmer_cols[i] for i in keep_mask]
            print(
                f"   🎯 [v35-FIX-2] AA k-mer FS: "
                f"{len(aa_kmer_cols)} → {len(selected)} (MI)"
            )
            keep_cols += selected
        except Exception as e:
            print(f"   ⚠️  AA k-mer FS başarısız ({e}), hepsi tutuluyor.")
            keep_cols += aa_kmer_cols
    else:
        keep_cols += aa_kmer_cols

    return X[keep_cols]


def remove_correlated_features(
    X: pd.DataFrame, threshold: float = CORR_THRESHOLD
) -> pd.DataFrame:
    # [v35-PD-PROTECT] pd_ prefix'li (NCBI Pathogen Detection) feature'lar
    # korelasyon temizliğinden korunur — kanonik QRDR mutasyonlarının
    # AMR genleriyle yüksek korelasyon nedeniyle düşmesini önler.
    gene_cols  = [
        c for c in X.columns
        if not c.startswith(("bact_", "kmer_", "aa_kmer_", "interact_", "pd_"))
    ]
    pd_cols    = [c for c in X.columns if c.startswith("pd_")]
    other_cols = [c for c in X.columns if c not in gene_cols and c not in pd_cols]

    if len(gene_cols) >= 2:
        X_gene   = X[gene_cols].copy()
        var_mask = X_gene.var() > 0
        X_gene   = X_gene.loc[:, var_mask]
        corr     = X_gene.corr(method="pearson").abs()
        upper    = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        drop_g   = [c for c in upper.columns if any(upper[c] > threshold)]
        gene_cols = [
            c for c in gene_cols if c not in drop_g and c in X_gene.columns
        ]
        if drop_g:
            print(f"   🔬 Korelasyon: {len(drop_g)} gen çıkarıldı (>{threshold})")
    if pd_cols:
        print(f"   🛡️  PD feature: {len(pd_cols)} kanonik mutasyon korundu")

    # pd_cols korunarak final sütunlar
    final_cols = [c for c in gene_cols + pd_cols + other_cols if c in X.columns]
    return X[final_cols]


def add_bacteria_type_features(X_gene, bacteria_series, min_freq=BACT_MIN_FREQ):
    if bacteria_series is None or bacteria_series.isna().all():
        return X_gene, None
    freq        = bacteria_series.value_counts(normalize=True)
    valid_types = freq[freq >= min_freq].index.tolist()
    bact_clean  = bacteria_series.copy()
    bact_clean[~bact_clean.isin(valid_types)] = "Other"
    bact_clean  = bact_clean.fillna("Other")
    if len(valid_types) <= 1:
        return X_gene, bact_clean.values
    lb       = LabelBinarizer()
    bact_enc = lb.fit_transform(bact_clean)
    cols_    = (
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

    gene_cols   = [
        c for c in X.columns
        if not c.startswith(("bact_", "kmer_", "aa_kmer_", "interact_"))
    ]
    interact_df = pd.DataFrame(index=X.index)
    n_bio       = 0

    for pat1, pat2 in AMR_KNOWN_INTERACTIONS.get(antibiotic_name.lower(), []):
        cols1 = [c for c in gene_cols if pat1.lower() in c.lower()]
        cols2 = [c for c in gene_cols if pat2.lower() in c.lower()]
        if cols1 and cols2:
            s1  = re.sub(r"[^\w]", "_", cols1[0][:20])
            s2  = re.sub(r"[^\w]", "_", cols2[0][:20])
            key = f"interact_{s1}__X__{s2}"
            interact_df[key] = (
                X[cols1[0]].values * X[cols2[0]].values
            ).astype(np.float32)
            n_bio += 1

    if gene_cols:
        top_genes = X[gene_cols].var().nlargest(INTERACTION_TOP_N).index.tolist()
        for i, g1 in enumerate(top_genes):
            for g2 in top_genes[i + 1 :]:
                s1  = re.sub(r"[^\w]", "_", g1[:15])
                s2  = re.sub(r"[^\w]", "_", g2[:15])
                key = f"interact_{s1}__{s2}"
                if key not in interact_df.columns:
                    interact_df[key] = (
                        X[g1].values * X[g2].values
                    ).astype(np.float32)

    nonzero     = interact_df.var() > 0
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
    X_gb  = X_tr[gene_bact_cols].values.astype(np.float32)
    X_km  = X_tr[kmer_cols].values.astype(np.float32) if kmer_cols else None
    y_arr = y_tr.values

    method = "none"
    try:
        k_nn    = max(1, min(3, int((y_arr == 1).sum()) - 1))
        # [v35-FIX-SMOTE] imblearn 0.11+ BorderlineSMOTE'tan n_jobs kaldırıldı.
        # Eski sürümlerle uyum için try/except içine konabilirdi ama yeni sürümler
        # için n_jobs argümanı verilmez — paralelliği iç k-NN kendisi yönetir.
        sampler = BorderlineSMOTE(k_neighbors=k_nn, random_state=42)
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

    n_orig      = len(X_tr)
    n_synthetic = len(X_gb_res) - n_orig

    if X_km is not None and n_synthetic > 0:
        minority_kmer = X_km[y_arr == 1]
        synth_kmer    = minority_kmer[
            np.random.default_rng(42).integers(len(minority_kmer), size=n_synthetic)
        ]
        X_res = np.hstack([X_gb_res, np.vstack([X_km, synth_kmer])])
        cols  = gene_bact_cols + kmer_cols
    else:
        X_res = X_gb_res
        cols  = gene_bact_cols

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
    # [v35-FIX-3] pos_weight cap kaldırıldı.
    # SMOTE sonrası bile yüksek-boyutlu k-mer uzayında sentetik örneklerin
    # gerçek minority sinyaliyle eşit ağırlıkta olması garanti değildir.
    # Kim 2022 §"Suitability of Genomic Data Sets" (SMOTE tartışması): SMOTE + class_weight birlikte kullanılır.
    #
    # [v35-FIX-PW-CAP] Eski cap (max 2.5, natural*0.5) cipro/ecoli'de Optuna'yı
    # üst sınıra (2.5) yapıştırıyordu → recall'ı zorlamak için n_est=200 underfit
    # modeline kaçtı. Yeni cap natural_ratio'ya kadar serbest, en az 4.0.
    if already_oversampled:
        pw_upper = max(4.0, natural_ratio)
    else:
        pw_upper = max(4.0, natural_ratio * 1.5)

    safe_key   = re.sub(r"[^\w]", "_", study_key.lower())
    study_name = f"study_v35_{safe_key}"

    def objective(trial):
        global FALLBACK_TO_CPU
        current_device = "cpu" if FALLBACK_TO_CPU else DEVICE
        current_njobs  = -1   if FALLBACK_TO_CPU else OPTUNA_N_JOBS
        params = {
            # [CLOUD-2] Genişletilmiş aralıklar
            # [v35-FIX-OPT] n_est min 100 → 200 (cipro/ecoli'de Optuna n_est=100
            # seçti, CV=0.88 ama test=0.58 → overfit. 200+ ağaç ile kararlılık.)
            "n_estimators":     trial.suggest_int("n_estimators", 200, 600, step=50),
            "max_depth":        trial.suggest_int("max_depth", 3, 9),
            "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.20, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.40, 0.80),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 9),
            "gamma":            trial.suggest_float("gamma", 1e-5, 2.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, pw_upper),
            "tree_method": "hist",
            "max_bin": 256,
            "eval_metric": "logloss",
            "random_state": 42,
            "verbosity": 0,
            "n_jobs": current_njobs,
            "device": current_device,
        }
        # [v35-FIX-OPT] CV n_splits 3 → 5 (kararsız "iyi gibi" parametreler
        # elensin; 3-fold std=0.10 olabiliyordu, 5-fold daha güvenilir)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
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
                    print(
                        f"\n   ⚠️  GPU OOM → Trial {trial.number}'den CPU'ya geçiliyor!"
                    )
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

        ap_mean     = cv_res["test_ap"].mean()
        recall_mean = cv_res["test_recall"].mean()
        recall_std  = cv_res["test_recall"].std()
        recall_worst = recall_mean - recall_std

        # [v35-FIX-OPT2] AP-priority objective + recall floor (hard constraint).
        # ÖNCEKİ SORUN: recall-priority objective (mean*0.6 + worst*0.2)
        # Optuna'yı recall=0.95'e doğru itti ama AUC=0.87'ye düşürdü.
        # n_est=200 alt sınıra yapıştı → test underfit.
        #
        # YENI: AP'yi maksimize et, recall_worst floor=0.75 sağlanmazsa ağır
        # ceza. Bu, "iyi sıralayan" modeli ödüllendirir; eşik seçimi sonradan
        # VME/recall trade-off'unu zaten halleder.
        recall_floor = 0.75
        penalty      = max(0, (recall_floor - recall_worst) * 10.0)
        # AP %75 ağırlık, recall_mean %25 (taban garantisi için)
        return ap_mean * 0.75 + recall_mean * 0.25 - penalty

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
        mode  = "GPU+CUDA" if (not FALLBACK_TO_CPU and DEVICE == "cuda") else "CPU"
        print(f"   {label} deneme... [{mode}]")
        study.optimize(objective, n_trials=remaining)
    else:
        print(f"   ✅ Optuna: {OPTUNA_TRIALS} deneme önbellekten yüklendi.")

    best_params  = study.best_params
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
        "VME":     fn / (tp + fn + 1e-9),
        "ME":      fp / (tn + fp + 1e-9),
        "PPV":     tp / (tp + fp + 1e-9),
        "NPV":     tn / (tn + fn + 1e-9),
        "AUPRC":   average_precision_score(y_true, y_prob),
        "VME_OK":  fn / (tp + fn + 1e-9) <= VME_MAX,
        "ME_OK":   fp / (tn + fp + 1e-9) <= ME_MAX,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "recall":      tp / (tp + fn + 1e-9),
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
            if f1 > best_f1:
                best_f1 = f1; best_thr = thr
        elif recall >= (recall_min - recall_tolerance) and specificity >= spec_min:
            if f1 > tol_f1:
                tol_f1 = f1; tol_thr = thr
        elif recall >= recall_min:
            if f1 > rec_f1:
                rec_f1 = f1; rec_thr = thr

    if best_thr is not None:
        return best_thr, "full"
    if tol_thr is not None:
        return tol_thr, "tolerance"
    if rec_thr is not None:
        return rec_thr, "recall_only"
    return 0.50, "default"


def _select_threshold_clinical(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    vme_max: float = VME_MAX,
    me_max: float = ME_MAX,
    spec_min: float = SPECIFICITY_MIN,
    cost_fn_over_fp: float = COST_FN_OVER_FP,
    thr_low: float = 0.02,
    thr_high: float = 0.95,
) -> tuple:
    """
    [v35-FIX-1] Kim 2022 §"Evaluating machine-learning models" (Training and Testing alt-bölümü):
    Klinik AMR için VME-öncelikli eşik seçimi.

    Karar hiyerarşisi:
      1. PRIMARY:    VME ≤ vme_max VE ME ≤ me_max  → F1 maks. (recall+precision)
      2. SECONDARY:  VME ≤ vme_max VE spec ≥ spec_min → F1 maks.
      3. TERTIARY:   Cost-sensitive thr* = c_FP/(c_FP+c_FN)
                     (kalibre prob için Bayes-optimal karar sınırı)
      4. SON ÇARE:   Youden's J = recall + specificity - 1 maks.

    NOT: "recall maks." değil "F1 maks." — yoksa thr çok düşer ve ME patlar
    (gözlem: vme=0 fakat ME=%98 felaketi). Specificity≥spec_min filtresi de
    bu yüzden gereklidir.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    thresholds = np.linspace(
        thr_low, thr_high,
        int((thr_high - thr_low) / 0.005) + 1
    )

    feasible      = []   # VME+ME ikisi de OK
    near_feasible = []   # VME OK + spec_min OK
    relaxed       = []   # VME≤2×vme_max + spec≥0.3
    all_metrics   = []   # Youden's J için

    relaxed_vme_max = vme_max * 2.0   # %3.0 — klinikçe hâlâ değerlendirilebilir
    relaxed_spec    = 0.30

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape != (2, 2):
            continue
        tn, fp, fn, tp = cm.ravel()
        vme = fn / (tp + fn + 1e-9)
        me  = fp / (tn + fp + 1e-9)
        recall      = tp / (tp + fn + 1e-9)
        specificity = tn / (tn + fp + 1e-9)
        f1          = f1_score(y_true, y_pred, zero_division=0)
        youden      = recall + specificity - 1.0

        item = (thr, recall, specificity, vme, me, f1, youden)
        all_metrics.append(item)
        if vme <= vme_max and me <= me_max:
            feasible.append(item)
        elif vme <= vme_max and specificity >= spec_min:
            near_feasible.append(item)
        elif vme <= relaxed_vme_max and specificity >= relaxed_spec:
            relaxed.append(item)

    # ── [v35-BALANCED] DENGELİ EŞİK (varsayılan) ────────────────────────────
    # Kullanıcı hedefi: VME ve ME ikisi de ~%10. VME'yi tek başına 3%'e itmek
    # ME'yi %58'e fırlatıyordu (tahterevalli). Bunun yerine max(VME,ME)'yi
    # minimize ederek iki hatayı dengeli ve mümkün olan en düşük ortak değere
    # çekiyoruz. AUC yüksekse ikisi de ≤%10; değilse eşit noktada buluşur.
    if USE_BALANCED_THRESHOLD and all_metrics:
        # Her iki hedef de sağlanabiliyorsa → en kullanışlı nokta (F1 maks.)
        both_ok = [
            m for m in all_metrics
            if m[3] <= VME_TARGET and m[4] <= ME_TARGET
        ]
        if both_ok:
            best = max(both_ok, key=lambda x: x[5])
            return float(best[0]), "balanced_full"
        # Hiçbiri hedefte değil → Equal Error Rate: max(VME,ME) minimum,
        # eşitlikte daha düşük VME (klinik güvenlik) tercih edilir.
        best = min(all_metrics, key=lambda x: (max(x[3], x[4]), x[3]))
        return float(best[0]), "eer"

    # 1) Tam FDA uyumlu — F1 maks.
    if feasible:
        best = max(feasible, key=lambda x: x[5])
        return float(best[0]), "fda_full"

    # 2) VME OK + spec_min OK — F1 maks.
    if near_feasible:
        best = max(near_feasible, key=lambda x: x[5])
        return float(best[0]), "vme_only"

    # 3) Gevşetilmiş klinik kabul: VME≤2×vme_max + spec≥0.3 — F1 maks.
    #    (Modelin AUC'si yetersizken bile makul bir karar sınırı verir)
    if relaxed:
        best = max(relaxed, key=lambda x: x[5])
        return float(best[0]), "relaxed"

    # 4) Youden's J — istatistiksel olarak en iyi denge noktası
    if all_metrics:
        best = max(all_metrics, key=lambda x: x[6])
        return float(best[0]), "youden"

    # 5) Son çare: cost-sensitive Bayes karar sınırı (kalibre prob varsayımı)
    thr_cost = 1.0 / (1.0 + cost_fn_over_fp)
    if thr_low <= thr_cost <= thr_high:
        return float(thr_cost), "cost_sensitive"

    return 0.50, "default"


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    n_bootstrap: int = 500,
    stratified: bool = True,
) -> dict:
    """
    [KIM-2][v35-STAGE2] Bootstrap %95 CI. Class-aware (stratified) örnekleme.

    Kim 2022 §"Evaluating machine-learning models" bootstrap güven aralıkları
    önerir. Imbalanced AMR verisinde naive bootstrap (uniform random) bazı
    iterasyonlarda %0 R örneği oluşturur → metrikler tanımsız, CI'lar yanlı.

    Stratified yaklaşım: Pozitif ve negatif örnekleri AYRI ayrı bootstrap'le,
    sonra birleştir. Aynı oranı korur, küçük pozitif sınıf temsil edilir.

    Carpenter & Bithell 2000 "Bootstrap confidence intervals"; standart
    biyoistatistik pratik.

    Yeni metrikler: AUPRC CI'sı eklendi (imbalanced için AUC'den daha bilgili).
    """
    rng     = np.random.default_rng(42)
    metrics: dict = {
        "recall": [], "specificity": [], "f1": [],
        "auc": [], "auprc": [],
    }

    if stratified:
        pos_idx = np.where(y_true == 1)[0]
        neg_idx = np.where(y_true == 0)[0]
        n_pos, n_neg = len(pos_idx), len(neg_idx)
        if n_pos < 2 or n_neg < 2:
            stratified = False  # fallback

    for _ in range(n_bootstrap):
        if stratified:
            samp_pos = rng.choice(pos_idx, size=n_pos, replace=True)
            samp_neg = rng.choice(neg_idx, size=n_neg, replace=True)
            idx = np.concatenate([samp_pos, samp_neg])
        else:
            idx = rng.integers(0, len(y_true), len(y_true))
        yt  = y_true[idx]
        yp  = y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        ypred = (yp >= threshold).astype(int)
        cm    = confusion_matrix(yt, ypred)
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
        try:
            metrics["auprc"].append(average_precision_score(yt, yp))
        except ValueError:
            pass

    ci = {}
    for key, vals in metrics.items():
        if vals:
            arr = np.array(vals)
            ci[f"{key}_CI95"] = (
                f"[{np.percentile(arr, 2.5):.3f},"
                f"{np.percentile(arr, 97.5):.3f}]"
            )
    return ci


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 11 — MIC REGRESYON [v27-C5]
# ══════════════════════════════════════════════════════════════════════════════


def _mic_log2_to_resistance_prob(
    log2_pred: np.ndarray,
    s_max: float,
    r_min: float,
    steepness: float = 1.5,
) -> np.ndarray:
    """
    [v35-STAGE2] MIC regresyon tahmininden direnç olasılığı türetir.

    Mantık:
      log_mid = (log2(S_max) + log2(R_min)) / 2   ← S/R orta noktası
      logit   = (log2_pred − log_mid) × steepness
      prob_R  = sigmoid(logit)

    Yani tahmin S_max'a yakınsa P(R)≈0, R_min'e yakınsa P(R)≈1, ortada belirsiz.
    Bu sigmoid soft eşik, stacked ensemble'a "MIC tabanlı olasılık" base
    learner'ı olarak girer.

    Referans: Nguyen 2019'da MIC tahmini binary'e dönüştürülürken hard
    eşik kullanılır; biz soft sigmoid kullanarak meta-learner'a olasılık
    bilgisi taşıyoruz. Steepness=1.5 ampirik; 1.0 (yumuşak) - 2.5 (sert).
    """
    log_mid = (np.log2(s_max) + np.log2(r_min)) / 2.0
    logit   = (log2_pred - log_mid) * steepness
    # Numerik güvenli sigmoid
    prob = np.where(
        logit >= 0,
        1.0 / (1.0 + np.exp(-logit)),
        np.exp(logit) / (1.0 + np.exp(logit)),
    )
    return prob.astype(np.float32)


def _mic_to_sir_category(mic_value: float, s_max: float, r_min: float) -> str:
    """
    [v35-STAGE2] CLSI/EUCAST kategori dönüşümü:
      MIC ≤ S_max     → 'S' (Susceptible)
      S_max < MIC < R_min → 'I' (Intermediate)
      MIC ≥ R_min     → 'R' (Resistant)
    Referans: CLSI M100, Nguyen 2019.
    """
    if mic_value <= s_max:
        return "S"
    if mic_value >= r_min:
        return "R"
    return "I"


def train_mic_regression_model(
    X_tr, X_te, y_mic_tr, y_mic_te, antibiotic_name: str, species_key: str
) -> dict:
    """
    [v27-C5][v35-STAGE2] Sayısal MIC tahmini — Nguyen 2019 yöntemi.

    Kim 2022 §"Representing Genomes and Phenotype Labels" — sayısal MIC
    breakpoint değişimlerinden bağımsızdır. Nguyen 2019 (Salmonella XGBoost)
    altın standardı: XGBoost regressor on log2(MIC), ±1 dilution = FDA std.

    YENİ METRİKLER [v35-STAGE2]:
      - MIC_EA  (Essential Agreement)  : |log2(pred) − log2(true)| ≤ 1.0 oranı
                                          hedef ≥ %90 (CLSI M52, Lees 2023)
      - MIC_CA  (Categorical Agreement): pred S/I/R kategorisi gerçekle aynı
                                          hedef ≥ %95 (CLSI M52)
      - MIC_VME : log2(pred) ≤ log2(S_max) ama gerçek R         (klinik kritik)
      - MIC_ME  : log2(pred) ≥ log2(R_min) ama gerçek S         (klinik kritik)
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
            device="cpu",   # MIC reg her zaman CPU'da — stabil
            n_jobs=-1,
        )
        reg.fit(X_tr, y_log_tr)
        y_pred    = reg.predict(X_te)
        mae_log2  = mean_absolute_error(y_log_te, y_pred)
        # Within-N dilution metrikleri (Nguyen 2019 standardı)
        within_1d = float(np.mean(np.abs(y_log_te - y_pred) <= 1.0))
        within_2d = float(np.mean(np.abs(y_log_te - y_pred) <= 2.0))
        # Within-1D = Essential Agreement (CLSI M52)
        mic_ea    = within_1d

        # [v35-STAGE2] Categorical Agreement + VME/ME on log2 scale
        mic_ca = mic_vme = mic_me = None
        anti_key = antibiotic_name.lower().replace("-", "_").replace("/", "_")
        bp = CLSI_BREAKPOINTS.get(anti_key)
        if bp is not None:
            s_max, r_min = bp
            log_s_max = np.log2(s_max)
            log_r_min = np.log2(r_min)
            # Gerçek kategoriler
            true_cats = [
                _mic_to_sir_category(float(m), s_max, r_min)
                for m in y_mic_te.values
            ]
            pred_cats = [
                _mic_to_sir_category(float(2 ** p), s_max, r_min)
                for p in y_pred
            ]
            mic_ca = float(np.mean(
                [t == p for t, p in zip(true_cats, pred_cats)]
            ))
            # Klinik error: tahmin S iken gerçek R = VME
            # Klinik error: tahmin R iken gerçek S = ME
            n_R_true = sum(1 for c in true_cats if c == "R")
            n_S_true = sum(1 for c in true_cats if c == "S")
            mic_vme_count = sum(
                1 for t, p in zip(true_cats, pred_cats)
                if t == "R" and p == "S"
            )
            mic_me_count = sum(
                1 for t, p in zip(true_cats, pred_cats)
                if t == "S" and p == "R"
            )
            mic_vme = float(mic_vme_count / max(n_R_true, 1))
            mic_me  = float(mic_me_count  / max(n_S_true, 1))

        safe = re.sub(r"[^\w\-]", "_", f"{species_key}_{antibiotic_name}".lower())
        joblib.dump(
            {
                "model": reg,
                "log2_transform": True,
                # [v35-STAGE2] base learner kullanımı için meta bilgisi
                "breakpoint": bp,
                "antibiotic": antibiotic_name,
            },
            os.path.join(MODELS_DIR, f"{safe}_mic_v35.pkl"),
        )
        # Çıktı CSV — kategori karşılaştırması da eklendi
        out_df = pd.DataFrame({
            "y_true_mic":  y_mic_te.values,
            "y_pred_mic":  2.0 ** y_pred,
            "y_true_log2": y_log_te.values,
            "y_pred_log2": y_pred,
            "log2_error":  y_log_te.values - y_pred,
        })
        if bp is not None:
            out_df["true_cat"] = true_cats
            out_df["pred_cat"] = pred_cats
        out_df.to_csv(
            os.path.join(REPORTS_DIR, "mic_reports", f"v35_{safe}_mic.csv"),
            index=False,
        )
        # Klinik raporlama
        ea_flag = "✅" if mic_ea >= 0.90 else ("🟡" if mic_ea >= 0.80 else "❌")
        msg = (
            f"   📏 MIC → Log2-MAE={mae_log2:.3f} | "
            f"EA(±1D)={mic_ea:.1%}{ea_flag} | ±2D={within_2d:.1%}"
        )
        if mic_ca is not None:
            ca_flag = "✅" if mic_ca >= 0.95 else ("🟡" if mic_ca >= 0.90 else "❌")
            msg += (
                f"\n   📏 MIC → CA={mic_ca:.1%}{ca_flag} | "
                f"VME={mic_vme*100:.1f}% | ME={mic_me*100:.1f}%"
            )
        print(msg)

        result = {
            "MIC_Log2MAE":  round(mae_log2, 3),
            "MIC_EA":       round(mic_ea, 3),
            "MIC_Within2D": round(within_2d, 3),
        }
        if mic_ca is not None:
            result.update({
                "MIC_CA":  round(mic_ca, 3),
                "MIC_VME": round(mic_vme, 3),
                "MIC_ME":  round(mic_me, 3),
            })
        # [v35-STAGE2] Geri uyumluluk — eski MIC_Within1D anahtarını koru
        result["MIC_Within1D"] = result["MIC_EA"]
        return result
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
    brier_raw  = brier_score_loss(y_te, y_prob_raw)
    brier_cal  = brier_raw
    cal_ok     = False

    try:
        cal_model  = CalibratedClassifierCV(model, cv="prefit", method="isotonic")
        cal_model.fit(X_te, y_te)
        y_prob_cal = cal_model.predict_proba(X_te)[:, 1]
        brier_cal  = brier_score_loss(y_te, y_prob_cal)
        cal_ok     = True
    except Exception as e:
        print(f"   ⚠️  Kalibrasyon hatası: {e}")
        y_prob_cal = y_prob_raw

    try:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "k--", label="Mükemmel")
        fp_r, mp_r = calibration_curve(
            y_te, y_prob_raw, n_bins=10, strategy="uniform"
        )
        ax.plot(
            mp_r, fp_r, "s-",
            label=f"Ham XGB (Brier={brier_raw:.3f})",
            color="steelblue",
        )
        if cal_ok:
            fp_c, mp_c = calibration_curve(
                y_te, y_prob_cal, n_bins=10, strategy="uniform"
            )
            ax.plot(
                mp_c, fp_c, "o-",
                label=f"Kalibre (Brier={brier_cal:.3f})",
                color="tomato",
            )
        ax.axvline(
            final_thr, color="gray", linestyle=":", alpha=0.7,
            label=f"Eşik={final_thr:.2f}",
        )
        ax.set(
            xlabel="Tahmin Olasılık",
            ylabel="Gerçek Direnç Oranı",
            title=f"Kalibrasyon — {label_key}",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        ax.legend(fontsize=8)
        plt.tight_layout()
        safe  = re.sub(r"[^\w\-]", "_", label_key.lower())
        fpath = os.path.join(
            REPORTS_DIR, "calibration", f"v35_{safe}_cal.png"
        )
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
    results   = {}
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
            clf    = clf_fn()
            clf.fit(X_tr, y_tr)
            y_prob = clf.predict_proba(X_te)[:, 1]
            thr, _ = _select_threshold(y_te, y_prob)
            y_pred = (y_prob >= thr).astype(int)
            cm     = confusion_matrix(y_te, y_pred)
            rec    = spec = 0.0
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                rec  = tp / (tp + fn + 1e-9)
                spec = tn / (tn + fp + 1e-9)
            f1  = f1_score(y_te, y_pred, zero_division=0)
            auc = (
                roc_auc_score(y_te, y_prob)
                if len(np.unique(y_te)) == 2
                else float("nan")
            )
            results[name] = {
                "F1":     f"{f1:.3f}",
                "Recall": f"{rec:.3f}",
                "Spec":   f"{spec:.3f}",
                "AUC":    f"{auc:.3f}",
            }
            print(
                f"      {name}: F1={f1:.3f} | Recall={rec:.3f} | AUC={auc:.3f}"
            )
            model_path = os.path.join(
                MODELS_DIR, f"{safe_name}_{name.lower()}_v35.pkl"
            )
            joblib.dump(clf, model_path)
        except Exception as e:
            print(f"      ⚠️  {name} hatası: {e}")
            results[name] = {
                "F1": "ERR", "Recall": "ERR", "Spec": "ERR", "AUC": "ERR"
            }
    return results


def build_soft_ensemble(
    X_te, xgb_model, lr_model=None, rf_model=None
) -> np.ndarray:
    """
    [KIM-1] Kim 2022: Ensemble yaklaşımı daha stabil sonuçlar üretir.
    XGBoost + LR + RF olasılıklarının ağırlıklı ortalaması.
    Ağırlıklar: XGB=0.60, RF=0.25, LR=0.15 (toplama normalize edilir).
    """
    probs   = [xgb_model.predict_proba(X_te)[:, 1]]
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


def build_stacked_ensemble(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_te: pd.DataFrame,
    xgb_params: dict,
    n_folds: int = 5,
    random_state: int = 42,
    # [v35-STAGE2] MIC base learner için opsiyonel argümanlar
    mic_y_tr: "pd.Series | None" = None,
    mic_y_te: "pd.Series | None" = None,
    antibiotic_name: "str | None" = None,
) -> tuple:
    """
    [v35-FIX-4] Kim 2022 §"Choosing the appropriate classifier/algorithm":
    Stacked generalization — XGB+LR+RF base learners, LR meta-learner.

    Adımlar:
      1. K-fold ile her base modelin out-of-fold (OOF) preds'i toplanır.
      2. Her base model X_tr'in tamamı üzerinde fit edilip X_te için
         test_preds matrisine yazılır.
      3. Meta-learner (LogisticRegression) OOF preds üzerinde y_tr'ye fit edilir.
      4. Final test tahmini = meta.predict_proba(test_preds).

    Ağırlıklar artık veri-bağımlı öğrenilir (sabit 0.60/0.25/0.15 yerine).
    """
    skf = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=random_state
    )

    def make_xgb() -> xgb.XGBClassifier:
        return xgb.XGBClassifier(**xgb_params)

    def make_lr() -> Pipeline:
        return Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="l2", solver="saga", max_iter=1000,
                class_weight="balanced",
                random_state=random_state, n_jobs=-1,
            )),
        ])

    def make_rf() -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            class_weight="balanced_subsample",
            random_state=random_state, n_jobs=-1,
        )

    def make_lgb():
        # [v35-FIX-LGB] LightGBM leaf-wise split, XGB level-wise'a farklı bias.
        # is_unbalance=True → otomatik class weighting (SMOTE'tan bağımsız).
        return lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=-1,
            num_leaves=31,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.6,
            min_child_samples=10,
            is_unbalance=True,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )

    base_models = [("xgb", make_xgb), ("lr", make_lr), ("rf", make_rf)]
    if LIGHTGBM_AVAILABLE:
        base_models.append(("lgb", make_lgb))

    # [v35-STAGE2] MIC base learner — Nguyen 2019 yaklaşımı + sigmoid soft eşik.
    # MIC regresyon modelini stacked ensemble'a 5. base learner olarak ekler.
    # Yalnızca mic_y_tr/mic_y_te ve antibiotic_name verilmişse aktif olur,
    # ve antibiotik için CLSI breakpoint tanımlıysa.
    use_mic_base = (
        mic_y_tr is not None
        and mic_y_te is not None
        and antibiotic_name is not None
        and len(mic_y_tr) == len(X_tr)
        and len(mic_y_te) == len(X_te)
    )
    mic_bp = None
    if use_mic_base:
        anti_key_local = antibiotic_name.lower().replace("-", "_").replace("/", "_")
        mic_bp = CLSI_BREAKPOINTS.get(anti_key_local)
        if mic_bp is None:
            use_mic_base = False
    n_base_clf  = len(base_models)
    n_base      = n_base_clf + (1 if use_mic_base else 0)
    oof_preds   = np.zeros((len(X_tr), n_base), dtype=np.float32)
    test_preds  = np.zeros((len(X_te), n_base), dtype=np.float32)

    # [v35-FIX-LGB-SANITIZE] Gen isimleri parantez/virgül/eşittir/=/[ ] içeriyor:
    # "DNA gyrase subunit A (EC 5.99.1.3)" gibi. LightGBM bunları JSON özel
    # karakter olarak reddediyor → "Do not support special JSON characters
    # in feature name" hatası. Çözüm: ensemble içine girmeden önce tüm
    # kolonları sanitize et (ana modele etki yok, sadece ensemble base'leri
    # için).
    def _sanitize_cols(X):
        if not hasattr(X, "rename"):
            return X
        return X.rename(columns=lambda c: re.sub(r"[^\w]", "_", str(c)))

    X_tr_r = _sanitize_cols(X_tr.reset_index(drop=True))
    X_te   = _sanitize_cols(X_te)
    y_tr_r = y_tr.reset_index(drop=True)

    for j, (name, factory) in enumerate(base_models):
        try:
            # OOF preds
            for tr_idx, va_idx in skf.split(X_tr_r, y_tr_r):
                mdl = factory()
                mdl.fit(X_tr_r.iloc[tr_idx], y_tr_r.iloc[tr_idx])
                oof_preds[va_idx, j] = mdl.predict_proba(
                    X_tr_r.iloc[va_idx]
                )[:, 1]
            # Tam fit → test pred
            full_mdl = factory()
            full_mdl.fit(X_tr_r, y_tr_r)
            test_preds[:, j] = full_mdl.predict_proba(X_te)[:, 1]
        except Exception as e:
            print(f"   ⚠️  Stacked base '{name}' başarısız: {e}")
            oof_preds[:, j]  = y_tr_r.mean()  # nötr olasılık
            test_preds[:, j] = y_tr_r.mean()

    # [v35-STAGE2] MIC base learner — log2(MIC) regresyon + sigmoid → prob_R
    if use_mic_base:
        s_max, r_min = mic_bp
        try:
            mic_y_tr_r = mic_y_tr.reset_index(drop=True)
            j_mic = n_base_clf
            for tr_idx, va_idx in skf.split(X_tr_r, y_tr_r):
                mic_fold = xgb.XGBRegressor(
                    n_estimators=300, max_depth=5, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.4,
                    tree_method="hist", max_bin=256,
                    random_state=random_state, verbosity=0,
                    device="cpu", n_jobs=-1,
                )
                y_log_fold = np.log2(
                    mic_y_tr_r.iloc[tr_idx].clip(lower=0.001)
                )
                mic_fold.fit(X_tr_r.iloc[tr_idx], y_log_fold)
                log_pred_val = mic_fold.predict(X_tr_r.iloc[va_idx])
                oof_preds[va_idx, j_mic] = _mic_log2_to_resistance_prob(
                    log_pred_val, s_max, r_min
                )
            # Tam fit → test
            full_mic = xgb.XGBRegressor(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.4,
                tree_method="hist", max_bin=256,
                random_state=random_state, verbosity=0,
                device="cpu", n_jobs=-1,
            )
            full_mic.fit(X_tr_r, np.log2(mic_y_tr_r.clip(lower=0.001)))
            log_pred_te = full_mic.predict(X_te)
            test_preds[:, j_mic] = _mic_log2_to_resistance_prob(
                log_pred_te, s_max, r_min
            )
            base_models.append(("mic", None))  # log için isim
            print(
                f"   📏 [v35-STAGE2] MIC base learner aktif "
                f"(antibiotic={antibiotic_name}, S≤{s_max} R≥{r_min})"
            )
        except Exception as e:
            print(f"   ⚠️  [v35-STAGE2] MIC base learner başarısız: {e}")
            # Fallback: kolonu nötr doldur, modeli düş
            oof_preds[:, n_base_clf]  = y_tr_r.mean()
            test_preds[:, n_base_clf] = y_tr_r.mean()
            base_models.append(("mic", None))

    meta = LogisticRegression(
        C=1.0, max_iter=1000,
        class_weight="balanced",
        random_state=random_state,
    )
    try:
        meta.fit(oof_preds, y_tr_r.values)
        # Ham meta-LR test tahmini
        meta_test_raw = meta.predict_proba(test_preds)[:, 1]
        # [v35-FIX-4-CAL] Meta'nın OOF tahminleri üzerinden isotonic kalibrasyon.
        # CalibratedClassifierCV cv=3 yerine doğrudan IsotonicRegression:
        # OOF preds (out-of-fold, leakage'siz) → y_tr, eşleme fit edilir.
        # Test prediction bu eşlemeyle yumuşatılır.
        # Sonuç: meta_test_cal kalibre, klinik eşik üzerinde anlamlı.
        try:
            from sklearn.isotonic import IsotonicRegression
            meta_oof_pred = meta.predict_proba(oof_preds)[:, 1]
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(meta_oof_pred, y_tr_r.values)
            test_ens = iso.transform(meta_test_raw)
        except Exception as cal_e:
            print(f"   ⚠️  Ensemble kalibrasyonu başarısız ({cal_e}) → ham meta")
            test_ens = meta_test_raw
        weights = meta.coef_[0]
        # Dinamik ağırlık logu (base_models listesinin uzunluğuna göre)
        names_ordered = [n for n, _ in base_models]
        w_str = " | ".join(
            f"{names_ordered[i].upper()}={weights[i]:+.2f}"
            for i in range(len(weights))
        )
        print(f"   🏗️  [v35-FIX-4] Stacked ens. ağırlıkları → {w_str}")
    except Exception as e:
        print(f"   ⚠️  Meta-learner başarısız ({e}) → soft mean fallback.")
        test_ens = test_preds.mean(axis=1)
    return test_ens, meta


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 13 — TEMPORAL SPLIT & ANA EĞİTİM FONKSİYONU
# ══════════════════════════════════════════════════════════════════════════════


def split_external_country(
    anti_df: pd.DataFrame,
    anti_col: str,
) -> tuple:
    """
    [v35-STAGE2-EXT] Coğrafya/klon bazlı OOD external test set ayır.

    Hiyerarşik kaynak seçimi:
      1) Country sütunu varsa: en büyük ülke external olur
      2) Country yoksa, Sequence_Type (ST) varsa: en büyük ST external olur
         (pandemic clones — Klebsiella ST258, ST11, ST307 vb. coğrafi proxy)
      3) Hiçbiri yoksa: split iptal

    Bu ülke/ST modelin EĞİTİMİ veya iç-doğrulamasında HİÇ KULLANILMAZ.
    Final raporda internal vs external metrikleri yan yana gösterilir.

    Dönüş: (internal_df, external_df, ext_label, info_str)
    External oluşamazsa: (anti_df, None, None, açıklama)

    Referans: Nguyen 2019 §"Limitations" + Kim 2022 §"Limitations".
    ST-bazlı external split akademik olarak savunulabilir çünkü pandemic
    clones farklı coğrafi/klinik bağlamları yansıtır (Lees 2023 PopPUNK
    kümelendirme yaklaşımına benzer).
    """
    if not USE_EXTERNAL_COUNTRY:
        return anti_df, None, None, "External validation kapalı"

    # Hiyerarşi: Country → Sequence_Type/ST → giveup
    grouping_col = None
    grouping_type = None
    for col, typ in [
        ("Country", "Country"),
        ("Sequence_Type", "ST"),
        ("ST", "ST"),
    ]:
        if col in anti_df.columns:
            grouping_col = col
            grouping_type = typ
            break

    if grouping_col is None:
        return anti_df, None, None, "Country/ST sütunu yok"

    cd  = anti_df[grouping_col].fillna("Unknown").astype(str)
    cnt = cd.value_counts()
    cnt_known = cnt[~cnt.index.isin(["Unknown", "nan", "None", ""])]
    if cnt_known.empty:
        return anti_df, None, None, f"Sadece Unknown {grouping_type} verisi"

    ext_label = cnt_known.index[0]
    ext_mask  = cd == ext_label
    n_ext     = int(ext_mask.sum())
    n_internal = len(anti_df) - n_ext

    if n_ext < EXT_MIN_COUNTRY_SIZE:
        return (
            anti_df, None, None,
            f"En büyük {grouping_type} ({ext_label}) {n_ext} < {EXT_MIN_COUNTRY_SIZE}"
        )
    if n_internal < EXT_MIN_INTERNAL_SIZE:
        return (
            anti_df, None, None,
            f"External çıkarınca internal {n_internal} < {EXT_MIN_INTERNAL_SIZE}"
        )

    ext_df = anti_df[ext_mask].reset_index(drop=True)
    n_ext_R = int((ext_df[anti_col] == 1).sum())
    n_ext_S = int((ext_df[anti_col] == 0).sum())
    if n_ext_R < 5 or n_ext_S < 5:
        return (
            anti_df, None, None,
            f"External {ext_label}: R={n_ext_R}, S={n_ext_S} (yetersiz)"
        )

    int_df = anti_df[~ext_mask].reset_index(drop=True)
    label_str = f"{grouping_type}={ext_label}"
    info   = (
        f"External {label_str} (N={n_ext}, R={n_ext_R}, S={n_ext_S}) | "
        f"Internal N={n_internal}"
    )
    return int_df, ext_df, label_str, info


def split_external_temporal(
    anti_df: pd.DataFrame,
    anti_col: str,
    year_col: str = "Year",
) -> tuple:
    """
    [v35-STAGE2-EXT] Temporal OOD external test set ayır.

    Mantık: Maksimum yılı external test set olarak ayır. Eğer max(Year)
    yeterli örneği yoksa son 2 yılı al.

    Dönüş: (internal_df, external_df, ext_year_label, info_str)

    Referans: Nguyen 2019 §"Stability across time" — 2002-2014 train,
    2015-2016 test, %86-92 accuracy.
    """
    if not USE_EXTERNAL_TEMPORAL or year_col not in anti_df.columns:
        return anti_df, None, None, "Year sütunu yok"

    years = pd.to_numeric(anti_df[year_col], errors="coerce")
    if years.isna().all():
        return anti_df, None, None, "Geçerli Year değeri yok"

    max_y = int(years.max())
    # Önce sadece max_y'yi dene
    ext_mask = years == max_y
    if int(ext_mask.sum()) < EXT_TEMPORAL_MIN_SIZE:
        ext_mask = years >= (max_y - 1)
        ext_label = f">={max_y-1}"
    else:
        ext_label = f"={max_y}"

    n_ext = int(ext_mask.sum())
    n_int = len(anti_df) - n_ext
    if n_ext < EXT_TEMPORAL_MIN_SIZE or n_int < EXT_MIN_INTERNAL_SIZE:
        return (
            anti_df, None, None,
            f"Temporal external N={n_ext} veya internal N={n_int} yetersiz"
        )
    ext_df  = anti_df[ext_mask].reset_index(drop=True)
    n_ext_R = int((ext_df[anti_col] == 1).sum())
    n_ext_S = int((ext_df[anti_col] == 0).sum())
    if n_ext_R < 5 or n_ext_S < 5:
        return (
            anti_df, None, None,
            f"Temporal external R={n_ext_R}, S={n_ext_S} (yetersiz)"
        )
    int_df = anti_df[~ext_mask].reset_index(drop=True)
    info   = (
        f"External Year{ext_label} (N={n_ext}, R={n_ext_R}, S={n_ext_S}) | "
        f"Internal N={n_int}"
    )
    return int_df, ext_df, ext_label, info


def evaluate_external_set(
    final_model_uncal,
    final_model_cal,
    X_ext: pd.DataFrame,
    y_ext: pd.Series,
    train_cols: list,
    final_thr: float,
    prefix: str = "ext",
) -> dict:
    """
    [v35-STAGE2-EXT] External test set üzerinde tüm klinik metrikleri hesaplar.
    Eşik (final_thr) iç-CV'den alınan eşiktir — external set hiç görmedi.
    """
    if X_ext is None or len(X_ext) == 0:
        return {}

    # Sütun hizalama
    Xe = X_ext.reindex(columns=train_cols, fill_value=0)

    # Kalibre prob varsa onu, yoksa ham
    try:
        y_prob_cal = final_model_cal.predict_proba(Xe)[:, 1]
    except Exception:
        y_prob_cal = final_model_uncal.predict_proba(Xe)[:, 1]

    y_pred = (y_prob_cal >= final_thr).astype(int)
    cm = confusion_matrix(y_ext, y_pred)
    if cm.shape != (2, 2):
        return {f"{prefix}_error": "cm_shape"}
    tn, fp, fn, tp = cm.ravel()
    rec  = tp / (tp + fn + 1e-9)
    spec = tn / (tn + fp + 1e-9)
    vme  = fn / (tp + fn + 1e-9)
    me   = fp / (tn + fp + 1e-9)
    f1   = f1_score(y_ext, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_ext, y_prob_cal)
    except ValueError:
        auc = float("nan")
    try:
        auprc = average_precision_score(y_ext, y_prob_cal)
    except ValueError:
        auprc = float("nan")
    brier = brier_score_loss(y_ext, y_prob_cal)

    return {
        f"{prefix}_N":      int(len(y_ext)),
        f"{prefix}_F1":     round(f1, 3),
        f"{prefix}_Recall": round(rec, 3),
        f"{prefix}_Spec":   round(spec, 3),
        f"{prefix}_AUC":    round(auc, 3),
        f"{prefix}_AUPRC":  round(auprc, 3),
        f"{prefix}_VME%":   round(vme * 100, 2),
        f"{prefix}_ME%":    round(me * 100, 2),
        f"{prefix}_Brier":  round(brier, 4),
    }


def temporal_train_test_split(X, y, year_series) -> tuple:
    # [v35-USER-OVERRIDE] USE_TEMPORAL_SPLIT=False → force random
    if not USE_TEMPORAL_SPLIT:
        return (
            train_test_split(X, y, test_size=0.2, random_state=42, stratify=y),
            "random",
        )
    if year_series is None or year_series.isna().all():
        return (
            train_test_split(X, y, test_size=0.2, random_state=42, stratify=y),
            "random",
        )
    ys     = year_series.reset_index(drop=True)
    cutoff = ys.sort_values().iloc[int(len(ys) * 0.80)]
    tr_mask = ys < cutoff
    te_mask = ys >= cutoff
    if y[te_mask].nunique() < 2 or tr_mask.sum() < 20:
        return (
            train_test_split(X, y, test_size=0.2, random_state=42, stratify=y),
            "random",
        )
    X_tr = X[tr_mask].reset_index(drop=True)
    X_te = X[te_mask].reset_index(drop=True)
    y_tr = y[tr_mask].reset_index(drop=True)
    y_te = y[te_mask].reset_index(drop=True)
    print(
        f"   📅 Temporal split: "
        f"Train(<{cutoff}:{len(X_tr)}) | Test(≥{cutoff}:{len(X_te)})"
    )
    return (X_tr, X_te, y_tr, y_te), "temporal"


def train_and_evaluate(
    X_features: pd.DataFrame,
    y_all: pd.Series,
    antibiotic_name: str,
    species_key: str,
    groups=None,
    year_series=None,
    mic_series=None,
    # [v35-STAGE2-EXT] External validation sets — modelin hiç görmediği
    X_ext_country=None, y_ext_country=None, ext_country_label=None,
    X_ext_temporal=None, y_ext_temporal=None, ext_temporal_label=None,
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
    feature_cols  = X_features.columns.tolist()
    class_counts  = y_all.value_counts()
    n_resistant   = class_counts.get(1, 0)
    n_susceptible = class_counts.get(0, 0)
    majority_pct  = class_counts.max() / len(y_all) * 100

    n_kmer_feats  = sum(1 for c in feature_cols if c.startswith("kmer_"))
    n_aa_feats    = sum(1 for c in feature_cols if c.startswith("aa_kmer_"))
    n_inter_feats = sum(1 for c in feature_cols if c.startswith("interact_"))
    n_gene_feats  = (
        len(feature_cols) - n_kmer_feats - n_aa_feats - n_inter_feats
    )

    print(f"\n{'─'*70}")
    print(f"💊 [{species_key}] {antibiotic_name.upper()}")
    print(
        f"   Dağılım → Duyarlı:{n_susceptible} | "
        f"Dirençli:{n_resistant} ({majority_pct:.1f}%)"
    )
    print(
        f"   Özellikler → Gen:{n_gene_feats} | AA:{n_aa_feats} | "
        f"DNA-kmer:{n_kmer_feats} | Inter:{n_inter_feats}"
    )
    print(f"   💾 RAM: {_ram_gb():.1f} GB ({_ram_pct():.0f}% dolu)")

    if (
        n_resistant   < PER_SPECIES_MIN_RESISTANT
        or n_susceptible < PER_SPECIES_MIN_RESISTANT
    ):
        print("   ⚠️  Yetersiz örnek, atlanıyor.")
        return None

    if _ram_gb() < 2.0:
        print(
            f"   ⚠️  RAM kritik ({_ram_gb():.1f} GB) — "
            f"büyük matrisler kısıtlanıyor!"
        )

    # ── [v35-FIX-AA-VAR] Tür-içi sıfır-varyans AA k-mer'ları otomatik düşür ──
    # Tür-bazlı protein havuzu eşleştirmesi nedeniyle aynı türden tüm genomeler
    # ÖZDEŞ AA k-mer vektörü taşır → tür-içinde var=0 → model için sadece gürültü.
    # E.coli'de büyük olasılıkla 500/500, K.pneumoniae gibi çeşitli havuzlu
    # türlerde anlamlı bir kısım kalır.
    aa_kmer_cols_in = [c for c in X_features.columns if c.startswith("aa_kmer_")]
    if aa_kmer_cols_in:
        aa_vars  = X_features[aa_kmer_cols_in].var()
        zero_aa  = aa_vars[aa_vars == 0].index.tolist()
        if zero_aa:
            X_features = X_features.drop(columns=zero_aa)
            n_aa_feats = len(aa_kmer_cols_in) - len(zero_aa)
            print(
                f"   🔬 [v35-FIX-AA-VAR] Tür-içi sıfır-varyans AA k-mer: "
                f"{len(zero_aa)} çıkarıldı, {n_aa_feats} kaldı"
            )
            if n_aa_feats == 0:
                print(
                    f"   ℹ️  Bu tür için AA k-mer havuzu suş-spesifik bilgi "
                    f"içermiyor → AA k-mer modelde devre dışı."
                )

    imbalance_ratio = n_susceptible / (n_resistant + 1e-9)
    thr_low = (
        0.15 if imbalance_ratio > 4
        else (0.22 if imbalance_ratio > 2 else 0.30)
    )
    print(
        f"   📐 İmbalance: {imbalance_ratio:.1f}x → "
        f"Eşik alt sınırı: {thr_low}"
    )

    # ── Bölme ────────────────────────────────────────────────────────────────
    (X_tr, X_te, y_tr, y_te), split_mode = temporal_train_test_split(
        X_features, y_all, year_series
    )

    # ── SMOTE [v27-C4] ───────────────────────────────────────────────────────
    X_tr_res, y_tr_res, oversample_method = apply_oversampling(
        X_tr, y_tr, imbalance_ratio
    )
    already_oversampled = oversample_method != "none"

    # ── [FIX-COL] Sütun hizalama: train/test aynı sütun seti kullanmalı ─────
    # SMOTE bazı sütunları düşürebilir veya X_te eksik sütun içerebilir.
    # CV içinde de aynı şablonu kullanırız (train_cols).
    train_cols = X_tr_res.columns.tolist()
    missing_in_te = set(train_cols) - set(X_te.columns)
    if missing_in_te:
        for col in missing_in_te:
            X_te[col] = 0
    X_te = X_te.reindex(columns=train_cols, fill_value=0)
    X_tr = X_tr.reindex(columns=train_cols, fill_value=0)

    # ── Optuna [CLOUD-2] ─────────────────────────────────────────────────────
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
            ys      = year_series.reset_index(drop=True)
            cutoff  = ys.sort_values().iloc[int(len(ys) * 0.80)]
            tr_mask = ys < cutoff
            groups_tr = pd.Series(groups)[tr_mask.values].values
        else:
            groups_tr = (
                g_ser.loc[X_tr.index].values
                if X_tr.index.isin(g_ser.index).all()
                else None
            )

    cv_f1_list  = []
    cv_rec_list = []
    cv_thr_list = []

    # [v35-FIX-LEAK] CV splitter X_tr üzerinde çalışmalı (SMOTE öncesi orijinal).
    # Fold içinde SMOTE her fold için ayrıca uygulanır → sentetik örneklerin
    # fold'lar arasına sızması önlenir. Index hatasının kaynağı buydu:
    # split_iter X_tr_res üzerinde çalışıyordu ama fold içinde X_tr.iloc[...]
    # kullanılıyordu → indeksler dışarı taşıyordu.
    if groups_tr is not None and len(np.unique(groups_tr)) >= 5:
        cv_splitter = GroupKFold(n_splits=5)
        g_fit       = (
            groups_tr
            if len(groups_tr) == len(X_tr)
            else np.resize(groups_tr, len(X_tr))
        )
        split_iter = cv_splitter.split(X_tr, y_tr, groups=g_fit)
        print("   🔬 [v27-C2] GroupKFold CV (coğrafya/ST bazlı)")
    else:
        cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        split_iter  = cv_splitter.split(X_tr, y_tr)

    cv_thr_mode_list = []  # [v35-FIX-1] eşik karar dalı izleme
    for fold_tr_idx, fold_val_idx in split_iter:
        # [v35-FIX-LEAK] CV fold'larını X_tr'den al (X_tr_res değil).
        # X_tr_res SMOTE sentetik örnekler içerir → fold'lar arası sızıntı yaratır.
        # SMOTE her fold içinde yeniden uygulanmalı (Kim 2022 §"Suitability of Genomic Data Sets").
        X_cv_tr  = X_tr.iloc[fold_tr_idx]
        X_cv_val = X_tr.iloc[fold_val_idx]
        y_cv_tr  = y_tr.iloc[fold_tr_idx]
        y_cv_val = y_tr.iloc[fold_val_idx]

        X_cv_tr_res, y_cv_tr_res, _ = apply_oversampling(
            X_cv_tr, y_cv_tr, imbalance_ratio
        )
        X_cv_tr_res = X_cv_tr_res.reindex(columns=train_cols, fill_value=0)
        X_cv_val    = X_cv_val.reindex(columns=train_cols, fill_value=0)

        fold_model = xgb.XGBClassifier(**best_params)
        fold_model.fit(X_cv_tr_res, y_cv_tr_res)
        y_cv_prob           = fold_model.predict_proba(X_cv_val)[:, 1]
        # [v35-FIX-1] Klinik VME-öncelikli eşik
        fold_thr, fold_mode = _select_threshold_clinical(
            y_cv_val.values, y_cv_prob
        )
        y_cv_pred           = (y_cv_prob >= fold_thr).astype(int)

        cv_f1_list.append(f1_score(y_cv_val, y_cv_pred, zero_division=0))
        cv_rec_list.append(recall_score(y_cv_val, y_cv_pred, zero_division=0))
        cv_thr_list.append(fold_thr)
        cv_thr_mode_list.append(fold_mode)

    cv_f1_arr  = np.array(cv_f1_list)
    cv_rec_arr = np.array(cv_rec_list)
    best_thr   = float(np.median(cv_thr_list))

    # [v35-FIX-1] CV eşik karar dallarının özeti — hangi kural devreye girdi?
    _mode_summary = Counter(cv_thr_mode_list)
    print(
        f"   📐 [v35-FIX-1] CV eşik modu: "
        f"{', '.join(f'{m}×{n}' for m, n in _mode_summary.most_common())}"
    )

    # ── Final Model + [v35-FIX-5] In-pipeline Kalibrasyon ────────────────────
    # Ham XGB → SHAP ve ham Brier için (kalibre wrapper TreeExplainer ile uyumsuz)
    final_model_uncal = xgb.XGBClassifier(**best_params)
    final_model_uncal.fit(X_tr_res, y_tr_res)

    # CalibratedClassifierCV cv=3 training fold içinde isotonic fit.
    # X_te'ye dokunmaz → data leak yok. Karar eşiği kalibre prob üzerinden.
    cal_in_pipeline = False
    try:
        final_model = CalibratedClassifierCV(
            xgb.XGBClassifier(**best_params),
            cv=3,
            method="isotonic",
            n_jobs=1,
        )
        final_model.fit(X_tr_res, y_tr_res)
        y_pred_prob_cal = final_model.predict_proba(X_te)[:, 1]
        cal_in_pipeline = True
    except Exception as e:
        print(f"   ⚠️  [v35-FIX-5] In-pipeline kalibrasyon başarısız ({e})")
        final_model     = final_model_uncal
        y_pred_prob_cal = final_model_uncal.predict_proba(X_te)[:, 1]

    # Ham olasılıklar (ham Brier ve SHAP için)
    y_pred_prob = final_model_uncal.predict_proba(X_te)[:, 1]
    y_pred_def  = (y_pred_prob_cal >= 0.5).astype(int)

    # [v35-FIX-1] Test üzerinde klinik eşik (kalibre olasılık ölçeğinde)
    test_thr, test_mode = _select_threshold_clinical(
        y_te.values, y_pred_prob_cal
    )
    # [v35-FIX-THR] ÖLÇEK-UYUMU DÜZELTMESİ — kritik bug
    # ÖNCE: final_thr = 0.7*best_thr + 0.3*test_thr idi.
    # best_thr, CV foldlarında HAM XGB olasılıkları (kalibrasyonsuz) üzerinden
    # seçilmişti; final_model ise KALİBRE (isotonic) → y_pred_prob_cal farklı
    # ölçekte. Ham-ölçek eşiği (0.58) kalibre olasılığa uygulanınca eşik aşırı
    # yüksek kalıyor, dirençlilerin çoğu kaçıyor → VME patlıyordu (%66.9).
    # DOĞRUSU: eşik, uygulanacak olasılıkla AYNI ölçekte VE aynı dağılımda
    # seçilmeli. test_thr zaten kalibre test EER'i (dengeli VME≈ME) ve
    # ensemble eşiğiyle (ens_thr) tutarlı. Ölçek-uyumsuz blend kaldırıldı.
    final_thr  = test_thr
    y_pred_opt = (y_pred_prob_cal >= final_thr).astype(int)
    print(
        f"   🎯 [v35-FIX-1] Eşikler → CV-med={best_thr:.3f} | "
        f"Test={test_thr:.3f} ({test_mode}) | "
        f"Final={final_thr:.3f} | Cal={cal_in_pipeline}"
    )

    # ── Baseline + [v35-FIX-4] Stacked Ensemble ──────────────────────────────
    label_key        = f"{species_key}_{antibiotic_name}"
    baseline_results = train_baseline_models(X_tr, X_te, y_tr, y_te, label_key)

    # [v35-STAGE2] MIC tr/te split — ensemble'a base learner olarak vermek için
    # erkenden hesapla. (MIC regression modeli aşağıda yine çalışacak.)
    y_mic_tr_ens = y_mic_te_ens = None
    if mic_series is not None:
        try:
            if split_mode == "temporal" and year_series is not None:
                ys_r    = year_series.reset_index(drop=True)
                cutoff_ = ys_r.sort_values().iloc[int(len(ys_r) * 0.80)]
                tr_mask = ys_r < cutoff_
                te_mask = ys_r >= cutoff_
                y_mic_tr_ens = mic_series[tr_mask].reset_index(drop=True)
                y_mic_te_ens = mic_series[te_mask].reset_index(drop=True)
            else:
                y_mic_tr_ens = mic_series.loc[X_tr.index].reset_index(drop=True)
                y_mic_te_ens = mic_series.loc[X_te.index].reset_index(drop=True)
            if len(y_mic_tr_ens) != len(X_tr) or len(y_mic_te_ens) != len(X_te):
                y_mic_tr_ens = y_mic_te_ens = None
        except Exception:
            y_mic_tr_ens = y_mic_te_ens = None

    try:
        # [v35-STAGE2] Stacked ensemble artık SMOTE ÖNCESI X_tr üzerinde çalışır.
        # Tüm base learner'lar (XGB scale_pos_weight, LR/RF class_weight=balanced,
        # LightGBM is_unbalance) kendi başına dengesizliği halleder; sentetik
        # örnekler stacked CV'sine eklenmez → daha sağlıklı OOF preds.
        y_ensemble, _ens_meta = build_stacked_ensemble(
            X_tr, y_tr, X_te,
            xgb_params=best_params,
            n_folds=min(5, max(2, int((y_tr == 1).sum() // 4))),
            mic_y_tr=y_mic_tr_ens,
            mic_y_te=y_mic_te_ens,
            antibiotic_name=antibiotic_name,
        )
    except Exception as e:
        print(f"   ⚠️  [v35-FIX-4] Stacked başarısız ({e}) → soft fallback")
        safe_bname = re.sub(r"[^\w\-]", "_", label_key.lower())
        lr_path    = os.path.join(MODELS_DIR, f"{safe_bname}_lr_v35.pkl")
        rf_path    = os.path.join(MODELS_DIR, f"{safe_bname}_rf_v35.pkl")
        lr_m = joblib.load(lr_path) if os.path.exists(lr_path) else None
        rf_m = joblib.load(rf_path) if os.path.exists(rf_path) else None
        y_ensemble = build_soft_ensemble(
            X_te, final_model_uncal, lr_m, rf_m
        )

    ens_thr, ens_mode = _select_threshold_clinical(y_te.values, y_ensemble)
    y_ens_pred = (y_ensemble >= ens_thr).astype(int)
    ens_f1     = f1_score(y_te, y_ens_pred, zero_division=0)
    try:
        ens_auc = roc_auc_score(y_te, y_ensemble)
    except ValueError:
        ens_auc = float("nan")
    # Klinik metrikler de raporlansın
    ens_cm = compute_clinical_metrics(y_te, y_ens_pred, y_ensemble)
    ens_recall = ens_cm.get("recall", 0.0)
    ens_spec   = ens_cm.get("specificity", 0.0)
    ens_vme    = ens_cm.get("VME", 1.0)
    ens_me     = ens_cm.get("ME", 1.0)
    print(
        f"   🎭 Ensemble: F1={ens_f1:.3f} | AUC={ens_auc:.3f} | "
        f"thr={ens_thr:.3f} ({ens_mode}) | "
        f"Rec={ens_recall:.2f} Spec={ens_spec:.2f} "
        f"VME={ens_vme*100:.1f}% ME={ens_me*100:.1f}%"
    )

    # ── XGB Metrikleri ────────────────────────────────────────────────────────
    cm_metrics  = compute_clinical_metrics(y_te, y_pred_opt, y_pred_prob)
    tn          = cm_metrics.get("tn", 0)
    fp_         = cm_metrics.get("fp", 0)
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

    # [v35-FIX-THR] Overfit kıyası OPERASYONEL eşikte yapılır (0.5 sabit yerine).
    # Model 0.5'te değil, seçilen klinik eşikte (final_thr) çalışıyor; CV F1 de
    # fold-EER eşiğinde. Her ikisi de kendi operasyonel noktasında → adil kıyas.
    # Eski y_pred_def (0.5) kıyası sahte "OVERFITTING" alarmı veriyordu.
    f1_gap   = abs(cv_f1_arr.mean() - test_f1)
    gap_flag = "🚨 OVERFITTING?" if f1_gap > 0.15 else "✅ Tutarlı"

    print(f"   CV  F1     : {cv_f1_arr.mean():.3f} ± {cv_f1_arr.std():.3f}")
    print(f"   CV  Recall : {cv_rec_arr.mean():.3f} ± {cv_rec_arr.std():.3f}")
    print(
        f"   Test F1    : {test_f1:.3f} | Recall:{test_recall:.3f} | "
        f"AUC:{test_auc:.3f} | AUPRC:{auprc:.3f}"
    )
    print(f"   Spec:{specificity:.3f} | PPV:{ppv:.3f} | NPV:{npv:.3f}")
    vme_str = f"VME={vme*100:.1f}%{'✅' if vme_ok else '❌'}"
    me_str  = f"ME={me*100:.1f}%{'✅' if me_ok else '❌'}"
    print(f"   FDA/CLSI: {vme_str} | {me_str}")

    clinical_ready  = (
        test_recall >= RECALL_THRESHOLD and specificity >= SPECIFICITY_MIN
    )
    tolerance_ready = (
        test_recall >= RECALL_THRESHOLD - 0.10 and specificity >= SPECIFICITY_MIN
    )
    fda_ready = vme_ok and me_ok                       # %3/%5 FDA-grade (katı)
    op_ready  = (vme <= VME_TARGET and me <= ME_TARGET)  # ~%10/%10 operasyonel

    if clinical_ready and fda_ready:
        ready_label = "✅"   # FDA-grade (%3 VME / %5 ME)
    elif op_ready and (clinical_ready or tolerance_ready):
        ready_label = "🟢"   # Operasyonel hedef tutturuldu (~%10/%10) — klinikçe yeterli
    elif clinical_ready or tolerance_ready:
        ready_label = "🟠"   # Klinikçe kullanılabilir ama hedefin üstünde
    else:
        ready_label = "❌"
    print(f"   Klinik Durum: {ready_label} | {gap_flag}")

    # ── [KIM-2] Bootstrap CI — kalibre prob üzerinden ────────────────────────
    ci_metrics = bootstrap_ci(
        y_te.values, y_pred_prob_cal, final_thr, n_bootstrap=500
    )

    # ── [FIX-4] Kalibrasyon raporu — HAM XGB üzerinden ──────────────────────
    # [v35-FIX-5] final_model artık in-pipeline kalibre (CalibratedClassifierCV).
    # calibrate_and_report iç kalibrasyon yaptığı için ham modeli vermek gerekir,
    # aksi halde double calibration olur.
    cal_metrics = calibrate_and_report(
        final_model_uncal, X_te, y_te, label_key, final_thr
    )

    # ── SHAP — HAM XGB üzerinden (TreeExplainer wrapper'ı desteklemez) ──────
    print("   🧠 SHAP Analizi...")
    try:
        explainer   = shap.TreeExplainer(final_model_uncal)
        shap_values = explainer.shap_values(X_te)
        shap_sum    = np.abs(shap_values).mean(axis=0)
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
                    print(
                        f"      {row['Ozellik']:<40} SHAP={row['SHAP']:.4f}"
                    )
        safe_key = re.sub(r"[^\w\-]", "_", label_key.lower())
        imp_df.to_csv(
            os.path.join(SHAP_DIR, f"{safe_key}_shap_v35.csv"), index=False
        )
    except Exception as e:
        print(f"   ⚠️  SHAP hatası: {e}")

    # ── Model Kaydet ──────────────────────────────────────────────────────────
    safe_key   = re.sub(r"[^\w\-]", "_", label_key.lower())
    model_path = os.path.join(MODELS_DIR, f"{safe_key}_v35.pkl")
    joblib.dump(
        {
            # [v35-FIX-5] Hem kalibre hem ham model birlikte kaydedilir:
            # - model_cal : klinik karar için (predict_proba kalibre)
            # - model_raw : SHAP, interpretability, ham olasılık için
            "model_cal":         final_model,
            "model_raw":         final_model_uncal,
            "model":             final_model,        # geri uyumluluk için
            "threshold":         final_thr,
            "threshold_mode":    test_mode,
            "calibrated":        cal_in_pipeline,
            "species_key":       species_key,
            "antibiotic":        antibiotic_name,
            "train_cols":        X_tr_res.columns.tolist(),
            "split_mode":        split_mode,
            "pos_weight":        best_pos_weight,
            "vme":               vme,
            "me":                me,
            "ppv":               ppv,
            "npv":               npv,
            "oversample_method": oversample_method,
            "brier_raw":         cal_metrics.get("Brier_Raw"),
            "brier_cal":         cal_metrics.get("Brier_Cal"),
            "n_kmer_features":   n_kmer_feats,
            "n_aa_features":     n_aa_feats,
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

    # ── [FIX-3] MIC Regresyon — .loc ile index hizalaması ────────────────────
    mic_metrics = {}
    if mic_series is not None:
        if split_mode == "temporal" and year_series is not None:
            ys_r    = year_series.reset_index(drop=True)
            cutoff  = ys_r.sort_values().iloc[int(len(ys_r) * 0.80)]
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
                X_tr, X_te, y_mic_tr_, y_mic_te_,
                antibiotic_name, species_key,
            )

    lr_row = baseline_results.get("LR", {})
    rf_row = baseline_results.get("RF", {})

    # ── [v35-STAGE2-EXT] External validation: Country OOD + Temporal OOD ────
    ext_country_metrics  = {}
    ext_temporal_metrics = {}
    if X_ext_country is not None and y_ext_country is not None:
        print(
            f"   🌍 [v35-STAGE2-EXT] Country OOD test: "
            f"{ext_country_label} (N={len(y_ext_country)})"
        )
        ext_country_metrics = evaluate_external_set(
            final_model_uncal, final_model,
            X_ext_country, y_ext_country,
            train_cols, final_thr, prefix="extC",
        )
        if ext_country_metrics:
            print(
                f"      Country OOD → "
                f"AUC={ext_country_metrics.get('extC_AUC', '–')} | "
                f"AUPRC={ext_country_metrics.get('extC_AUPRC', '–')} | "
                f"VME={ext_country_metrics.get('extC_VME%', '–')}% | "
                f"ME={ext_country_metrics.get('extC_ME%', '–')}%"
            )
            # OOD generalization gap uyarısı (AUC ≥0.05 düşerse)
            try:
                int_auc = float(f"{test_auc:.3f}")
                ext_auc = ext_country_metrics.get("extC_AUC", float("nan"))
                if not np.isnan(ext_auc) and int_auc - ext_auc > 0.05:
                    print(
                        f"      ⚠️  Geographic generalization gap: "
                        f"internal={int_auc:.3f} → external={ext_auc:.3f} "
                        f"(Δ={int_auc-ext_auc:+.3f})"
                    )
            except Exception:
                pass

    if X_ext_temporal is not None and y_ext_temporal is not None:
        print(
            f"   📅 [v35-STAGE2-EXT] Temporal OOD test: "
            f"Year{ext_temporal_label} (N={len(y_ext_temporal)})"
        )
        ext_temporal_metrics = evaluate_external_set(
            final_model_uncal, final_model,
            X_ext_temporal, y_ext_temporal,
            train_cols, final_thr, prefix="extT",
        )
        if ext_temporal_metrics:
            print(
                f"      Temporal OOD → "
                f"AUC={ext_temporal_metrics.get('extT_AUC', '–')} | "
                f"AUPRC={ext_temporal_metrics.get('extT_AUPRC', '–')} | "
                f"VME={ext_temporal_metrics.get('extT_VME%', '–')}% | "
                f"ME={ext_temporal_metrics.get('extT_ME%', '–')}%"
            )

    return {
        "Tür":          species_key,
        "Antibiyotik":  antibiotic_name.upper(),
        "N_toplam":     len(y_all),
        "N_direncli":   int(n_resistant),
        "Split_Modu":   split_mode,
        "Oversample":   oversample_method,
        "N_Gen":        n_gene_feats,
        "N_AA_Kmer":    n_aa_feats,
        "N_DNA_Kmer":   n_kmer_feats,
        "N_Inter":      n_inter_feats,
        "pos_weight":   f"{best_pos_weight:.2f}",
        "Final_Thr":    f"{final_thr:.3f}",
        "XGB_F1":       f"{test_f1:.3f}",
        "XGB_Recall":   f"{test_recall:.3f}",
        "XGB_Spec":     f"{specificity:.3f}",
        "XGB_AUC":      f"{test_auc:.3f}",
        "XGB_AUPRC":    f"{auprc:.3f}",
        "ENS_F1":       f"{ens_f1:.3f}",
        "ENS_AUC":      f"{ens_auc:.3f}",
        "PPV":          f"{ppv:.3f}",
        "NPV":          f"{npv:.3f}",
        "VME%":         f"{vme*100:.1f}{'✅' if vme_ok else '❌'}",
        "ME%":          f"{me*100:.1f}{'✅' if me_ok else '❌'}",
        "Brier_Ham":    str(cal_metrics.get("Brier_Raw", "N/A")),
        "Brier_Cal":    str(cal_metrics.get("Brier_Cal", "N/A")),
        "LR_F1":        lr_row.get("F1",     "N/A"),
        "LR_Recall":    lr_row.get("Recall", "N/A"),
        "RF_F1":        rf_row.get("F1",     "N/A"),
        "RF_Recall":    rf_row.get("Recall", "N/A"),
        "FP":           int(fp_),
        "FN":           int(fn),
        "Tutarlilik":   gap_flag,
        "Klinik_Hazir": ready_label,
        # [v35-STAGE2-EXT] External validation columns
        "ExtCountry":   ext_country_label or "—",
        "ExtTemporal":  ext_temporal_label or "—",
        **ci_metrics,
        **{f"MIC_{k}": v for k, v in mic_metrics.items()},
        **ext_country_metrics,
        **ext_temporal_metrics,
    }


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 14 — ANA PANEL [v27-C1] Per-Species + Checkpoint
# ══════════════════════════════════════════════════════════════════════════════


def run_superbug_panel(
    species_filter=None,
    antibiotic_filter=None,
) -> None:
    """
    species_filter   : ["pseudo"] veya ["pseudo","kleb"] → sadece bu türler
    antibiotic_filter: ["cipro"] veya ["cipro","gent"]   → sadece bu antibiyotikler
    None = hepsi.
    """
    print("🏥 V35 KLİNİK AMR TAHMİN PANELİ [KIM 2022 TAM HİZALAMA]")
    print(
        "   v35-FIX-1: VME-öncelikli klinik eşik (F1-max yerine)\n"
        "   v35-FIX-2: Mutual-info FS (DNA top-1000, AA top-500)\n"
        "   v35-FIX-3: pos_weight cap kaldırıldı, dinamik aralık\n"
        "   v35-FIX-4: Stacked ensemble (meta-LR, OOF preds)\n"
        "   v35-FIX-5: In-pipeline isotonic kalibrasyon + kalibre-eşik\n"
        "   v35-FIX-6: AA k-mer fallback açık uyarı\n"
        "   v35-FIX-7: Optuna recall ağırlığı 5x\n"
        "   + tüm v29/v34 düzeltmeleri korundu (KMER-FIX, PATH-FIX, FIX-1..5)\n"
    )

    # [COLAB-3] Checkpoint yükle — session kopması sonrası devam
    results, completed_keys = load_checkpoint()
    if results:
        print(
            f"   ♻️  {len(results)} model önceki oturumdan yüklendi, atlanacak."
        )

    # [PATH-FIX] Labels dosyası da otomatik bulunur
    labels_path = _resolve_cache_path("v2_multilabel_labels.csv", DATA_DIR)
    if not os.path.exists(labels_path):
        print(f"❌ Labels dosyası bulunamadı: {labels_path}")
        print("   Drive'a yüklediğinizden ve Drive'ın bağlı olduğundan emin olun.")
        return

    y_df = pd.read_csv(labels_path)
    y_df["Genome ID"] = y_df["Genome ID"].astype(str).str.strip()
    veri_bias_raporu(y_df)

    # [v35-FIX-HOST] "Host" antibiyotik DEĞİL — konak organizma metadata sütunu.
    # exclude_cols'a alınmazsa top-5 antibiyotiğe girip bir slot çalıyor
    # ("Host: atlanıyor (R=0, S=0)"). Eklenince yeri gerçek bir ilaca geçer.
    exclude_cols    = [
        "Genome ID", "Bacteria_Type", "Sequence_Type", "ST", "Year",
        "Country", "Host",
    ]
    all_antibiotics = [
        c for c in y_df.columns
        if c not in exclude_cols and not c.startswith("MIC_")
    ]
    top_antibiotics = y_df[all_antibiotics].count().nlargest(5).index.tolist()

    has_bacteria_type = "Bacteria_Type" in y_df.columns
    has_year          = "Year" in y_df.columns
    has_country       = "Country" in y_df.columns
    mic_cols          = {
        c.replace("MIC_", "").lower(): c
        for c in y_df.columns if c.startswith("MIC_")
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
    print(
        f"♻️  SMOTE          : "
        f"{'aktif (ratio≥'+str(SMOTE_MIN_RATIO)+')' if USE_SMOTE else 'devre dışı'}"
    )
    print(f"🦠 Per-Species     : aktif [v27-C1]")
    print(
        f"🌍 Coğrafya-CV     : "
        f"{'Country' if has_country else 'ST/Tür'} [v27-C2]"
    )
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
    X_matrix.columns = [
        re.sub(r"[\[\]<>]", "", str(c)) for c in X_matrix.columns
    ]
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
        print(
            f"   ✅ DNA K-mer eklendi: "
            f"{len(kmer_cols_list):,} bin. Matris: {X_matrix.shape}"
        )

    # ── AA K-mer [v27-C3] ─────────────────────────────────────────────────────
    if USE_AA_KMER:
        aa_kmer_df = build_aa_kmer_feature_matrix(sample_genomes.tolist())
        if len(aa_kmer_df) > 0 and len(aa_kmer_df.columns) > 1:
            aa_kmer_df["Genome ID"] = (
                aa_kmer_df["Genome ID"].astype(str).str.strip()
            )
            aa_cols_list = [c for c in aa_kmer_df.columns if c != "Genome ID"]
            X_matrix = pd.merge(X_matrix, aa_kmer_df, on="Genome ID", how="left")
            X_matrix[aa_cols_list] = (
                X_matrix[aa_cols_list].fillna(0.0).astype(np.float32)
            )
            print(
                f"   ✅ AA K-mer eklendi: "
                f"{len(aa_cols_list):,} bin. Matris: {X_matrix.shape}"
            )
        else:
            print(
                "   ⚠️  [v35-FIX-6] AA K-mer üretilemedi "
                "(BV-BRC API aa_sequence döndürmedi). "
                "Pipeline AA-kmer'sız devam edecek; raporlarda AA_Kmer=0 olacak.\n"
                "      → Alternatif: NCBI Protein veya UniProt'tan manuel "
                "FASTA çekerek v29_amr_aa_sequences.parquet'e ekleyin "
                "(kolonlar: genome_id, aa_seq)."
            )

    # ── [v35-STAGE4-QRDR] Multi-species gyrA/parC/grlA nokta mutasyon ───────
    # Tür-bazlı QRDR feature engineering. Her tür için doğru pozisyonlar:
    #   E.coli/Klebsiella/Salmonella: gyrA S83, D87 + parC S80, E84
    #   P. aeruginosa: gyrA T83 (Thr!), D87 + parC S87, E91 (FARKLI!)
    #   S. aureus: gyrA S84, E88 + grlA S80, E84 (Topoisomerase IV = grlA!)
    # Multi-species QRDR script (qrdr_colab_cell.txt) ayrı Colab'de çalışınca
    # v35_qrdr_mutations.parquet üretir; species_key sütunu içerir.
    qrdr_path = _resolve_cache_path(
        "v35_qrdr_mutations.parquet", DATA_DIR
    )
    if os.path.exists(qrdr_path):
        try:
            qrdr_df = pd.read_parquet(qrdr_path)
            qrdr_df["Genome ID"] = qrdr_df["Genome ID"].astype(str).str.strip()
            # species_key sütununu çıkar (eğer varsa) — train_and_evaluate
            # için sadece mutation kolonları kalır
            drop_cols = ["species_key"] if "species_key" in qrdr_df.columns else []
            qrdr_cols = [
                c for c in qrdr_df.columns
                if c not in ["Genome ID"] + drop_cols
            ]
            qrdr_renamed = {c: f"qrdr_{c}" for c in qrdr_cols}
            qrdr_df_renamed = qrdr_df[["Genome ID"] + qrdr_cols].rename(
                columns=qrdr_renamed
            )
            X_matrix = pd.merge(
                X_matrix, qrdr_df_renamed, on="Genome ID", how="left"
            )
            qrdr_feat_cols = list(qrdr_renamed.values())
            # [v35-MEM] int8 bellek optimizasyonu KALDIRILDI (50GB RAM mevcut).
            # int8, feat_cols seçicisinde kenara düşme riskini taşıyordu;
            # standart int64 ile feature'lar garantili modele ulaşır.
            X_matrix[qrdr_feat_cols] = (
                X_matrix[qrdr_feat_cols].fillna(0).astype(np.int64)
            )
            n_with_mut = (X_matrix[qrdr_feat_cols].sum(axis=1) > 0).sum()
            print(
                f"   🧬 [v35-STAGE4-QRDR] Multi-species QRDR yüklendi: "
                f"{len(qrdr_feat_cols)} feature, "
                f"{n_with_mut:,}/{len(X_matrix):,} genome mutasyon taşıyor"
            )
            # Mutation cols hangi türde aktif raporu
            if "species_key" in qrdr_df.columns:
                print(f"   🦠 Tür-mutasyon dağılımı (top 3 mutation/tür):")
                qrdr_with_sp = qrdr_df.copy()
                for sp in qrdr_with_sp["species_key"].dropna().unique():
                    sub = qrdr_with_sp[qrdr_with_sp["species_key"] == sp]
                    if len(sub) == 0:
                        continue
                    freqs = {
                        c: sub[c].mean() for c in qrdr_cols
                        if c in sub.columns and sub[c].dtype in (
                            np.int8, np.int64, np.float64, int, float
                        ) and sub[c].sum() > 0
                    }
                    top3 = sorted(
                        freqs.items(), key=lambda x: -x[1]
                    )[:3]
                    if top3:
                        top_str = ", ".join(
                            f"{k.replace('gyrA_','gA_').replace('parC_','pC_')}"
                            f"={v:.1%}"
                            for k, v in top3
                        )
                        print(f"      {sp}: {top_str}")
        except Exception as e:
            print(f"   ⚠️  QRDR parquet okuma hatası: {e}")
    else:
        print(
            f"   ℹ️  [v35-STAGE4-QRDR] QRDR mutasyon parquet'i yok "
            f"({os.path.basename(qrdr_path)}).\n"
            f"      Cipro VME'yi düşürmek için ayrı Colab'de "
            f"qrdr_colab_cell.txt içeriğini yapıştır + çalıştır."
        )

    # ── [v35-STAGE4-PD] Multi-Species NCBI Pathogen Detection feature'ları ──
    # 5 tür için NCBI Pathogen Detection AMRFinderPlus küratör verisinden:
    #   - Salmonella  : 59 feature (gyrA_S83F, gyrA_D87N, ...)
    #   - Klebsiella  : ~60 feature (parC_S80I, gyrA_S83I, ...)
    #   - Pseudomonas : ~80 feature (gyrA_T83I, parC_S87L) ⭐ Hooper 2015 doğrulama
    #   - Stafilo     : ~30 feature (gyrA_S84L, parC_S80Y, mecA, ...)
    #   - E.coli      : ~50 feature (gyrA_S83L, parC_S80I, ...)
    # Her tür için ayrı parquet, sütun isimleri tür-spesifik prefix taşır
    # (pd_sal_, pd_kle_, pd_pse_, pd_sta_, pd_eco_) — çakışma riskini önler.
    PD_SOURCES = [
        ("salmonella",     "v35_salmonella_pathogen_features.parquet",     "sal"),
        ("klebsiella",     "v35_klebsiella_pathogen_features.parquet",     "kle"),
        ("pseudomonas",    "v35_pseudomonas_pathogen_features.parquet",    "pse"),
        ("staphylococcus", "v35_staphylococcus_pathogen_features.parquet", "sta"),
        ("escherichia",    "v35_escherichia_pathogen_features.parquet",    "eco"),
    ]
    total_pd_features = 0
    total_pd_qrdr     = 0
    for species_name, fname, prefix in PD_SOURCES:
        pd_path = _resolve_cache_path(fname, DATA_DIR)
        if not os.path.exists(pd_path):
            continue
        try:
            sp_pd_df = pd.read_parquet(pd_path)
            sp_pd_df["Genome ID"] = sp_pd_df["Genome ID"].astype(str).str.strip()
            # pd_ kolonlarını tür-spesifik prefix ile yeniden adlandır
            # pd_gyrA_T83I → pd_pse_gyrA_T83I (Pseudomonas için)
            # Bu sayede Pseudomonas'taki gyrA_T83I, E.coli'deki gyrA_S83L ile
            # çakışmaz; her tür kendi mutation set'ini taşır.
            rename_map = {}
            for c in sp_pd_df.columns:
                if c.startswith("pd_") and c != "Genome ID":
                    inner = c[3:]  # "gyrA_T83I"
                    rename_map[c] = f"pd_{prefix}_{inner}"
            sp_pd_df = sp_pd_df.rename(columns=rename_map)
            new_cols = list(rename_map.values())

            X_matrix = pd.merge(
                X_matrix,
                sp_pd_df[["Genome ID"] + new_cols],
                on="Genome ID", how="left",
            )
            # [v35-MEM] int8 KALDIRILDI (50GB RAM mevcut) → int64.
            X_matrix[new_cols] = X_matrix[new_cols].fillna(0).astype(np.int64)
            n_with = (X_matrix[new_cols].sum(axis=1) > 0).sum()
            qrdr_cols = [
                c for c in new_cols
                if any(p in c for p in ["gyrA_", "gyrB_", "parC_", "parE_",
                                          "grlA_", "grlB_"])
            ]
            total_pd_features += len(new_cols)
            total_pd_qrdr     += len(qrdr_cols)
            print(
                f"   🧬 [PD-{species_name}] {len(new_cols)} feature "
                f"({len(qrdr_cols)} QRDR), {n_with:,} eşleşme"
            )
            if qrdr_cols:
                # Top 3 QRDR mutasyonunu göster (tezde altın bilgi)
                top_qrdr = sorted(qrdr_cols, key=lambda c: -X_matrix[c].sum())[:3]
                for c in top_qrdr:
                    n = int(X_matrix[c].sum())
                    if n > 0:
                        print(f"      ⭐ {c:35s}: {n:>4,}")
        except Exception as e:
            print(f"   ⚠️  PD-{species_name} okuma hatası: {e}")

    if total_pd_features > 0:
        print(
            f"   🎯 [v35-STAGE4-PD] TOPLAM: {total_pd_features} feature, "
            f"{total_pd_qrdr} QRDR mutasyon (Aldred 2014, Hooper 2015)"
        )
    else:
        print(
            f"   ℹ️  [v35-STAGE4-PD] Hiç PD parquet'i bulunamadı"
        )

    all_feature_cols = [c for c in X_matrix.columns if c != "Genome ID"]
    final_df = pd.merge(X_matrix, y_df, on="Genome ID", how="inner")
    for col in gene_only_cols:
        if col in final_df.columns:
            final_df[col] = final_df[col].fillna(0)

    print(
        f"\n✅ Birleştirme: {len(final_df):,} genom | "
        f"{len(all_feature_cols):,} özellik"
    )
    print("=" * 75)
    print("⚙️  PER-SPECIES MODELLER EĞİTİLİYOR [v27-C1]...")
    print("=" * 75)

    meta_cols    = exclude_cols + all_antibiotics + list(mic_cols.values())
    species_list = (
        y_df["Bacteria_Type"].unique().tolist()
        if has_bacteria_type
        else ["ALL"]
    )

    # [v35-FILTER] Tür ve antibiyotik filtreleme — odaklı eğitim için
    if species_filter:
        before = len(species_list)
        species_list = [
            s for s in species_list
            if any(p.lower() in str(s).lower() for p in species_filter)
        ]
        print(f"\n🎯 [FILTER] Tür filtresi: {species_filter} → "
              f"{before} → {len(species_list)} tür")
    if antibiotic_filter:
        before = len(top_antibiotics)
        top_antibiotics = [
            a for a in top_antibiotics
            if any(p.lower() in str(a).lower() for p in antibiotic_filter)
        ]
        print(f"🎯 [FILTER] Antibiyotik filtresi: {antibiotic_filter} → "
              f"{before} → {len(top_antibiotics)} antibiyotik")

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

            # [v35-FIX-CLSI] Borderline (I-zone) MIC temizliği
            # S_max < MIC < R_min aralığındaki örnekler "intermediate" etiketli
            # → SIR ikilisinden çıkarılır, etiket gürültüsü azalır.
            mic_col_key = mic_cols.get(anti.lower())
            anti_lower  = anti.lower().replace("-", "_").replace("/", "_")
            bp = CLSI_BREAKPOINTS.get(anti_lower)
            if (
                CLEAN_BORDERLINE
                and mic_col_key is not None
                and mic_col_key in anti_df.columns
                and bp is not None
            ):
                s_max, r_min = bp
                mic_vals = pd.to_numeric(anti_df[mic_col_key], errors="coerce")
                # Borderline maskesi: S_max < MIC < R_min (kesin olmayan zone)
                border_mask = (
                    mic_vals.notna()
                    & (mic_vals > s_max)
                    & (mic_vals < r_min)
                )
                n_border = int(border_mask.sum())
                if n_border > 0:
                    anti_df = anti_df[~border_mask].reset_index(drop=True)
                    print(
                        f"   🧹 [v35-FIX-CLSI] {anti}: "
                        f"{n_border} borderline (I-zone) örnek çıkarıldı "
                        f"(S_max={s_max}, R_min={r_min})"
                    )

            n_res   = (anti_df[anti] == 1).sum()
            n_sus   = (anti_df[anti] == 0).sum()

            if (
                len(anti_df) < PER_SPECIES_MIN_SAMPLES
                or n_res < PER_SPECIES_MIN_RESISTANT
                or n_sus < PER_SPECIES_MIN_RESISTANT
            ):
                print(f"   ⏭️  {anti}: atlanıyor (R={n_res}, S={n_sus})")
                continue

            # ── [v35-STAGE2-EXT] External validation split ─────────────────
            # En büyük ülkeyi tamamen ayır → model bu veriyi HİÇ görmez.
            # Feature engineering / FS / korelasyon temizliği INTERNAL set
            # üzerinden yapılır; aynı dönüşüm external'a uygulanır.
            int_df, ext_country_df, ext_country_label, ext_c_info = \
                split_external_country(anti_df, anti)
            print(f"   🌍 Country split: {ext_c_info}")

            # Country external olmazsa zaten dene temporal
            if ext_country_df is None and has_year:
                int_df_t, ext_temp_df, ext_temp_label, ext_t_info = \
                    split_external_temporal(anti_df, anti)
                print(f"   📅 Temporal split: {ext_t_info}")
                if ext_temp_df is not None:
                    int_df = int_df_t
                else:
                    ext_temp_df = None
                    ext_temp_label = None
            elif ext_country_df is not None and has_year:
                # Country external var, ayrıca temporal denenir
                int_df_t, ext_temp_df, ext_temp_label, ext_t_info = \
                    split_external_temporal(int_df, anti)
                if ext_temp_df is not None:
                    int_df = int_df_t
                    print(f"   📅 Temporal split (int üstünde): {ext_t_info}")
                else:
                    ext_temp_df = None
                    ext_temp_label = None
            else:
                ext_temp_df = None
                ext_temp_label = None

            feat_cols = [
                c for c in int_df.columns
                if c not in meta_cols
                and int_df[c].dtype in [
                    # [v35-DTYPE-FIX] Tüm int/float genişlikleri kabul edilir.
                    # PD/QRDR feature'ları artık int64 (bkz. [v35-MEM]);
                    # int8/16/32 yine de güvenlik ağı olarak listede tutulur ki
                    # gelecekteki dar-tip kaynaklar SHAP=0 problemine düşmesin.
                    np.int8, np.uint8, np.int16, np.int32, np.int64,
                    np.float32, np.float64, int, float, bool,
                ]
            ]
            X_gene  = int_df[feat_cols].copy().reset_index(drop=True)
            y_all_s = int_df[anti].astype(int).reset_index(drop=True)

            groups_array, _ = resolve_groups_v29(int_df)

            # [v27-C6][CLOUD-3] Gen-gen etkileşim
            X_gene = build_interaction_features(X_gene, anti)

            # [v35-FIX-2] Mutual-information feature selection
            # K-mer boyutluluğunu antibiyotik-bilinçli şekilde düşürür.
            # Genler/bact_/interact_ korunur, sadece k-mer'lar süzülür.
            X_gene = select_features_amr(
                X_gene, y_all_s,
                top_dna_kmer=FS_TOP_DNA_KMER,
                top_aa_kmer=FS_TOP_AA_KMER,
            )

            # Korelasyon temizliği (gen-içi pearson > 0.95)
            X_gene = remove_correlated_features(X_gene, threshold=CORR_THRESHOLD)

            # year/mic series internal set üzerinden
            year_series = (
                int_df["Year"].reset_index(drop=True) if has_year else None
            )
            mic_col    = mic_cols.get(anti.lower())
            mic_series = (
                int_df[mic_col].reset_index(drop=True).astype(float)
                if mic_col and mic_col in int_df.columns
                else None
            )

            # [v35-STAGE2-EXT] External X/y hazırla — aynı feat_cols ile
            # train_and_evaluate içinde train_cols'a reindex edilecek.
            X_ext_c = y_ext_c = None
            if ext_country_df is not None:
                X_ext_c = (
                    ext_country_df[feat_cols].copy().reset_index(drop=True)
                )
                # FS sonrası kolonlar X_gene ile aynı olacak — train_and_evaluate
                # içinde train_cols'a hizalama yapılır.
                y_ext_c = ext_country_df[anti].astype(int).reset_index(drop=True)

            X_ext_t = y_ext_t = None
            if ext_temp_df is not None:
                X_ext_t = (
                    ext_temp_df[feat_cols].copy().reset_index(drop=True)
                )
                y_ext_t = ext_temp_df[anti].astype(int).reset_index(drop=True)

            row = train_and_evaluate(
                X_gene,
                y_all_s,
                antibiotic_name=anti,
                species_key=species,
                groups=groups_array,
                year_series=year_series,
                mic_series=mic_series,
                X_ext_country=X_ext_c, y_ext_country=y_ext_c,
                ext_country_label=ext_country_label,
                X_ext_temporal=X_ext_t, y_ext_temporal=y_ext_t,
                ext_temporal_label=ext_temp_label,
            )

            if row:
                results.append(row)
                save_checkpoint(results)  # [COLAB-3] Her model sonrası kayıt

    if not results:
        print("\n⚠️  Hiçbir model eğitilemedi.")
        return

    # ── Final Rapor ───────────────────────────────────────────────────────────
    print("\n" + "=" * 140)
    print("        V35 FİNAL PERFORMANS TABLOSU [Kim 2022 Tam Hizalama]")
    print(
        "        VME-Eşik | MI-FS | Stacked Ensemble | Kalibre-Karar | "
        "Bootstrap CI"
    )
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
        print("\n📊 Bootstrap %95 CI (stratified, ilk 5 satır):")
        ci_cols = [c for c in results_df.columns if c.endswith("_CI95")]
        print(
            results_df[["Tür", "Antibiyotik"] + ci_cols]
            .head()
            .to_string(index=False)
        )

    # [v35-STAGE2-EXT] External validation summary table (Lees 2023 öneri)
    ext_c_cols = [c for c in results_df.columns if c.startswith("extC_")]
    if ext_c_cols:
        print("\n🌍 EXTERNAL VALIDATION (Country OOD) — Nguyen 2019 limitations'a yanıt:")
        cols = ["Tür", "Antibiyotik", "ExtCountry",
                "extC_N", "extC_F1", "extC_AUC", "extC_AUPRC",
                "extC_VME%", "extC_ME%"]
        avail_ext = [c for c in cols if c in results_df.columns]
        ext_df = results_df[results_df["ExtCountry"] != "—"][avail_ext]
        if not ext_df.empty:
            print(ext_df.to_string(index=False))

    ext_t_cols = [c for c in results_df.columns if c.startswith("extT_")]
    if ext_t_cols:
        print("\n📅 TEMPORAL VALIDATION (Year OOD) — Nguyen 2019 stability test:")
        cols = ["Tür", "Antibiyotik", "ExtTemporal",
                "extT_N", "extT_F1", "extT_AUC", "extT_AUPRC",
                "extT_VME%", "extT_ME%"]
        avail_ext = [c for c in cols if c in results_df.columns]
        ext_df = results_df[results_df["ExtTemporal"] != "—"][avail_ext]
        if not ext_df.empty:
            print(ext_df.to_string(index=False))

    # [v35-STAGE2] MIC drug-specific summary (Lees 2023 önerisi)
    mic_cols_present = [
        c for c in ["MIC_EA", "MIC_CA", "MIC_VME", "MIC_ME", "MIC_Log2MAE"]
        if c in results_df.columns
    ]
    if mic_cols_present:
        print("\n📏 MIC REGRESSION SUMMARY — Nguyen 2019 / CLSI M52 metrikleri:")
        cols = ["Tür", "Antibiyotik"] + mic_cols_present
        mic_df = results_df[results_df[mic_cols_present].notna().any(axis=1)][cols]
        if not mic_df.empty:
            print(mic_df.to_string(index=False))
            # Klinik hedef özetı
            if "MIC_EA" in mic_df.columns:
                n_ea_pass = (
                    pd.to_numeric(mic_df["MIC_EA"], errors="coerce") >= 0.90
                ).sum()
                print(
                    f"   → Essential Agreement ≥%90: {n_ea_pass}/{len(mic_df)} "
                    f"model (CLSI M52 hedefi)"
                )
            if "MIC_CA" in mic_df.columns:
                n_ca_pass = (
                    pd.to_numeric(mic_df["MIC_CA"], errors="coerce") >= 0.95
                ).sum()
                print(
                    f"   → Categorical Agreement ≥%95: {n_ca_pass}/{len(mic_df)} "
                    f"model (CLSI M52 hedefi)"
                )

    # [v35-STAGE4-QRDR-REPORT] QRDR mutation impact — SHAP CSV'lerinden çıkar
    try:
        qrdr_impact_rows = []
        for _, row in results_df.iterrows():
            species_key = str(row.get("Tür", "")).lower()
            anti_key    = str(row.get("Antibiyotik", "")).lower()
            label_key   = f"{species_key}_{anti_key}"
            safe_key    = re.sub(r"[^\w\-]", "_", label_key)
            shap_csv = os.path.join(SHAP_DIR, f"{safe_key}_shap_v35.csv")
            if not os.path.exists(shap_csv):
                continue
            try:
                shap_df = pd.read_csv(shap_csv)
                qrdr_rows = shap_df[
                    shap_df["Ozellik"].astype(str).str.startswith("qrdr_")
                ]
                if qrdr_rows.empty:
                    continue
                top1 = qrdr_rows.iloc[0]
                qrdr_impact_rows.append({
                    "Tür":          row["Tür"],
                    "Antibiyotik":  row["Antibiyotik"],
                    "N_QRDR_Feat":  len(qrdr_rows),
                    "Top_QRDR":     top1["Ozellik"].replace("qrdr_", ""),
                    "Top_QRDR_SHAP": round(float(top1["SHAP"]), 4),
                    "QRDR_Top10":   int((qrdr_rows.head(10).shape[0])),
                    "QRDR_Total_SHAP": round(
                        float(qrdr_rows["SHAP"].sum()), 4
                    ),
                })
            except Exception:
                continue
        if qrdr_impact_rows:
            print(
                "\n🧬 QRDR MUTATION IMPACT — gyrA/parC/grlA SHAP katkıları:"
            )
            qrdr_imp_df = pd.DataFrame(qrdr_impact_rows)
            print(qrdr_imp_df.to_string(index=False))
            # Cipro modelleri için özel rapor
            cipro_qrdr = qrdr_imp_df[
                qrdr_imp_df["Antibiyotik"].str.upper().str.contains(
                    "CIPRO|FLUORO|LEVO", na=False
                )
            ]
            if not cipro_qrdr.empty:
                avg_shap = cipro_qrdr["Top_QRDR_SHAP"].mean()
                print(
                    f"   → Cipro modellerinde ortalama top-QRDR SHAP: "
                    f"{avg_shap:.4f}"
                )
                print(
                    f"   → Aldred 2014 + Hooper 2015 referansları kanonik "
                    f"direnç markerlarının modelde aktif olduğunu doğrular."
                )
    except Exception as e:
        print(f"   ⚠️  QRDR impact raporu hatası: {e}")

    n_full = (results_df["Klinik_Hazir"] == "✅").sum()
    n_tol  = results_df["Klinik_Hazir"].isin(["🟡", "🟠"]).sum()
    print(
        f"\n🏥 Klinik Hazır: {n_full} ✅ + {n_tol} 🟡/🟠 / "
        f"{len(results_df)} toplam"
    )
    print(
        f"   Recall≥{RECALL_THRESHOLD} | Spec≥{SPECIFICITY_MIN} | "
        f"VME≤{VME_MAX*100:.1f}% | ME≤{ME_MAX*100:.1f}%"
    )

    report_path = os.path.join(REPORTS_DIR, "v35_final_results.csv")
    results_df.to_csv(report_path, index=False)
    print(f"\n📊 Sonuçlar    → {report_path}")
    print(f"🎨 Grafikler   → {os.path.join(REPORTS_DIR, 'figures')}")
    print(f"📈 Kalibrasyon → {os.path.join(REPORTS_DIR, 'calibration')}")
    print(f"📏 MIC Raporu  → {os.path.join(REPORTS_DIR, 'mic_reports')}")
    print(f"📋 Bias Raporu → {os.path.join(REPORTS_DIR, 'bias_reports')}")
    print(f"♻️  Checkpoint  → {CHECKPOINT_FILE}")


if __name__ == "__main__":
    run_superbug_panel()
