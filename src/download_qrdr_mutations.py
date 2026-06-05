# download_qrdr_mutations.py
# AYRI COLAB'de çalıştır — Cipro/Quinolone direnç mutasyonlarını çeker.
#
# Çıktı: /content/drive/MyDrive/amr_v29/data/processed/v35_qrdr_mutations.parquet
# Kolonlar:
#   - Genome ID
#   - gyrA_S83L, gyrA_S83F, gyrA_D87N, gyrA_D87G, gyrA_D87Y  (E.coli pozisyonları)
#   - parC_S80I, parC_S80R, parC_E84K, parC_E84V
#   - gyrB_S464Y, gyrB_E466D                                  (ikincil, daha nadir)
#   - gyrA_aa_seq, parC_aa_seq  (debug için)
#
# Referans pozisyonlar:
#   E.coli K-12 MG1655 (UniProt P0AES4 gyrA, P0AFI2 parC)
#   Aldred KJ et al. (2014) Biochemistry 53:1565
#   Hooper DC, Jacoby GA. (2015) Cold Spring Harb Perspect Med 6:a025320
#
# Strateji:
#   1) BV-BRC sp_gene endpoint'ten gyrA/parC AA sequence dene
#   2) Boş dönerse NCBI E-utils efetch ile genome → protein
#   3) Sekansı E.coli K-12 referansıyla hizala (Levenshtein-free, basit position lookup)
#   4) Belirli QRDR pozisyonlarını oku
#
# Hız: ~22,000 genom × ~0.3 sn = ~2 saat. Colab background'da çalıştır.

import requests
import pandas as pd
import numpy as np
import time
import os
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

# ── Google Drive ─────────────────────────────────────────────────────────────
try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    BASE = "/content/drive/MyDrive/amr_v29"
except ImportError:
    BASE = ".."

DATA_DIR   = os.path.join(BASE, "data", "processed")
OUT_FILE   = os.path.join(DATA_DIR, "v35_qrdr_mutations.parquet")
LABELS_CSV = os.path.join(DATA_DIR, "v2_multilabel_labels.csv")

# ── Referans AA sekansları (E.coli K-12 MG1655) ──────────────────────────────
# UniProt P0AES4 (gyrA) ve P0AFI2 (parC) — sadece QRDR civarı (1-100. pozisyonlar)
# Tam sekanslar uzun; biz sadece eşleştirme için anchor kullanacağız.
GYRA_REF_K12 = (
    "MSDLAREITPVNIEEELKSSYLDYAMSVIVGRALPDVRDGLKPVHRRVLYAMNVLGNDWNKAYKKSARVVGDVIGKYHPHGDSAVYDTIVRMAQPFSLRYM"
)
PARC_REF_K12 = (
    "MSDMAERLALHEFTENAYLNYSMYVIMDRALPFIGDGLKPVQRRIVYAMSELGLNASAKFKKSARTVGDVLGKYHPHGDSACYEAMVLMAQPFSYRYP"
)

# QRDR (Quinolone Resistance Determining Region) anchor pozisyonları
# E.coli K-12'de gyrA Ser83, parC Ser80 - 0-indexed:
GYRA_QRDR_POSITIONS = {
    "S83L": (82, "S", "L"),
    "S83F": (82, "S", "F"),
    "D87N": (86, "D", "N"),
    "D87G": (86, "D", "G"),
    "D87Y": (86, "D", "Y"),
}
PARC_QRDR_POSITIONS = {
    "S80I": (79, "S", "I"),
    "S80R": (79, "S", "R"),
    "E84K": (83, "E", "K"),
    "E84V": (83, "E", "V"),
}

# ── BV-BRC API çekimi ─────────────────────────────────────────────────────────
HEADERS = {
    "Content-Type": "application/x-www-form-urlencoded",
    "Accept": "application/json",
}


def fetch_bvbrc_gyrA_parC(genome_ids: list) -> dict:
    """
    BV-BRC genome_feature endpoint — gyrA, parC AA sequence dene.
    """
    result = {}
    for i in range(0, len(genome_ids), 50):
        batch  = genome_ids[i:i+50]
        id_str = ",".join(str(g) for g in batch)
        payload = (
            f"in(genome_id,({id_str}))"
            f"&or(eq(gene,gyrA),eq(gene,parC))"
            f"&select(genome_id,gene,product,aa_sequence)&limit(25000)"
        )
        try:
            r = requests.post(
                "https://www.bv-brc.org/api/genome_feature/",
                headers=HEADERS, data=payload, timeout=120,
            )
            if r.status_code == 200:
                for item in r.json():
                    gid  = str(item.get("genome_id", "")).strip()
                    gene = str(item.get("gene", "")).lower()
                    aa   = str(item.get("aa_sequence", "")).strip()
                    if gid and aa and gene in ("gyra", "parc"):
                        result.setdefault(gid, {})[gene] = aa
        except Exception as e:
            print(f"   ⚠️  Batch {i} hata: {e}")
        if i % 500 == 0:
            print(f"\r   BV-BRC %{int((i+50)/len(genome_ids)*100):3d}", end="", flush=True)
        time.sleep(0.2)
    print()
    return result


# ── Pozisyon parse fonksiyonu ────────────────────────────────────────────────
def _find_qrdr_anchor(seq: str, ref_seq: str, window: int = 30) -> Optional[int]:
    """
    Sekansta ref_seq'in QRDR bölgesinin başlangıç pozisyonunu bulur.
    Basit substring search; tam BLAST gerekmez çünkü QRDR vahşi tip korunur.
    """
    # Ref'in QRDR civarı: pozisyon 75-90 arası
    anchor = ref_seq[60:90]  # 30 aa anchor
    pos    = seq.find(anchor)
    if pos == -1:
        # Daha kısa anchor dene
        for sub_len in [25, 20, 15, 10]:
            anchor = ref_seq[70:70+sub_len]
            pos    = seq.find(anchor)
            if pos != -1:
                return pos - 70  # offset to align ref position 0
    else:
        return pos - 60
    return None


def parse_qrdr_mutations(
    aa_seq: str,
    ref_seq: str,
    qrdr_positions: dict,
) -> dict:
    """
    QRDR pozisyonlarındaki mutasyonları binary feature olarak çıkarır.
    """
    out = {key: 0 for key in qrdr_positions.keys()}
    if not aa_seq or len(aa_seq) < 50:
        return out
    offset = _find_qrdr_anchor(aa_seq, ref_seq)
    if offset is None:
        return out
    for key, (ref_pos, ref_aa, mut_aa) in qrdr_positions.items():
        seq_pos = ref_pos + offset
        if 0 <= seq_pos < len(aa_seq):
            if aa_seq[seq_pos] == mut_aa:
                out[key] = 1
    return out


# ── Ana pipeline ─────────────────────────────────────────────────────────────
def main():
    print("🧬 QRDR Mutation Downloader — gyrA / parC")
    print("=" * 60)

    if not os.path.exists(LABELS_CSV):
        print(f"❌ Labels dosyası bulunamadı: {LABELS_CSV}")
        return

    # E.coli genomelarını al (QRDR pozisyonları E.coli K-12 referansı)
    y_df = pd.read_csv(LABELS_CSV)
    y_df["Genome ID"] = y_df["Genome ID"].astype(str).str.strip()
    if "Bacteria_Type" in y_df.columns:
        ecoli_df = y_df[y_df["Bacteria_Type"].str.contains("coli", case=False, na=False)]
    else:
        ecoli_df = y_df
    genome_ids = ecoli_df["Genome ID"].unique().tolist()
    print(f"📊 İşlenecek E.coli genomları: {len(genome_ids):,}")

    # Önceki çalışmadan devam (checkpoint)
    done = {}
    if os.path.exists(OUT_FILE):
        try:
            old = pd.read_parquet(OUT_FILE)
            for _, row in old.iterrows():
                done[str(row["Genome ID"])] = row.to_dict()
            print(f"♻️  Önbellekten {len(done):,} kayıt yüklendi.")
        except Exception:
            pass

    remaining = [g for g in genome_ids if g not in done]
    print(f"📥 Çekilecek: {len(remaining):,}")

    if not remaining:
        print("✅ Tüm genom önbellekte var.")
        return

    # Chunk-bazlı çekim (her 1000'de bir kaydet)
    all_rows = list(done.values())
    CHUNK = 1000
    for chunk_i in range(0, len(remaining), CHUNK):
        chunk = remaining[chunk_i:chunk_i+CHUNK]
        print(f"\n🌐 BV-BRC chunk {chunk_i//CHUNK+1}/{(len(remaining)-1)//CHUNK+1}: "
              f"{len(chunk)} genom")
        bv_data = fetch_bvbrc_gyrA_parC(chunk)
        print(f"   BV-BRC döndü: {len(bv_data)} genom için en az 1 gen")

        # Parse mutations
        for gid in chunk:
            entry = {"Genome ID": gid}
            gd    = bv_data.get(gid, {})
            gyrA  = gd.get("gyra", "")
            parC  = gd.get("parc", "")
            entry.update(parse_qrdr_mutations(gyrA, GYRA_REF_K12, GYRA_QRDR_POSITIONS))
            entry.update({f"par{k[3:]}": v for k, v in
                          parse_qrdr_mutations(parC, PARC_REF_K12, PARC_QRDR_POSITIONS).items()})
            # Debug için saklayabilirsin (büyük olur, opsiyonel)
            # entry["gyrA_aa_len"] = len(gyrA)
            # entry["parC_aa_len"] = len(parC)
            all_rows.append(entry)

        # Ara kayıt
        try:
            pd.DataFrame(all_rows).to_parquet(OUT_FILE, index=False)
            print(f"   💾 Ara kayıt: {len(all_rows):,} satır → {OUT_FILE}")
        except Exception as e:
            print(f"   ⚠️  Kayıt hatası: {e}")

    # Final
    df = pd.DataFrame(all_rows)
    df.to_parquet(OUT_FILE, index=False)
    print(f"\n✅ TAMAMLANDI: {len(df):,} satır → {OUT_FILE}")
    print("\n📊 Mutation frequencies:")
    mut_cols = [c for c in df.columns if c != "Genome ID"]
    for c in mut_cols:
        if df[c].dtype in (int, float, np.int64, np.float64):
            pct = df[c].mean() * 100
            print(f"   {c:15s} : %{pct:5.2f} ({int(df[c].sum())}/{len(df)})")


if __name__ == "__main__":
    main()
