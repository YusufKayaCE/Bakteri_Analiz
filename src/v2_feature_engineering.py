import pandas as pd
import numpy as np
import os
import requests
from collections import Counter
import itertools
import time

# Ayarlar
INPUT_LABELS = "../data/processed/v2_multilabel_labels.csv"
OUTPUT_FEATURES = "../data/processed/v2_X_features.csv"
K = 6  
BATCH_SIZE = 100 

# SABİT SÖZLÜK OLUŞTURMA (Sütun kaymasını kesin olarak engeller)
print("Sabit K-mer sözlüğü oluşturuluyor...")
bases = ['A', 'C', 'G', 'T']
# Sadece temiz 4096 kombinasyonu baştan yaratıyoruz
ALL_KMERS = [''.join(p) for p in itertools.product(bases, repeat=K)]
# Tablomuzun sütunları sonsuza dek bu sırayla sabit kalacak:
COLUMNS = ['Genome ID'] + ALL_KMERS

def get_kmer_counts(sequence, k):
    kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    raw_counts = Counter(kmers)
    
    # Sadece bizim 4096 listemizde olan K-mer'leri al (N harfli olan kirli verileri filtrele)
    clean_counts = {kmer: raw_counts.get(kmer, 0) for kmer in ALL_KMERS}
    return clean_counts

def fetch_genome_sequence(genome_id):
    url = f"https://www.bv-brc.org/api/genome_sequence/?eq(genome_id,{genome_id})&select(sequence)&http_accept=application/json"
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if data:
                return "".join([d['sequence'] for d in data])
    except:
        pass
    return None

def run_v2_feature_engineering():
    df_labels = pd.read_csv(INPUT_LABELS)
    genome_ids = df_labels['Genome ID'].unique()
    total_genomes = len(genome_ids)
    
    print(f"\nV2.0 Başlıyor: Toplam {total_genomes} genom işlenecek.")
    
    start_index = 0
    if os.path.exists(OUTPUT_FEATURES):
        existing_df = pd.read_csv(OUTPUT_FEATURES, usecols=['Genome ID'])
        start_index = len(existing_df)
        print(f"Devam ediliyor... Kaldığı yer: {start_index}")

    for i in range(start_index, total_genomes, BATCH_SIZE):
        batch_ids = genome_ids[i:i+BATCH_SIZE]
        batch_data = []
        
        print(f"İşleniyor: {i} - {i+len(batch_ids)} arası...")
        
        for g_id in batch_ids:
            seq = fetch_genome_sequence(g_id)
            if seq:
                counts = get_kmer_counts(seq, K)
                counts['Genome ID'] = g_id
                batch_data.append(counts)
            time.sleep(0.1) 
            
        if batch_data:
            # Sütunları sabit listemizle zorluyoruz. Artık kayma ihtimali SIFIR.
            batch_df = pd.DataFrame(batch_data, columns=COLUMNS).fillna(0)
            
            if not os.path.exists(OUTPUT_FEATURES):
                batch_df.to_csv(OUTPUT_FEATURES, index=False)
            else:
                batch_df.to_csv(OUTPUT_FEATURES, mode='a', header=False, index=False)
        
        print(f"--- %{(i+len(batch_ids))/total_genomes*100:.2f} tamamlandı ---")

if __name__ == "__main__":
    run_v2_feature_engineering()