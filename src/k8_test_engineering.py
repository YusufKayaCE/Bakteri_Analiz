import pandas as pd
import numpy as np
import os
import requests
from collections import Counter
import itertools
import concurrent.futures

# Ayarlar
INPUT_LABELS = "../data/processed/v2_multilabel_labels.csv"
OUTPUT_FEATURES = "../data/processed/k8_X_features_fast.csv"
K = 8  
BATCH_SIZE = 20 # Paketleri biraz büyüttük çünkü hızlandık
MAX_TEST_GENOME = 1000 
WORKERS = 10 # SİHİR BURADA: Aynı anda 10 işçi internetten veri çekecek!

print(f"K={K} için Sabit Sözlük oluşturuluyor... (65.536 Sütun)")
bases = ['A', 'C', 'G', 'T']
ALL_KMERS = [''.join(p) for p in itertools.product(bases, repeat=K)]
COLUMNS = ['Genome ID'] + ALL_KMERS

def get_kmer_counts(sequence, k=K):
    kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    raw_counts = Counter(kmers)
    return {kmer: raw_counts.get(kmer, 0) for kmer in ALL_KMERS}

def fetch_and_process(genome_id):
    """Her bir işçinin (Thread) tek başına yapacağı görev"""
    url = f"https://www.bv-brc.org/api/genome_sequence/?eq(genome_id,{genome_id})&select(sequence)&http_accept=application/json"
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if data:
                seq = "".join([d['sequence'] for d in data])
                counts = get_kmer_counts(seq)
                counts['Genome ID'] = genome_id
                return counts
    except:
        pass # Sunucu o anlık reddederse sessizce geç
    return None

def run_fast_engineering():
    df_labels = pd.read_csv(INPUT_LABELS)
    amp_df = df_labels.dropna(subset=['ampicillin'])
    genome_ids = amp_df['Genome ID'].unique()[:MAX_TEST_GENOME]
    total_genomes = len(genome_ids)
    
    print(f"\n🚀 TURBO MOD AKTİF: {total_genomes} genom {WORKERS} işçi ile eşzamanlı indirilecek.")
    
    start_index = 0
    if os.path.exists(OUTPUT_FEATURES):
        existing_df = pd.read_csv(OUTPUT_FEATURES, usecols=['Genome ID'])
        start_index = len(existing_df)
        print(f"Devam ediliyor... Kaldığı yer: {start_index}")

    for i in range(start_index, total_genomes, BATCH_SIZE):
        batch_ids = genome_ids[i:i+BATCH_SIZE]
        batch_data = []
        
        print(f"İndiriliyor: {i} - {i+len(batch_ids)} arası...")
        
        # --- ÇOKLU İŞ PARÇACIĞI HAVUZU (THREAD POOL) ---
        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
            # İşçileri sahaya sal ve sonuçları bekle
            results = list(executor.map(fetch_and_process, batch_ids))
            
            for res in results:
                if res is not None:
                    batch_data.append(res)
                    
        if batch_data:
            batch_df = pd.DataFrame(batch_data, columns=COLUMNS).fillna(0)
            
            # RAM patlamasın diye veriyi küçültüyoruz
            for col in ALL_KMERS:
                batch_df[col] = batch_df[col].astype('float32')
                
            if not os.path.exists(OUTPUT_FEATURES):
                batch_df.to_csv(OUTPUT_FEATURES, index=False)
            else:
                batch_df.to_csv(OUTPUT_FEATURES, mode='a', header=False, index=False)
        
        print(f"--- %{(i+len(batch_ids))/total_genomes*100:.2f} tamamlandı ---")

if __name__ == "__main__":
    run_fast_engineering()