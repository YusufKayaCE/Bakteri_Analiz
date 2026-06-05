import pandas as pd
import requests
import os
import time
from collections import Counter
from itertools import product
import urllib3

# SSL uyarılarını terminalde gizlemek için:
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def generate_kmer_features(k: int = 6):
    """
    Olası tüm k-mer kombinasyonlarını üretir.
    Zaman Karmaşıklığı: O(4^k)
    """
    bases = ['A', 'C', 'G', 'T']
    kmers = [''.join(p) for p in product(bases, repeat=k)]
    return kmers

def fetch_genome_with_retry(genome_id: str, kmers_list: list, k: int = 6, max_retries: int = 3) -> dict:
    """
    API bağlantı kopmalarına karşı 'Retry' mantığı ile güçlendirilmiş k-mer çıkarma fonksiyonu.
    """
    url = f"https://patricbrc.org/api/genome_sequence/?eq(genome_id,{genome_id})&select(sequence)&limit(10000)"
    headers = {"accept": "application/json"}
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=45, verify=False)
            
            if response.status_code == 200:
                data = response.json()
                if not data:
                    print(f"  -> [UYARI] {genome_id} API'de bulundu ama 'sequence' verisi BOŞ.")
                    return None
                    
                # K-mer'lerin 0 çıkmasını engelleyen hayat kurtarıcı .upper() fonksiyonumuz
                full_sequence = "".join([item['sequence'] for item in data if 'sequence' in item]).upper()
                
                if not full_sequence:
                    print(f"  -> [UYARI] {genome_id} içinde geçerli bir dizi bulunamadı.")
                    return None
                
                # K-mer sayımı
                kmer_counts = Counter(full_sequence[i:i+k] for i in range(len(full_sequence) - k + 1))
                feature_dict = {kmer: kmer_counts.get(kmer, 0) for kmer in kmers_list}
                feature_dict['Genome ID'] = genome_id
                
                return feature_dict
                
            elif response.status_code == 429:
                wait_time = (attempt + 1) * 3
                print(f"  -> [HATA 429] API sınırı aşıldı. {wait_time} saniye bekleniyor...")
                time.sleep(wait_time)
                
            else:
                print(f"  -> [HATA {response.status_code}] Sunucu hatası.")
                time.sleep(2)
                
        except requests.exceptions.Timeout:
            print(f"  -> [ZAMAN AŞIMI] API çok yavaş yanıt veriyor.")
            time.sleep(3)
        except requests.exceptions.ConnectionError:
            print(f"  -> [BAĞLANTI KOPTU] İnternet veya sunucu kaynaklı hata.")
            time.sleep(3)
            
    print(f"  -> [BAŞARISIZ] {genome_id} için {max_retries} deneme yapıldı ama veri alınamadı.")
    return None

def build_feature_matrix(labels_path: str, output_path: str, k: int = 6, limit: int = None):
    """
    Etiketlenmiş verideki Genome ID'leri okur, API'den k-mer özelliklerini çeker ve matrisi oluşturur.
    """
    print(f"Özellik Çıkarımı başlatıldı... (K={k})")
    
    df_labels = pd.read_csv(labels_path)
    
    # -------- İŞTE XGBOOST'U ÇÖKERTEN HATAYI ÇÖZEN KISIM --------
    # Veri setini baştan aşağı rastgele karıştırıyoruz (Shuffle).
    # Böylece ilk 200 verinin içine hem Dirençli (1) hem Duyarlı (0) bakteriler eşit dağılacak.
    df_labels = df_labels.sample(frac=1, random_state=42).reset_index(drop=True)
    # -------------------------------------------------------------
    
    genome_ids = df_labels['Genome ID'].unique()
    
    if limit:
        genome_ids = genome_ids[:limit]
        print(f"!!! TEST MODU AKTİF: Sadece {limit} genom işlenecek !!!")
        
    kmers_list = generate_kmer_features(k)
    X_data = []
    
    for i, g_id in enumerate(genome_ids):
        print(f"İşleniyor [{i+1}/{len(genome_ids)}]: {g_id}")
        features = fetch_genome_with_retry(g_id, kmers_list, k, max_retries=3)
        
        if features:
            X_data.append(features)
        
        # API'yi boğmamak için bekleme
        time.sleep(0.5)
            
    if not X_data:
        print("Hiç veri çekilemedi. API bağlantınızı kontrol edin.")
        return
        
    df_features = pd.DataFrame(X_data)
    
    # Bellek Optimizasyonu
    for col in df_features.columns:
        if col != 'Genome ID':
            df_features[col] = pd.to_numeric(df_features[col], downcast='unsigned')
            
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_features.to_csv(output_path, index=False)
    
    print(f"\nİşlem tamam! Özellik Matrisi (X) kaydedildi: {output_path}")
    print(f"Oluşan Veri Setinin Boyutu: {df_features.shape}")

if __name__ == "__main__":
    LABELS_FILE = "../data/processed/y_labels.csv"
    FEATURES_OUTPUT = "../data/processed/X_features.csv"
    
    # Hızlıca sonucu görmek için limit=200 yaptık.
    build_feature_matrix(LABELS_FILE, FEATURES_OUTPUT, k=6, limit=None)