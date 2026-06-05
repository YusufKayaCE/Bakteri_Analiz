import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import requests
import warnings
import time
import re  # XGBoost'un sevmediği karakterleri temizlemek için eklendi
warnings.filterwarnings('ignore')

# Ayarlar
LABELS_FILE = "../data/processed/v2_multilabel_labels.csv"
MAX_GENOMES = 500 # API'yi yormamak için şimdilik 500 bakteri ile test edeceğiz.
BATCH_SIZE = 20   # Sunucuyu boğmamak için 50'den 20'ye düşürdük

def fetch_amr_genes_from_bvbrc(genome_ids):
    """Bütün özel genleri ham olarak çeker, Direnç Geni ayıklamasını Python'da yapar!"""
    print("🌐 BV-BRC Veritabanına Balyozla Giriliyor (Sabırlı Çekim)...")
    
    all_genes = []
    total = len(genome_ids)
    
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json"
    }
    
    for i in range(0, total, BATCH_SIZE):
        batch = genome_ids[i:i+BATCH_SIZE]
        id_str = ",".join(batch)
        
        url = "https://www.bv-brc.org/api/sp_gene/"
        
        # SİHİRLİ VURUŞ: eq(property...) kısmını SİLDİK! Bütün özel genleri (limit 25000) çekiyoruz!
        payload = f"in(genome_id,({id_str}))&select(genome_id,property,gene,product)&limit(25000)"
        
        try:
            response = requests.post(url, headers=headers, data=payload, timeout=120)
            if response.status_code == 200:
                data = response.json()
                
                # FİLTREYİ SUNUCU DEĞİL, BİZİM PYTHON KODUMUZ YAPIYOR!
                for item in data:
                    prop = str(item.get('property', '')).lower()
                    
                    if 'resist' in prop or 'antimicrobial' in prop or 'antibiotic' in prop:
                        gene_name = item.get('gene', '')
                        if not gene_name:
                            gene_name = item.get('product', '')
                            
                        if gene_name:
                            all_genes.append({
                                'Genome ID': item['genome_id'],
                                'AMR_Gene': gene_name
                            })
                            
            print(f"Çekilen Bakteri: {min(i+len(batch), total)} / {total} | ⚔️ Yakalanan Gen Silahı: {len(all_genes)}")
            time.sleep(2) 
            
        except Exception as e:
            print(f"Hata oluştu (Atlanıyor, mola veriliyor): {e}")
            time.sleep(5)
            
    return pd.DataFrame(all_genes)

def train_paper_method():
    print("🧬 V11.0 MAKALE METODOLOJİSİ (GEN VARLIĞI / YOKLUĞU) BAŞLIYOR...")
    
    # 1. Etiketleri (Labels) Yükle
    y_df = pd.read_csv(LABELS_FILE)
    anti = 'ampicillin'
    clean_labels = y_df.dropna(subset=[anti])
    
    # Test için rastgele 500 bakteri seçelim
    sample_genomes = clean_labels['Genome ID'].astype(str).unique()[:MAX_GENOMES]
    clean_labels['Genome ID'] = clean_labels['Genome ID'].astype(str)
    y_target = clean_labels[clean_labels['Genome ID'].isin(sample_genomes)][['Genome ID', anti]]
    
    # 2. Makale Yöntemi: Gerçek AMR Genlerini Çek
    genes_df = fetch_amr_genes_from_bvbrc(sample_genomes)
    
    if len(genes_df) == 0:
        print("❌ BV-BRC API'den hiçbir direnç geni çekilemedi. İşlem iptal edildi.")
        return
        
    print(f"\nToplam {len(genes_df)} adet Biyolojik Direnç Geni (Silah) bulundu!")
    
    # 3. Gen Matrisi Oluşturma (Presence / Absence Matrix)
    genes_df['Değer'] = 1
    X_matrix = genes_df.pivot_table(index='Genome ID', columns='AMR_Gene', values='Değer', aggfunc='max', fill_value=0).reset_index()
    
    print(f"Yapay Zekaya {X_matrix.shape[1] - 1} farklı GERÇEK direnç geni sunulacak.")
    
    # 4. Verileri Birleştir
    final_df = pd.merge(X_matrix, y_target, on="Genome ID", how="inner")
    
    if len(final_df) == 0:
        print("❌ Genleri çekilen bakteriler ile etiketli bakteriler eşleşmedi!")
        return
        
    X_data = final_df.drop(columns=['Genome ID', anti])
    
    # =========================================================================
    # SİHİRLİ TEMİZLİK: XGBoost'un sevmediği [, ], <, > karakterlerini siliyoruz
    # =========================================================================
    X_data.columns = [re.sub(r'[\[\]<>]', '', str(col)) for col in X_data.columns]
    
    y_data = final_df[anti]
    
    # 5. Modeli Eğit 
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        eval_metric='logloss'
    )
    
    print("\n🤖 Yapay Zeka GERÇEK Biyolojik Silahları İnceleyerek Eğitiliyor...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print("\n" + "="*50)
    print(f"🏆 MAKALE BİREBİR REPLİKASYON SONUCU ({anti.upper()})")
    print("="*50)
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"✅ Doğru Bildiği Duyarlılar (0) : {cm[0][0]}")
    print(f"🔥 DOĞRU BİLDİĞİ DİRENÇLİLER (1): {cm[1][1]} <--- GÖZÜMÜZ BURADA!")
    print(f"❌ Hata Sayısı                    : {cm[0][1] + cm[1][0]}")

if __name__ == "__main__":
    train_paper_method()