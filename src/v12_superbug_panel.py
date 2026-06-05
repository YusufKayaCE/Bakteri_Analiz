import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score
import requests
import warnings
import time
import re
import os
warnings.filterwarnings('ignore')

# Ayarlar
LABELS_FILE = "../data/processed/v2_multilabel_labels.csv"
CACHE_FILE = "../data/processed/v12_amr_genes_cache.csv" 
MAX_GENOMES = 1500 
BATCH_SIZE = 50

def fetch_amr_genes_from_bvbrc(genome_ids):
    """Veriyi önce bilgisayarda (Cache) arar, bulamazsa API'den çeker."""
    if os.path.exists(CACHE_FILE):
        print(f"📦 YEREL ÖNBELLEK BULUNDU! '{CACHE_FILE}' dosyasından okunuyor...")
        df = pd.read_csv(CACHE_FILE)
        # TİP DÜZELTME: Okunan CSV'deki Genome ID'leri kesinlikle string yap
        df['Genome ID'] = df['Genome ID'].astype(str)
        return df
    
    print(f"🌐 BV-BRC Veritabanından {len(genome_ids)} Bakteri İçin Gen Çekiliyor...")
    all_genes = []
    total = len(genome_ids)
    headers = {"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"}
    
    for i in range(0, total, BATCH_SIZE):
        batch = genome_ids[i:i+BATCH_SIZE]
        id_str = ",".join(batch)
        url = "https://www.bv-brc.org/api/sp_gene/"
        payload = f"in(genome_id,({id_str}))&select(genome_id,property,gene,product)&limit(25000)"
        
        try:
            response = requests.post(url, headers=headers, data=payload, timeout=120)
            if response.status_code == 200:
                data = response.json()
                for item in data:
                    prop = str(item.get('property', '')).lower()
                    if 'resist' in prop or 'antimicrobial' in prop or 'antibiotic' in prop:
                        gene_name = item.get('gene', '') or item.get('product', '')
                        if gene_name:
                            all_genes.append({'Genome ID': str(item['genome_id']), 'AMR_Gene': gene_name})
            print(f"İlerleme: %{int(min(i+len(batch), total)/total*100)} | Toplanan Gen: {len(all_genes)}")
            time.sleep(1)
        except Exception as e:
            print(f"Hata (Atlanıyor): {e}")
            time.sleep(3)
            
    df = pd.DataFrame(all_genes)
    if not df.empty:
        df['Genome ID'] = df['Genome ID'].astype(str) # Kaydetmeden önce emin ol
        df.to_csv(CACHE_FILE, index=False)
        print(f"💾 Gen verileri kaydedildi: {CACHE_FILE}")
    return df

def run_superbug_panel():
    print("🏥 V13.2 KLİNİK SÜPER BAKTERİ PANELİ (SIZINTISIZ VE TİP HATASI DÜZELTİLMİŞ)...\n")
    
    y_df = pd.read_csv(LABELS_FILE)
    exclude_cols = ['Genome ID', 'Bacteria_Type']
    all_antibiotics = [col for col in y_df.columns if col not in exclude_cols]
    top_5_antibiotics = y_df[all_antibiotics].count().nlargest(5).index.tolist()
    
    sample_genomes = y_df['Genome ID'].astype(str).unique()[:MAX_GENOMES]
    y_df['Genome ID'] = y_df['Genome ID'].astype(str)
    
    genes_df = fetch_amr_genes_from_bvbrc(sample_genomes)
    if len(genes_df) == 0: return
        
    genes_df['Değer'] = 1
    # Ekstra güvenlik: Merge öncesi ikisini de kesinlikle string yap
    genes_df['Genome ID'] = genes_df['Genome ID'].astype(str)
    
    X_matrix = genes_df.pivot_table(index='Genome ID', columns='AMR_Gene', values='Değer', aggfunc='max', fill_value=0).reset_index()
    X_matrix.columns = [re.sub(r'[\[\]<>]', '', str(col)) for col in X_matrix.columns]
    
    final_df = pd.merge(X_matrix, y_df, on="Genome ID", how="inner")
    
    results = []
    
    print("\n" + "="*75)
    print("⚙️ VERİ SIZINTISI ENGELLENDİ: CV ve Sınıf Ağırlıkları Sadece Train Setinden Alınıyor...")
    print("="*75)
    
    for anti in top_5_antibiotics:
        anti_df = final_df.dropna(subset=[anti])
        if len(anti_df) < 50: continue
            
        X_data = anti_df.drop(columns=['Genome ID', 'Bacteria_Type'] + all_antibiotics, errors='ignore')
        y_data = anti_df[anti]
        
        # SIZINTI ÇÖZÜMÜ ADIM 1: Veriyi EN BAŞTA bölüyoruz!
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42, stratify=y_data)
        
        # SIZINTI ÇÖZÜMÜ ADIM 2: Pos_weight sadece Train verisinden hesaplanıyor!
        class_counts = y_train.value_counts()
        pos_weight = class_counts[0] / class_counts[1] if 1 in class_counts and class_counts[1] > 0 else 1
        
        model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, eval_metric='logloss', scale_pos_weight=pos_weight)
        
        # SIZINTI ÇÖZÜMÜ ADIM 3: CV işlemi TEST SETİNİ GÖRMEDEN, sadece Train üzerinde yapılıyor!
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_f1_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='f1')
        cv_f1_mean = cv_f1_scores.mean()
        
        # Final Testi
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Eklenen Metrikler (Recall ve F1)
        test_f1 = f1_score(y_test, y_pred, zero_division=0)
        test_recall = recall_score(y_test, y_pred, zero_division=0)
        
        # CV vs Test Karşılaştırması (Overfitting Uyarı Sistemi)
        f1_diff = test_f1 - cv_f1_mean
        status = "✅ Normal" if abs(f1_diff) < 0.15 else "⚠️ Overfit Şüphesi"
        
        cm = confusion_matrix(y_test, y_pred)
        true_resistant = cm[1][1] if len(cm) > 1 and len(cm[0]) > 1 else 0
        actual_resistant_in_test = sum(y_test == 1)
        
        results.append({
            'Antibiyotik': anti.upper(),
            'CV F1 (Train)': f"{cv_f1_mean:.3f}",
            'Test F1': f"{test_f1:.3f}",
            'CV/Test Farkı': f"{f1_diff:+.3f} ({status})",
            'Test Recall': f"%{test_recall*100:.1f}",
            'Bulunan Dirençli': f"{true_resistant}/{actual_resistant_in_test}"
        })

    print("\n" + "🩺"*40)
    print("      V13.2 KUSURSUZ DENETİM RAPORU (SIFIR SIZINTI, FULL ŞEFFAFLIK)      ")
    print("🩺"*40)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    run_superbug_panel()