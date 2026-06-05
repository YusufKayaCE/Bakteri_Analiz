import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# Ayarlar
FEATURES_FILE = "../data/processed/v2_X_features_clean.csv"
LABELS_FILE = "../data/processed/v2_multilabel_labels.csv"
MODEL_DIR = "../models/v2_models/"

def train_v3_hybrid_models():
    print("V3.0 Hibrit Model: DNA + Biyolojik Özellikler Yükleniyor...\n")
    
    # 1. Verileri Oku
    X_df = pd.read_csv(FEATURES_FILE)
    y_df = pd.read_csv(LABELS_FILE)
    
    # 2. X ve y'yi kusursuzca eşleştir
    merged_df = pd.merge(X_df, y_df, on="Genome ID", how="inner")
    
    # K-mer sütunlarını tespit et (4096 adet DNA sütunu)
    kmer_cols = [col for col in X_df.columns if col != 'Genome ID']
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    hedef_antibiyotikler = ['ampicillin', 'ciprofloxacin', 'gentamicin']
    
    for anti in hedef_antibiyotikler:
        if anti not in merged_df.columns:
            continue
            
        print("=" * 45)
        print(f"🧬 HİBRİT YAPAY ZEKA (DNA + METADATA): {anti.upper()}")
        print("=" * 45)
        
        # Sadece test edilmiş verileri al
        clean_df = merged_df.dropna(subset=[anti]).copy()
        
        # --- YENİ PARAMETRELER EKLENİYOR ---
        print("Yeni parametreler (Bakteri Türü ve Genom Uzunluğu) matrise işleniyor...")
        
        # Parametre 1: Bakteri Türü (One-Hot Encoding)
        # Örn: Tür_ECOLI: 1, Tür_SALMONELLA: 0 gibi yeni sütunlar yaratır
        bacteria_dummies = pd.get_dummies(clean_df['Bacteria_Type'], prefix='Tür')
        
        # Parametre 2: Toplam Genom Uzunluğu (Tüm K-mer sayılarının toplamı)
        genome_length = clean_df[kmer_cols].sum(axis=1)
        
        # Özellikler Matrisini (X) İnşa Et
        X_data = clean_df[kmer_cols].copy() # 4096 DNA sütunu
        X_data = pd.concat([X_data, bacteria_dummies], axis=1) # Bakteri türlerini ekle
        X_data['Genom_Uzunlugu'] = genome_length # Uzunluğu ekle
        
        y_data = clean_df[anti]
        
        # %80 Eğitim, %20 Test
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
        
        # Modeli Eğit (Dengeli hiperparametrelerle, sistemi çökertmeden!)
        model = xgb.XGBClassifier(
            eval_metric='logloss',
            tree_method='hist', 
            max_depth=6,
            learning_rate=0.1
        )
        
        print("Model yeni özellikleri kullanarak öğreniyor...")
        model.fit(X_train, y_train)
        
        # Test ve Başarı Ölçümü
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f">>> {anti.upper()} HİBRİT BAŞARI ORANI: %{acc*100:.2f} <<<")
        
        # Modeli Kaydet
        model_path = os.path.join(MODEL_DIR, f"xgboost_{anti}_v3_hybrid.pkl")
        joblib.dump(model, model_path)
        print(f"[BAŞARILI] Hibrit model kasaya kilitlendi.\n")

if __name__ == "__main__":
    train_v3_hybrid_models()