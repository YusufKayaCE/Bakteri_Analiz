import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Ayarlar
FEATURES_FILE = "../data/processed/v2_X_features_clean.csv"
LABELS_FILE = "../data/processed/v2_multilabel_labels.csv"
MODEL_DIR = "../models/v4_models/" # Yeni kasamız

def train_v4_smote_models():
    print("V4.0 SMOTE (Sentetik Veri) Operasyonu Başlıyor...\n")
    
    X_df = pd.read_csv(FEATURES_FILE)
    y_df = pd.read_csv(LABELS_FILE)
    
    merged_df = pd.merge(X_df, y_df, on="Genome ID", how="inner")
    exclude_cols = ['Genome ID', 'Bacteria_Type']
    antibiotics = [col for col in y_df.columns if col not in exclude_cols]
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Şimdilik düşük kalan iki hedefimize odaklanalım
    hedef_antibiyotikler = ['ampicillin', 'ciprofloxacin']
    
    for anti in hedef_antibiyotikler:
        if anti not in merged_df.columns:
            continue
            
        print("=" * 50)
        print(f"🎯 KESKİN NİŞANCI MODELİ EĞİTİLİYOR: {anti.upper()}")
        print("=" * 50)
        
        clean_df = merged_df.dropna(subset=[anti])
        
        X_data = clean_df.drop(columns=antibiotics + exclude_cols)
        y_data = clean_df[anti]
        
        # 1. Veriyi Ayır (%80 Eğitim, %20 Test)
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
        
        # 2. SİHİRLİ DOKUNUŞ: Sadece eğitim verisine Sentetik Bakteri ekle!
        print(f"Eğitim öncesi orjinal veri sayısı: {len(y_train)}")
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        print(f"SMOTE sonrası sentetik genlerle zenginleşmiş veri sayısı: {len(y_train_smote)}")
        
        # 3. XGBoost'u Kur (Daha keskin, dengeli parametreler)
        model = xgb.XGBClassifier(
            eval_metric='logloss',
            tree_method='hist',
            max_depth=7,           # Biraz daha derine in
            learning_rate=0.08,    # Dikkatlice öğren
            n_estimators=200,      # Yeterli sayıda ağaç
            subsample=0.9,         
            colsample_bytree=0.9
        )
        
        print("Model sentetik klonlarla birlikte eğitiliyor...")
        model.fit(X_train_smote, y_train_smote)
        
        # 4. Test (Model HİÇ GÖRMEDİĞİ gerçek verilerle test edilecek)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"\n>>> {anti.upper()} V4.0 YENİ BAŞARI ORANI: %{acc*100:.2f} <<<")
        
        model_path = os.path.join(MODEL_DIR, f"xgboost_{anti}_v4_smote.pkl")
        joblib.dump(model, model_path)
        print("[BAŞARILI] Model kasaya kilitlendi.\n")

if __name__ == "__main__":
    train_v4_smote_models()