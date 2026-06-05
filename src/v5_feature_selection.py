import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
import os
import warnings
warnings.filterwarnings('ignore')

FEATURES_FILE = "../data/processed/v2_X_features_clean.csv"
LABELS_FILE = "../data/processed/v2_multilabel_labels.csv"

def train_v5_filtered_models():
    print("V5.0 Gürültü Filtreleme (Feature Selection) Başlıyor...\n")
    
    X_df = pd.read_csv(FEATURES_FILE)
    y_df = pd.read_csv(LABELS_FILE)
    
    merged_df = pd.merge(X_df, y_df, on="Genome ID", how="inner")
    exclude_cols = ['Genome ID', 'Bacteria_Type']
    antibiotics = [col for col in y_df.columns if col not in exclude_cols]
    
    # Hedefler
    hedef_antibiyotikler = ['ampicillin', 'ciprofloxacin']
    
    for anti in hedef_antibiyotikler:
        if anti not in merged_df.columns:
            continue
            
        print("=" * 55)
        print(f"🧬 BİYOLOJİK FİLTRELEME & YAPAY ZEKA: {anti.upper()}")
        print("=" * 55)
        
        clean_df = merged_df.dropna(subset=[anti])
        
        X_data = clean_df.drop(columns=antibiotics + exclude_cols)
        y_data = clean_df[anti]
        
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
        
        # --- SİHİRLİ DOKUNUŞ: FEATURE SELECTION ---
        # 4096 genden en etkili 250 tanesini seç!
        print("İstatistiksel analiz yapılıyor... En kritik 250 gen seçiliyor...")
        selector = SelectKBest(score_func=f_classif, k=250)
        
        # Filtreyi eğitim verisine göre ayarla ve uygula
        X_train_filtered = selector.fit_transform(X_train, y_train)
        # Aynı filtreyi test verisine de uygula
        X_test_filtered = selector.transform(X_test)
        
        # XGBoost'u bu saflaştırılmış veriyle eğit
        model = xgb.XGBClassifier(
            eval_metric='logloss',
            tree_method='hist',
            max_depth=6,
            learning_rate=0.1,
            n_estimators=150
        )
        
        print("Model sadece en kritik genler üzerinden eğitiliyor...")
        model.fit(X_train_filtered, y_train)
        
        y_pred = model.predict(X_test_filtered)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"\n>>> {anti.upper()} V5.0 FİLTRELENMİŞ BAŞARI ORANI: %{acc*100:.2f} <<<")
        print("-" * 55)

if __name__ == "__main__":
    train_v5_filtered_models()