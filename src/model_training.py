import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

def load_and_align_data(features_path: str, labels_path: str, target_col: str):
    """
    Özellik matrisi (X) ile etiketleri (y) 'Genome ID' üzerinden birleştirir (Inner Join).
    """
    print("1. Veri setleri hizalanıyor (Data Alignment)...")
    df_features = pd.read_csv(features_path)
    df_labels = pd.read_csv(labels_path)
    
    df_merged = pd.merge(df_features, df_labels, on='Genome ID', how='inner')
    
    y = df_merged[target_col]
    X = df_merged.drop(columns=['Genome ID', target_col])
    
    print(f" -> Hizalama Başarılı! X Boyutu: {X.shape}, y Boyutu: {y.shape}")
    return X, y

def plot_feature_importance(model, feature_names, top_n=15):
    """
    XGBoost modelinin karar verirken en çok hangi K-mer'lere (DNA dizilimlerine)
    baktığını görselleştirir. Akademik sunumlar için kritiktir.
    Zaman Karmaşıklığı: O(N log N) - N adet özelliğin sıralanması
    """
    print(f"\n4. En Önemli {top_n} Biyolojik Özellik (K-mer) Çıkarılıyor...")
    
    # Özellik önem derecelerini (Information Gain) al
    importances = model.feature_importances_
    
    # DataFrame oluştur ve sırala
    importance_df = pd.DataFrame({
        'K-mer': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(top_n)
    
    # Görselleştirme (Seaborn ile akademik ve profesyonel bir stil)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='K-mer', data=importance_df, hue='K-mer', palette='viridis', legend=False)
    plt.title(f'Salmonella AMR Tahmininde En Önemli {top_n} DNA Dizilimi', fontsize=14, fontweight='bold')
    plt.xlabel('Önem Derecesi (Feature Gain)', fontsize=12)
    plt.ylabel('K-mer (DNA Dizilimi)', fontsize=12)
    plt.tight_layout()
    
    # Grafiği 'reports/figures' klasörüne kaydet (Jüri sunumu için)
    save_path = "../reports/figures/feature_importance.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f" -> Grafik başarıyla kaydedildi: {save_path}")
    
    # Ekranda göster
    plt.show()

def train_evaluate_xgboost(X, y):
    """
    XGBoost modelini eğitir ve tıbbi veri setleri için kritik olan metrikleri hesaplar.
    """
    print("\n2. Model Eğitimi Başlıyor (XGBoost)...")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = xgb.XGBClassifier(
        n_estimators=100,        
        max_depth=5,             
        learning_rate=0.1,       
        random_state=42,
        n_jobs=-1,               
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    print(" -> XGBoost Modeli başarıyla eğitildi!")
    
    print("\n3. Model Değerlendirmesi (Evaluation)...")
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\n>>> Genel Doğruluk (Accuracy): %{acc * 100:.2f} <<<")
    print("-" * 50)
    print("Sınıflandırma Raporu (Precision, Recall, F1-Score):")
    print(classification_report(y_test, y_pred))
    
    return model

if __name__ == "__main__":
    FEATURES_FILE = "../data/processed/X_features.csv"
    LABELS_FILE = "../data/processed/y_labels.csv"
    TARGET_COLUMN = 'Target' 
    
    # Adım 1: Veriyi hazırla
    X, y = load_and_align_data(FEATURES_FILE, LABELS_FILE, target_col=TARGET_COLUMN)
    
    # Adım 2: Modeli eğit ve test et
    trained_model = train_evaluate_xgboost(X, y)
    
    # Adım 3: En önemli özellikleri görselleştir
    # X.columns modelin karar verirken kullandığı 4096 adet K-mer ismidir (AAAAAA, AAAAC vb.)
    # --- YENİ EKLENEN: MODELİ HARD DİSKE KAYDETME KISMI ---
    os.makedirs("../models", exist_ok=True)
    model_yolu = "../models/xgboost_salmonella_v1.pkl"
    joblib.dump(trained_model, model_yolu)
    print(f"\n[BAŞARILI] Modelin beyni hard diske kaydedildi: {model_yolu}")
    # --------------------------------------------------------
    
    # Adım 3: En önemli özellikleri görselleştir
    plot_feature_importance(trained_model, X.columns)