import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
import numpy as np

# Ayarlar
FEATURES_FILE = "../data/processed/v2_X_features_clean.csv"
LABELS_FILE = "../data/processed/v2_multilabel_labels.csv"

def train_v7_functional_model():
    print("🚀 V7.0 BİLGİ ODAKLI (KNOWLEDGE-DRIVEN) MODEL BAŞLIYOR...")
    
    X_df = pd.read_csv(FEATURES_FILE)
    y_df = pd.read_csv(LABELS_FILE)
    merged_df = pd.merge(X_df, y_df, on="Genome ID", how="inner")
    
    # Hedefimiz Ampicillin (Çünkü plazmid/gen bazlı direnci en yüksek olan bu)
    anti = 'ampicillin'
    clean_df = merged_df.dropna(subset=[anti])
    
    # 1. ÖZELLİK MÜHENDİSLİĞİ (Makaledeki Çözüm)
    # Tüm sütunlar arasından varyansı en yüksek (yani en çok değişim gösteren) genleri seçiyoruz
    # Bu genelde "Direnç Genlerinin" olduğu bölgelere işaret eder.
    kmer_cols = [col for col in X_df.columns if col != 'Genome ID']
    X_raw = clean_df[kmer_cols]
    
    # Varyans Filtresi: Hiç değişmeyen (sıkıcı) genleri at, en hareketli 500 geni al
    top_500_vars = X_raw.var().sort_values(ascending=False).head(500).index
    X_data = X_raw[top_500_vars]
    y_data = clean_df[anti]

    # 2. VERİ DENGESİ (Undersampling)
    rus = RandomUnderSampler(random_state=42)
    X_res, y_res = rus.fit_resample(X_data, y_data)
    
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
    
    # 3. İLERİ SEVİYE XGBOOST (Gürültüye karşı bağışıklık)
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4, # Sığ ağaçlar kullanarak ezberlemeyi (overfitting) önlüyoruz
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss'
    )
    
    print(f"Eğitim başlıyor... {len(X_train)} dengeli örnek üzerinden {len(top_500_vars)} kritik gen inceleniyor.")
    model.fit(X_train, y_train)
    
    # 4. SONUÇLAR
    y_pred = model.predict(X_test)
    
    print("\n" + "="*50)
    print(f"📊 V7.0 HİBRİT MODEL SONUÇLARI ({anti.upper()})")
    print("="*50)
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n✅ Doğru Bildiği Duyarlılar (0): {cm[0][0]}")
    print(f"🔥 DOĞRU BİLDİĞİ DİRENÇLİLER (1): {cm[1][1]} <--- İŞTE BU!")
    print(f"❌ Hatalı Tahmin Sayısı: {cm[0][1] + cm[1][0]}")

if __name__ == "__main__":
    train_v7_functional_model()