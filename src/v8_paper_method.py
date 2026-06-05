import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

# Dosya Yolları
FEATURES_FILE = "../data/processed/v2_X_features_clean.csv"
LABELS_FILE = "../data/processed/v2_multilabel_labels.csv"

def run_paper_methodology():
    print("🧬 V8.0 MAKALE METODOLOJİSİ (PMC9491192) UYGULANIYOR...")
    
    # 1. Veriyi Yükle
    X_df = pd.read_csv(FEATURES_FILE)
    y_df = pd.read_csv(LABELS_FILE)
    merged_df = pd.merge(X_df, y_df, on="Genome ID", how="inner")
    
    # Hedefimiz: Ampicillin (Plazmid bazlı direnç olduğu için yakalanma ihtimali en yüksek olan)
    anti = 'ampicillin'
    clean_df = merged_df.dropna(subset=[anti])
    
    exclude_cols = ['Genome ID', 'Bacteria_Type']
    antibiotics = [col for col in y_df.columns if col not in exclude_cols]
    
    X_raw = clean_df.drop(columns=antibiotics + exclude_cols)
    y_raw = clean_df[anti]
    
    print(f"\n1. ADIM: Sınıf Dengesizliği Çözülüyor (Undersampling)...")
    rus = RandomUnderSampler(random_state=42)
    X_balanced, y_balanced = rus.fit_resample(X_raw, y_raw)
    print(f"Eşitlenmiş Veri Seti: {len(y_balanced[y_balanced==0])} Duyarlı, {len(y_balanced[y_balanced==1])} Dirençli.")
    
    print(f"\n2. ADIM: Makale Mantığı - Direnç İmzaları (SNV/Gene) Taranıyor...")
    # SİHİRLİ DOKUNUŞ: 4096 genin hepsine bakmak yerine, Direnç (1) ve Duyarlı (0)
    # arasında istatistiksel olarak EN BÜYÜK uçurumu yaratan 50 "İmza K-mer"i seçiyoruz.
    selector = SelectKBest(score_func=f_classif, k=50)
    X_signatures = selector.fit_transform(X_balanced, y_balanced)
    
    # 3. Eğitim ve Test Setlerini Ayır
    X_train, X_test, y_train, y_test = train_test_split(X_signatures, y_balanced, test_size=0.2, random_state=42)
    
    # 4. Makale Standartlarında XGBoost (Sığ Ağaçlar, Genelleme Yeteneği)
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=3,          # Ezberlemeyi önlemek için sığ tutuyoruz
        learning_rate=0.05,
        subsample=0.8,
        eval_metric='logloss'
    )
    
    print("\n3. ADIM: Yapay Zeka sadece bu 50 Biyolojik İmza üzerinden eğitiliyor...")
    model.fit(X_train, y_train)
    
    # 5. Gerçek Yüzleşme (Test)
    y_pred = model.predict(X_test)
    
    print("\n" + "="*50)
    print(f"🏆 MAKALE (PMC9491192) MODELİ SONUÇLARI ({anti.upper()})")
    print("="*50)
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"✅ Doğru Bildiği Duyarlılar (0) : {cm[0][0]}")
    print(f"🔥 DOĞRU BİLDİĞİ DİRENÇLİLER (1): {cm[1][1]}")
    print(f"❌ Kafasının Karıştığı (Hata)    : {cm[0][1] + cm[1][0]}")

if __name__ == "__main__":
    run_paper_methodology()