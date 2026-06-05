import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

FEATURES_FILE = "../data/processed/v2_X_features_clean.csv"
LABELS_FILE = "../data/processed/v2_multilabel_labels.csv"

def train_neural_network():
    print("🧠 V10.0 DERİN ÖĞRENME (YAPAY SİNİR AĞI) BAŞLATILIYOR...")
    print("XGBoost devre dışı. Model tüm DNA imzasını bir bütün olarak okuyacak.\n")
    
    # 1. Veriyi Yükle
    X_df = pd.read_csv(FEATURES_FILE)
    y_df = pd.read_csv(LABELS_FILE)
    merged_df = pd.merge(X_df, y_df, on="Genome ID", how="inner")
    
    anti = 'ampicillin'
    clean_df = merged_df.dropna(subset=[anti])
    
    exclude_cols = ['Genome ID', 'Bacteria_Type']
    antibiotics = [col for col in y_df.columns if col not in exclude_cols]
    
    X_raw = clean_df.drop(columns=antibiotics + exclude_cols)
    y_raw = clean_df[anti]
    
    # 2. Adil Sınav (Undersampling)
    rus = RandomUnderSampler(random_state=42)
    X_balanced, y_balanced = rus.fit_resample(X_raw, y_raw)
    
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)
    
    # 3. SİHİRLİ DOKUNUŞ: Standardizasyon (Sinir Ağları için Hayatidir!)
    # Nöronların kafası karışmasın diye tüm DNA kelime sayılarını 0 ile 1 arasında ölçekliyoruz.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. ÇOK KATMANLI SİNİR AĞI (Multi-Layer Perceptron)
    # 3 Gizli Katman: 256 nöron -> 128 nöron -> 64 nöron
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64), 
        activation='relu',
        solver='adam',
        alpha=0.001,           # L2 Regülarizasyon (Ezberlemeyi önler)
        learning_rate_init=0.001,
        max_iter=300,
        random_state=42,
        early_stopping=True    # Model ezberlemeye başlarsa eğitimi durdur
    )
    
    print("🤖 Nöronlar eğitiliyor... Bu işlem biraz sürebilir (Tüm genetik yapı aynı anda sentezleniyor)...")
    mlp.fit(X_train_scaled, y_train)
    
    # 5. Gerçek Yüzleşme
    y_pred = mlp.predict(X_test_scaled)
    
    print("\n" + "="*50)
    print(f"🧬 SİNİR AĞI (NEURAL NETWORK) SONUÇLARI ({anti.upper()})")
    print("="*50)
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"✅ Doğru Bildiği Duyarlılar (0) : {cm[0][0]}")
    print(f"🔥 DOĞRU BİLDİĞİ DİRENÇLİLER (1): {cm[1][1]}")
    print(f"❌ Kafasının Karıştığı (Hata)    : {cm[0][1] + cm[1][0]}")

if __name__ == "__main__":
    train_neural_network()