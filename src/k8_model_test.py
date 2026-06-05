import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Ayarlar
FEATURES_FILE = "../data/processed/k8_X_features_fast.csv"
LABELS_FILE = "../data/processed/v2_multilabel_labels.csv"

def test_k8_model():
    print("K=8 Dev Matris Yükleniyor (1000 Satır x 65.536 Sütun)...")
    X_df = pd.read_csv(FEATURES_FILE)
    y_df = pd.read_csv(LABELS_FILE)

    # Eşleştirme (İndirdiğimiz 1000 genom ile etiketleri birleştiriyoruz)
    merged_df = pd.merge(X_df, y_df, on="Genome ID", how="inner")
    
    # Sadece Ampicillin için
    anti = 'ampicillin'
    clean_df = merged_df.dropna(subset=[anti])
    
    print(f"Toplam {len(clean_df)} bakteri ile AMPICILLIN K=8 Testi Başlıyor...")

    # Özellikler (65.536 K-mer) ve Hedef Ayrımı
    exclude_cols = ['Genome ID', 'Bacteria_Type']
    antibiotics = [col for col in y_df.columns if col not in exclude_cols]
    
    X_data = clean_df.drop(columns=antibiotics + exclude_cols)
    y_data = clean_df[anti]

    # %80 Eğitim, %20 Test
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

    # XGBoost Büyük Veri Modu (Histogram)
    model = xgb.XGBClassifier(
        eval_metric='logloss',
        tree_method='hist', # 65.536 sütun için bu parametre HAYATİDİR!
        max_depth=6,
        learning_rate=0.1
    )

    print("XGBoost 65.536 farklı genetik örüntüyü inceliyor (Biraz sürebilir)...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*45)
    print(f"🧬 K=8 AMPICILLIN YENİ BAŞARI ORANI: %{acc*100:.2f}")
    print("="*45)
    print("Eski K=6 başarımız: %65.42 idi.")

if __name__ == "__main__":
    test_k8_model()