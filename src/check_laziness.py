import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

FEATURES_FILE = "../data/processed/v2_X_features_clean.csv"
LABELS_FILE = "../data/processed/v2_multilabel_labels.csv"

def run_lie_detector():
    print("🕵️‍♂️ YAPAY ZEKA YALAN MAKİNESİNE BAĞLANIYOR...\n")
    
    X_df = pd.read_csv(FEATURES_FILE)
    y_df = pd.read_csv(LABELS_FILE)
    merged_df = pd.merge(X_df, y_df, on="Genome ID", how="inner")
    
    exclude_cols = ['Genome ID', 'Bacteria_Type']
    antibiotics = [col for col in y_df.columns if col not in exclude_cols]
    
    # Şüphe duyduğumuz Ciprofloxacin modelini test edelim
    anti = 'gentamicin'
    print(f"Hedef: {anti.upper()} (Başarımız %75.21 çıkmıştı)\n")
    
    clean_df = merged_df.dropna(subset=[anti])
    X_data = clean_df.drop(columns=antibiotics + exclude_cols)
    y_data = clean_df[anti]
    
    # Modelin test edildiği o %20'lik kısmı tekrar ayır
    _, X_test, _, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
    
    # Kasadaki efsane V2 modelimizi çağırıyoruz
    model_path = f"../models/v2_models/xgboost_{anti}_v2.pkl"
    model = joblib.load(model_path)
    
    # Modelden tahmin yapmasını iste
    y_pred = model.predict(X_test)
    
    print("=" * 50)
    print("📊 KARMAŞIKLIK MATRİSİ (Confusion Matrix)")
    print("=" * 50)
    cm = confusion_matrix(y_test, y_pred)
    print(f"Gerçekte DUYARLI (0) olup, modelin doğru bildiği: {cm[0][0]}")
    print(f"Gerçekte DİRENÇLİ (1) olup, modelin doğru bildiği: {cm[1][1]}")
    print(f"Modelin kafasının karıştığı toplam vaka: {cm[0][1] + cm[1][0]}")
    
    print("\n" + "=" * 50)
    print("📈 DETAYLI KARNE (Classification Report)")
    print("=" * 50)
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    run_lie_detector()