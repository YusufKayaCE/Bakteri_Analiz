import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings('ignore')

FEATURES_FILE = "../data/processed/v2_X_features_clean.csv"
LABELS_FILE = "../data/processed/v2_multilabel_labels.csv"

def run_true_learning():
    print("⚔️ V6.0 UNDERSAMPLING (ACI GERÇEKLER) OPERASYONU...\n")
    
    X_df = pd.read_csv(FEATURES_FILE)
    y_df = pd.read_csv(LABELS_FILE)
    merged_df = pd.merge(X_df, y_df, on="Genome ID", how="inner")
    
    exclude_cols = ['Genome ID', 'Bacteria_Type']
    antibiotics = [col for col in y_df.columns if col not in exclude_cols]
    
    anti = 'ampicillin' # O sahte şampiyonumuzu test ediyoruz
    print(f"HEDEF: {anti.upper()}")
    
    clean_df = merged_df.dropna(subset=[anti])
    X_data = clean_df.drop(columns=antibiotics + exclude_cols)
    y_data = clean_df[anti]
    
    # Veriyi ayır (Test setine dokunmuyoruz, o gerçek hayat!)
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
    
    print(f"Eğitimdeki Orjinal Dağılım:\n{y_train.value_counts()}\n")
    
    # SİHİRLİ DOKUNUŞ: Tırpanlıyoruz! (Undersampling)
    rus = RandomUnderSampler(random_state=42)
    X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)
    
    print(f"Tırpan Sonrası Eşit Dağılım:\n{y_train_rus.value_counts()}\n")
    
    # Model artık tembellik yapamayacak, öğrenmeye mecbur
    model = xgb.XGBClassifier(
        eval_metric='logloss',
        tree_method='hist',
        max_depth=5,
        learning_rate=0.1
    )
    
    print("Yapay Zeka nihayet DNA okumaya zorlanıyor...")
    model.fit(X_train_rus, y_train_rus)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print("=" * 50)
    print(f"GERÇEK BAŞARI ORANI: %{acc*100:.2f}")
    print("=" * 50)
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"Gerçekte DUYARLI (0) olup, modelin doğru bildiği: {cm[0][0]}")
    print(f"Gerçekte DİRENÇLİ (1) olup, modelin doğru bildiği: {cm[1][1]}")

if __name__ == "__main__":
    run_true_learning()