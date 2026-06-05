import pandas as pd
import os

def prepare_target_labels(resistant_path: str, susceptible_path: str, output_path: str) -> pd.DataFrame:
    """
    Dirençli ve Duyarlı fenotip verilerini birleştirir, temizler, binary formata çevirir
    ve işlenmiş veriyi kaydeder.
    
    Zaman Karmaşıklığı: O(N)
    Alan Karmaşıklığı: O(N)
    
    Parametreler:
    - resistant_path (str): Dirençli verilerin CSV yolu
    - susceptible_path (str): Duyarlı verilerin CSV yolu
    - output_path (str): İşlenmiş verinin kaydedileceği yol
    """
    print("Veri hazırlama ve etiketleme (Labeling) işlemi başlatıldı...")
    
    # 1. Verileri Oku (Sadece gerekli sütunlar - RAM Optimizasyonu)
    use_cols = ['Genome ID', 'Resistant Phenotype']
    df_res = pd.read_csv(resistant_path, usecols=use_cols)
    df_sus = pd.read_csv(susceptible_path, usecols=use_cols)
    
    # 2. DataFrame'leri Birleştir
    df_combined = pd.concat([df_res, df_sus], ignore_index=True)
    
    # 3. Tekrar eden Genome ID'leri temizle (Dublicate handling)
    initial_shape = df_combined.shape[0]
    df_combined.drop_duplicates(subset=['Genome ID'], keep='first', inplace=True)
    print(f"-> {initial_shape - df_combined.shape[0]} adet tekrar eden Genome ID temizlendi.")
    
    # 4. Binary Label Mapping ve Bellek Optimizasyonu (Downcasting to int8)
    label_mapping = {'Resistant': 1, 'Susceptible': 0}
    df_combined['Target'] = df_combined['Resistant Phenotype'].map(label_mapping).astype('int8')
    df_combined.drop(columns=['Resistant Phenotype'], inplace=True)
    
    # 5. Sonuçları 'processed' klasörüne kaydet
    # Klasör yoksa oluştur
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_combined.to_csv(output_path, index=False)
    
    print(f"-> Hedef Değişken (y) başarıyla hazırlandı! Toplam Örnek: {df_combined.shape[0]}")
    print(f"-> Dosya kaydedildi: {output_path}")
    
    return df_combined

# Bu dosya doğrudan çalıştırılırsa test et:
if __name__ == "__main__":
    # Yolların senin proje mimarine (AMR_Prediction_Project) uygun olduğundan emin ol.
    RESISTANT_FILE = "../data/raw/BVBRC_genome_amr_resisdant.csv"
    SUSCEPTIBLE_FILE = "../data/raw/BVBRC_genome_amr_suspect.csv"
    OUTPUT_FILE = "../data/processed/y_labels.csv"
    
    # Fonksiyonu çalıştır
    prepare_target_labels(RESISTANT_FILE, SUSCEPTIBLE_FILE, OUTPUT_FILE)