import pandas as pd
import glob
import os

def create_multilabel_matrix(raw_data_path="../data/raw/amr_*.csv"):
    print("V2.0: Çoklu Etiketli Matris Oluşturuluyor (Multi-Label Pivot)...\n")
    
    all_matrices = []
    
    file_list = glob.glob(raw_data_path)
    
    for file in file_list:
        bakteri_adi = os.path.basename(file).replace("amr_", "").replace(".csv", "").upper()
        print(f"-> İşleniyor: {bakteri_adi}")
        
        # 1. Veriyi oku
        df = pd.read_csv(file, usecols=['Genome ID', 'Antibiotic', 'Resistant Phenotype'], dtype=str)
        df['Antibiotic'] = df['Antibiotic'].str.lower()
        
        # 2. Sadece dirençli (1) ve duyarlı (0) olanları al
        df = df[df['Resistant Phenotype'].isin(['Resistant', 'Susceptible'])]
        df['Target'] = df['Resistant Phenotype'].map({'Resistant': 1, 'Susceptible': 0})
        
        # 3. En popüler 5 antibiyotiği belirle
        top_5 = df['Antibiotic'].value_counts().head(5).index.tolist()
        df_filtered = df[df['Antibiotic'].isin(top_5)]
        
        # 4. PIVOT: Uzun formattan geniş formata geçiş
        # Index: Genome ID, Sütunlar: Antibiyotik isimleri, Değerler: Target (1/0)
        matrix = df_filtered.pivot_table(index='Genome ID', 
                                        columns='Antibiotic', 
                                        values='Target', 
                                        aggfunc='first')
        
        # Bakteri türünü de bir sütun olarak ekleyelim (Hangi modelin çözeceğini bilmek için)
        matrix['Bacteria_Type'] = bakteri_adi
        
        all_matrices.append(matrix)
        print(f"   {bakteri_adi} tamamlandı. Satır sayısı: {len(matrix)}")

    # 5. Tüm bakterileri tek bir devasa tabloda birleştir
    final_df = pd.concat(all_matrices, axis=0)
    
    # 6. Kaydet
    output_path = "../data/processed/v2_multilabel_labels.csv"
    os.makedirs("../data/processed", exist_ok=True)
    final_df.to_csv(output_path)
    
    print(f"\n[BAŞARILI] V2.0 Etiket Matrisi Hazır: {output_path}")
    print(f"Toplam Benzersiz Bakteri Sayısı: {len(final_df)}")
    print("Sütunlar (Hedef Antibiyotikler):", final_df.columns.tolist())

if __name__ == "__main__":
    create_multilabel_matrix()