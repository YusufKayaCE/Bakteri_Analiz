import pandas as pd
import requests

# Verimizden rastgele bir bakteri ID'si alıyoruz
LABELS_FILE = "../data/processed/v2_multilabel_labels.csv"
y_df = pd.read_csv(LABELS_FILE)
test_id = y_df['Genome ID'].astype(str).iloc[0] # İlk bakteriyi seç

print(f"🕵️‍♂️ Hedef Bakteri ID: {test_id}")
print("Tüm özel genler filtresiz çekiliyor...\n")

# HİÇBİR FİLTRE YOK (eq(property...) kısmını sildik)
url = f"https://www.bv-brc.org/api/sp_gene/?eq(genome_id,{test_id})&limit(5)&http_accept=application/json"

response = requests.get(url)
data = response.json()

if data:
    print("İşte API'nin bize döndürdüğü ham veri formatı:")
    import json
    print(json.dumps(data[0], indent=4)) # İlk genin tüm detaylarını güzelce yazdır
else:
    print("Bu bakteride özel gen bulunamadı. Başka bir ID deneyelim.")