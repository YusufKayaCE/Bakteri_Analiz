import itertools

input_file = "../data/processed/v2_X_features.csv"
output_file = "../data/processed/v2_X_features_clean.csv"

# Sadece temiz A,C,G,T harflerinden oluşan o 4096 K-mer başlıklarını hazırlıyoruz
bases = ['A', 'C', 'G', 'T']
ALL_KMERS = [''.join(p) for p in itertools.product(bases, repeat=6)]
header = "Genome ID," + ",".join(ALL_KMERS) + "\n"

print("Operasyon Neşter: Bozuk veriler ayrıştırılıyor...")

valid_lines = 0
with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    outfile.write(header) # En başa tertemiz başlığımızı koyuyoruz
    
    for line in infile:
        # Sadece tam 4096 virgül içeren (yani tam 4097 sütunlu) KUSURSUZ satırları kurtar
        if line.count(',') == 4096:
            # Başlık satırlarını değil, sadece Genome ID (sayı) ile başlayan verileri al
            if line[0].isdigit(): 
                outfile.write(line)
                valid_lines += 1

print(f"\n[BAŞARILI] Kurtarılan Kusursuz Genom Sayısı: {valid_lines}")
print(f"Tertemiz dosyan hazır: {output_file}")