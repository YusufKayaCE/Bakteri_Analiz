# Bakteri Genomlarından Makine Öğrenmesi ile Antimikrobiyal Direnç (AMR) Tahmini

Kim ve ark. (2022, *Clin Microbiol Rev* 35(3):e00179-21) metodolojik çerçevesine hizalı,
**çok-tür / çok-antibiyotik**, klinik-güvenli bir AMR tahmin sistemi.
Bir bakteri izolatının genomundan (AMR gen varlığı + kanonik nokta mutasyonları) belirli bir
antibiyotiğe **Dirençli (R)** mi **Duyarlı (S)** mi olduğunu tahmin eder.

---

## 🎯 Problem
Geleneksel antibiyogram (kültür + AST) **24–72 saat** sürer. Bu sürede klinisyen ampirik
(tahmini) antibiyotik vermek zorundadır. Tüm-genom dizileme ucuzladığından, genomdan
**saatler içinde** doğrudan direnç tahmini mümkündür. Amaç: bunu **klinik-güvenli hata
oranlarıyla** (VME/ME dengesi) yapmaktır.

- **VME (Very Major Error):** dirençliyi duyarlı sanmak — en tehlikeli hata.
- **ME (Major Error):** duyarlıyı dirençli sanmak — gereksiz geniş-spektrum.

## 🧬 Yöntem (pipeline)
1. **Veri:** BV-BRC (genom + AMR özel-genler + `genome_amr` fenotip) + NCBI Pathogen
   Detection / AMRFinderPlus (QRDR nokta mutasyonları). Etiketler **sadece laboratuvar
   ölçümü** (evidence filtresi — BV-BRC'nin ~%92'si computational tahmindir, elenir).
2. **Özellikler:** AMR gen varlığı (ikili), tür-spesifik QRDR mutasyonları (gyrA/parC/grlA),
   gen-gen etkileşimleri. Mutual-information ile özellik seçimi.
3. **Model:** Tür-bazlı (per-species) **stacked ensemble** — XGBoost + LightGBM + RandomForest
   + LogisticRegression, meta-learner LR. Optuna ile hiperparametre optimizasyonu (GPU).
4. **Dengesizlik:** `scale_pos_weight` (v36'da SMOTE kapalı — sentetik gürültü yerine sınıf
   ağırlığı).
5. **Kalibrasyon + eşik:** İzotonik/adaptif-sigmoid olasılık kalibrasyonu; dengeli (Equal
   Error Rate) klinik karar eşiği.
6. **Doğrulama:** Coğrafya-bazlı **GroupKFold** çapraz-doğrulama (klonal sızıntıya karşı),
   stratified bootstrap %95 güven aralıkları.
7. **Yorumlanabilirlik:** SHAP — her tahminin biyolojik gerekçesi.

## 📊 Sonuçlar (v36 — büyütme öncesi)
- **16 model** (tür × antibiyotik); **14'ü** VME ≤ %20 **ve** ME ≤ %20 aralığında.
- Ortalama ensemble AUC ≈ **0,92**.
- En iyi: *S. aureus* / ciprofloxacin — **VME %9,8, ME %8,5, AUC 0,96**.
- SHAP kararları bilinen mekanizmalarla örtüşür: cipro → *gyrA/parC*, beta-laktam →
  TEM/CTX-M/CMY, *S. aureus* → *mecA* (PBP2a), gentamisin → AAC enzimleri.

Etkileşimli sonuç panosu: **`web/AMR_v36_dashboard.html`** (çift tıkla aç).
Akademik özet belge: **`AMR_Tez_Dokuman.pdf`**.

## 📁 Dosya yapısı
```
src/v36.py                 # Ana pipeline (SMOTE-off + adaptif kalibrasyon; tek dosya, self-contained)
src/reporting_module.py    # Opsiyonel akademik rapor üretimi (v36 import eder)
src/*_cell.txt             # Colab veri-çekme / büyütme hücreleri (BV-BRC, QRDR, NCBI PD)
make_amr_pdf.py            # Tez/rapor PDF üretici
AMR_Tez_Dokuman.pdf        # Problem / literatür / dataset / çözüm belgesi
web/AMR_v36_dashboard.html # Etkileşimli sonuç panosu (standalone)
requirements.txt
```
> `data/`, `models/`, `reports/` depoya dahil **değildir** (büyük); veriler BV-BRC ve
> NCBI Pathogen Detection'dan üretilir.

## ▶️ Çalıştırma (Google Colab)
1. `pip install -r requirements.txt`
2. Google Drive'ı bağla; `v2_multilabel_labels.csv` ve gen önbelleğini
   `/content/drive/MyDrive/amr_v29/data/processed/` altına koy.
3. `src/v36.py` içeriğini bir hücreye yapıştırıp çalıştır → `run_superbug_panel()`
   tüm paneli eğitir. Tek tür/ilaç için: `run_superbug_panel(species_filter=["ecoli"], antibiotic_filter=["cipro"])`.

## 📚 Başlıca referanslar
- Kim JI ve ark. (2022) *Clin Microbiol Rev* 35(3):e00179-21 — ana çerçeve.
- Davis JJ ve ark. (2016) *Sci Rep* 6:27930 — PATRIC/RAST.

