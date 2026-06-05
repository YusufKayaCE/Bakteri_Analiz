import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

# Grafiklerin akademik makale formatında (şık ve okunaklı) olması için genel ayarlar
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16})

def generate_academic_reports(model_path, X_test, y_test, antibiotic_name, output_dir="../reports/figures"):
    """
    Eğitilmiş bir modeli kullanarak jüri sunumu için akademik grafikler üretir.
    """
    os.makedirs(output_dir, exist_ok=True)
    safe_name = antibiotic_name.lower().replace(" ", "_")
    
    # 1. Modeli ve Ayarları Yükle
    try:
        saved_data = joblib.load(model_path)
        model = saved_data["model"]
        threshold = saved_data["threshold"]
    except Exception as e:
        print(f"⚠️ Model yüklenemedi ({antibiotic_name}): {e}")
        return

    # Olasılıkları ve Klinik Kararları Hesapla
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    print(f"\n🎨 {antibiotic_name.upper()} için grafikler çiziliyor...")

    # ---------------------------------------------------------
    # GÖRSEL 1: KLİNİK KARMAŞIKLIK MATRİSİ (Confusion Matrix)
    # ---------------------------------------------------------
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(6, 5))
    # Hastalık (Direnç) tahmini olduğu için renk paletini tıp domainine uygun seçiyoruz (Kırmızı tonları uyarıcıdır)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=False,
                xticklabels=['Duyarlı (0)', 'Dirençli (1)'],
                yticklabels=['Duyarlı (0)', 'Dirençli (1)'],
                annot_kws={"size": 16, "weight": "bold"})
    
    plt.title(f'{antibiotic_name.capitalize()} - Karmaşıklık Matrisi (Thr: {threshold:.2f})')
    plt.ylabel('Gerçek Sınıf (Ground Truth)')
    plt.xlabel('Modelin Tahmini (Predicted Sınıf)')
    
    cm_path = os.path.join(output_dir, f"{safe_name}_confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300) # 300 DPI akademik yayın standartıdır
    plt.close()

    # ---------------------------------------------------------
    # GÖRSEL 2: ROC EĞRİSİ VE AUC (Alıcı İşletim Karakteristiği)
    # ---------------------------------------------------------
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Eğrisi (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Şans Seviyesi')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Yanlış Pozitif Oranı (FPR)')
    plt.ylabel('Doğru Pozitif Oranı (TPR - Recall)')
    plt.title(f'{antibiotic_name.capitalize()} - ROC Eğrisi')
    plt.legend(loc="lower right")
    
    roc_path = os.path.join(output_dir, f"{safe_name}_roc_curve.png")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # GÖRSEL 3: SHAP SUMMARY PLOT (Özellik Etki Dağılımı)
    # ---------------------------------------------------------
    # SHAP grafiklerini Matplotlib ile kaydetmek biraz trick'lidir
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(10, 8))
    # show=False parametresi, grafiğin ekrana basılmadan dosyaya kaydedilmesini sağlar
    shap.summary_plot(shap_values, X_test, max_display=10, show=False)
    plt.title(f"{antibiotic_name.capitalize()} - SHAP Gen Etkisi (İlk 10)", fontsize=16)
    
    shap_path = os.path.join(output_dir, f"{safe_name}_shap_summary.png")
    plt.tight_layout()
    plt.savefig(shap_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   ✅ Görseller kaydedildi -> {output_dir}")