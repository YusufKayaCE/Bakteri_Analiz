# -*- coding: utf-8 -*-
"""AMR tez/rapor PDF üretici — Türkçe, reportlab Platypus."""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily

# ── Türkçe font (Arial) ──────────────────────────────────────────────────────
FD = "C:/Windows/Fonts/"
pdfmetrics.registerFont(TTFont("Arial",        FD + "arial.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Bold",   FD + "arialbd.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Italic", FD + "ariali.ttf"))
registerFontFamily("Arial", normal="Arial", bold="Arial-Bold",
                   italic="Arial-Italic", boldItalic="Arial-Bold")

styles = getSampleStyleSheet()
def S(name, **kw):
    kw.setdefault("fontName", "Arial")
    return ParagraphStyle(name, parent=styles["Normal"], **kw)

title_s   = S("t",  fontName="Arial-Bold", fontSize=19, leading=24, alignment=TA_CENTER, spaceAfter=4, textColor=colors.HexColor("#0B3D2E"))
subt_s    = S("st", fontSize=11, leading=15, alignment=TA_CENTER, textColor=colors.HexColor("#444444"), spaceAfter=14)
h1_s      = S("h1", fontName="Arial-Bold", fontSize=14, leading=18, spaceBefore=14, spaceAfter=6, textColor=colors.HexColor("#0B3D2E"))
h2_s      = S("h2", fontName="Arial-Bold", fontSize=11.5, leading=15, spaceBefore=8, spaceAfter=3, textColor=colors.HexColor("#1A5C44"))
body_s    = S("b",  fontSize=10.3, leading=15, alignment=TA_JUSTIFY, spaceAfter=6)
bullet_s  = S("bl", fontSize=10.3, leading=14.5, alignment=TA_JUSTIFY, leftIndent=14, bulletIndent=4, spaceAfter=3)
small_s   = S("sm", fontSize=8.6, leading=11.5, textColor=colors.HexColor("#555555"))
cell_s    = S("c",  fontSize=8.8, leading=11.5)
cellb_s   = S("cb", fontName="Arial-Bold", fontSize=8.8, leading=11.5, textColor=colors.white)

story = []
def P(t, st=body_s): story.append(Paragraph(t, st))
def B(t): story.append(Paragraph("• " + t, bullet_s))
def SP(h=6): story.append(Spacer(1, h))
def HR(): story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#1A5C44"), spaceBefore=2, spaceAfter=8))

# ── Başlık ───────────────────────────────────────────────────────────────────
P("Bakteri Genomlarından Makine Öğrenmesi ile<br/>Antimikrobiyal Direnç (AMR) Tahmini", title_s)
P("Kim ve ark. (2022, <i>Clin Microbiol Rev</i>) metodolojik çerçevesine hizalı, "
  "çok-tür / çok-antibiyotik, klinik-güvenli tahmin sistemi", subt_s)
HR()

# ════════════════════════════════════════════════════════════════ 1
P("1. Problem Tanımı", h1_s)

P("1.1. Antimikrobiyal Direnç ve Küresel Yük", h2_s)
P("Antimikrobiyal direnç (AMR), bakterilerin kendilerini etkisiz hale getirmesi gereken "
  "antibiyotiklere karşı hayatta kalma yeteneği kazanmasıdır. Murray ve ark. (2022, "
  "<i>Lancet</i>) tahminlerine göre 2019 yılında dünya genelinde yaklaşık 4,95 milyon ölüm "
  "AMR ile <b>ilişkili</b>, ~1,27 milyon ölüm ise doğrudan AMR'ye <b>atfedilebilir</b> "
  "bulunmuştur. Dünya Sağlık Örgütü (WHO) AMR'yi insanlığın karşı karşıya olduğu en büyük "
  "on küresel sağlık tehdidinden biri olarak ilan etmiştir. Yanlış veya gecikmiş antibiyotik "
  "seçimi hem tedavi başarısızlığına ve mortaliteye, hem de dirençli suşların yayılmasına "
  "yol açar.")

P("1.2. Geleneksel Tanının Darboğazı", h2_s)
P("Direnç belirlemede altın standart, kültür sonrası uygulanan fenotipik antibiyogramdır "
  "(AST): minimum inhibitör konsantrasyon (MIC) ölçümü veya disk difüzyon. Bu süreç "
  "tipik olarak <b>24–72 saat</b> sürer. Bu süre boyunca klinisyen, kanıta dayalı bir "
  "duyarlılık bilgisi olmadan ampirik (tahmini) ve çoğunlukla geniş-spektrumlu antibiyotik "
  "vermek zorunda kalır; bu da hem uygunsuz tedavi riskini hem de gereksiz geniş-spektrum "
  "kullanımını (yeni direncin başlıca sürücüsü) artırır.")

P("1.3. Genomik Fırsat ve Bu Çalışmanın Problemi", h2_s)
P("Tüm-genom dizilemenin (WGS) maliyeti büyük ölçüde düşmüş ve bir bakteri genomu saatler "
  "içinde elde edilebilir hale gelmiştir. Bir bakteri genomu, direnç genlerini (ör. "
  "beta-laktamazlar) ve direnci belirleyen kanonik nokta mutasyonlarını (ör. florokinolonlar "
  "için QRDR bölgesindeki <i>gyrA</i>/<i>parC</i> mutasyonları) barındırır. "
  "<b>Problem:</b> Bir bakteri izolatının genomundan elde edilen özelliklerden (AMR gen "
  "varlığı ve kanonik nokta mutasyonları) hareketle, izolatın belirli bir antibiyotiğe "
  "<b>Dirençli (R)</b> mi yoksa <b>Duyarlı (S)</b> mi olduğunu, klinik açıdan güvenli hata "
  "oranlarıyla makine öğrenmesi kullanarak tahmin etmek.")

P("1.4. Klinik Kısıtlar — Hata Türleri", h2_s)
P("Klinik AMR tahmininde tüm hatalar eşit değildir. İki kritik hata türü:", body_s)
B("<b>Çok Büyük Hata (VME — Very Major Error):</b> Gerçekte dirençli olan izolatı duyarlı "
  "tahmin etmek. Hasta etkisiz bir antibiyotik alır → <b>en tehlikeli hata.</b>")
B("<b>Büyük Hata (ME — Major Error):</b> Gerçekte duyarlı olanı dirençli tahmin etmek. "
  "Gereksiz geniş-spektrum antibiyotik kullanımı.")
P("FDA/CLSI araştırma-derecesi kabul eşikleri tipik olarak VME ≤ %3 ve ME ≤ %5'tir; "
  "operasyonel klinik kabul bundan daha geniş olabilir. VME ve ME bir tahterevallidir: "
  "tek bir karar eşiği bu ödünleşim üzerinde kayar ve modelin ROC/AUC eğrisinin kalitesi, "
  "iki hatanın <b>aynı anda</b> ne kadar düşürülebileceğinin tavanını belirler "
  "(her ikisini de ≤%10 yapmak yaklaşık AUC ≥ 0,95 gerektirir).")

SP(); HR()

# ════════════════════════════════════════════════════════════════ 2
P("2. Makale / Tez ve Literatür Taraması", h1_s)
P("Sistem, doğrudan AMR-ML literatürünün yerleşik çalışmalarına dayandırılmış ve "
  "özellikle Kim ve ark. (2022) derlemesinin metodolojik bölümlerine birebir hizalanmıştır.", body_s)

lit = [
    ["Çalışma", "Katkısı ve sisteme yansıması"],
    ["Kim, Maguire, Tsang ve ark. (2022), Clin Microbiol Rev 35(3)",
     "ANA ÇERÇEVE. ML-AMR'nin mevcut pratiği, sınırları ve klinik perspektifi: veri seti "
     "uygunluğu, genom/fenotip temsili, yorumlanabilir modeller için özellik seçimi, "
     "eğitim-test, kalibrasyon ve sınırlamalar. Pipeline'ın tüm aşamaları bu derlemenin "
     "bölümlerine atıfla tasarlandı."],
    ["Davis ve ark. (2016), Sci Rep 6:27930",
     "PATRIC/RAST üzerinde adaptive boosting ile ilk büyük ölçekli genom-bazlı AMR ML "
     "çalışması. Tür-spesifik k-mer + AMR metadata yaklaşımının temeli."],
    ["Nguyen ve ark. (2019), J Clin Microbiol 57(2)",
     "Salmonella için XGBoost ile log2(MIC) regresyonu; ±1 dilution içinde %95 doğruluk, "
     "ortalama VME=%2,7 ME=%0,1 (alan içi altın standart). MIC-temelli yaklaşım breakpoint "
     "değişimlerinden bağımsızdır."],
    ["Lees ve ark. (2023/24), Microbial Genomics",
     "Klebsiella pneumoniae MIC tahmini; Elastic Net + Random Forest + FaST-LMM, pan-genom "
     "(Panaroo) ve PopPUNK ile popülasyon (klonal) yapısı düzeltmesi. Coğrafi/klonal "
     "karıştırıcıların değerlendirmedeki kritik rolünü gösterir."],
    ["Aldred, Kerns, Osheroff (2014), Biochemistry 53(10)",
     "Kinolon etki ve direnç mekanizması; QRDR mutasyonları gyrA Ser83Leu ve parC Ser80Ile. "
     "Sistemde siprofloksasin tahmininin nedensel (mekanistik) temeli."],
    ["Hooper & Jacoby (2015)",
     "Florokinolon direnç mekanizmaları (hedef mutasyonu, efflux, plazmid qnr). "
     "P. aeruginosa'da gyrA T83I ve parC S87L — E. coli'den FARKLI pozisyonlar — "
     "tür-spesifik mutasyon haritalamasını gerektirir."],
    ["CLSI M100 / M52",
     "Klinik breakpoint'ler ve Essential Agreement (EA, ±1 dilution) / Categorical "
     "Agreement (CA, S-I-R kategori uyumu) tanımları."],
    ["Yöntemsel temeller",
     "Wolpert (1992) stacked generalization; Chawla ve ark. (2002) SMOTE ve sınırları; "
     "Niculescu-Mizil & Caruana (2005) olasılık kalibrasyonu (küçük örnekte isotonic "
     "yerine sigmoid); Lundberg & Lee (2017) SHAP ile yorumlanabilirlik."],
]
lt = Table([[Paragraph(c, cellb_s if i == 0 else (cell_s if j == 1 else cellb_s if False else cell_s))
             for j, c in enumerate(row)] for i, row in enumerate(lit)],
           colWidths=[52*mm, 116*mm])
lt.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1A5C44")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Arial-Bold"),
    ("FONTSIZE", (0, 0), (-1, 0), 9),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BBBBBB")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EEF5F1")]),
    ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
]))
# başlık hücrelerini beyaz yaz
lt._cellvalues[0] = [Paragraph(c, cellb_s) for c in lit[0]]
story.append(lt)

SP(); HR()

# ════════════════════════════════════════════════════════════════ 3
P("3. Veri Seti (Dataset)", h1_s)

P("3.1. Kaynaklar", h2_s)
B("<b>BV-BRC</b> (Bacterial and Viral Bioinformatics Resource Center; eski PATRIC): genom "
  "dizileri, kürasyonlu AMR özel-gen (specialty gene) verisi ve <i>genome_amr</i> fenotip "
  "tablosu (R/S ve MIC).")
B("<b>NCBI Pathogen Detection + AMRFinderPlus:</b> kanonik nokta mutasyonları "
  "(QRDR: gyrA, parC, grlA ve diğerleri), tür-spesifik olarak.")

P("3.2. Kapsam", h2_s)
B("<b>5 öncelikli patojen:</b> <i>Escherichia coli</i>, <i>Salmonella enterica</i>, "
  "<i>Klebsiella pneumoniae</i>, <i>Pseudomonas aeruginosa</i>, <i>Staphylococcus aureus</i>.")
B("<b>5 antibiyotik:</b> siprofloksasin (florokinolon), gentamisin (aminoglikozit), "
  "ampisilin ve seftazidim (beta-laktam), tetrasiklin.")
B("Yaklaşık <b>27.600 etiketli genom</b>; özellik seti tam olan ~<b>12.000 genom</b> "
  "modellemede kullanıldı. Toplam <b>~3.570 özellik</b>.")

P("3.3. Özellikler (X)", h2_s)
B("<b>AMR gen varlığı</b> (ikili): efflux pompaları, beta-laktamazlar (TEM, CTX-M, KPC, "
  "CMY), aminoglikozit-modifiye edici enzimler (AAC, ANT, APH) vb.")
B("<b>QRDR / Pathogen-Detection nokta mutasyonları:</b> tür-spesifik ön ek ile çakışma "
  "önlenir (pd_eco_, pd_sal_, pd_kle_, pd_pse_, pd_sta_).")
B("<b>Gen-gen etkileşim özellikleri:</b> biyolojik olarak bilinen çiftler + varyans-bazlı "
  "üst çiftler (epistasis uzayı).")

P("3.4. Etiketler (y) ve Veri Dürüstlüğü — Kritik Bulgu", h2_s)
P("Etiket, ilgili antibiyotik için <b>laboratuvarda ölçülmüş</b> R/S değeridir. Çalışmanın "
  "kritik metodolojik bulgusu şudur: BV-BRC <i>genome_amr</i> kayıtlarının yaklaşık "
  "<b>%92'si computational</b> (yani başka bir hesaplama modelinin <i>tahminidir</i>), "
  "yalnızca ~<b>%7,4'ü laboratuvar ölçümüdür</b>. Kayıtlar <i>evidence</i> alanına göre "
  "filtrelenerek SADECE laboratuvar ölçümleri kullanılmıştır. Aksi halde model 'başka bir "
  "modeli taklit eden' dairesel (circular) ve klinik açıdan geçersiz bir sisteme dönüşürdü.")

P("3.5. Veri Zorlukları", h2_s)
B("<b>Sınıf dengesizliği:</b> bazı tür×ilaç çiftlerinde dirençli (R) örnek kıttır "
  "(ör. Salmonella/gentamisin).")
B("<b>Coğrafi/klonal yapı:</b> rastgele bölme klonal sızıntıya (iyimser sonuç) yol açar; "
  "bu nedenle ülke-bazlı GroupKFold çapraz-doğrulama ile daha dürüst değerlendirme yapılır.")
B("<b>İçsel direnç ve klinik kullanılmama:</b> bazı tür×ilaç çiftleri hiç test edilmez "
  "(ör. Klebsiella ampisiline içsel dirençli) → teorik 5×5=25 ızgaradan klinik/biyolojik "
  "olarak anlamlı ~16 model kalır.")

SP(); HR()

# ════════════════════════════════════════════════════════════════ 4
P("4. Problem Hangi Sorunu Çözüyor?", h1_s)

P("4.1. Hızlı Klinik Karar Desteği", h2_s)
P("Kültür/antibiyogram sonucunu (24–72 saat) beklemeden, genomdan saatler içinde R/S "
  "tahmini üretir ve ampirik antibiyotik seçimini bilgilendirir; uygun tedaviye erken "
  "geçişi sağlar.")

P("4.2. Antibiyotik Yönetimi (Stewardship)", h2_s)
P("Erken duyarlılık öngörüsü, geniş-spektrumdan dar-spektrum tedaviye geçişe imkân tanır; "
  "böylece gereksiz geniş-spektrum kullanımı ve buna bağlı yeni direnç baskısı azalır.")

P("4.3. Halk Sağlığı Sürveyansı", h2_s)
P("Genom-bazlı, büyük ölçekli direnç izleme; AMR yayılımının takibi için bir altyapı sunar "
  "(Kim ve ark. 2022, Halk Sağlığı Sürveyansı bölümü).")

P("4.4. Yorumlanabilirlik (Klinik Güven)", h2_s)
P("SHAP analizi her tahminin biyolojik gerekçesini açığa çıkarır ve bunlar bilinen "
  "mekanizmalarla örtüşür: siprofloksasin → <i>gyrA</i>/<i>parC</i> mutasyonları; "
  "beta-laktamlar → TEM / CTX-M / CMY beta-laktamazları; S. aureus → <i>mecA</i> (PBP2a); "
  "gentamisin → AAC(3) / AAC(6'). Model bir kara-kutu değildir; çıktıları klinisyence "
  "doğrulanabilir.")

P("4.5. Klinik-Güvenli Karar Eşiği", h2_s)
P("VME (dirençliyi kaçırma) ile ME arasındaki ödünleşim, dengeli (Equal Error Rate) eşik "
  "seçimi ve olasılık kalibrasyonu ile yönetilir; böylece en tehlikeli hata olan VME "
  "denetim altında tutulur.")

P("4.6. Mevcut Başarı Durumu (büyütme öncesi)", h2_s)
P("Sistem şu an 16 (tür×antibiyotik) modeli kapsamaktadır. Çoğu model klinik-kullanım "
  "aralığındadır (VME yaklaşık %10–20). Öne çıkanlar:", body_s)
res = [
    ["Model", "VME", "ME", "AUC", "Durum"],
    ["S. aureus / siprofloksasin", "%9,8", "%8,5", "0,96", "Çok iyi"],
    ["K. pneumoniae / seftazidim", "%10,0", "%10,4", "0,96", "İyi"],
    ["K. pneumoniae / gentamisin", "%11,9", "%11,6", "0,96", "İyi"],
    ["Salmonella / siprofloksasin", "%12,0", "%12,0", "0,94", "İyi"],
    ["E. coli / siprofloksasin", "%16,1", "%14,6", "0,92", "Kullanılabilir"],
    ["P. aeruginosa / seftazidim", "%25,6", "%29,9", "0,79", "Zayıf (veri kısıtı)"],
]
rt = Table([[Paragraph(c, cellb_s) if i == 0 else Paragraph(c, cell_s)
             for c in row] for i, row in enumerate(res)],
           colWidths=[64*mm, 22*mm, 22*mm, 22*mm, 38*mm])
rt.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1A5C44")),
    ("FONTNAME", (0, 0), (-1, 0), "Arial-Bold"),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("ALIGN", (1, 0), (3, -1), "CENTER"),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BBBBBB")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EEF5F1")]),
    ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
]))
story.append(rt)
SP(4)
P("Sınırlamalar dürüstçe raporlanmıştır: <i>P. aeruginosa</i>/seftazidim gibi kıt-veri ve "
  "mekanizması karmaşık (AmpC dereprese olması, efflux, porin kaybı) çiftlerde AUC ve "
  "dolayısıyla hata oranları sınırlıdır; bu çiftler ancak gerçek-laboratuvar veri artışıyla "
  "iyileştirilebilir. Bu şeffaflık, klinik konuşlandırma için Kim ve ark. (2022) tarafından "
  "vurgulanan bir gerekliliktir.", body_s)

SP(10)
story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#1A5C44")))
P("Bu belge, projenin v36 (SMOTE-kapalı + adaptif kalibrasyon) sürümünün, zayıf-tür veri "
  "büyütmesi <b>uygulanmadan önceki</b> durumuna dayanmaktadır.", small_s)

# ── Footer (sayfa no) ─────────────────────────────────────────────────────────
def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Arial", 8)
    canvas.setFillColor(colors.HexColor("#888888"))
    canvas.drawCentredString(A4[0] / 2, 12 * mm, f"— {doc.page} —")
    canvas.drawString(20 * mm, 12 * mm, "AMR Tahmin Sistemi")
    canvas.drawRightString(A4[0] - 20 * mm, 12 * mm, "Kim ve ark. 2022 hizalı")
    canvas.restoreState()

OUT = "C:/Users/Casper/Desktop/Projeler/Bakteri_Analiz/AMR_Tez_Dokuman.pdf"
doc = SimpleDocTemplate(OUT, pagesize=A4,
                        leftMargin=20*mm, rightMargin=20*mm,
                        topMargin=18*mm, bottomMargin=20*mm,
                        title="AMR Tahmin Sistemi — Tez Dokümanı")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("OK ->", OUT, "|", os.path.getsize(OUT), "bytes")
