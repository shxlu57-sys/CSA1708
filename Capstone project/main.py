"""
AI-Powered Medical Report Analysis and Patient Health Insight System
CSA1708 - Artificial Intelligence | SIMATS Engineering

Flow (same as the project modules):
  Module 1 : Upload PDF / image  -> PyPDF2 / PyMuPDF + OpenCV + Tesseract OCR -> text
             text -> health values (regex) -> Pandas DataFrame
  Module 2 : Compare with reference ranges + Scikit-learn risk model
  Module 3 : Show simple, large-font, colour-coded insights in
             English / Telugu / Hindi / Tamil (for elderly patients)

Run:  python medical_report_assistant.py
"""

import os
import re
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, font as tkfont
from tkinter.scrolledtext import ScrolledText

import numpy as np
import pandas as pd

# ---- optional libraries (program still opens if one is missing) -------------
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None
try:
    import fitz  # PyMuPDF - renders scanned PDF pages to images (no poppler needed)
except ImportError:
    fitz = None
try:
    import cv2
except ImportError:
    cv2 = None
try:
    import pytesseract
    from PIL import Image
except ImportError:
    pytesseract = None
    Image = None
try:
    from sklearn.tree import DecisionTreeClassifier
except ImportError:
    DecisionTreeClassifier = None

# Windows users: set the path where Tesseract is installed
if sys.platform.startswith("win"):
    _tess = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if pytesseract and os.path.exists(_tess):
        pytesseract.pytesseract.tesseract_cmd = _tess


# =============================================================================
# 1. REFERENCE RANGES  (typical adult values - doctor's lab range always wins)
# =============================================================================
# lo/hi may be a single number or {"M": x, "F": y}.  only_low=True -> a value
# above range is NOT a problem (e.g. HDL "good" cholesterol).
PARAMS = [
    dict(key="hba1c", unit="%", lo=4.0, hi=5.6, plaus=(2, 20),
         aliases=[r"hba1c", r"glycosylated\s+h[a]?emoglobin", r"glycated\s+h[a]?emoglobin"]),
    dict(key="hemoglobin", unit="g/dL", lo={"M": 13.0, "F": 12.0}, hi={"M": 17.0, "F": 15.5},
         plaus=(2, 25), aliases=[r"h[a]?emoglobin", r"\bhb\b", r"\bhgb\b"]),
    dict(key="glucose", unit="mg/dL", lo=70, hi=100, plaus=(20, 800),
         aliases=[r"fasting\s+blood\s+sugar", r"fasting\s+(?:blood\s+)?glucose",
                  r"blood\s+glucose", r"blood\s+sugar", r"glucose", r"\bfbs\b"]),
    dict(key="ldl", unit="mg/dL", lo=0, hi=100, plaus=(10, 400),
         aliases=[r"\bldl\b(?:[\s\-]*cholesterol)?"]),
    dict(key="hdl", unit="mg/dL", lo=40, hi=1000, only_low=True, plaus=(5, 150),
         aliases=[r"\bhdl\b(?:[\s\-]*cholesterol)?"]),
    dict(key="triglycerides", unit="mg/dL", lo=0, hi=150, plaus=(10, 2000),
         aliases=[r"triglycerides?", r"\btg\b"]),
    dict(key="cholesterol", unit="mg/dL", lo=0, hi=200, plaus=(50, 600),
         aliases=[r"total\s+cholesterol", r"cholesterol[\s,\-]*total", r"cholesterol"]),
    dict(key="creatinine", unit="mg/dL", lo={"M": 0.7, "F": 0.6}, hi={"M": 1.3, "F": 1.1},
         plaus=(0.1, 15), aliases=[r"creatinine"]),
    dict(key="wbc", unit="cells/µL", lo=4000, hi=11000, plaus=(500, 200000),
         aliases=[r"\bwbc\b", r"total\s+(?:wbc|leu[ck]ocyte)\s*count", r"white\s+blood\s+cells?"]),
    dict(key="platelets", unit="cells/µL", lo=150000, hi=450000, plaus=(5000, 2000000),
         aliases=[r"platelets?(?:\s+count)?", r"\bplt\b"]),
]
BP_ROWS = [
    dict(key="systolic", unit="mmHg", lo=90, hi=120, plaus=(50, 260)),
    dict(key="diastolic", unit="mmHg", lo=60, hi=80, plaus=(30, 160)),
]
ALL_PARAMS = {p["key"]: p for p in PARAMS + BP_ROWS}


def _pick(v, sex):
    return v[sex] if isinstance(v, dict) else v


def _normalise_units(key, v):
    """Reports write counts in different styles (4.5 / 4500 / 2.5 lakh)."""
    if key == "wbc" and v < 100:
        return v * 1000
    if key == "platelets":
        if v < 20:
            return v * 100000      # 2.5 lakh
        if v < 1000:
            return v * 1000        # 250 (thousand)
    return v


# =============================================================================
# 2. MODULE 1 - FILE -> TEXT  (PDF / image / OCR)
# =============================================================================
def preprocess_image(path):
    """OpenCV clean-up so Tesseract reads scanned / phone-camera photos better."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Could not open the image.")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.fastNlMeansDenoising(gray, h=15)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 15)


def ocr_image_array(arr):
    return pytesseract.image_to_string(Image.fromarray(arr), config="--oem 3 --psm 6")


def extract_text(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        text = ""
        if PyPDF2:                                   # digital PDFs
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = "\n".join((pg.extract_text() or "") for pg in reader.pages)
        if len(text.strip()) < 30:                   # scanned PDF -> OCR each page
            if not (fitz and pytesseract and cv2):
                raise RuntimeError("Scanned PDF needs: pymupdf, opencv-python, pytesseract.")
            doc = fitz.open(path)
            pages = []
            for page in doc:
                pix = page.get_pixmap(dpi=250)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                if pix.n == 1:
                    gray = img[:, :, 0]
                else:
                    gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY if pix.n == 4 else cv2.COLOR_RGB2GRAY)
                bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 31, 15)
                pages.append(ocr_image_array(bw))
            text = "\n".join(pages)
        return text
    if ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"):
        if not (cv2 and pytesseract):
            raise RuntimeError("Images need: opencv-python and pytesseract (+ Tesseract installed).")
        return ocr_image_array(preprocess_image(path))
    if ext == ".txt":
        with open(path, encoding="utf-8", errors="ignore") as f:
            return f.read()
    raise ValueError("Please choose a PDF or an image (JPG / PNG).")


# =============================================================================
# 3. MODULE 1 (cont.) - TEXT -> HEALTH VALUES -> DataFrame
# =============================================================================
NUM = r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"


def extract_values(text):
    work = text.replace("|", " ")
    found = {}

    # blood pressure, e.g. "Blood Pressure : 150/95 mmHg"
    m = re.search(r"(?:blood\s+pressure|\bbp\b)[^0-9\n]{0,25}(\d{2,3})\s*/\s*(\d{2,3})", work, re.I)
    if m:
        s, d = float(m.group(1)), float(m.group(2))
        if BP_ROWS[0]["plaus"][0] <= s <= BP_ROWS[0]["plaus"][1]:
            found["systolic"], found["diastolic"] = s, d
        work = work[:m.start()] + " " * (m.end() - m.start()) + work[m.end():]

    for p in PARAMS:                       # order matters (HbA1c before Hb, LDL before cholesterol)
        for alias in p["aliases"]:
            m = re.search(alias + r"[^0-9\n]{0,30}" + NUM, work, re.I)
            if not m:
                continue
            v = float(m.group(m.lastindex).replace(",", ""))
            v = _normalise_units(p["key"], v)
            lo, hi = p["plaus"]
            if lo <= v <= hi:
                found[p["key"]] = v
                work = work[:m.start()] + " " * (m.end() - m.start()) + work[m.end():]
                break
    return found


def analyse(values, sex="M"):
    """Module 2 - compare with reference ranges. Returns a Pandas DataFrame."""
    rows = []
    for key, v in values.items():
        p = ALL_PARAMS[key]
        lo, hi = _pick(p["lo"], sex), _pick(p["hi"], sex)
        only_low = p.get("only_low", False)
        if v < lo:
            status, dev = "low", (lo - v) / lo
        elif v > hi and not only_low:
            status, dev = "high", (v - hi) / hi
        else:
            status, dev = "normal", 0.0
        rows.append(dict(key=key, value=v, unit=p["unit"], low=lo, high=hi,
                         status=status, deviation=round(dev, 3)))
    return pd.DataFrame(rows)


# ---- Scikit-learn risk model -------------------------------------------------
# DEMO model: trained on synthetic cases generated from the reference ranges.
# Replace with a real labelled dataset (e.g. from a hospital / Kaggle) for real use.
def _features(df):
    if df.empty:
        return [0, 0, 0, 0]
    n_bad = (df.status != "normal").sum()
    key_bad = df[(df.status != "normal") & df.key.isin(
        ["glucose", "hba1c", "cholesterol", "ldl", "systolic", "diastolic", "creatinine"])]
    return [n_bad / len(df), df.deviation.max(), df.deviation.mean(), len(key_bad)]


_risk_model = None


def _train_risk_model():
    global _risk_model
    if DecisionTreeClassifier is None:
        return None
    rng = np.random.default_rng(42)
    X, y = [], []
    for _ in range(1500):
        frac = rng.random()
        mx = rng.random() * 1.2 * frac
        mean = mx * rng.random()
        kb = int(rng.integers(0, 6) * frac)
        score = 2 * frac + 2 * mx + kb * 0.4
        X.append([frac, mx, mean, kb])
        y.append(0 if score < 0.7 else (1 if score < 2.0 else 2))
    _risk_model = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X, y)
    return _risk_model


def predict_risk(df):
    """0 = looks fine, 1 = needs attention, 2 = see a doctor soon"""
    if df.empty:
        return 0
    f = _features(df)
    model = _risk_model or _train_risk_model()
    if model is not None:
        return int(model.predict([f])[0])
    score = 2 * f[0] + 2 * f[1] + f[3] * 0.4          # fallback rule
    return 0 if score < 0.7 else (1 if score < 2.0 else 2)


# =============================================================================
# 4. MODULE 3 - SIMPLE LANGUAGE (English / Telugu / Hindi / Tamil)
# =============================================================================
LANGS = {"English": "en", "తెలుగు (Telugu)": "te", "हिन्दी (Hindi)": "hi", "தமிழ் (Tamil)": "ta"}

UI = {
 "en": dict(title="Easy Medical Report Reader", lang="Language", sex="Gender", male="Male", female="Female",
            upload="1. Choose Report (PDF / Photo)", analyze="2. Explain My Report", sample="Try Sample Report",
            save="Save Result", none_selected="No file chosen", working="Reading your report, please wait...",
            value="Your value", rng="Normal range", meaning="What is this?", normal="NORMAL", high="HIGH", low="LOW",
            low_msg="Lower than normal. Please talk to your doctor about this.",
            high_msg="Higher than normal. Please talk to your doctor about this.",
            ok_msg="This is in the normal range.",
            all_ok="Good news! All the values we could read are normal.",
            some_bad="{n} value(s) are outside the normal range. Please show this report to your doctor.",
            risk=["Overall: Looks fine", "Overall: Needs some attention", "Overall: Please see a doctor soon"],
            none_found="We could not find any health values. Please try a clearer PDF or photo.",
            disclaimer="This is an AI helper, not a doctor. Always confirm with your doctor.",
            summary="SUMMARY", details="DETAILS"),
 "te": dict(title="సులభ వైద్య నివేదిక చదువరి", lang="భాష", sex="లింగం", male="పురుషుడు", female="స్త్రీ",
            upload="1. నివేదికను ఎంచుకోండి (PDF / ఫోటో)", analyze="2. నా నివేదికను వివరించండి", sample="నమూనా నివేదిక",
            save="ఫలితం సేవ్ చేయండి", none_selected="ఫైల్ ఎంచుకోలేదు", working="మీ నివేదికను చదువుతున్నాం, దయచేసి వేచి ఉండండి...",
            value="మీ విలువ", rng="సాధారణ పరిధి", meaning="ఇది ఏమిటి?", normal="సాధారణం", high="ఎక్కువ", low="తక్కువ",
            low_msg="సాధారణం కంటే తక్కువగా ఉంది. దీని గురించి మీ డాక్టర్‌తో మాట్లాడండి.",
            high_msg="సాధారణం కంటే ఎక్కువగా ఉంది. దీని గురించి మీ డాక్టర్‌తో మాట్లాడండి.",
            ok_msg="ఇది సాధారణ పరిధిలో ఉంది.",
            all_ok="శుభవార్త! మేము చదవగలిగిన అన్ని విలువలు సాధారణంగా ఉన్నాయి.",
            some_bad="{n} విలువలు సాధారణ పరిధికి బయట ఉన్నాయి. దయచేసి ఈ నివేదికను మీ డాక్టర్‌కు చూపించండి.",
            risk=["మొత్తం మీద: బాగానే ఉంది", "మొత్తం మీద: కొంత శ్రద్ధ అవసరం", "మొత్తం మీద: త్వరగా డాక్టర్‌ను కలవండి"],
            none_found="ఆరోగ్య విలువలు కనుగొనలేకపోయాం. దయచేసి స్పష్టమైన PDF లేదా ఫోటో ప్రయత్నించండి.",
            disclaimer="ఇది AI సహాయకుడు మాత్రమే, డాక్టర్ కాదు. ఎల్లప్పుడూ మీ డాక్టర్‌ను సంప్రదించండి.",
            summary="సారాంశం", details="వివరాలు"),
 "hi": dict(title="आसान मेडिकल रिपोर्ट रीडर", lang="भाषा", sex="लिंग", male="पुरुष", female="महिला",
            upload="1. रिपोर्ट चुनें (PDF / फोटो)", analyze="2. मेरी रिपोर्ट समझाएं", sample="नमूना रिपोर्ट",
            save="नतीजा सेव करें", none_selected="कोई फ़ाइल नहीं चुनी", working="आपकी रिपोर्ट पढ़ी जा रही है, कृपया प्रतीक्षा करें...",
            value="आपका मान", rng="सामान्य सीमा", meaning="यह क्या है?", normal="सामान्य", high="ज्यादा", low="कम",
            low_msg="सामान्य से कम है। कृपया इस बारे में अपने डॉक्टर से बात करें।",
            high_msg="सामान्य से अधिक है। कृपया इस बारे में अपने डॉक्टर से बात करें।",
            ok_msg="यह सामान्य सीमा में है।",
            all_ok="अच्छी खबर! जो भी मान पढ़े गए वे सभी सामान्य हैं।",
            some_bad="{n} मान सामान्य सीमा से बाहर हैं। कृपया यह रिपोर्ट अपने डॉक्टर को दिखाएं।",
            risk=["कुल मिलाकर: ठीक है", "कुल मिलाकर: कुछ ध्यान देने की जरूरत", "कुल मिलाकर: जल्द डॉक्टर से मिलें"],
            none_found="कोई स्वास्थ्य मान नहीं मिला। कृपया साफ़ PDF या फोटो आज़माएं।",
            disclaimer="यह एक AI सहायक है, डॉक्टर नहीं। हमेशा अपने डॉक्टर से पुष्टि करें।",
            summary="सारांश", details="विवरण"),
 "ta": dict(title="எளிய மருத்துவ அறிக்கை வாசிப்பான்", lang="மொழி", sex="பாலினம்", male="ஆண்", female="பெண்",
            upload="1. அறிக்கையைத் தேர்ந்தெடுக்கவும் (PDF / படம்)", analyze="2. என் அறிக்கையை விளக்கவும்", sample="மாதிரி அறிக்கை",
            save="முடிவைச் சேமி", none_selected="கோப்பு தேர்ந்தெடுக்கவில்லை", working="உங்கள் அறிக்கையைப் படிக்கிறோம், காத்திருக்கவும்...",
            value="உங்கள் மதிப்பு", rng="இயல்பான அளவு", meaning="இது என்ன?", normal="இயல்பு", high="அதிகம்", low="குறைவு",
            low_msg="இயல்பை விட குறைவாக உள்ளது. இதைப் பற்றி உங்கள் மருத்துவரிடம் பேசுங்கள்.",
            high_msg="இயல்பை விட அதிகமாக உள்ளது. இதைப் பற்றி உங்கள் மருத்துவரிடம் பேசுங்கள்.",
            ok_msg="இது இயல்பான அளவில் உள்ளது.",
            all_ok="நல்ல செய்தி! படிக்க முடிந்த அனைத்து மதிப்புகளும் இயல்பாக உள்ளன.",
            some_bad="{n} மதிப்புகள் இயல்பான அளவுக்கு வெளியே உள்ளன. இந்த அறிக்கையை உங்கள் மருத்துவரிடம் காட்டுங்கள்.",
            risk=["மொத்தத்தில்: நன்றாக உள்ளது", "மொத்தத்தில்: கவனம் தேவை", "மொத்தத்தில்: விரைவில் மருத்துவரைப் பாருங்கள்"],
            none_found="உடல்நல மதிப்புகள் கிடைக்கவில்லை. தெளிவான PDF அல்லது படத்தை முயற்சிக்கவும்.",
            disclaimer="இது ஒரு AI உதவியாளர் மட்டுமே, மருத்துவர் அல்ல. எப்போதும் உங்கள் மருத்துவரிடம் உறுதிப்படுத்துங்கள்.",
            summary="சுருக்கம்", details="விவரங்கள்"),
}

# (name, plain-language meaning) per language
INFO = {
 "hemoglobin": {
  "en": ("Hemoglobin", "Carries oxygen in your blood. Low value can mean anaemia (weakness, tiredness)."),
  "te": ("హిమోగ్లోబిన్", "ఇది రక్తంలో ఆక్సిజన్‌ను మోస్తుంది. తక్కువైతే రక్తహీనత, నీరసం ఉండవచ్చు."),
  "hi": ("हीमोग्लोबिन", "यह खून में ऑक्सीजन ले जाता है। कम होने पर खून की कमी और कमजोरी हो सकती है।"),
  "ta": ("ஹீமோகுளோபின்", "இது இரத்தத்தில் ஆக்சிஜனை எடுத்துச் செல்கிறது. குறைந்தால் இரத்தசோகை, சோர்வு ஏற்படலாம்.")},
 "glucose": {
  "en": ("Blood Sugar (Fasting)", "Sugar level in your blood. High value may point towards diabetes."),
  "te": ("రక్తంలో చక్కెర (ఖాళీ కడుపుతో)", "రక్తంలో చక్కెర స్థాయి. ఎక్కువైతే మధుమేహం (షుగర్) సూచన కావచ్చు."),
  "hi": ("ब्लड शुगर (खाली पेट)", "खून में शक्कर की मात्रा। ज्यादा होने पर मधुमेह (डायबिटीज) का संकेत हो सकता है।"),
  "ta": ("இரத்த சர்க்கரை (வெறும் வயிறு)", "இரத்தத்தில் சர்க்கரையின் அளவு. அதிகமானால் நீரிழிவு நோயின் அறிகுறியாக இருக்கலாம்.")},
 "hba1c": {
  "en": ("HbA1c (3-month sugar average)", "Shows your average blood sugar over the last 3 months."),
  "te": ("HbA1c (3 నెలల సగటు చక్కెర)", "గత 3 నెలల్లో మీ రక్తంలో సగటు చక్కెర స్థాయిని చూపుతుంది."),
  "hi": ("HbA1c (3 महीने की औसत शक्कर)", "पिछले 3 महीनों में खून में शक्कर का औसत स्तर दिखाता है।"),
  "ta": ("HbA1c (3 மாத சராசரி சர்க்கரை)", "கடந்த 3 மாதங்களில் உங்கள் இரத்த சர்க்கரையின் சராசரி அளவைக் காட்டுகிறது.")},
 "cholesterol": {
  "en": ("Total Cholesterol", "Fat in your blood. High value can raise the risk of heart problems."),
  "te": ("మొత్తం కొలెస్ట్రాల్", "రక్తంలోని కొవ్వు. ఎక్కువైతే గుండె సమస్యల ప్రమాదం పెరుగుతుంది."),
  "hi": ("कुल कोलेस्ट्रॉल", "खून में चर्बी की मात्रा। ज्यादा होने पर दिल की बीमारी का खतरा बढ़ता है।"),
  "ta": ("மொத்த கொலஸ்ட்ரால்", "இரத்தத்தில் உள்ள கொழுப்பு. அதிகமானால் இதய நோய் அபாயம் கூடும்.")},
 "ldl": {
  "en": ("LDL (bad cholesterol)", "The 'bad' cholesterol that can block blood vessels when high."),
  "te": ("LDL (చెడు కొలెస్ట్రాల్)", "ఎక్కువైతే రక్తనాళాల్లో అడ్డంకులు కలిగించే 'చెడు' కొలెస్ట్రాల్."),
  "hi": ("LDL (खराब कोलेस्ट्रॉल)", "ज्यादा होने पर नसों में रुकावट पैदा कर सकता है।"),
  "ta": ("LDL (கெட்ட கொலஸ்ட்ரால்)", "அதிகமானால் இரத்தக் குழாய்களில் அடைப்பை ஏற்படுத்தலாம்.")},
 "hdl": {
  "en": ("HDL (good cholesterol)", "The 'good' cholesterol that protects your heart. Higher is better."),
  "te": ("HDL (మంచి కొలెస్ట్రాల్)", "గుండెను కాపాడే 'మంచి' కొలెస్ట్రాల్. ఎక్కువగా ఉంటే మంచిది."),
  "hi": ("HDL (अच्छा कोलेस्ट्रॉल)", "दिल की रक्षा करने वाला कोलेस्ट्रॉल। ज्यादा होना अच्छा है।"),
  "ta": ("HDL (நல்ல கொலஸ்ட்ரால்)", "இதயத்தைப் பாதுகாக்கும் கொலஸ்ட்ரால். அதிகமாக இருப்பது நல்லது.")},
 "triglycerides": {
  "en": ("Triglycerides", "Another type of blood fat. High value is linked to heart and sugar problems."),
  "te": ("ట్రైగ్లిజరైడ్స్", "రక్తంలోని మరో రకమైన కొవ్వు. ఎక్కువైతే గుండె, చక్కెర సమస్యలతో సంబంధం ఉంటుంది."),
  "hi": ("ट्राइग्लिसराइड्स", "खून में एक और तरह की चर्बी। ज्यादा होने पर दिल और शुगर की समस्या का खतरा रहता है।"),
  "ta": ("டிரைகிளிசரைடுகள்", "இரத்தத்தில் உள்ள மற்றொரு வகை கொழுப்பு. அதிகமானால் இதயம், சர்க்கரை பிரச்சினைகளுடன் தொடர்புடையது.")},
 "systolic": {
  "en": ("Blood Pressure - upper (Systolic)", "Pressure when the heart pumps. High value means high BP."),
  "te": ("రక్తపోటు – పై విలువ (సిస్టోలిక్)", "గుండె కొట్టుకున్నప్పుడు ఉండే ఒత్తిడి. ఎక్కువైతే హై బీపీ."),
  "hi": ("ब्लड प्रेशर – ऊपर का (सिस्टोलिक)", "दिल के धड़कने पर दबाव। ज्यादा होने पर हाई बीपी है।"),
  "ta": ("இரத்த அழுத்தம் – மேல் எண் (சிஸ்டாலிக்)", "இதயம் துடிக்கும்போது உள்ள அழுத்தம். அதிகமானால் உயர் இரத்த அழுத்தம்.")},
 "diastolic": {
  "en": ("Blood Pressure - lower (Diastolic)", "Pressure when the heart rests between beats."),
  "te": ("రక్తపోటు – కింది విలువ (డయాస్టోలిక్)", "గుండె కొట్టుకునే మధ్య విశ్రాంతి సమయంలో ఉండే ఒత్తిడి."),
  "hi": ("ब्लड प्रेशर – नीचे का (डायस्टोलिक)", "धड़कनों के बीच दिल के आराम के समय का दबाव।"),
  "ta": ("இரத்த அழுத்தம் – கீழ் எண் (டயஸ்டாலிக்)", "இதயம் துடிப்புகளுக்கு இடையே ஓய்வெடுக்கும்போது உள்ள அழுத்தம்.")},
 "creatinine": {
  "en": ("Creatinine", "Shows how well your kidneys are working."),
  "te": ("క్రియాటినిన్", "మీ కిడ్నీలు ఎంత బాగా పనిచేస్తున్నాయో చూపుతుంది."),
  "hi": ("क्रिएटिनिन", "बताता है कि आपकी किडनी कितनी ठीक काम कर रही है।"),
  "ta": ("கிரியாட்டினின்", "உங்கள் சிறுநீரகங்கள் எவ்வளவு நன்றாக வேலை செய்கின்றன என்பதைக் காட்டுகிறது.")},
 "wbc": {
  "en": ("White Blood Cells (WBC)", "Cells that fight infection. High value may mean an infection."),
  "te": ("తెల్ల రక్త కణాలు (WBC)", "ఇన్ఫెక్షన్‌తో పోరాడే కణాలు. ఎక్కువైతే ఇన్ఫెక్షన్ ఉండవచ్చు."),
  "hi": ("सफेद रक्त कोशिकाएं (WBC)", "संक्रमण से लड़ने वाली कोशिकाएं। ज्यादा होने पर संक्रमण हो सकता है।"),
  "ta": ("வெள்ளை இரத்த அணுக்கள் (WBC)", "நோய்த்தொற்றுடன் போராடும் செல்கள். அதிகமானால் தொற்று இருக்கலாம்.")},
 "platelets": {
  "en": ("Platelets", "Help your blood to clot. Low value can cause easy bleeding."),
  "te": ("ప్లేట్‌లెట్స్", "రక్తం గడ్డకట్టడానికి సహాయపడతాయి. తక్కువైతే రక్తస్రావం సులభంగా అవుతుంది."),
  "hi": ("प्लेटलेट्स", "खून जमने में मदद करते हैं। कम होने पर आसानी से खून बह सकता है।"),
  "ta": ("பிளேட்லெட்டுகள்", "இரத்தம் உறைய உதவுகின்றன. குறைந்தால் எளிதில் இரத்தப்போக்கு ஏற்படலாம்.")},
}

SAMPLE_REPORT = """
CITY CARE DIAGNOSTICS - PATIENT REPORT
Hemoglobin            10.2   g/dL     13.0 - 17.0
HbA1c                 7.1    %        4.0 - 5.6
Fasting Blood Sugar   132    mg/dL    70 - 100
Total Cholesterol     228    mg/dL    < 200
LDL Cholesterol       148    mg/dL    < 100
HDL Cholesterol       46     mg/dL    > 40
Triglycerides         160    mg/dL    < 150
Serum Creatinine      0.9    mg/dL    0.7 - 1.3
WBC Count             7,200  cells/cumm
Platelet Count        2.5    lakhs/cumm
Blood Pressure        150/95 mmHg
"""


def build_report(df, lang, risk):
    """Return a list of (text, tag) ready to print - also used for 'Save'."""
    t = UI[lang]
    out = [(t["summary"] + "\n", "h1")]
    if df.empty:
        out.append((t["none_found"] + "\n", "bad"))
        return out
    n_bad = int((df.status != "normal").sum())
    out.append((t["all_ok"] + "\n" if n_bad == 0 else t["some_bad"].format(n=n_bad) + "\n",
                "good" if n_bad == 0 else "bad"))
    out.append((t["risk"][risk] + "\n\n", ["good", "warn", "bad"][risk]))
    out.append((t["details"] + "\n", "h1"))
    order = {"high": 0, "low": 0, "normal": 1}
    for _, r in df.sort_values("status", key=lambda s: s.map(order)).iterrows():
        name, meaning = INFO[r.key][lang]
        tag = {"normal": "good", "high": "bad", "low": "warn"}[r.status]
        arrow = {"normal": "✔", "high": "▲", "low": "▼"}[r.status]
        val = f"{r.value:,.1f}".rstrip("0").rstrip(".") if r.value < 1000 else f"{r.value:,.0f}"
        out.append((f"\n{arrow} {name}\n", "name"))
        out.append((f"   {t['value']}: {val} {r.unit}   →   {t[r.status]}\n", tag))
        lo, hi = r.low, r.high
        rng = f"{lo:g} – {hi:g}" if hi < 1000 and lo > 0 else (f"< {hi:g}" if lo == 0 else f"> {lo:g}")
        out.append((f"   {t['rng']}: {rng} {r.unit}\n", "plain"))
        out.append((f"   {t['meaning']} {meaning}\n", "plain"))
        msg = {"normal": t["ok_msg"], "high": t["high_msg"], "low": t["low_msg"]}[r.status]
        out.append((f"   {msg}\n", tag))
    out.append(("\n" + t["disclaimer"] + "\n", "small"))
    return out


# =============================================================================
# 5. GUI (Tkinter)  - big fonts, high contrast, language switch
# =============================================================================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.geometry("980x780")
        self.configure(bg="#F4F8FB")
        self.path = None
        self.df = pd.DataFrame()
        self.risk = 0
        self.font_size = 16
        fams = set(tkfont.families())
        self.family = next((f for f in ("Nirmala UI", "Noto Sans Telugu", "Noto Sans", "Latha",
                                        "Arial Unicode MS", "Segoe UI") if f in fams), "TkDefaultFont")

        self.lang_var = tk.StringVar(value="English")
        self.sex_var = tk.StringVar(value="M")

        bar = tk.Frame(self, bg="#0B5394")
        bar.pack(fill="x")
        self.title_lbl = tk.Label(bar, fg="white", bg="#0B5394", font=(self.family, 20, "bold"), pady=10)
        self.title_lbl.pack()

        ctl = tk.Frame(self, bg="#F4F8FB", pady=8)
        ctl.pack(fill="x", padx=12)
        self.lang_lbl = tk.Label(ctl, bg="#F4F8FB", font=(self.family, 14, "bold"))
        self.lang_lbl.grid(row=0, column=0, padx=4)
        cb = ttk.Combobox(ctl, textvariable=self.lang_var, values=list(LANGS), state="readonly",
                          width=16, font=(self.family, 14))
        cb.grid(row=0, column=1, padx=4)
        cb.bind("<<ComboboxSelected>>", lambda e: self.refresh())
        self.sex_lbl = tk.Label(ctl, bg="#F4F8FB", font=(self.family, 14, "bold"))
        self.sex_lbl.grid(row=0, column=2, padx=(16, 4))
        self.rb_m = tk.Radiobutton(ctl, variable=self.sex_var, value="M", bg="#F4F8FB",
                                   font=(self.family, 14), command=self.reanalyse)
        self.rb_f = tk.Radiobutton(ctl, variable=self.sex_var, value="F", bg="#F4F8FB",
                                   font=(self.family, 14), command=self.reanalyse)
        self.rb_m.grid(row=0, column=3)
        self.rb_f.grid(row=0, column=4)
        tk.Button(ctl, text="A+", font=(self.family, 12, "bold"),
                  command=lambda: self.zoom(2)).grid(row=0, column=5, padx=(24, 2))
        tk.Button(ctl, text="A−", font=(self.family, 12, "bold"),
                  command=lambda: self.zoom(-2)).grid(row=0, column=6)

        btns = tk.Frame(self, bg="#F4F8FB")
        btns.pack(fill="x", padx=12, pady=4)
        mk = lambda c: dict(font=(self.family, 14, "bold"), fg="white", bg=c, activebackground=c,
                            relief="flat", padx=14, pady=8, cursor="hand2")
        self.b_up = tk.Button(btns, command=self.choose_file, **mk("#1B7F3B"))
        self.b_an = tk.Button(btns, command=self.run_analysis, **mk("#0B5394"))
        self.b_sm = tk.Button(btns, command=self.use_sample, **mk("#6A6A6A"))
        self.b_sv = tk.Button(btns, command=self.save_result, **mk("#8E5B00"))
        for b in (self.b_up, self.b_an, self.b_sm, self.b_sv):
            b.pack(side="left", padx=4)
        self.file_lbl = tk.Label(self, bg="#F4F8FB", fg="#444", font=(self.family, 12), anchor="w")
        self.file_lbl.pack(fill="x", padx=16)

        self.box = ScrolledText(self, wrap="word", padx=16, pady=12, relief="solid", bd=1)
        self.box.pack(fill="both", expand=True, padx=12, pady=10)
        self.box.configure(state="disabled")
        self.apply_fonts()
        self.refresh()

    # ---- helpers -----------------------------------------------------------
    @property
    def lang(self):
        return LANGS[self.lang_var.get()]

    def apply_fonts(self):
        s, f = self.font_size, self.family
        self.box.configure(font=(f, s))
        self.box.tag_configure("h1", font=(f, s + 6, "bold"), foreground="#0B5394", spacing3=6)
        self.box.tag_configure("name", font=(f, s + 2, "bold"), foreground="#111111")
        self.box.tag_configure("plain", font=(f, s), foreground="#333333")
        self.box.tag_configure("small", font=(f, s - 3, "italic"), foreground="#666666")
        self.box.tag_configure("good", font=(f, s, "bold"), foreground="#1B7F3B")
        self.box.tag_configure("warn", font=(f, s, "bold"), foreground="#C26A00")
        self.box.tag_configure("bad", font=(f, s, "bold"), foreground="#C62828")

    def zoom(self, d):
        self.font_size = max(11, min(34, self.font_size + d))
        self.apply_fonts()

    def show(self, items):
        self.box.configure(state="normal")
        self.box.delete("1.0", "end")
        for text, tag in items:
            self.box.insert("end", text, tag)
        self.box.configure(state="disabled")

    def refresh(self):
        t = UI[self.lang]
        self.title("AI Medical Report Analysis")
        self.title_lbl.config(text="🩺  " + t["title"])
        self.lang_lbl.config(text=t["lang"] + ":")
        self.sex_lbl.config(text=t["sex"] + ":")
        self.rb_m.config(text=t["male"])
        self.rb_f.config(text=t["female"])
        self.b_up.config(text=t["upload"])
        self.b_an.config(text=t["analyze"])
        self.b_sm.config(text=t["sample"])
        self.b_sv.config(text=t["save"])
        self.file_lbl.config(text=os.path.basename(self.path) if self.path else t["none_selected"])
        if self.df.empty and not self.path:
            self.show([(t["disclaimer"] + "\n", "small")])
        else:
            self.show(build_report(self.df, self.lang, self.risk))

    # ---- actions -----------------------------------------------------------
    def choose_file(self):
        p = filedialog.askopenfilename(title="Medical report",
                                       filetypes=[("Reports", "*.pdf *.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                                                  ("All files", "*.*")])
        if p:
            self.path = p
            self.df = pd.DataFrame()
            self.refresh()

    def use_sample(self):
        self.path = "sample_report.txt"
        self._process_text(SAMPLE_REPORT)

    def run_analysis(self):
        t = UI[self.lang]
        if not self.path or not os.path.exists(self.path):
            self.choose_file()
            if not self.path:
                return
        self.show([(t["working"] + "\n", "h1")])
        self.update_idletasks()
        try:
            text = extract_text(self.path)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.refresh()
            return
        self._process_text(text)

    def _process_text(self, text):
        self.values = extract_values(text)
        self.reanalyse()

    def reanalyse(self):
        if not getattr(self, "values", None) and not self.path:
            return
        self.df = analyse(getattr(self, "values", {}), self.sex_var.get())
        self.risk = predict_risk(self.df)
        self.refresh()

    def save_result(self):
        if self.df.empty:
            return
        p = filedialog.asksaveasfilename(defaultextension=".txt",
                                         filetypes=[("Text", "*.txt"), ("CSV table", "*.csv")])
        if not p:
            return
        if p.lower().endswith(".csv"):
            self.df.to_csv(p, index=False, encoding="utf-8-sig")
        else:
            with open(p, "w", encoding="utf-8") as f:
                f.write("".join(txt for txt, _ in build_report(self.df, self.lang, self.risk)))


if __name__ == "__main__":
    App().mainloop()
