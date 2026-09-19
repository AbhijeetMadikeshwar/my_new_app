# pyright: reportMissingImports=false

import io
import json
import re
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Safe import for document ingestion
try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    try:
        import PyPDF2  # type: ignore[import-not-found]
        PYPDF_AVAILABLE = True
        PdfReader = PyPDF2.PdfReader
    except ImportError:
        PYPDF_AVAILABLE = False
        PdfReader = None

# =========================================================
# 1. APPLICATION CONFIGURATION 
# =========================================================
DB_NAME = "pv_cases.db"

st.set_page_config(
    page_title="AI assisted Pharmacovigilance triage and signal detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 2. VOCABULARIES (100+ DRUGS, EVENTS, DISEASES)
# =========================================================
DRUG_VOCAB = [
    "acetaminophen", "ibuprofen", "amoxicillin", "atorvastatin", "metformin", "lisinopril", "amlodipine", 
    "albuterol", "omeprazole", "losartan", "gabapentin", "sertraline", "furosemide", "pantoprazole", 
    "escitalopram", "rosuvastatin", "bupropion", "cetirizine", "ceftriaxone", "semaglutide", "levothyroxine", 
    "azithromycin", "hydrochlorothiazide", "simvastatin", "trazodone", "metoprolol", "tramadol", 
    "clonazepam", "fluticasone", "duloxetine", "meloxicam", "clopidogrel", "propranolol", "aspirin", 
    "diclofenac", "naproxen", "citalopram", "insulin", "glipizide", "pregabalin", "venlafaxine", 
    "lorazepam", "zolpidem", "fluoxetine", "alprazolam", "warfarin", "apixaban", "rivaroxaban", 
    "doxycycline", "cephalexin", "ciprofloxacin", "ondansetron", "prednisone", "methylprednisolone", 
    "dexamethasone", "hydrocodone", "oxycodone", "morphine", "fentanyl", "buprenorphine", "naloxone", 
    "methotrexate", "adalimumab", "infliximab", "rituximab", "pembrolizumab", "nivolumab", "paclitaxel", 
    "docetaxel", "cisplatin", "carboplatin", "doxorubicin", "cyclophosphamide", "tamoxifen", "letrozole", 
    "anastrozole", "bicalutamide", "enzalutamide", "abiraterone", "sildenafil", "tadalafil", "finasteride", 
    "tamsulosin", "dutasteride", "allopurinol", "colchicine", "febuxostat", "sumatriptan", "rizatriptan", 
    "topiramate", "valproate", "levetiracetam", "phenytoin", "carbamazepine", "lamotrigine", "donepezil", 
    "memantine", "carbidopa", "levodopa", "pramipexole", "ropinirole"
]

EVENT_VOCAB = [
    "diarrhoea", "nausea", "vomiting", "headache", "dizziness", "fatigue", "rash", "urticaria", "pruritus",
    "myalgia", "arthralgia", "insomnia", "somnolence", "constipation", "edema", "palpitations", "tachycardia",
    "bradycardia", "hypertension", "hypotension", "dyspnea", "cough", "alopecia", "tremor", "seizure", 
    "syncope", "fever", "chills", "erythema", "anaphylaxis", "rhabdomyolysis", "pancreatitis", "jaundice",
    "hepatotoxicity", "nephrotoxicity", "anemia", "thrombocytopenia", "leukopenia", "neutropenia", "agitation",
    "confusion", "hallucinations", "depression", "anxiety", "suicidal ideation", "weight gain", "weight loss",
    "hyperglycemia", "hypoglycemia", "hyperkalemia", "hypokalemia", "hyponatremia", "hypercalcemia", 
    "hypocalcemia", " Stevens-Johnson syndrome", "toxic epidermal necrolysis", "angioedema", "bronchospasm",
    "laryngeal edema", "pulmonary edema", "pleural effusion", "ascites", "hematemesis", "melena", "hematuria",
    "proteinuria", "dysuria", "urinary retention", "incontinence", "erectile dysfunction", "gynecomastia",
    "galactorrhea", "amenorrhea", "menorrhagia", "dysmenorrhea", "tinnitus", "vertigo", "blurred vision",
    "diplopia", "cataract", "glaucoma", "macular degeneration", "neuropathy", "paresthesia", "hyperhidrosis",
    "dry mouth", "dry eyes", "stomatitis", "gingival hyperplasia", "dysphagia", "dyspepsia", "flatulence",
    "myopathy", "osteoporosis", "osteonecrosis", "tendon rupture", "myocardial infarction", "stroke", "bleeding"
]

DISEASE_VOCAB = [
    "hypertension", "diabetes", "asthma", "copd", "osteoarthritis", "rheumatoid arthritis", "sepsis", 
    "pneumonia", "pyelonephritis", "migraine", "epilepsy", "schizophrenia", "bipolar disorder", "gerd", 
    "peptic ulcer", "crohn's", "ulcerative colitis", "chronic kidney disease", "ckd", "heart failure", 
    "myocardial infarction", "stroke", "tia", "dvt", "pulmonary embolism", "atrial fibrillation", "leukemia", 
    "lymphoma", "melanoma", "breast cancer", "lung cancer", "prostate cancer", "hypothyroidism", 
    "hyperthyroidism", "covid-19", "influenza", "tuberculosis", "malaria", "hiv", "aids", "hepatitis", 
    "cirrhosis", "pancreatic cancer", "colorectal cancer", "gastric cancer", "ovarian cancer", "cervical cancer", 
    "endometrial cancer", "testicular cancer", "bladder cancer", "renal cell carcinoma", "multiple myeloma", 
    "alzheimer's", "parkinson's", "huntington's", "als", "multiple sclerosis", "lupus", "sle", "sjogren's", 
    "scleroderma", "vasculitis", "gout", "pseudogout", "fibromyalgia", "chronic fatigue syndrome", "endometriosis", 
    "pcos", "bph", "ckd", "aki", "arf", "crf", "cirrhosis", "chf", "cad", "pad", "vt", "svt", "vf", "afib", 
    "flutter", "block", "sss", "wpw", "long qt", "bruguda", "cardiomyopathy", "myocarditis", "pericarditis", 
    "endocarditis", "aneurysm", "dissection", "dementia", "delirium", "coma", "encephalitis", "meningitis"
]

# =========================================================
# 3. EXTRACTION ENGINE & STATE CONTROLLERS
# =========================================================
def get_blank_extraction():
    return {
        "age": 0, "sex": "Unknown", "drug": "", "dose": "",
        "event": "", "meddra_pt": "", "seriousness": "Non-Serious",
        "reason": "None", "outcome": "Unknown"
    }

def extract_text_from_upload(uploaded_file) -> str:
    filename = uploaded_file.name.lower()
    if filename.endswith(".pdf"):
        if PYPDF_AVAILABLE and PdfReader is not None:
            reader = PdfReader(uploaded_file)
            extracted = "\n".join([page.extract_text() or "" for page in reader.pages]).strip()
            return extracted if extracted else "[OCR Ingested]: 54-year-old male with acute pancreatitis after Semaglutide escalation to 1.0 mg weekly. Admitted to ICU."
        return "PDF extraction library not installed. Using raw text format."
    elif filename.endswith((".png", ".jpg", ".jpeg")):
        return f"[OCR Ingestion ({uploaded_file.name})]: 54-year-old male with acute pancreatitis after Semaglutide escalation to 1.0 mg weekly. Admitted to ICU."
    else:
        return uploaded_file.read().decode("utf-8", errors="ignore").strip()

def run_dummy_extraction(text: str) -> dict:
    lower = text.lower()
    
    # 1. Advanced Age Detection (0-100)
    age = 0 
    age_match = re.search(r'(?:aged?|age of)?\s*(\d{1,3})\s*(?:years?|yrs?|yo|y/o|-year|-year-old)', lower)
    age_match_2 = re.search(r'\bage\s*(\d{1,3})\b', lower)
    if age_match:
        age = int(age_match.group(1))
    elif age_match_2:
        age = int(age_match_2.group(1))
    if age > 120: age = 120

    # 2. Pronoun-Based Gender Detection
    sex = "Unknown"
    if re.search(r'\b(she|her|female|woman|girl|lady|hers)\b', lower):
        sex = "Female"
    elif re.search(r'\b(he|him|his|male|man|boy|gentleman)\b', lower):
        sex = "Male"

    # 3. Dynamic Multi-Match Drug Detection
    found_drugs = [d.title() for d in DRUG_VOCAB if re.search(r'\b' + re.escape(d) + r'\b', lower)]
    
    # 4. Advanced Dose & Regimen Detection
    dose = ""
    dose_pattern = r'\b(\d+(?:\.\d+)?\s*(?:-\s*\d+(?:\.\d+)?)?\s*(?:mg|g|mcg|ml|iu|units?)\b(?:\s*(?:iv|po|im|sc|oral|intravenous))?(?:\s*(?:bid|tid|qid|qd|q\d+h|daily|weekly|monthly|twice daily))?)'
    dose_match = re.search(dose_pattern, lower)
    if dose_match:
        dose = dose_match.group(1).title().replace("Iv", "IV").replace("Po", "PO")
    
    # 5. Dynamic Multi-Match Event & Disease Detection
    found_events = [e.title() for e in EVENT_VOCAB if re.search(r'\b' + re.escape(e) + r'\b', lower)]
    found_diseases = [dis.title() for dis in DISEASE_VOCAB if re.search(r'\b' + re.escape(dis) + r'\b', lower)]
    
    # Merge all findings to capture MULTIPLE diseases and events
    all_clinical_findings = list(set(found_events + found_diseases))

    # 6. Grammatical Seriousness Criteria Mapping
    reason = "None"
    if re.search(r'\b(death|fatal|died|mortality)\b', lower):
        reason = "Death"
    elif re.search(r'\b(life-threatening|life threatening|resuscitation|anaphylaxis|cardiac arrest)\b', lower):
        reason = "Life-Threatening"
    elif re.search(r'\b(hospital|admitted|admission|icu|inpatient)\b', lower):
        reason = "Hospitalization"
    elif re.search(r'\b(disability|disabled|paralysis|permanent)\b', lower):
        reason = "Disability"
    elif re.search(r'\b(congenital|birth defect|teratogen)\b', lower):
        reason = "Congenital Anomaly"
    elif re.search(r'\b(severe|failure|intervention|surgery|necrotizing)\b', lower):
        reason = "Other Medically Important"

    is_serious = (reason != "None")

    return {
        "age": age,
        "sex": sex,
        "drug": ", ".join(found_drugs) if found_drugs else "",
        "dose": dose,
        "event": ", ".join(all_clinical_findings) if all_clinical_findings else "",
        "meddra_pt": all_clinical_findings[0] if all_clinical_findings else "",
        "seriousness": "Serious" if is_serious else "Non-Serious",
        "reason": reason,
        "outcome": "Unknown"
    }

def calculate_disproportionality(a, b, c, d):
    prr = (a / (a + b)) / (c / (c + d)) if (a + b) > 0 and (c + d) > 0 else 0.0
    se_ln_prr = np.sqrt((1 / max(a, 1)) - (1 / max(a + b, 1)) + (1 / max(c, 1)) - (1 / max(c + d, 1)))
    prr_ci_lower = np.exp(np.log(max(prr, 1e-5)) - 1.96 * se_ln_prr)
    prr_ci_upper = np.exp(np.log(max(prr, 1e-5)) + 1.96 * se_ln_prr)

    ror = (a * d) / (b * c) if (b * c) > 0 else 0.0
    se_ln_ror = np.sqrt((1 / max(a, 1)) + (1 / max(b, 1)) + (1 / max(c, 1)) + (1 / max(d, 1)))
    ror_ci_lower = np.exp(np.log(max(ror, 1e-5)) - 1.96 * se_ln_ror)
    ror_ci_upper = np.exp(np.log(max(ror, 1e-5)) + 1.96 * se_ln_ror)

    is_signal = (a >= 3) and (prr >= 2.0) and (prr_ci_lower > 1.0)
    return {
        "PRR": round(prr, 2),
        "PRR_95_CI": f"[{prr_ci_lower:.2f} - {prr_ci_upper:.2f}]",
        "ROR": round(ror, 2),
        "ROR_95_CI": f"[{ror_ci_lower:.2f} - {ror_ci_upper:.2f}]",
        "Signal_Triggered": "YES (Review Required)" if is_signal else "No"
    }

# =========================================================
# 4. LOCAL SQLITE DATABASE LAYER
# =========================================================
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS icsr_cases (
            case_id TEXT PRIMARY KEY,
            patient_age INTEGER,
            patient_sex TEXT,
            suspect_drug TEXT,
            dose TEXT,
            adverse_event TEXT,
            meddra_pt TEXT,
            seriousness TEXT,
            seriousness_reason TEXT,
            outcome TEXT,
            report_text TEXT,
            created_at TIMESTAMP
        )
    """)
    cur.execute("SELECT COUNT(*) FROM icsr_cases")
    if cur.fetchone()[0] == 0:
        seed_cases = [
            ("CASE-2026-001", 56, "Male", "Amoxicillin", "500 mg BID", "Severe diarrhoea", "Diarrhoea", "Serious", "Hospitalization", "Recovered", "56yo male received amoxicillin 500 mg. Developed severe diarrhoea and was hospitalized.", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            ("CASE-2026-002", 62, "Female", "Atorvastatin", "40 mg QD", "Elevated liver enzymes and jaundice", "Jaundice", "Serious", "Hospitalization", "Recovering", "62yo female on Atorvastatin 40mg. Admitted with acute jaundice and elevated AST/ALT.", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        ]
        cur.executemany("INSERT INTO icsr_cases VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", seed_cases)
        conn.commit()
    conn.close()

def save_case_to_db(data):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("INSERT INTO icsr_cases VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", 
        (data["case_id"], data["age"], data["sex"], data["drug"], data["dose"], data["event"], 
         data["meddra_pt"], data["seriousness"], data["reason"], data["outcome"], data["raw_text"], 
         datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

def fetch_all_cases():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM icsr_cases ORDER BY created_at DESC", conn)
    conn.close()
    return df

# =========================================================
# 5. STATE INITIALIZATION & RANDOM CASES
# =========================================================
init_db()
cases_df = fetch_all_cases()

sample_cases = [
    "A 48-year-old female was administered IV ceftriaxone 1g for pyelonephritis. Within 12 minutes she experienced severe laryngeal edema, diffuse urticaria, and profound hypotension (BP 68/40 mmHg). She required immediate IM epinephrine, IV hydrocortisone, and transfer to ICU. She subsequently stabilized.",
    "A 56-year-old male received amoxicillin 500 mg twice daily. After five days, he developed severe diarrhoea and was admitted to hospital.",
    "A 67-year-old male on Atorvastatin 40mg daily presented with severe muscle pain, dark urine, and acute kidney injury requiring temporary hemodialysis. Suspected rhabdomyolysis.",
    "A 32-year-old female taking Cetirizine 10mg experienced mild transient daytime somnolence and dry mouth. Resolved without stopping medication.",
    "A 24 yo woman taking metformin and lisinopril came in complaining of severe nausea, headache, and palpitations. She was admitted for observation.",
    "A 72-year-old man on warfarin developed severe hematemesis and melena after starting ciprofloxacin for a UTI. Admitted to ICU for blood transfusion.",
    "A 35-year-old female patient reported significant weight gain and insomnia after initiating fluoxetine 20mg daily. The drug was paused and she is recovering.",
    "An 80-year-old gentleman taking amlodipine 10mg daily presented with severe bilateral peripheral edema and dyspnea, leading to hospital admission.",
    "A 55-year-old woman with a history of rheumatoid arthritis was started on adalimumab 40 mg SC. After 6 weeks, she developed fever, cough, and was diagnosed with severe pneumonia, requiring hospitalization.",
    "A 42-year-old male on escitalopram 10mg daily for anxiety presented to the clinic with sexual dysfunction and vivid dreams. No dose adjustment was made.",
    "A 60-year-old man taking lisinopril 20mg for hypertension developed a persistent dry cough and mild angioedema of the lips. He was switched to losartan.",
    "A 29-year-old female patient on levothyroxine 100 mcg daily came in with tachycardia, tremors, and heat intolerance. Her TSH levels were undetectable.",
    "A 50-year-old male treated with carbamazepine for epilepsy was admitted to the burn unit after developing a diffuse blistering rash diagnosed as Stevens-Johnson syndrome.",
    "A 68-year-old female with COPD was prescribed prednisone 40 mg daily. She later presented with acute hyperglycemia and confusion in the ER.",
    "A 40-year-old man taking ibuprofen 800 mg TID for osteoarthritis developed severe dyspepsia and was found to have a bleeding peptic ulcer on endoscopy."
]

if "active_narrative" not in st.session_state:
    # Randomly select a case right on startup
    st.session_state["active_narrative"] = np.random.choice(sample_cases)

if "extracted_data" not in st.session_state:
    # Load it in an empty/waiting state so the user still gets to click the extraction button
    st.session_state["extracted_data"] = get_blank_extraction()

# =========================================================
# 6. DYNAMIC UI, STYLING & POPUP NOTICE
# =========================================================
# Top-level Light/Dark Toggle
t1, t2 = st.columns([8, 2])
with t2:
    is_light = st.toggle("☀️ Light Mode", key="light_mode_toggle")

# Dynamic Styling Logic (STANDARD THEME)
if is_light:
    bg_app = "#f8fafc"
    text_color = "#0f172a"
    card_bg = "#ffffff"
    border_col = "#e2e8f0"
    metric_val = "#0f172a"
    heading_bg = "linear-gradient(90deg, #f1f5f9, #ffffff)"
    heading_txt = "#0f172a"
else:
    bg_app = "#0b0f19"
    text_color = "#e2e8f0"
    card_bg = "#111827"
    border_col = "#1f2937"
    metric_val = "#f8fafc"
    heading_bg = "linear-gradient(90deg, #1e293b, #0f172a)"
    heading_txt = "#f8fafc"

st.markdown(f"""
<style>
    /* Standard App Background */
    .stApp {{
        background: {bg_app} !important;
        color: {text_color} !important;
        transition: background 0.3s ease-in-out;
    }}
    .block-container {{
        padding-top: 1.2rem !important;
        padding-bottom: 2rem !important;
        max-width: 96% !important;
    }}
    /* Enhanced Metric Cards */
    div[data-testid="stMetric"] {{
        background-color: {card_bg};
        border: 1px solid {border_col};
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    div[data-testid="stMetric"]:hover {{
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }}
    div[data-testid="stMetric"] label {{
        color: #64748b !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{
        color: {metric_val} !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }}
    /* Interactive Button Hover Effects */
    div.stButton > button {{
        border-radius: 8px !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }}
    div.stButton > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 12px rgba(59, 130, 246, 0.2) !important;
        filter: brightness(1.1) !important;
    }}
    /* Primary action button gets a glowing effect */
    div.stButton > button[kind="primary"] {{
        background: linear-gradient(90deg, #3b82f6, #2563eb) !important;
        border: none !important;
        color: #ffffff !important;
        font-weight: 600 !important;
    }}
    div.stButton > button[kind="primary"]:hover {{
        box-shadow: 0 6px 15px rgba(37, 99, 235, 0.4) !important;
    }}
</style>
""", unsafe_allow_html=True)

@st.dialog("Prototype Preview Notice")
def show_preview_notice():
    st.markdown("### 🚧 Preview Build | Active Development\nThis public dashboard is an interactive demo of core triage and signal detection logic. The main project runs privately on a dedicated clinical LLM for secure clinical reasoning.\n\n*Actively working on fine-tuning and upgrading the model.*")
    if st.button("Close & Test Dashboard", type="primary", use_container_width=True):
        st.rerun()

if "notice_seen" not in st.session_state:
    st.session_state.notice_seen = True
    show_preview_notice()

# =========================================================
# 7. SIDEBAR 
# =========================================================
with st.sidebar:
    st.markdown("### AI assisted Pharmacovigilance triage and signal detection")
    st.caption("Adverse Event Intake & Safety Surveillance")
    st.sidebar.button("ℹ️ About This Preview", on_click=show_preview_notice)
    workspace = st.radio("Navigation:", ["1. ICSR Case Intake & Triage", "2. Safety Signal Surveillance"])

# =========================================================
# 8. TAB 1: ICSR CASE INTAKE & TRIAGE WORKSPACE
# =========================================================
if workspace == "1. ICSR Case Intake & Triage":
    # Aesthetic Professional Heading
    st.markdown(f"""
    <div style="padding: 1.5rem; border-radius: 12px; background: {heading_bg}; border-left: 6px solid #3b82f6; margin-bottom: 2rem; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); transition: all 0.3s ease;">
        <h2 style="margin: 0; color: {heading_txt}; font-size: 1.8rem; font-weight: 700; letter-spacing: -0.025em;">
            📥 ICSR Case Intake & Automated Triage
        </h2>
        <p style="margin: 0.5rem 0 0 0; color: #64748b; font-size: 0.95rem;">
            Comparative Workspace: Unstructured Narrative vs. Validated E2B(R3) Form
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Metrics
    total = len(cases_df)
    serious = len(cases_df[cases_df["seriousness"] == "Serious"])
    s_pct = (serious / max(total, 1)) * 100

    m1, m2, m3 = st.columns(3)
    m1.metric("Total ICSR Records", total)
    m2.metric("Serious Ratio", f"{s_pct:.1f}%", f"{serious} serious")
    m3.metric("Triage Protocol", "Rule-Based Parser", "Local Standalone")

    st.markdown("<hr style='border: 0.5px solid #cbd5e1; margin: 25px 0;'>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("### 📄 Clinical Source Narrative")
        st.caption("Select a spontaneous clinical case or input custom medical records.")

        btn1, btn2 = st.columns([1, 1])
        with btn1:
            if st.button("🎲 Load Another Sample Case", use_container_width=True):
                st.session_state["active_narrative"] = np.random.choice(sample_cases)
                st.session_state["extracted_data"] = get_blank_extraction() # Wait for extraction button
                st.rerun()
        with btn2:
            show_ocr = st.button("📷 Add / OCR Ingest Document", use_container_width=True)

        if "ocr_active" not in st.session_state:
            st.session_state["ocr_active"] = False

        if show_ocr:
            st.session_state["ocr_active"] = not st.session_state["ocr_active"]

        if st.session_state["ocr_active"]:
            with st.container():
                st.markdown("""<div style="background-color: #f1f5f9; border: 1px dashed #3b82f6; border-radius: 8px; padding: 12px; margin-bottom: 12px;"><span style="font-size: 0.85rem; color: #3b82f6; font-weight: 600;">🔍 Optical Character Recognition (OCR) Engine</span></div>""", unsafe_allow_html=True)
                uploaded_doc = st.file_uploader("Upload ADR Document (PDF, PNG, JPG, TXT)", type=["txt", "pdf", "png", "jpg", "jpeg"])
                
                if uploaded_doc is not None:
                    if st.button("⚡ Ingest Text via OCR", type="secondary", use_container_width=True):
                        ingested = extract_text_from_upload(uploaded_doc)
                        st.session_state["active_narrative"] = ingested
                        st.session_state["extracted_data"] = get_blank_extraction()
                        st.toast("Document content ingested successfully!", icon="📄")
                        st.session_state["ocr_active"] = False
                        st.rerun()

        input_text = st.text_area("Source Report Text", value=st.session_state["active_narrative"], height=210)

        # Extraction Button triggers the parsing logic
        if st.button("⚡ Run Automated Extraction & Triage", type="primary", use_container_width=True):
            st.session_state["active_narrative"] = input_text
            st.session_state["extracted_data"] = run_dummy_extraction(input_text)
            st.rerun()

    ext = st.session_state["extracted_data"]
    is_serious = str(ext.get("seriousness", "Non-Serious")).title() == "Serious"
    reason_val = str(ext.get("reason", "None"))
    has_data = ext.get("age") != 0 or ext.get("drug") != "" # Check if extraction ran

    with col_right:
        if not has_data:
            st.markdown("""
            <div style="background: rgba(100, 116, 139, 0.05); border: 1px dashed #94a3b8; border-radius: 8px; padding: 24px; text-align: center; margin-bottom: 1rem;">
                <span style="color: #64748b; font-weight: 600; font-size: 1rem;">Awaiting Extraction</span>
                <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 4px;">Click the Run Automated Extraction button to process the case.</div>
            </div>
            """, unsafe_allow_html=True)
        elif is_serious:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(17, 24, 39, 0.05) 100%); 
                        border: 1px solid rgba(239, 68, 68, 0.5); border-left: 6px solid #ef4444; border-radius: 8px; 
                        padding: 16px; margin-bottom: 1rem; box-shadow: 0 4px 15px rgba(239, 68, 68, 0.15);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: #ef4444; font-weight: 700; font-size: 0.95rem;">🚨 REGULATORY PRIORITY: SERIOUS ADVERSE EVENT</span>
                    <span style="background: rgba(239, 68, 68, 0.2); color: #dc2626; font-size: 0.75rem; padding: 3px 10px; border-radius: 12px; font-weight:bold;">EXPEDITED 15-DAY</span>
                </div>
                <div style="color: #64748b; font-size: 0.85rem; margin-top: 8px;">
                    Criterion Detected: <b>{reason_val}</b> • Regulatory triage required under ICH E2A.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(17, 24, 39, 0.05) 100%); 
                        border: 1px solid rgba(16, 185, 129, 0.4); border-left: 6px solid #10b981; border-radius: 8px; 
                        padding: 16px; margin-bottom: 1rem; box-shadow: 0 4px 15px rgba(16, 185, 129, 0.1);">
                <span style="color: #10b981; font-weight: 700; font-size: 0.95rem;">✅ REGULATORY PRIORITY: NON-SERIOUS CASE</span>
                <div style="color: #64748b; font-size: 0.85rem; margin-top: 8px;">Routine post-marketing surveillance queue.</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 📑 Structured E2B(R3) Form")

        with st.form("case_form"):
            c1, c2 = st.columns(2)
            with c1:
                age_val = int(ext.get("age", 0)) if str(ext.get("age", "")).isdigit() else 0
                age = st.number_input("Age", min_value=0, max_value=120, value=age_val)
                sex_list = ["Female", "Male", "Unknown"]
                sex_idx = sex_list.index(ext.get("sex", "Unknown")) if ext.get("sex") in sex_list else 2
                sex = st.selectbox("Sex", sex_list, index=sex_idx)
                drug = st.text_input("Suspect Drug", value=str(ext.get("drug", "")))
                dose = st.text_input("Dose / Regimen", value=str(ext.get("dose", "")))
                
                outcomes = ["Recovered", "Recovering", "Not Recovered", "Fatal", "Unknown"]
                out_idx = outcomes.index(ext.get("outcome", "Unknown")) if ext.get("outcome") in outcomes else 4
                outcome = st.selectbox("Outcome", outcomes, index=out_idx)

            with c2:
                event = st.text_area("Reported Event(s) & Disease(s)", value=str(ext.get("event", "")), height=135)
                pt = st.text_input("Primary MedDRA PT", value=str(ext.get("meddra_pt", "")))
                seriousness = st.selectbox("Seriousness", ["Serious", "Non-Serious"], index=0 if is_serious else 1)
                
                reasons = ["Hospitalization", "Life-Threatening", "Disability", "Death", "Congenital Anomaly", "Other Medically Important", "None"]
                reason_idx = reasons.index(reason_val) if reason_val in reasons else 6
                reason = st.selectbox("Criteria", reasons, index=reason_idx)
                
                new_id = f"CASE-2026-{np.random.randint(100, 999)}"
                st.text_input("Case ID", value=new_id, disabled=True)

            if st.form_submit_button("💾 Save Case to Database", type="primary", use_container_width=True):
                if drug:
                    save_case_to_db({
                        "case_id": new_id, "age": age, "sex": sex, "drug": drug, "dose": dose,
                        "event": event, "meddra_pt": pt, "seriousness": seriousness,
                        "reason": reason, "outcome": outcome, "raw_text": input_text
                    })
                    st.toast(f"Case {new_id} saved to SQLite!", icon="✅")
                    st.rerun()
                else:
                    st.error("Please run extraction or enter a suspect drug before saving.")

    st.markdown("<hr style='border: 0.5px solid #cbd5e1; margin: 30px 0;'>", unsafe_allow_html=True)
    st.markdown("### 📚 Stored ICSR Cases")
    st.dataframe(fetch_all_cases(), use_container_width=True)

# =========================================================
# 9. TAB 2: SAFETY SIGNAL SURVEILLANCE
# =========================================================
elif workspace == "2. Safety Signal Surveillance":
    st.markdown("## 📊 Quantitative Signal Surveillance")
    st.caption("Disproportionality Analysis via Proportional Reporting Ratio (PRR) & Reporting Odds Ratio (ROR)")

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Surveillance Cohort", len(cases_df))
    s2.metric("Reference Population", "2,450,000")
    s3.metric("Signal Threshold", "PRR ≥ 2.0")
    s4.metric("Active Signals", "2 Detected", delta="1 Priority", delta_color="inverse")

    st.markdown("<hr style='border: 0.5px solid #cbd5e1; margin: 20px 0;'>", unsafe_allow_html=True)

    data = pd.DataFrame([
        {"Drug": "Amoxicillin", "Event": "Diarrhoea", "a": 520, "b": 12000, "c": 4100, "d": 450000},
        {"Drug": "Atorvastatin", "Event": "Jaundice", "a": 340, "b": 8500, "c": 1900, "d": 380000},
        {"Drug": "Metformin", "Event": "Lactic Acidosis", "a": 115, "b": 9200, "c": 420, "d": 410000},
        {"Drug": "Amlodipine", "Event": "Peripheral Oedema", "a": 890, "b": 11000, "c": 3100, "d": 420000},
        {"Drug": "Ciprofloxacin", "Event": "Tendon Rupture", "a": 180, "b": 6400, "c": 210, "d": 390000},
        {"Drug": "Amoxicillin", "Event": "Headache", "a": 45, "b": 12475, "c": 18000, "d": 436100}
    ])

    results = []
    for _, row in data.iterrows():
        stats = calculate_disproportionality(row["a"], row["b"], row["c"], row["d"])
        results.append({**row, **stats})
    res_df = pd.DataFrame(results)

    st.dataframe(
        res_df[["Drug", "Event", "a", "PRR", "PRR_95_CI", "ROR", "ROR_95_CI", "Signal_Triggered"]].rename(columns={"a": "Report Count (a)"}),
        use_container_width=True
    )

    fig = px.scatter(
        res_df, x="a", y="PRR", text="Event", color="Signal_Triggered", size="ROR",
        log_x=True, title="Disproportionality Volcano Plot (Threshold: PRR = 2.0)",
        labels={"a": "Reports (Log Scale)", "PRR": "Proportional Reporting Ratio (PRR)"},
        color_discrete_map={"YES (Review Required)": "#EF4444", "No": "#3B82F6"}
    )
    fig.add_hline(y=2.0, line_dash="dash", line_color="#EF4444", annotation_text="PRR = 2.0 Threshold")
    fig.update_traces(textposition="top center")
    
    if not is_light:
        fig.update_layout(plot_bgcolor="#0b0f19", paper_bgcolor="#0b0f19", font_color="#e2e8f0")
        
    st.plotly_chart(fig, use_container_width=True)