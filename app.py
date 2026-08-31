import streamlit as st
from io import StringIO
import docx2txt
import pdfplumber
import re
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, util
import torch
import time

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Tamil Scribe AI | Smart Document Summarizer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# SESSION STATE INITIALIZATION
# =========================================================
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "summary_result" not in st.session_state:
    st.session_state.summary_result = None
if "extractive_result" not in st.session_state:
    st.session_state.extractive_result = None
if "last_processed_type" not in st.session_state:
    st.session_state.last_processed_type = ""
if "detected_lang" not in st.session_state:
    st.session_state.detected_lang = "ta"
if "stats" not in st.session_state:
    st.session_state.stats = {}

# Sample texts for 1-click testing
SAMPLE_TAMIL_TECH = """செயற்கை நுண்ணறிவு (Artificial Intelligence - AI) என்பது கணினி அறிவியல் துறையில் மிக வேகமாக வளர்ந்து வரும் ஒரு புரட்சிகரமான தொழில்நுட்பமாகும். மனித மூளையின் சிந்தனைத் திறன், கற்றல் முறை மற்றும் முடிவெடுக்கும் ஆற்றல் ஆகியவற்றை கணினி அமைப்புகளுக்கு வழங்குவதே இதன் முக்கிய நோக்கமாகும்.

தற்காலத்தில் இயற்கை மொழி செயலாக்கம் (Natural Language Processing), கணினி பார்வை (Computer Vision) மற்றும் தானியங்கி வாகனங்கள் போன்ற பல துறைகளில் செயற்கை நுண்ணறிவு முன்னணியில் திகழ்கிறது. குறிப்பாக தமிழ் போன்ற பாரம்பரிய மொழிகளில் மொழிபெயர்ப்பு, பேச்சுணர்தல் மற்றும் தானியங்கி உரை சுருக்கம் ஆகியவற்றில் செயற்கை நுண்ணறிவு பெரும் மாற்றங்களை ஏற்படுத்தி வருகிறது.

கல்வித்துறையில் மாணவர்களுக்கு தனிப்பயனாக்கப்பட்ட கற்றல் அனுபவங்களை வழங்க AI மென்பொருட்கள் பெரிதும் பயன்படுகின்றன. மருத்துவத் துறையில் நோய்களை ஆரம்ப கட்டத்திலேயே துல்லியமாகக் கண்டறியவும், புதிய மருந்துகளைக் கண்டறியவும் AI வழிவகுக்கிறது. விவசாயத்தில் பயிர் நோய்களைக் கண்டறிதல் மற்றும் வானிலை முன்னறிவிப்புகளில் இது முக்கியப் பங்கு வகிக்கிறது.

எதிர்காலத்தில் செயற்கை நுண்ணறிவு மனிதர்களின் அன்றாட வாழ்க்கையில் பிரிக்க முடியாத அங்கமாக மாறும் என்று தொழில்நுட்ப வல்லுநர்கள் கணிக்கின்றனர். அதே வேளையில், தரவு பாதுகாப்பு மற்றும் நெறிமுறைகள் சார்ந்த விழிப்புணர்வும் மிக அவசியமாகிறது."""

SAMPLE_TAMIL_LIT = """திருக்குறள் உலகப் பொதுமறை என்று போற்றப்படும் மிகச்சிறந்த தமிழ் வாழ்வியல் நூலாகும். இதனை இயற்றியவர் திருவள்ளுவர். இந்நூல் அறத்துப்பால், பொருட்பால், காமத்துப்பால் என்ற மூன்று பெரும் பிரிவுகளையும், 133 அதிகாரங்களையும், மொத்தம் 1330 குறட்பாக்களையும் கொண்டுள்ளது.

மனிதன் எவ்வாறு வாழ வேண்டும், சமுதாயத்தில் பிறருடன் எவ்வாறு பழக வேண்டும், ஒரு அரசன் நாட்டை எவ்வாறு ஆள வேண்டும், நட்பின் பெருமை, கல்வியின் மேன்மை, ஒழுக்கத்தின் சிறப்பு ஆகிய அனைத்து விழுமியங்களையும் இரண்டு வரிகளில் மிக ஆழமாக விளக்குகிறது திருக்குறள்.

எந்த ஒரு குறிப்பிட்ட மதம், சாதி, இனம் அல்லது மொழியைச் சாராமல், உலக மக்கள் அனைவருக்கும் பொதுவான மனிதநேயக் கருத்துக்களைக் கூறுவதால் இது 'உலகப் பொதுமறை' என்ற உலகளாவிய அங்கீகாரத்தைப் பெற்றுள்ளது. உலகின் நூற்றுக்கும் மேற்பட்ட மொழிகளில் மொழிபெயர்க்கப்பட்டுள்ள பெருமை திருக்குறளுக்கு உண்டு."""

SAMPLE_ENGLISH_TECH = """Artificial Intelligence and Large Language Models (LLMs) have fundamentally transformed how humans interact with digital information. Modern transformer architectures, equipped with self-attention mechanisms, are capable of processing vast amounts of textual data, synthesizing key concepts, and generating human-like summaries in fractions of a second.

Extractive summarization identifies and scores the most salient sentences directly from the source text using high-dimensional sentence embeddings and cosine similarity. In contrast, abstractive summarization interprets the core semantics and generates entirely new sentences that succinctly capture the main ideas.

The integration of multilingual embeddings enables systems to understand low-resource and classical languages such as Tamil, Kannada, and Telugu alongside English. This democratizes access to information, allowing researchers, students, and legal professionals to quickly analyze long documents, legal contracts, research papers, and technical manuals without language barriers."""

SAMPLE_ENGLISH_BUSINESS = """Quarterly Business Performance & Strategic Outlook: The fiscal year 2024 concluded with record revenue growth of 24.5% year-over-year, driven primarily by enterprise adoption of cloud-native infrastructure and automated customer support solutions. Total revenue reached $142.8 million compared to $114.7 million in the previous fiscal period.

Operating expenses increased moderately by 8.2%, reflecting strategic investments in research and development and international market expansion across the Asia-Pacific region. Operating margins expanded by 340 basis points to 28.6%, showcasing high operational leverage.

Looking forward to 2025, the company plans to launch three flagship AI modules, expand partner ecosystems, and increase customer retention rates above 96%. Key risk factors include global macroeconomic volatility, foreign exchange fluctuations, and evolving regulatory compliance requirements."""

# =========================================================
# CUSTOM CSS DESIGN SYSTEM (Theme-Adaptive & Typography)
# =========================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Mukta+Malar:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Root typography & basic reset */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', 'Mukta Malar', sans-serif;
    }

    /* Gradient Hero Container */
    .hero-container {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(168, 85, 247, 0.18) 50%, rgba(236, 72, 153, 0.12) 100%);
        border: 1px solid rgba(147, 51, 234, 0.3);
        border-radius: 20px;
        padding: 30px 24px 24px 24px;
        text-align: center;
        margin-bottom: 25px;
        position: relative;
        backdrop-filter: blur(12px);
        box-shadow: 0 10px 30px -10px rgba(99, 102, 241, 0.2);
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: linear-gradient(90deg, #6366F1, #8B5CF6);
        color: #ffffff !important;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        padding: 5px 14px;
        border-radius: 9999px;
        margin-bottom: 12px;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.35);
    }

    .hero-title {
        font-size: 38px;
        font-weight: 800;
        letter-spacing: -0.8px;
        background: linear-gradient(135deg, #6366F1 0%, #A855F7 50%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 8px 0;
        line-height: 1.2;
    }

    .hero-subtitle {
        font-size: 15.5px;
        font-weight: 400;
        opacity: 0.9;
        margin-bottom: 16px;
        max-width: 680px;
        margin-left: auto;
        margin-right: auto;
    }

    .hero-tags-row {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 10px;
    }

    .hero-tag {
        background: rgba(99, 102, 241, 0.12);
        border: 1px solid rgba(99, 102, 241, 0.25);
        color: inherit;
        font-size: 12px;
        font-weight: 600;
        padding: 5px 13px;
        border-radius: 8px;
        backdrop-filter: blur(4px);
    }

    /* Metric Cards - Theme Adaptive */
    .metric-box {
        background: rgba(99, 102, 241, 0.07);
        border: 1px solid rgba(99, 102, 241, 0.22);
        border-radius: 14px;
        padding: 14px 16px;
        text-align: center;
        backdrop-filter: blur(8px);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .metric-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.2);
        border-color: rgba(99, 102, 241, 0.5);
    }

    .metric-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
        opacity: 0.8;
        margin-bottom: 4px;
    }

    .metric-value {
        font-size: 20px;
        font-weight: 800;
        color: inherit;
        font-family: 'JetBrains Mono', 'Plus Jakarta Sans', monospace;
    }

    .metric-sub {
        font-size: 11px;
        font-weight: 600;
        color: #10b981;
        margin-top: 2px;
    }

    .summary-header-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.3);
        color: #818cf8;
        font-size: 12px;
        font-weight: 700;
        padding: 5px 14px;
        border-radius: 8px;
        margin-bottom: 16px;
    }

    /* Key Points Highlight Box */
    .key-point-item {
        background: rgba(99, 102, 241, 0.06);
        border-left: 4px solid #6366f1;
        border-radius: 0 10px 10px 0;
        border-top: 1px solid rgba(99, 102, 241, 0.12);
        border-right: 1px solid rgba(99, 102, 241, 0.12);
        border-bottom: 1px solid rgba(99, 102, 241, 0.12);
        padding: 14px 18px;
        margin-bottom: 12px;
        font-size: 15px;
        line-height: 1.65;
        color: inherit;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 12px !important;
        font-weight: 700 !important;
        letter-spacing: 0.3px !important;
        padding: 10px 20px !important;
        transition: all 0.25s ease !important;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #9333ea 100%) !important;
        border: none !important;
        color: #ffffff !important;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.35) !important;
    }

    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(124, 58, 237, 0.5) !important;
    }

    .sidebar-brand-card {
        background: linear-gradient(135deg, #4338ca 0%, #6366f1 100%);
        color: #ffffff !important;
        border-radius: 14px;
        padding: 16px 18px;
        margin-bottom: 20px;
        box-shadow: 0 6px 16px rgba(67, 56, 202, 0.3);
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 11px;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 9999px;
        margin-top: 6px;
    }

    .status-active {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .status-fallback {
        background: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }

    /* Footer styling */
    .custom-footer {
        text-align: center;
        padding: 24px 0 12px 0;
        margin-top: 40px;
        border-top: 1px solid rgba(148, 163, 184, 0.2);
        opacity: 0.8;
        font-size: 13px;
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 28px;
        }
        .hero-subtitle {
            font-size: 14px;
        }
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# MODELS & API INITIALIZATION
# =========================================================

# Helper to configure Gemini API Key
def get_gemini_client(api_key_override=None):
    """Initializes and returns the Gemini model instance safely."""
    api_key = None
    
    # 1. Check user override from input field
    if api_key_override and api_key_override.strip():
        api_key = api_key_override.strip()
    else:
        # 2. Check Streamlit secrets safely without throwing StreamlitSecretNotFoundError
        try:
            if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
                api_key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            api_key = None

    if not api_key:
        return None, "No API key configured."

    try:
        genai.configure(api_key=api_key)
        # Check model availability - default to gemini-1.5-pro
        model_instance = genai.GenerativeModel("gemini-1.5-pro")
        return model_instance, "Active"
    except Exception as e:
        return None, f"Configuration Error: {str(e)}"


# Sentence Transformer for Local Extractive Summarization (Cached)
@st.cache_resource(show_spinner=False)
def load_sentence_model():
    """Loads multilingual sentence transformer for semantic extractive summarization."""
    try:
        return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    except Exception:
        # Fallback in case of local cache or network issue
        return None


sentence_model = load_sentence_model()


# =========================================================
# UTILITY FUNCTIONS
# =========================================================

def extract_text_from_file(uploaded_file):
    """Extracts raw text from uploaded TXT, DOCX, or PDF files."""
    if uploaded_file is None:
        return ""
        
    file_type = uploaded_file.type
    filename = uploaded_file.name.lower()

    try:
        # Plain text
        if file_type == "text/plain" or filename.endswith(".txt"):
            return StringIO(uploaded_file.getvalue().decode("utf-8", errors="ignore")).read()

        # Word DOCX
        elif (
            file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            or filename.endswith(".docx")
        ):
            return docx2txt.process(uploaded_file)

        # PDF Document
        elif file_type == "application/pdf" or filename.endswith(".pdf"):
            extracted_pages = []
            with pdfplumber.open(uploaded_file) as pdf:
                for idx, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        extracted_pages.append(page_text.strip())
            return "\n\n".join(extracted_pages)

        else:
            return ""
    except Exception as err:
        st.error(f"Error parsing file '{uploaded_file.name}': {str(err)}")
        return ""


def detect_language(text):
    """Detects if document is predominantly Tamil ('ta') or English ('en')."""
    if not text or not text.strip():
        return "ta"
    tamil_chars = re.findall(r'[\u0B80-\u0BFF]', text)
    # If substantial Tamil unicode characters are found, classify as Tamil
    return "ta" if len(tamil_chars) > 10 else "en"


def compute_reading_time(word_count, wpm=180):
    """Estimates reading time in minutes and seconds."""
    if word_count <= 0:
        return "0 sec"
    minutes = word_count / wpm
    if minutes < 1:
        return f"{max(1, int(minutes * 60))} sec"
    return f"{minutes:.1f} min"


def split_into_sentences(text):
    """Splits multilingual text into sentences preserving Tamil and English delimiters."""
    # Split on sentence terminals like ., !, ?, or Tamil newline terminators
    raw_sentences = re.split(r'(?<=[.!?|।\n])\s+', text.strip())
    cleaned = []
    for s in raw_sentences:
        clean_s = s.strip()
        # Discard ultra-short sentence fragments (less than 15 chars)
        if len(clean_s) >= 15:
            cleaned.append(clean_s)
    return cleaned


def extract_key_sentences_global(text, n=5):
    """Extracts top n semantically salient sentences using SentenceTransformer embeddings."""
    if sentence_model is None:
        sentences = split_into_sentences(text)
        return sentences[:n]

    sentences = split_into_sentences(text)
    if len(sentences) <= n:
        return sentences

    try:
        embeddings = sentence_model.encode(sentences, convert_to_tensor=True)
        doc_embedding = sentence_model.encode(text, convert_to_tensor=True)

        scores = util.pytorch_cos_sim(embeddings, doc_embedding).squeeze()
        
        # Handle 1D or 0D tensor scores
        if scores.dim() == 0:
            return [sentences[0]]
            
        top_k = min(n, len(sentences))
        top_idxs = torch.topk(scores, k=top_k).indices
        top_idxs = sorted(top_idxs.tolist())

        return [sentences[i] for i in top_idxs]
    except Exception:
        # Fallback gracefully to first n sentences
        return sentences[:n]


def summarize_with_fallback(text, n=5):
    """Generates a clean extractive summary using local sentence embeddings."""
    summary_points = extract_key_sentences_global(text, n=n)
    seen = set()
    unique_summary = []

    for point in summary_points:
        stripped = point.strip()
        if stripped and stripped not in seen:
            seen.add(stripped)
            unique_summary.append(stripped)

    return unique_summary


def summarize_with_gemini(text, lang="ta", style="bullet", length="balanced", model_instance=None):
    """Generates an intelligent abstractive summary using Gemini with customized formatting."""
    if model_instance is None:
        return ""

    # Style definitions
    style_instruction_en = {
        "bullet": "Present the summary as distinct, structured bullet points highlighting the most essential takeaways.",
        "executive": "Write an executive summary with a 2-sentence overarching overview followed by key operational/strategic points.",
        "takeaways": "Highlight the Top 5 Core Takeaways and Actionable Insights in clear, numbered bullet points.",
        "synopsis": "Write a cohesive, well-written narrative paragraph synthesis of the document."
    }.get(style, "Present the result as clear bullet points.")

    style_instruction_ta = {
        "bullet": "முக்கியமான கருத்துக்களை தெளிவான புள்ளிவிவரக் குறிப்புகளாக (Bullet points) வரிசைப்படுத்தித் தருக.",
        "executive": "ஆவணத்தின் முக்கிய சாரம்சத்தை ஒரு செயல்முறைச் சுருக்கமாக (Executive Summary) தருக.",
        "takeaways": "முக்கிய 5 சிறப்பம்சங்கள் மற்றும் தீர்வுகளை வரிசைப்படுத்தித் தருக.",
        "synopsis": "முழு ஆவணத்தையும் தெளிவான, சுருக்கமான பத்தியாக (Paragraph synopsis) தருக."
    }.get(style, "தெளிவான புள்ளி விவரங்களாக தருக.")

    # Length guidelines
    length_instruction_en = {
        "concise": "Keep the summary ultra-concise, under 100-150 words.",
        "balanced": "Provide a balanced, comprehensive summary of around 150-250 words.",
        "detailed": "Provide an in-depth, thorough summary covering all nuances and sub-points."
    }.get(length, "Keep it balanced and comprehensive.")

    length_instruction_ta = {
        "concise": "சுருக்கம் மிகச் சுருக்கமாக 100-150 சொற்களுக்குள் இருக்க வேண்டும்.",
        "balanced": "சுருக்கம் சமநிலையான 150-250 சொற்களுக்குள் இருக்க வேண்டும்.",
        "detailed": "அனைத்து முக்கிய விவரங்களையும் உள்ளடக்கிய விரிவான சுருக்கமாகத் தருக."
    }.get(length, "சமநிலையான சுருக்கமாகத் தருக.")

    if lang == "ta":
        prompt = f"""நீங்கள் ஒரு தமிழ் ஆவண சுருக்க நிபுணர் மற்றும் AI உதவியாளர் (Tamil Document Summarization Expert).

கீழே கொடுக்கப்பட்டுள்ள ஆவணத்தை பகுப்பாய்வு செய்து உயர்தர சுருக்கத்தை தமிழில் வழங்கவும்.

வழிகாட்டுதல்கள்:
1. ஆவணத்தின் மையக்கருத்தையும் முக்கிய செய்தியையும் முழுமையாகப் பாதுகாக்கவும்.
2. நபர்களின் பெயர்கள், தேதிகள், புள்ளிவிவரங்கள், எண்கள், இடங்கள் ஆகியவற்றைத் துல்லியமாகப் பாதுகாக்கவும்.
3. வடிவம்: {style_instruction_ta}
4. நீளம்: {length_instruction_ta}
5. ஆவணத்தில் இல்லாத தகவல்களை நீங்களாகச் சேர்க்க வேண்டாம்.
6. வாசகருக்கு எளிதில் புரியும் வகையிலும், வாசிப்பதற்கு நேர்த்தியாகவும் அமைய வேண்டும்.

ஆவணம்:
\"\"\"
{text}
\"\"\"

தமிழில் சுருக்கம்:"""

    else:
        prompt = f"""You are a professional document summarization expert.

Analyze the following document and generate a high-quality, professional summary.

Requirements:
1. Preserve the core meaning, primary conclusions, and critical context.
2. Preserve key entities, including names, dates, financial figures, metrics, and locations accurately.
3. Format: {style_instruction_en}
4. Length: {length_instruction_en}
5. Do not hallucinate or include information not found in the original source.
6. Use clean, professional formatting with markdown.

Document:
\"\"\"
{text}
\"\"\"

Summary:"""

    try:
        response = model_instance.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.warning(f"⚠️ Gemini API Request Note: {e}")
        return ""


# =========================================================
# SIDEBAR CONFIGURATION & CONTROLS
# =========================================================

with st.sidebar:
    # App Branding Card
    st.markdown("""
    <div class="sidebar-brand-card">
        <div style="font-size: 24px; margin-bottom: 4px;">📝 <b>Tamil Scribe AI</b></div>
        <div style="font-size: 12px; opacity: 0.9;">Multilingual Document Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    # Initialize Gemini model
    gemini_model, _ = get_gemini_client()

    # Summary Settings
    st.markdown("### ⚙️ Summary Preferences")

    summary_style = st.selectbox(
        "Summarization Style",
        options=["bullet", "executive", "takeaways", "synopsis"],
        format_func=lambda x: {
            "bullet": "📌 Key Bullet Points (புள்ளி விவரங்கள்)",
            "executive": "📑 Executive Summary (செயல்முறை சுருக்கம்)",
            "takeaways": "🎯 Top Takeaways (முக்கிய அம்சங்கள்)",
            "synopsis": "📖 Narrative Synopsis (விளக்கப் பத்தி)"
        }[x],
        index=0
    )

    summary_length = st.select_slider(
        "Summary Depth / Length",
        options=["concise", "balanced", "detailed"],
        value="balanced",
        format_func=lambda x: {
            "concise": "⚡ Concise (சுருக்கம்)",
            "balanced": "⚖️ Balanced (சமநிலை)",
            "detailed": "🔍 In-depth (விரிவானது)"
        }[x]
    )

    extractive_sentence_count = st.slider(
        "Extractive Highlight Count",
        min_value=3,
        max_value=12,
        value=5,
        help="Number of core sentences extracted by the semantic transformer model."
    )

    st.markdown("---")

    # Quick Samples
    st.markdown("### ⚡ Quick Sample Tests")
    st.caption("Click to instantly populate and test the summarizer:")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("🇮🇳 தமிழ் AI செய்தி", use_container_width=True):
            st.session_state.input_text = SAMPLE_TAMIL_TECH
            st.rerun()
        if st.button("🇬🇧 English Tech", use_container_width=True):
            st.session_state.input_text = SAMPLE_ENGLISH_TECH
            st.rerun()
            
    with col_s2:
        if st.button("📜 திருக்குறள்", use_container_width=True):
            st.session_state.input_text = SAMPLE_TAMIL_LIT
            st.rerun()
        if st.button("📊 Business Doc", use_container_width=True):
            st.session_state.input_text = SAMPLE_ENGLISH_BUSINESS
            st.rerun()

    st.markdown("---")
    
    # Supported Specs Box
    st.markdown("""
    <div style="font-size: 12px; color: #64748b; line-height: 1.6;">
        <b>Supported Languages:</b> 🇮🇳 தமிழ் (Tamil) & 🇬🇧 English<br>
        <b>Supported Formats:</b> PDF, DOCX, TXT<br>
        <b>AI Architecture:</b> Google Gemini 1.5 & Sentence Transformers (MiniLM-L12)
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# MAIN CONTENT HERO AREA
# =========================================================

st.markdown("""
<div class="hero-container">
    <div class="hero-badge">
        <span>✨ AI-Powered Intelligence</span>
    </div>
    <div class="hero-title">📝 Tamil Scribe AI</div>
    <div class="hero-subtitle">
        Intelligent Tamil & English Document Summarization. Transform lengthy articles, research papers, and documents into clear, actionable executive insights in seconds.
    </div>
    <div class="hero-tags-row">
        <span class="hero-tag">🇮🇳 தமிழ் & 🇬🇧 English</span>
        <span class="hero-tag">⚡ Abstractive AI</span>
        <span class="hero-tag">🧠 Semantic Extractive Engine</span>
        <span class="hero-tag">📄 PDF / DOCX / Text</span>
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# INPUT TABS & DOCUMENT SELECTION
# =========================================================

input_tab1, input_tab2, input_tab3 = st.tabs([
    "✍️ Direct Text Input",
    "📄 PDF Document",
    "📘 Word Document (DOCX)"
])

# Global variable to capture extracted text for processing
raw_input_text = ""

# TAB 1: Direct Text Input
with input_tab1:
    col_t1, col_t2 = st.columns([4, 1])
    with col_t1:
        st.markdown("**Enter or paste your text below:**")
    with col_t2:
        if st.button("🗑️ Clear Text", use_container_width=True):
            st.session_state.input_text = ""
            st.session_state.summary_result = None
            st.session_state.extractive_result = None
            st.rerun()

    text_area_content = st.text_area(
        "Direct Text Input",
        value=st.session_state.input_text,
        height=240,
        placeholder="Paste your Tamil or English document, article, notes, or essay here...",
        label_visibility="collapsed"
    )
    if text_area_content != st.session_state.input_text:
        st.session_state.input_text = text_area_content
    raw_input_text = st.session_state.input_text

# TAB 2: PDF Upload
with input_tab2:
    st.markdown("**Upload a PDF document to extract and summarize:**")
    uploaded_pdf = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        help="Upload multi-page or single-page PDF files.",
        key="pdf_uploader"
    )
    if uploaded_pdf is not None:
        with st.spinner("📖 Reading and extracting PDF text..."):
            extracted_pdf_text = extract_text_from_file(uploaded_pdf)
            if extracted_pdf_text.strip():
                raw_input_text = extracted_pdf_text
                st.session_state.input_text = extracted_pdf_text
                st.success(f"✅ Successfully extracted {len(extracted_pdf_text.split())} words from **{uploaded_pdf.name}**")
            else:
                st.error("❌ Unable to extract readable text from this PDF. Please ensure it contains selectable text, not scanned images.")

# TAB 3: DOCX Upload
with input_tab3:
    st.markdown("**Upload a Microsoft Word (.docx) document:**")
    uploaded_docx = st.file_uploader(
        "Upload Word Document",
        type=["docx"],
        help="Upload Microsoft Word .docx files.",
        key="docx_uploader"
    )
    if uploaded_docx is not None:
        with st.spinner("📖 Reading DOCX content..."):
            extracted_docx_text = extract_text_from_file(uploaded_docx)
            if extracted_docx_text.strip():
                raw_input_text = extracted_docx_text
                st.session_state.input_text = extracted_docx_text
                st.success(f"✅ Successfully extracted {len(extracted_docx_text.split())} words from **{uploaded_docx.name}**")
            else:
                st.error("❌ Unable to extract readable text from this Word document.")


# =========================================================
# LIVE TEXT ANALYTICS DASHBOARD
# =========================================================

current_text = raw_input_text.strip()

if current_text:
    detected_lang_code = detect_language(current_text)
    st.session_state.detected_lang = detected_lang_code
    
    words_list = current_text.split()
    total_words = len(words_list)
    total_chars = len(current_text)
    est_read_time = compute_reading_time(total_words)
    est_sentences = len(split_into_sentences(current_text))
    lang_display = "தமிழ் 🇮🇳" if detected_lang_code == "ta" else "English 🇬🇧"

    st.markdown("### 📊 Document Overview")
    
    # Custom interactive metric dashboard
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    
    with m_col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">🌐 Detected Language</div>
            <div class="metric-value">{lang_display}</div>
            <div class="metric-sub">Auto-classified</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">📝 Word Count</div>
            <div class="metric-value">{total_words:,}</div>
            <div class="metric-sub">{total_chars:,} characters</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">⏱️ Est. Read Time</div>
            <div class="metric-value">{est_read_time}</div>
            <div class="metric-sub">Original pace</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">📄 Sentences</div>
            <div class="metric-value">{est_sentences}</div>
            <div class="metric-sub">Analyzed units</div>
        </div>
        """, unsafe_allow_html=True)

    # Document Preview Expander
    with st.expander("🔍 Click to view full document text", expanded=False):
        st.text_area(
            "Document Raw Content",
            value=current_text,
            height=200,
            disabled=True,
            label_visibility="collapsed"
        )


# =========================================================
# SUMMARIZE TRIGGER BUTTON
# =========================================================

st.markdown("<div style='margin-top: 15px; margin-bottom: 25px;'>", unsafe_allow_html=True)
btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])

with btn_col2:
    summarize_clicked = st.button(
        "✨ SUMMARIZE DOCUMENT / ஆவணத்தைச் சுருக்கு",
        type="primary",
        use_container_width=True
    )
st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# SUMMARIZATION EXECUTION
# =========================================================

if summarize_clicked:
    if not current_text:
        st.error("⚠️ Please provide text input or upload a document before summarizing.")
    else:
        progress_bar = st.progress(0, text="🚀 Initializing intelligent summarization pipeline...")
        time.sleep(0.2)
        
        detected_lang = detect_language(current_text)
        
        # Step 1: Semantic Extractive Processing
        progress_bar.progress(30, text="🧠 Computing semantic embeddings and key sentence weights...")
        fallback_points = summarize_with_fallback(current_text, n=extractive_sentence_count)
        st.session_state.extractive_result = fallback_points
        
        # Step 2: Generative AI Summarization
        progress_bar.progress(65, text="🤖 Generating high-accuracy AI abstractive summary...")
        
        if gemini_model is not None:
            gemini_summary = summarize_with_gemini(
                current_text,
                lang=detected_lang,
                style=summary_style,
                length=summary_length,
                model_instance=gemini_model
            )
        else:
            gemini_summary = ""

        progress_bar.progress(100, text="✨ Complete!")
        time.sleep(0.2)
        progress_bar.empty()

        # Save results to session state
        st.session_state.summary_result = gemini_summary
        
        # Calculate summary statistics
        original_words = len(current_text.split())
        summary_text_for_calc = gemini_summary if gemini_summary else " ".join(fallback_points)
        summary_words = len(summary_text_for_calc.split())
        
        compression_ratio = round(original_words / max(1, summary_words), 1)
        savings_percent = max(0, round(((original_words - summary_words) / max(1, original_words)) * 100))
        summary_read_time = compute_reading_time(summary_words)

        st.session_state.stats = {
            "original_words": original_words,
            "summary_words": summary_words,
            "compression_ratio": compression_ratio,
            "savings_percent": savings_percent,
            "summary_read_time": summary_read_time,
            "used_gemini": bool(gemini_summary)
        }


# =========================================================
# RESULTS PRESENTATION DASHBOARD
# =========================================================

if st.session_state.summary_result is not None or st.session_state.extractive_result is not None:
    stats = st.session_state.stats
    has_gemini = bool(st.session_state.summary_result)
    
    st.markdown("---")
    st.markdown("## 🎯 Summary & Document Intelligence")

    # High-impact summary metrics bar
    if stats:
        sm_col1, sm_col2, sm_col3, sm_col4 = st.columns(4)
        with sm_col1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">⚡ Reading Time Saved</div>
                <div class="metric-value" style="color: #10b981;">{stats.get('savings_percent', 0)}% Faster</div>
                <div class="metric-sub">~{stats.get('summary_read_time', '1 min')} read time</div>
            </div>
            """, unsafe_allow_html=True)
            
        with sm_col2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">✍️ Summary Words</div>
                <div class="metric-value">{stats.get('summary_words', 0):,}</div>
                <div class="metric-sub">From {stats.get('original_words', 0):,} words</div>
            </div>
            """, unsafe_allow_html=True)
            
        with sm_col3:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">📊 Compression Factor</div>
                <div class="metric-value">1 : {stats.get('compression_ratio', 1)}</div>
                <div class="metric-sub">Concise reduction</div>
            </div>
            """, unsafe_allow_html=True)

        with sm_col4:
            engine_title = "Gemini 1.5 Pro" if stats.get('used_gemini') else "Transformer"
            engine_badge = "🟢 Abstractive AI" if stats.get('used_gemini') else "🟡 Extractive"
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">🤖 Active Engine</div>
                <div class="metric-value" style="font-size: 16px;">{engine_title}</div>
                <div class="metric-sub">{engine_badge}</div>
            </div>
            """, unsafe_allow_html=True)

    # Result Tabs
    res_tab1, res_tab2, res_tab3 = st.tabs([
        "🌟 Generated AI Summary",
        "🔍 Key Semantic Highlights",
        "📝 Side-by-Side Comparison"
    ])

    final_summary_text = st.session_state.summary_result if has_gemini else "\n\n".join([f"• {pt}" for pt in (st.session_state.extractive_result or [])])

    # TAB 1: AI Generated Summary
    with res_tab1:
        if has_gemini:
            st.markdown("""
            <div class="summary-header-badge">
                <span>✨ Gemini AI Synthesis</span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(st.session_state.summary_result)
        else:
            st.info("ℹ️ Local Extractive Mode: Below is the distilled summary composed of top semantically scored sentences from your document.")
            for idx, pt in enumerate(st.session_state.extractive_result or [], 1):
                st.markdown(f"**{idx}.** {pt}")

        # Action bar: Download & Export options
        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
        exp_col1, exp_col2, exp_col3 = st.columns([1, 1, 2])
        
        with exp_col1:
            st.download_button(
                label="📥 Download as TXT",
                data=final_summary_text,
                file_name="Tamil_Scribe_Summary.txt",
                mime="text/plain",
                use_container_width=True
            )
            
        with exp_col2:
            st.download_button(
                label="📄 Download as Markdown",
                data=f"# Document Summary\n\nGenerated by Tamil Scribe AI\n\n---\n\n{final_summary_text}",
                file_name="Tamil_Scribe_Summary.md",
                mime="text/markdown",
                use_container_width=True
            )

    # TAB 2: Semantic Highlights
    with res_tab2:
        st.markdown("**Top ranked sentences based on high-dimensional semantic cosine similarity:**")
        if st.session_state.extractive_result:
            for rank, sentence in enumerate(st.session_state.extractive_result, 1):
                st.markdown(f"""
                <div class="key-point-item">
                    <div style="font-size: 11px; font-weight: 700; color: #818cf8; text-transform: uppercase; margin-bottom: 4px;">
                        📌 Highlight #{rank}
                    </div>
                    {sentence}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.write("No highlights available.")

    # TAB 3: Side-by-Side Comparison
    with res_tab3:
        cmp_col1, cmp_col2 = st.columns(2)
        with cmp_col1:
            st.markdown("#### 📄 Original Source")
            st.markdown(f"""
            <div style="background: rgba(99, 102, 241, 0.06); border: 1px solid rgba(99, 102, 241, 0.2); border-radius: 12px; padding: 16px; max-height: 450px; overflow-y: auto; font-size: 14.5px; line-height: 1.7;">
                {current_text.replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)
            
        with cmp_col2:
            st.markdown("#### ✍️ Distilled Summary")
            st.markdown(f"""
            <div style="background: rgba(99, 102, 241, 0.06); border: 1px solid rgba(99, 102, 241, 0.2); border-radius: 12px; padding: 16px; max-height: 450px; overflow-y: auto; font-size: 14.5px; line-height: 1.7;">
                {final_summary_text.replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="custom-footer">
    <div style="font-size: 14px; font-weight: 600; color: #334155; margin-bottom: 6px;">
        📝 Tamil Scribe AI | AI-Powered Multilingual Document Summarization
    </div>
    <div style="font-size: 12px; color: #94a3b8;">
        Engineered with Google Gemini 1.5 &bull; PyTorch &bull; Sentence Transformers &bull; Streamlit
    </div>
</div>
""", unsafe_allow_html=True)