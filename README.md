# 📝 Tamil Scribe AI - Multilingual Document Summarizer

An AI-powered document summarization web application designed specifically for **Tamil (தமிழ்)** and **English** documents. It supports both **Abstractive Summarization** (via Google Gemini 1.5) and **Local Semantic Extractive Summarization** (via Sentence Transformers).

---

## ✨ Features

- 🇮🇳 **Multilingual Support**: High-accuracy summarization for both Tamil and English documents with automatic language classification.
- 📄 **Multi-Format Processing**: Upload and summarize **PDF**, **DOCX (Microsoft Word)**, and **Plain Text (TXT)** documents.
- 🧠 **Dual AI Architecture**:
  - **Generative AI Mode**: Powered by Google Gemini 1.5 Pro with customizable styles (Bullet points, Executive summary, Top takeaways, Narrative synopsis).
  - **Local Extractive Mode**: Offline-ready fallback using `paraphrase-multilingual-MiniLM-L12-v2` with cosine semantic similarity ranking.
- 📊 **Smart Analytics**: Real-time word count, character count, estimated reading time, compression ratio, and time saved.
- 🎨 **Modern Theme-Adaptive UI**: Built with Streamlit featuring responsive glassmorphism styling, clean typography, and seamless Dark/Light mode support.
- 📥 **Export & Download**: One-click download of generated summaries as **TXT** or **Markdown (.md)** files.

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/DHANURAGAVIS466/Text-summarization.git
cd Text-summarization
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Configure Gemini API Key
To enable Generative AI summaries, create a file named `.streamlit/secrets.toml` in the root folder:
```toml
GEMINI_API_KEY = "your-google-gemini-api-key"
```
*(If no API key is provided, the application seamlessly runs using the local Sentence Transformer extractive engine).*

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```

---

## 🛠️ Tech Stack

- **Frontend / Framework**: Streamlit
- **Generative LLM**: Google Gemini 1.5 Pro (`google-generativeai`)
- **Semantic Embeddings**: `sentence-transformers` & PyTorch
- **Document Parsers**: `pdfplumber`, `docx2txt`

---

## 📄 License
MIT License
