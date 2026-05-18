# PDF Chatbot

A Streamlit chatbot that answers questions about an uploaded PDF using
semantic search (FAISS + sentence-transformers) and the Hugging Face
Inference API for answer generation.

## Features

- Upload any text-based PDF (multi-page, paragraphs and tables)
- Chunked semantic search with FAISS
- Switch between Mistral, Llama 3, and Falcon instruct models
- Shows the source chunks used to answer each question
- Per-session chat history
- Clear error messages for unreadable PDFs, timeouts, and API outages

## Project structure

```
pdf-chatbot/
├── app.py              # main Streamlit app
├── pdf_processor.py    # PDF extraction and chunking
├── embeddings.py       # FAISS index and search
├── llm.py              # Hugging Face API calls
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

1. **Clone and enter the directory**

   ```bash
   cd pdf-chatbot
   ```

2. **Create a virtual environment** (recommended)

   ```bash
   python -m venv .venv
   source .venv/bin/activate    # macOS/Linux
   # .venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure your Hugging Face token**

   ```bash
   cp .env.example .env
   ```

   Edit `.env` and set your token:

   ```
   HUGGING_FACE_API_TOKEN=hf_your_token_here
   ```

   Get a token at <https://huggingface.co/settings/tokens>.
   For Llama 3, also accept the model license on its model page.

5. **Run the app**

   ```bash
   streamlit run app.py
   ```

   The app will open in your browser at <http://localhost:8501>.

## How it works

1. **Extraction** — `pdfplumber` pulls text and tables from each page.
2. **Chunking** — text is split into ~500-character chunks with 50-character overlap.
3. **Embedding** — chunks are embedded with `sentence-transformers/all-MiniLM-L6-v2`
   and added to a FAISS inner-product index (cosine similarity on normalized vectors).
4. **Retrieval** — your question is embedded and the top-k most similar chunks are pulled.
5. **Generation** — the chunks plus your question are sent to the chosen Hugging Face
   model with an instruction to answer only from the context.

## Notes

- The first run downloads the embedding model (~90 MB).
- Scanned/image-only PDFs will fail extraction — OCR is not included.
- Some Hugging Face models (Llama 3) require explicit license acceptance on your account.
# PdfChatbot
