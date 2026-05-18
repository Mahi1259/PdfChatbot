# PDF Chatbot

Drop a PDF in, ask questions about it, get answers. Built with Streamlit on top of FAISS for retrieval and the Hugging Face Inference API for the actual answers.

I built this because skimming long PDFs is painful and Ctrl-F only gets you so far when you don't know the exact wording you're looking for.

## What it can do

- Takes any text-based PDF  multi-page, paragraphs, tables, the usual
- Splits the text into chunks and does semantic search over them with FAISS
- Lets you switch between Mistral, Llama 3, and Falcon (whichever you prefer)
- Shows you the chunks it used to answer, so you can sanity-check the response
- Keeps a chat history for the session
- Tries to fail gracefully  unreadable PDFs, API timeouts, that kind of thing

## What's in here

```
pdf-chatbot/
├── app.py              # the Streamlit UI
├── pdf_processor.py    # pulls text/tables out of the PDF
├── embeddings.py       # builds the FAISS index and runs the search
├── llm.py              # talks to the Hugging Face API
├── requirements.txt
├── .env.example
└── README.md
```

## Getting it running

1. **Grab the code**

   ```bash
   cd pdf-chatbot
   ```

2. **Set up a virtual environment**  strongly recommended unless you enjoy dependency conflicts

   ```bash
   python -m venv .venv
   source .venv/bin/activate    # macOS/Linux
   # .venv\Scripts\activate     # Windows
   ```

3. **Install the dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Add your Hugging Face token**

   ```bash
   cp .env.example .env
   ```

   Then open `.env` and drop in your token:

   ```
   HUGGING_FACE_API_TOKEN=hf_your_token_here
   ```

   You can get one from <https://huggingface.co/settings/tokens>. Heads up: if you want to use Llama 3, you'll also need to accept its license on the model page first  Hugging Face is strict about that one.

5. **Fire it up**

   ```bash
   streamlit run app.py
   ```

   It should pop open in your browser at <http://localhost:8501>.

## How it actually works

Roughly, the pipeline goes like this:

1. **Extraction**  `pdfplumber` reads the PDF and pulls out the text and any tables.
2. **Chunking**  that text gets sliced into ~500-character pieces with a bit of overlap (50 chars) so we don't cut sentences in half at the boundaries.
3. **Embedding**  each chunk gets turned into a vector with `sentence-transformers/all-MiniLM-L6-v2` and stored in a FAISS index. Inner-product search on normalized vectors, which is effectively cosine similarity.
4. **Retrieval**  when you ask something, your question gets embedded the same way and we grab the top-k closest chunks.
5. **Generation**  those chunks plus your question get sent to whichever Hugging Face model you picked, with a prompt telling it to answer only from what's in the context. (It mostly listens.)

## A few things to know

- First run will download the embedding model  around 90 MB, so give it a minute.
- This won't work on scanned PDFs or anything that's basically just images. There's no OCR step.
- Llama 3 (and a few others) need you to accept the license on Hugging Face before the API will let you use them. If you get a 403, that's usually why.
