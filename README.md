# DocuChat

**[Live App](https://docuchat-pm07.streamlit.app/)**

A fully free, open-LLM document chat app. Upload PDFs, ask questions, get answers grounded in what the documents actually say. Built in 2024.

I built this because every "chat with your PDF" tool I could find was either paywalled, rate-limited into uselessness, or required handing your documents over to OpenAI. DocuChat costs nothing to run: Streamlit Community Cloud for hosting, HuggingFace Inference API free tier for the LLM, and local open-source embeddings. No API keys with billing attached, no document lock-in, no sneaky usage caps.

## What's under the hood

**PDF parsing** uses `unstructured` with title-based chunking. Rather than naively splitting on character count, it respects document structure (headings, sections, paragraphs). Tables are extracted separately from body text. Chunks are capped at 4000 characters with overlap tuned for coherent sections.

**Hybrid retrieval** combines dense semantic search (Qdrant in-memory + `all-MiniLM-L6-v2` embeddings) with sparse BM25 keyword search in a weighted ensemble. BM25 gets the heavier weight (0.7 vs 0.3 for dense) because document Q&A rewards precise keyword matching, and BM25 is fast without needing GPU.

**Cross-encoder reranking** with `mixedbread-ai/mxbai-rerank-xsmall-v1` does a second pass over retrieved chunks, scoring each one against the query with a proper cross-attention model rather than just cosine similarity. This is the difference between a retrieval pipeline that sort of works and one that actually surfaces the right context.

**Generation** via HuggingFace Inference API hitting Llama 3 8B Instruct. The chatbot maintains conversation history and streams output token-by-token.

## Setup

### HuggingFace Access Token

The model (`meta-llama/Meta-Llama-3-8B-Instruct`) is gated, so you need to accept Meta's licence before you can use it.

**Accept the licence first:**

1. Go to [huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
2. Click **Agree and access repository** and fill in the form
3. Approval is usually near-instant

**Create a token:**

1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click **New token**
3. Set type to **Fine-grained** (or **Read**)
4. Under Permissions, make sure **Make calls to Inference Providers** is enabled
5. Generate and copy the token

Paste the token into the sidebar field in the app. It's sent directly to HuggingFace and not stored anywhere.

## Running locally

```bash
git clone https://github.com/prodm93/docuchat.git
cd docuchat
pip install -r requirements.txt
streamlit run Hello.py
```

You'll need `poppler` for PDF rendering (`apt install poppler-utils` / `brew install poppler`) and `tesseract` for OCR (`apt install tesseract-ocr`).

## Stack

- Streamlit (UI + hosting)
- LangChain (RAG orchestration)
- Qdrant (in-memory vector store)
- sentence-transformers / `all-MiniLM-L6-v2` (local embeddings, no API call needed)
- rank-bm25 (sparse retrieval)
- mixedbread-ai/mxbai-rerank-xsmall-v1 (cross-encoder reranking)
- unstructured (PDF parsing)
- HuggingFace Inference API / Llama 3 8B Instruct (generation)

## Note on recent commits

This project was originally built and deployed in 2024, without subsequent regular maintenance. The recent commits are exclusively quick fixes for breaking upstream changes that accumulated since then, not feature work or refactoring:

- **Streamlit Community Cloud** updated its sandbox kernel to block executable stacks, which killed ChromaDB's default ONNX embedding function at import time (even when unused). Swapped to Qdrant in-memory as the vector store.
- **NLTK** renamed the `punkt` tokeniser resource to `punkt_tab`, breaking PDF text extraction.
- **HuggingFace** deprecated and shut down the `api-inference.huggingface.co` endpoint entirely, and the new inference router reclassified chat models from `text-generation` to `conversational`, requiring a switch from `text_generation()` to `chat_completion()`.

(All original dependencies were version-pinned at the time of deployment, so these were upstream interface changes over the course of two years rather than dependency drift. Please do reach out to me and/or submit a pull request if you'd like to use the app but run into trouble!)

## Known limitations

- **Large PDFs are slow.** Reranking is CPU-bound on Streamlit Community Cloud's free tier. Fine for research papers and reports, less ideal for 500-page manuals.
- **HuggingFace free tier has rate limits.** If you hit them, wait a minute and retry.
- **Scanned PDFs** will go through OCR via tesseract, but accuracy depends on scan quality. Natively digital PDFs work best.
- **Tables** are extracted but not currently fed into the RAG retrieval pipeline. They're collected separately.
