# Dulset — Complete Setup & Configuration Guide

This guide explains how your Dulset pipeline works, how to configure API keys, and how to run it locally or on Vercel.

---

## 1. Zero-Key Default (Out of the Box)

By default, your app runs **100% locally and free without requiring any paid API keys**:
- **Speech-to-Text**: Handled directly in the user's browser using the native **Web Speech API** (Chrome / Edge).
- **RAG Embeddings & Search**: Handled via lightweight **Hashing-TFIDF + BM25 Okapi + Cosine Similarity** (<1ms latency).
- **Answer Generation**: Handled via **Extractive Passage Synthesis** (grounded in the MSMARCO-XI dataset).

---

## 2. How & Where to Add API Keys (Optional)

If you want to plug in external cloud AI providers, **never write keys into Python or JavaScript files**. Use environment variables:

### Option A: Local Development (`.env` file)
1. Create a file named `.env` in your project root (`voice-rag/.env`).
2. Add your keys:

```ini
# --- Optional: Sarvam AI (for Speech-to-Text) ---
STT_PROVIDER=sarvam
SARVAM_API_KEY=your_sarvam_api_key_here

# --- Optional: ElevenLabs (for Speech-to-Text) ---
# STT_PROVIDER=elevenlabs
# ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
```

*(Note: `.env` is automatically ignored by `.gitignore` so your keys are never uploaded to GitHub).*

---

### Option B: On Vercel Cloud
1. Go to your [Vercel Dashboard](https://vercel.com/dashboard).
2. Select your project **voice-rag** → **Settings** → **Environment Variables**.
3. Add these variables for **Production** (and Preview if needed):

```ini
PINECONE_API_KEY=your_new_pinecone_key
PINECONE_INDEX_NAME=voicerag-index
PINECONE_NAMESPACE=
# Optional but recommended for a new Pinecone index:
# PINECONE_HOST=https://your-index-host.pinecone.io

# Optional voice services
SARVAM_API_KEY=your_sarvam_key
STT_PROVIDER=sarvam
ELEVENLABS_API_KEY=your_elevenlabs_key
```

4. Click **Save**, then redeploy. On Vercel, the API uses Pinecone only;
   it does not build or download the MSMARCO-XI dataset.

Create the Pinecone index before running the importer with dimension `384`,
metric `cosine`, and the exact value used for `PINECONE_INDEX_NAME`. Copy the
index host from Pinecone into `PINECONE_HOST` for a new index. Use the same
index name, namespace, and host in GitHub Actions and Vercel.

The Pinecone index must already be populated by running
`scripts/index_to_pinecone.py` from a machine with internet access. That
script streams MSMARCO-XI directly from Hugging Face and does not save the
dataset locally.

**Security:** the Pinecone key previously pasted into chat is exposed. Revoke
it in Pinecone, create a replacement, and add only the replacement to Vercel.

---

## 3. How to Run Locally

```bash
python run.py
```
Open **[http://localhost:8000](http://localhost:8000)** in Chrome or Edge.

---

## 4. How to Deploy to Vercel

1. **Commit & Push to GitHub**:
   Ensure `data/index/` and `src/` are included in your commit.
2. **Turn off Vercel Authentication / Deployment Protection**:
   In your Vercel project: **Settings** → **Deployment Protection** → set **Vercel Authentication** to **Disabled**.
3. **Deploy**:
   Vercel will build and serve your app globally with zero configuration!

After deployment, verify `https://your-project.vercel.app/api/health`. A healthy
response should show `status: "ok"`, `embedder: "hashing-tfidf"`, and a
non-zero `vectors_indexed` value. A Pinecone configuration problem is reported
explicitly instead of silently using sample data.
