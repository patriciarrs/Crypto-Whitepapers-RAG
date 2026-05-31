# Crypto Whitepapers RAG

A Retrieval-Augmented Generation (RAG) system that answers questions about the Bitcoin and Ethereum whitepapers. Built iteratively across five versions, each motivated by the findings of the previous one.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/patriciarrs/Rag-finalproject/blob/main/v5/RAG_FinalProject.ipynb)

---

## Tech Stack

| Component | Tool |
|---|---|
| Framework | LangChain |
| Vector Database | Pinecone |
| Embeddings | OpenAI `text-embedding-3-small` |
| LLM | OpenAI `gpt-4o-mini` |
| PDF Loader | PyPDFLoader |
| Retrieval | Baseline · Multi-query + FlashRank · Adaptive |
| User Interface | Gradio |
| Evaluation | RAGAS |
| Environment | Google Colab |

---

## The Journey

### v1 — Baseline RAG

**Goal:** build the mandatory foundation: ingest two whitepapers and answer questions about them.

The pipeline loads the Bitcoin and Ethereum whitepapers, cleans PDF noise (extra whitespace, page numbers), splits the text into overlapping chunks (1000 chars, 200 overlap), embeds them with OpenAI `text-embedding-3-small`, and stores them in Pinecone. At query time, the top-3 most similar chunks are retrieved and passed to `gpt-4o-mini` with a prompt template to generate an answer. Chat history (last 5 turns) is included so follow-up questions work correctly.

Two small decisions worth noting: a lightweight LLM (`gpt-3.5-turbo`, `temperature=0`) auto-detects each document's topic and tags it as metadata on every chunk, and a `load_secrets()` helper handles API keys in both Colab and local `.env` environments without changing any code.

---

### v2 — Gradio Interface

**Motivation:** the system worked but required writing code to use it. A chat interface would make it accessible and easier to demo.

Added a `gr.ChatInterface` with example questions and a public shareable link via `share=True`. Gradio manages conversation history internally in `[user, assistant]` pair format, which required a conversion step to match the `{role, content}` format used internally.

---

### v3 — RAGAS Evaluation

**Motivation:** the system produced answers but there was no way to know whether they were actually good. Without measurement, any future improvement would be unverifiable.

Added RAGAS evaluation with four metrics: Faithfulness (is the answer grounded in retrieved context?), Answer Relevancy (does it address the question?), Context Precision (are the best chunks ranked first?), and Context Recall (were all necessary chunks retrieved?). A hand-written evaluation dataset of 6 questions with ground truth answers was added.

**Results (v3):**

| Metric | Score |
|---|---|
| Faithfulness | 0.748 |
| Answer Relevancy | 0.924 |
| Context Precision | 0.917 |
| Context Recall | 0.583 |

**Findings:** context recall was the weakest metric at 0.583. Two questions scored 0.0 recall. Initial diagnosis pointed to the Ethereum PDF not being ingested — but checking Pinecone confirmed the chunks were there. The real issue was a combination of two problems: ground truth mismatch (the Q6 ground truth described content that wasn't in the retrieved chunks) and duplicate ingestion (running `ingest_documents` more than once created multiple copies of the same chunks in Pinecone, causing the retriever to return the same chunk 5 times instead of 5 different ones).

**Struggle:** `scores.to_pandas()` in RAGAS drops the `question` column from the output DataFrame. Adding it back manually with `df.insert(0, "question", ...)` is the fix. Also encountered a `numpy` binary incompatibility error (`numpy.dtype size changed`) caused by `ragas` and `datasets` being compiled against `numpy < 2.0` while a newer version was already loaded in memory. Fix: pin `numpy<2.0` at the top of the installation cell and restart the runtime before importing anything.

---

### v4 — Multi-query + Reranking

**Motivation:** context recall was 0.583 and the root cause (beyond the data issues) was that a single query phrasing sometimes missed relevant chunks. Multi-query retrieval generates multiple variations of the query and searches with each, which should improve coverage. FlashRank reranking then re-scores the combined results with a cross-encoder, which is more accurate than cosine similarity alone.

The Gradio interface was also updated: a strategy selector dropdown was added so users can switch between Baseline and Multi-query + Reranking live in the chat without restarting the session.

After fixing the duplicate ingestion issue (deleting and recreating the Pinecone index, then re-ingesting once), the baseline recall jumped from 0.583 to 0.861 — the largest single improvement in the entire project, and it came from a data quality fix, not an algorithmic one.

**Struggle:** adding `additional_inputs` to `gr.ChatInterface` requires each entry in `examples` to be a list that includes a value for every input — `[message, strategy]` — not a flat string. Gradio raises a `ValueError` that makes this clear but the fix is not obvious at first glance.

**Results (v4):**

| Metric | Baseline | Multi-query + Reranking | Δ |
|---|---|---|---|
| Faithfulness | 0.932 | 0.938 | +0.006 |
| Answer Relevancy | 0.903 | 0.904 | +0.001 |
| Context Precision | 1.000 | 0.782 | -0.218 |
| Context Recall | 0.861 | 0.833 | -0.028 |

**Findings:** multi-query + reranking did not universally improve results. Faithfulness improved marginally, but context precision dropped significantly (-0.218) and context recall slightly (-0.028). The explanation: multi-query generates a large pool of candidate chunks, some loosely related. FlashRank reranks this pool but occasionally promotes off-topic chunks above relevant ones. The `top_n=4` cutoff also discards some chunks that RAGAS considers necessary. This is a known recall/precision tradeoff — more retrieval improves coverage but introduces noise that hurts ranking.

The key insight was that multi-query is not always better, and applying it to every query pays an unnecessary cost (extra LLM calls to generate variations) on questions the baseline already handles well.

---

### v5 — Adaptive Retrieval

**Motivation:** v4 showed that multi-query helps on some questions and hurts on others. The logical response was not to pick one strategy and apply it everywhere, but to choose dynamically based on how confident the baseline result actually is.

Adaptive retrieval works in two steps: run `similarity_search_with_score` to get cosine similarity scores alongside the chunks, then check the top score against a threshold (0.75). If the score is at or above the threshold, the baseline result is good enough and is used directly (cheap path). If it falls below, the system escalates to multi-query + reranking (expensive path). The threshold is a named constant so it can be tuned without touching the logic.

**Results (v5):**

| Metric | Baseline | Multi-query + Reranking | Adaptive | Δ Baseline→Adaptive |
|---|---|---|---|---|
| Faithfulness | 0.915 | 0.959 | 0.898 | -0.018 |
| Answer Relevancy | 0.907 | 0.933 | 0.927 | +0.020 |
| Context Precision | 0.972 | 0.856 | 0.843 | -0.130 |
| Context Recall | 0.861 | 0.917 | 1.000 | +0.139 |

**Findings:** Adaptive achieved perfect context recall (1.000) — the only strategy to do so, and a complete reversal from the 0.583 recorded in v3. The precision drop (-0.130) reflects the same recall/precision tradeoff seen in v4: when the system escalates, it prioritises retrieving all relevant information over ranking order. For a Q&A use case where missing information is more costly than including some noise, this is an acceptable tradeoff. The adaptive design also means the expensive multi-query pipeline only runs when it is actually needed, keeping costs lower than always using multi-query.

---

## Key Takeaways

**Data quality matters more than algorithm complexity.** The single biggest recall improvement in the project came from fixing duplicate ingestion, not from any retrieval technique.

**Measure before optimising.** Adding RAGAS in v3 before attempting any retrieval improvements meant that every subsequent change could be validated with numbers. Without it, the v4 multi-query experiment would have looked like a success based on intuition alone — the scores showed it actually hurt two metrics.

**More complex retrieval does not automatically win.** Multi-query + reranking improved faithfulness but degraded precision and recall versus the clean baseline. The right question is not "which strategy is best" but "which strategy is best for this question."

**Evaluation datasets require careful ground truth design.** Two questions scored 0.0 context recall in v3 not because the retrieval was broken but because the ground truths described content that was not in the retrieved chunks. RAGAS is only as useful as the quality of the ground truths it is scored against.

---

## Getting Started

### 1. Prerequisites

- A [Pinecone](https://pinecone.io) account (free tier works)
- An [OpenAI](https://platform.openai.com) API key

### 2. Open in Colab

Click the **Open in Colab** badge at the top of this README.

### 3. Set up secrets

In Colab, open the **Secrets** panel (🔑 icon in the left sidebar) and add:

| Key | Where to find it |
|---|---|
| `OPENAI_API_KEY` | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| `PINECONE_API_KEY` | Pinecone Console → API Keys |

### 4. Add the Ethereum whitepaper

The Bitcoin whitepaper loads automatically from a public URL. For Ethereum, download the PDF and upload it to your Colab session:

- Download: [ethereum.org/en/whitepaper](https://ethereum.org/en/whitepaper)
- In Colab: click the **Files** icon (📁) in the left sidebar → upload `ethereum.pdf`

### 5. Run the notebook

Run all cells from top to bottom. The ingestion step only needs to run once — after that, the chunks are stored in Pinecone and you can skip straight to inference.

> ⚠️ After running Section 3 (Installation), go to **Runtime → Restart session** before continuing. Skipping this causes a numpy binary incompatibility error on import.

---

## Project Structure

```
RAG_FinalProject.ipynb   # Main notebook (v5)
ethereum.pdf             # Ethereum whitepaper (you provide this)
README.md                # This file
```

---

## Example Questions

```
What is Bitcoin?
How does proof-of-work prevent double spending?
What is the purpose of the timestamp server?
What is Ethereum and how is it different from Bitcoin?
What is the incentive for miners to support the network?
```
