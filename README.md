# Crypto Whitepapers RAG

A Retrieval-Augmented Generation (RAG) system that answers questions about the Bitcoin and Ethereum whitepapers. Built iteratively across six versions, each motivated by the findings of the previous one.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/patriciarrs/Crypto-Whitepapers-RAG/blob/main/v7/Patr%C3%ADcia_RAG_FinalProject.ipynb)

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

**Initial results (v3, before fix):**

| Metric | Score |
|---|---|
| Faithfulness | 0.748 |
| Answer Relevancy | 0.924 |
| Context Precision | 0.917 |
| Context Recall | 0.583 |

**Findings:** context recall was the weakest metric at 0.583. Two questions scored 0.0 recall. Initial diagnosis pointed to the Ethereum PDF not being ingested — but checking Pinecone confirmed the chunks were there. The root cause turned out to be duplicate ingestion: running `ingest_documents` more than once created multiple copies of the same chunks in Pinecone, causing the retriever to return the same chunk several times instead of distinct ones. The index was deleted, recreated, and re-ingested once to fix this. A second issue — ground truth mismatch on Q6, whose ground truth described content not present in the retrieved chunks — was also identified here but left for v4. After fixing the duplicate ingestion, scores improved significantly:

**Results (v3, after fixing duplicate ingestion):**

| Metric | Score |
|---|---|
| Faithfulness | 0.925 |
| Answer Relevancy | 0.908 |
| Context Precision | 0.903 |
| Context Recall | 0.917 |

The recall jump from 0.583 to 0.917 is fully explained by the fix. The faithfulness jump (+0.177) is harder to attribute: deduplicating chunks gives the LLM more diverse context instead of the same passage repeated, which could plausibly improve grounding, but a gain of this size is also consistent with LLM non-determinism across runs. Q5 (incentive for nodes) and Q6 (Ethereum vs Bitcoin) still scored below 1.0 on some metrics, pointing to the remaining ground truth mismatch as the next issue to address.

**Struggle:** `scores.to_pandas()` in RAGAS drops the `question` column from the output DataFrame. Adding it back manually with `df.insert(0, "question", ...)` is the fix. Also encountered a `numpy` binary incompatibility error (`numpy.dtype size changed`) caused by `ragas` and `datasets` being compiled against `numpy < 2.0` while a newer version was already loaded in memory. This was resolved by pinning `ragas==0.2.6`, which is compatible with Colab's numpy environment and does not require a runtime restart after installation.

---

### v4 — Multi-query + Reranking

**Motivation:** v3 showed that even after fixing duplicate ingestion, some questions still scored below 1.0 on recall. Multi-query retrieval generates multiple variations of the query and searches with each, which should improve coverage. FlashRank reranking then re-scores the combined results with a cross-encoder, which is more accurate than cosine similarity alone.

The ground truth mismatch identified in v3 was resolved here by rewriting the Q6 ground truth to accurately reflect the content of the Ethereum whitepaper chunks. The v3 post-fix baseline (0.917 recall) was used as the starting point for v4 comparisons.

The Gradio interface was also updated: a strategy selector dropdown was added so users can switch between Baseline and Multi-query + Reranking live in the chat without restarting the session.

**Struggle:** adding `additional_inputs` to `gr.ChatInterface` requires each entry in `examples` to be a list that includes a value for every input — `[message, strategy]` — not a flat string. Gradio raises a `ValueError` that makes this clear but the fix is not obvious at first glance.

**Results (v4):**

| Metric | Baseline | Multi-query + Reranking | Δ |
|---|---|---|---|
| Faithfulness | 0.931 | 0.955 | +0.024 |
| Answer Relevancy | 0.902 | 0.929 | +0.027 |
| Context Precision | 0.944 | 0.856 | -0.088 |
| Context Recall | 0.903 | 0.778 | -0.125 |

**Findings:** the v4 baseline differs slightly from the v3 post-fix baseline (e.g. faithfulness 0.931 vs 0.925) despite no change to the retrieval or generation pipeline — only the Q6 ground truth string was rewritten. These small differences are attributable to LLM non-determinism across runs rather than any system change. The within-run deltas between baseline and multi-query are more meaningful since both strategies were evaluated in the same run under the same conditions.

Multi-query + reranking improved faithfulness and answer relevancy, but degraded both precision and recall noticeably. The explanation: multi-query generates a large pool of candidate chunks, some loosely related. FlashRank reranks this pool but occasionally promotes off-topic chunks above relevant ones, and the `top_n=4` cutoff discards some chunks that RAGAS considers necessary. This is a known recall/precision tradeoff — broader retrieval improves answer quality when it finds the right chunks, but introduces noise that hurts ranking and coverage.

The key insight was that multi-query is not always better, and applying it to every query pays an unnecessary cost (extra LLM calls to generate variations) on questions the baseline already handles well.

---

### v5 — Adaptive Retrieval

**Motivation:** v4 showed that multi-query helps on some questions and hurts on others. The logical response was not to pick one strategy and apply it everywhere, but to choose dynamically based on how confident the baseline result actually is.

Adaptive retrieval works in two steps: run `similarity_search_with_score` to get cosine similarity scores alongside the chunks, then check the top score against a threshold (0.75). If the score is at or above the threshold, the baseline result is good enough and is used directly (cheap path). If it falls below, the system escalates to multi-query + reranking (expensive path). The threshold is a named constant so it can be tuned without touching the logic.

**Results (v5):**

| Metric | Baseline | Multi-query + Reranking | Adaptive | Δ Baseline→Adaptive |
|---|---|---|---|---|
| Faithfulness | 0.853 | 0.906 | 0.930 | +0.078 |
| Answer Relevancy | 0.899 | 0.911 | 0.934 | +0.034 |
| Context Precision | 0.972 | 0.843 | 0.847 | -0.125 |
| Context Recall | 0.861 | 0.944 | 0.889 | +0.028 |

**Findings:** Adaptive delivered the best scores on faithfulness and answer relevancy across all three strategies, improving on the baseline by +0.078 and +0.034 respectively. Multi-query led on context recall (0.944), while Adaptive sat between multi-query and baseline on that metric (0.889). Neither advanced strategy matched the baseline's context precision (0.972), reflecting the same recall/precision tradeoff seen in v4: broader retrieval improves answer quality but introduces some noise that affects ranking. Overall, Adaptive is the best balance — it improves generation quality over the baseline without the full precision cost of always running multi-query, and the expensive pipeline only triggers on questions where the baseline confidence is genuinely low.

One caveat: the v5 baseline faithfulness (0.853) is notably lower than the v4 baseline (0.931) despite being the same system. This gap is too large to attribute entirely to normal variance and likely reflects LLM non-determinism across separate runs. It means the Adaptive delta of +0.078 on faithfulness should be read as a directional signal rather than a precise measurement — the true gain on a given run may be larger or smaller.

---

### v6 — Prompt Engineering

**Motivation:** across v1–v5 all improvements targeted retrieval. Adaptive retrieval in v5 pushed faithfulness to 0.930 but there was still room to improve it further on the generation side — the LLM's answers given what was retrieved. The prompt was the lever left untouched.

The improved prompt adds three grounding rules: a strict no-prior-knowledge rule, a requirement to cite which document (Bitcoin or Ethereum whitepaper) supports each claim, and an explicit refusal phrase for when the context is insufficient. It also adds two UX-oriented instructions: tone alignment (matching response register to the complexity of the question) and frustration detection (acknowledging when the user repeats a question or signals dissatisfaction and trying a different angle).

To isolate the prompt as the only variable, both prompts are evaluated using the same adaptive retrieval strategy from v5. `create_rag_chain()` was refactored to accept a `prompt_template` parameter, with `IMPROVED_PROMPT` as the default. `ORIGINAL_PROMPT` is kept as a named constant for comparison. `run_ragas_evaluation()` was updated to accept the same parameter, so any combination of retrieval strategy and prompt can be evaluated with one function call.

**Other prompt engineering concerns considered and not addressed:**

- **RBAC and governance** — not applicable. The system has no concept of users or roles. Every user sees the same two documents, so there is nothing to restrict or govern.
- **Red-teaming** — a testing methodology, not a prompt feature. Documenting adversarial attack scenarios and proving resistance to them is a project on its own, separate from what a prompt change can demonstrate.
- **Toxicity filtering** — not applicable to this domain. A Q&A system about cryptocurrency whitepapers has essentially no toxicity risk. Adding a filter would be defensive engineering against a threat that does not exist here.

**Results (v6):**

| Metric | Original Prompt | Improved Prompt | Δ |
|---|---|---|---|
| Faithfulness | 0.860 | 0.934 | +0.074 |
| Answer Relevancy | 0.897 | 0.895 | -0.002 |
| Context Precision | 0.889 | 0.843 | -0.046 |
| Context Recall | 0.944 | 0.944 | +0.000 |

**Findings:** faithfulness improved by +0.074, which is exactly what the prompt was designed to target. The strict no-prior-knowledge rule and source citation requirement directly reduce the LLM's tendency to draw on knowledge outside the retrieved context. Context recall was unchanged (0.944 both), meaning the improved prompt neither helped nor hurt retrieval coverage. Context precision dropped slightly (-0.046): the stricter grounding instructions occasionally cause the LLM to lean on chunks it cites explicitly, which can affect how RAGAS scores ranking quality. Answer relevancy was essentially flat (-0.002), an acceptable tradeoff for a meaningful faithfulness gain.

---

## Key Takeaways

**RAGAS scores are directional, not exact.** The LLM runs with non-zero temperature and RAGAS uses its own internal LLM calls to score answers, so results vary between runs. Within-version deltas (same run, different strategies) are the most reliable signal. Cross-version baseline comparisons should be treated as approximate — a gap of a few hundredths is likely variance; a consistent directional pattern across metrics is meaningful.

**Prompt engineering is measurable — selectively.** Strict grounding and source citation improved faithfulness and context recall in ways RAGAS can quantify. Tone alignment and frustration detection improve user experience but are better evaluated qualitatively in a demo. Knowing which tool fits which concern — RAGAS vs human judgment — is itself a skill worth demonstrating.

**Data quality matters more than algorithm complexity.** The single biggest recall improvement in the project came from fixing duplicate ingestion, not from any retrieval technique.

**Measure before optimising.** Adding RAGAS in v3 before attempting any retrieval improvements meant that every subsequent change could be validated with numbers. Without it, the v4 multi-query experiment would have looked like a success based on intuition alone — the scores showed it actually hurt two metrics.

**More complex retrieval does not automatically win.** Multi-query + reranking improved faithfulness but degraded precision and recall versus the clean baseline. The right question is not "which strategy is best" but "which strategy is best for this question."

**Evaluation datasets require careful ground truth design.** Two questions scored 0.0 context recall in v3 not because the retrieval was broken but because of duplicate ingestion and a ground truth mismatch. RAGAS is only as useful as the quality of the data it is scored against.

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
RAG_FinalProject.ipynb   # Main notebook (v6)
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
