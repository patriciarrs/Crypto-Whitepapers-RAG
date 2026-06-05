# Crypto Whitepapers RAG

A Retrieval-Augmented Generation (RAG) system that answers questions about the Bitcoin and Ethereum whitepapers using LangChain, Pinecone, and OpenAI. v2 adds a Gradio chat interface. v3 adds RAGAS evaluation. v4 adds multi-query + FlashRank reranking with a before/after comparison. v5 adds adaptive retrieval. v6 improves the prompt to enforce strict grounding and source citation, and uses RAGAS to measure the impact on faithfulness and answer relevancy. v7 loads both whitepapers from public URLs so the notebook runs end-to-end without any manual file uploads, and fixes a stale environment variable bug in `load_secrets()`.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/patriciarrs/Crypto-Whitepapers-RAG/blob/main/v7/Patr%C3%ADcia_RAG_FinalProject.ipynb)

---

## Overview

This project implements a RAG pipeline that:

1. Loads and preprocesses the Bitcoin and Ethereum whitepapers (PDFs)
2. Splits the text into overlapping chunks and stores them in Pinecone as vector embeddings
3. At query time, retrieves the most relevant chunks and passes them to an LLM to generate an answer
4. Maintains conversation history to support multi-turn follow-up questions

---

## Features

- **Text preprocessing** — cleans PDF noise (extra whitespace, page numbers) before indexing
- **Auto topic detection** — uses an LLM to detect and tag each document's topic automatically
- **Semantic search** — retrieves relevant chunks from Pinecone using cosine similarity
- **Chat history** — keeps track of the last 5 conversation turns for context-aware answers
- **Gradio UI** — chat interface with a strategy selector dropdown and a public shareable link
- **RAGAS evaluation** — scores the system on Faithfulness, Answer Relevancy, Context Precision, and Context Recall
- **Multi-query retrieval** — generates query variations to retrieve more relevant chunks
- **FlashRank reranking** — re-scores retrieved chunks with a cross-encoder for better relevance ordering
- **Adaptive retrieval** — runs baseline first; escalates to multi-query + reranking only when confidence score < 0.75
- **Improved prompt** — enforces strict grounding, requires source citation, and uses an explicit refusal phrase when context is insufficient
- **Prompt injection defence** — instructs the LLM to ignore any instructions embedded in user questions or retrieved documents
- **Tone alignment** — matches response register to the complexity of the question
- **Frustration detection** — acknowledges when the user repeats a question or signals dissatisfaction and tries a different angle
- **Public URL ingestion** — both whitepapers load automatically from public URLs; no manual file upload required

---

## What Changed in v7

Two small fixes to make the notebook fully self-contained:

**1. Ethereum whitepaper loaded from a public URL**

Previously, the Ethereum PDF had to be downloaded and uploaded manually to the Colab session before ingestion. The notebook now loads it directly from `ethereum.org`. The manual upload path is kept as a commented fallback in case the URL is unavailable.

**2. Stale environment variable fix in `load_secrets()`**

`load_secrets()` now explicitly deletes an existing key from `os.environ` before setting the new value. Without this, re-running the setup cell in the same session could leave a stale key silently in place, causing hard-to-diagnose authentication errors on a second run.

No changes were made to the retrieval pipeline, prompt, evaluation logic, or any other part of the system. The RAGAS scores from v6 carry forward unchanged as the final benchmark.

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

### 4. Run the notebook

Run all cells from top to bottom. Both whitepapers load automatically from public URLs — no file uploads needed. The ingestion step only needs to run once; after that, the chunks are stored in Pinecone and you can skip straight to inference.

---

## Project Structure

```
Patrícia_RAG_FinalProject.ipynb   # Main notebook
README.md                         # This file
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
