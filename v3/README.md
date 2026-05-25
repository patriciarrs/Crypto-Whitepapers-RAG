# Crypto Whitepapers RAG

A Retrieval-Augmented Generation (RAG) system that answers questions about the Bitcoin and Ethereum whitepapers using LangChain, Pinecone, and OpenAI. v2 adds a Gradio chat interface for interactive querying without writing any code. v3 adds RAGAS evaluation to measure retrieval and answer quality with objective metrics.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/patriciarrs/Rag-finalproject/blob/main/v3/RAG_FinalProject.ipynb)

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
- **Gradio UI** — interactive chat interface with example questions and a public shareable link
- **RAGAS evaluation** — scores the system on Faithfulness, Answer Relevancy, Context Precision, and Context Recall

---

## Tech Stack

| Component | Tool |
|---|---|
| Framework | LangChain |
| Vector Database | Pinecone |
| Embeddings | OpenAI `text-embedding-3-small` |
| LLM | OpenAI `gpt-4o-mini` |
| PDF Loader | PyPDFLoader |
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

### 4. Add the Ethereum whitepaper

The Bitcoin whitepaper loads automatically from a public URL. For Ethereum, download the PDF and upload it to your Colab session:

- Download: [ethereum.org/en/whitepaper](https://ethereum.org/en/whitepaper)
- In Colab: click the **Files** icon (📁) in the left sidebar → upload `ethereum.pdf`

### 5. Run the notebook

Run all cells from top to bottom. The ingestion step only needs to run once — after that, the chunks are stored in Pinecone and you can skip straight to inference.

---

## Project Structure

```
RAG_FinalProject.ipynb   # Main notebook
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
