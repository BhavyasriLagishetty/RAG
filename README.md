# RAG Document Q&A
### Retrieval-Augmented Generation with Groq & Llama3

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Groq](https://img.shields.io/badge/Groq-API-FF6B35?style=for-the-badge&logo=groq&logoColor=white)](https://groq.com/)
[![Llama3](https://img.shields.io/badge/Llama3.1-8B-0467DF?style=for-the-badge&logo=meta&logoColor=white)](https://ai.meta.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

> An intelligent document questioning system that lets you chat with your PDF research papers using **state-of-the-art retrieval-augmented generation**. No manual searching. No reading through hundreds of pages. Just ask questions in natural language and get precise, sourced answers instantly.

[![GitHub](https://img.shields.io/badge/GitHub-BhavyasriLagishetty-181717?style=flat-square&logo=github)](https://github.com/BhavyasriLagishetty)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-BhavyasriLagishetty-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/bhavyasri-lagishetty/)
[![Email](https://img.shields.io/badge/Email-bhavyasri.lagishetty%40gmail.com-EA4335?style=flat-square&logo=gmail)](mailto:bhavyasri.lagishetty@gmail.com)

---

## What Is This?

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline using **Streamlit**, **LangChain**, and **Groq's** ultra-fast inference. It processes PDF research papers, converts them into searchable vector embeddings using **FAISS** and **HuggingFace** models, then answers your questions by retrieving relevant context and generating responses with **Llama3.1-8B**.

The result: an AI research assistant that goes from raw documents to intelligent answers — entirely self-contained.

---

## Key Features

- **Custom research paper processing** — handles multiple PDFs with automatic text extraction
- **9-beam semantic search** — FAISS-powered vector similarity with HuggingFace all-MiniLM-L6-v2 embeddings
- **11-dimensional state retrieval** — document chunks + context matching + relevance scoring
- **Deep document understanding** — RecursiveCharacterTextSplitter with 1000-char chunks, 200-char overlap
- **Groq LPU inference** — sub-second response times with Llama3.1-8B
- **Dense retrieval scoring** — similarity matching, relevance ranking, context windowing
- **Persistent vector store** — FAISS index auto-saves in session state; recreates on demand
- **Live Q&A interface** — real-time question answering with source chunk display
- **Source transparency** — view exactly which document sections support each answer
- **Clean modular codebase** — document loading, embeddings, retrieval, generation all separated

---

## Demo
plain
Copy
      ──── Document Processing Pipeline ────
PDFs → Load → Split → Embed → Store → Retrieve → Generate → Answer
↓      ↓      ↲       ↓       ↓        ↓          ↓         ↓
[Attention.pdf]  [1000-char chunks]  [FAISS]  [Top-k docs]  [Llama3.1-8B]
[LLM.pdf]        [all-MiniLM-L6-v2]  [Index]  [Context]     [Response]
plain
Copy

The system always retrieves the most relevant document chunks first. The user asks questions in natural language — the agent retrieves context, generates answers, and displays source references.

---

## Architecture
┌─────────────────────────────────────────────────────────────┐
│                        RAG Pipeline                         │
│                                                             │
│  Documents (PDFs)     Processing Layer       Vector Store   │
│  ┌──────────────┐     ┌─────────────┐       ┌──────────┐   │
│  │ Attention.pdf│────▶│ PyPDF Loader│──────▶│          │   │
│  │ LLM.pdf      │     │ Text Splitter       │  FAISS   │   │
│  └──────────────┘     │ Embeddings  │──────▶│  Index   │   │
│                       └─────────────┘       └────┬─────┘   │
│                                                  │          │
│  Query (User)        Retrieval Layer             │          │
│  ┌──────────────┐     ┌─────────────┐            │          │
│  │ "What is...?"│────▶│ Similarity  │◀───────────┘          │
│  │              │     │ Search      │                       │
│  └──────────────┘     └──────┬──────┘                       │
│                              │                              │
│  Generation Layer            ▼                              │
│  ┌──────────────┐     ┌─────────────┐      ┌──────────┐    │
│  │ Prompt with  │◀────│ Top-k Chunks│      │ Answer   │    │
│  │ Context      │     │ (5 docs)    │      │ + Sources│    │
│  └──────┬───────┘     └─────────────┘      └──────────┘    │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────┐                                           │
│  │ Groq LLM     │────▶ Streaming Response                   │
│  │ Llama3.1-8B  │                                           │
│  └──────────────┘                                           │
└─────────────────────────────────────────────────────────────┘
plain
Copy

### Neural Network Components

| Component | Architecture | Purpose |
|-----------|--------------|---------|
| Input | 11 values (9 sensors + angle + distance) | State representation |
| Embeddings | all-MiniLM-L6-v2 (384-dim) | Convert text to vectors |
| Vector Store | FAISS (Facebook AI) | Fast similarity search |
| LLM | Llama3.1-8B-Instruct (8B params) | Generate natural language answers |
| Context Window | 5 retrieved chunks + query | Grounded generation |

---

## How the System Works

### 1 — Ingest
The system reads PDF documents from the `research_papers/` directory:

- **PyPDFDirectoryLoader** extracts text from all PDFs
- **RecursiveCharacterTextSplitter** creates 1000-character chunks with 200-character overlap
- **HuggingFaceEmbeddings** (all-MiniLM-L6-v2) converts chunks to 384-dimensional vectors
- **FAISS** indexes vectors for millisecond retrieval

### 2 — Retrieve (Similarity Search)
When you ask a question:
user_query → Embedding Model → FAISS Search → Top-5 Chunks Retrieved
plain
Copy

1. User query is embedded using the same model
2. FAISS performs similarity search across all document chunks
3. Top-5 most relevant chunks are retrieved as context

### 3 — Generate (Groq Inference)

```python
retrieved_context = " ".join(top_5_chunks)
prompt = f"Answer using only this context:\n\n{retrieved_context}\n\nQuestion: {user_query}"
response = llama3_1_8b.generate(prompt)
The Llama3.1-8B model generates answers strictly from provided context — no hallucinations, no external knowledge.
4 — Display
Streaming answer display with markdown formatting
Source chunks expandable in sidebar
Response time measurement (typically 0.5-2 seconds)
Document similarity scores visible
Usage Phases
Table
Phase	Sessions	What the System Does
Setup	1	Install dependencies, set API keys, load PDFs
Indexing	1	Create embeddings, build FAISS index (one-time)
Exploration	1-5	Basic questions, testing retrieval quality
Research	5-20	Deep questions, cross-document analysis
Mastery	20+	Complex multi-hop queries, full paper synthesis
The Documents — AI Research Papers
A hand-selected collection of foundational AI papers for immediate testing:
plain
Copy
              [Attention Is All You Need]
                     (Transformer)
                          ↓
              [Large Language Models Survey]
                     (LLM Overview)
Table
Paper	Content	Key Topics
Attention.pdf	"Attention Is All You Need" (Vaswani et al.)	Transformers, self-attention, positional encoding
LLM.pdf	LLM survey paper	Architecture, training, capabilities, limitations
The environment handles academic formatting, mathematical notation, citations, and multi-column layouts.
Project Structure
plain
Copy
rag-document-qa/
│
├── app.py                   ← Entry point — run this
├── requirements.txt         ← All dependencies
├── .env                     ← API keys (create this)
├── .gitignore
├── README.md
│
├── research_papers/         ← Put your PDFs here
│   ├── Attention.pdf        ← Sample: Transformer paper
│   └── LLM.pdf              ← Sample: LLM survey
│
└── (Session State)          ← Runtime created
    └── vectors              ← FAISS index (in-memory)
Getting Started
Prerequisites
Python 3.8 or higher
pip
Groq API key (free at console.groq.com)
Installation
bash
Copy
# 1. Clone the repository
git clone https://github.com/BhavyasriLagishetty/RAG.git
cd RAG

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install streamlit langchain-groq langchain-community langchain-text-splitters python-dotenv faiss-cpu

# 4. Set up environment variables
cp .env.example .env            # Or create manually
# Edit .env and add your GROQ_API_KEY

# 5. Prepare documents
mkdir -p research_papers
# Copy your PDFs into research_papers/ folder
Run Application
bash
Copy
streamlit run app.py
Two interfaces open immediately:
Streamlit web app — interactive Q&A interface at http://localhost:8501
Terminal logs — processing status and debug information
The app resumes with existing embeddings if session state persists.
Hyperparameters
Table
Parameter	Value	Effect
CHUNK_SIZE	1000 chars	Large context windows per chunk
CHUNK_OVERLAP	200 chars	Overlap maintains context continuity
EMBEDDING_MODEL	all-MiniLM-L6-v2	Fast, high-quality sentence embeddings
VECTOR_DIM	384	Dense vector representation
TOP_K_RETRIEVAL	5 chunks	Balance between context breadth and precision
LLM_MODEL	llama-3.1-8b-instant	Fast, capable instruction-following model
TEMPERATURE	0.7	Balanced creativity and determinism
MAX_TOKENS	1024	Sufficient for detailed answers
Dependencies
Table
Package	Version	Purpose
streamlit	≥ 1.28.0	Web application interface
langchain-groq	≥ 0.1.0	Groq LLM integration
langchain-community	≥ 0.0.10	Community loaders and vector stores
langchain-text-splitters	≥ 0.0.1	Document chunking strategies
faiss-cpu	≥ 1.7.4	Facebook AI Similarity Search (CPU)
sentence-transformers	≥ 2.2.0	HuggingFace embeddings backend
python-dotenv	≥ 1.0.0	Environment variable management
pypdf	≥ 3.0.0	PDF parsing and text extraction
License
Distributed under the MIT License. See LICENSE for details.
Author
Bhavyasri Lagishetty
Building intelligent systems that understand documents
https://github.com/BhavyasriLagishetty
https://www.linkedin.com/in/bhavyasri-lagishetty/
mailto:bhavyasri.lagishetty@gmail.com
If you found this project useful, please consider giving it a ⭐
