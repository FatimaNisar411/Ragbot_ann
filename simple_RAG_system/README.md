# Simple RAG System 🔍🧠

This is a minimal Retrieval-Augmented Generation (RAG) system built with FastAPI, Hugging Face models, and ChromaDB. It allows question answering based on a custom context file.

## 💡 Features

- Embeds and stores context using `all-MiniLM-L6-v2` with ChromaDB
- Retrieves relevant chunks using similarity search
- Uses a Hugging Face language model to generate answers
- REST API via FastAPI (`/ask` endpoint)

## 🛠️ Setup

1. **Clone the repo**

```bash
git clone https://github.com/FatimaNisar411/simple_RAG_system.git
cd RAG
