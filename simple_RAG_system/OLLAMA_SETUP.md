# 🦙 Ollama Setup Guide for RAG System

## Installation Steps

### 1. Download and Install Ollama
- Go to: https://ollama.ai/download
- Download for Windows
- Install Ollama (about 100MB installer)

### 2. Start Ollama Service
```bash
ollama serve
```
(Keep this running in a terminal)

### 3. Download a Llama Model

#### Small Models (Recommended for laptops):
```bash
# Llama 3.2 3B - ~2.0 GB - Good balance
ollama pull llama3.2:3b

# Llama 3.2 1B - ~0.9 GB - Fastest, basic quality  
ollama pull llama3.2:1b
```

#### Medium Models (Better quality):
```bash
# Llama 3.1 8B - ~4.7 GB - Excellent quality
ollama pull llama3.1:8b

# Code Llama 7B - ~3.8 GB - Good for technical content
ollama pull codellama:7b
```

### 4. Configure Your RAG System

Add to your `.env` file:
```bash
# Use Ollama instead of Groq
OLLAMA_MODEL=llama3.2:3b

# Keep Groq as backup (optional)
GROQ_API_KEY=your_groq_key_here
```

### 5. Test Ollama
```bash
ollama run llama3.2:3b "Hello, how are you?"
```

## Model Recommendations by Use Case

### 💻 **Laptop/Limited RAM (8-16GB)**:
- **llama3.2:1b** (0.9 GB) - Fast responses
- **llama3.2:3b** (2.0 GB) - Good quality, reasonable speed

### 🖥️ **Desktop/Good RAM (16-32GB)**:
- **llama3.1:8b** (4.7 GB) - Excellent quality
- **codellama:7b** (3.8 GB) - Great for technical content

### 🚀 **Powerful Machine (32GB+ RAM)**:
- **llama3.1:70b** (40 GB) - Best quality

## Priority Order in Your RAG System

1. **Ollama** (if OLLAMA_MODEL set) - Local, private, fast
2. **Groq API** (if GROQ_API_KEY set) - Free cloud API
3. **Local Hugging Face** - Fallback local model

## Usage
Just restart your FastAPI server after setting OLLAMA_MODEL in .env!

Your system will automatically use Ollama first, then fall back to Groq if needed.
