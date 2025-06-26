from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os
import PyPDF2
import pdfplumber
from pathlib import Path
import requests
import json
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

print("🔧 Starting ANN RAG setup...")

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Using device: {device}")

# Initialize embedding model
print("📌 Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Setup ChromaDB
print("📦 Setting up ChromaDB...")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(
    name="ann_docs",
    embedding_function=SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")
)

# Check for API keys (multiple free options)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")  # Hugging Face Inference API

# Determine which API to use (prefer free options)
use_groq = GROQ_API_KEY is not None
use_together = TOGETHER_API_KEY is not None
use_hf = HF_API_KEY is not None

if use_groq:
    print("🚀 Using FREE Groq API for Llama-3.1 model!")
    generator = None  # Won't need local generator
    api_provider = "groq"
elif use_together:
    print("🚀 Using FREE Together.ai API for Llama models!")
    generator = None
    api_provider = "together"
elif use_hf:
    print("🚀 Using FREE Hugging Face Inference API!")
    generator = None
    api_provider = "huggingface"
else:
    print("🏠 No API keys found, loading local model...")
    api_provider = "local"
    # Load instruction-following model for better responses
    print("🔧 Loading Llama model for laptop...")
    model_name = "microsoft/Phi-3-mini-4k-instruct"  # Small but powerful Llama-style model

    try:
        # Load the Phi-3 model (Llama architecture, laptop-friendly)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add padding token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for CPU
            device_map=None,  # CPU only
            trust_remote_code=True
        )
        
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=-1,  # CPU
            return_full_text=False,  # Only return generated text
            pad_token_id=tokenizer.eos_token_id
        )
        print("✅ Phi-3 Llama model loaded successfully!")
        
    except Exception as e:
        print(f"⚠️ Failed to load Phi-3: {e}")
        print("🔄 Falling back to FLAN-T5...")
        generator = pipeline("text2text-generation", model="google/flan-t5-base", device=-1)
        model_name = "google/flan-t5-base"

print("✅ Model setup complete.")

_loaded = False

def call_groq_api(prompt: str) -> str:
    """Call the free Groq API for Llama inference with retry logic"""
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            print(f"🔍 Calling Groq API (attempt {attempt + 1}/{max_retries}) with prompt length: {len(prompt)}")
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "llama-3.1-8b-instant",  # Free Llama model
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are a helpful AI assistant. Answer questions accurately based on the provided documentation when relevant, or use your general knowledge when the question is outside the documentation scope. Provide clear, detailed explanations without extra formatting or emojis."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "max_tokens": 800,  # Increased for longer, complete responses
                "temperature": 0.7,
                "stream": False
            }
            
            print("📡 Sending request to Groq API...")
            response = requests.post(url, headers=headers, json=data, timeout=60)  # Increased timeout to 60 seconds
            
            if response.status_code == 200:
                result = response.json()
                answer = result["choices"][0]["message"]["content"].strip()
                print(f"✅ Groq API success! Response length: {len(answer)}")
                return answer
            else:
                print(f"⚠️ Groq API error: {response.status_code} - {response.text}")
                if attempt < max_retries - 1:
                    print(f"🔄 Retrying in 2 seconds...")
                    time.sleep(2)
                    continue
                return None
                
        except requests.exceptions.Timeout:
            print(f"⏰ Groq API timeout on attempt {attempt + 1}")
            if attempt < max_retries - 1:
                print(f"🔄 Retrying in 3 seconds...")
                time.sleep(3)
                continue
            print("❌ All Groq API attempts timed out")
            return None
        except Exception as e:
            print(f"❌ Groq API call failed on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                print(f"🔄 Retrying in 2 seconds...")
                time.sleep(2) 
                continue
            return None
    
    return None

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF using multiple methods for best results"""
    text = ""
    
    try:
        # Method 1: Try pdfplumber first (better for complex layouts)
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
        
        if text.strip():
            print(f"✅ Extracted {len(text)} characters using pdfplumber")
            return text
            
    except Exception as e:
        print(f"⚠️ pdfplumber failed: {e}")
    
    try:
        # Method 2: Fallback to PyPDF2
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
        
        if text.strip():
            print(f"✅ Extracted {len(text)} characters using PyPDF2")
            return text
            
    except Exception as e:
        print(f"❌ PyPDF2 also failed: {e}")
    
    return text

def load_documents():
    global _loaded
    if _loaded:
        return
    
    data_dir = "data"
    if not os.path.exists(data_dir):
        print("⚠️ data directory not found!")
        return
    
    # Get all .txt and .pdf files in the data directory
    txt_files = [f for f in os.listdir(data_dir) if f.endswith('.txt') and f != 'context.txt']
    pdf_files = [f for f in os.listdir(data_dir) if f.endswith('.pdf')]
    
    all_files = txt_files + pdf_files
    
    if not all_files:
        print("⚠️ No .txt or .pdf files found in data directory!")
        return
    
    print(f"📄 Found {len(txt_files)} text files and {len(pdf_files)} PDF files to process...")
    
    chunk_id = 0
    for filename in all_files:
        file_path = os.path.join(data_dir, filename)
        print(f"📖 Processing {filename}...")
        
        try:
            # Extract content based on file type
            if filename.endswith('.pdf'):
                content = extract_text_from_pdf(file_path)
                if not content.strip():
                    print(f"⚠️ No text extracted from {filename}")
                    continue
            else:  # .txt file
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            
            # Split content into chunks by paragraphs (double newlines)
            chunks = [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]
            
            # If no paragraph breaks, split by sentences or fixed length
            if len(chunks) == 1 and len(content) > 1000:
                # Split by sentences (rough approximation)
                sentences = content.split('. ')
                chunks = []
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk + sentence) > 500:  # Max chunk size
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + ". "
                    else:
                        current_chunk += sentence + ". "
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
            
            # Add chunks to the collection
            for chunk in chunks:
                if len(chunk) > 50:  # Only add meaningful chunks
                    collection.add(
                        documents=[chunk], 
                        ids=[f"doc_{chunk_id}"],
                        metadatas=[{"source": filename, "type": "pdf" if filename.endswith('.pdf') else "txt"}]
                    )
                    chunk_id += 1
                    
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            continue
    
    print(f"✅ Loaded {chunk_id} chunks from {len(all_files)} files ({len(txt_files)} txt, {len(pdf_files)} pdf)")
    _loaded = True

def answer_question(query: str) -> str:
    global use_groq, generator, model_name  # Access global variables
    
    # Retrieve more relevant documents for better context
    results = collection.query(query_texts=[query], n_results=5)  # Increased from 3
    retrieved_docs = "\n\n".join(results["documents"][0])
    
    # Get source files for context
    sources = [meta.get("source", "Unknown") for meta in results.get("metadatas", [[]])[0]]
    
    # Create a better prompt for complete explanations
    prompt = f"""You are an expert in neural networks. Based on the documentation below, provide a complete and detailed explanation.

Neural Network Documentation:
{retrieved_docs}

Question: {query}

Instructions: Provide a comprehensive answer that fully explains the concept. Include details about how it works, why it's important, and any key technical aspects mentioned in the documentation. Make sure to complete your explanation fully.

Complete Answer:"""

    try:
        answer = None
        
        # First try Groq API if available (FREE)
        if use_groq:
            print("🚀 Using FREE Groq API...")
            print("📝 DEBUG - Prompt being sent:")
            print("=" * 80)
            print(prompt)
            print("=" * 80)
            answer = call_groq_api(prompt)
            if answer:
                print("✅ Got response from Groq API")
            else:
                print("⚠️ Groq API failed, falling back to local model...")
        
        # Use local model if Groq not available or failed
        if not answer and generator:
            print("🏠 Using local model...")
            # Check if we're using Phi-3 (Llama) or T5
            if "phi-3" in model_name.lower():
                # For Phi-3 Llama model - use text generation
                response = generator(
                    prompt, 
                    max_new_tokens=300,  # Generate new tokens
                    do_sample=True, 
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )[0]['generated_text']
                answer = response.strip()
            else:
                # For T5 model - use text2text generation
                response = generator(
                    prompt, 
                    max_length=500,
                    min_length=100,
                    do_sample=True, 
                    temperature=0.7,
                    no_repeat_ngram_size=3
                )[0]['generated_text']
                answer = response.strip()
        
        # Final fallback if no answer was generated
        if not answer:
            answer = f"Based on the documentation: {retrieved_docs[:400]}..."
    
    except Exception as e:
        print(f"❌ Error generating response: {e}")
        answer = f"Based on the documentation, here's what I found: {retrieved_docs[:400]}..."
    
    # Format the final response
    source_info = f"\n\n📁 Sources: {', '.join(set(sources))}" if sources else ""
    
    return f"""🧠 **ANN Expert Bot**

❓ **Your Question:** {query}

💡 **Answer:** {answer}

📚 **Referenced Documentation:**
{retrieved_docs[:500]}{'...' if len(retrieved_docs) > 500 else ''}{source_info}"""

def answer_question_structured(query: str) -> dict:
    """Return structured answer with sources separated"""
    global use_groq, generator, model_name
    
    # Retrieve more relevant documents for better context
    results = collection.query(query_texts=[query], n_results=5)
    retrieved_docs = results["documents"][0]
    retrieved_docs_text = "\n\n".join(retrieved_docs)
    
    # Get source files for context
    sources = [meta.get("source", "Unknown") for meta in results.get("metadatas", [[]])[0]]
    
    # Create a prompt that returns just the clean answer without extra formatting
    prompt = f"""Based on the following documentation, answer the question accurately and comprehensively. If the question is not related to the documentation, answer based on your general knowledge.

Documentation:
{retrieved_docs_text}

Question: {query}

Provide a clear, detailed answer without any extra formatting, emojis, or source citations:"""

    try:
        answer = None
        
        # First try Groq API if available (FREE)
        if use_groq:
            print("🚀 Using FREE Groq API...")
            answer = call_groq_api(prompt)
            if answer:
                print("✅ Got response from Groq API")
            else:
                print("⚠️ Groq API failed, falling back to local model...")
        
        # Use local model if Groq not available or failed
        if not answer and generator:
            print("🏠 Using local model...")
            # Check if we're using Phi-3 (Llama) or T5
            if "phi-3" in model_name.lower():
                # For Phi-3 Llama model - use text generation
                response = generator(
                    prompt, 
                    max_new_tokens=300,
                    do_sample=True, 
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )[0]['generated_text']
                answer = response.strip()
            else:
                # For T5 model - use text2text generation
                response = generator(
                    prompt, 
                    max_length=500,
                    min_length=100,
                    do_sample=True, 
                    temperature=0.7,
                    no_repeat_ngram_size=3
                )[0]['generated_text']
                answer = response.strip()
        
        # Final fallback if no answer was generated
        if not answer:
            answer = f"Based on the documentation: {retrieved_docs_text[:400]}..."
    
    except Exception as e:
        print(f"❌ Error generating response: {e}")
        answer = f"Based on the documentation, here's what I found: {retrieved_docs_text[:400]}..."
    
    return {
        "answer": answer,
        "sources": list(set(sources)),  # Remove duplicates
        "retrieved_docs": retrieved_docs
    }
