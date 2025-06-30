"""
Response Generation Module
Handles LLM interactions and answer generation
"""

import os
import requests
import json
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("🤖 Response Generator Module Initialized")

# GPU optimization settings
if os.getenv("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES")
if os.getenv("OLLAMA_GPU_LAYERS"):
    os.environ["OLLAMA_GPU_LAYERS"] = os.getenv("OLLAMA_GPU_LAYERS")

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Using device: {device}")

# Check for API keys and local options
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

# Determine which API/model to use (prefer Groq API for Llama 3.1:8b)
use_groq = GROQ_API_KEY is not None
use_ollama = OLLAMA_MODEL is not None
use_together = TOGETHER_API_KEY is not None
use_hf = HF_API_KEY is not None

# Initialize local model if needed
generator = None
model_name = None
tokenizer = None

if use_groq:
    print("🚀 Using FREE Groq API for Llama-3.1-8b-instant!")
    api_provider = "groq"
elif use_ollama:
    print(f"🦙 Using local Ollama model as fallback: {OLLAMA_MODEL}")
    api_provider = "ollama"
elif use_together:
    print("🚀 Using FREE Together.ai API for Llama models!")
    api_provider = "together"
elif use_hf:
    print("🚀 Using FREE Hugging Face Inference API!")
    api_provider = "huggingface"
else:
    print("🏠 No API keys found, loading local model...")
    api_provider = "local"
    print("🔧 Loading Llama model for laptop...")
    model_name = "microsoft/Phi-3-mini-4k-instruct"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=None,
            trust_remote_code=True
        )
        
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=-1,
            return_full_text=False,
            pad_token_id=tokenizer.eos_token_id
        )
        print("✅ Phi-3 Llama model loaded successfully!")
        
    except Exception as e:
        print(f"⚠️ Failed to load Phi-3: {e}")
        print("🔄 Falling back to FLAN-T5...")
        generator = pipeline("text2text-generation", model="google/flan-t5-base", device=-1)
        model_name = "google/flan-t5-base"

print("✅ Response Generator setup complete.")

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
                "model": "llama-3.1-8b-instant",
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
                "max_tokens": 800,
                "temperature": 0.7,
                "stream": False
            }
            
            print("📡 Sending request to Groq API...")
            response = requests.post(url, headers=headers, json=data, timeout=60)
            
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

def call_ollama_api(prompt: str, model: str) -> str:
    """Call local Ollama API for Llama inference"""
    try:
        print(f"🦙 Calling Ollama API with model: {model}")
        url = "http://localhost:11434/api/generate"
        
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 800
            }
        }
        
        print("📡 Sending request to Ollama...")
        response = requests.post(url, json=data, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get("response", "").strip()
            print(f"✅ Ollama success! Response length: {len(answer)}")
            return answer
        else:
            print(f"⚠️ Ollama error: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("❌ Ollama not running! Start it with: ollama serve")
        return None
    except Exception as e:
        print(f"❌ Ollama API call failed: {e}")
        return None

def generate_response(prompt: str) -> str:
    """Generate response using the best available model"""
    global use_ollama, use_groq, generator, model_name, tokenizer
    
    answer = None
    
    try:
        # First try Groq API (primary choice for Llama 3.1:8b)
        if use_groq:
            print("🚀 Using FREE Groq API for Llama 3.1:8b...")
            answer = call_groq_api(prompt)
            if answer:
                print("✅ Got response from Groq API")
            else:
                print("⚠️ Groq API failed, falling back to local Ollama...")
        
        # Try local Ollama if Groq not available or failed
        if not answer and use_ollama:
            print("🦙 Using local Ollama model as fallback...")
            answer = call_ollama_api(prompt, OLLAMA_MODEL)
            if answer:
                print("✅ Got response from Ollama")
            else:
                print("⚠️ Ollama failed, falling back to local model...")
        
        # Use local model if both Groq and Ollama not available or failed
        if not answer and generator:
            print("🏠 Using local model...")
            if "phi-3" in model_name.lower():
                response = generator(
                    prompt, 
                    max_new_tokens=300,
                    do_sample=True, 
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )[0]['generated_text']
                answer = response.strip()
            else:
                response = generator(
                    prompt, 
                    max_length=500,
                    min_length=100,
                    do_sample=True, 
                    temperature=0.7,
                    no_repeat_ngram_size=3
                )[0]['generated_text']
                answer = response.strip()
        
        # Final fallback
        if not answer:
            answer = "I'm sorry, I couldn't generate a response. Please try again."
    
    except Exception as e:
        print(f"❌ Error generating response: {e}")
        answer = f"Error generating response: {str(e)}"
    
    return answer

def create_prompt(query: str, retrieved_docs: str) -> str:
    """Create a well-formatted prompt for the LLM with conversation awareness"""
    
    # Check if this is a contextualized query (contains conversation context)
    if "Conversation context:" in query and "Current question:" in query:
        # Split contextualized query
        parts = query.split("Current question:", 1)
        context_part = parts[0].replace("Conversation context:", "").strip()
        actual_question = parts[1].strip()
        
        # Check if the question contains explicit exclusions (like "not", "only about", "exclude")
        exclusion_keywords = ["not", "only about", "exclude", "don't mention", "without"]
        has_exclusion = any(keyword in actual_question.lower() for keyword in exclusion_keywords)
        
        if has_exclusion:
            return f"""Based on the following documentation, answer the question accurately and comprehensively. Pay careful attention to any exclusions or specific limitations mentioned in the question.

Documentation:
{retrieved_docs}

Question: {actual_question}

IMPORTANT: Follow the specific instructions in the question exactly. If the question asks to exclude certain topics or mentions "not" or "only about", respect those constraints completely. Do not include information that was explicitly excluded."""
        else:
            return f"""Based on the following documentation and conversation context, answer the question accurately and comprehensively. Use the conversation context to understand references like "it", "them", "this", etc.

Documentation:
{retrieved_docs}

Conversation Context:
{context_part}

Current Question: {actual_question}

Provide a clear, detailed answer that takes into account the conversation context. Reference previous topics when relevant:"""
    
    else:
        # Standard prompt for new conversations
        return f"""Based on the following documentation, answer the question accurately and comprehensively. If the question is not related to the documentation, answer based on your general knowledge.

Documentation:
{retrieved_docs}

Question: {query}

Provide a clear, detailed answer without any extra formatting, emojis, or source citations:"""

def get_system_info() -> dict:
    """Get information about the response generation system"""
    return {
        "api_provider": api_provider,
        "ollama_model": OLLAMA_MODEL,
        "use_ollama": use_ollama,
        "use_groq": use_groq,
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "local_model": model_name if model_name else "None"
    }

if __name__ == "__main__":
    # Test the response generator
    print("🧪 Testing response generator...")
    info = get_system_info()
    print(f"📊 System info: {info}")
    
    # Test prompt creation
    test_prompt = create_prompt(
        "What is a neural network?", 
        "A neural network is a computational model inspired by biological neural networks."
    )
    print(f"📝 Test prompt created: {len(test_prompt)} characters")
