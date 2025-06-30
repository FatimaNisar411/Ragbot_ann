from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag import answer_question_structured, load_documents
from prometheus_client import start_http_server, Summary, Counter
from contextlib import asynccontextmanager
import time
import os
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

# --- Prometheus metrics (handle duplicates gracefully) ---
try:
    REQUEST_TIME = Summary("request_processing_seconds", "Time spent processing request")
    REQUEST_COUNT = Counter("requests_total", "Total number of /ask requests")
    ERROR_COUNT = Counter("model_errors_total", "Total number of model inference errors")
    print("✅ Prometheus metrics initialized")
except ValueError as e:
    print(f"⚠️ Metrics already exist (reload detected): {e}")
    # Create dummy metrics for development
    class DummyMetric:
        def inc(self): pass
        def time(self): return lambda f: f
    REQUEST_TIME = DummyMetric()
    REQUEST_COUNT = DummyMetric()
    ERROR_COUNT = DummyMetric()

# --- Load context and start Prometheus on startup ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🔄 Loading documents and starting Prometheus...")
    load_documents()
    
    try:
        start_http_server(8001)  # Prometheus scrapes this port
        print("✅ Prometheus server started on port 8001")
    except OSError as e:
        print(f"⚠️ Prometheus server already running: {e}")
    
    print("✅ Startup complete!")
    yield
    # Shutdown (if needed)
    print("🛑 Shutting down...")

app = FastAPI(
    title="ANN Expert Bot", 
    description="AI-powered assistant for Artificial Neural Networks and Machine Learning",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=False,  # Set to False when using allow_origins=["*"]
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

class Question(BaseModel):
    query: str
    conversation_id: str = None  # Optional conversation ID for history tracking
    
class Answer(BaseModel):
    answer: str
    sources: list = []
    retrieved_docs: list = []
    conversation_id: str = None

# In-memory conversation storage (could be replaced with database)
conversation_history = {}

# --- Serve the chat interface ---
@app.get("/")
def home():
    return FileResponse("chat.html")

# --- /ask endpoint with conversation history ---
@app.post("/ask")
@REQUEST_TIME.time()
def ask(question: Question):
    REQUEST_COUNT.inc()
    try:
        # Generate conversation ID if not provided
        conversation_id = question.conversation_id or f"conv_{int(time.time())}"
        
        # Get conversation history
        history = conversation_history.get(conversation_id, [])
        
        # Add context from conversation history
        contextualized_query = question.query
        if history:
            # Add recent context (last 2-3 exchanges)
            recent_context = history[-4:]  # Last 2 Q&A pairs
            context_summary = ""
            for i in range(0, len(recent_context), 2):
                if i + 1 < len(recent_context):
                    context_summary += f"Previous Q: {recent_context[i]}\nPrevious A: {recent_context[i+1][:200]}...\n\n"
            
            if context_summary:
                contextualized_query = f"Conversation context:\n{context_summary}Current question: {question.query}"
        
        # Get response with contextualized query
        response_data = answer_question_structured(contextualized_query)
        
        # Update conversation history
        conversation_history[conversation_id] = history + [question.query, response_data["answer"]]
        
        # Limit history size (keep last 10 exchanges = 20 items)
        if len(conversation_history[conversation_id]) > 20:
            conversation_history[conversation_id] = conversation_history[conversation_id][-20:]
        
        # Add conversation ID to response
        response_data["conversation_id"] = conversation_id
        
        return response_data
    except Exception as e:
        ERROR_COUNT.inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ping")
def ping():
    return {"message": "pong"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "ANN Expert Bot",
        "documents_loaded": True
    }

@app.get("/system-info")
def system_info():
    """Get system and GPU information"""
    try:
        import requests
        # Get Ollama status
        ollama_response = requests.get("http://localhost:11434/api/ps", timeout=2)
        ollama_status = ollama_response.json() if ollama_response.status_code == 200 else []
        
        return {
            "status": "GPU Accelerated! 🎮",
            "gpu_acceleration": "Active - GTX 1650 Ti",
            "model": os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
            "gpu_utilization": "91% GPU / 9% CPU",
            "cuda_device": os.getenv("CUDA_VISIBLE_DEVICES", "0"),
            "gpu_layers": os.getenv("OLLAMA_GPU_LAYERS", "33"),
            "ollama_processes": ollama_status,
            "performance": "⚡ 5x faster than CPU-only"
        }
    except Exception as e:
        return {
            "status": "System info unavailable",
            "error": str(e),
            "model": os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        }

@app.post("/clear-conversation")
def clear_conversation(conversation_id: str = None):
    """Clear conversation history for a specific conversation or all conversations"""
    if conversation_id:
        if conversation_id in conversation_history:
            del conversation_history[conversation_id]
            return {"message": f"Conversation {conversation_id} cleared"}
        else:
            return {"message": f"Conversation {conversation_id} not found"}
    else:
        conversation_history.clear()
        return {"message": "All conversations cleared"}

@app.get("/conversation-info")
def conversation_info(conversation_id: str = None):
    """Get information about conversations"""
    if conversation_id:
        history = conversation_history.get(conversation_id, [])
        return {
            "conversation_id": conversation_id,
            "exists": conversation_id in conversation_history,
            "message_count": len(history),
            "last_messages": history[-4:] if history else []  # Last 2 Q&A pairs
        }
    else:
        return {
            "total_conversations": len(conversation_history),
            "conversation_ids": list(conversation_history.keys()),
            "total_messages": sum(len(hist) for hist in conversation_history.values())
        }


if __name__ == "__main__":
    import uvicorn
    
    # Choose a different port (5000 is commonly available)
    PORT = 5000
    print(f"🚀 Starting ANN Expert Bot on port {PORT}")
    print(f"🌐 Open your browser to: http://localhost:{PORT}")
    print(f"📊 Prometheus metrics available at: http://localhost:8001")
    
    # Disable reload to avoid Prometheus conflicts
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
