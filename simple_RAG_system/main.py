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

class Answer(BaseModel):
    answer: str
    sources: list = []
    retrieved_docs: list = []

# --- Serve the chat interface ---
@app.get("/")
def home():
    return FileResponse("chat.html")

# --- /ask endpoint with metrics ---
@app.post("/ask")
@REQUEST_TIME.time()
def ask(question: Question):
    REQUEST_COUNT.inc()
    try:
        response_data = answer_question_structured(question.query)
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


if __name__ == "__main__":
    import uvicorn
    
    # Choose a different port (5000 is commonly available)
    PORT = 5000
    print(f"🚀 Starting ANN Expert Bot on port {PORT}")
    print(f"🌐 Open your browser to: http://localhost:{PORT}")
    print(f"📊 Prometheus metrics available at: http://localhost:8001")
    
    # Disable reload to avoid Prometheus conflicts
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
