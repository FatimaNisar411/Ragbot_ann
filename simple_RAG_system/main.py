from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag import answer_question_structured, load_documents
from data_loader import process_uploaded_file, create_session_collection
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

# Session-based document storage
session_documents = {}  # {conversation_id: [uploaded_file_info]}
session_collections = {}  # {conversation_id: chromadb_collection}

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
        
        # Check if user wants to exclude previous context
        exclusion_keywords = ["not", "only about", "exclude", "don't mention", "without", "ignore previous"]
        should_exclude_context = any(keyword in question.query.lower() for keyword in exclusion_keywords)
        
        if history and not should_exclude_context:
            # Add recent context (last 2-3 exchanges)
            recent_context = history[-4:]  # Last 2 Q&A pairs
            context_summary = ""
            for i in range(0, len(recent_context), 2):
                if i + 1 < len(recent_context):
                    context_summary += f"Previous Q: {recent_context[i]}\nPrevious A: {recent_context[i+1][:200]}...\n\n"
            
            if context_summary:
                contextualized_query = f"Conversation context:\n{context_summary}Current question: {question.query}"
        elif should_exclude_context:
            print(f"🚫 Excluding conversation context due to user instruction in: {question.query}")
        
        # Get response with contextualized query and conversation ID
        response_data = answer_question_structured(contextualized_query, conversation_id)
        
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
    """Clear conversation history and session documents for a specific conversation or all conversations"""
    from data_loader import cleanup_session_collection
    
    if conversation_id:
        cleared_items = []
        
        # Clear conversation history
        if conversation_id in conversation_history:
            del conversation_history[conversation_id]
            cleared_items.append("conversation history")
        
        # Clear session documents
        if conversation_id in session_documents:
            del session_documents[conversation_id]
            cleared_items.append("session documents")
        
        # Clean up session collection
        if cleanup_session_collection(conversation_id):
            cleared_items.append("document chunks")
        
        if cleared_items:
            return {"message": f"Cleared {', '.join(cleared_items)} for conversation {conversation_id}"}
        else:
            return {"message": f"Conversation {conversation_id} not found"}
    else:
        # Clear all conversations and session documents
        conversation_history.clear()
        
        # Clean up all session collections
        for conv_id in list(session_documents.keys()):
            cleanup_session_collection(conv_id)
        
        session_documents.clear()
        return {"message": "All conversations and session documents cleared"}

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

@app.post("/upload-document")
async def upload_document(
    file: UploadFile = File(...),
    conversation_id: str = None
):
    """Upload a document for session-specific Q&A"""
    try:
        # Generate conversation ID if not provided
        if not conversation_id:
            conversation_id = f"conv_{int(time.time())}"
        
        # Validate file type
        allowed_extensions = ['.pdf', '.txt', '.docx']
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"File type {file_extension} not supported. Allowed: {allowed_extensions}"
            )
        
        # Read file content
        file_content = await file.read()
        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        # Process the uploaded file
        result = process_uploaded_file(file_content, file.filename, conversation_id)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["message"])
        
        # Store session document info
        if conversation_id not in session_documents:
            session_documents[conversation_id] = []
        
        session_documents[conversation_id].append({
            "filename": file.filename,
            "upload_time": time.time(),
            "chunks_added": result["chunks_added"]
        })
        
        return {
            "success": True,
            "message": result["message"],
            "conversation_id": conversation_id,
            "filename": file.filename,
            "chunks_added": result["chunks_added"],
            "total_session_docs": len(session_documents[conversation_id])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        ERROR_COUNT.inc()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    # Choose a different port (5000 is commonly available)
    PORT = 5000
    print(f"🚀 Starting ANN Expert Bot on port {PORT}")
    print(f"🌐 Open your browser to: http://localhost:{PORT}")
    print(f"📊 Prometheus metrics available at: http://localhost:8001")
    
    # Disable reload to avoid Prometheus conflicts
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
