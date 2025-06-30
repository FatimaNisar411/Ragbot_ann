"""
Data Loading and PDF Processing Module
Handles document extraction, chunking, and ChromaDB setup
"""

import os
import PyPDF2
import pdfplumber
from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import SentenceTransformer

print("📚 Data Loader Module Initialized")

# Initialize embedding model
print("📌 Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Setup ChromaDB with persistent storage
print("📦 Setting up ChromaDB with persistent storage...")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="ann_docs",
    embedding_function=SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")
)

_loaded = False

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

def chunk_text(content: str, max_chunk_size: int = 500) -> list:
    """Split content into meaningful chunks"""
    # Split content into chunks by paragraphs (double newlines)
    chunks = [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]
    
    # If no paragraph breaks, split by sentences or fixed length
    if len(chunks) == 1 and len(content) > 1000:
        # Split by sentences (rough approximation)
        sentences = content.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) > max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
            else:
                current_chunk += sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
    
    return chunks

def process_file(file_path: str, filename: str) -> list:
    """Process a single file and return chunks"""
    print(f"📖 Processing {filename}...")
    
    try:
        # Extract content based on file type
        if filename.endswith('.pdf'):
            content = extract_text_from_pdf(file_path)
            if not content.strip():
                print(f"⚠️ No text extracted from {filename}")
                return []
        else:  # .txt file
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        
        # Split into chunks
        chunks = chunk_text(content)
        
        # Filter meaningful chunks
        meaningful_chunks = []
        for chunk in chunks:
            if len(chunk) > 50:  # Only add meaningful chunks
                meaningful_chunks.append({
                    "content": chunk,
                    "source": filename,
                    "type": "pdf" if filename.endswith('.pdf') else "txt"
                })
        
        print(f"✅ Created {len(meaningful_chunks)} chunks from {filename}")
        return meaningful_chunks
        
    except Exception as e:
        print(f"❌ Error processing {filename}: {e}")
        return []

def load_documents():
    """Load all documents from the data directory into ChromaDB"""
    global _loaded
    
    # Check if documents are already persisted
    try:
        existing_count = collection.count()
        if existing_count > 0:
            print(f"📚 Found {existing_count} existing chunks in persistent storage")
            _loaded = True
            return
    except Exception as e:
        print(f"⚠️ Error checking existing documents: {e}")
    
    if _loaded:
        print("📚 Documents already loaded")
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
    total_chunks = 0
    
    for filename in all_files:
        file_path = os.path.join(data_dir, filename)
        
        # Process file and get chunks
        file_chunks = process_file(file_path, filename)
        
        # Add chunks to ChromaDB collection
        for chunk_data in file_chunks:
            collection.add(
                documents=[chunk_data["content"]], 
                ids=[f"doc_{chunk_id}"],
                metadatas=[{
                    "source": chunk_data["source"],
                    "type": chunk_data["type"]
                }]
            )
            chunk_id += 1
            total_chunks += 1
    
    print(f"✅ Loaded {total_chunks} chunks from {len(all_files)} files ({len(txt_files)} txt, {len(pdf_files)} pdf)")
    _loaded = True

def get_collection():
    """Get the ChromaDB collection (for use by other modules)"""
    return collection

def get_relevant_chunks(query: str, n_results: int = 5):
    """Retrieve relevant document chunks for a query"""
    results = collection.query(query_texts=[query], n_results=n_results)
    return {
        "documents": results["documents"][0],
        "metadatas": results.get("metadatas", [[]])[0],
        "ids": results.get("ids", [[]])[0]
    }

def get_document_stats():
    """Get statistics about loaded documents"""
    try:
        total_count = collection.count()
        return {
            "total_chunks": total_count,
            "status": "loaded" if _loaded else "not_loaded",
            "collection_name": "ann_docs"
        }
    except Exception as e:
        return {
            "total_chunks": 0,
            "status": "error",
            "error": str(e)
        }

def create_session_collection(conversation_id: str):
    """Create a temporary collection for a specific conversation session"""
    try:
        session_collection_name = f"session_{conversation_id}"
        session_collection = chroma_client.get_or_create_collection(
            name=session_collection_name,
            embedding_function=SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")
        )
        print(f"✅ Created session collection: {session_collection_name}")
        return session_collection
    except Exception as e:
        print(f"❌ Error creating session collection: {e}")
        return None

def process_uploaded_file(file_content: bytes, filename: str, conversation_id: str) -> dict:
    """Process an uploaded file for a specific conversation session"""
    try:
        import tempfile
        import io
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{filename}") as temp_file:
            temp_file.write(file_content)
            temp_path = temp_file.name
        
        try:
            # Process the file
            chunks = process_file(temp_path, filename)
            
            if not chunks:
                return {"success": False, "message": f"No content extracted from {filename}"}
            
            # Get or create session collection
            session_collection = create_session_collection(conversation_id)
            if not session_collection:
                return {"success": False, "message": "Failed to create session collection"}
            
            # Add chunks to session collection
            chunk_id = 0
            for chunk_data in chunks:
                session_collection.add(
                    documents=[chunk_data["content"]], 
                    ids=[f"session_{conversation_id}_doc_{chunk_id}"],
                    metadatas=[{
                        "source": chunk_data["source"],
                        "type": chunk_data["type"],
                        "session": conversation_id,
                        "uploaded": True
                    }]
                )
                chunk_id += 1
            
            return {
                "success": True, 
                "message": f"Successfully processed {filename}",
                "chunks_added": len(chunks),
                "filename": filename
            }
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except:
                pass
                
    except Exception as e:
        print(f"❌ Error processing uploaded file {filename}: {e}")
        return {"success": False, "message": f"Error processing {filename}: {str(e)}"}

def get_relevant_chunks_with_session(query: str, conversation_id: str = None, n_results: int = 5):
    """Retrieve relevant chunks from both main collection and session collection"""
    all_results = {"documents": [], "metadatas": [], "ids": []}
    
    # Get chunks from main collection (50% of results)
    main_results_count = max(1, n_results // 2)
    main_results = collection.query(query_texts=[query], n_results=main_results_count)
    
    if main_results["documents"] and main_results["documents"][0]:
        all_results["documents"].extend(main_results["documents"][0])
        all_results["metadatas"].extend(main_results.get("metadatas", [[]])[0])
        all_results["ids"].extend(main_results.get("ids", [[]])[0])
    
    # Get chunks from session collection if available (50% of results)
    if conversation_id:
        try:
            session_collection_name = f"session_{conversation_id}"
            session_collection = chroma_client.get_collection(name=session_collection_name)
            
            session_results_count = n_results - len(all_results["documents"])
            if session_results_count > 0:
                session_results = session_collection.query(query_texts=[query], n_results=session_results_count)
                
                if session_results["documents"] and session_results["documents"][0]:
                    all_results["documents"].extend(session_results["documents"][0])
                    all_results["metadatas"].extend(session_results.get("metadatas", [[]])[0])
                    all_results["ids"].extend(session_results.get("ids", [[]])[0])
                    
        except Exception as e:
            print(f"⚠️ No session collection found for {conversation_id}: {e}")
    
    return all_results

def cleanup_session_collection(conversation_id: str):
    """Clean up session collection when conversation ends"""
    try:
        session_collection_name = f"session_{conversation_id}"
        chroma_client.delete_collection(name=session_collection_name)
        print(f"🗑️ Cleaned up session collection: {session_collection_name}")
        return True
    except Exception as e:
        print(f"⚠️ Error cleaning up session collection: {e}")
        return False

if __name__ == "__main__":
    # Test the data loader
    print("🧪 Testing data loader...")
    load_documents()
    stats = get_document_stats()
    print(f"📊 Document stats: {stats}")
    
    # Test retrieval
    test_query = "What is a neuron?"
    chunks = get_relevant_chunks(test_query, 2)
    print(f"🔍 Test query: '{test_query}'")
    print(f"📄 Retrieved {len(chunks['documents'])} relevant chunks")
