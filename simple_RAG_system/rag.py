"""
Main RAG System - Orchestrates data loading and response generation
"""

from data_loader import load_documents, get_relevant_chunks, get_relevant_chunks_with_session
from response_generator import generate_response, create_prompt

print("� Starting ANN RAG setup...")

def answer_question_structured(query: str, conversation_id: str = None) -> dict:
    """Return structured answer with sources separated, including session documents"""
    
    # Retrieve relevant documents (both main collection and session-specific)
    print(f"🔍 Searching for: {query}")
    if conversation_id:
        print(f"📂 Including session documents for conversation: {conversation_id}")
        chunk_results = get_relevant_chunks_with_session(query, conversation_id, n_results=5)
    else:
        chunk_results = get_relevant_chunks(query, n_results=5)
    
    retrieved_docs = chunk_results["documents"]
    retrieved_docs_text = "\n\n".join(retrieved_docs)
    
    # Get source files for context, distinguish between main and session docs
    sources = []
    session_sources = []
    for meta in chunk_results["metadatas"]:
        source = meta.get("source", "Unknown")
        if meta.get("uploaded", False):
            session_sources.append(f"📎 {source} (uploaded)")
        else:
            sources.append(source)
    
    all_sources = sources + session_sources
    
    # Create prompt
    prompt = create_prompt(query, retrieved_docs_text)
    
    # Generate response
    answer = generate_response(prompt)
    
    return {
        "answer": answer,
        "sources": list(set(all_sources)),  # Remove duplicates
        "retrieved_docs": retrieved_docs
    }
