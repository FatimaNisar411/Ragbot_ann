"""
Main RAG System - Orchestrates data loading and response generation
"""

from data_loader import load_documents, get_relevant_chunks
from response_generator import generate_response, create_prompt

print("� Starting ANN RAG setup...")

def answer_question_structured(query: str) -> dict:
    """Return structured answer with sources separated"""
    
    # Retrieve relevant documents
    print(f"� Searching for: {query}")
    chunk_results = get_relevant_chunks(query, n_results=5)
    
    retrieved_docs = chunk_results["documents"]
    retrieved_docs_text = "\n\n".join(retrieved_docs)
    
    # Get source files for context
    sources = [meta.get("source", "Unknown") for meta in chunk_results["metadatas"]]
    
    # Create prompt
    prompt = create_prompt(query, retrieved_docs_text)
    
    # Generate response
    answer = generate_response(prompt)
    
    return {
        "answer": answer,
        "sources": list(set(sources)),  # Remove duplicates
        "retrieved_docs": retrieved_docs
    }
