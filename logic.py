import os
import json
import re
import shutil
import gc
import streamlit as st
from dotenv import load_dotenv

# --- LIGHTWEIGHT SETUP (Runs instantly) ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# --- CONFIGURATION ---
CHUNK_SIZE = 1000 
CHUNK_OVERLAP = 200
MAX_CHUNKS = 100      

# --- CACHING THE EMBEDDING MODEL ---
# show_spinner=False ensures this runs SILENTLY without a loading bar
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """Load the heavy model only when this function is called, not at startup."""
    # 1. HEAVY IMPORTS MOVED INSIDE
    import torch
    from langchain_huggingface import HuggingFaceEmbeddings

    # 2. GPU LOGIC (Safe for Python 3.13 / CPU fallback)
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
        
    print(f"⏳ Loading HuggingFace Embedding Model on {device.upper()}...")
    
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': False}
    )

def get_pdf_text(pdf_docs):
    from PyPDF2 import PdfReader  # Lazy Import
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text += content
    return text

def create_vector_store(text):
    # Lazy Imports
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = text_splitter.split_text(text)
    
    if len(chunks) > MAX_CHUNKS:
        print(f"⚠️ Demo Mode: Limiting to first {MAX_CHUNKS} chunks.")
        chunks = chunks[:MAX_CHUNKS]

    gc.collect()
    if os.path.exists("./chroma_db"):
        try:
            shutil.rmtree("./chroma_db")
        except PermissionError:
            pass
    
    # Load model now (UI is already visible)
    embed_model = load_embedding_model()
    
    print(f"Processing {len(chunks)} chunks locally...")
    db = Chroma.from_texts(
        texts=chunks, 
        embedding=embed_model, 
        persist_directory="./chroma_db"
    )
    return True

# --- HELPER: FIND VALID MODEL NAME ---
def get_active_model_name(genai, keyword):
    """Dynamically finds a valid model name from the API to avoid 404s."""
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                if keyword in m.name:
                    return m.name
    except Exception:
        pass
    
    # Fallback constants if dynamic check fails
    if keyword == "flash": return "models/gemini-1.5-flash"
    if keyword == "pro": return "models/gemini-1.5-pro"
    return "models/gemini-pro"

def get_rag_response(question, model_selection="flash"):
    # Lazy Imports
    import google.generativeai as genai
    from langchain_chroma import Chroma
    
    # Configure API
    genai.configure(api_key=api_key)

    # 1. Determine correct model name
    target_keyword = "flash" if "flash" in model_selection.lower() else "pro"
    active_model = get_active_model_name(genai, target_keyword)

    optimized_query = question 
    embed_model = load_embedding_model()

    try:
        db = Chroma(persist_directory="./chroma_db", embedding_function=embed_model)
        docs = db.similarity_search(optimized_query, k=5) 
        context_text = "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        return f"Error during retrieval: {e}"
    
    prompt = f"""
    You are a RAG assistant. Answer the 'Original Question' based strictly on the 'Context'.
    Original Question: {question}
    Context: {context_text}
    Answer:
    """
    
    model = genai.GenerativeModel(active_model)
    
    try:
        response_object = model.generate_content(prompt)
        final_response = str(response_object.text).strip()
        return (final_response, optimized_query) 
    except Exception as e:
        return f"Error using {active_model}: {e}"

def generate_quiz(topic, model_selection="flash"):
    import google.generativeai as genai  # Lazy Import
    genai.configure(api_key=api_key)

    # 1. Determine correct model name
    target_keyword = "flash" if "flash" in model_selection.lower() else "pro"
    active_model = get_active_model_name(genai, target_keyword)

    prompt = f"""
    Generate a 3-question MCQ quiz about "{topic}". Output strict JSON.
    Format: {{"questions": [{{"question": "...", "options": ["..."], "answer": "...", "explanation": "..."}}]}}
    """
    
    model = genai.GenerativeModel(active_model)
    
    try:
        response = model.generate_content(prompt)
        clean_json = re.sub(r'```json\s*|\s*```', '', response.text).strip()
        return json.loads(clean_json)
    except Exception as e:
        return None