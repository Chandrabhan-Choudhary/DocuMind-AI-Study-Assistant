import os
import json
import re
import time
import shutil
import gc
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import streamlit as st
import torch  # Required for GPU detection

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# --- INITIAL SETUP & MODEL DETECTION ---
try:
    genai.configure(api_key=api_key)
    
    def get_model_name(keyword):
        """Finds the first available model containing the keyword."""
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                if keyword in m.name:
                    return m.name
        return None

    # Auto-detect the correct string
    FLASH_MODEL_NAME = get_model_name("flash")
    PRO_MODEL_NAME = get_model_name("pro")
    
except Exception as e:
    # Fallback to known stable names if API list fails
    print(f"⚠️ API Configuration Warning: {e}. Using fallback model names.")
    FLASH_MODEL_NAME = "models/gemini-1.5-flash-001"
    PRO_MODEL_NAME = "models/gemini-1.5-pro-001"

if not FLASH_MODEL_NAME:
    FLASH_MODEL_NAME = "models/gemini-1.5-flash-001" 
if not PRO_MODEL_NAME:
    PRO_MODEL_NAME = "models/gemini-1.5-pro-001"

print(f"🔑 USING KEY ENDING IN: ...{api_key[-4:] if api_key else 'NONE'}")
print(f"🤖 Active Flash Model: {FLASH_MODEL_NAME}")
print(f"🧠 Active Pro Model:   {PRO_MODEL_NAME}")


# --- CONFIGURATION ---
CHUNK_SIZE = 1000 
CHUNK_OVERLAP = 200
BATCH_SIZE = 10       
MAX_CHUNKS = 100      

# --- CACHING THE EMBEDDING MODEL (GPU OPTIMIZED) ---
@st.cache_resource
def load_embedding_model():
    """Load the local model once and keep it in memory for instant access."""
    
    # Check for GPU (NVIDIA CUDA) or Mac (MPS), fallback to CPU
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
        
    print(f"⏳ Loading HuggingFace Embedding Model on {device.upper()}...")
    
    # Load with device parameter
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': False}
    )

EMBED_MODEL = load_embedding_model()


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text += content
    return text

def create_vector_store(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = text_splitter.split_text(text)
    
    if len(chunks) > MAX_CHUNKS:
        print(f"⚠️ Demo Mode: Limiting to first {MAX_CHUNKS} chunks.")
        chunks = chunks[:MAX_CHUNKS]

    # Windows File Lock Fix
    gc.collect()
    if os.path.exists("./chroma_db"):
        try:
            shutil.rmtree("./chroma_db")
            print("✅ Old database cleared.")
        except PermissionError:
            print("⚠️ Database locked. Appending to existing DB.")
    
    # Create DB using Local Embeddings
    print(f"Processing {len(chunks)} chunks locally...")
    db = Chroma.from_texts(
        texts=chunks, 
        embedding=EMBED_MODEL, 
        persist_directory="./chroma_db"
    )
    print("✅ Vector Store Created Successfully")
    return True

def get_rag_response(question, model_selection="flash"):
    """
    Retrieves context using the original query and prompts the LLM to self-correct
    during generation (Query-Aware Generation).
    Returns: (response_text, original_query) OR an error string.
    """
    # 1. The query remains the original for retrieval to avoid extra API calls.
    # The LLM will correct the query internally during the final step.
    optimized_query = question 

    # 2. Retrieve Context (Locally) using the original query
    try:
        db = Chroma(persist_directory="./chroma_db", embedding_function=EMBED_MODEL)
        docs = db.similarity_search(optimized_query, k=5) 
        context_text = "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        return f"Error during retrieval: {e}"
    
    # 3. Modify the Prompt to Self-Correct the Query *During Generation*
    prompt = f"""
    You are a RAG assistant. First, analyze the 'Original Question' and the 'Context' provided below. 
    Then, answer the 'Original Question' based strictly on the 'Context'.
    
    Original Question: {question}
    
    Context: {context_text}
    
    Answer:
    """
    
    # 4. Select the correct model and Generate Answer
    active_model = FLASH_MODEL_NAME if "flash" in model_selection.lower() else PRO_MODEL_NAME
        
    model = genai.GenerativeModel(active_model)
    try:
        response_object = model.generate_content(prompt)
        final_response = str(response_object.text).strip()
        
        # SUCCESS: Return both the response AND the original query (for display)
        return (final_response, optimized_query) 
        
    except Exception as e:
        # Final Quota Check
        if "429" in str(e) and active_model == PRO_MODEL_NAME:
            return f"Error: Pro model quota exhausted. Please switch to Gemini Flash for instant answers."
        
        return f"Error using {active_model}: {e}"

def generate_quiz(topic, model_selection="flash"):
    prompt = f"""
    Generate a 3-question MCQ quiz about "{topic}". Output strict JSON.
    Format: {{"questions": [{{"question": "...", "options": ["..."], "answer": "...", "explanation": "..."}}]}}
    """
    
    active_model = FLASH_MODEL_NAME if "flash" in model_selection.lower() else PRO_MODEL_NAME
        
    model = genai.GenerativeModel(active_model)
    
    try:
        response = model.generate_content(prompt)
        clean_json = re.sub(r'```json\s*|\s*```', '', response.text).strip()
        return json.loads(clean_json)
    except Exception as e:
        if "429" in str(e) and active_model == PRO_MODEL_NAME:
            # Fallback to Flash for quiz generation if Pro fails
            print("⚠️ Pro model failed to generate quiz. Falling back to Flash...")
            model_backup = genai.GenerativeModel(FLASH_MODEL_NAME)
            response_backup = model_backup.generate_content(prompt)
            clean_json = re.sub(r'```json\s*|\s*```', '', response_backup.text).strip()
            return json.loads(clean_json)
        return None