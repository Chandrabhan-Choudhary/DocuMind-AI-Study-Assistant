import streamlit as st
import os
import time
from dotenv import load_dotenv
# ALL NECESSARY FUNCTIONS MUST BE IMPORTED HERE
from logic import get_pdf_text, create_vector_store, get_rag_response, generate_quiz

# --- Initial Setup and Loading ---
st.set_page_config(page_title="DocuMind", page_icon="🧠", layout="wide")
load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    st.error("⚠️ API Key missing! Check your .env file.")
    st.stop()

# NOTE: The "Startup Spinner" block has been removed to ensure the UI renders instantly.
# The embedding model will now load automatically when you first use it.

# Session Persistence Check
if os.path.exists("./chroma_db"):
    st.session_state['processed'] = True

# --- SIDEBAR ---
with st.sidebar:
    st.title("🧠 DocuMind")
    st.caption("AI Study Assistant")
    
    # --- MODEL SWITCHER ---
    st.subheader("⚙️ AI Configuration")
    model_choice = st.selectbox(
        "Select Model:",
        ["Gemini Flash (Fast)", "Gemini Pro (Smart)"],
        index=0
    )
    
    st.divider()

    # Upload Section
    uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type="pdf")
    
    if st.button("Process Docs", type="primary"):
        if uploaded_files:
            # The spinner will run here while the model loads in the background
            with st.spinner("Analyzing Content (This may take a moment first time)..."):
                raw_text = get_pdf_text(uploaded_files)
                success = create_vector_store(raw_text)
                if success:
                    st.success("✅ Knowledge Base Ready!")
                    st.session_state['processed'] = True
                    st.rerun()
        else:
            st.warning("Please upload a file first.")
            
    # Reset Button
    if st.button("Clear Data"):
        if os.path.exists("./chroma_db"):
            import shutil
            try:
                shutil.rmtree("./chroma_db")
            except:
                pass 
        st.session_state.clear()
        st.rerun()

# --- MAIN UI ---
st.header("📚 DocuMind Dashboard")
st.caption(f"🚀 Active Model: **{model_choice}**")

if 'processed' in st.session_state:
    tab1, tab2 = st.tabs(["💬 Chat", "📝 Quiz"])
    
    # Chat Tab
    with tab1:
        if "messages" not in st.session_state: st.session_state.messages = []
        
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])
            
        if prompt := st.chat_input("Ask a question..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            
            # --- START SIMPLE STABLE FEEDBACK ---
            try:
                # Use a single, reliable spinner for the entire process
                with st.spinner(f"Thinking with {model_choice}..."):
                    response_data = get_rag_response(prompt, model_choice)
                
                # Check if the result is a tuple (success) or a string (error)
                if isinstance(response_data, tuple):
                    response, optimized_query = response_data
                    
                    # 1. Show Optimized Query (Demonstration Feature)
                    st.markdown(f"> **Query for Search:** `{optimized_query}`")
                    st.markdown("---")
                    
                    # 2. Show Response and Save Message
                    st.chat_message("assistant").write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                
                else:
                    # Handle the error string returned from logic.py
                    st.chat_message("assistant").error(response_data)
                    # We do not save the error message to session state
                    
            except Exception as final_error:
                # If a crash happens, print the error to the screen instead of going white
                st.error(f"FATAL UI ERROR: {final_error}")

    # Quiz Tab
    with tab2:
        st.subheader("Generate a Quiz")
        topic = st.text_input("Enter Topic")
        
        if st.button("Create Quiz"):
            with st.spinner(f"Generating with {model_choice}..."):
                st.session_state['quiz'] = generate_quiz(topic, model_choice)
                st.session_state['user_answers'] = {}
        
        if 'quiz' in st.session_state and st.session_state['quiz']:
            with st.form("quiz_form"):
                score = 0
                questions = st.session_state['quiz']['questions']
                
                for i, q in enumerate(questions):
                    st.write(f"**Q{i+1}: {q['question']}**")
                    st.session_state['user_answers'][i] = st.radio(
                        "Choose:", q['options'], key=f"q{i}", index=None
                    )
                    st.markdown("---")
                
                if st.form_submit_button("Submit"):
                    for i, q in enumerate(questions):
                        if st.session_state['user_answers'].get(i) == q['answer']:
                            score += 1
                    st.success(f"Score: {score}/{len(questions)}")
                    with st.expander("Show Answers"):
                        for q in questions:
                            st.write(f"**{q['question']}** \n*Answer:* {q['answer']} \n*Reason:* {q['explanation']}")
else:
    st.info("👈 Upload PDF to start.")