import streamlit as st
import rag_utils
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Dynamic PDF Chatbot",
    page_icon="🤖",
    layout="wide"
)

# --- Constants ---
VECTORSTORE_FOLDER = "vectorstore"

# --- Main Application ---
st.title("Interactive PDF Q&A Chatbot 🤖")
st.write("Upload your PDF documents and ask questions directly, or use pre-processed documents.")

# --- Session State Initialization and Pre-Load ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "text_chunks" not in st.session_state:
    st.session_state.text_chunks = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ready" not in st.session_state:
    st.session_state.ready = False # Flag to indicate if documents are loaded/processed

# Attempt to load vector store on first run if it exists
if not st.session_state.ready and st.session_state.vector_store is None:
    # Check if pre-processed files exist and load them
    if os.path.exists(os.path.join(VECTORSTORE_FOLDER, "faiss_index.bin")):
        with st.spinner("Loading pre-processed documents..."):
            st.session_state.vector_store, st.session_state.text_chunks = rag_utils.load_vectorstore(VECTORSTORE_FOLDER)
            if st.session_state.vector_store is not None:
                st.session_state.messages.append({"role": "assistant", "content": f"**Pre-processed documents loaded!** ({len(st.session_state.text_chunks)} chunks). You can now ask questions."})
                st.session_state.ready = True
    else:
        st.session_state.ready = True # Set ready to True even if no files loaded, to prevent re-loading on every rerun

# --- Sidebar for PDF Upload ---
with st.sidebar:
    st.header("Upload Your Documents")
    
    uploaded_files = st.file_uploader(
        "Upload your PDF files here and click 'Process'", 
        accept_multiple_files=True,
        type="pdf"
    )

    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Processing documents... This may take a moment."):
                # Process the files and store the result in session state
                vector_store, text_chunks = rag_utils.process_uploaded_pdfs(uploaded_files)
                
                if vector_store is not None:
                    st.session_state.vector_store = vector_store
                    st.session_state.text_chunks = text_chunks
                    st.session_state.messages = [{"role": "assistant", "content": f"**Documents processed!** ({len(text_chunks)} chunks). You can now ask questions."}]
                    st.session_state.ready = True
                    st.success("Documents processed successfully!")
                else:
                    st.warning("Could not process documents (likely empty files).")
        else:
            st.warning("Please upload at least one PDF file.")

# --- Chat Interface ---
# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Main chat logic
if user_query := st.chat_input("Ask a question about your documents..."):
    # First, check if documents have been processed/loaded
    if st.session_state.vector_store is None or st.session_state.text_chunks is None:
        st.warning("Please upload and process your documents or run 'run_once.py' before asking questions.")
        st.stop()

    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate and display assistant response
    nt", "content": response})
print(False)
