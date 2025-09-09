# app.py
import streamlit as st
import rag_utils
import os
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🧠",
    layout="wide"
)

# --- Custom CSS for a unique look ---
st.markdown("""
<style>
    .stChatMessage {
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


# --- Constants ---
VECTORSTORE_FOLDER = "vectorstore"

# --- Functions ---
@st.cache_data
def load_data():
    """Loads the FAISS index and text chunks once and caches them."""
    if not os.path.exists(VECTORSTORE_FOLDER) or not os.listdir(VECTORSTORE_FOLDER):
        st.error("Vectorstore not found. Please run 'python run_once.py' first to build it.")
        st.stop()
    return rag_utils.load_vectorstore(VECTORSTORE_FOLDER)

# --- Sidebar ---
with st.sidebar:
    st.header("DocuMind AI 🧠")
    st.write("Your personal assistant for querying PDF documents.")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Chat history cleared. How can I help you with your documents?"}
        ]
        st.rerun()
    st.markdown("---")
    st.write("Built with the RAG pipeline using:")
    st.markdown("- **Hugging Face**")
    st.markdown("- **Sentence-Transformers**")
    st.markdown("- **FAISS**")
    st.markdown("- **Streamlit**")


# --- Main Application ---
st.title("Ask Questions About Your Documents")
st.write("This chatbot uses a RAG pipeline to answer questions from your custom PDFs.")

# Load the vectorstore
vector_store, text_chunks = load_data()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I help you with your documents today?"}
    ]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "timestamp" in message:
            st.caption(f"Sent at {message['timestamp']}")

# Accept user input
if user_query := st.chat_input("What is your question?"):
    # Add user message to history
    user_message = {
        "role": "user", 
        "content": user_query,
        "timestamp": datetime.now().strftime("%I:%M %p, %d-%b-%Y")
    }
    st.session_state.messages.append(user_message)
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_query)
        st.caption(f"Sent at {user_message['timestamp']}")

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            retrieved_chunks = rag_utils.retrieve_relevant_chunks(user_query, vector_store, text_chunks)
            response = rag_utils.generate_answer(user_query, retrieved_chunks)
            
            assistant_message = {
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now().strftime("%I:%M %p, %d-%b-%Y")
            }
            
            st.markdown(response)
            st.caption(f"Sent at {assistant_message['timestamp']}")
    
    # Add assistant response to history
    st.session_state.messages.append(assistant_message)