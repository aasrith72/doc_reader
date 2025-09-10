import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import streamlit as st
# Load environment variables from .env file
load_dotenv()
HF_TOKEN = st.secrets["Hugging_face"]
if not HF_TOKEN:
    raise ValueError("Hugging Face token not found. Please set HUGGINGFACE_HUB_TOKEN in your .env file.")

# Initialize clients
LLM_CLIENT = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.3", token=HF_TOKEN)
EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

def process_uploaded_pdfs(uploaded_files) -> tuple:
    """
    Processes uploaded PDF files into a searchable vectorstore.
    
    Args:
        uploaded_files: A list of uploaded file objects from Streamlit.

    Returns:
        A tuple containing the FAISS vector store and the list of text chunks.
    """
    raw_text = ""
    for file in uploaded_files:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            raw_text += page.extract_text() or ""
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    
    embeddings = EMBEDDING_MODEL.encode(chunks, show_progress_bar=True)
    
    vector_store = faiss.IndexFlatL2(embeddings.shape[1])
    vector_store.add(embeddings)
    
    return vector_store, chunks

def retrieve_relevant_chunks(query: str, vector_store, text_chunks: list[str], k: int = 5) -> list[str]:
    """Retrieves the top k most relevant chunks for a given query."""
    query_embedding = EMBEDDING_MODEL.encode([query])
    _, indices = vector_store.search(query_embedding, k)
    return [text_chunks[i] for i in indices[0]]

def generate_answer(query: str, relevant_chunks: list[str]) -> str:
    """Generates an answer using the LLM based on the query and retrieved context."""
    context = "\n\n".join(relevant_chunks)
    
    prompt = f"""
    Use the following context to answer the user's question. If the answer is not in the context, say you don't know.

    Context:
    {context}

    Question:
    {query}
    """
    
    messages = [{"role": "user", "content": prompt}]
    
    try:
        response = LLM_CLIENT.chat_completion(
            messages=messages,
            max_tokens=250,
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:

        return f"An error occurred while generating the answer: {e}"


