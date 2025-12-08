import os
import json
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HF_TOKEN:
    # If the token is not set in .env, try to get it from the environment directly
    HF_TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN")

if not HF_TOKEN:
    raise ValueError("Hugging Face token not found. Please set HUGGINGFACE_HUB_TOKEN in your .env file or environment variables.")

# Initialize clients
LLM_CLIENT = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.2", token=HF_TOKEN)
EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

# --- Helper Functions ---

def get_text_from_pdfs(pdf_paths: list) -> str:
    """Extracts text from a list of PDF file paths or uploaded file objects."""
    raw_text = ""
    for path_or_file in pdf_paths:
        # Check if it's a Streamlit UploadedFile object or a file path string
        if isinstance(path_or_file, str):
            # It's a file path, open it
            with open(path_or_file, 'rb') as file:
                pdf_reader = PdfReader(file)
                for page in pdf_reader.pages:
                    raw_text += page.extract_text() or ""
        else:
            # It's a Streamlit UploadedFile object, pass it directly
            pdf_reader = PdfReader(path_or_file)
            for page in pdf_reader.pages:
                raw_text += page.extract_text() or ""
    return raw_text

def chunk_text(raw_text: str) -> list[str]:
    """Splits raw text into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(raw_text)

# --- Core RAG Functions ---

def process_uploaded_pdfs(uploaded_files) -> tuple:
    """
    Processes uploaded PDF files into an in-memory searchable vectorstore.
    
    Args:
        uploaded_files: A list of uploaded file objects from Streamlit.

    Returns:
        A tuple containing the FAISS vector store and the list of text chunks.
    """
    raw_text = get_text_from_pdfs(uploaded_files)
    chunks = chunk_text(raw_text)
    
    embeddings = EMBEDDING_MODEL.encode(chunks, show_progress_bar=False)
    
    # Check dimensions for FAISS index creation
    if embeddings.shape[0] == 0:
        return None, None
        
    vector_store = faiss.IndexFlatL2(embeddings.shape[1])
    vector_store.add(embeddings)
    
    return vector_store, chunks

def create_and_save_vectorstore(docs_folder: str, vectorstore_folder: str):
    """Processes all PDF files in docs_folder and saves the FAISS index and chunks."""
    pdf_files = [os.path.join(docs_folder, f) for f in os.listdir(docs_folder) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"No PDF files found in {docs_folder}. Skipping vectorstore creation.")
        return
        
    raw_text = get_text_from_pdfs(pdf_files)
    chunks = chunk_text(raw_text)
    
    embeddings = EMBEDDING_MODEL.encode(chunks, show_progress_bar=True)
    
    if embeddings.shape[0] == 0:
        print("No text chunks created. Skipping vectorstore save.")
        return

    # Create in-memory vector store
    vector_store = faiss.IndexFlatL2(embeddings.shape[1])
    vector_store.add(embeddings)
    
    # Save the index and chunks
    faiss.write_index(vector_store, os.path.join(vectorstore_folder, "faiss_index.bin"))
    with open(os.path.join(vectorstore_folder, "text_chunks.json"), 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=4)
    print(f"Saved {len(chunks)} chunks and FAISS index.")

def load_vectorstore(vectorstore_folder: str) -> tuple | tuple[None, None]:
    """Loads the FAISS index and text chunks from disk."""
    index_path = os.path.join(vectorstore_folder, "faiss_index.bin")
    chunks_path = os.path.join(vectorstore_folder, "text_chunks.json")
    
    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        print(f"Vectorstore files not found in {vectorstore_folder}. Returning None.")
        return None, None
        
    print(f"Loading vectorstore from {vectorstore_folder}...")
    try:
        index = faiss.read_index(index_path)
        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        return index, chunks
    except Exception as e:
        print(f"Error loading vectorstore: {e}")
        return None, None

def retrieve_relevant_chunks(query: str, vector_store, text_chunks: list[str], k: int = 5) -> list[str]:
    """Retrieves the top k most relevant chunks for a given query."""
    query_embedding = EMBEDDING_MODEL.encode([query])
    _, indices = vector_store.search(query_embedding, k)
    # indices[0] contains the actual indices array
    return [text_chunks[i] for i in indices[0]]

def generate_answer(query: str, relevant_chunks: list[str]) -> str:
    """Generates an answer using the LLM based on the query and retrieved context."""
    context = "\n\n".join(relevant_chunks)
    
    prompt = f"""
    Use the following context to answer the user's question. If the answer is not in the context, say you don't know and politely explain that your knowledge is limited to the provided documents.

    Context:
    {context}

    Question:
    {query}
    """
    
    messages = [{"role": "user", "content": prompt}]
    
    try:
        response = LLM_CLIENT.chat_completion(
            messages=messages,
            max_tokens=512, # Increased max tokens for better answers
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred while generating the answer. Please check your Hugging Face token and network connection. Error: {e}"

