# run_once.py
import rag_utils
import os

if __name__ == "__main__":
    print("Starting the vectorstore creation process...")
    
    # Define paths
    docs_folder = "docs"
    vectorstore_folder = "vectorstore"
    
    # Check if the vectorstore directory exists, if not create it
    if not os.path.exists(vectorstore_folder):
        os.makedirs(vectorstore_folder)
        
    # Create and save the vectorstore
    rag_utils.create_and_save_vectorstore(docs_folder, vectorstore_folder)
    
    print(f"Vectorstore created and saved in the '{vectorstore_folder}' directory.")
    print("You can now run the main application with 'streamlit run app.py'")