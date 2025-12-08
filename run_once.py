import rag_utils
import os

if __name__ == "__main__":
    print("--- Starting Offline Vectorstore Creation ---")
    
    # Define paths
    docs_folder = "docs"
    vectorstore_folder = "vectorstore"
    
    # Check if the vectorstore directory exists, if not create it
    if not os.path.exists(vectorstore_folder):
        os.makedirs(vectorstore_folder)
        print(f"Created directory: '{vectorstore_folder}'")

    # Ensure the docs folder exists
    if not os.path.exists(docs_folder):
        os.makedirs(docs_folder)
        print(f"Created directory: '{docs_folder}'. Please place your PDF files inside it.")
        
    # Create and save the vectorstore
    rag_utils.create_and_save_vectorstore(docs_folder, vectorstore_folder)
    
    print("\n--- Process Complete ---")
    print(f"Vectorstore created and saved in the '{vectorstore_folder}' directory.")
    print("You can now run the main application with 'streamlit run app.py'")
