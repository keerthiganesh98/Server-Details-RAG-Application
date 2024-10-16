import streamlit as st
import os
import time
from datetime import datetime
import PyPDF2
import openai
#from langchain_community.embeddings import OpenAIEmbeddings
#from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_community.llms import OpenAI
from langchain_openai import OpenAI,OpenAIEmbeddings
from langchain.chains import RetrievalQA
import shutil


# Set your OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Function to create necessary directories
def create_directories():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_dir = f"data_{timestamp}"
    vector_dir = f"vector_storage_{timestamp}"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(vector_dir, exist_ok=True)
    return data_dir, vector_dir

# Function to extract text from PDF files
def extract_text_from_pdfs(uploaded_files, data_dir):
    text_content = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(data_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with open(file_path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                text_content.append(page.extract_text())
    return "\n".join(text_content)

# Function to process text and create vector store
def process_text_and_create_vectorstore(text_content, vector_dir):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(text_content)

    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_texts(texts, embeddings, persist_directory=vector_dir)
    #vectorstore.persist()
    return vectorstore

# Cleanup old directories
def cleanup_old_dirs(current_data_dir):
    
    for dir_name in os.listdir():
        if dir_name.startswith("data_") and dir_name != current_data_dir:
            shutil.rmtree(dir_name)
    

# Streamlit app
def main():
    st.title("Server Details RAG Application")

    # Initialize session state
    if 'data_dir' not in st.session_state:
        st.session_state.data_dir, st.session_state.vector_dir = create_directories()

    # File upload
    uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type="pdf")

    if uploaded_files:
        if st.button("Process Files"):
            with st.spinner("Processing PDF files..."):
                text_content = extract_text_from_pdfs(uploaded_files, st.session_state.data_dir)
                vectorstore = process_text_and_create_vectorstore(text_content, st.session_state.vector_dir)
                st.session_state.vectorstore = vectorstore
            st.success("Files processed and vector store created successfully!")

    # Query input
    if 'vectorstore' in st.session_state:
        query = st.text_input("Ask a question about the server details:")
        if query:
            with st.spinner("Searching for answer..."):
                qa_chain = RetrievalQA.from_chain_type(
                    llm=OpenAI(),
                    chain_type="stuff",
                    retriever=st.session_state.vectorstore.as_retriever()
                )
                response = qa_chain.run(query)
                st.write("Answer:", response)

    # Cleanup old directoriess
    cleanup_old_dirs(st.session_state.data_dir)


if __name__ == "__main__":
    main()