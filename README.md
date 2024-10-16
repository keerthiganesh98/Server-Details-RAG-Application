
# Server Details RAG Application

This project is a **Retrieval-Augmented Generation (RAG)** application for extracting and processing text from uploaded PDF files, specifically designed to handle server-related details. It leverages **LangChain** for text processing, **Chroma** for vector storage, and **OpenAI's GPT** to provide intelligent query responses based on extracted text.

## Features
- **PDF File Upload**: Allows multiple PDF files to be uploaded for processing.
- **Text Extraction**: Extracts text content from the uploaded PDF files.
- **Text Splitting**: Breaks down extracted content into manageable chunks using `RecursiveCharacterTextSplitter`.
- **Vector Store Creation**: Processes the split text and creates a vector store using **Chroma** and **OpenAI embeddings** for efficient retrieval.
- **Query Interface**: Allows users to input questions related to the uploaded documents and retrieves relevant information using a pre-trained OpenAI model.
- **Automatic Cleanup**: Removes outdated directories to maintain a clean working environment.

## Requirements
The application requires the following Python libraries:
- `streamlit`
- `PyPDF2`
- `openai`
- `langchain`
- `chromadb`
- `shutil`
  
Additionally, you need to set up your **OpenAI API Key** in the Streamlit secrets configuration:
1. In your project directory, create a `.streamlit/secrets.toml` file.
2. Add your OpenAI API key in the following format:
   ```toml
   [secrets]
   OPENAI_API_KEY = "your_openai_api_key_here"
   ```

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/server-details-rag.git
   cd server-details-rag
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API Key by adding it to the `secrets.toml` file as mentioned above.

## Usage

1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Upload PDF files containing server details via the file uploader.

3. Ask questions related to the uploaded server details in the query input field.

4. The app will extract relevant information from the PDF content and return an answer based on your query.

## Directory Structure

When you run the application, two directories are dynamically created:
- `data_YYYYMMDD_HHMMSS`: Stores the uploaded PDF files.
- `vector_storage_YYYYMMDD_HHMMSS`: Stores the vector store used for retrieving text embeddings.

Old directories will be cleaned up automatically after processing.

## Future Enhancements

- Add more advanced query capabilities such as summarization and contextual responses.
- Extend support for additional file formats beyond PDF.
- Implement session-based history of queries and responses.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue if you'd like to contribute.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
