
# Intelligent RAG Q&A Chatbot

A powerful Retrieval-Augmented Generation (RAG) chatbot built with Python and Streamlit that can process multiple document formats and answer questions using AI.

## Features

- **Multi-format Document Support**: PDF, CSV, TXT, and image files (PNG, JPG, JPEG)
- **OCR Capabilities**: Extract text from images using Tesseract
- **Vector Search**: Semantic search using Supabase vector database
- **AI-Powered Responses**: Integration with OpenRouter API (Claude, Gemini, etc.)
- **Dark Theme UI**: Clean and modern Streamlit interface
- **Real-time Processing**: Upload and query documents instantly

## Setup Instructions

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Supabase Configuration

1. Go to [Supabase](https://supabase.com/) and create a new project
2. In the SQL Editor, run this query to create the documents table:

```sql
-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the documents table
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding VECTOR(384),
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index for similarity search
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Create the match_documents function for similarity search
CREATE OR REPLACE FUNCTION match_documents(
    query_embedding VECTOR(384),
    match_threshold FLOAT DEFAULT 0.3,
    match_count INT DEFAULT 3
)
RETURNS TABLE(
    id BIGINT,
    content TEXT,
    metadata JSONB,
    similarity FLOAT
)
LANGUAGE sql
AS $$
    SELECT
        id,
        content,
        metadata,
        1 - (documents.embedding <=> query_embedding) AS similarity
    FROM documents
    WHERE 1 - (documents.embedding <=> query_embedding) > match_threshold
    ORDER BY documents.embedding <=> query_embedding
    LIMIT match_count;
$$;
```

3. Set environment variables:

```bash
export SUPABASE_URL="your-supabase-project-url"
export SUPABASE_ANON_KEY="your-supabase-anon-key"
```

### 3. Install Tesseract (for OCR)

**Windows:**
- Download from: https://github.com/UB-Mannheim/tesseract/wiki
- Add to PATH

**macOS:**
```bash
brew install tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install tesseract-ocr
```

### 4. Run the Application

```bash
streamlit run app.py
```

## Usage

1. **Upload Documents**: Use the sidebar to upload PDF, CSV, TXT, or image files
2. **Process Documents**: Click "Process Documents" to extract text and create embeddings
3. **Ask Questions**: Type your questions in the chat interface
4. **Get AI Responses**: The system will find relevant documents and generate answers

## Project Structure

```
python_rag_project/
├── app.py                 # Main Streamlit application
├── document_processor.py  # Document processing and text extraction
├── vector_store.py       # Supabase vector database integration
├── rag_engine.py         # RAG logic with OpenRouter API
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Configuration

- **OpenRouter API Key**: Already configured in the code
- **Model Selection**: Choose from Claude, Gemini, or other models
- **Chunk Size**: Adjustable text chunking for better retrieval
- **Similarity Threshold**: Fine-tune document relevance

## Testing with Loan Approval Dataset

The system has been tested with the Loan Approval Prediction dataset from Kaggle, demonstrating its effectiveness in decision-support scenarios.

## Deployment

### Streamlit Community Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Set environment variables in the app settings
5. Deploy!

### Other Platforms

- **Heroku**: Add `Procfile` with `web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
- **Railway**: Direct deployment with automatic environment detection
- **Google Cloud Run**: Containerize with Docker

## Troubleshooting

1. **Supabase Connection Issues**: Verify URL and API key
2. **OCR Not Working**: Ensure Tesseract is properly installed
3. **Memory Issues**: Reduce batch size for large documents
4. **API Rate Limits**: Implement retry logic if needed

## License

MIT License - Feel free to use and modify for your projects!
