
import streamlit as st
import os
from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag_engine import RAGEngine
import tempfile

# Set page config
st.set_page_config(
    page_title="Intelligent RAG Q&A Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply dark theme
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stSidebar {
        background-color: #262730;
    }
    .stTextInput > div > div > input {
        background-color: #262730;
        color: #FAFAFA;
    }
    .stSelectbox > div > div > select {
        background-color: #262730;
        color: #FAFAFA;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = VectorStore()
    if 'document_processor' not in st.session_state:
        st.session_state.document_processor = DocumentProcessor()
    if 'rag_engine' not in st.session_state:
        st.session_state.rag_engine = RAGEngine()

def main():
    initialize_session_state()
    
    st.title("🤖 Intelligent RAG Q&A Chatbot")
    st.markdown("Ask questions about your uploaded documents using AI-powered retrieval and generation.")
    
    # Sidebar for document upload and settings
    with st.sidebar:
        st.header("📁 Document Management")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=['csv', 'txt',],
            accept_multiple_files=True,
            help="Supported formats:  CSV, TXT"
        )
        
        # Model selection
        st.header("⚙️ Settings")
        model_choice = st.selectbox(
            "Select AI Model",
            ["anthropic/claude-3-haiku", "google/gemini-pro", "microsoft/wizardlm-2-8x22b"]
        )
        
        # Process documents button
        if uploaded_files and st.button("🔄 Process Documents"):
            with st.spinner("Processing documents..."):
                for uploaded_file in uploaded_files:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    # Process document
                    try:
                        chunks = st.session_state.document_processor.process_document(tmp_file_path, uploaded_file.name)
                        st.session_state.vector_store.add_documents(chunks)
                        st.success(f"✅ Processed {uploaded_file.name}")
                    except Exception as e:
                        st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                    finally:
                        os.unlink(tmp_file_path)
        
        # Display document count
        doc_count = st.session_state.vector_store.get_document_count()
        st.info(f"📊 Documents in database: {doc_count}")
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for i, (question, answer) in enumerate(st.session_state.chat_history):
                st.markdown(f"**🙋 You:** {question}")
                st.markdown(f"**🤖 Assistant:** {answer}")
                st.markdown("---")
        
        # Query input
        query = st.text_input("💬 Ask a question about your documents:", key="query_input")
        
        col_send, col_clear = st.columns([1, 1])
        with col_send:
            if st.button("🚀 Send", use_container_width=True):
                if query and doc_count > 0:
                    with st.spinner("Generating response..."):
                        try:
                            # Get relevant documents
                            relevant_docs = st.session_state.vector_store.similarity_search(query, k=3)
                            
                            # Generate response
                            response = st.session_state.rag_engine.generate_response(
                                query, relevant_docs, model_choice
                            )
                            
                            # Add to chat history
                            st.session_state.chat_history.append((query, response))
                            st.experimental_rerun()
                            
                        except Exception as e:
                            st.error(f"❌ Error generating response: {str(e)}")
                elif doc_count == 0:
                    st.warning("⚠️ Please upload and process documents first.")
                else:
                    st.warning("⚠️ Please enter a question.")
        
        with col_clear:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.experimental_rerun()
    
    with col2:
        st.header("📈 Statistics")
        st.metric("Total Questions", len(st.session_state.chat_history))
        st.metric("Documents", doc_count)
        
        if st.session_state.chat_history:
            st.header("🕒 Recent Questions")
            for question, _ in st.session_state.chat_history[-3:]:
                st.text(f"• {question[:50]}...")

if __name__ == "__main__":
    main()
