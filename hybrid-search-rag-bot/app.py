import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from ingest import DocumentIngester
from rag_engine import RAGEngine
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Enterprise RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🤖 Enterprise RAG Chatbot")
st.markdown(
    """
    A production-ready Retrieval-Augmented Generation (RAG) chatbot 
    powered by Groq's Llama 3.3 70B model.
    """
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = None

if "uploaded_documents" not in st.session_state:
    st.session_state.uploaded_documents = {}

if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = []

if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None

# Initialize the document ingester
ingester = DocumentIngester()

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key input
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Get your API key from https://console.groq.com"
    )
    
    if api_key:
        st.session_state.groq_api_key = api_key
    
    # Try to load from secrets if available
    if not st.session_state.groq_api_key:
        try:
            st.session_state.groq_api_key = st.secrets.get("GROQ_API_KEY")
        except:
            pass
    
    st.divider()
    
    # Initialize RAG Engine if we have all required credentials
    if st.session_state.groq_api_key and not st.session_state.rag_engine:
        try:
            pinecone_key = os.getenv("PINECONE_API_KEY") or st.secrets.get("PINECONE_API_KEY")
            pinecone_env = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")
            pinecone_index = os.getenv("PINECONE_INDEX_NAME", "rag-chatbot")
            
            if pinecone_key:
                with st.spinner("Initializing RAG Engine..."):
                    st.session_state.rag_engine = RAGEngine(
                        groq_api_key=st.session_state.groq_api_key,
                        pinecone_api_key=pinecone_key,
                        pinecone_environment=pinecone_env,
                        pinecone_index_name=pinecone_index
                    )
                st.success("✅ RAG Engine initialized!")
        except Exception as e:
            st.warning(f"⚠️ Could not initialize RAG Engine: {str(e)}")
    
    st.divider()
    
    st.subheader("📄 Document Management")
    
    # PDF upload
    uploaded_pdf = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        help="Upload a PDF document to index"
    )
    
    if uploaded_pdf is not None:
        if uploaded_pdf.name not in st.session_state.uploaded_documents:
            with st.spinner(f"Processing {uploaded_pdf.name}..."):
                try:
                    # Process the PDF
                    chunks = ingester.process_pdf(uploaded_pdf, uploaded_pdf.name)
                    st.session_state.uploaded_documents[uploaded_pdf.name] = {
                        "file": uploaded_pdf,
                        "chunk_count": len(chunks)
                    }
                    st.session_state.document_chunks.extend(chunks)
                    
                    # Index chunks in RAG engine if available
                    if st.session_state.rag_engine:
                        with st.spinner("Indexing chunks in Pinecone..."):
                            success = st.session_state.rag_engine.index_documents(chunks)
                            if success:
                                st.success(f"✅ Processed {uploaded_pdf.name} ({len(chunks)} chunks indexed)")
                            else:
                                st.warning(f"⚠️ Processed {uploaded_pdf.name} but indexing failed")
                    else:
                        st.success(f"✅ Processed {uploaded_pdf.name} ({len(chunks)} chunks)")
                except Exception as e:
                    st.error(f"❌ Error processing PDF: {str(e)}")
        else:
            st.info(f"ℹ️ {uploaded_pdf.name} already uploaded")
    
    # Display uploaded documents
    if st.session_state.uploaded_documents:
        st.write("**Uploaded Documents:**")
        for doc_name, doc_info in st.session_state.uploaded_documents.items():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"📄 {doc_name}")
            with col2:
                st.caption(f"{doc_info['chunk_count']} chunks")
        
        # Clear documents button
        if st.button("🗑️ Clear All Documents"):
            st.session_state.uploaded_documents = {}
            st.session_state.document_chunks = []
            st.rerun()
    
    # Debug section
    with st.expander("🔍 Debug - View Chunks"):
        if st.session_state.document_chunks:
            st.write(f"**Total chunks: {len(st.session_state.document_chunks)}**")
            selected_chunk = st.number_input(
                "Select chunk to view",
                min_value=0,
                max_value=len(st.session_state.document_chunks) - 1,
                value=0
            )
            chunk = st.session_state.document_chunks[selected_chunk]
            st.json({
                "chunk_id": chunk["chunk_id"],
                "metadata": chunk["metadata"],
                "content_preview": chunk["content"][:200] + "..."
            })
        else:
            st.write("No chunks available yet. Upload a PDF first!")

# Main chat interface
st.subheader("💬 Chat")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Check for API key
    if not st.session_state.groq_api_key:
        st.error("❌ Please provide your Groq API Key in the sidebar first!")
        st.stop()
    
    # Check if documents are uploaded
    if not st.session_state.document_chunks:
        st.toast("📄 Please upload a document first!", icon="📄")
        st.stop()
    
    # Check if RAG engine is initialized
    if not st.session_state.rag_engine:
        st.error("❌ RAG Engine not initialized. Check your Pinecone API key in environment variables.")
        st.stop()
    
    # Add user message to chat history
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Retrieve relevant context
    with st.spinner("🔍 Retrieving relevant documents..."):
        retrieved_chunks = st.session_state.rag_engine.retrieve(prompt, top_k=5)
    
    if not retrieved_chunks:
        st.warning("⚠️ No relevant documents found. Try rephrasing your question.")
        st.session_state.messages.pop()  # Remove the user message
        st.stop()
    
    # Generate response using streaming
    try:
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            # Stream response
            for chunk in st.session_state.rag_engine.generate_response_streaming(
                prompt, 
                retrieved_chunks,
                chat_history=st.session_state.messages[:-1]  # Exclude current user message
            ):
                full_response += chunk
                response_placeholder.write(full_response)
            
            # Display source citations
            if retrieved_chunks:
                st.divider()
                st.markdown("### 📚 Sources")
                citations = st.session_state.rag_engine._extract_citations(retrieved_chunks)
                for i, citation in enumerate(citations, 1):
                    st.caption(f"{i}. **{citation['filename']}** (Page {citation['page']}) - Score: {citation['score']:.2%}")
        
        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response
        })
        
    except Exception as e:
        st.error(f"❌ Error generating response: {str(e)}")
        # Remove the user message if there was an error
        st.session_state.messages.pop()
