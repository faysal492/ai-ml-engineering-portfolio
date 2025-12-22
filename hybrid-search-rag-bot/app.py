import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from ingest import DocumentIngester
import os
import json

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
    
    # Add user message to chat history
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Generate response using Groq
    try:
        chat = ChatGroq(
            temperature=0.7,
            groq_api_key=st.session_state.groq_api_key,
            model_name="llama-3.3-70b-versatile",
        )
        
        # Build messages for LLM
        messages = []
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        
        # Get response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            # Stream response
            for chunk in chat.stream(messages):
                full_response += chunk.content
                response_placeholder.write(full_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response
        })
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        # Remove the user message if there was an error
        st.session_state.messages.pop()
