"""
RAG Engine Module
Handles vector embeddings, Pinecone indexing, retrieval, and LLM-based generation.
"""

import os
import logging
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Handles local embeddings using Sentence Transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the sentence transformer model
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.dimension}")
    
    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single piece of text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        embedding = self.model.encode(text, convert_to_numpy=False)
        return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts efficiently.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(texts, convert_to_numpy=False)
        return [e.tolist() if hasattr(e, 'tolist') else list(e) for e in embeddings]


class PineconeVectorDB:
    """Manages Pinecone vector database operations with fallback to local storage."""
    
    def __init__(self, api_key: str, environment: str, index_name: str, dimension: int = 384):
        """
        Initialize Pinecone vector database.
        
        Args:
            api_key: Pinecone API key
            environment: Pinecone environment (e.g., 'us-east-1-aws')
            index_name: Name of the index
            dimension: Vector dimension (default: 384 for all-MiniLM-L6-v2)
        """
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.dimension = dimension
        self._local_vectors = {}  # Fallback local storage
        
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=api_key)
        self.index = None
        
        logger.info(f"Initialized Pinecone with index: {index_name}")
    
    def ensure_index_exists(self):
        """Create index if it doesn't exist."""
        try:
            # Check if index exists
            indexes = self.pc.list_indexes()
            index_names = [idx.get('name') for idx in indexes.get('indexes', [])]
            
            if self.index_name not in index_names:
                logger.info(f"Creating index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                logger.info(f"Index {self.index_name} created successfully")
            else:
                logger.info(f"Index {self.index_name} already exists")
            
            # Get index reference
            self.index = self.pc.Index(self.index_name)
            return True
        
        except Exception as e:
            logger.error(f"Error ensuring index exists: {str(e)}")
            logger.warning("Falling back to in-memory vector storage")
            # Initialize in-memory storage as fallback
            self._local_vectors = {}
            return False
    
    def upsert_vectors(self, vectors: List[Tuple[str, List[float], Dict]]):
        """
        Upsert vectors to Pinecone or local storage.
        
        Args:
            vectors: List of (id, embedding, metadata) tuples
        """
        try:
            if self.index:
                # Use Pinecone
                vectors_to_upsert = [
                    (vec_id, embedding, metadata)
                    for vec_id, embedding, metadata in vectors
                ]
                
                # Upsert in batches
                batch_size = 100
                for i in range(0, len(vectors_to_upsert), batch_size):
                    batch = vectors_to_upsert[i:i + batch_size]
                    self.index.upsert(vectors=batch)
                    logger.info(f"Upserted batch {i//batch_size + 1} ({len(batch)} vectors)")
                
                logger.info(f"Total vectors upserted to Pinecone: {len(vectors_to_upsert)}")
            else:
                # Fallback: Use local storage
                for vec_id, embedding, metadata in vectors:
                    self._local_vectors[vec_id] = {
                        'embedding': embedding,
                        'metadata': metadata
                    }
                logger.info(f"Vectors stored in local memory: {len(vectors)}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error upserting vectors: {str(e)}")
            return False
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """
        Search for similar vectors in Pinecone or local storage.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of search results with metadata
        """
        try:
            if self.index:
                # Use Pinecone
                results = self.index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    include_metadata=True
                )
                
                # Format results
                formatted_results = []
                for match in results.get('matches', []):
                    formatted_results.append({
                        'id': match.get('id'),
                        'score': match.get('score'),
                        'metadata': match.get('metadata', {})
                    })
                
                return formatted_results
            else:
                # Fallback: Local search using cosine similarity
                import numpy as np
                from sklearn.metrics.pairwise import cosine_similarity
                
                if not self._local_vectors:
                    return []
                
                query_vec = np.array(query_embedding).reshape(1, -1)
                results = []
                
                for vec_id, data in self._local_vectors.items():
                    embed = np.array(data['embedding']).reshape(1, -1)
                    score = cosine_similarity(query_vec, embed)[0][0]
                    results.append({
                        'id': vec_id,
                        'score': float(score),
                        'metadata': data['metadata']
                    })
                
                # Sort by score and return top-k
                results.sort(key=lambda x: x['score'], reverse=True)
                return results[:top_k]
        
        except Exception as e:
            logger.error(f"Error searching vectors: {str(e)}")
            return []


class RAGEngine:
    """Complete RAG pipeline combining embeddings, retrieval, and generation."""
    
    def __init__(self, 
                 groq_api_key: str,
                 pinecone_api_key: str,
                 pinecone_environment: str,
                 pinecone_index_name: str,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 groq_model: str = "llama-3.3-70b-versatile"):
        """
        Initialize the RAG engine.
        
        Args:
            groq_api_key: Groq API key
            pinecone_api_key: Pinecone API key
            pinecone_environment: Pinecone environment
            pinecone_index_name: Pinecone index name
            embedding_model: Name of embedding model
            groq_model: Name of Groq model
        """
        logger.info("Initializing RAG Engine...")
        
        # Initialize embeddings
        self.embeddings = EmbeddingModel(embedding_model)
        
        # Initialize vector database
        self.vector_db = PineconeVectorDB(
            api_key=pinecone_api_key,
            environment=pinecone_environment,
            index_name=pinecone_index_name,
            dimension=self.embeddings.dimension
        )
        
        # Ensure index exists
        if not self.vector_db.ensure_index_exists():
            logger.error("Failed to initialize Pinecone index")
        
        # Initialize LLM
        self.llm = ChatGroq(
            api_key=groq_api_key,
            model_name=groq_model,
            temperature=0.7,
            max_tokens=2048
        )
        
        # RAG prompt template
        self.system_prompt = """You are a helpful AI assistant powered by a Retrieval-Augmented Generation (RAG) system.
            
Use the provided context to answer the user's question accurately and helpfully.
If the context doesn't contain relevant information, say so honestly.
Always cite your sources when using information from the provided documents.

Context:
{context}"""
        
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{query}")
        ])
        
        logger.info("RAG Engine initialized successfully")
    
    def index_documents(self, chunks: List[Dict]) -> bool:
        """
        Index document chunks in Pinecone.
        
        Args:
            chunks: List of chunks with content and metadata
            
        Returns:
            Success status
        """
        logger.info(f"Indexing {len(chunks)} chunks...")
        
        try:
            # Extract texts for embedding
            texts = [chunk['content'] for chunk in chunks]
            
            # Embed all texts
            logger.info("Embedding texts...")
            embeddings = self.embeddings.embed_batch(texts)
            
            # Prepare vectors for upsert
            vectors = []
            for chunk, embedding in zip(chunks, embeddings):
                vector_id = chunk['chunk_id']
                metadata = {
                    'filename': chunk['metadata'].get('filename', 'unknown'),
                    'page': chunk['metadata'].get('page', 1),
                    'chunk_index': chunk['metadata'].get('chunk_index', 0),
                    'content': chunk['content'][:500]  # Store preview
                }
                vectors.append((vector_id, embedding, metadata))
            
            # Upsert to Pinecone
            success = self.vector_db.upsert_vectors(vectors)
            
            if success:
                logger.info(f"Successfully indexed {len(chunks)} chunks")
            
            return success
        
        except Exception as e:
            logger.error(f"Error indexing documents: {str(e)}")
            return False
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: User query
            top_k: Number of results to return
            
        Returns:
            List of relevant chunks with metadata
        """
        try:
            # Embed query
            query_embedding = self.embeddings.embed_text(query)
            
            # Search in vector DB
            results = self.vector_db.search(query_embedding, top_k=top_k)
            
            logger.info(f"Retrieved {len(results)} relevant chunks")
            return results
        
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []
    
    def generate_response(self, 
                         query: str, 
                         context_chunks: List[Dict],
                         chat_history: List[Dict] = None) -> Tuple[str, List[Dict]]:
        """
        Generate a response using retrieved context and chat history.
        
        Args:
            query: User query
            context_chunks: Retrieved context chunks
            chat_history: Previous chat messages
            
        Returns:
            Tuple of (response_text, source_citations)
        """
        try:
            # Format context
            context_text = self._format_context(context_chunks)
            
            # Build messages for LLM
            messages = []
            
            # Add system message
            messages.append(SystemMessage(content=self.system_prompt))
            
            # Add chat history
            if chat_history:
                for msg in chat_history:
                    if msg['role'] == 'user':
                        messages.append(HumanMessage(content=msg['content']))
                    else:
                        messages.append(AIMessage(content=msg['content']))
            
            # Add current query with context
            formatted_prompt = self.system_prompt.format(context=context_text) + "\n\nUser: " + query
            messages.append(HumanMessage(content=formatted_prompt))
            
            # Generate response
            response = self.llm.invoke(messages)
            response_text = response.content
            
            # Extract source citations
            citations = self._extract_citations(context_chunks)
            
            logger.info("Response generated successfully")
            return response_text, citations
        
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}", []
    
    def generate_response_streaming(self,
                                   query: str,
                                   context_chunks: List[Dict],
                                   chat_history: List[Dict] = None):
        """
        Generate a response with streaming output.
        
        Args:
            query: User query
            context_chunks: Retrieved context chunks
            chat_history: Previous chat messages
            
        Yields:
            Response text chunks
        """
        try:
            # Format context
            context_text = self._format_context(context_chunks)
            
            # Build messages
            messages = []
            messages.append(SystemMessage(content=self.system_prompt))
            
            if chat_history:
                for msg in chat_history:
                    if msg['role'] == 'user':
                        messages.append(HumanMessage(content=msg['content']))
                    else:
                        messages.append(AIMessage(content=msg['content']))
            
            formatted_prompt = self.system_prompt.format(context=context_text) + "\n\nUser: " + query
            messages.append(HumanMessage(content=formatted_prompt))
            
            # Stream response
            for chunk in self.llm.stream(messages):
                yield chunk.content
        
        except Exception as e:
            logger.error(f"Error in streaming response: {str(e)}")
            yield f"Error: {str(e)}"
    
    def _format_context(self, chunks: List[Dict]) -> str:
        """Format retrieved chunks for LLM context."""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            filename = chunk['metadata'].get('filename', 'Unknown')
            page = chunk['metadata'].get('page', 'N/A')
            content = chunk['metadata'].get('content', '')
            
            context_parts.append(
                f"[Source {i}: {filename} (Page {page})]\n{content}\n"
            )
        
        return "\n".join(context_parts)
    
    def _extract_citations(self, chunks: List[Dict]) -> List[Dict]:
        """Extract source citations from context chunks."""
        citations = []
        seen = set()
        
        for chunk in chunks:
            source_key = (
                chunk['metadata'].get('filename'),
                chunk['metadata'].get('page')
            )
            
            if source_key not in seen:
                citations.append({
                    'filename': chunk['metadata'].get('filename'),
                    'page': chunk['metadata'].get('page'),
                    'score': chunk.get('score')
                })
                seen.add(source_key)
        
        return citations


# Example usage and testing
if __name__ == "__main__":
    # Load environment variables
    groq_key = os.getenv("GROQ_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    pinecone_env = os.getenv("PINECONE_ENVIRONMENT")
    pinecone_index = os.getenv("PINECONE_INDEX_NAME")
    
    if not all([groq_key, pinecone_key, pinecone_env, pinecone_index]):
        print("Error: Missing required environment variables")
        exit(1)
    
    # Initialize RAG engine
    rag = RAGEngine(
        groq_api_key=groq_key,
        pinecone_api_key=pinecone_key,
        pinecone_environment=pinecone_env,
        pinecone_index_name=pinecone_index
    )
    
    # Test retrieval
    test_query = "What is machine learning?"
    results = rag.retrieve(test_query, top_k=3)
    print(f"\nRetrieved {len(results)} results for: {test_query}")
    
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"  Score: {result['score']:.4f}")
        print(f"  Source: {result['metadata'].get('filename')} (Page {result['metadata'].get('page')})")
