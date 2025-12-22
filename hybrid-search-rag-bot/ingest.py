"""
Data Ingestion Module
Handles PDF extraction, text processing, and semantic chunking.
"""

import io
import re
from typing import List, Tuple
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentIngester:
    """Handles document ingestion and text extraction from PDFs."""
    
    def __init__(self):
        """Initialize the text splitter for semantic chunking."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def extract_text_from_pdf(self, pdf_file) -> Tuple[str, int]:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_file: File-like object or bytes containing PDF data
            
        Returns:
            Tuple of (extracted_text, total_pages)
        """
        try:
            # Convert file to bytes if needed
            if hasattr(pdf_file, 'read'):
                pdf_bytes = pdf_file.read()
            else:
                pdf_bytes = pdf_file
            
            # Read PDF
            pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
            total_pages = len(pdf_reader.pages)
            
            # Extract text from all pages
            extracted_text = ""
            for page_num, page in enumerate(pdf_reader.pages, 1):
                text = page.extract_text()
                # Add page separator for tracking
                extracted_text += f"\n--- Page {page_num} ---\n{text}"
            
            return extracted_text, total_pages
        
        except Exception as e:
            raise ValueError(f"Error extracting PDF: {str(e)}")
    
    def chunk_text(self, text: str, metadata: dict = None) -> List[dict]:
        """
        Split text into semantic chunks with metadata.
        
        Args:
            text: Text to be chunked
            metadata: Optional metadata to attach to chunks (filename, page, etc.)
            
        Returns:
            List of chunks with metadata
        """
        if metadata is None:
            metadata = {}
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Create chunk objects with metadata
        chunk_objects = []
        for i, chunk in enumerate(chunks):
            # Extract page number if present in chunk
            page_match = re.search(r'--- Page (\d+) ---', chunk)
            page_number = int(page_match.group(1)) if page_match else 1
            
            chunk_obj = {
                "content": chunk,
                "chunk_id": f"{metadata.get('filename', 'unknown')}_{i}",
                "metadata": {
                    **metadata,
                    "page": page_number,
                    "chunk_index": i,
                }
            }
            chunk_objects.append(chunk_obj)
        
        return chunk_objects
    
    def process_pdf(self, pdf_file, filename: str) -> List[dict]:
        """
        Complete pipeline: Extract PDF → Chunk text → Return chunks with metadata.
        
        Args:
            pdf_file: File-like object or bytes
            filename: Name of the PDF file
            
        Returns:
            List of chunks with metadata
        """
        # Extract text from PDF
        text, total_pages = self.extract_text_from_pdf(pdf_file)
        
        # Chunk the text with metadata
        metadata = {
            "filename": filename,
            "source": filename,
            "total_pages": total_pages
        }
        
        chunks = self.chunk_text(text, metadata)
        
        return chunks


# Example usage and testing
if __name__ == "__main__":
    ingester = DocumentIngester()
    
    # Test with sample text
    sample_text = """
    This is a sample document for testing the chunking functionality.
    """ * 500  # Make it long enough to create multiple chunks
    
    chunks = ingester.chunk_text(sample_text, {"filename": "test.pdf"})
    
    print(f"Total chunks: {len(chunks)}")
    print(f"First chunk content length: {len(chunks[0]['content'])}")
    print(f"First chunk metadata: {chunks[0]['metadata']}")
