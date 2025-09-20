"""
Document ingestion pipeline for processing PDF and text files.
Handles text extraction, chunking, and metadata generation.
"""

import os
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import PyPDF2
import tiktoken


class DocumentProcessor:
    """Handles document processing, text extraction, and chunking."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Target size for text chunks in tokens
            chunk_overlap: Number of tokens to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
                
                return text.strip()
        except Exception as e:
            raise Exception(f"Error extracting text from PDF {file_path}: {str(e)}")
    
    def extract_text_from_txt(self, file_path: str) -> str:
        """
        Extract text from a text file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Extracted text content
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except Exception as e:
            raise Exception(f"Error extracting text from TXT {file_path}: {str(e)}")
    
    def extract_text(self, file_path: str) -> str:
        """
        Extract text from supported file types.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Extracted text content
        """
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_ext == '.txt':
            return self.extract_text_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned text content
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        return text.strip()
    
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in text.
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        return len(self.encoding.encode(text))
    
    def split_text_into_chunks(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Input text to chunk
            metadata: Metadata to attach to each chunk
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        # Clean the text first
        text = self.clean_text(text)
        
        # Split into sentences for better chunking
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        chunk_id = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            # If adding this sentence would exceed chunk size, save current chunk
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                chunks.append({
                    'chunk_id': f"{metadata['doc_id']}_{chunk_id}",
                    'text': current_chunk.strip(),
                    'tokens': current_tokens,
                    'metadata': {
                        **metadata,
                        'chunk_index': chunk_id,
                        'start_char': text.find(current_chunk),
                        'end_char': text.find(current_chunk) + len(current_chunk)
                    }
                })
                
                # Start new chunk with overlap
                overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else ""
                current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                current_tokens = self.count_tokens(current_chunk)
                chunk_id += 1
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_tokens += sentence_tokens
        
        # Add the last chunk if it has content
        if current_chunk.strip():
            chunks.append({
                'chunk_id': f"{metadata['doc_id']}_{chunk_id}",
                'text': current_chunk.strip(),
                'tokens': current_tokens,
                'metadata': {
                    **metadata,
                    'chunk_index': chunk_id,
                    'start_char': text.find(current_chunk),
                    'end_char': text.find(current_chunk) + len(current_chunk)
                }
            })
        
        return chunks
    
    def process_document(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Process a single document and return chunks.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of processed chunks
        """
        # Extract text
        text = self.extract_text(file_path)
        
        # Generate metadata
        file_name = Path(file_path).name
        file_size = os.path.getsize(file_path)
        
        metadata = {
            'doc_id': Path(file_path).stem,
            'filename': file_name,
            'file_path': file_path,
            'file_size': file_size,
            'file_type': Path(file_path).suffix.lower(),
            'total_tokens': self.count_tokens(text)
        }
        
        # Split into chunks
        chunks = self.split_text_into_chunks(text, metadata)
        
        return chunks
    
    def process_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        Process all supported documents in a directory.
        
        Args:
            directory_path: Path to the directory
            
        Returns:
            List of all processed chunks from all documents
        """
        all_chunks = []
        directory = Path(directory_path)
        
        # Find all supported files
        supported_extensions = ['.pdf', '.txt']
        files = []
        
        for ext in supported_extensions:
            files.extend(directory.glob(f"**/*{ext}"))
        
        print(f"Found {len(files)} documents to process...")
        
        for file_path in files:
            try:
                print(f"Processing: {file_path}")
                chunks = self.process_document(str(file_path))
                all_chunks.extend(chunks)
                print(f"  -> Generated {len(chunks)} chunks")
            except Exception as e:
                print(f"  -> Error processing {file_path}: {str(e)}")
        
        return all_chunks


def main():
    """Test the document processor."""
    processor = DocumentProcessor()
    
    # Test with a sample directory if it exists
    test_dir = "test_docs"
    if os.path.exists(test_dir):
        chunks = processor.process_directory(test_dir)
        print(f"\nTotal chunks generated: {len(chunks)}")
        
        # Print sample chunk
        if chunks:
            print(f"\nSample chunk:")
            print(f"ID: {chunks[0]['chunk_id']}")
            print(f"Text: {chunks[0]['text'][:200]}...")
            print(f"Tokens: {chunks[0]['tokens']}")
            print(f"Metadata: {chunks[0]['metadata']}")


if __name__ == "__main__":
    main()
