"""
Embedding generation and FAISS vector database management.
Handles text embedding creation, storage, and retrieval.
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import faiss
import json

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"Warning: sentence_transformers import failed: {e}")
    print("Please install: pip install sentence-transformers")
    SentenceTransformer = None


class EmbeddingManager:
    """Manages text embeddings and FAISS vector database operations."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", index_path: str = "data/faiss_index"):
        """
        Initialize the embedding manager.
        
        Args:
            model_name: Name of the sentence transformer model
            index_path: Path to store the FAISS index and metadata
        """
        self.model_name = model_name
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize the embedding model
        if SentenceTransformer is None:
            raise ImportError("sentence_transformers is not available. Please install it with: pip install sentence-transformers")
        
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Initialize FAISS index
        self.index = None
        self.chunk_metadata = []
        self.chunk_id_to_index = {}
        self.is_loaded = False
        
        # Try to load existing index
        self.load_index()
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Create embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Numpy array of embeddings
        """
        print(f"Creating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings
    
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Add new chunks to the FAISS index.
        
        Args:
            chunks: List of chunk dictionaries with text and metadata
        """
        if not chunks:
            return
        
        # Extract texts for embedding
        texts = [chunk['text'] for chunk in chunks]
        
        # Create embeddings
        embeddings = self.create_embeddings(texts)
        
        # Initialize index if it doesn't exist
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
            self.chunk_metadata = []
            self.chunk_id_to_index = {}
        
        # Add embeddings to index
        self.index.add(embeddings.astype('float32'))
        
        # Store metadata
        start_idx = len(self.chunk_metadata)
        for i, chunk in enumerate(chunks):
            chunk_idx = start_idx + i
            self.chunk_metadata.append(chunk)
            self.chunk_id_to_index[chunk['chunk_id']] = chunk_idx
        
        print(f"Added {len(chunks)} chunks to index. Total chunks: {len(self.chunk_metadata)}")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using a query.
        
        Args:
            query: Query text
            top_k: Number of top results to return
            
        Returns:
            List of similar chunks with scores
        """
        if self.index is None or len(self.chunk_metadata) == 0:
            return []
        
        # Create query embedding
        query_embedding = self.model.encode([query])
        
        # Search the index
        scores, indices = self.index.search(query_embedding.astype('float32'), min(top_k, len(self.chunk_metadata)))
        
        # Format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunk_metadata):
                chunk = self.chunk_metadata[idx].copy()
                chunk['similarity_score'] = float(score)
                results.append(chunk)
        
        return results
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific chunk by its ID.
        
        Args:
            chunk_id: ID of the chunk to retrieve
            
        Returns:
            Chunk dictionary or None if not found
        """
        if chunk_id in self.chunk_id_to_index:
            idx = self.chunk_id_to_index[chunk_id]
            return self.chunk_metadata[idx]
        return None
    
    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """
        Get all chunks in the index.
        
        Returns:
            List of all chunks
        """
        return self.chunk_metadata.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current index.
        
        Returns:
            Dictionary with index statistics
        """
        if self.index is None:
            return {
                'total_chunks': 0,
                'embedding_dimension': self.embedding_dim,
                'model_name': self.model_name,
                'index_type': 'None'
            }
        
        return {
            'total_chunks': len(self.chunk_metadata),
            'embedding_dimension': self.embedding_dim,
            'model_name': self.model_name,
            'index_type': 'IndexFlatIP',
            'is_trained': self.index.is_trained,
            'ntotal': self.index.ntotal
        }
    
    def save_index(self) -> None:
        """Save the FAISS index and metadata to disk."""
        if self.index is None:
            print("No index to save.")
            return
        
        # Save FAISS index
        faiss_path = self.index_path / "faiss_index.bin"
        faiss.write_index(self.index, str(faiss_path))
        
        # Save metadata
        metadata_path = self.index_path / "metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.chunk_metadata, f)
        
        # Save chunk ID mapping
        mapping_path = self.index_path / "chunk_mapping.pkl"
        with open(mapping_path, 'wb') as f:
            pickle.dump(self.chunk_id_to_index, f)
        
        # Save configuration
        config_path = self.index_path / "config.json"
        config = {
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dim,
            'total_chunks': len(self.chunk_metadata)
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Index saved to {self.index_path}")
    
    def load_index(self) -> bool:
        """
        Load the FAISS index and metadata from disk.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            faiss_path = self.index_path / "faiss_index.bin"
            metadata_path = self.index_path / "metadata.pkl"
            mapping_path = self.index_path / "chunk_mapping.pkl"
            
            if not all([faiss_path.exists(), metadata_path.exists(), mapping_path.exists()]):
                print("No existing index found.")
                return False
            
            # Load FAISS index
            self.index = faiss.read_index(str(faiss_path))
            
            # Load metadata
            with open(metadata_path, 'rb') as f:
                self.chunk_metadata = pickle.load(f)
            
            # Load chunk mapping
            with open(mapping_path, 'rb') as f:
                self.chunk_id_to_index = pickle.load(f)
            
            self.is_loaded = True
            print(f"Loaded index with {len(self.chunk_metadata)} chunks")
            return True
            
        except Exception as e:
            print(f"Error loading index: {str(e)}")
            return False
    
    def clear_index(self) -> None:
        """Clear the current index and metadata."""
        self.index = None
        self.chunk_metadata = []
        self.chunk_id_to_index = {}
        self.is_loaded = False
        print("Index cleared.")
    
    def rebuild_index(self) -> None:
        """Rebuild the entire index from scratch."""
        if not self.chunk_metadata:
            print("No chunks to rebuild index with.")
            return
        
        print("Rebuilding index...")
        self.clear_index()
        
        # Re-add all chunks
        chunks = self.chunk_metadata.copy()
        self.chunk_metadata = []
        self.add_chunks(chunks)
        
        print("Index rebuilt successfully.")


class DocumentIndexer:
    """High-level interface for document indexing operations."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", index_path: str = "data/faiss_index"):
        """
        Initialize the document indexer.
        
        Args:
            model_name: Name of the embedding model
            index_path: Path to store the index
        """
        self.embedding_manager = EmbeddingManager(model_name, index_path)
    
    def index_documents(self, chunks: List[Dict[str, Any]], save: bool = True) -> None:
        """
        Index a list of document chunks.
        
        Args:
            chunks: List of chunk dictionaries
            save: Whether to save the index after adding chunks
        """
        print(f"Indexing {len(chunks)} chunks...")
        self.embedding_manager.add_chunks(chunks)
        
        if save:
            self.embedding_manager.save_index()
    
    def search_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of relevant chunks with scores
        """
        return self.embedding_manager.search(query, top_k)
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about the indexed documents."""
        return self.embedding_manager.get_stats()


def main():
    """Test the embedding manager."""
    # Create a test indexer
    indexer = DocumentIndexer()
    
    # Test with sample chunks
    sample_chunks = [
        {
            'chunk_id': 'test_1',
            'text': 'This is a sample document about artificial intelligence and machine learning.',
            'tokens': 15,
            'metadata': {
                'doc_id': 'test_doc',
                'filename': 'test.txt',
                'chunk_index': 0
            }
        },
        {
            'chunk_id': 'test_2',
            'text': 'Natural language processing is a subset of AI that focuses on text understanding.',
            'tokens': 16,
            'metadata': {
                'doc_id': 'test_doc',
                'filename': 'test.txt',
                'chunk_index': 1
            }
        }
    ]
    
    # Index the chunks
    indexer.index_documents(sample_chunks)
    
    # Test search
    results = indexer.search_documents("What is artificial intelligence?", top_k=2)
    print(f"\nSearch results:")
    for i, result in enumerate(results):
        print(f"{i+1}. Score: {result['similarity_score']:.3f}")
        print(f"   Text: {result['text']}")
        print(f"   Chunk ID: {result['chunk_id']}")
        print()


if __name__ == "__main__":
    main()
