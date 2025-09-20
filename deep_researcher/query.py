"""
Query handling and answer generation system.
Handles natural language queries, retrieval, and LLM-based answer generation.
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from .embeddings import DocumentIndexer
import json
import re


class QueryProcessor:
    """Handles query processing, retrieval, and answer generation."""
    
    def __init__(self, indexer: DocumentIndexer, model_name: str = "google/flan-t5-small"):
        """
        Initialize the query processor.
        
        Args:
            indexer: Document indexer instance
            model_name: Name of the T5 model for answer generation
        """
        self.indexer = indexer
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load the T5 model and tokenizer
        print(f"Loading T5 model: {model_name}")
        print(f"Using device: {self.device}")
        
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: Natural language query
            top_k: Number of chunks to retrieve
            
        Returns:
            List of relevant chunks with similarity scores
        """
        return self.indexer.search_documents(query, top_k)
    
    def format_context(self, chunks: List[Dict[str, Any]], max_context_length: int = 2000) -> str:
        """
        Format retrieved chunks into context for the LLM.
        
        Args:
            chunks: List of relevant chunks
            max_context_length: Maximum length of context in characters
            
        Returns:
            Formatted context string
        """
        if not chunks:
            return "No relevant information found."
        
        context_parts = []
        current_length = 0
        
        for i, chunk in enumerate(chunks):
            # Format chunk with source information
            chunk_text = f"Source {i+1} (from {chunk['metadata']['filename']}):\n{chunk['text']}\n"
            
            if current_length + len(chunk_text) > max_context_length:
                break
            
            context_parts.append(chunk_text)
            current_length += len(chunk_text)
        
        return "\n".join(context_parts)
    
    def generate_answer(self, query: str, context: str) -> str:
        """
        Generate an answer using the T5 model.
        
        Args:
            query: User query
            context: Retrieved context from documents
            
        Returns:
            Generated answer
        """
        # Create prompt for the model
        prompt = f"Question: {query}\nContext: {context}\nAnswer:"
        
        # Tokenize input
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
        inputs = inputs.to(self.device)
        
        # Generate answer
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=256,
                num_beams=4,
                early_stopping=True,
                do_sample=False,
                temperature=0.7
            )
        
        # Decode output
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up the answer
        answer = answer.strip()
        if answer.startswith("Answer:"):
            answer = answer[7:].strip()
        
        return answer
    
    def process_query(self, query: str, top_k: int = 5, include_sources: bool = True) -> Dict[str, Any]:
        """
        Process a complete query and return answer with sources.
        
        Args:
            query: Natural language query
            top_k: Number of chunks to retrieve
            include_sources: Whether to include source information
            
        Returns:
            Dictionary containing answer, sources, and metadata
        """
        print(f"Processing query: {query}")
        
        # Retrieve relevant chunks
        chunks = self.retrieve_relevant_chunks(query, top_k)
        
        if not chunks:
            return {
                'answer': "I couldn't find any relevant information to answer your question.",
                'sources': [],
                'chunks_used': [],
                'query': query,
                'confidence': 0.0
            }
        
        # Format context
        context = self.format_context(chunks)
        
        # Generate answer
        answer = self.generate_answer(query, context)
        
        # Prepare sources
        sources = []
        if include_sources:
            for chunk in chunks:
                source_info = {
                    'chunk_id': chunk['chunk_id'],
                    'filename': chunk['metadata']['filename'],
                    'similarity_score': chunk['similarity_score'],
                    'text_preview': chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text']
                }
                sources.append(source_info)
        
        # Calculate confidence based on similarity scores
        avg_similarity = sum(chunk['similarity_score'] for chunk in chunks) / len(chunks)
        confidence = min(avg_similarity, 1.0)
        
        return {
            'answer': answer,
            'sources': sources,
            'chunks_used': chunks,
            'query': query,
            'confidence': confidence,
            'context_used': context
        }
    
    def generate_summary(self, query: str = "Summarize the main findings", top_k: int = 10) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of the indexed documents.
        
        Args:
            query: Query for summarization
            top_k: Number of chunks to use for summary
            
        Returns:
            Dictionary containing summary and sources
        """
        print("Generating document summary...")
        
        # Get more chunks for summarization
        chunks = self.retrieve_relevant_chunks(query, top_k)
        
        if not chunks:
            return {
                'summary': "No documents available for summarization.",
                'sources': [],
                'total_chunks_used': 0
            }
        
        # Format context for summarization
        context = self.format_context(chunks, max_context_length=4000)
        
        # Generate summary
        summary = self.generate_answer(query, context)
        
        # Prepare sources
        sources = []
        for chunk in chunks:
            source_info = {
                'chunk_id': chunk['chunk_id'],
                'filename': chunk['metadata']['filename'],
                'similarity_score': chunk['similarity_score'],
                'text_preview': chunk['text'][:150] + "..." if len(chunk['text']) > 150 else chunk['text']
            }
            sources.append(source_info)
        
        return {
            'summary': summary,
            'sources': sources,
            'total_chunks_used': len(chunks),
            'query_used': query
        }
    
    def export_answer_to_markdown(self, result: Dict[str, Any], output_path: str) -> None:
        """
        Export query result to a Markdown file.
        
        Args:
            result: Query result dictionary
            output_path: Path to save the Markdown file
        """
        markdown_content = f"""# Query Result

## Query
{result['query']}

## Answer
{result['answer']}

## Confidence Score
{result['confidence']:.2f}

## Sources Used
"""
        
        for i, source in enumerate(result['sources'], 1):
            markdown_content += f"""
### Source {i}
- **File**: {source['filename']}
- **Chunk ID**: {source['chunk_id']}
- **Similarity Score**: {source['similarity_score']:.3f}
- **Text Preview**: {source['text_preview']}

---
"""
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        print(f"Answer exported to {output_path}")
    
    def get_available_documents(self) -> List[Dict[str, Any]]:
        """
        Get information about all available documents.
        
        Returns:
            List of document information
        """
        all_chunks = self.indexer.embedding_manager.get_all_chunks()
        
        # Group chunks by document
        docs = {}
        for chunk in all_chunks:
            doc_id = chunk['metadata']['doc_id']
            if doc_id not in docs:
                docs[doc_id] = {
                    'doc_id': doc_id,
                    'filename': chunk['metadata']['filename'],
                    'file_path': chunk['metadata']['file_path'],
                    'file_type': chunk['metadata']['file_type'],
                    'file_size': chunk['metadata']['file_size'],
                    'total_tokens': chunk['metadata']['total_tokens'],
                    'chunk_count': 0
                }
            docs[doc_id]['chunk_count'] += 1
        
        return list(docs.values())


class ResearchAssistant:
    """High-level interface for the research assistant."""
    
    def __init__(self, index_path: str = "data/faiss_index", model_name: str = "google/flan-t5-small"):
        """
        Initialize the research assistant.
        
        Args:
            index_path: Path to the FAISS index
            model_name: Name of the T5 model
        """
        self.indexer = DocumentIndexer(index_path=index_path)
        self.query_processor = QueryProcessor(self.indexer, model_name)
    
    def ask_question(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Ask a question and get an answer.
        
        Args:
            question: The question to ask
            top_k: Number of chunks to retrieve
            
        Returns:
            Query result with answer and sources
        """
        return self.query_processor.process_query(question, top_k)
    
    def summarize_documents(self, query: str = "Summarize the main findings", top_k: int = 10) -> Dict[str, Any]:
        """
        Generate a summary of all documents.
        
        Args:
            query: Query for summarization
            top_k: Number of chunks to use
            
        Returns:
            Summary result
        """
        return self.query_processor.generate_summary(query, top_k)
    
    def get_document_info(self) -> Dict[str, Any]:
        """
        Get information about indexed documents.
        
        Returns:
            Document statistics and information
        """
        stats = self.indexer.get_document_stats()
        documents = self.query_processor.get_available_documents()
        
        return {
            'stats': stats,
            'documents': documents,
            'total_documents': len(documents)
        }
    
    def export_result(self, result: Dict[str, Any], output_path: str) -> None:
        """
        Export a query result to Markdown.
        
        Args:
            result: Query result to export
            output_path: Output file path
        """
        self.query_processor.export_answer_to_markdown(result, output_path)


def main():
    """Test the query processor."""
    # This would require an existing index to test
    print("Query processor initialized. Use with an existing document index.")


if __name__ == "__main__":
    main()
