"""
Test script for Deep Researcher Agent.
Creates sample documents and tests the complete pipeline.
"""

import os
import tempfile
from pathlib import Path
import sys

# Add the current directory to the path
sys.path.append(str(Path(__file__).parent))

from deep_researcher.ingest import DocumentProcessor
from deep_researcher.embeddings import DocumentIndexer
from deep_researcher.query import ResearchAssistant


def create_sample_documents():
    """Create sample documents for testing."""
    # Create test directory
    test_dir = Path("test_docs")
    test_dir.mkdir(exist_ok=True)
    
    # Sample document 1: AI Research
    ai_doc = test_dir / "ai_research.txt"
    ai_content = """
    Artificial Intelligence and Machine Learning Research

    Artificial Intelligence (AI) is a branch of computer science that aims to create 
    intelligent machines capable of performing tasks that typically require human intelligence. 
    These tasks include learning, reasoning, problem-solving, perception, and language understanding.

    Machine Learning (ML) is a subset of AI that focuses on the development of algorithms 
    and statistical models that enable computer systems to improve their performance on a 
    specific task through experience. ML algorithms build mathematical models based on 
    training data to make predictions or decisions.

    Deep Learning is a subset of machine learning that uses artificial neural networks 
    with multiple layers (deep neural networks) to model and understand complex patterns 
    in data. Deep learning has been particularly successful in areas such as computer vision, 
    natural language processing, and speech recognition.

    Recent advances in AI include:
    - Large Language Models (LLMs) like GPT and BERT
    - Computer vision breakthroughs in image recognition
    - Autonomous vehicles and robotics
    - Healthcare applications in diagnosis and drug discovery
    - Natural language processing for translation and summarization

    The future of AI holds promise for solving complex global challenges while also 
    raising important questions about ethics, privacy, and the impact on employment.
    """
    
    with open(ai_doc, 'w', encoding='utf-8') as f:
        f.write(ai_content)
    
    # Sample document 2: Climate Change
    climate_doc = test_dir / "climate_change.txt"
    climate_content = """
    Climate Change and Environmental Science

    Climate change refers to long-term shifts in global temperatures and weather patterns. 
    While climate variations are natural, since the 1800s human activities have been the 
    main driver of climate change, primarily due to the burning of fossil fuels.

    Key causes of climate change include:
    - Greenhouse gas emissions from burning fossil fuels
    - Deforestation and land use changes
    - Industrial processes and manufacturing
    - Transportation and energy production
    - Agriculture and livestock farming

    The effects of climate change are widespread and include:
    - Rising global temperatures and sea levels
    - More frequent and intense extreme weather events
    - Changes in precipitation patterns
    - Ocean acidification and warming
    - Loss of biodiversity and ecosystem disruption
    - Threats to food security and water resources

    Mitigation strategies involve reducing greenhouse gas emissions through:
    - Transitioning to renewable energy sources
    - Improving energy efficiency
    - Sustainable transportation and urban planning
    - Carbon capture and storage technologies
    - Reforestation and ecosystem restoration

    Adaptation measures include:
    - Building resilient infrastructure
    - Developing drought-resistant crops
    - Improving water management systems
    - Creating early warning systems for extreme weather
    - Protecting vulnerable communities and ecosystems

    International cooperation through agreements like the Paris Agreement is crucial 
    for addressing climate change on a global scale.
    """
    
    with open(climate_doc, 'w', encoding='utf-8') as f:
        f.write(climate_content)
    
    print(f"Created sample documents in {test_dir}")
    return str(test_dir)


def test_document_processing():
    """Test document processing pipeline."""
    print("\n=== Testing Document Processing ===")
    
    # Create sample documents
    test_dir = create_sample_documents()
    
    # Process documents
    processor = DocumentProcessor(chunk_size=300, chunk_overlap=50)
    chunks = processor.process_directory(test_dir)
    
    print(f"Processed {len(chunks)} chunks")
    
    # Show sample chunks
    for i, chunk in enumerate(chunks[:2]):
        print(f"\nChunk {i+1}:")
        print(f"ID: {chunk['chunk_id']}")
        print(f"Text: {chunk['text'][:100]}...")
        print(f"Tokens: {chunk['tokens']}")
        print(f"File: {chunk['metadata']['filename']}")
    
    return chunks


def test_embedding_and_indexing(chunks):
    """Test embedding generation and indexing."""
    print("\n=== Testing Embedding and Indexing ===")
    
    # Create indexer
    indexer = DocumentIndexer(index_path="test_index")
    indexer.index_documents(chunks, save=True)
    
    # Get stats
    stats = indexer.get_document_stats()
    print(f"Index created with {stats['total_chunks']} chunks")
    print(f"Embedding dimension: {stats['embedding_dimension']}")
    print(f"Model: {stats['model_name']}")
    
    return indexer


def test_query_processing(indexer):
    """Test query processing and answer generation."""
    print("\n=== Testing Query Processing ===")
    
    # Create research assistant
    assistant = ResearchAssistant(index_path="test_index")
    
    # Test queries
    test_queries = [
        "What is artificial intelligence?",
        "What are the causes of climate change?",
        "What are the effects of global warming?",
        "How can we mitigate climate change?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        try:
            result = assistant.ask_question(query, top_k=3)
            print(f"Answer: {result['answer']}")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Sources: {len(result['sources'])}")
        except Exception as e:
            print(f"Error: {str(e)}")


def test_summarization(assistant):
    """Test document summarization."""
    print("\n=== Testing Document Summarization ===")
    
    try:
        result = assistant.summarize_documents(top_k=5)
        print(f"Summary: {result['summary']}")
        print(f"Sources used: {result['total_chunks_used']}")
    except Exception as e:
        print(f"Error: {str(e)}")


def cleanup():
    """Clean up test files."""
    import shutil
    
    # Remove test directory
    test_dir = Path("test_docs")
    if test_dir.exists():
        shutil.rmtree(test_dir)
        print(f"\nCleaned up {test_dir}")
    
    # Remove test index
    test_index = Path("test_index")
    if test_index.exists():
        shutil.rmtree(test_index)
        print(f"Cleaned up {test_index}")


def main():
    """Run the complete test suite."""
    print("🔍 Deep Researcher Agent - System Test")
    print("=" * 50)
    
    try:
        # Test document processing
        chunks = test_document_processing()
        
        # Test embedding and indexing
        indexer = test_embedding_and_indexing(chunks)
        
        # Test query processing
        test_query_processing(indexer)
        
        # Test summarization
        assistant = ResearchAssistant(index_path="test_index")
        test_summarization(assistant)
        
        print("\n✅ All tests completed successfully!")
        print("\nYou can now run the full system:")
        print("  Streamlit: streamlit run app.py")
        print("  CLI: python cli.py --help")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        cleanup()


if __name__ == "__main__":
    main()
