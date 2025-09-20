"""
Demo script for Deep Researcher Agent.
Shows how to use the system programmatically.
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


def create_demo_document():
    """Create a demo document for testing."""
    demo_content = """
    Introduction to Machine Learning

    Machine learning is a subset of artificial intelligence that enables computers 
    to learn and make decisions from data without being explicitly programmed. 
    It has revolutionized many industries including healthcare, finance, and technology.

    Types of Machine Learning:

    1. Supervised Learning: Uses labeled training data to learn a mapping from 
       inputs to outputs. Examples include classification and regression.

    2. Unsupervised Learning: Finds hidden patterns in data without labeled examples. 
       Examples include clustering and dimensionality reduction.

    3. Reinforcement Learning: Learns through interaction with an environment, 
       receiving rewards or penalties for actions.

    Popular Algorithms:
    - Linear Regression
    - Decision Trees
    - Random Forest
    - Support Vector Machines
    - Neural Networks
    - K-Means Clustering

    Applications:
    - Image recognition and computer vision
    - Natural language processing
    - Recommendation systems
    - Fraud detection
    - Medical diagnosis
    - Autonomous vehicles

    The future of machine learning looks promising with advances in deep learning, 
    quantum computing, and edge AI deployment.
    """
    
    # Create demo directory
    demo_dir = Path("demo_docs")
    demo_dir.mkdir(exist_ok=True)
    
    # Write demo document
    demo_file = demo_dir / "machine_learning.txt"
    with open(demo_file, 'w', encoding='utf-8') as f:
        f.write(demo_content)
    
    print(f"Created demo document: {demo_file}")
    return str(demo_dir)


def run_demo():
    """Run the complete demo."""
    print("🔍 Deep Researcher Agent - Demo")
    print("=" * 40)
    
    try:
        # Step 1: Create demo document
        print("\n1. Creating demo document...")
        demo_dir = create_demo_document()
        
        # Step 2: Process document
        print("\n2. Processing document...")
        processor = DocumentProcessor(chunk_size=200, chunk_overlap=30)
        chunks = processor.process_directory(demo_dir)
        print(f"   Created {len(chunks)} chunks")
        
        # Step 3: Create embeddings and index
        print("\n3. Creating embeddings and index...")
        indexer = DocumentIndexer(index_path="demo_index")
        indexer.index_documents(chunks, save=True)
        print("   Index created successfully")
        
        # Step 4: Initialize research assistant
        print("\n4. Initializing research assistant...")
        assistant = ResearchAssistant(index_path="demo_index")
        print("   Assistant ready!")
        
        # Step 5: Ask questions
        print("\n5. Asking questions...")
        questions = [
            "What is machine learning?",
            "What are the types of machine learning?",
            "What are some popular algorithms?",
            "What are the applications of machine learning?"
        ]
        
        for i, question in enumerate(questions, 1):
            print(f"\n   Question {i}: {question}")
            result = assistant.ask_question(question, top_k=3)
            print(f"   Answer: {result['answer']}")
            print(f"   Confidence: {result['confidence']:.2f}")
            print(f"   Sources: {len(result['sources'])}")
        
        # Step 6: Generate summary
        print("\n6. Generating summary...")
        summary = assistant.summarize_documents(top_k=5)
        print(f"   Summary: {summary['summary']}")
        
        # Step 7: Show document info
        print("\n7. Document information:")
        info = assistant.get_document_info()
        print(f"   Total chunks: {info['stats']['total_chunks']}")
        print(f"   Documents: {info['total_documents']}")
        
        print("\n✅ Demo completed successfully!")
        print("\nNext steps:")
        print("  - Run 'streamlit run app.py' for the web interface")
        print("  - Run 'python cli.py --help' for CLI commands")
        print("  - Upload your own documents and start researching!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        import shutil
        demo_dir = Path("demo_docs")
        demo_index = Path("demo_index")
        
        if demo_dir.exists():
            shutil.rmtree(demo_dir)
            print(f"\nCleaned up {demo_dir}")
        
        if demo_index.exists():
            shutil.rmtree(demo_index)
            print(f"Cleaned up {demo_index}")


if __name__ == "__main__":
    run_demo()
