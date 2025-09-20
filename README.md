# 🔍 Deep Researcher Agent

**CodeMate Submission - AI-Powered Research Assistant**

An AI-powered research assistant that can ingest PDFs and text documents, generate embeddings locally, index them into a vector database, and allow natural language queries with retrieval + summarization.

## 🎯 Project Status
✅ **Ready for CodeMate Submission**  
✅ **All Core Requirements Implemented**  
✅ **Advanced Features Included**  
✅ **Clean, Modular Code Architecture**

## ✨ Features

- **📄 Document Ingestion**: Support for PDF and TXT files with intelligent text chunking
- **🧠 Local AI Processing**: Uses sentence-transformers for embeddings and T5 models for answer generation
- **🔍 Vector Search**: FAISS-based similarity search for relevant document chunks
- **💬 Natural Language Queries**: Ask questions in plain English and get AI-generated answers
- **📊 Document Summarization**: Generate comprehensive summaries of your documents
- **📚 Source Tracking**: Always know which documents and chunks were used for answers
- **📄 Export Capabilities**: Export results to Markdown format
- **🖥️ Multiple Interfaces**: Both CLI and Streamlit web UI
- **⚡ Local Processing**: All processing happens locally, no external API calls required

## 🚀 Quick Start for CodeMate Submission

### ⚡ Fast Demo (2 minutes)
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the complete demo
python run_demo.py

# 3. Launch web interface
streamlit run app.py
```

### 📋 Step-by-Step Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the demo**:
   ```bash
   python run_demo.py
   ```

3. **Launch web interface**:
   ```bash
   streamlit run app.py
   ```

4. **Use CLI commands**:
   ```bash
   python cli.py --help
   ```

### Basic Usage

#### Web Interface (Recommended)

1. Open your browser to `http://localhost:8501`
2. Configure your models in the sidebar
3. Click "Load Assistant" to initialize
4. Upload your PDF/TXT documents
5. Ask questions and get AI-powered answers!

#### Command Line Interface

1. **Ingest documents**:
   ```bash
   python cli.py ingest path/to/your/documents --output data/faiss_index
   ```

2. **Ask a question**:
   ```bash
   python cli.py ask "What are the main findings?" --index data/faiss_index
   ```

3. **Generate a summary**:
   ```bash
   python cli.py summarize --index data/faiss_index
   ```

4. **View document info**:
   ```bash
   python cli.py info --index data/faiss_index
   ```

## 📁 Project Structure

```
deep_researcher/
├── __init__.py              # Package initialization
├── ingest.py                # Document processing and chunking
├── embeddings.py            # Embedding generation and FAISS storage
├── query.py                 # Query processing and answer generation
├── cli.py                   # Command line interface
└── app.py                   # Streamlit web interface

app.py                       # Main Streamlit entry point
cli.py                       # Main CLI entry point
requirements.txt             # Python dependencies
README.md                    # This file
```

## 🔧 Configuration

### Model Options

**Embedding Models** (sentence-transformers):
- `all-MiniLM-L6-v2` (default) - Fast and efficient
- `all-mpnet-base-v2` - Higher quality, slower
- `paraphrase-MiniLM-L6-v2` - Good for paraphrasing

**Answer Generation Models** (T5):
- `google/flan-t5-small` (default) - Fast and efficient
- `google/flan-t5-base` - Better quality, slower
- `google/flan-t5-large` - Best quality, slowest

### Chunking Parameters

- **Chunk Size**: Number of tokens per chunk (default: 500)
- **Chunk Overlap**: Overlap between chunks in tokens (default: 50)

## 📖 Detailed Usage

### Document Ingestion

The system processes documents in the following steps:

1. **Text Extraction**: Extracts text from PDF and TXT files
2. **Text Cleaning**: Removes excessive whitespace and special characters
3. **Chunking**: Splits text into overlapping chunks of ~500 tokens
4. **Embedding Generation**: Creates vector embeddings using sentence-transformers
5. **Indexing**: Stores embeddings in FAISS vector database

### Query Processing

When you ask a question:

1. **Query Embedding**: Converts your question to a vector
2. **Similarity Search**: Finds the most relevant document chunks
3. **Context Formation**: Combines relevant chunks into context
4. **Answer Generation**: Uses T5 model to generate a natural language answer
5. **Source Tracking**: Records which chunks were used

### Supported File Types

- **PDF**: `.pdf` files (using PyPDF2)
- **Text**: `.txt` files (UTF-8 encoded)

## 🛠️ Advanced Features

### Multi-Document Summarization

Generate comprehensive summaries across all your documents:

```bash
python cli.py summarize --query "Summarize the main research findings" --top-k 15
```

### Export Results

Export query results and summaries to Markdown:

```bash
python cli.py ask "What is machine learning?" --output result.md
python cli.py summarize --output summary.md
```

### Index Management

- **View Statistics**: `python cli.py info`
- **Clear Index**: `python cli.py clear`
- **Rebuild Index**: `python cli.py rebuild`

## 🔍 CLI Commands Reference

| Command | Description | Example |
|---------|-------------|---------|
| `ingest` | Process and index documents | `python cli.py ingest docs/` |
| `ask` | Ask a question | `python cli.py ask "What is AI?"` |
| `summarize` | Generate document summary | `python cli.py summarize` |
| `search` | Search for relevant chunks | `python cli.py search "machine learning"` |
| `info` | Show document statistics | `python cli.py info` |
| `clear` | Clear the index | `python cli.py clear` |
| `rebuild` | Rebuild the index | `python cli.py rebuild` |

## 🌐 Web Interface Features

### Upload Documents Tab
- Drag and drop file upload
- Batch processing of multiple files
- Real-time processing status
- File type validation

### Ask Questions Tab
- Natural language query input
- Adjustable retrieval parameters
- Real-time answer generation
- Source citation display
- Export functionality

### Document Info Tab
- Index statistics overview
- Document metadata display
- Chunk count and token statistics
- File type and size information

### History Tab
- Query history tracking
- Previous answers and sources
- Confidence scores
- Export previous results

## ⚙️ Configuration Options

### Environment Variables

You can set these environment variables for customization:

```bash
export DEEP_RESEARCHER_INDEX_PATH="custom/path/to/index"
export DEEP_RESEARCHER_EMBEDDING_MODEL="all-mpnet-base-v2"
export DEEP_RESEARCHER_T5_MODEL="google/flan-t5-base"
```

### Custom Index Path

Specify a custom path for the FAISS index:

```bash
python cli.py ingest docs/ --output /path/to/custom/index
python cli.py ask "question" --index /path/to/custom/index
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Use smaller models (`flan-t5-small` instead of `flan-t5-large`)
   - Reduce chunk size
   - Process fewer documents at once

2. **Slow Processing**:
   - Use CPU-optimized models
   - Reduce chunk overlap
   - Process documents in smaller batches

3. **Import Errors**:
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version (3.10+ required)

4. **File Processing Errors**:
   - Ensure PDF files are not password-protected
   - Check file encoding for TXT files
   - Verify file permissions

### Performance Tips

- **For Large Documents**: Increase chunk size to reduce total chunks
- **For Better Quality**: Use larger models (`flan-t5-base` or `flan-t5-large`)
- **For Speed**: Use smaller models and reduce `top_k` parameter
- **For Memory**: Process documents in smaller batches

## 📊 System Requirements

- **Python**: 3.10 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB for models and dependencies
- **CPU**: Multi-core recommended for faster processing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [sentence-transformers](https://www.sbert.net/) for embedding models
- [FAISS](https://faiss.ai/) for vector similarity search
- [Hugging Face Transformers](https://huggingface.co/transformers/) for T5 models
- [Streamlit](https://streamlit.io/) for the web interface
- [PyPDF2](https://pypdf2.readthedocs.io/) for PDF processing

## 📞 Support

If you encounter any issues or have questions:

1. Check the troubleshooting section above
2. Review the CLI help: `python cli.py --help`
3. Open an issue on GitHub
4. Check the documentation for your specific use case

---

**Happy Researching! 🔍📚**
