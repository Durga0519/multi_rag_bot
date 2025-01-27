# Multi-Agentic Retrieval-Augmented Generation System

This project demonstrates a **Multi-Agentic Retrieval-Augmented Generation (RAG) System** implemented using Streamlit. The system is designed to handle text, CSV, and JSON files, retrieve relevant documents using FAISS, rank them by relevance, and generate a concise summary using a language model. It is suitable for large-scale document querying, even with storage-intensive datasets.

---

## Key Features

1. **Multi-Agent Workflow**:
    - **Query Parsing (Agent 1)**: Extracts keywords and important terms from user queries.
    - **Document Retrieval (Agent 2)**: Uses FAISS to retrieve documents based on vector similarity.
    - **Document Ranking (Agent 3)**: Ranks documents using cosine similarity.
    - **Response Generation (Agent 4)**: Synthesizes a concise response from top-ranked documents using a summarization model.

2. **Support for Multiple File Formats**:
    - Plain text (`.txt`)
    - Comma-Separated Values (`.csv`)
    - JSON files (`.json`)

3. **Efficient Storage & Indexing**:
    - FAISS for scalable and fast indexing of document embeddings.
    - Support for large datasets with minimal performance overhead.

4. **Interactive UI**:
    - Allows users to upload files, input queries, and view results interactively.
    - Displays retrieved documents, rankings, and the generated response.

---

## Prerequisites and Storage Requirements

### Storage Considerations

- The application requires **sufficient disk space** to store document embeddings and models.
- A minimum of **5 GB of free disk space** is recommended to account for the following:
    - Cached models (e.g., Sentence Transformer and Summarizer from HuggingFace).
    - FAISS index for embedding storage (varies with the number of documents).
    - Temporary storage for uploaded files and indexed data.
- For systems with limited storage, consider running the application on a cloud environment like AWS, GCP, or Azure, or ensure local storage capacity is expanded as needed.

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-repo-url.git
cd your-repo-directory
```

### 2. Create a Virtual Environment (Optional)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Required Libraries

Install the dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

### 5. Upload Files & Query the System

1. Upload `.txt`, `.csv`, or `.json` files via the UI.
2. Enter your query in the input box.
3. View retrieved documents, ranked results, and the generated summary.

---

## File Structure

```plaintext
project-directory/
|-- app.py                  # Main application file
|-- requirements.txt        # Python dependencies
|-- README.md               # Documentation
```

---

## Usage Workflow

### Upload Files

- Upload files in one of the supported formats.
- The system parses the files, splits content (if needed), and generates embeddings for indexing.

### Query Processing

- Enter a query in natural language.
- The system retrieves top-k documents, ranks them, and generates a concise response.

### Results

- View **Retrieved Documents**: Top documents relevant to the query.
- View **Ranked Documents**: Documents ordered by similarity.
- View **Generated Response**: A summarized answer to the query.

---

## Example Queries

1. Query: *"Explain the benefits of vector similarity for document retrieval."*
    - Retrieves technical articles on FAISS and vector databases.
    - Summarizes key points into a concise paragraph.

2. Query: *"How do I process JSON files in Python?"*
    - Fetches relevant FAQs or tutorials from uploaded files.

---

## Troubleshooting

### Common Issues

1. **Insufficient Disk Space**:
    - Ensure at least 5 GB of free storage before running the application.
    - Delete unnecessary files or use cloud storage solutions for large datasets.

2. **Model Download Issues**:
    - The first run downloads large pre-trained models.
    - Ensure a stable internet connection and sufficient storage.

3. **Memory Errors**:
    - For large datasets, consider increasing your system's RAM or reducing the number of retrieved documents (adjustable via the sidebar).

4. **Slow Query Processing**:
    - Ensure embeddings are indexed correctly in FAISS.
    - Reduce the `k` value for faster retrieval.

---

## System Diagram

### Agent Interaction Flow

```plaintext
+-----------------+       +-----------------+       +-----------------+
|                 |       |                 |       |                 |
| Query Parser    +------>+ Document        +------>+ Document Ranker |
| (Agent 1)       |       | Retriever       |       | (Agent 3)       |
|                 |       | (Agent 2)       |       |                 |
+-----------------+       +-----------------+       +-----------------+
         |                         |                          |
         |                         |                          |
         +-------------------------+                          |
                  |                                           |
                  +-------------------------------------------+
                                      |
                                      v
                           +-----------------------+
                           | Response Generator    |
                           | (Agent 4)             |
                           +-----------------------+
```

---

## License

This project is licensed under the MIT License. Feel free to use and modify it for your own purposes.

---



