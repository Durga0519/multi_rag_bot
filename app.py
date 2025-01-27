import streamlit as st
import os
import faiss
import pandas as pd
import json
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Cache the models so they are not downloaded each time
@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer('all-MiniLM-L6-v2')  # Sentence Transformer for embeddings

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn", revision="main")  # HuggingFace summarization pipeline

# Initialize models
st_model = load_sentence_transformer()  # Sentence Transformer for embeddings
summarizer = load_summarizer()  # HuggingFace summarization pipeline

# Initialize FAISS Index
dimension = 384  # Embedding size of all-MiniLM-L6-v2
index = faiss.IndexFlatL2(dimension)  # L2 distance index
doc_store = []  # List to hold document contents
doc_embeddings = []  # List to store document embeddings
metadata_store = []  # Store metadata like filenames


# Helper Functions for File Parsing
def parse_text(file):
    """Parse a plain text file."""
    return file.read().decode("utf-8")


def parse_csv(file):
    """Parse a CSV file and concatenate all rows into a single text."""
    df = pd.read_csv(file)
    return df.to_string(index=False)


def parse_json(file):
    """Parse a JSON file and convert it into a text string."""
    data = json.load(file)
    return json.dumps(data, indent=2)


# Query Parser (Agent 1)
def parse_query(query):
    """Extract keywords from the user query."""
    return query  # Simple pass-through for now


# Document Retriever (Agent 2)
def retrieve_documents(query, index, docs, k=5):
    """Retrieve top-k relevant documents using FAISS."""
    query_embedding = st_model.encode([query])  # Encode query
    st.write(f"Query embedding: {query_embedding}")  # Debug: Check the query embedding
    st.write(f"Using k={k} for search.")  # Debug: Print k value before search
    distances, indices = index.search(query_embedding, k)
    retrieved_docs = [docs[i] for i in indices[0]]
    return retrieved_docs, distances[0]


def rank_documents(query, doc_embeddings, docs):
    """Rank documents based on cosine similarity."""
    query_embedding = st_model.encode([query])  # This should be a 2D array
    doc_embeddings = np.vstack(doc_embeddings)  # Ensure doc_embeddings is 2D (n_documents, embedding_size)
    
    # Calculate cosine similarity (query_embedding should be 2D, doc_embeddings should be 2D)
    scores = cosine_similarity(query_embedding, doc_embeddings)[0]  # Get the similarity for the query
    ranked_indices = np.argsort(scores)[::-1]  # Sort the documents based on scores
    ranked_docs = [docs[i] for i in ranked_indices]  # Get the sorted documents
    ranked_scores = [scores[i] for i in ranked_indices]  # Get the sorted scores
    return ranked_docs, ranked_scores


# Response Generator (Agent 4)
def generate_response(top_docs):
    """Generate a response from the top documents."""
    combined_text = " ".join(top_docs)
    response = summarizer(combined_text, max_length=100, min_length=30, do_sample=False)
    return response[0]['summary_text']


# Streamlit UI
st.title("Multi-Agentic Retrieval-Augmented Generation System")
st.sidebar.title("System Options")
k = st.sidebar.slider("Number of Retrieved Documents", min_value=1, max_value=10, value=5)

uploaded_files = st.file_uploader(
    "Upload Text, CSV, or JSON Files", accept_multiple_files=True, type=["txt", "csv", "json"]
)
if uploaded_files:
    st.write("Uploading and indexing documents...")
    for file in uploaded_files:
        filename = file.name
        file_type = filename.split(".")[-1]
        
        # Parse file content based on type
        try:
            if file_type == "txt":
                content = parse_text(file)
            elif file_type == "csv":
                content = parse_csv(file)
            elif file_type == "json":
                content = parse_json(file)
            else:
                st.warning(f"Unsupported file type: {file_type}")
                continue

            # Add content to document store
            doc_store.append(content)
            metadata_store.append(filename)
            
            # Generate embeddings and index them
            embedding = st_model.encode([content])
            doc_embeddings.append(embedding)  # Store the embeddings
            index.add(embedding)  # Add embeddings to FAISS index

        except Exception as e:
            st.error(f"Error processing {filename}: {e}")
    
    st.success(f"Indexed {len(doc_store)} documents successfully!")

query = st.text_input("Enter your query:")
if query:
    # Ensure k is always positive
    k = max(k, 1)  # Prevent k from being zero or negative

    # Ensure the query is not empty
    if not query:
        st.warning("Please enter a valid query.")
    else:
        st.write("Processing your query...")

        # Agent Workflow
        parsed_query = parse_query(query)
        retrieved_docs, distances = retrieve_documents(parsed_query, index, doc_store, k=k)
        ranked_docs, ranked_scores = rank_documents(parsed_query, np.array(doc_embeddings), doc_store)

        # Display top-k retrieved documents
        st.subheader("Top Retrieved Documents:")
        for i, doc in enumerate(retrieved_docs):
            st.write(f"Document {i+1} (Score: {distances[i]:.4f}):")
            st.write(doc)
            st.write("---")

        # Optionally, display ranked documents and their similarity scores
        st.subheader("Ranked Documents by Similarity:")
        for i, (doc, score) in enumerate(zip(ranked_docs[:k], ranked_scores[:k])):
            st.write(f"Document {i+1} (Score: {score:.4f}):")
            st.write(doc)
            st.write("---")

        # Generate the response from the top documents
        response = generate_response(ranked_docs[:k])

        # Display the Generated Response only
        st.subheader("Generated Response:")
        st.write(response)
