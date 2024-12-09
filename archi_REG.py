import json
from sentence_transformers import SentenceTransformer
from weaviate import Client
import streamlit as st
from dotenv import load_dotenv
import os
import logging
from multiprocessing import Pool, cpu_count
import pickle
from openai import OpenAI  # For LLM interactions
import hashlib

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")


def load_data_from_uploaded_file(uploaded_file):
    """
    Load and parse amazon_data from an uploaded JSONL file.

    Parameters:
    -----------
    uploaded_file : file-like object
        A file object containing JSONL formatted amazon_data

    Returns:
    --------
    list
        A list of parsed JSON objects from the file
    """
    data = []
    for line in uploaded_file:
        data.append(json.loads(line))
    return data


def simple_text_splitter(text, chunk_size=512, overlap=128):
    """
    Split a long text into smaller chunks with overlapping.

    Parameters:
    -----------
    text : str
        The input text to be split
    chunk_size : int, optional
        Maximum size of each text chunk (default: 512)
    overlap : int, optional
        Number of characters to overlap between chunks (default: 128)

    Returns:
    --------
    list
        A list of text chunks
    """
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]
    return chunks


def process_entry(item):
    """
    Process a single amazon_data entry by combining and splitting its text components.

    Parameters:
    -----------
    item : dict
        A dictionary containing product information with keys like 'title', 'features', etc.

    Returns:
    --------
    list
        A list of text chunks derived from the product information
    """
    title = item.get('title', '')
    features = " ".join(item.get('features', []))
    categories = " ".join(item.get('categories', []))
    details = " ".join([f"{k}: {v}" for k, v in item.get('details', {}).items()])
    combined_text = f"{title} {features} {categories} {details}".strip()
    if combined_text:
        return simple_text_splitter(combined_text)
    return []


def preprocess_data_parallel(data):
    """
    Process multiple amazon_data entries in parallel using multiprocessing.

    Parameters:
    -----------
    amazon_data : list
        A list of amazon_data entries to be processed

    Returns:
    --------
    list
        A flattened list of text chunks from all processed entries
    """
    with Pool(cpu_count()) as pool:
        results = pool.map(process_entry, data)
    # Flatten the list of lists
    processed_data = [chunk for sublist in results for chunk in sublist]
    return processed_data


def generate_embeddings_in_batches(data, model, batch_size=32):
    """
    Generate vector embeddings for text amazon_data in batches.

    Parameters:
    -----------
    amazon_data : list
        A list of text chunks to be embedded
    model : SentenceTransformer
        The embedding model to use
    batch_size : int, optional
        Number of texts to embed in each batch (default: 32)

    Returns:
    --------
    list
        A list of vector embeddings corresponding to the input texts
    """
    embeddings = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        batch_embeddings = model.encode(batch)
        embeddings.extend(batch_embeddings)
    return embeddings


def generate_file_hash(uploaded_file):
    """
    Generate a unique hash for the uploaded file to use as a cache key.

    Parameters:
    -----------
    uploaded_file : file-like object
        The uploaded file to hash

    Returns:
    --------
    str
        MD5 hash of the file content
    """
    # Reset file pointer to the beginning
    uploaded_file.seek(0)
    # Read the file content
    file_content = uploaded_file.read()
    # Generate hash
    file_hash = hashlib.md5(file_content).hexdigest()
    # Reset file pointer for future use
    uploaded_file.seek(0)
    return file_hash


@st.cache_data
def process_and_embed_data(uploaded_file):
    """
    Comprehensive amazon_data processing function that loads, preprocesses,
    and generates embeddings for the uploaded file.

    Parameters:
    -----------
    uploaded_file : file-like object
        The JSONL file to process

    Returns:
    --------
    tuple
        A tuple containing:
        - processed_data: List of text chunks
        - embeddings: List of vector embeddings
        - model: The sentence transformer model used for embedding
    """
    # Load amazon_data
    data = load_data_from_uploaded_file(uploaded_file)

    # Preprocess amazon_data
    processed_data = preprocess_data_parallel(data)

    # Load model and generate embeddings
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
    embeddings = generate_embeddings_in_batches(processed_data, model)

    return processed_data, embeddings, model


@st.cache_resource
def initialize_weaviate_client():
    """
    Initialize and return a Weaviate client with connection verification.

    Returns:
    --------
    Client
        A configured Weaviate client

    Raises:
    -------
    ConnectionError
        If unable to connect to Weaviate server
    """
    weaviate_url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
    client = Client(weaviate_url)

    # Verify connection
    try:
        # Check if server is ready
        if not client.is_ready():
            raise ConnectionError("Weaviate server is not ready")

        # Try to get meta info to verify connection
        client.get_meta()
        logging.info(f"Successfully connected to Weaviate at {weaviate_url}")
        return client
    except Exception as e:
        error_msg = f"Failed to connect to Weaviate at {weaviate_url}: {str(e)}"
        logging.error(error_msg)
        raise ConnectionError(error_msg)


def setup_weaviate_schema(client):
    """
    Create or verify the Weaviate schema for storing product descriptions.

    Parameters:
    -----------
    client : Client
        The Weaviate client to use for schema setup

    Returns:
    --------
    str
        The name of the created/existing class
    """
    class_name = "ProductDescription"
    schema = {
        "classes": [
            {
                "class": class_name,
                "description": "A collection of product descriptions with embeddings",
                "properties": [
                    {"name": "text", "dataType": ["text"], "description": "The product description text"},
                    {"name": "embedding", "dataType": ["number[]"],
                     "description": "The vector embedding of the description"}
                ],
                "vectorizer": "none",
            }
        ]
    }

    if not client.schema.exists(class_name):
        client.schema.create(schema)
        logging.info(f"Weaviate schema for {class_name} created.")
    else:
        logging.info(f"Weaviate schema for {class_name} already exists.")
    return class_name


def add_data_to_weaviate(client, class_name, embeddings, texts, batch_size=100):
    """
    Add processed text and embeddings to Weaviate in batches.

    Parameters:
    -----------
    client : Client
        The Weaviate client
    class_name : str
        The name of the Weaviate class to store amazon_data
    embeddings : list
        List of vector embeddings
    texts : list
        List of text chunks corresponding to embeddings
    batch_size : int, optional
        Number of objects to insert in each batch (default: 100)
    """
    # Clear existing amazon_data in the class
    try:
        client.schema.delete_class(class_name)
        setup_weaviate_schema(client)
    except Exception as e:
        logging.warning(f"Error clearing existing amazon_data: {e}")

    with client.batch as batch:
        for i in range(0, len(texts), batch_size):
            for text, embedding in zip(texts[i:i + batch_size], embeddings[i:i + batch_size]):
                batch.add_data_object(
                    data_object={"text": text},
                    class_name=class_name,
                    vector=embedding
                )
            try:
                batch.create_objects()
                logging.info(f"Batch {i // batch_size + 1} inserted successfully.")
            except Exception as e:
                logging.error(f"Error inserting batch {i // batch_size + 1}: {e}")


def query_weaviate(client, class_name, query_embedding, limit=5):
    """
    Query Weaviate for similar documents.

    Parameters:
    -----------
    client : Client
        The Weaviate client
    class_name : str
        The name of the Weaviate class to query
    query_embedding : list
        The vector embedding of the query
    limit : int, optional
        Number of results to return (default: 5)

    Returns:
    --------
    list
        A list of retrieved text documents
    """
    try:
        results = (
            client.query
            .get(class_name, ["text"])
            .with_near_vector({"vector": query_embedding})
            .with_limit(limit)
            .do()
        )

        # Handle the response structure correctly
        if results and "data" in results:
            # Access the correct path in the response
            class_results = results["data"]["Get"][class_name]
            return [item["text"] for item in class_results]
        else:
            logging.warning("No results found or unexpected response structure")
            return []

    except Exception as e:
        logging.error(f"Error querying Weaviate: {e}")
        return []


def ask_llm(documents, question, temperature=0.7, top_p=0.9):
    """
    Query a Large Language Model (LLM) with retrieved documents and a user question.

    Parameters:
    -----------
    documents : str
        Concatenated relevant documents retrieved from vector search
    question : str
        The user's input question
    temperature : float, optional
        Controls randomness in LLM response (default: 0.7)
    top_p : float, optional
        Controls diversity of LLM response (default: 0.9)

    Returns:
    --------
    str
        The LLM's response to the question
    """
    prompt = f"""
    You are an assistant that answers questions using only the provided information.
    If no relevant information is available, respond with: "I don't know based on the provided information."

    Relevant documents:
    {documents}

    Question: {question}
    Provide a concise and accurate response.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.error("OpenAI API key is missing from the env file.")
        raise ValueError("OpenAI API key is missing.")

    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p
    )

    return response.choices[0].message.content


def main():
    """
    Main Streamlit application function with updated query handling.
    """
    st.title("RAG System for Product Descriptions")
    st.write("Upload a `.jsonl` file to get started!")

    # File uploader
    uploaded_file = st.file_uploader("Upload your .jsonl file", type=["jsonl"])
    if uploaded_file is None:
        st.warning("Please upload a .jsonl file to proceed.")
        return

    # Process data and generate embeddings (cached)
    with st.spinner("Processing data..."):
        try:
            processed_data, embeddings, model = process_and_embed_data(uploaded_file)
            st.success("Data processed successfully!")
        except Exception as e:
            st.error(f"Error processing data: {e}")
            return

    # Initialize Weaviate client
    try:
        client = initialize_weaviate_client()
    except Exception as e:
        st.error(f"Error connecting to Weaviate: {e}")
        return

    # Set up schema and add data
    class_name = setup_weaviate_schema(client)
    add_data_to_weaviate(client, class_name, embeddings, processed_data)

    st.write("Ask questions about the data!")

    # User input for question
    question = st.text_input("Enter your question:")

    # Temperature and top_p sliders
    temperature = st.slider("Set the temperature (creativity):", 0.0, 1.0, 0.7, 0.1)
    top_p = st.slider("Set the top-p (nucleus sampling):", 0.0, 1.0, 0.9, 0.1)

    if question:
        # Encode the question
        query_embedding = model.encode(question).tolist()

        # Retrieve documents using the new query function
        documents = query_weaviate(client, class_name, query_embedding)

        if documents:
            # Call LLM with retrieved documents
            response = ask_llm("\n".join(documents), question, temperature, top_p)
            st.write("Response:")
            st.write(response)
        else:
            st.warning("No relevant documents found for your question.")


# Run the Streamlit app
if __name__ == "__main__":
    main()