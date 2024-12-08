# Import required libraries
# Core libraries
import streamlit as st
import json
import os

# NLP and ML libraries
import spacy
from sklearn.cluster import KMeans, DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer

# Deep Learning libraries
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Data processing and visualization
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Initialize spaCy model with error handling
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.warning("Downloading spaCy model 'en_core_web_sm'...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


# Text Preprocessing Functions
def preprocess_text_spacy(text, nlp):
    """
    Preprocess text using spaCy with enhanced filtering.

    Args:
        text (str): Input text to preprocess
        nlp: spaCy language model

    Returns:
        list: Filtered and lemmatized tokens
    """
    if not isinstance(text, str):
        text = str(text)
    doc = nlp(text.lower())
    # Filter tokens based on multiple criteria
    tokens = [
        token.lemma_ for token in doc
        if (not token.is_stop and  # Remove stop words
            not token.is_punct and  # Remove punctuation
            not token.like_num and  # Remove numbers
            not token.like_url and  # Remove URLs
            len(token.text.strip()) > 1)  # Remove single characters
    ]
    return tokens


# Data Loading Functions
@st.cache_data
def load_jsonl(uploaded_files):
    """
    Load and merge JSONL files (reviews and metadata).

    Args:
        uploaded_files: List of uploaded JSONL files

    Returns:
        list: Merged review amazon_data with metadata
    """
    reviews = []
    metadata = {}

    # Process each uploaded file
    for file in uploaded_files:
        if file.name == "meta.jsonl":
            for line in file:
                item = json.loads(line.decode("utf-8").strip())
                metadata[item.get("review_id")] = item
        elif file.name == "reviews.jsonl":
            for line in file:
                reviews.append(json.loads(line.decode("utf-8").strip()))

    # Merge metadata with reviews if available
    if metadata:
        for review in reviews:
            review_meta = metadata.get(review.get("review_id"), {})
            review.update(review_meta)

    return reviews


@st.cache_data
def preprocess_reviews(reviews):
    """
    Preprocess all reviews using spaCy.

    Args:
        reviews (list): List of review dictionaries

    Returns:
        list: Processed reviews with lemmatized tokens
    """
    processed_reviews = []
    for review in reviews:
        processed_review = {
            "title": preprocess_text_spacy(review.get("title", ""), nlp),
            "text": preprocess_text_spacy(review.get("text", ""), nlp),
            "rating": review.get("rating")
        }
        processed_reviews.append(processed_review)
    return processed_reviews


# Feature Extraction Functions
def extract_bigrams(texts):
    """
    Extract significant bigrams from texts.

    Args:
        texts: List of texts to process

    Returns:
        list: Top 10 most common bigrams with their counts
    """
    bigram_counts = Counter()
    for text in texts:
        doc = nlp(" ".join(text) if isinstance(text, list) else text)
        # Create bigrams using index-based iteration
        bigrams = []
        for i in range(len(doc) - 1):
            token = doc[i]
            next_token = doc[i + 1]
            if not (token.is_stop or next_token.is_stop):
                bigrams.append(f"{token.text}_{next_token.text}")
        bigram_counts.update(bigrams)
    return bigram_counts.most_common(10)


@st.cache_data
def generate_embeddings(corpus, use_tfidf):
    """
    Generate document embeddings using either TF-IDF or SentenceTransformer.

    Args:
        corpus: List of documents
        use_tfidf (bool): Whether to use TF-IDF instead of SentenceTransformer

    Returns:
        numpy.array: Document embeddings
    """
    if use_tfidf:
        vectorizer = TfidfVectorizer(max_features=1000)
        embeddings = vectorizer.fit_transform(corpus).toarray()
    else:
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        embeddings = model.encode(corpus)
    return embeddings


# Clustering Functions
@st.cache_data
def perform_clustering(embeddings, method, num_clusters=None, eps=0.5, min_samples=5):
    """
    Perform document clustering using either KMeans or DBSCAN.

    Args:
        embeddings: Document embeddings
        method (str): Clustering method ('kmeans' or 'dbscan')
        num_clusters (int, optional): Number of clusters for KMeans
        eps (float): DBSCAN epsilon parameter
        min_samples (int): DBSCAN minimum samples parameter

    Returns:
        tuple: (cluster assignments, clustering model)
    """
    if method == "kmeans":
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        return clusters.tolist(), kmeans
    elif method == "dbscan":
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        clusters = dbscan.fit_predict(embeddings)
        return clusters.tolist(), None


@st.cache_data
def calculate_silhouette(embeddings, clusters):
    """
    Calculate silhouette score for clustering evaluation.

    Args:
        embeddings: Document embeddings
        clusters: Cluster assignments

    Returns:
        float: Silhouette score
    """
    unique_clusters = set(clusters)
    if len(unique_clusters) > 1 and -1 not in unique_clusters:
        return silhouette_score(embeddings, clusters)
    return 0


# Visualization Preparation
@st.cache_data
def reduce_dimensions(embeddings):
    """
    Reduce embedding dimensions for visualization using PCA.

    Args:
        embeddings: High-dimensional embeddings

    Returns:
        numpy.array: 2D embeddings for visualization
    """
    pca = PCA(n_components=2)
    return pca.fit_transform(embeddings)


# Cluster Analysis
def analyze_clusters(clusters, corpus, processed_reviews):
    """
    Analyze clusters for patterns and extract key information.

    Args:
        clusters: Cluster assignments
        corpus: Original text corpus
        processed_reviews: Preprocessed reviews

    Returns:
        dict: Cluster analysis results
    """
    clustered_docs = defaultdict(list)
    clustered_processed = defaultdict(list)

    # Group documents by cluster
    for idx, cluster_id in enumerate(clusters):
        clustered_docs[int(cluster_id)].append(corpus[idx])
        clustered_processed[int(cluster_id)].append(processed_reviews[idx]["text"])

    cluster_analysis = {}
    for cluster_id in clustered_docs.keys():
        # Count word frequencies
        word_counter = Counter()
        for doc in clustered_processed[cluster_id]:
            word_counter.update(doc)

        # Extract bigrams
        bigrams = extract_bigrams(clustered_processed[cluster_id])

        # Identify named entities
        named_entities = Counter()
        for doc in clustered_docs[cluster_id]:
            spacy_doc = nlp(doc)
            named_entities.update([(ent.text, ent.label_) for ent in spacy_doc.ents])

        cluster_analysis[int(cluster_id)] = {
            "frequent_words": word_counter.most_common(10),
            "bigrams": bigrams,
            "named_entities": named_entities.most_common(5),
            "document_count": len(clustered_docs[cluster_id])
        }

    return cluster_analysis


# Sentiment Analysis Components
class ReviewDataset(Dataset):
    """Custom dataset for sentiment analysis."""

    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0)
        }


@st.cache_resource
def load_sentiment_model():
    """
    Load pretrained sentiment analysis model and tokenizer.

    Returns:
        tuple: (model, tokenizer)
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        "nlptown/bert-base-multilingual-uncased-sentiment"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "nlptown/bert-base-multilingual-uncased-sentiment"
    )
    return model, tokenizer


def predict_sentiments(dataloader, model, device):
    """
    Predict sentiment scores for reviews.

    Args:
        dataloader: DataLoader containing reviews
        model: Sentiment analysis model
        device: Computing device (CPU/GPU)

    Returns:
        list: Predicted sentiment scores
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy() + 1)

    return predictions


# Streamlit UI
st.title("Enhanced NLP Analysis Workflow")
st.sidebar.header("Upload Datasets")
uploaded_files = st.sidebar.file_uploader(
    "Choose JSONL files (reviews.jsonl and meta.jsonl)",
    type="jsonl",
    accept_multiple_files=True
)

if uploaded_files:
    reviews = load_jsonl(uploaded_files)
    st.success(f"Loaded {len(reviews)} reviews successfully.")

    tabs = st.tabs(["Preprocessing", "Clustering", "NER", "Sentiment Analysis"])

    # Initialize session state
    for key in ["preprocessed_reviews", "clustering_results", "ner_results", "sentiment_results"]:
        if key not in st.session_state:
            st.session_state[key] = None

    # Preprocessing Tab
    with tabs[0]:
        st.header("Step 1: Text Preprocessing")
        if st.button("Execute Preprocessing"):
            with st.spinner("Preprocessing reviews..."):
                st.session_state.preprocessed_reviews = preprocess_reviews(reviews)

                # Display sample
                if st.session_state.preprocessed_reviews:
                    st.write("Sample of Processed Reviews:")
                    df = pd.DataFrame([
                        {
                            "Original": r.get("text", ""),
                            "Processed": " ".join(p["text"])
                        }
                        for r, p in zip(
                            reviews[:5],
                            st.session_state.preprocessed_reviews[:5]
                        )
                    ])
                    st.dataframe(df)

    # Clustering Tab
    with tabs[1]:
        st.header("Step 2: Document Clustering")
        col1, col2 = st.columns(2)

        with col1:
            embedding_method = st.selectbox(
                "Embedding Method",
                ["SentenceTransformer", "TF-IDF"]
            )

        with col2:
            clustering_method = st.selectbox(
                "Clustering Method",
                ["KMeans", "DBSCAN"]
            )

        if clustering_method == "KMeans":
            num_clusters = st.slider("Number of Clusters", 2, 10, 5)
            clustering_params = {"num_clusters": num_clusters}
        else:
            eps = st.slider("DBSCAN eps", 0.1, 1.0, 0.5)
            min_samples = st.slider("DBSCAN min_samples", 2, 10, 5)
            clustering_params = {"eps": eps, "min_samples": min_samples}

        if st.button("Execute Clustering"):
            if st.session_state.preprocessed_reviews:
                with st.spinner("Generating embeddings..."):
                    corpus = [
                        " ".join(doc["text"])
                        for doc in st.session_state.preprocessed_reviews
                    ]
                    embeddings = generate_embeddings(
                        corpus,
                        embedding_method == "TF-IDF"
                    )
                    reduced_embeddings = reduce_dimensions(embeddings)

                with st.spinner("Performing clustering..."):
                    clusters, model = perform_clustering(
                        embeddings,
                        clustering_method.lower(),
                        **clustering_params
                    )

                    silhouette = calculate_silhouette(
                        np.array(embeddings),
                        clusters
                    )

                    cluster_analysis = analyze_clusters(
                        clusters,
                        corpus,
                        st.session_state.preprocessed_reviews
                    )

                st.session_state.clustering_results = {
                    "clusters": clusters,
                    "reduced_embeddings": reduced_embeddings,
                    "analysis": cluster_analysis,
                    "silhouette": silhouette
                }
            else:
                st.warning("Please execute preprocessing first.")

        if st.session_state.clustering_results:
            results = st.session_state.clustering_results

            # Display metrics
            st.metric("Silhouette Score", f"{results['silhouette']:.3f}")

            # Visualizations
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Scatter plot
            reduced_embeddings = np.array(results["reduced_embeddings"])
            clusters = np.array(results["clusters"])

            scatter = ax1.scatter(
                reduced_embeddings[:, 0],
                reduced_embeddings[:, 1],
                c=clusters,
                cmap='viridis'
            )
            ax1.set_title("Document Clusters (PCA)")
            ax1.set_xlabel("Component 1")
            ax1.set_ylabel("Component 2")
            plt.colorbar(scatter, ax=ax1)

            # Cluster sizes
            cluster_sizes = Counter(clusters)
            ax2.bar(cluster_sizes.keys(), cluster_sizes.values())
            ax2.set_title("Cluster Sizes")
            ax2.set_xlabel("Cluster ID")
            ax2.set_ylabel("Number of Documents")

            st.pyplot(fig)

            # Detailed cluster analysis
            st.write("### Cluster Analysis")
            for cluster_id, analysis in results["analysis"].items():
                with st.expander(
                        f"Cluster {cluster_id} ({analysis['document_count']} documents)"
                ):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.write("Most Frequent Words")
                        st.write(pd.DataFrame(
                            analysis["frequent_words"],
                            columns=["Word", "Count"]
                        ))

                    with col2:
                        st.write("Top Bigrams")
                        st.write(pd.DataFrame(
                            analysis["bigrams"],
                            columns=["Bigram", "Count"]
                        ))

                    with col3:
                        st.write("Named Entities")
                        st.write(pd.DataFrame(
                            analysis["named_entities"],
                            columns=["Entity (Type)", "Count"]
                        ))

    # NER Tab
    with tabs[2]:
        st.header("Step 3: Named Entity Recognition")
        ner_model = st.selectbox("NER Model", ["spaCy", "Transformers"])

        if st.button("Execute NER"):
            if st.session_state.preprocessed_reviews:
                with st.spinner("Performing Named Entity Recognition..."):
                    corpus = [
                        " ".join(doc["text"])
                        for doc in st.session_state.preprocessed_reviews
                    ]

                    if ner_model == "spaCy":
                        st.session_state.ner_results = [
                            {
                                "text": doc,
                                "entities": [
                                    {
                                        "text": ent.text,
                                        "label": ent.label_,
                                        "start": ent.start_char,
                                        "end": ent.end_char
                                    }
                                    for ent in nlp(doc).ents
                                ]
                            }
                            for doc in corpus
                        ]
                    else:
                        # Add missing parts for NER and Sentiment Analysis tabs

                        # NER Tab (continuation)
                        ner_pipeline = pipeline(
                            "ner",
                            model="dbmdz/bert-large-cased-finetuned-conll03-english"
                        )
                        st.session_state.ner_results = []
                        for doc in corpus:
                            entities = ner_pipeline(doc)
                            st.session_state.ner_results.append({
                                "text": doc,
                                "entities": entities
                            })

                        # Display NER results
                    if st.session_state.ner_results:
                        for idx, result in enumerate(st.session_state.ner_results[:5]):  # Show first 5 for brevity
                            with st.expander(f"Document {idx + 1}"):
                                st.write("Text:", result["text"][:200] + "...")  # Show truncated text
                                st.write("Entities:")
                                if result["entities"]:
                                    entities_df = pd.DataFrame(result["entities"])
                                    st.dataframe(entities_df)
                                else:
                                    st.write("No entities found")
                    else:
                        st.warning("Please execute preprocessing first.")

                        # Sentiment Analysis Tab
                    with tabs[3]:
                        st.header("Step 4: Sentiment Analysis")

                        if st.button("Execute Sentiment Analysis"):
                            if st.session_state.preprocessed_reviews:
                                with st.spinner("Performing sentiment analysis..."):
                                    # Load model and tokenizer
                                    sentiment_model, sentiment_tokenizer = load_sentiment_model()

                                    # Prepare amazon_data
                                    texts = [" ".join(review["text"]) for review in
                                             st.session_state.preprocessed_reviews]
                                    true_ratings = [review["rating"] for review in reviews if "rating" in review]

                                    # Set up device
                                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                                    sentiment_model.to(device)

                                    # Create dataset and dataloader
                                    dataset = ReviewDataset(texts, sentiment_tokenizer)
                                    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

                                    # Get predictions
                                    predicted_ratings = predict_sentiments(dataloader, sentiment_model, device)

                                    # Calculate correlation if true ratings exist
                                    correlation = None
                                    if true_ratings:
                                        correlation, p_value = pearsonr(true_ratings, predicted_ratings)

                                    st.session_state.sentiment_results = {
                                        "true_ratings": true_ratings,
                                        "predicted_ratings": predicted_ratings,
                                        "correlation": correlation
                                    }

                                    # Display results
                                    if correlation:
                                        st.metric("Pearson Correlation", f"{correlation:.3f}")

                                    # Plotting
                                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

                                    # Distribution of predicted ratings
                                    sns.histplot(predicted_ratings, bins=5, ax=ax1)
                                    ax1.set_title("Distribution of Predicted Ratings")
                                    ax1.set_xlabel("Rating")
                                    ax1.set_ylabel("Count")

                                    # True vs Predicted ratings if available
                                    if true_ratings:
                                        ax2.scatter(true_ratings, predicted_ratings, alpha=0.5)
                                        ax2.set_title("True vs Predicted Ratings")
                                        ax2.set_xlabel("True Rating")
                                        ax2.set_ylabel("Predicted Rating")
                                        # Add perfect prediction line
                                        min_val = min(min(true_ratings), min(predicted_ratings))
                                        max_val = max(max(true_ratings), max(predicted_ratings))
                                        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)

                                    st.pyplot(fig)

                                    # Show detailed results
                                    results_df = pd.DataFrame({
                                        'Text': texts[:10],  # Show first 10 for brevity
                                        'Predicted Rating': predicted_ratings[:10]
                                    })
                                    if true_ratings:
                                        results_df['True Rating'] = true_ratings[:10]

                                    st.write("Sample Results:")
                                    st.dataframe(results_df)
                            else:
                                st.warning("Please execute preprocessing first.")