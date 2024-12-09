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
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Data processing and visualization
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import random

# Initialize spaCy model with error handling
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.warning("Downloading spaCy model 'en_core_web_sm'...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


# Core Functions
def preprocess_text_spacy(text, nlp):
    """Preprocess text using spaCy with enhanced filtering."""
    if not isinstance(text, str):
        text = str(text)
    doc = nlp(text.lower())
    tokens = [
        token.lemma_ for token in doc
        if (not token.is_stop and
            not token.is_punct and
            not token.like_num and
            not token.like_url and
            len(token.text.strip()) > 1)
    ]
    return tokens


@st.cache_data
def load_jsonl(uploaded_files):
    """Load and merge JSONL files."""
    reviews = []
    metadata = {}

    for file in uploaded_files:
        if file.name == "meta.jsonl":
            for line in file:
                item = json.loads(line.decode("utf-8").strip())
                metadata[item.get("review_id")] = item
        elif file.name == "reviews.jsonl":
            for line in file:
                reviews.append(json.loads(line.decode("utf-8").strip()))

    if metadata:
        for review in reviews:
            review_meta = metadata.get(review.get("review_id"), {})
            review.update(review_meta)

    return reviews


@st.cache_data
def preprocess_reviews(reviews):
    """Preprocess all reviews using spaCy."""
    processed_reviews = []
    for review in reviews:
        processed_review = {
            "title": preprocess_text_spacy(review.get("title", ""), nlp),
            "text": preprocess_text_spacy(review.get("text", ""), nlp),
            "rating": review.get("rating")
        }
        processed_reviews.append(processed_review)
    return processed_reviews


def save_preprocessed_data(processed_reviews, filename="preprocessed_reviews", format="json"):
    """Save preprocessed reviews to file."""
    serializable_reviews = []
    for review in processed_reviews:
        serializable_review = {
            "title": " ".join(review["title"]),
            "text": " ".join(review["text"]),
            "rating": review["rating"]
        }
        serializable_reviews.append(serializable_review)

    full_filename = f"{filename}.{format}"

    if format == "json":
        with open(full_filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_reviews, f, ensure_ascii=False, indent=2)
    else:  # CSV format
        df = pd.DataFrame(serializable_reviews)
        df.to_csv(full_filename, index=False, encoding='utf-8')

    return full_filename


# Analysis Functions
def extract_keywords_tfidf(texts_by_cluster, n_keywords=10):
    """Extract keywords using class-based TF-IDF."""
    vectorizer = TfidfVectorizer(max_features=1000)

    all_texts = []
    cluster_doc_indices = {}
    current_idx = 0

    for cluster_id, texts in texts_by_cluster.items():
        cluster_doc_indices[cluster_id] = list(range(current_idx, current_idx + len(texts)))
        all_texts.extend(texts)
        current_idx += len(texts)

    tfidf_matrix = vectorizer.fit_transform(all_texts)
    feature_names = vectorizer.get_feature_names_out()

    keywords_by_cluster = {}
    for cluster_id, doc_indices in cluster_doc_indices.items():
        if doc_indices:
            cluster_tfidf = tfidf_matrix[doc_indices].mean(axis=0).A1
            top_indices = cluster_tfidf.argsort()[-n_keywords:][::-1]
            keywords_by_cluster[cluster_id] = [
                (feature_names[idx], cluster_tfidf[idx])
                for idx in top_indices
            ]

    return keywords_by_cluster


@st.cache_data
def generate_embeddings(corpus, use_tfidf):
    """Generate document embeddings."""
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
    """Perform document clustering."""
    if method == "kmeans":
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        return clusters.tolist(), kmeans
    else:  # dbscan
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        clusters = dbscan.fit_predict(embeddings)
        return clusters.tolist(), None
def analyze_cluster_topics(cluster_texts):
    """
    Analyze the topics and themes of a cluster.
    """
    word_freq = Counter()
    for text in cluster_texts:
        doc = nlp(text)
        words = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        word_freq.update(words)

    return word_freq.most_common(10)


@st.cache_data
def calculate_silhouette(embeddings, clusters):
    """Calculate silhouette score."""
    unique_clusters = set(clusters)
    if len(unique_clusters) > 1 and -1 not in unique_clusters:
        return silhouette_score(embeddings, clusters)
    return 0


@st.cache_data
def reduce_dimensions(embeddings):
    """Reduce dimensions for visualization."""
    pca = PCA(n_components=2)
    return pca.fit_transform(embeddings)


def analyze_clusters(clusters, corpus, processed_reviews):
    """Analyze clusters for patterns and extract key information."""
    clustered_docs = defaultdict(list)
    clustered_processed = defaultdict(list)

    for idx, cluster_id in enumerate(clusters):
        clustered_docs[int(cluster_id)].append(corpus[idx])
        clustered_processed[int(cluster_id)].append(processed_reviews[idx]["text"])

    cluster_analysis = {}
    for cluster_id in clustered_docs.keys():
        # Word frequencies
        word_counter = Counter()
        for doc in clustered_processed[cluster_id]:
            word_counter.update(doc)

        # Extract bigrams
        bigrams = extract_bigrams(clustered_processed[cluster_id])

        # Named entities
        named_entities = Counter()
        for doc in clustered_docs[cluster_id]:
            doc = nlp(doc)
            named_entities.update([(ent.text, ent.label_) for ent in doc.ents])

        cluster_analysis[int(cluster_id)] = {
            "frequent_words": word_counter.most_common(10),
            "bigrams": bigrams,  # Added back
            "named_entities": named_entities.most_common(5),
            "document_count": len(clustered_docs[cluster_id])
        }

    return cluster_analysis

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
    """Load pretrained sentiment analysis model."""
    model = AutoModelForSequenceClassification.from_pretrained(
        "nlptown/bert-base-multilingual-uncased-sentiment"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "nlptown/bert-base-multilingual-uncased-sentiment"
    )
    return model, tokenizer


def predict_sentiments(dataloader, model, device):
    """Predict sentiment scores."""
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy() + 1)
    return predictions


def calculate_metrics(true_ratings, predicted_ratings):
    """Calculate evaluation metrics."""
    true_array = np.array(true_ratings)
    pred_array = np.array(predicted_ratings)

    correlation, p_value = pearsonr(true_ratings, predicted_ratings)

    return {
        'pearson_correlation': correlation,
        'pearson_p_value': p_value,
        'exact_precision': np.mean(true_array == pred_array),
        'within_one_precision': np.mean(np.abs(true_array - pred_array) <= 1),
        'mae': np.mean(np.abs(true_array - pred_array)),
        'rmse': np.sqrt(np.mean((true_array - pred_array) ** 2))
    }

# Streamlit UI
st.title("NLP Analysis")
st.sidebar.header("Upload Datasets")
uploaded_files = st.sidebar.file_uploader(
    "Choose JSONL files (reviews.jsonl)",
    type="jsonl",
    accept_multiple_files=True
)

if uploaded_files:
    reviews = load_jsonl(uploaded_files)
    st.success(f"Loaded {len(reviews)} reviews successfully.")

    tabs = st.tabs(["Preprocessing", "Clustering", "Sentiment Analysis"])

    # Initialize session state
    for key in ["preprocessed_reviews", "clustering_results", "sentiment_results"]:
        if key not in st.session_state:
            st.session_state[key] = None

    # Preprocessing Tab
    with tabs[0]:
        st.header("Step 1: Text Preprocessing")

        # Add save options with radio buttons for format
        col1, col2 = st.columns(2)

        with col1:
            save_format = st.radio("Select format to save",
                               options=["JSON", "CSV"],
                               key="format_selection_radio")

        with col2:
            base_filename = st.text_input("Filename (without extension)",
                                      value="preprocessed_reviews",
                                      key="filename_input_field")

        if st.button("Execute Preprocessing", key="execute_preprocess_btn"):
            with st.spinner("Preprocessing reviews..."):
                st.session_state.preprocessed_reviews = preprocess_reviews(reviews)

                try:
                    # Save file in selected format
                    saved_file = save_preprocessed_data(
                        st.session_state.preprocessed_reviews,
                        filename=base_filename,
                        format=save_format.lower()
                    )
                    st.success(f"✅ File saved as {save_format} to: {saved_file}")

                    # Add download button for the saved file
                    with open(saved_file, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                        st.download_button(
                            label=f"Download {save_format} file",
                            data=file_content,
                            file_name=saved_file,
                            mime="application/json" if save_format == "JSON" else "text/csv",
                            key="download_processed_btn"
                        )

                except Exception as e:
                    st.error(f"Error saving preprocessed data: {str(e)}")

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
                    st.dataframe(df, key="sample_preview_df")

                    # Add download button for the sample
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download Sample as CSV",
                        data=csv,
                        file_name="preprocessed_sample.csv",
                        mime="text/csv",
                        key="download_sample_btn"
                    )
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
                    # Create corpus from preprocessed reviews
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

                    # Store results in session state including corpus
                    st.session_state.clustering_results = {
                        "clusters": clusters,
                        "corpus": corpus,  # Store corpus here
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

            # Detailed cluster analysis
            st.write("### Advanced Cluster Analysis")

            # Get corpus and clusters from results
            corpus = results["corpus"]
            clusters = results["clusters"]

            # Prepare texts by cluster for all analysis methods
            texts_by_cluster = defaultdict(list)
            for idx, cluster_id in enumerate(clusters):
                texts_by_cluster[cluster_id].append(corpus[idx])

            # Analysis method selection
            analysis_method = st.selectbox(
                "Select keyword extraction method",
                ["TF-IDF", "NER", "Bigrams", "Topic Analysis"],
                key="keyword_method"
            )

            for cluster_id, texts in texts_by_cluster.items():
                with st.expander(f"Cluster {cluster_id} Analysis ({len(texts)} documents)"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("#### Keywords & Topics")
                        if analysis_method == "TF-IDF":
                            keywords = extract_keywords_tfidf({cluster_id: texts}, n_keywords=10)
                            st.write("TF-IDF Keywords:")
                            keyword_df = pd.DataFrame(
                                keywords[cluster_id],
                                columns=["Keyword", "Score"]
                            )
                            st.dataframe(keyword_df)

                        elif analysis_method == "NER":
                            ner_counts = Counter()
                            for text in texts:
                                doc = nlp(text)
                                ner_counts.update([f"{ent.text} ({ent.label_})" for ent in doc.ents])
                            st.write("Named Entities:")
                            st.dataframe(pd.DataFrame(
                                ner_counts.most_common(10),
                                columns=["Entity (Type)", "Count"]
                            ))

                        elif analysis_method == "Bigrams":
                            st.write("Top Bigrams:")
                            bigrams = extract_bigrams(texts)
                            st.dataframe(pd.DataFrame(
                                bigrams,
                                columns=["Bigram", "Count"]
                            ))

                        else:  # Topic Analysis
                            st.write("Topic Analysis:")
                            word_freq = Counter()
                            for text in texts:
                                doc = nlp(text)
                                words = [token.lemma_ for token in doc
                                         if not token.is_stop and not token.is_punct]
                                word_freq.update(words)

                            st.dataframe(pd.DataFrame(
                                word_freq.most_common(10),
                                columns=["Topic Term", "Frequency"]
                            ))

                    with col2:
                        st.write("#### Cluster Statistics")
                        # Calculate cluster statistics
                        avg_length = np.mean([len(text.split()) for text in texts])
                        stats = {
                            "Document Count": len(texts),
                            "Average Document Length": f"{avg_length:.1f} words",
                            "Total Words": sum(len(text.split()) for text in texts)
                        }
                        st.write(pd.Series(stats))

                        # Show word frequency distribution
                        word_freq = Counter()
                        for text in texts:
                            word_freq.update(text.split())

                        fig, ax = plt.subplots(figsize=(8, 4))
                        words, counts = zip(*word_freq.most_common(10))
                        sns.barplot(x=list(counts), y=list(words))
                        plt.title("Top Words Distribution")
                        st.pyplot(fig)

                    # Show sample documents
                    st.write("#### Sample Documents")
                    n_samples = min(3, len(texts))
                    for i, text in enumerate(random.sample(texts, n_samples)):
                        st.text_area(
                            f"Document {i + 1}",
                            text[:500] + "..." if len(text) > 500 else text,
                            height=100,
                            key=f"sample_doc_{cluster_id}_{i}"
                        )

    # Sentiment Analysis Tab
    with tabs[2]:  # Note: Changed from tabs[2] to tabs[3] since it's the fourth tab
        st.header("Step 3: Sentiment Analysis")

        if st.button("Execute Sentiment Analysis"):
            if not st.session_state.preprocessed_reviews:
                st.warning("Please execute preprocessing first.")
            else:
                try:
                    with st.spinner("Loading sentiment analysis model..."):
                        # Load model and tokenizer
                        sentiment_model, sentiment_tokenizer = load_sentiment_model()

                        # Prepare data
                        texts = [" ".join(review["text"]) for review in
                                 st.session_state.preprocessed_reviews]
                        true_ratings = [review.get("rating") for review in reviews]
                        true_ratings = [r for r in true_ratings if
                                        r is not None]  # Filter out None values

                        # Set up device
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        st.info(f"Using device: {device}")
                        sentiment_model.to(device)

                        # Create dataset and dataloader
                        try:
                            with st.spinner("Processing reviews..."):
                                dataset = ReviewDataset(texts, sentiment_tokenizer)
                                dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

                                # Get predictions
                                predicted_ratings = predict_sentiments(dataloader, sentiment_model,
                                                                       device)

                                # Store results in session state
                                st.session_state.sentiment_results = {
                                    "texts": texts,
                                    "predicted_ratings": predicted_ratings,
                                    "true_ratings": true_ratings if true_ratings else None
                                }

                                if true_ratings and len(true_ratings) == len(predicted_ratings):
                                    metrics = calculate_metrics(true_ratings, predicted_ratings)

                                    st.write("### Analysis Metrics")

                                    # Create three columns for metrics display
                                    col1, col2, col3 = st.columns(3)

                                    with col1:
                                        st.metric(
                                            "Pearson Correlation",
                                            f"{metrics['pearson_correlation']:.3f}",
                                            help="Correlation between true and predicted ratings (-1 to 1)"
                                        )
                                        st.metric(
                                            "P-value",
                                            f"{metrics['pearson_p_value']:.3e}",
                                            help="Statistical significance of correlation"
                                        )

                                    with col2:
                                        st.metric(
                                            "Exact Precision",
                                            f"{metrics['exact_precision']:.2%}",
                                            help="Percentage of exactly matching predictions"
                                        )
                                        st.metric(
                                            "Within ±1 Precision",
                                            f"{metrics['within_one_precision']:.2%}",
                                            help="Percentage of predictions within 1 point of true rating"
                                        )

                                    with col3:
                                        st.metric(
                                            "Mean Absolute Error",
                                            f"{metrics['mae']:.2f}",
                                            help="Average absolute difference between true and predicted ratings"
                                        )
                                        st.metric(
                                            "Root Mean Square Error",
                                            f"{metrics['rmse']:.2f}",
                                            help="Root mean square error of predictions"
                                        )

                                    # Add a detailed correlation interpretation
                                    st.write("### Correlation Interpretation")
                                    corr = metrics['pearson_correlation']
                                    if abs(corr) > 0.7:
                                        strength = "strong"
                                    elif abs(corr) > 0.3:
                                        strength = "moderate"
                                    else:
                                        strength = "weak"

                                    direction = "positive" if corr > 0 else "negative"

                                    st.info(
                                        f"The model shows a {strength} {direction} correlation "
                                        f"({corr:.3f}) with the true ratings. "
                                        f"The p-value is {metrics['pearson_p_value']:.3e}, "
                                        f"{'indicating statistical significance' if metrics['pearson_p_value'] < 0.05 else 'suggesting the correlation might not be statistically significant'}."
                                    )

                                    # Plotting
                                    fig, ax = plt.subplots(figsize=(12, 6))

                                    # Plot both distributions
                                    if true_ratings and len(true_ratings) == len(predicted_ratings):
                                        # Create the distributions plot
                                        sns.kdeplot(data=true_ratings, label='True Ratings',
                                                    color='blue', alpha=0.6)
                                        sns.kdeplot(data=predicted_ratings, label='Predicted Ratings',
                                                    color='red', alpha=0.6)

                                        # Add histograms with transparency
                                        plt.hist(true_ratings, bins=5, color='blue', alpha=0.3,
                                                 density=True, label='True Ratings (Histogram)')
                                        plt.hist(predicted_ratings, bins=5, color='red', alpha=0.3,
                                                 density=True, label='Predicted Ratings (Histogram)')

                                        plt.title("Distribution of True vs Predicted Ratings")
                                        plt.xlabel("Rating")
                                        plt.ylabel("Density")
                                        plt.legend()

                                        # Add mean lines
                                        true_mean = np.mean(true_ratings)
                                        pred_mean = np.mean(predicted_ratings)
                                        plt.axvline(x=true_mean, color='blue', linestyle='--',
                                                    alpha=0.5, label=f'True Mean: {true_mean:.2f}')
                                        plt.axvline(x=pred_mean, color='red', linestyle='--',
                                                    alpha=0.5, label=f'Predicted Mean: {pred_mean:.2f}')

                                        # Update legend with mean values
                                        plt.legend(title="Rating Distributions")

                                    else:
                                        plt.text(0.5, 0.5, 'No true ratings available for comparison',
                                                 ha='center', va='center')
                                        plt.title("Ratings Comparison Not Available")

                                    plt.tight_layout()
                                    st.pyplot(fig)

                                    # Add summary statistics
                                    if true_ratings and len(true_ratings) == len(predicted_ratings):
                                        st.write("### Summary Statistics")
                                        stats_df = pd.DataFrame({
                                            'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                                            'True Ratings': [
                                                np.mean(true_ratings),
                                                np.median(true_ratings),
                                                np.std(true_ratings),
                                                np.min(true_ratings),
                                                np.max(true_ratings)
                                            ],
                                            'Predicted Ratings': [
                                                np.mean(predicted_ratings),
                                                np.median(predicted_ratings),
                                                np.std(predicted_ratings),
                                                np.min(predicted_ratings),
                                                np.max(predicted_ratings)
                                            ]
                                        })
                                        stats_df = stats_df.round(2)
                                        st.dataframe(stats_df)

                                # Add download button for full results
                                full_results_df = pd.DataFrame({
                                    'Text': texts,
                                    'Predicted Rating': predicted_ratings
                                })
                                if true_ratings:
                                    full_results_df['True Rating'] = true_ratings

                                csv = full_results_df.to_csv(index=False)
                                st.download_button(
                                    label="Download Full Results CSV",
                                    data=csv,
                                    file_name="sentiment_analysis_results.csv",
                                    mime="text/csv"
                                )

                        except Exception as e:
                            st.error(f"Error during sentiment analysis: {str(e)}")
                            st.error(
                                "Please make sure you have enough memory and the correct dependencies installed.")

                except Exception as e:
                    st.error(f"Error loading sentiment model: {str(e)}")
                    st.error(
                        "Please make sure you have the required model files and dependencies installed.")