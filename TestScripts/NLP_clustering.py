import json
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from collections import defaultdict, Counter
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import spacy

def load_cleaned_reviews(file_path):
    """Load preprocessed reviews from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def generate_embeddings(corpus, use_tfidf=False):
    """Generate embeddings for the given corpus."""
    if use_tfidf:
        vectorizer = TfidfVectorizer(max_features=1000)
        return vectorizer.fit_transform(corpus).toarray()
    else:
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        return model.encode(corpus)


def apply_clustering(embeddings, method="kmeans", num_clusters=6):
    """Cluster embeddings using the specified method."""
    if method == "kmeans":
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        return clusters, kmeans
    elif method == "dbscan":
        dbscan = DBSCAN(eps=0.5, min_samples=5, metric='cosine')
        clusters = dbscan.fit_predict(embeddings)
        return clusters, None
    else:
        raise ValueError(f"Unsupported clustering method: {method}")


def analyze_clusters(clusters, corpus):
    """Analyze clusters to extract keywords and named entities."""
    clustered_docs = defaultdict(list)
    for idx, cluster_id in enumerate(clusters):
        clustered_docs[cluster_id].append(corpus[idx])

    nlp = spacy.load("en_core_web_sm")
    cluster_keywords = {}

    for cluster_id, docs in clustered_docs.items():
        word_counter = Counter()
        for doc in docs:
            tokens = doc.split()
            word_counter.update(tokens)

        # Get top 10 most frequent words
        frequent_words = word_counter.most_common(10)

        # Extract named entities
        named_entities = Counter()
        for doc in docs:
            spacy_doc = nlp(doc)
            named_entities.update([ent.text for ent in spacy_doc.ents])
        top_entities = named_entities.most_common(5)

        cluster_keywords[cluster_id] = {
            "frequent_words": frequent_words,
            "named_entities": top_entities
        }

    return cluster_keywords


def visualize_clusters(embeddings, clusters):
    """Visualize clusters using PCA for dimensionality reduction."""
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 7))
    for cluster_id in set(clusters):
        cluster_points = reduced_embeddings[clusters == cluster_id]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster_id}")

    plt.title("Document Clusters")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.show()


def main():
    # Step 0: Load amazon_data
    cleaned_reviews_file = "ner_results_transformers.json"
    documents = load_cleaned_reviews(cleaned_reviews_file)
    corpus = [" ".join(doc["text"]) for doc in documents]

    # Step 1: Generate embeddings
    use_tfidf = False  # Set to True to use TF-IDF
    embeddings = generate_embeddings(corpus, use_tfidf=use_tfidf)

    # Step 2: Apply clustering
    clustering_method = "kmeans"  # Options: "kmeans", "dbscan"
    num_clusters = 6  # Only for KMeans
    clusters, model = apply_clustering(embeddings, method=clustering_method, num_clusters=num_clusters)

    # Step 3: Analyze clusters
    cluster_keywords = analyze_clusters(clusters, corpus)

    # Step 4: Display cluster analysis
    for cluster_id, keywords in cluster_keywords.items():
        print(f"Cluster {cluster_id}:")
        print("  Frequent Words:")
        for word, freq in keywords["frequent_words"]:
            print(f"    {word}: {freq}")
        print("  Named Entities:")
        for entity, freq in keywords["named_entities"]:
            print(f"    {entity}: {freq}")
        print()

    # Step 5: Evaluate clusters
    if clustering_method == "kmeans" and num_clusters > 1:
        silhouette_avg = silhouette_score(embeddings, clusters)
        print(f"Silhouette Score: {silhouette_avg:.4f}")

    # Step 6: Visualize clusters
    visualize_clusters(embeddings, clusters)


if __name__ == "__main__":
    main()

