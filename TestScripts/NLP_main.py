import json
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load JSONL file function
def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]

# Preprocess text using spaCy
def preprocess_text_spacy(text):
    doc = nlp(text.lower())  # Process text with spaCy
    tokens = [
        token.lemma_ for token in doc
        if not token.is_stop            # Exclude stop words
        and not token.is_punct          # Exclude punctuation
        and not token.like_num          # Exclude numbers
        and not token.is_space          # Exclude spaces
        and not token.like_url          # Exclude URLs
    ]
    return tokens

# Main processing function
def process_reviews(reviews_file, output_file):
    # Load reviews amazon_data
    reviews = load_jsonl(reviews_file)

    # Process reviews
    processed_reviews = []
    for review in reviews:
        title_tokens = preprocess_text_spacy(review.get("title", ""))
        text_tokens = preprocess_text_spacy(review.get("text", ""))
        processed_reviews.append({
            "title": title_tokens,
            "text": text_tokens
        })

    # Save processed amazon_data
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(processed_reviews, file, ensure_ascii=False, indent=4)

    print(f"Processed reviews saved to {output_file}")

# File paths
reviews_file_path = "../amazon_data/reviews.jsonl"
output_file_path = "processed_reviews.json"

# Run the processing
process_reviews(reviews_file_path, output_file_path)
