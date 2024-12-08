import json
import spacy
from transformers import pipeline

# Load Preprocessed Reviews
def load_cleaned_reviews(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# Save NER Results
def save_results(results, output_file):
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=4)
    print(f"NER results saved to {output_file}")

# Apply NER with spaCy
def extract_entities_spacy(documents):
    nlp = spacy.load("en_core_web_sm")  # Load spaCy model
    results = []
    for doc in documents:
        text = " ".join(doc["text"])  # Combine tokens into a single string
        spacy_doc = nlp(text)
        entities = [{"text": ent.text, "label": ent.label_} for ent in spacy_doc.ents]
        results.append({"text": text, "entities": entities})
    return results

# Apply NER with Hugging Face Transformers
def extract_entities_transformers(documents):
    ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")  # Load Transformers model
    results = []
    for doc in documents:
        text = " ".join(doc["text"])  # Combine tokens into a single string
        entities = ner_pipeline(text)
        # Convert float32 to float for JSON compatibility
        for entity in entities:
            entity["score"] = float(entity["score"])
        results.append({"text": text, "entities": entities})
    return results

# Main Function
def main(input_file, output_file_spacy, output_file_transformers):
    # Load preprocessed reviews
    documents = load_cleaned_reviews(input_file)

    # Apply spaCy NER
    print("Applying spaCy NER...")
    results_spacy = extract_entities_spacy(documents)
    save_results(results_spacy, output_file_spacy)

    # Apply Transformers NER
    print("Applying Hugging Face Transformers NER...")
    results_transformers = extract_entities_transformers(documents)
    save_results(results_transformers, output_file_transformers)

# Paths
input_file = "processed_reviews.json"  # Input file with preprocessed reviews
output_file_spacy = "ner_results_spacy.json"  # Output file for spaCy results
output_file_transformers = "ner_results_transformers.json"  # Output file for Transformers results

# Run the script
if __name__ == "__main__":
    main(input_file, output_file_spacy, output_file_transformers)

