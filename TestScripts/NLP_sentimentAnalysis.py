import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, precision_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


# Load JSONL Reviews
def load_reviews(file_path):
    reviews = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            reviews.append(json.loads(line.strip()))  # Parse each line as JSON
    return reviews


# Define Dataset
class ReviewDataset(Dataset):
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
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }


# Predict Sentiments
def predict_sentiments(dataloader, model, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Get logits
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Convert logits to probabilities and predict classes
            probs = torch.nn.functional.softmax(logits, dim=-1)
            pred_classes = torch.argmax(probs, dim=-1)
            predictions.extend(pred_classes.cpu().numpy())
    return predictions


# Evaluate Predictions
def evaluate_predictions(true_ratings, predicted_ratings):
    pearson_corr, _ = pearsonr(true_ratings, predicted_ratings)
    mse = mean_squared_error(true_ratings, predicted_ratings)
    mae = mean_absolute_error(true_ratings, predicted_ratings)
    r2 = r2_score(true_ratings, predicted_ratings)
    precision = precision_score(true_ratings, predicted_ratings, average="macro")  # For multiclass precision
    print(f"Pearson Correlation: {pearson_corr:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R^2 Score: {r2:.2f}")
    print(f"Precision (Macro): {precision:.2f}")
    return pearson_corr, mse, mae, r2, precision


# Plot Distributions
def plot_distributions(true_ratings, predicted_ratings):
    plt.figure(figsize=(10, 6))
    plt.hist(true_ratings, bins=5, alpha=0.6, label="True Ratings", range=(1, 6), align='left')
    plt.hist(predicted_ratings, bins=5, alpha=0.6, label="Predicted Ratings", range=(1, 6), align='left')
    plt.title("Comparison of True and Predicted Ratings")
    plt.xlabel("Ratings")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()


# Main Function
def main():
    # File Path
    file_path = "../amazon_data/reviews.jsonl"  # Adjust path if necessary

    # Load Data
    reviews = load_reviews(file_path)
    texts = [review["text"] for review in reviews]
    true_ratings = [int(review["rating"]) for review in reviews]  # Ensure ratings are integers

    # Load Model and Tokenizer
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    # model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    # model_name = "bert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Prepare Dataset and DataLoader
    dataset = ReviewDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=25)  # Adjust batch size based on your resources

    # Set Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Predict Sentiments
    print("Predicting sentiments...")
    predicted_ratings = predict_sentiments(dataloader, model, device)

    # Convert Predicted Ratings (0-4) to 1-5 Scale
    predicted_ratings = [pred + 1 for pred in predicted_ratings]

    # Evaluate Predictions
    print("Evaluating predictions...")
    pearson_corr, mse, mae, r2, precision = evaluate_predictions(true_ratings, predicted_ratings)

    # Visualize Results
    print("Visualizing results...")
    plot_distributions(true_ratings, predicted_ratings)

    # Save Model Performance
    output_file = f"performance-{model_name.replace('/', '-')}.json"
    model_performance = {
        "model name": model_name,
        "Pearson Correlation": pearson_corr,
        "Mean Squared Error (MSE)": mse,
        "Mean Absolute Error (MAE)": mae,
        "R^2 Score": r2,
        "Precision (Macro)": precision
    }
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(model_performance, file, ensure_ascii=False, indent=4)
    print(f"Model performance saved to {output_file}")


# Run Script
if __name__ == "__main__":
    main()