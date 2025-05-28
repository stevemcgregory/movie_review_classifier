from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F
import argparse

model = AutoModelForSequenceClassification.from_pretrained("models/movie_review_classifier")
tokenizer = AutoTokenizer.from_pretrained("models/movie_review_classifier")

def classify_review(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_class].item()
    return predicted_class, confidence

label_map = {0: "Negative", 1: "Positive"}

def main():
    parser = argparse.ArgumentParser(description="Run sentiment classification on a review.")
    parser.add_argument("--text", type=str, required=True, help="Review text to classify.")
    args = parser.parse_args()

    label, confidence = classify_review(args.text)
    print(f"Prediction: {label_map[label]} ({confidence:.2%} confidence)")

if __name__ == "__main__":
    main()
