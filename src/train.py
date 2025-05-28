import argparse
import logging
import os
from datasets import load_dataset
from evaluate import load as load_metric
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def preprocess_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True)

def compute_metrics(eval_pred):
    metric = load_metric("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def main(model_name, output_dir, dataset_name, num_labels, batch_size, epochs):
    logger.info("Loading dataset...")
    dataset = load_dataset(dataset_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_dataset = dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
        report_to="none",
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training completed.")

    logger.info("Evaluating...")
    results = trainer.evaluate()
    logger.info(f"Evaluation results: {results}")

    model.save_pretrained("models/movie_review_classifier")
    tokenizer.save_pretrained("models/movie_review_classifier")
    logger.info("Model saved to models/movie_review_classifier")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a text classification model with Transformers")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--dataset_name", type=str, default="imdb")
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    main(
        model_name=args.model_name,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        num_labels=args.num_labels,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )
