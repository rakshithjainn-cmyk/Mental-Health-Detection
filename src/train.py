import os
import random
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from src.dataset import MHTextDataset
from src.preprocess import load_and_clean

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_metrics(p):
    from sklearn.metrics import accuracy_score, f1_score
    preds = np.argmax(p.predictions, axis=1)
    acc = accuracy_score(p.label_ids, preds)
    f1 = f1_score(p.label_ids, preds, average='weighted')
    return {"accuracy": acc, "f1_weighted": f1}

def main(
    data_path='data/sample.csv',
    model_name='bert-base-uncased',
    output_dir='models/bert-mh',
    epochs=3,
    batch_size=16,
    max_length=128
):
    seed_everything()
    os.makedirs(output_dir, exist_ok=True)

    df = load_and_clean(data_path)
    label2id = {l: i for i, l in enumerate(sorted(df['label'].unique()))}
    id2label = {v: k for k, v in label2id.items()}
    df['label'] = df['label'].map(label2id)

    X_train, X_val, y_train, y_val = train_test_split(
        df['text'].tolist(), df['label'].tolist(), test_size=0.15, random_state=42, stratify=df['label']
    )

    train_dataset = MHTextDataset(X_train, y_train, model_name=model_name, max_length=max_length)
    val_dataset = MHTextDataset(X_val, y_val, model_name=model_name, max_length=max_length)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(label2id), id2label=id2label, label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model='f1_weighted',
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        save_total_limit=2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(output_dir)

    import json
    with open(os.path.join(output_dir, 'label_map.json'), 'w') as f:
        json.dump(id2label, f)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data/sample.csv')
    parser.add_argument('--model_name', default='bert-base-uncased')
    parser.add_argument('--output_dir', default='models/bert-mh')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_length', type=int, default=128)
    args = parser.parse_args()
    main(**vars(args))
