# src/eval.py
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import json




def predict_texts(texts, model_dir, max_length=128, device=None):
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
device = device or ("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
all_preds = []
with torch.no_grad():
for t in texts:
inputs = tokenizer(t, return_tensors='pt', truncation=True, padding='max_length', max_length=max_length)
inputs = {k:v.to(device) for k,v in inputs.items()}
logits = model(**inputs).logits
pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
all_preds.append(int(pred))
return all_preds




def evaluate(test_texts, test_labels, model_dir):
preds = predict_texts(test_texts, model_dir)
print(classification_report(test_labels, preds))
print('Confusion matrix:')
print(confusion_matrix(test_labels, preds))
