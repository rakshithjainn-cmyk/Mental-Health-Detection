# src/inference.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json




def load_model(model_dir='models/bert-mh'):
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
with open(f'{model_dir}/label_map.json','r') as f:
label_map = json.load(f)
return tokenizer, model, label_map, device




def predict(text, model_dir='models/bert-mh', max_length=128):
tokenizer, model, label_map, device = load_model(model_dir)
model.eval()
inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=max_length)
inputs = {k:v.to(device) for k,v in inputs.items()}
with torch.no_grad():
logits = model(**inputs).logits
prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
idx = int(prob.argmax())
return {'label_id': idx, 'label_name': label_map[str(idx)] if str(idx) in label_map else label_map[idx], 'probs': prob.tolist()}


if __name__ == '__main__':
import sys
txt = ' '.join(sys.argv[1:]) or "i feel very sad and hopeless"
print(predict(txt))
