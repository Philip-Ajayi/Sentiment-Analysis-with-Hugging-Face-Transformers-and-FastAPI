import torch
from transformers import BertTokenizer, BertForSequenceClassification

class SentimentModel:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.model.eval()  # Set the model to evaluation mode

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
        return prediction  # 0 for negative, 1 for positive
