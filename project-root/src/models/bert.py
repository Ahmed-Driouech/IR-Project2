from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class BERTRanker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def score(self, query, doc):
        inputs = self.tokenizer(query, doc, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            score = torch.sigmoid(logits).item()  # Convert to probability
        return score

    def rerank(self, query, docs):
        scored = [(docid, self.score(query, text)) for docid, text in docs]
        return sorted(scored, key=lambda x: x[1], reverse=True)

