from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

class T5Ranker:
    def __init__(self, model_name="castorini/monot5-base-msmarco"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def score(self, query, doc):
        # Format input as MonoT5 expects: "Query: <query> Document: <doc> Relevant:"
        input_text = f"Query: {query} Document: {doc} Relevant:"
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            # Generate the "true" token probability
            true_id = self.tokenizer.encode("true")[0]
            outputs = self.model.generate(**inputs, return_dict_in_generate=True, output_scores=True)
            logits = outputs.scores[0]  # Scores for the first generated token
            probs = torch.softmax(logits, dim=-1)
            score = probs[0, true_id].item()  # Probability of "true"
        return score

    def rerank(self, query, docs):
        scored = [(docid, self.score(query, text)) for docid, text in docs]
        return sorted(scored, key=lambda x: x[1], reverse=True)

    def retrieve(self, query, docs, top_k=100):
        # For consistency, treat as reranking over a candidate set
        reranked = self.rerank(query, docs)
        return reranked[:top_k]
