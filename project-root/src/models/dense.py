from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer
import torch
import faiss
import numpy as np

class DenseRetriever:
    def __init__(self, query_model="facebook/dpr-question_encoder-single-nq-base",
                 ctx_model="facebook/dpr-ctx_encoder-single-nq-base"):
        self.q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(query_model)
        self.q_encoder = DPRQuestionEncoder.from_pretrained(query_model)
        self.ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(ctx_model)
        self.ctx_encoder = DPRContextEncoder.from_pretrained(ctx_model)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_encoder.to(self.device)
        self.ctx_encoder.to(self.device)
        self.index = None
        self.doc_ids = None

    def index(self, passages):
        # Passages: list of (docid, text)
        embeddings = []
        self.doc_ids = [docid for docid, _ in passages]
        for _, text in passages:
            inputs = self.ctx_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                emb = self.ctx_encoder(**inputs).pooler_output.cpu().numpy()
            embeddings.append(emb)
        embeddings = np.vstack(embeddings)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner product (cosine similarity after normalization)
        self.index.add(embeddings)

    def retrieve(self, query, top_k=100):
        if self.index is None:
            raise RuntimeError("Index not built. Call index() first.")
        inputs = self.q_tokenizer(query, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            q_emb = self.q_encoder(**inputs).pooler_output.cpu().numpy()
        distances, indices = self.index.search(q_emb, top_k)
        return [(self.doc_ids[idx], float(dist)) for idx, dist in zip(indices[0], distances[0])]
