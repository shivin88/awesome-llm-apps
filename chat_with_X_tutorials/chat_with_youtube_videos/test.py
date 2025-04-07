import torch
from sentence_transformers import SentenceTransformer

try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode("This is a test sentence.")
    print("Embeddings shape:", embeddings.shape)
except RuntimeError as e:
    print(f"RuntimeError during sentence-transformers import or usage: {e}")
except Exception as e:
    print(f"Other error: {e}")