import numpy as np
from sentence_transformers import SentenceTransformer

class SentenceEmbeddingPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Inicializa o modelo de Sentence Transformers.
        A licença deste modelo é Apache 2.0.
        """
        
        self.model = SentenceTransformer(model_name)

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Transforma um texto/query em um vetor de dimensão 384.
        """
        return self.model.encode(text, convert_to_numpy=True).astype(np.float32)