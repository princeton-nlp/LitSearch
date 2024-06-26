import sentence_transformers
import numpy as np
from typing import List, Any
from sklearn.metrics.pairwise import cosine_similarity
from eval.retrieval.kv_store import KVStore
from eval.retrieval.kv_store import TextType
from utils import utils

class E5(KVStore):
    def __init__(self, index_name: str, model_path: str = "intfloat/e5-large-v2"):
        super().__init__(index_name, 'e5')
        self.model_path = model_path
        self._model = sentence_transformers.SentenceTransformer(model_path, device="cuda", cache_folder=utils.get_cache_dir()).bfloat16()
    
    def _format_text(self, text: str, type: TextType) -> str:
        if type == TextType.KEY:
            text = "passage: " + text
        elif type == TextType.QUERY:
            text = "query: " + text
        else:
            raise ValueError("Invalid TextType")
        return text
    
    def _encode_batch(self, texts: List[str], type: TextType, show_progress_bar: bool = True) -> List[Any]:
        texts = [self._format_text(text, type) for text in texts]
        return self._model.encode(texts, batch_size=256, normalize_embeddings=True, show_progress_bar=show_progress_bar).astype(np.float16)
    
    def _query(self, encoded_query: Any, n: int) -> List[int]:
        cosine_similarities = cosine_similarity([encoded_query], self.encoded_keys)[0]
        top_indices = cosine_similarities.argsort()[-n:][::-1]
        return top_indices
    
    def load(self, path: str):
        super().load(path)
        self._model = sentence_transformers.SentenceTransformer(self.model_path, device="cuda", cache_folder=utils.get_cache_dir())
        return self
        
