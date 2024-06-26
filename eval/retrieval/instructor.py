import numpy as np
from typing import List, Any
from sklearn.metrics.pairwise import cosine_similarity
from InstructorEmbedding import INSTRUCTOR
from utils import utils
from eval.retrieval.kv_store import KVStore
from eval.retrieval.kv_store import TextType

class Instructor(KVStore):
    def __init__(self, index_name: str, key_instruction: str, query_instruction: str, model_path: str = "hkunlp/instructor-xl"):
        super().__init__(index_name, 'instructor')
        self.model_path = model_path
        self.key_instruction = key_instruction
        self.query_instruction = query_instruction
        self._model = INSTRUCTOR(model_path, device="cuda", cache_folder=utils.get_cache_dir())
    
    def _format_text(self, text: str, type: TextType) -> List[str]:
        if type == TextType.KEY:
            return [self.key_instruction, text]
        elif type == TextType.QUERY:
            return [self.query_instruction, text]
        else:
            raise ValueError("Invalid TextType")
    
    def _encode_batch(self, texts: List[str], type: TextType, show_progress_bar: bool = True) -> List[Any]:
        texts = [self._format_text(text, type) for text in texts]
        return self._model.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=show_progress_bar).astype(np.float16)
    
    def _query(self, encoded_query: Any, n: int) -> List[int]:
        cosine_similarities = cosine_similarity([encoded_query], self.encoded_keys)[0]
        top_indices = cosine_similarities.argsort()[-n:][::-1]
        return top_indices
    
    def load(self, path: str):
        super().load(path)
        self._model = INSTRUCTOR(self.model_path, device="cuda", cache_folder=utils.get_cache_dir())
        return self
        
