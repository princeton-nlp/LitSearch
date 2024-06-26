import numpy as np
from typing import List, Any
from sklearn.metrics.pairwise import cosine_similarity
from gritlm import GritLM
from eval.retrieval.kv_store import KVStore
from eval.retrieval.kv_store import TextType

class GRIT(KVStore):
    def __init__(self, index_name: str, raw_instruction: str, model_path: str = "GritLM/GritLM-7B"):
        super().__init__(index_name, 'grit')
        self.model_path = model_path
        self.raw_instruction = raw_instruction
        self._model = GritLM(model_path, torch_dtype="auto", device_map="auto", mode="embedding")
    
    def _get_instruction(self, type: TextType) -> str:
        if type == TextType.KEY:
            return "<|embed|>\n"
        elif type == TextType.QUERY:
            return "<|user|>\n" + self.raw_instruction + "\n<|embed|>\n"
        else:
            raise ValueError("Invalid TextType")
    
    def _encode_batch(self, texts: List[str], type: TextType, show_progress_bar: bool = True) -> List[Any]:
        return self._model.encode(texts, batch_size=256, instruction=self._get_instruction(type), show_progress_bar=show_progress_bar).astype(np.float16)
    
    def _query(self, encoded_query: Any, n: int) -> List[int]:
        try:
            cosine_similarities = cosine_similarity([encoded_query], self.encoded_keys)[0]
        except:
            for i, encoded_key in enumerate(self.encoded_keys):
                if np.any(np.isnan(encoded_key)):
                    self.encoded_keys[i] = np.zeros_like(encoded_key)
            cosine_similarities = cosine_similarity([encoded_query], self.encoded_keys)[0]
        top_indices = cosine_similarities.argsort()[-n:][::-1]
        return top_indices
    
    def load(self, path: str):
        super().load(path)
        self._model = GritLM(self.model_path, torch_dtype="auto", device_map="auto", mode="embedding")
        return self
        
