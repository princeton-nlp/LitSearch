import os
import argparse
import datasets
from tqdm import tqdm
from utils import utils
from eval.retrieval.kv_store import KVStore

def load_index(index_path: str) -> KVStore:
    index_type = os.path.basename(index_path).split(".")[-1]
    if index_type == "bm25":
        from eval.retrieval.bm25 import BM25
        index = BM25(None).load(index_path)
    elif index_type == "instructor":
        from eval.retrieval.instructor import Instructor
        index = Instructor(None, None, None).load(index_path)
    elif index_type == "e5":
        from eval.retrieval.e5 import E5
        index = E5(None).load(index_path)
    elif index_type == "gtr":
        from eval.retrieval.gtr import GTR
        index = GTR(None).load(index_path)
    elif index_type == "grit":
        from eval.retrieval.grit import GRIT
        index = GRIT(None, None).load(index_path)
    else:
        raise ValueError("Invalid index type")
    return index

parser = argparse.ArgumentParser()
parser.add_argument("--index_name", type=str, required=True)

parser.add_argument("--top_k", type=int, required=False, default=200)
parser.add_argument("--retrieval_results_root_dir", type=str, required=False, default="results/retrieval")
parser.add_argument("--index_root_dir", type=str, required=False, default="retrieval_indices")
parser.add_argument("--dataset_path", required=False, default="princeton-nlp/LitSearch")
args = parser.parse_args()

index = load_index(os.path.join(args.index_root_dir, args.index_name))
query_set = [query for query in datasets.load_dataset(args.dataset_path, "query", split="full")]
for query in tqdm(query_set):
    query_text = query["query"]
    top_k = index.query(query_text, args.top_k)
    query["retrieved"] = top_k

os.makedirs(args.retrieval_results_root_dir, exist_ok=True)
output_path = os.path.join(args.retrieval_results_root_dir, f"{args.index_name}.jsonl")
utils.write_json(query_set, output_path)
