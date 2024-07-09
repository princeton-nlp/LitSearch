import os
import copy
import argparse
import datasets
from tqdm import tqdm
from utils import utils

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, required=True)

parser.add_argument("--output_dir", required=False, default="results/onehop/prereranking")
parser.add_argument("--cutoff_count", type=int, required=False, default=50)
parser.add_argument("--total_count", type=int, required=False, default=200)
parser.add_argument("--dataset_path", required=False, default="princeton-nlp/LitSearch")
args = parser.parse_args()

original_retrieval_results = utils.read_json(args.input_path)
corpus_data = utils.get_clean_dict(datasets.load_dataset(args.dataset_path, "corpus_clean", split="full"))

union_retrieval_results = []
for result in tqdm(original_retrieval_results):
    original_retrieved_ids = result["retrieved"][:args.cutoff_count]
    union_retrieved_ids = copy.deepcopy(original_retrieved_ids)
    citation_counts = []
    for paper_id in original_retrieved_ids:
        cited_papers = utils.get_clean_citations(corpus_data[paper_id])
        citation_counts.append(len(cited_papers))
        for cited_paper in cited_papers:
            if cited_paper not in union_retrieved_ids:
                union_retrieved_ids.append(cited_paper)
        if len(union_retrieved_ids) >= args.total_count:
            break

    result["original_retrieved"] = result["retrieved"]
    result["retrieved"] = union_retrieved_ids[:args.total_count]
    union_retrieval_results.append(result)

os.makedirs(args.output_dir, exist_ok=True)
output_path = os.path.join(args.output_dir, os.path.basename(args.input_path).replace(".json", ".union.json"))
utils.write_json(union_retrieval_results, output_path)
