import os
import argparse
import datasets
from typing import List
from utils import utils
from eval.retrieval.kv_store import KVStore

def get_index_name(args: argparse.Namespace) -> str:
    return os.path.basename(args.dataset_path) + "." + args.key

def create_index(args: argparse.Namespace) -> KVStore:
    index_name = get_index_name(args)

    if args.index_type == "bm25":
        from eval.retrieval.bm25 import BM25
        index = BM25(index_name)
    elif args.index_type == "instructor":
        from eval.retrieval.instructor import Instructor
        if args.key == "title_abstract":
            query_instruction = "Represent the research question for retrieving relevant research paper abstracts:"
            key_instruction = "Represent the title and abstract of the research paper for retrieval:"
        elif args.key == "full_paper":
            query_instruction = "Represent the research question for retrieving relevant research papers:"
            key_instruction = "Represent the research paper for retrieval:"
        elif args.key == "paragraphs":
            query_instruction = "Represent the research question for retrieving passages from relevant research papers:"
            key_instruction = "Represent the passage from the research paper for retrieval:"
        else:
            raise ValueError("Invalid key")
        index = Instructor(index_name, key_instruction, query_instruction)
    elif args.index_type == "e5":
        from eval.retrieval.e5 import E5
        index = E5(index_name)
    elif args.index_type == "gtr":
        from eval.retrieval.gtr import GTR
        index = GTR(index_name)
    elif args.index_type == "grit":
        from eval.retrieval.grit import GRIT
        if args.key == "title_abstract":
            raw_instruction = "Given a research query, retrieve the title and abstract of the relevant research paper"
        elif args.key == "full_paper":
            raw_instruction = "Given a research query, retrieve the relevant research paper"
        elif args.key == "paragraphs":
            raw_instruction = "Given a research query, retrieve the passage from the relevant research paper"
        else:
            raise ValueError("Invalid key")
        index = GRIT(index_name, raw_instruction)
    else:
        raise ValueError("Invalid index type")
    return index

def create_kv_pairs(data: List[dict], key: str) -> dict:
    if key == "title_abstract":
        kv_pairs = {utils.get_clean_title_abstract(record): utils.get_clean_corpusid(record) for record in data}
    elif key == "full_paper":
        kv_pairs = {utils.get_clean_full_paper(record): utils.get_clean_corpusid(record) for record in data}
    elif key == "paragraphs":
        kv_pairs = {}
        for record in data:
            corpusid = utils.get_clean_corpusid(record)
            paragraphs = utils.get_clean_paragraphs(record)
            for paragraph_idx, paragraph in enumerate(paragraphs):
                kv_pairs[paragraph] = (corpusid, paragraph_idx)
    else:
        raise ValueError("Invalid key")
    return kv_pairs

parser = argparse.ArgumentParser()
parser.add_argument("--index_type", required=True) # bm25, instructor, e5, gtr, grit
parser.add_argument("--key", required=True), # title_absract, full_paper, paragraphs

parser.add_argument("--dataset_path", required=False, default="princeton-nlp/LitSearch")
parser.add_argument("--index_root_dir", required=False, default="retrieval_indices")
args = parser.parse_args()

corpus_data = datasets.load_dataset(args.dataset_path, "corpus_clean", split="full")
index = create_index(args)
kv_pairs = create_kv_pairs(corpus_data, args.key)
index.create_index(kv_pairs)

index_name = get_index_name(args)
index.save(args.index_root_dir)
