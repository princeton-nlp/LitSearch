# LitSearch

This repository contains the code and data for paper LitSearch: A Retrieval Benchmark for Scientific **Lit**erature **Search**. In this paper, we introduce a benchmark consisting of a set of 597 realistic literature search queries about recent ML and NLP papers. We provide the code we used for benchmarking state-of-the-art retrieval models and two LLM-based reranking pipelines.

<img src="https://github.com/princeton-nlp/LitSearch/assets/6129506/569873f4-ba04-4bb2-a86f-ed44c77c6ded" alt="LitSearch" width="100%" >

## Requirements
Please install the latest versions of PyTorch (`torch`), NumPy (`numpy`), HuggingFace Transformers (`transformers`), HuggingFace Datasets (`datasets`), SentenceTransformers (`sentence-transformers`), InstructorEmbedding (`InstructorEmbedding`), Rank-BM25 (`rank-bm25`), GritLM (`gritlm`) and the OpenAI API package (`openai`). This codebase is tested on `torch==1.13.1`, `numpy==1.23.5`, `transformers==4.30.2`, `datasets==2.20.0`, `sentence-transformers==2.2.2`, `InstructorEmbedding==1.0.1`, `rank-bm25==0.2.2`, `gritlm==1.0.0` and `openai==1.33.0` with Python 3.10.14.

Note: We used a standalone environment for GritLM since its dependencies were incompatible with other packages.

## Data
We provide the LitSearch query set and retrieval corpus as separate HuggingFace `datasets` configurations under [`princeton-nlp/LitSearch`](https://huggingface.co/datasets/princeton-nlp/LitSearch). We also provide the retrieval corpus in the Semantic Scholar Open Research Corpus (S2ORC) format along with all available metadata to facilitate exploration of retrieval strategies more advanced than the ones we implement in this codebase. The data can be downloaded using the `datasets` package using
```python
from datasets import load_dataset

query_data = load_dataset("princeton-nlp/LitSearch", "query", split="full")
corpus_clean_data = load_dataset("princeton-nlp/LitSearch", "corpus_clean", split="full")
corpus_s2orc_data = load_dataset("princeton-nlp/LitSearch", "corpus_s2orc", split="full")
```

## Code Structure
* `eval/retrieval/`
    * Contains a parent class for retrievers in `kv_store.py` and implementations of 5 retrieval pipelines including BM25 (`bm25.py`), GTR (`gtr.py`), Instructor (`instructor.py`), E5 (`e5.py`) and GRIT (`grit.py`).
    * Contains `build_index.py` for building a retrieval index of the required type using a given retrieval corpus.
    * Contains `evaluate_index.py` for evaluating a retriever using the associated retrieval index and a query set.
* `eval/reranking/rerank.py` contains code for reranking a provided set of retrieval results using GPT4. This code is adapted from [Rank-GPT](https://github.com/sunnweiwei/RankGPT).
* `eval/onehop/get_onehop_union.py` contains code that implements the first stage of the one-hop reranking operation described in section 3.2 of our paper. Once the union is computed using this script, GPT4-based reranking is applied as before using `eval/reranking/rerank.py`.

## Evaluation
This repository provides support for running evaluations using the BM25, GTR, Instructor, E5 and GRIT retrievers, reranking using GPT-4, and executing a one-hop reranking strategy. We provide sample commands for running the corresponding scripts:

#### Build retrieval index
```bash
python3 -m eval.retrieval.build_index --index_type bm25 --key title_abstract
```

#### Run retrieval using built index
```bash
python3 -m eval.retrieval.evaluate_index --index_name LitSearch.title_abstract.bm25
```

#### Run GPT-4-based reranking
```bash
python3 -m eval.reranking.rerank --retrieval_results_file results/retrieval/LitSearch.title_abstract.bm25.jsonl 
```

#### Run one-hop strategy (union + reranking)
```bash
python3 -m eval.onehop.get_onehop_union --input_path results/retrieval/LitSearch.title_abstract.bm25.jsonl
python3 -m eval.reranking.rerank --retrieval_results_file results/onehop/prereranking/LitSearch.title_abstract.bm25.union.jsonl --output_dir results/onehop/postreranking --max_k 200 
```

## Bug or Questions?

If you have any questions related to the code or the paper, feel free to email Anirudh (`anirudh.ajith@princeton.edu`). If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker!

## Citation

Please cite our paper if you use LitSearch in your work:
```bibtex
@article{ajith2024litsearch,
  title={LitSearch: A Retrieval Benchmark for Scientific Literature Search},
  author={Ajith, Anirudh and Xia, Mengzhou and Chevalier, Alexis and Goyal, Tanya and Chen, Danqi and Gao, Tianyu},
  year={2024}
}
```
