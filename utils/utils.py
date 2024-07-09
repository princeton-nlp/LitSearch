import os
import json
from typing import List, Any, Tuple
from datasets import Dataset
from utils.openai_utils import OPENAIBaseEngine

##### file reading and writing #####

def read_json(filename: str, silent: bool = False) -> List[Any]:
    with open(filename, 'r') as file:
        if filename.endswith(".json"):
            data = json.load(file)
        elif filename.endswith(".jsonl"):
            data = [json.loads(line) for line in file]
        else:
            raise ValueError("Input file must be either a .json or .jsonl file")
    
    if not silent:
        print(f"Loaded {len(data)} records from {filename}")
    return data

def write_json(data: List[Any], filename: str, silent: bool = False) -> None:
    if filename.endswith(".json"):
        with open(filename, 'w') as file:
            json.dump(data, file, indent=4)
    elif filename.endswith(".jsonl"):
        with open(filename, 'w') as file:
            for item in data:
                file.write(json.dumps(item) + "\n")
    else:
        raise ValueError("Output file must be either a .json or .jsonl file")
    
    if not silent:
        print(f"Saved {len(data)} records to {filename}")

def read_txt(filename: str) -> str:
    with open(filename, 'r') as file:
        text = file.read()
    return text

##### evaluation metrics #####

def calculate_recall(retrieved: List[int], relevant_docs: List[int]) -> float:
    num_relevant_retrieved = len(set(retrieved).intersection(set(relevant_docs)))
    num_relevant = len(relevant_docs)
    recall = num_relevant_retrieved / num_relevant if num_relevant > 0 else 0
    return recall

def calculate_ndcg(retrieved: List[int], relevant_docs: List[int]) -> float:
    dcg = 0
    for idx, docid in enumerate(retrieved):
        if docid in relevant_docs:
            dcg += 1 / (idx + 1)
    idcg = sum([1 / (idx + 1) for idx in range(len(relevant_docs))])
    ndcg = dcg / idcg if idcg > 0 else 0
    return ndcg

def calculate_ngram_overlap(query: str, text: str) -> float:
    query_ngrams = set(query.split())
    text_ngrams = set(text.split())
    overlap = len(query_ngrams.intersection(text_ngrams)) / len(query_ngrams)
    return overlap

##### reading fields from corpus_s2orc #####

def get_s2orc_corpusid(item: dict) -> int:
    return item['corpusid']

def get_s2orc_title(item: dict) -> str:
    try:
        title_info = json.loads(item['content']['annotations']['title'])
        title_start, title_end = title_info[0]['start'], title_info[0]['end']
        return get_s2orc_text(item, title_start, title_end)
    except:
        return ""

def get_s2orc_abstract(item: dict) -> str:
    try:
        abstract_info = json.loads(item['content']['annotations']['abstract'])
        abstract_start, abstract_end = abstract_info[0]['start'], abstract_info[0]['end']
        return get_s2orc_text(item, abstract_start, abstract_end)
    except:
        return ""

def get_s2orc_title_abstract(item: dict) -> str:
    title = get_s2orc_title(item)
    abstract = get_s2orc_abstract(item)
    return f"Title: {title}\nAbstract: {abstract}"

def get_s2orc_full_paper(item: dict) -> str:
    if "content" in item and "text" in item['content'] and item['content']['text'] is not None:
        return item['content']['text']
    else:
        return ""

def get_s2orc_paragraph_indices(item: dict) -> List[Tuple[int, int]]:
    text = get_s2orc_full_paper(item)
    paragraph_indices = []
    paragraph_start = 0
    paragraph_end = 0
    while paragraph_start < len(text):
        paragraph_end = text.find("\n\n", paragraph_start)
        if paragraph_end == -1:
            paragraph_end = len(text)
        paragraph_indices.append((paragraph_start, paragraph_end))
        paragraph_start = paragraph_end + 2
    return paragraph_indices

def get_s2orc_text(item: dict, start_idx: int, end_idx: int) -> str:
    assert start_idx >= 0 and end_idx >= 0
    assert start_idx <= end_idx
    assert end_idx <= len(item['content']['text'])
    if "content" in item and "text" in item['content']:
        return item['content']['text'][start_idx:end_idx]
    else:
        return ""

def get_s2orc_paragraphs(item: dict, min_words: int = 10) -> List[str]:
    paragraph_indices = get_s2orc_paragraph_indices(item)
    paragraphs = [get_s2orc_text(item, paragraph_start, paragraph_end) for paragraph_start, paragraph_end in paragraph_indices]
    paragraphs = [paragraph for paragraph in paragraphs if len(paragraph.split()) >= min_words]
    return paragraphs

def get_s2orc_citations(item: dict, corpus_data: dict = None) -> List[int]:
    try:
        bibentry_string = item['content']['annotations']['bibentry']
        bibentry_data = json.loads(bibentry_string)
        citations = set()
        for ref in bibentry_data:
            if "attributes" in ref and "matched_paper_id" in ref["attributes"]:
                if (corpus_data is None) or (ref["attributes"]["matched_paper_id"] in corpus_data):
                    citations.add(ref["attributes"]["matched_paper_id"])
        return list(citations)
    except:
        return []

def get_s2orc_dict(data: Dataset) -> dict:
    return {get_s2orc_corpusid(item): item for item in data}

##### reading fields from corpus_clean #####

def get_clean_corpusid(item: dict) -> int:
    return item['corpusid']

def get_clean_title(item: dict) -> str:
    return item['title']

def get_clean_abstract(item: dict) -> str:
    return item['abstract']

def get_clean_title_abstract(item: dict) -> str:
    title = get_clean_title(item)
    abstract = get_clean_abstract(item)
    return f"Title: {title}\nAbstract: {abstract}"

def get_clean_full_paper(item: dict) -> str:
    return item['full_paper']

def get_clean_paragraph_indices(item: dict) -> List[Tuple[int, int]]:
    text = get_clean_full_paper(item)
    paragraph_indices = []
    paragraph_start = 0
    paragraph_end = 0
    while paragraph_start < len(text):
        paragraph_end = text.find("\n\n", paragraph_start)
        if paragraph_end == -1:
            paragraph_end = len(text)
        paragraph_indices.append((paragraph_start, paragraph_end))
        paragraph_start = paragraph_end + 2
    return paragraph_indices

def get_clean_text(item: dict, start_idx: int, end_idx: int) -> str:
    text = get_clean_full_paper(item)
    assert start_idx >= 0 and end_idx >= 0
    assert start_idx <= end_idx
    assert end_idx <= len(text)
    return text[start_idx:end_idx]

def get_clean_paragraphs(item: dict, min_words: int = 10) -> List[str]:
    paragraph_indices = get_clean_paragraph_indices(item)
    paragraphs = [get_clean_text(item, paragraph_start, paragraph_end) for paragraph_start, paragraph_end in paragraph_indices]
    paragraphs = [paragraph for paragraph in paragraphs if len(paragraph.split()) >= min_words]
    return paragraphs

def get_clean_citations(item: dict) -> List[int]:
    return item['citations']

def get_clean_dict(data: Dataset) -> dict:
    return {get_clean_corpusid(item): item for item in data}

##### openai gpt-4 model #####

def get_gpt4_model(model_name: str = "gpt-4-1106-preview", azure: bool = True) -> OPENAIBaseEngine:
    model = OPENAIBaseEngine(model_name, azure)
    model.test_api()
    return model

def prompt_gpt4_model(model: OPENAIBaseEngine, prompt: str = None, messages: List[dict] = None) -> str:
    if prompt is not None:
        messages = [{"role": "assistant", "content": prompt}]
    elif messages is None:
        raise ValueError("Either prompt or messages must be provided")
    
    response = model.safe_completion(messages)
    if response["finish_reason"] != "stop":
        print(f"Unexpected stop reason: {response['finish_reason']}")
    return response["content"]

##### cache directory #####

def get_cache_dir() -> str:
    return os.environ['HF_HOME']