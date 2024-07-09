import os
import copy
import json
import argparse 
import datasets
from tqdm import tqdm
from typing import List, Tuple
from utils import utils
from utils.openai_utils import OPENAIBaseEngine

###### QUERY CONSTRUCTION FUNCTIONS ######
def create_prompt_messages(item: dict, rank_start: int, rank_end: int, index_type: str) -> List[dict]:
    query = item['query']
    num_docs = len(item['documents'][rank_start:rank_end])

    if index_type == "title_abstract":
        messages = [{'role': 'system', 'content': "You are RankGPT, an intelligent assistant that can rank papers based on their relevancy to a research query."},
                    {'role': 'user', 'content': f"I will provide you with the abstracts of {num_docs} papers, each indicated by number identifier []. \nRank the papers based on their relevance to research question: {query}."},
                    {'role': 'assistant', 'content': 'Okay, please provide the papers.'}]
        max_length = 300
    elif index_type == "full_paper":
        messages = [{'role': 'system', 'content': "You are RankGPT, an intelligent assistant that can rank papers based on their relevancy to a research query."},
                    {'role': 'user', 'content': f"I will provide you with {num_docs} papers, each indicated by number identifier []. \nRank the papers based on their relevance to research question: {query}."},
                    {'role': 'assistant', 'content': 'Okay, please provide the papers.'}]
        max_length = 10000
    else:
        raise ValueError(f"Invalid index type: {index_type}")
    
    for rank, document in enumerate(item['documents'][rank_start: rank_end]):
        content = document['content'].replace('Title: Content: ', '').strip()
        content = ' '.join(content.split()[:int(max_length)])
        messages.append({'role': 'user', 'content': f"[{rank+1}] {content}"})
        messages.append({'role': 'assistant', 'content': f'Received passage [{rank+1}].'})
    postfix_prompt = f"Search Query: {query}. \nRank the {num_docs} papers above based on their relevance to the research query. The papers should be listed in descending order using identifiers. The most relevant papers should be listed first. The output format should be [] > [], e.g., [1] > [2]. Only respond with the ranking results, do not say any words or explain."
    messages.append({'role': 'user', 'content': postfix_prompt})
    return messages

###### RESPONSE PROCESSING FUNCTIONS ######
def clean_response(response: str):
    new_response = ''
    for c in response:
        new_response += (c if c.isdigit() else ' ')
    new_response = new_response.strip()
    return new_response

def remove_duplicate(response):
    new_response = []
    for c in response:
        if c not in new_response:
            new_response.append(c)
    return new_response

def receive_permutation(item, permutation, rank_start, rank_end):
    response = clean_response(permutation)
    response = [int(x) - 1 for x in response.split()]
    response = remove_duplicate(response)
    cut_range = copy.deepcopy(item['documents'][rank_start: rank_end])
    original_rank = [tt for tt in range(len(cut_range))]
    response = [ss for ss in response if ss in original_rank]
    response = response + [tt for tt in original_rank if tt not in response]
    for j, x in enumerate(response):
        item['documents'][j + rank_start] = copy.deepcopy(cut_range[x])
        if 'rank' in item['documents'][j + rank_start]:
            item['documents'][j + rank_start]['rank'] = cut_range[j]['rank']
        if 'score' in item['documents'][j + rank_start]:
            item['documents'][j + rank_start]['score'] = cut_range[j]['score']
    return item

def permutation_pipeline(model: OPENAIBaseEngine, item: dict, rank_start: int, rank_end: int, index_type: str) -> dict:
    decrement_rate = (rank_end - rank_start) // 5
    min_count = (rank_end - rank_start) // 2
    
    while rank_end - rank_start >= min_count:
        try:
            messages = create_prompt_messages(item, rank_start, rank_end, index_type)
            permutation = utils.prompt_gpt4_model(model, messages=messages)
            return receive_permutation(item, permutation, rank_start, rank_end)
        except Exception as e: # the context window might be overflowing; reduce the number of documents and try again;
            rank_end -= decrement_rate
            print(f"Error: context window overflow; reducing the number of documents to {rank_end - rank_start}")
    print(f"Error: unable to rerank the documents. Returning the original order.")
    return item
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval_results_file", type=str, required=True)

    parser.add_argument("--model", type=str, help="Simulator LLM", default="gpt-4-1106-preview")
    parser.add_argument("--max_k", default=100, type=int, help="Max number of retrieved documents to rerank")
    parser.add_argument("--output_dir", type=str, required=False, default="results/reranking/")
    parser.add_argument("--dataset_path", required=False, default="princeton-nlp/LitSearch")
    args = parser.parse_args()

    corpus_data = datasets.load_dataset(args.dataset_path, "corpus_clean", split="full")
    retrieval_results = utils.read_json(args.retrieval_results_file)
    model = utils.get_gpt4_model(args.model, azure=True)
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, os.path.basename(args.retrieval_results_file).replace(".json", ".reranked.json"))

    index_type = os.path.basename(args.retrieval_results_file).split(".")[1]
    if index_type == "title_abstract":
        corpusid_to_text = {utils.get_clean_corpusid(item): utils.get_clean_title_abstract(item) for item in corpus_data}
    elif index_type == "full_paper":
        corpusid_to_text = {utils.get_clean_corpusid(item): utils.get_clean_full_paper(item) for item in corpus_data}
    else:
        raise ValueError(f"Invalid index type: {index_type}")
    
    # truncate retrieval results to max_k
    for result in retrieval_results:
        result["retrieved"] = result["retrieved"][:args.max_k]

    ###### RERANKING ######
    # put retrieval results into format required by reranking pipeline
    reranking_inputs = []
    for query_info in retrieval_results:
        reranking_inputs.append({
            "query": query_info["query"],
            "documents": [{
                "content": corpusid_to_text[retrieved_corpusid],
                "corpusid": retrieved_corpusid
            } for retrieved_corpusid in query_info["retrieved"]]
        })
        
    # rerank
    if not os.path.exists(output_file):
        reranking_outputs = copy.deepcopy(retrieval_results)
        utils.write_json(reranking_outputs, output_file)
    
    for item_idx, item in enumerate(tqdm(reranking_inputs)):
        reranking_outputs = utils.read_json(output_file)
        if "pre_reranked" not in reranking_outputs[item_idx]:
            reranked_item = permutation_pipeline(model, item, rank_start=0, rank_end=len(item["documents"]), index_type=index_type)
            reranking_outputs[item_idx]["pre_reranked"] = reranking_outputs[item_idx]["retrieved"]
            reranking_outputs[item_idx]["retrieved"] = [document["corpusid"] for document in reranked_item["documents"]]    
            utils.write_json(reranking_outputs, output_file, silent=True) # save after each iteration
