# from sentence_transformers import SentenceTransformer
import pandas as pd
# import json
from python_algo.config import cache_cost_per_token, cache_cost_per_hour, cache_time, token_in, token_out
# import math

data_dup = pd.read_csv(r"data\question_dup_count.csv")

# def semantic_search(new_input,
#                     database
#                     ):
#     # embedder = SentenceTransformer("all-MiniLM-L6-v2")
#     embedder = SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True)

#     # Corpus with example sentences
#     corpus = database
#     # Use "convert_to_tensor=True" to keep the tensors on GPU (if available)
#     corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
#     # Query sentences:
#     query = new_input
#     query_embedding = embedder.encode(query, convert_to_tensor=True)
#     similarity_scores = embedder.similarity(query_embedding, corpus_embeddings)[0]

#     return similarity_scores.item()

def eval_instruction(id, ranking_result):
    count = 0
    for row in ranking_result:
        if id == int(int(row[0])/100):
            if row[3] == 1:
                count+=1
    total_dup = 0
    for i, row in data_dup.iterrows():
        if row['id'] == id:
            total_dup = row['count_dup']
            break
    return total_dup, count

def cost_in_out(metadata: object) -> float:
    return (metadata.candidates_token_count * token_out) + ((metadata.prompt_token_count - metadata.cached_content_token_count)*token_in) + (metadata.cached_content_token_count * cache_cost_per_token)

def total_cost(metadata: object,
               in_out_cost: float) -> float:
    print(cache_cost_per_hour*cache_time*metadata.cached_content_token_count)
    return ((cache_cost_per_hour*cache_time*metadata.cached_content_token_count) + in_out_cost + (metadata.cached_content_token_count * token_in))