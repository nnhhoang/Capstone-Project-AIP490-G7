from python_algo.config import MODEL_SEMANTIC, Test_set_path
from sentence_transformers import util
import pandas as pd
import ast
import torch
def Semantic_eval(predict_set):
    df = pd.read_csv(Test_set_path)
    sentences = df['instruction'].tolist()

    predict_embeddings = MODEL_SEMANTIC.encode(predict_set, convert_to_tensor=True)
    test_embeddings = MODEL_SEMANTIC.encode(sentences, convert_to_tensor=True)
    cosine_similarities = util.cos_sim(predict_embeddings, test_embeddings)

    num_questions = len(df)
    Sum = 0
    for i in range(num_questions):
        Sum+=cosine_similarities[i][i]

    return Sum/num_questions

def Semantic_calculate(question, 
                       id_set,
                       data
                       ):
    
    embed_data = data
    
    sentence_input = question['question_content']

    input_embeddings = MODEL_SEMANTIC.encode(sentence_input, convert_to_tensor=True)
    list_embed = []
    compare_vectors = []
    
    for id in id_set:
        compare_vector = embed_data[embed_data['id']== int(id)]['embed_column'].iloc[0]
        embed_list = ast.literal_eval(compare_vector)
        compare_vectors.append(embed_list)
    print('done vector compare')
    
    for vector in compare_vectors:  
        cosine_similarities = util.cos_sim(input_embeddings, vector)
        list_embed.append(cosine_similarities)
    print('cosine done')
    return list_embed
# def Semantic_calculate(question, id_set, data):
    
#     # 1. Prepare Input Embeddings
#     input_embeddings = MODEL_SEMANTIC.encode(question['question_content'], convert_to_tensor=True)

#     # 2. Filter and Extract Embeddings
#     compare_data = data[data['id'].isin(id_set)]
#     compare_embeddings = compare_data['embed_column'].apply(ast.literal_eval).tolist()
#     compare_embeddings = [torch.tensor(emb) for emb in compare_embeddings]

#     # 3. Batch Cosine Similarity Calculation
#     all_similarities = util.cos_sim(input_embeddings, compare_embeddings)
#     similarities_list = all_similarities[0].tolist()  # Extract similarities as a list

#     return similarities_list