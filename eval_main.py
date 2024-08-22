import pandas as pd
from python_algo.gemini_api import LLM
from python_algo.evaluation import cost_in_out, total_cost, eval_instruction
from python_algo.database_management import question_database_manage
from python_algo.semantic_process import Semantic_eval
import json


data_in_path = r"data\question_data3.csv"
database_path = r"data\question_data_full.csv"
data_in = pd.read_csv(r'data\question_data3.csv')
data_base = pd.read_csv(r"data\question_data_full.csv")
Gemini = LLM()
question_database = question_database_manage(Gemini, database_path)

# print(data_base)

metadata_gemini: object
count = 0
cost = 0
list_rank: dict = {}
data_dup = []
list_ins = []

for row in data_in.itertuples():
    # prompt_ins = Gemini.get_prompt(1, question_1={"id": row.id, "question_content": row.question_content, "ans": row.ans}, question_2=None)    
    # result, metadata = Gemini.get_completion_for_eval(prompt=prompt_ins)
    # cost += cost_in_out(metadata=metadata)
    # metadata_gemini = metadata
    # print(result, metadata, type(result))
    
    # try:
    #     result= json.loads(result)
    #     instruction = result["Correct answer explanation"]
    # except json.JSONDecodeError as e:
    #     print(f"Error decoding JSON: {e}")
    
    # list_ins.append(result)
    
    question_main = {"id": row.id, 
                     "question_content": row.question_content, 
                     "ans": row.ans, 
                     "instruction": row.instruction, #result
                     "difficulty": None,
                     "subchapters": row.subchapters, 
                     "paragraph": row.paragraph,
                     "spatial_matrix": json.loads(row.spatial_matrix)}
    
    ranking_result, final_result, metadata = question_database.new_ranking_question(question_main,k=10)
    # cost += cost_in_out(metadata=metadata)
    # ranking_result = sorted(ranking_result, key=lambda x: (x[3], x[1], x[2]), reverse=True)
    # print(ranking_result, metadata)
    # real_dup, out_dup = eval_instruction(row.id, ranking_result)
    # data_dup.append([row.id, real_dup, out_dup])
    # list_rank[row.id]=ranking_result

    # count+=1
    # if count==1: 
    #     break

# prompt_eval = Gemini.get_prompt(3,question_1=data_in, question_2=list_ins)
# ins_result, metadata = Gemini.get_completion_for_eval(prompt=prompt_eval)
# semantic_eval = Semantic_eval(list_ins)

index = ['id', 'real_dup', 'out_dup']
data_dup = pd.DataFrame(data_dup, columns=index)
data_dup.to_csv('data\dup_result.csv', index=False)

# Serializing json
json_object = json.dumps(list_rank)
 
# Writing to sample.json
with open("result.json", "w") as outfile:
    outfile.write(json_object)

cost = total_cost(metadata= metadata_gemini, in_out_cost=cost)


print("-------------------")
print(list_rank)
print("-------------------")
print(data_dup)
print("-------------------")
print(cost)
print("-------------------")