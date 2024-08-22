import pandas as pd
import numpy as np
import ast
import operator
import json
from fuzzywuzzy import fuzz

from python_algo.semantic_process import Semantic_calculate
import time

class question_database_manage:
    def __init__(self, LLM, question_database_path):
        self.data_path = question_database_path
        self.data = pd.read_csv(question_database_path)
        self.LLM = LLM
    
    def add_question(self, question_content):
        new_row_df = pd.DataFrame([question_content])  # Convert question data to DataFrame
        self.data = pd.concat([self.data, new_row_df], ignore_index=True)
        self.data.to_csv(self.data_path, index=False)
    
    def calculate_iou(self, matrix1, matrix2):
        intersection = np.logical_and(matrix1, matrix2)
        union = np.logical_or(matrix1, matrix2)
        iou = np.sum(intersection) / np.sum(union)
        return iou

    def get_rows_with_non_zero(self, matrix):
        return np.where(np.any(matrix != 0, axis=1))[0]

    def compare_matrices(self, matrix1, matrix2):
        # Get rows with non-zero elements in both matrices
        rows_matrix1 = self.get_rows_with_non_zero(matrix1)
        rows_matrix2 = self.get_rows_with_non_zero(matrix2)
        
        # Union of both row indices
        rows_to_take = np.union1d(rows_matrix1, rows_matrix2)
        
        # Extract the relevant rows from both matrices
        matrix1_small = matrix1[rows_to_take]
        matrix2_small = matrix2[rows_to_take]
        
        # Calculate IOU between the two small matrices
        iou = self.calculate_iou(matrix1_small, matrix2_small)
        
        return iou
    def transform_str_numpy_array(self, string_matrix):
        
        listMatrix = json.loads(string_matrix)
        numpy_array = np.array(listMatrix)
        
        return numpy_array
    
    def search_spatial(self, question_content):
        
        list_rank = {}
        for row in self.data.iloc:
           
            new_matrix = np.array(question_content['spatial_matrix'])
            old_matrix = self.transform_str_numpy_array(row['spatial_matrix'])
            list_rank[str(row['id'])] = self.compare_matrices(new_matrix, old_matrix)
        return list_rank
    
    def ranking_question(self, question_content, k= 30):
        final_result = []
        spatial_result  = self.search_spatial(question_content)
        
        #need output from gemini in json to continue
        #filter_df = self.data[(self.data['difficulty'] == question_content['difficulty'].iloc[0]) & (self.data['learning_outcome'] == question_content['learning_outcome'].iloc[0])]
        sorted_ranks = sorted(spatial_result.items(), key=operator.itemgetter(1), reverse=True)

        # Handle Ties (Optional)
        top_scores = sorted_ranks[:k]  # Get top 5 initially
        current_score = top_scores[-1][1]  # Score of the 5th item
        for i in range(k, len(sorted_ranks)):
            if sorted_ranks[i][1] == current_score:
                top_scores.append(list(sorted_ranks[i]))  # Add ties to top_scores
            else:
                break  # Stop if scores are lower
        
        new_top_score = []
        for item in top_scores:
            item = list(item)
            input_question = {"id": question_content['id'], "question_content": question_content["question_content"], "ans": question_content["ans"], "instruction": question_content["instruction"]}
            compare_question_id = item[0]
            compare_question_content = self.data[self.data['id'] == int(compare_question_id)]
            
            compare_input = {"id": compare_question_id, "question_content": compare_question_content["question_content"].iloc[0], "ans": compare_question_content["ans"].iloc[0], "instruction": None}
            prompt = self.LLM.get_prompt(2, input_question, compare_input, item[1])
            answer = self.LLM.get_completion(prompt)
            print(answer)
            level = int(answer.split(',')[0].split(':')[-1].strip(' "'))
            item.append(item[0])
            item.append(level)
            similar = fuzz.ratio(question_content["question_content"], compare_question_content["question_content"].iloc[0] )
            new_item = [item[0], item[1], similar , level]
            new_top_score.append(new_item)
            final_result.append((question_content['id'],compare_question_id,level,answer))
            
        return new_top_score, final_result
  

    def new_ranking_question(self, question_content, k= 40):
        time1 = time.time()
        spatial_result  = self.search_spatial(question_content)
        time2 = time.time()
        print('got spatial result', time2 - time1)
        
        time3 = time.time()
        sorted_ranks = sorted(spatial_result.items(), key=operator.itemgetter(1), reverse=True)
        top_scores = []
        for i in range(0, k):
            top_scores.append(list(sorted_ranks[i]))
        # Handle Ties (Optional)
        # top_scores = sorted_ranks[:k]  # Get top 5 initially
        # current_score = top_scores[-1][1]  # Score of the 5th item
        for i in range(k, len(sorted_ranks)):
            if int(sorted_ranks[i][1]) == 1:
                top_scores.append(list(sorted_ranks[i]))  # Add ties to top_scores
            else:
                break  # Stop if scores are lower

        top_scores.sort(key=lambda x: x[1], reverse=True)
        new_top_score = []
        time4 = time.time()
        print(time4 - time3, 'top_score done')
        
        time5 = time.time()
        embed_data = pd.read_csv('embed_database_full.csv')
        list_semantic = []
        for item in top_scores:
            item = list(item)
            item_id = item[0]
            compare_question = self.data[self.data['id'] == int(item_id)]
            list_semantic.append(item_id)
        semantic_result = Semantic_calculate(question_content, list_semantic, embed_data)
        time6 = time.time()
        print(time6 -time5,'semantic_done------------', semantic_result)
        

        time7 = time.time()
        count = 0
        list_question=""
        input_question = {"id": question_content['id'], "question_content": question_content["question_content"], "ans": question_content["ans"], "instruction": question_content["instruction"]}
        for item in top_scores:
            item = list(item)
            compare_question_id = item[0]
    
            compare_question_content = self.data[self.data['id'] == int(compare_question_id)]['question_content'].iloc[0]
            compare_question_ans = self.data[self.data['id'] == int(compare_question_id)]["ans"].iloc[0]
            compare_question_ins = self.data[self.data['id'] == int(compare_question_id)]["instruction"].iloc[0]
            list_question += " [id: " + str(compare_question_id) + ", Content: " + str(compare_question_content) + ", Correct answer:" + str(compare_question_ans) + ", Instruction:" + str(compare_question_ins) + ", IoU score:" + str(item[1]) + ", Semantic score:" + str(semantic_result[count]) + "] "           
            count+=1
        prompt = self.LLM.get_prompt(2,input_question,list_question)
        time8 = time.time()
        print('done', time8 - time7)
        #answer, metadata = self.LLM.get_completion_for_eval(prompt)
        # index=0
        # answer= json.loads(answer)
        
        # for id, details in answer.items():
        #     #search for item in top_scores that has the same id
        #     for i in range(len(top_scores)):
        #         if top_scores[i][0] == id:
        #             index = i
        #     new_top_score.append([id, top_scores[index][1], semantic_result[0][index], details['Level'], details['Reason']])
        #     # new_top_score.append([id, top_scores[index][1], '70%', semantic_result[index], details['Reason']])
      
        
        return new_top_score,  prompt
