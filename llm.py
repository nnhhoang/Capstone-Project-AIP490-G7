from google.generativeai import caching
import google.generativeai as genai
import datetime
import os

from config import GOOGLE_API_KEY, generation_config, safety_settings, book_path
from data_analysis import read_pdf

class Prompt():

    def prompt_chapter(self, question):
        out_format = """
        {
            "Id": "...",
            "Correct answer explanation": "..."
        }
        """
        prompt = f"""
        'pythonlearn.pdf', which is a textbook about Python programming language. This
        textbook provides an Informatics-oriented introduction to programming. You responds have
        to based on the knowledge and context contained in this notebook.
        Now, remember your primary objective is to dissect and address each problem with a rigorous
        and detailed approach. This task involves:
        1. Clearly identifying and understanding the question.
        2. Breaking down the question into question part and selection part.
        3. Focus on question part, find out all the python syntax or python code.
        Applying relevant Python principles and techniques to solve code line-by-line,
        explain the concept contained in each line. Do not try to explain code in selection part.
        Instead, remember the number of line of explanation must equal to the number of line of
        provided code.
        4. Understand the requirement and synthesizing the line-by-line to formulate a
        comprehensive answer to explain the correct answer.
        Integrate step-by-step reasoning to solve Python problems under following JSON structure: {out_format}
        ###
        Here is description about each attribute:
        "Id": Index of the question
        "Correct answer explanation": Explain based on the correct answer. Do not explain anything about incorrect answer.
        Access to given file "Unit3.csv" to apply your task to all the question in the file.
        Begin your task from the question at the beginning and try to go through all the question.
        ###
        Question: {question['question_content']}
        Answer: {question['ans']}
        Id: {question['id']}
        """

        return prompt

    def prompt_check_dup(self):
        prompt = f"""

        """

        return prompt

class LLM(Prompt):

    def __init__(self):
        book = read_pdf(book_path)
        os.environ['GENAI_API_KEY'] = GOOGLE_API_KEY
        genai.configure(api_key=os.environ['GENAI_API_KEY'])
        
        self.cache = caching.CachedContent.create(
            model="models/gemini-1.5-pro-001",
            display_name="python for everyone", # used to identify the cache
            system_instruction="You are a code assistant, a highly advanced large language model have in-depth knowledge of Python programming. Your core strengths lie in tackling complex Python questions,utilizing intricate reasoning, and delivering solutions through methodical problem-solving. Throughout this interaction, you will encounter a variety of Python problems, ranging from basic theories to advanced algorithms.",
            contents=[book],
            ttl=datetime.timedelta(minutes=5),
        )
        self.LLM = genai.GenerativeModel.from_cached_content(cached_content=self.cache, generation_config=generation_config, safety_settings=safety_settings)


    def get_prompt(self,
                   task_num: int,
                   question_1,
                   question_2,
                   ):
        
        return (super().prompt_chapter(question_1) if task_num==1 else (
                super().prompt_check_dup(question_1,question_2)))
        
    def get_completion(self, 
                        prompt
                        ):
        result = self.LLM.generate_content(prompt)
        return result.text


# class Similar_checking():
    
    # def similar_paragraph(question_in: Question_attribute,
    #                       question_added: List[Question_attribute],
    #                       leng: int
    #                       ):

    #     ques_add = np.zeros((len(question_added)+1,5))
    #     feature = ["Index", "Z", "S", "E", "spatial_matrix"]
    #     ques_add = pd.DataFrame(ques_add, columns=feature)
                
    #     index_values = ques_add['Index']
    #     result = np.zeros(1,len(question_added))
    #     result = pd.DataFrame(result, columns=index_values[1:])

    #     if question_in.paragraph.get("Z") == "0":
    #         return result

    #     ques_add.at[1,"Index"] = question_in.id
    #     ques_add.at[1,"Z"] = question_in.paragraph.get("Z")
    #     ques_add.at[1,"S"] = question_in.paragraph.get("S")
    #     ques_add.at[1,"E"] = question_in.paragraph.get("E")

    #     for i in range(1,len(question_added)+1):
    #         specific_question = question_added[i]
    #         ques_add.at[i,"Index"] = specific_question.id
    #         ques_add.at[i,"Z"] = specific_question.paragraph.get("Z")
    #         ques_add.at[i,"Z"] = specific_question.paragraph.get("S")
    #         ques_add.at[i,"Z"] = specific_question.paragraph.get("E")

    #     for index, data in ques_add.iterrows():
    #         spatial_matrix = np.zeros((leng,297))
    #         z_list = list(map(int, data['Z'].split(',')))
    #         s_list = list(map(int, data['S'].split(',')))
    #         e_list = list(map(int, data['E'].split(',')))
            
    #         for z, s, e in zip(z_list, s_list, e_list):
    #             if z==0:
    #                 continue
    #             spatial_matrix[z,s:e] = 1
    #         ques_add.at[index, "spatial_matrix"] = spatial_matrix

    #     for index, data in ques_add.iterrows():
    #         if index == 0:
    #             continue
    #         spatialA = (ques_add.iloc[0].loc['spatial'] == 1).sum()
    #         spatialB = (data['spatial'] == 1).sum()
    #         compare = (data['spatial'] == 1) & (ques_add.iloc[0].loc['spatial'] == 1)
    #         num_equal = compare.sum()
    #         if num_equal==0:
    #             value = 0
    #         else:
    #             value = num_equal*100/(spatialA+spatialB-num_equal)
    #         result.at[0,data['spatial']] = value
                
    #     return result

    