from langchain_google_genai import GoogleGenerativeAI

import google.generativeai as genai
import datetime
import os
import PyPDF2
from python_algo.config import GOOGLE_API_KEY, generation_config, safety_settings, cache_time

global book_path
book_path = r"data\python_learn.pdf"

def read_pdf(file_path: str):
    pdf_data = ""  
    with open(file_path, "rb") as f:
        pdf = PyPDF2.PdfReader(f)
        for page in pdf.pages:
            new_page = page.extract_text()
            pdf_data = pdf_data + new_page
    return pdf_data

class Prompt():
    def prompt_chapter(self, question):
        out_format = """
        {
            "Id": "...",
            "Correct answer explanation": "..."
        }
        """
        prompt = f"""
        You are an expert in Python and a duplication checking agent, you have an in-depth knowledge
        of Python programming. Your core strengths lie in tackling complex Python questions,
        utilizing intricate reasoning, and delivering solutions through methodical problem-solving.
        Throughout this interaction, you will encounter a variety of Python problems,
        ranging from basic theories to advanced algorithms.

        'python for everyone', which is a textbook about Python programming language. This
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
        Integrate step-by-step reasoning to solve Python problems under following structure: {out_format}
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

    def prompt_check_dup(self, question1, question2):
        response_format = """
        [
            {"Id": "id of compare question", "Level": "level number", "Reason": "reason" },
            {"Id": "id of compare question", "Level": "level number", "Reason": "reason" },
            ...
            {"Id": "id of compare question", "Level": "level number", "Reason": "reason" }
            
        ]
        """
        prompt = f"""
        You are an expert in Python and a duplication checking agent, you have an in-depth knowledge
        of Python programming. Your core strengths lie in tackling complex Python questions,
        utilizing intricate reasoning, and delivering solutions through methodical problem-solving.
        Throughout this interaction, you will encounter a variety of Python problems,
        ranging from basic theories to advanced algorithms.
        ###
        Your primary objective is to dissect and address each problem with a rigorous and detailed
        approach. This involves:
        1. Clearly identifying and understanding the problem statement.
        2. Breaking down the problem into manageable components to understand the topic, concept, 
        example, and context in question.
        3. Analyzing the correct answer and instruction of the question to understand how to solve 
        the problem and the step-by-step to the correct answer.
        4. Compare given question with original question to find out whether they are duplicate or 
        not. Analyze the grammatical structure and meaning of two questions. Determine whether they 
        have the same subject, main verb, and semantic object. Use 'python for everyone' textbook as 
        a knowledge base or ontology to determine whether two questions refer to the same concept or 
        entity.
        ###
        I will give you an original question and a list of question. You have to comply with above 
        thought process to compare each question in the list with the original question. Each 
        question will be evaluated through the following fields:
            'Question': Orginal question,
            'Correct answer': Correct answer of the question,
            'Instruction': Correct answer explanation of the question,
            'IoU score': Similarity score based on knowledge from the book between this question and original question
            'Semantic score': the semantic score calculated between questions
        ###
        Your response must follow below JSON format: 
        {{
        "Id": {{"Level": "level number", "Reason": "an explaination"}},
        ...
        "Id": {{"Level": "level number", "Reason": "an explaination"}}
        }}
        With 'Id' is the Id of the question in list of question to be compared with original question, 'Level' is 0 for not duplicate or 1 for duplicate base on involves 4.
        Reason is the reason why you conclude that level. Reason is a string.
        This is an example: 
        {{
        "34": {{"Level": "1", "Reason": "an explaination"}},
        "33": {{"Level": "0", "Reason": "an explaination"}},
        ...
        "39": {{"Level": "1", "Reason": "an explaination"}}
        }}
        ###
        Original question: {question1}
        List of question: {question2}
        """

        return prompt
    
    def prompt_eval(self, question1, question2):
        format_out = """
        {
            'Id':...,
            'Score':...
        }
        """
        prompt =f"""
        You are an expert in evaluate the correction of how to solve a Python problem. Evaluate 
        the response to the below question, taking into account the correct answer supplied by 
        the teacher. You should give an evaluation between 0 and 5, with the following meanings:
        5: This is an excellent, accurate answer.
        4: Good answer, almost everything is correct.
        3: Mostly correct.
        2: Contains innacurracies.
        1: Mostly innaccurate.
        0: Completely wrong.
        Original question asked: {question1['question_content']}
        Correct answer: {question1['instruction']}
        Response to be evaluated by prompt technique: {question2}
        Please show your reasoning by summarizing the correct answer and summarizing the answer 
        from the response to be evaluated by prompt technique. Break the summary to smaller step .
        Then, comparing whether each correspond step in summary are the same or not. If there 
        are lack of information or redundant of information. The step-by-step must be sequentially, 
        if the first step is define something, the correspond step in evaluated response must be 
        also have 'define' keyword.
        Please note that your score should be in a json format as below:{format_out}
        """

        return prompt

class LLM:
    def __init__(self):
        book = read_pdf(book_path)
        os.environ['GENAI_API_KEY'] = GOOGLE_API_KEY
        genai.configure(api_key=os.environ['GENAI_API_KEY'])

        self.prompt = Prompt()

       
        genai.configure(api_key= GOOGLE_API_KEY)
        self.LLM = genai.GenerativeModel('gemini-1.5-flash', generation_config=generation_config)

    def get_prompt(self, task_num, question_1, question_2 = None):
        prompt = self.prompt
        if task_num == 1:
            return prompt.prompt_chapter(question_1)
        elif task_num == 2:
            return prompt.prompt_check_dup(question_1,question_2)
        else:
            return prompt.prompt_eval(question_1,question_2)
     
    def get_completion(self, 
                        prompt
                        ):
        result = self.LLM.generate_content(prompt)
        return result.text
    
    def get_completion_for_eval(self,
                                prompt
                                ):
        result = self.LLM.generate_content(prompt)
        return result
    
