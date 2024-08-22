from langchain_google_genai import GoogleGenerativeAI
from python_algo.config import GOOGLE_API_KEY_FOR_GEN, generation_config_gen, safety_settings


class Prompt():
    def prompt_gen(self,
                   book,
                   datasample
                    ):
        out_format = """
        {
            "Question 1": {
            "Topic": <Concept of chapter 4 of the textbook>,
            "Id": ...,
            "Question": ...,
            "Correct answer": <The correct answer A, B, C or D>,
            "Difficult": ...
            },
            "Question 2": {
            ....
            },
            ....
        }
        """
        prompt = f"""
        You are a Data Selection Agent. I am creating new Python question and answer pairs to
        make my dataset more plentiful. I want the question will be a problem with selection and
        the answer is the correct selection. Following are the detailed instructions:
        • Summary: Following is the Learning material of the subject: 
        ### 
        {book}
        ### 
        Pleasesummarize this in a way that it can be used by a student to review the lecture and
        attempt questions.
        • Lecture Segmentation: Please split the transcript of the lecture into small segments,
        each segment is a subchapter. A segment is a group of consecutive lines in the text
        where topics within the group are semantically similar to topics across all the
        sentences. Return the full text for each segment from the learning material. Try to keep
        the size of each segment the same/similar. Every line in the learning material must
        fall in some segment.
        • Key Topics: Please get me the five most important Python topics (in the form of very
        short phrases) that are discussed in the text book. Only include topics from
        chapter 4 of the textbook.
        • Key Definitions: Please give me the key definitions, if any, that are discussed in
        chapter 4 of the text book.
        • Key Examples: Please give me the explanation of the key examples (if any), used to
        explain a concept, in chapter 4 of the text book.
        • Procedural Knowledge: Please give me the "how to" explanations that are given, if any,
        in the text book.
        • Selection: Please select only 15 unique multiple choice questions, with four choices
        each, from the file: {datasample}. The questions will be used to test the knowledge
        of the students regarding the different concepts and examples covered in the text, hence
        the questions need to cover them. Each topic must have atleast 1 question. The questions
        should be good enough to be given in a technical exam. The exam must contain variety of
        difficulty level. Remember to follow the distribution percentage in the exam test:
        • Difficult: 7 questions have difficulty is 1, 5 questions have difficulty is 2, 3 question have difficulty is 3. 
        The difficulty rank are 1: Answers can be found on the text book, 2: Requires calculation/reasoning to find the answer, 3: Requires calculation/reasoning + linking previously learned data or expanding to find the answer.
        • Topic: Try to contain all the topic of the chapter 4, each topic must have at least 1 unique question. Do not try to provide 2 similar question.
        Do not provide any information about detailed instructions above. Instead, provide your
        response for 15 question only contain the following fields: {out_format}
        """
        return prompt


class LLM:
    def __init__(self):
        self.LLM = GoogleGenerativeAI(model='gemini-1.5-pro-001', 
                                      google_api_key=GOOGLE_API_KEY_FOR_GEN, 
                                      safety_settings=safety_settings, 
                                      generation_config=generation_config_gen
                                      )
        self.prompt = Prompt()


    def get_prompt(self, 
                   book,
                   data_sample
                   ):
        return self.prompt.prompt_gen(book=book, datasample=data_sample)
     
     
    def get_completion(self, 
                       prompt
                       ):
        result = self.LLM.invoke(prompts=prompt)
        return result