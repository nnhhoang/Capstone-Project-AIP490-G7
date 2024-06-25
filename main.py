import asyncio
from typing import List
from data_analysis import Data_analysis
import question
import llm
import os


LLM_instance = llm.LLM()
data_attribute: List[question.Question_attribute] = Data_analysis.read_csv(csv_file_path=r"Data\Unit3 - Sheet1.csv")
# pdf_document, table_content = Data_analysis.read_pdf(file_path="Data\pythonlearn.pdf")
# data_predicted: List[question.Question_attribute]
# with open("Data\document.txt", "w", encoding="utf-8") as f:
#     f.write(pdf_document)

# file = open("Data\content.txt", "r", encoding="utf-8")
# table_content = file.readlines()
# print(table_content)

# name = LLM_instance.upload_pdf("Data\document.txt")
# print(name)

# print(type(pdf_document))
# raw_text = Data_analysis.get_pdf_text("Data\pythonlearn.pdf")
# for question_attribute in data_attribute:
#     prompt_out =LLM_instance.get_completion(quesion_template=question_attribute)
#     print(prompt_out)
# for ques in data_attribute:
#     print(ques,"\n")

# print(table_content,"\n")

