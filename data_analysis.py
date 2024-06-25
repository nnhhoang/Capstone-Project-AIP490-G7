import pandas as pd
from question import Question_attribute, Knowledge_attribute
from pydantic import Field
from typing import List, Optional, Sequence, Mapping
import PyPDF2
from tabula import read_pdf
from tabulate import tabulate
import pymupdf
from PyPDF2 import PdfReader
import json

import llm


class Data_analysis():

    #read CSV data
    def read_csv(csv_file_path: str):
        question_bank: List[Question_attribute] = []
        #Question - Selection - Answer - Difficulty - CLO
        datas = pd.read_csv(csv_file_path, quotechar='"')
        # print(datas)
        # add data to question_bank:
        # Index,Question,Answer,LO,Difficult,Name concept,Supchapter,Page,Paragraph,Instruction,Inbook
        for index, data in datas.iterrows():
            new_data = Question_attribute(id = int(data['Index']), 
                                          question = data['Question'], 
                                          answer = data['Answer'],
                                          learning_outcome = data['LO'],
                                          difficulty = data['Difficult'],
                                          instruction = data['Instruction'],
                                          paragraph = {'Z':data['Z'], 'S':data['S'], 'E':data['E']}
                                          )
            for i in range(len(str(data['Name concept']).split(","))):
                new_knowledge = Knowledge_attribute(
                    concept_name = str(data['Name concept']).split(",")[i],
                    contained_book = str(data['Inbook']).split(",")[i],
                    chapter = str(data['Supchapter']).split(",")[i],
                    page = str(data['Page']).split(",")[i]
                )
                new_data.knowledge[i] = new_knowledge
            question_bank.append(new_data)
        return question_bank
        

    #read pdf and update pdf embedding
    def read_pdf(file_path: str):
        pdf_data = ""  # Initialize pdf_data as an empty string
        with open(file_path, "rb") as f:
            pdf = PyPDF2.PdfReader(f)
            for page in pdf.pages:
                new_page = page.extract_text()
                pdf_data = pdf_data + new_page

        return pdf_data


    #read learning outcome from txt file
    def read_CLO(self,
                 file_path: str,
                 ):
        file_path = "Data\CLO.txt"
        clo: Optional[Mapping[str,str]]

        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = str(line).split("\t")
                key, value = parts[1], parts[2] 
                clo[key] = value
        return clo

    #get question with id
    def find_question_with_id(self, 
                              question_id: int, 
                              question_bank: List[Question_attribute]
                              ):
        for question in question_bank:
            if question.id == question_id:
                return question
        return None


    #update new attribute <Need update>
    def update_Data(self, 
                    update_Data: Optional[Sequence[Question_attribute]], 
                    question_bank: List[Question_attribute]
                    ):
        for up_question in update_Data:
            #up_question include: id, intent, knowledge [knowledge name, keyword, concept, page, paragraph]
            question = self.find_question_with_id(up_question.id, question_bank)
            if question == None:
                continue
            question.intent = up_question.intent
            question.knowledge = up_question.knowledge

    def json_instruction(path):
        # Read the JSON data from the file
        with open(path) as f:
            datajson = json.load(f)

        # Create an empty list to store the data for each instruction
        data_list = []

        # Iterate over each item in the datajson list
        for item in datajson:
            # Extract the Id and Instruction values from the item
            instruction_id = int(item['Id'])
            instruction_text = item['Instruction']['Line-by-line explanation']
            if instruction_text == "Not available":
                instruction = "None"
            else: 
                instruction = list(instruction_text.values())
                instruction = ", ".join('"{}"'.format(s) for s in instruction)
                # print(type(instruction_text))

            # Create a dictionary for each instruction and append it to the list
            data_list.append({'Index': instruction_id, 'Instruction': instruction})

        # Convert the data list to a pandas DataFrame
        data_instruction = pd.DataFrame(data_list)
        return data_instruction

    #From similar data to dataframe
    def export_similar_data(self):
        return


    #From attribute data to dataframe
    def export_data_base(self):

        return
