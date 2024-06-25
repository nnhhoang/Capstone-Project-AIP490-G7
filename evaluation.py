from typing import List
from ragas import evaluate
from ragas.metrics import (answer_similarity, context_recall, context_precision)
import pandas as pd
from datasets import Dataset 

from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from sentence_transformers import SentenceTransformer

from question import Question_attribute
from data_analysis import Data_analysis
from config import GOOGLE_API_KEY



class Evaluation_class():
    def __init__(self) -> None:
        pass
    
    def semantic_search(new_input,
                    database
                    ):
        embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # Corpus with example sentences
        corpus = database
        # Use "convert_to_tensor=True" to keep the tensors on GPU (if available)
        corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

        # Query sentences:
        query = new_input

        query_embedding = embedder.encode(query, convert_to_tensor=True)

        similarity_scores = embedder.similarity(query_embedding, corpus_embeddings)[0]

        return similarity_scores.item()

    def eval_instruction(self,
                         prompt_data,
                         data_test
                        ):
        index = data_test['Index']
        data = pd.DataFrame({
            'question': 0,
            'answer': 0,
            'ground_truth': 0,
            'similarity_score': 0
        })
        data.index = index

        for index, answer in prompt_data.iterrows():
            question: Question_attribute = Data_analysis.find_question_with_id(index,data_test)
            if question == None:
                continue
            data.at[index,'question'] = question.question
            data.at[index,'answer'] = answer
            data.at[index,'ground_truth'] = question.instruction

        for i in range(len(data)):
            data.at[i, 'similarity_score'] = self.semantic_search(data.at[i,'answer'],data.at[i,'ground_truth'])

        for index, data_row in data.iterrows():
            if data_row['answer']=="None" and data_row['ground_truth']=="None":
                data.at[index, 'similarity_score'] = 1
                continue
            if data_row['answer']=="None" or data_row['ground_truth']=="None":
                data.at[index, 'similarity_score'] = 0
        return sum(data['similarity_score'])/len(data)


    def eval_instruction_2(prompt_data,
                         data_test
                         ):
        index = data_test['Index']
        data = pd.DataFrame({
            'question': 0,
            'answer': 0,
            'contexts': 0,
            'ground_truth': 0,
        })
        data.index = index

        for index, answer in prompt_data.iterrows():
            question: Question_attribute = Data_analysis.find_question_with_id(index,data_test)
            if question == None:
                continue
            data.at[index,'question'] = question.question
            data.at[index,'answer'] = answer
            data.at[index,'contexts'] = question.paragraph
            data.at[index,'ground_truth'] = question.instruction

        contexts_list = data["contexts"].tolist()
        contexts_list = [str(context).split(', ') for context in contexts_list]
        data["contexts"] = contexts_list
        dataset = Dataset.from_dict(data[['question','ground_truth','answer','contexts']])

        langchain_llm = GoogleGenerativeAI(google_api_key=GOOGLE_API_KEY, model='models/gemini-1.5-pro-latest')
        langchain_embeddings = GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY, model='models/text-embedding-004')

        llm = LangchainLLMWrapper(langchain_llm)
        embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)

        score = evaluate(
            dataset=dataset,
            metrics=[answer_similarity],
            column_map={'ground_truth': 'ground_truth', 'answer': 'answer'},
            llm= llm,
            embeddings= embeddings
        )
        score_df = score.to_pandas()
        similar_score = pd.DataFrame(score_df)
        for index, data in similar_score.iterrows():
            if data['answer']=="None" and data['ground_truth']=="None":
                similar_score.at[index, 'answer_similarity'] = 1
                continue
            if data['answer']=="None" or data['ground_truth']=="None":
                similar_score.at[index, 'answer_similarity'] = 0

        similar_score.index = index

        return similar_score