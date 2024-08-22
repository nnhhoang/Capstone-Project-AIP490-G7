from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory
from sentence_transformers import SentenceTransformer

#GOOGLE_API_KEY='x'
GOOGLE_API_KEY= 'x'
# GOOGLE_API_KEY= 'x'
# GOOGLE_API_KEY = 'x'
# GOOGLE_API_KEY='x'

GOOGLE_API_KEY_FOR_GEN = 'x'

generation_config_gen = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain"
        }

cache_time = 160

generation_config = {
        "temperature": 0,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "application/json"
        }

safety_settings = {
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}

MODEL_SEMANTIC = SentenceTransformer('all-MiniLM-L6-v2')
# MODEL_SEMANTIC = SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True)
# MODEL_SEMANTIC = SentenceTransformer('microsoft/codebert-base')

Test_set_path = r'data\question_data3.csv'

cache_cost_per_token = 0.875/1000000
cache_cost_per_hour = (4.5/1000000)/60
token_in = 3.5/1000000
token_out = 10.5/1000000
