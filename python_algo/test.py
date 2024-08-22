
from gemini_api import LLM, Prompt



gemini_call = LLM()

s = gemini_call.haha(1)
answer = gemini_call.get_prompt(1, 'what is cat', 'None')