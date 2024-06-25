from pydantic import BaseModel, Field
from typing import Optional, Mapping, List


class Knowledge_attribute(BaseModel):
    concept_name: Optional[str] = Field(default=None)
    contained_book: Optional[str] = Field(default=True)
    chapter: Optional[str] = Field(default=None)
    page: Optional[str] = Field(default=None)

class Question_attribute(BaseModel):
    id: Optional[int] = Field(default=None)
    question: Optional[str] = Field(default=None)
    answer: Optional[str] = Field(default=None)
    knowledge: Optional[Mapping[str,Knowledge_attribute]] = Field(default={})
    difficulty: Optional[str] = Field(default="Easy")
    learning_outcome: Optional[str] = Field(default=None)
    instruction: Optional[str] = Field(default=None)
    paragraph: Optional[str] = Field(default=None)
    paragraph_map: Optional[Mapping[str:str]] = Field(default={'Z':"0", 'S':"0", 'E':"0"})