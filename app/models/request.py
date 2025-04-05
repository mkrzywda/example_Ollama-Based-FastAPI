from pydantic import BaseModel
from typing import List


class OllamaRequest(BaseModel):
    prompt: str


class OllamaResponse(BaseModel):
    ollama_response: str

