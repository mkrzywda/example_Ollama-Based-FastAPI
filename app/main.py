from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from typing_extensions import Annotated
from openai import AsyncOpenAI
import time
from prompts import generate_prompt
from models.request import OllamaRequest, OllamaResponse
from llm import safe_get_response_llm
import os
import secrets

import logging
from logging.handlers import TimedRotatingFileHandler

logger = logging.getLogger("ollama_based_api")
log_file = "ollama_based_api.log"
handler = TimedRotatingFileHandler(
    filename=log_file, when="W0", interval=1, encoding="utf-8"
)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[handler, logging.StreamHandler()],
)

app = FastAPI()
security = HTTPBasic()

VALID_USERNAME = os.getenv("username")
VALID_PASSWORD = os.getenv("password")

if not VALID_USERNAME or not VALID_PASSWORD:
    raise RuntimeError("Environment variables username and password must be set.")


def verify_credentials(credentials: Annotated[HTTPBasicCredentials, Depends(security)]):
    current_username_bytes = credentials.username.encode("utf8")
    correct_username_bytes = VALID_USERNAME.encode("utf8")
    current_password_bytes = credentials.password.encode("utf8")
    correct_password_bytes = VALID_PASSWORD.encode("utf8")

    is_correct_username = secrets.compare_digest(
        current_username_bytes, correct_username_bytes
    )
    is_correct_password = secrets.compare_digest(
        current_password_bytes, correct_password_bytes
    )

    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


async def get_client():
    client = AsyncOpenAI(base_url="http://llm-ollama-deepseek:11434/v1/", api_key="ollama")
    try:
        yield client
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="A server connection error occurred. Please attempt your request again later.",
        )


@app.post("/generate", response_model=OllamaResponse)
async def generate(
    request: OllamaRequest,
    username: Annotated[str, Depends(verify_credentials)],
    client: AsyncOpenAI = Depends(get_client),
) -> OllamaResponse | None:
    start_time = time.time()
    logger.info(
        f"Starting generate endpoint for prompt: {request.prompt} by user: {username}"
    )
    user_prompt = f"Based on user query and context provided in System prompt generate: {request.prompt}"

    
    llm_response, token_count = await safe_get_response_llm(
            client, generate_prompt, user_prompt
    )

    elapsed_time_ms = (time.time() - start_time) * 1000
    elapsed_time_s = elapsed_time_ms / 1000
    tokens_per_second = (
            token_count / elapsed_time_s
            if elapsed_time_s > 0 and token_count > 0
            else 0
    )

    logger.info(
            f"generate response completed successfully in {elapsed_time_ms:.0f} ms, "
            f"processed {token_count} tokens, {tokens_per_second:.1f} tokens/second"
    )
    return OllamaResponse(ollama_response=llm_response)