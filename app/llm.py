from openai import AsyncOpenAI
from typing import Tuple
from tenacity import retry, stop_after_attempt, wait_fixed

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


@retry(stop=stop_after_attempt(5), wait=wait_fixed(2))
async def safe_get_response_llm(
    client: AsyncOpenAI, prompt: str, original_prompt: str
):
    return await get_response_llm(client, prompt, original_prompt)


async def get_response_llm(
    client: AsyncOpenAI, system_prompt: str, user_prompt: str
) -> Tuple[str, int]:
    response = await client.chat.completions.create(
        model="gemma3:1b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
    )

    content = str(response.choices[0].message.content.strip())
    logger.debug(f"LLM Raw Content Response: {content=}\n")
    token_count = response.usage.total_tokens
    return content, token_count
