import logging
import os
from typing import List

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def complete(
    content: str,
    history: List[dict] = [],  # Not allowed yet,
    model="gemini-pro",
    system_message="",
    logit_bias=None,  # dictionary {2435: 0.7, 333: 100}
    return_raw_completion=False,
    temperature=None,
    **kwargs,
):
    if history:
        logger.warning("History is not supported yet")

    if system_message:
        logger.warning("System message is not supported yet")

    if logit_bias:
        logger.warning("Logit bias is not supported yet")

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])

    model = genai.GenerativeModel(model_name=model)

    config = genai.GenerationConfig(
        candidate_count=1,  # Only 1 is supported for now
        temperature=temperature,
    )
    completion = model.generate_content(
        content,
        generation_config=config,
    )

    if return_raw_completion:
        return completion

    return completion.text


if __name__ == "__main__":
    # Example usage
    content = "What is the meaning of life?"
    return_raw_completion = False

    completion = complete(
        content,
        return_raw_completion=False,
    )
    print(completion)
