import logging
import os
from typing import List

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def get_logit_bias(targets: List[int], penalty=0.5) -> dict:
    # https://platform.openai.com/docs/api-reference/chat/create#chat-create-logit_bias
    # Create a logit bias to encourage the model to talk about the targets
    logit_bias = {target: penalty for target in targets}
    return logit_bias


def complete(
    content: str,
    history: List[dict] = [],
    model="gpt-3.5-turbo",
    system_message="",
    logit_bias=None,  # dictionary {2435: 0.7, 333: 100}
    return_raw_completion=False,
    **kwargs,
):
    if logit_bias:
        logger.warning("Logit bias is not supported yet")
        logit_bias = None

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            *history,
            {"role": "user", "content": content},
        ],
        logit_bias=logit_bias,
    )

    message = completion.choices[0].message.content

    if return_raw_completion:
        return message, completion

    return message


if __name__ == "__main__":
    # Example usage
    content = "What is the meaning of life?"
    history = [
        {"role": "system", "content": "The meaning of life is 42."},
        {"role": "user", "content": "What is the meaning of life?"},
    ]
    model = "llama3-8b-8192"
    system_message = "The meaning of life is 42."
    logit_bias = get_logit_bias([2435, 333])
    return_raw_completion = False

    completion = complete(
        content,
        history,
        model,
        system_message,
        logit_bias=logit_bias,
        return_raw_completion=False,
    )
    print(completion)
