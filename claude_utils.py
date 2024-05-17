import logging
from typing import List

import anthropic
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


client = anthropic.Anthropic()


def get_logit_bias(targets: List[int], penalty=0.5) -> dict:
    # Create a logit bias to encourage the model to talk about the targets
    logit_bias = {target: penalty for target in targets}
    return logit_bias


def complete(
    content: str,
    history: List[dict] = [],
    model="claude-3-opus-20240229",
    system_message="",
    logit_bias=None,  # dictionary {2435: 0.7, 333: 100}
    return_raw_completion=False,
    max_tokens=1000,
    temperature=0.0,
    **kwargs,
):
    if logit_bias:
        logger.warning("Logit bias is not supported yet")
        logit_bias = None

    messages = history + [{"role": "user", "content": content}]

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_message,
        messages=messages,
    )

    if return_raw_completion:
        return response

    return response.content[0].text


if __name__ == "__main__":
    # Example usage
    content = "What is the meaning of life?"
    history = [
        {"role": "user", "content": "What is the meaning of life?"},
        {"role": "assistant", "content": "The meaning of life is 42."},
    ]
    model = "claude-3-opus-20240229"
    system_message = "The meaning of life is 42."
    logit_bias = get_logit_bias([2435, 333])
    return_raw_completion = False

    completion = complete(
        content,
        history,
        model,
        system_message,
        logit_bias,
        return_raw_completion=False,
    )
    print(completion)
