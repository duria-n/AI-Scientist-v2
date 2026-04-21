import json
import logging
import os
import time

from .utils import FunctionSpec, OutputType, opt_messages_to_list, backoff_create
from ai_scientist.utils.network import normalize_httpx_proxy_env
from funcy import notnone, once, select_values
import openai
from rich import print

logger = logging.getLogger("ai-scientist")


OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


DEEPSEEK_MODEL_ALIASES = {
    "deepseek-coder-v2-0724": "deepseek-coder",
}


def _is_deepseek_model(model: str) -> bool:
    return model.startswith("deepseek-") or model in DEEPSEEK_MODEL_ALIASES


def _resolve_deepseek_model(model: str) -> str:
    return DEEPSEEK_MODEL_ALIASES.get(model, model)


def get_ai_client(model: str, max_retries=2) -> openai.OpenAI:
    normalize_httpx_proxy_env()
    if model.startswith("ollama/"):
        client = openai.OpenAI(
            base_url="http://localhost:11434/v1", 
            max_retries=max_retries
        )
    elif _is_deepseek_model(model):
        client = openai.OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
            max_retries=max_retries,
        )
    else:
        client = openai.OpenAI(max_retries=max_retries)
    return client


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    client = get_ai_client(model_kwargs.get("model"), max_retries=0)
    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore

    messages = opt_messages_to_list(system_message, user_message)

    if func_spec is not None:
        filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]
        # force the model to use the function
        filtered_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict

    if filtered_kwargs.get("model", "").startswith("ollama/"):
       filtered_kwargs["model"] = filtered_kwargs["model"].replace("ollama/", "")
    elif _is_deepseek_model(filtered_kwargs.get("model", "")):
        filtered_kwargs["model"] = _resolve_deepseek_model(filtered_kwargs["model"])

    t0 = time.time()
    completion = backoff_create(
        client.chat.completions.create,
        OPENAI_TIMEOUT_EXCEPTIONS,
        messages=messages,
        **filtered_kwargs,
    )
    req_time = time.time() - t0

    choice = completion.choices[0]

    if func_spec is None:
        output = choice.message.content
    else:
        assert (
            choice.message.tool_calls
        ), f"function_call is empty, it is not a function call: {choice.message}"
        assert (
            choice.message.tool_calls[0].function.name == func_spec.name
        ), "Function name mismatch"
        try:
            print(f"[cyan]Raw func call response: {choice}[/cyan]")
            output = json.loads(choice.message.tool_calls[0].function.arguments)
        except json.JSONDecodeError as e:
            logger.error(
                f"Error decoding the function arguments: {choice.message.tool_calls[0].function.arguments}"
            )
            raise e

    in_tokens = completion.usage.prompt_tokens
    out_tokens = completion.usage.completion_tokens

    info = {
        "system_fingerprint": completion.system_fingerprint,
        "model": completion.model,
        "created": completion.created,
    }

    return output, req_time, in_tokens, out_tokens, info
