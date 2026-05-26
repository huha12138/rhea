"""Judge-model generation helpers for local and API-hosted chat models."""

import os
from time import time
from typing import Dict, List, Optional, Tuple

import strictfire
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_LOCAL_JUDGE = os.getenv(
    "RHEA_LOCAL_JUDGE_MODEL", "Qwen/Qwen2.5-32B-Instruct"
)
API_MODEL_PREFIXES = ("chatgpt", "gpt-", "gpt4", "o1", "o3", "o4")

_LOCAL_MODEL_NAME: Optional[str] = None
_LOCAL_TOKENIZER = None
_LOCAL_MODEL = None
_OPENAI_CLIENT = None


def _is_api_model(model_name: str) -> bool:
    return model_name.lower().startswith(API_MODEL_PREFIXES)


def _prepare_messages(
    prompt: str, messages: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    return [{"role": "user", "content": prompt}] if prompt else messages


def _get_openai_client():
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        from openai import OpenAI

        kwargs = {}
        base_url = os.getenv("OPENAI_BASE_URL")
        if base_url:
            kwargs["base_url"] = base_url
        _OPENAI_CLIENT = OpenAI(**kwargs)
    return _OPENAI_CLIENT


def _load_local_model(model_name: str):
    global _LOCAL_MODEL_NAME, _LOCAL_TOKENIZER, _LOCAL_MODEL
    if _LOCAL_MODEL is not None and _LOCAL_MODEL_NAME == model_name:
        return _LOCAL_TOKENIZER, _LOCAL_MODEL

    print("evaluate_model_name:", model_name)
    _LOCAL_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
    _LOCAL_MODEL = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    _LOCAL_MODEL_NAME = model_name
    return _LOCAL_TOKENIZER, _LOCAL_MODEL


def _generate_with_api(
    model_name: str,
    prompt: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    end_tokens: List[str],
) -> Tuple[str, int, float]:
    start_time = time()
    completion = _get_openai_client().chat.completions.create(
        model=model_name,
        messages=_prepare_messages(prompt, messages),
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        stop=end_tokens or None,
    )
    used_time = time() - start_time

    usage = completion.usage
    prompt_len = usage.prompt_tokens if usage else 0
    num_output_tokens = usage.completion_tokens if usage else 0
    content = completion.choices[0].message.content or ""
    token_per_second = num_output_tokens / used_time if used_time else 0
    return content, prompt_len, token_per_second


def _generate_with_local_model(
    model_name: str,
    prompt: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
) -> Tuple[str, int, float]:
    tokenizer, model = _load_local_model(model_name)
    text = tokenizer.apply_chat_template(
        _prepare_messages(prompt, messages),
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    start_time = time()
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_tokens,
        do_sample=False,
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
    used_time = time() - start_time

    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    content = tokenizer.decode(
        output_ids[index:], skip_special_tokens=True
    ).strip("\n")
    prompt_len = model_inputs.input_ids.shape[1]
    token_per_second = len(output_ids) / used_time if used_time else 0
    return content, prompt_len, token_per_second


def generate(
    model_name: str = DEFAULT_LOCAL_JUDGE,
    prompt: str = "",
    messages: List[Dict[str, str]] = [],
    print_prompt: bool = False,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 1,
    seed: int = 111,
    end_tokens: List[str] = [],
):
    prompt = prompt.replace("\\n", "\n")
    if print_prompt:
        print("Prompt:", prompt)
        print("=" * 50)

    if _is_api_model(model_name):
        return _generate_with_api(
            model_name=model_name,
            prompt=prompt,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            end_tokens=end_tokens,
        )

    return _generate_with_local_model(
        model_name=model_name,
        prompt=prompt,
        messages=messages,
        max_tokens=max_tokens,
    )


def main(
    model_name: str = DEFAULT_LOCAL_JUDGE,
    prompt: str = "",
    print_prompt: bool = False,
    max_tokens: int = 128,
    temperature: float = 1,
    top_p: float = 1,
):
    print(
        generate(
            model_name=model_name,
            prompt=prompt,
            print_prompt=print_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
    )


if __name__ == "__main__":
    strictfire.StrictFire(main)
