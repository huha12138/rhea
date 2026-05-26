import os
import json
import torch
import numpy as np
import importlib.util

from rank_bm25 import BM25Okapi
import re

from copy import deepcopy
from transformers import (
    GenerationConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteriaList,
    StoppingCriteria,
    AutoModel
)
from datasets import load_from_disk
from strictfire import StrictFire
from tqdm import tqdm

from utils.misc import get_logger, config
from utils.qwen3 import generate
from utils.constants import INFERENCE_OUTPUT
from typing import Dict, Any, List, Literal
from time import time

package_name = "flash_attn"
spec = importlib.util.find_spec(package_name)
FLASH_AVAILABLE = True
if spec is None:
    print("no flash_attn")
    FLASH_AVAILABLE = False

SEED = 111
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, tokenizer, stops=[]):
        super().__init__()
        self.tokenizer = tokenizer
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        tokens = self.tokenizer.decode(input_ids[0][-15:])
        return any([stop in tokens[-len(stop) :] for stop in self.stops])

def llama_generate(
    prompt,
    model,
    tokenizer,
    debug: bool = False,
    end_tokens: List[str] = [],
    **kwargs,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    if tokenizer.eos_token:
        end_tokens.append(tokenizer.eos_token)
    stopping_criteria = StoppingCriteriaSub(tokenizer, end_tokens)
    if not debug:
        input_ids = input_ids.to(DEVICE)
    if debug:
        output = "some dummy text."
        return output, input_ids.shape[1]
    else:
        with torch.no_grad():
            start_time = time()
            generation_output = model.generate(
                input_ids=input_ids,
                return_dict_in_generate=True,
                output_scores=True,
                stopping_criteria=StoppingCriteriaList([stopping_criteria]),
                **kwargs,
            )
            used_time = time() - start_time
    s = generation_output.sequences[0]
    output_tokens = s[input_ids.shape[1] :]
    num_output_tokens = len(output_tokens)
    output = tokenizer.decode(output_tokens)
    for stop_token in end_tokens:
        output = output.replace(stop_token, "")
    return output, input_ids.shape[1], num_output_tokens / used_time

def get_prompt_from_docs(docs, question, system_message=None, only_user=False, model="mistral"):
    messages = docs + [question]
    seps = [" ", "</s>"]
    if model == "mistral":
        roles=("[INST]", "[/INST]")
    elif model == "longalpaca":
        roles=("### Instruction:\n", "### Response:\n")
    elif model == "memochat":
        roles=("### User:", "### Assistant:")
    
    if system_message:
        ret = f"{roles[0]} {system_message}\n"
    else:
        ret = f"{roles[0]} "

    for i, message in enumerate(messages):
        if only_user and i % 2 == 1:
            message = " "
        tag = roles[i % 2]
        if i == 0:
            ret += message + " "
        else:
            ret += tag + " " + message + seps[i % 2]

    ret += roles[1]
    return ret

def retrieve_top5_qa_pairs(question, docs):
    """Retrieve the five most relevant QA pairs while preserving dialogue order."""
    qa_pairs = []
    qa_indices = []
    for i in range(0, len(docs) - 1, 2):
        qa_text = docs[i] + " " + docs[i + 1]
        qa_pairs.append(qa_text)
        qa_indices.append(i)

    tokenized_corpus = [re.findall(r'\w+', doc.lower()) for doc in qa_pairs]
    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = re.findall(r'\w+', question.lower())
    scores = bm25.get_scores(tokenized_query)
    print("Scores:", scores)
    top5_qa_indices = sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)[:5]

    # Restore the retrieved QA pairs to their original dialogue order.
    top5_original_indices = sorted([qa_indices[i] for i in top5_qa_indices])

    result = []
    for i in top5_original_indices:
        result.extend(docs[i:i + 2])
    return result

def main(
    model_name: str,
    task_name: Literal[
        "refinement_single",
        "refinement_multi",
        "expansion_single",
        "expansion_multi",
        "follow-up_single",
        "follow-up_multi",
        "recollection_single_cls",
        "recollection_multi_cls",
        "recollection_single_global-inst",
        "recollection_multi_global-inst",
        "cls_ablation_gold",
        "cls_ablation_dgc",
        "cls_ablation_sgc",
        "cls_ablation_rc",
    ],
    rhea: bool = False,    
    first_end: bool = False,   
    only_assitant: bool = False,  
    w_retieval_1: bool = False,
    attention: bool = False,
    abolation:int = 10,
    valid: bool = False,
    w_retieval_2: bool = False,
    only_user: bool = False, 
    only_reply: bool = False, 
    use_bm25: bool = False,
    TopK: int = 5,
    recent_topk: bool = False,
    summary: bool = False,
    query_correlation: bool = False,
    longalpaca: bool = False,
    memochat: bool = False,
    mode:int = 0, 
    tl:float = 0.5, 
    th:float = 0.8,  
    abandan: bool = False,
    preserve: bool = False,
    conv_key: str = "conv",
    system_message: str = "You are a helpful, respectful and honest assistant.",
    output_key: str = "gen_resp",
    load_8bit: bool = False,
    temperature: float = 1.0,
    top_p: float = 1,
    top_k: int = 50,
    do_sample: bool = False,
    max_new_tokens: int = 1024,
    load_model_args: Dict[str, Any] = {},
    end_tokens: List[str] = [],
    resume: bool = False,
    use_gold_history: bool = False,
    n_forward: int = -1,
):

    print("abandan:",abandan)
    print("preserve:",preserve)
    # Build the output path from the task family and subtype.
    print(model_name)
    task_type, task_subtype = task_name.split("_", 1)
    if "ablation" in task_name:
        task_type = "_".join(task_name.split("_")[:2])
        task_subtype = "_".join(task_name.split("_")[2:])
        out_filename = os.path.join(
            INFERENCE_OUTPUT,
            task_type,
            f"{task_subtype}_{model_name}.jsonl",
        )
    elif use_gold_history and "ablation" not in task_name:
        out_filename = os.path.join(
            INFERENCE_OUTPUT,
            task_type,
            f"{task_subtype}_gold_{model_name}.jsonl",
        )
    else:
        out_filename = os.path.join(
            INFERENCE_OUTPUT,
            task_type,
            f"{task_subtype}_{model_name}.jsonl",
        )
    logger = get_logger(
        name=__name__,
        console_level="info",
        file_level="debug",
        log_path=os.path.join(
            "log",
            f"{task_name}_{model_name}.log",
        ),
        maxBytes=10000000,
    )

    if not end_tokens:
        end_tokens = config[model_name]["end_tokens"]
        logger.info(f"Changed end_tokens to {end_tokens}")

    if "ablation" in task_name:
        data = [
            json.loads(row)
            for row in open(os.path.join("data", f"{task_name}.jsonl"))
        ]
    else:
        print(task_name)
        data = load_from_disk(
            f"data/MTEval/{task_name}/test"
        ).to_list()

    if valid:
        print(task_name)

        if task_name in ["refinement_single", "refinement_multi"]:
            file_name = "refinement.json"
        elif task_name in ["follow-up_single", "follow-up_multi"]:
            file_name = "follow-up.json"
        elif task_name in ["recollection_single_cls", "recollection_multi_cls"]:
            file_name = "recollection-cls.json"
        elif task_name in ["recollection_single_global-inst", "recollection_multi_global-inst"]:
            file_name = "recollection-global-inst.json"
        elif task_name == "expansion_single":
            file_name = "expansion_1.json" 
        elif task_name == "expansion_multi":
            file_name = "expansion.json"
        else:
            raise ValueError(f"No validation label mapping is configured for `{task_name}`.")

        file_path = f'data/MTEval/result/{file_name}'

        with open(file_path, 'r', encoding='utf-8') as file:
            labels = json.load(file)


    print(data[0].keys())
    print(len(data))

    out_data = []
    if out_filename and os.path.exists(out_filename):
        out_data = [json.loads(line) for line in open(out_filename)]

    print_first_prompt = False
    total_forward = sum(
        turn["do_inference"] for row in data for turn in row[conv_key]
    )

    if resume and out_data:
        matched = 0
        ori_row_map = {
            f"{row['id']}#{turn['id']}": turn
            for row in data
            for turn in row[conv_key]
        }
        for row in out_data:
            for turn in row[conv_key]:
                if not turn["do_inference"]:
                    continue
                _key = f"{row['id']}#{turn['id']}"
                if output_key in turn and _key in ori_row_map:
                    ori_row_map[_key][output_key] = turn[output_key]
                    matched += 1
        if total_forward == matched:
            print(f"{out_filename} has finished.")
            return

        logger.info(f"Resumed {matched} instances from {out_filename}.")

    os.makedirs(os.path.dirname(out_filename), exist_ok=True)
    use_openai = False
    model_path = config[model_name]["path"]
    if "gpt-3.5" in model_name or "gpt-4" in model_name:
        use_openai = True
    elif rhea:
        model = AutoModel.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).to('cuda')
    else:
        use_flash = config[model_name]["use_flash_attn"] and FLASH_AVAILABLE
        if use_flash:
            logger.info("Using flash attention2.")
            
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if use_flash else None,
            **load_model_args,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        tokenizer.padding_side = "left"

        try:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        except AttributeError as e:
            logger.info(e)
            logger.info(
                "Can't set the tokenizer.pad_token_id but it's probably"
                " ok if the model is chatglm."
            )

    generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )    

    logger.info(f"loaded model `{model_name}`")
    if n_forward > 0:
        logger.info(f"Only inference on the first {n_forward} examples.")
        pbar = tqdm(total=n_forward)
    else:
        pbar = tqdm(total=total_forward)

    pbar.set_description(f"Inferencing {out_filename}")
    token_per_second_list = []

    for i, row in enumerate(data):
        # Create the model-specific conversation prompt.
        if use_openai:
            conv = deepcopy(config["gpt-4"]["chat_template"])
        else:
            conv = deepcopy(config[model_name]["chat_template"])
            
        docs = []
        docs2 = []
        if system_message:
            conv.set_system_message(system_message)
            docs.append(system_message)
        for j, turn in enumerate(row[conv_key]):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            conv.append_message(conv.roles[0], turn["user"])
            conv.append_message(conv.roles[1], turn["sys"])

            if not turn["do_inference"]:
                pbar.update(1)
                # Some tasks include context-only turns that do not require inference.
                docs.append(turn["user"])
                docs.append(turn["sys"])
                docs2.append(turn["user"])
                docs2.append(turn["sys"])
                continue

            if resume and output_key in turn:
                pbar.update(1)
                if not use_gold_history:
                    conv.update_last_message(turn[output_key])
                continue
            conv.update_last_message(None)
            
            question = turn["user"]

            if first_end:
                prompt = get_prompt_from_docs(docs2,question,system_message,only_user)
            elif only_user:
                # Keep only user instructions in the serialized history.
                system_message = docs[0]
                docs3 = docs[1:]
                prompt = get_prompt_from_docs(docs3,question,system_message,only_user)
            elif only_reply:
                # Mask historical user instructions and keep assistant replies.
                system_message = docs[0]
                for i, _ in enumerate(docs):
                    if i%2 == 1:
                        docs[i] = " "
                docs3 = docs[1:]
                prompt = get_prompt_from_docs(docs3,question,system_message,only_user)
            elif use_bm25:
                # Retrieve the most relevant dialogue history with BM25.
                system_message = docs[0]
                if len(docs)>10:
                    docs3 = retrieve_top5_qa_pairs(question, docs[1:])
                else:
                    docs3 = docs[1:]
                prompt = get_prompt_from_docs(docs3,question,system_message,only_user)
            elif recent_topk:
                system_message = docs[0]
                k = max(1,len(docs)-10)
                docs3 = docs[k:]
                prompt = get_prompt_from_docs(docs3,question,system_message,only_user)
            elif summary:
                system_message = docs[0]
                docs3 = docs[1:]
                sum_query_1 = f"""Given the above conversation history, distill the key points from the dialogue that are directly relevant to answering the query. Return a concise paragraph summarizing, so model can use it as complete context without re-reading the full history."""
                
                sum_query_2 = f"""Given the above conversation history and the following user's query: \n 
                            {question}  \n
                            Distill the key points from the dialogue that are directly relevant to answering the query. Return a concise paragraph summarizing, so model can use it as complete context without re-reading the full history."""
                sum_query = sum_query_2 if query_correlation else sum_query_1
                prompt = get_prompt_from_docs(docs3, sum_query, system_message, only_user)
            elif longalpaca:
                system_message = docs[0]
                docs3 = docs[1:]
                prompt = get_prompt_from_docs(docs3, question, system_message, only_user, "longalpaca")
            elif memochat:
                system_message = docs[0]
                docs3 = docs[1:]
                prompt = get_prompt_from_docs(docs3, question, system_message, only_user, "memochat")
            else:
                prompt = conv.get_prompt()

            error_occurred = False

            if use_openai:
                resp, prompt_len, token_per_second = generate(
                    model_name=model_name,
                    prompt="",
                    messages=conv.to_openai_api_messages(),
                    temperature=temperature,
                    top_p=top_p,
                    end_tokens=end_tokens,
                    max_tokens=max_new_tokens,
                )
                token_per_second_list.append(token_per_second)

            elif rhea:
                try:
                    # Use the configured Rhea retrieval/generation variant.
                    st = time()
                    if only_assitant:
                        resps,prompts = model.generate_from_text1(documents=[docs], questions=[question], generation_config=generation_config)
                        resp = resps[0]
                        prompt = prompts[0]
                    elif w_retieval_1:
                        resps,prompts = model.generate_from_text_w_Retrieval(documents=[docs], questions=[question], mode=mode, generation_config=generation_config)
                        resp = resps[0]
                        prompt = prompts[0]
                    elif w_retieval_2:
                        with torch.no_grad():
                            resps,prompts = model.generate_from_text_w_Retrieval_8(documents=[docs], questions=[question], mode=mode, tl=tl, th=th, generation_config=generation_config)
                        resp = resps[0]
                        prompt = prompts[0]
                    elif abolation in [11, 12, 13]:
                        with torch.no_grad():
                            if abolation == 11:
                                resps,prompts = model.generate_from_text_w_Retrieval_11(documents=[docs], questions=[question], mode=mode, tl=tl, th=th, generation_config=generation_config)
                            elif abolation == 12:
                                resps,prompts = model.generate_from_text_w_Retrieval_12(documents=[docs], questions=[question], mode=mode, tl=tl, th=th, generation_config=generation_config)
                            elif abolation == 13:
                                resps,prompts = model.generate_from_text_w_Retrieval_13(documents=[docs], questions=[question], mode=mode, tl=tl, th=th, generation_config=generation_config)
                        resp = resps[0]
                        prompt = prompts[0]
                    elif abandan or preserve:
                        print("abandan:",abandan)
                        print("preserve:",preserve)
                        with torch.no_grad():
                            resps,prompts = model.generate_from_text_w_Retrieval_9(documents=[docs], questions=[question], mode=mode, abandan=abandan, preserve=preserve, generation_config=generation_config)
                        resp = resps[0]
                        prompt = prompts[0]
                    elif valid:
                        label = labels[i][j].get("label")
                        resps,prompts = model.generate_from_text_w_Retrieval_label(documents=[docs], questions=[question], label=label, generation_config=generation_config)
                        resp = resps[0]
                        prompt = prompts[0]

                    used_time = time() - st
                    prompt_len = 10
                    token_per_second = 1
                    token_per_second_list.append(token_per_second)
                except Exception as e:
                    error_occurred = True
                    logger.exception(f"Error occurred at {i}.")

            else:
                if summary:
                    st = time()
                    resp, prompt_len, token_per_second = llama_generate(
                        prompt=prompt,
                        model=model,
                        tokenizer=tokenizer,
                        generation_config=generation_config,
                        end_tokens=end_tokens,
                    )
                    # Rebuild the final prompt from the generated summary.
                    docs4 = [resp]
                    prompt = get_prompt_from_docs(docs4,question,system_message,only_user)

                    resp, prompt_len, token_per_second = llama_generate(
                        prompt=prompt,
                        model=model,
                        tokenizer=tokenizer,
                        generation_config=generation_config,
                        end_tokens=end_tokens,
                    )

                    used_time = time() - st
                    token_per_second_list.append(token_per_second)
                else:
                    st = time()
                    resp, prompt_len, token_per_second = llama_generate(
                        prompt=prompt,
                        model=model,
                        tokenizer=tokenizer,
                        generation_config=generation_config,
                        end_tokens=end_tokens,
                    )
                    used_time = time() - st
                    token_per_second_list.append(token_per_second)

            # Update progress and persist generation metadata.
            if not print_first_prompt and not error_occurred:
                tqdm.write(prompt)
                tqdm.write(resp)
                
            turn["error"] = error_occurred
            pbar.set_postfix({"t/s": f"{np.mean(token_per_second_list):.2f}"})
            pbar.update(1)

            
            if first_end:
                if j <= 1:
                    docs2.append(question)
                    if use_gold_history:
                        docs2.append(turn["sys"])
                    else:
                        docs2.append(resp)
                else:
                    docs2[-2] = question
                    if use_gold_history:
                        docs2[-1] = turn["sys"]
                    else:
                        docs2[-1] = resp
                    assert len(docs2) == 4 or len(docs2) == 6, len(docs2)

            docs.append(question)
            if use_gold_history:
                conv.update_last_message(turn["sys"])
                docs.append(turn["sys"])
            else:
                conv.update_last_message(resp)
                docs.append(resp)

            if not error_occurred:
                turn["prompt"] = prompt
                turn["prompt_len"] = prompt_len
                turn[output_key] = resp
                turn['time'] = round(used_time, 2)

            if i % 10 == 0:
                with open(out_filename, "w", encoding="utf-8") as f:
                    f.write(
                        "\n".join([
                            json.dumps(row, ensure_ascii=False) for row in data
                        ])
                    )
                logger.debug(
                    f"Ran {i+1}/{len(data)}."
                    f" prompt_len={prompt_len if not error_occurred else 'ERROR'}."
                    f" Saved to {out_filename}"
                )

        print_first_prompt = True

    with open(out_filename, "w", encoding="utf-8") as f:
        f.write(
            "\n".join([json.dumps(row, ensure_ascii=False) for row in data])
        )

    logger.info(f"Finished running. Output saved in {out_filename}.")


if __name__ == "__main__":
    StrictFire(main)
