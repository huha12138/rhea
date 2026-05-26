import re
import warnings
import os
import torch
import gc
import time

from string import Template
from torch import nn
from jinja2.exceptions import TemplateError
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel, PretrainedConfig, AutoModel, AutoConfig
from huggingface_hub import hf_hub_download
from typing import List, Optional, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
from nltk.tokenize import sent_tokenize

import torch

def add_memory_tokens_to_inputs(input_ids: torch.Tensor, attention_mask: torch.Tensor, n_mem_tokens: int, tokenizer):
    """
    Concatenate memory-token ids and extend the corresponding attention mask.
    """
    assert len(tokenizer.mem_tokens) == n_mem_tokens, f"{len(tokenizer.mem_tokens)} VS {n_mem_tokens}"

    mem_tokens = torch.stack([tokenizer.mem_token_ids_pt] * input_ids.size(0), 0)
    assert len(mem_tokens.size()) == 2
    assert len(mem_tokens) == input_ids.size(0)
    assert len(mem_tokens[0]) == n_mem_tokens
    input_ids = torch.cat([input_ids, mem_tokens], dim=1)
    attention_mask = torch.cat([attention_mask, torch.ones(input_ids.size(0), n_mem_tokens)], dim=1)
    return input_ids, attention_mask

class RheaConfig(PretrainedConfig):

    model_type = "rhea"
    def __init__(self,
                decoder_model_name: str = "meta-llama/Llama-2-7b-chat-hf",
                doc_max_length: int = 128,
                quantization: str = 'no',
                sep: bool = False,
                compr_model_name: str = "google-bert/bert-base-uncased",
                compr_rate: int = 64,
                compr_n_layers: int = None, # only for surgical mistral compressor
                compr_every_n_layer: int = None,
                compr_base_model_name: str = 'mistralai/Mistral-7B-Instruct-v0.2',
                compr_rms_norm: bool = False, # only for surgical mistral compressor: if true, rms norm applied on h-s
                compr_mlp_hidden_dim: int = 8096,
                compr_use_mlp: bool = True, 
                lora: bool = False, # lora on decoder (and decoder as compr)
                lora_compressor: bool = False, # lora only on the compressor if it exists
                training_form: str = "both",
                lora_r: int = 16,
                lora_r_compressor: int = None,
                load_adapters: bool = True,
                kbtc_training: bool = False,
                optimize_mem_tokens: bool = False,
                different_mem_tokens: bool = False,
                attn_implementation: str = 'flash_attention_2',
                device_map = None,
                **kwargs):
        super().__init__(**kwargs)

        self.decoder_model_name = decoder_model_name # model name of decoder
        self.doc_max_length = doc_max_length # the maximum length of document that can be used by this model (it is used to compute number of mem tokens !)
        self.quantization = quantization # quantization, could be no, int4, int8
        self.sep = sep # boolean type, whether to use sep token
        
        self.compr_model_name = compr_model_name # model name of compressor # null
        self.compr_rate = compr_rate # compression rate
        self.compr_use_mlp = compr_use_mlp
        self.compr_mlp_hidden_dim = compr_mlp_hidden_dim
        self.compr_n_layers = compr_n_layers
        self.compr_every_n_layer = compr_every_n_layer
        self.compr_base_model_name = compr_base_model_name
        self.compr_rms_norm = compr_rms_norm
        
        self.lora = lora # boolean type, whether to use lora trsining
        self.lora_compressor = lora_compressor
        self.training_form = training_form # training form, could be compressor: training only comprssor; both: training both
        # Or both_separately: training both with separate adapters
        self.lora_r = lora_r # lora_r for lora training, we use 16 throughout the experiment.
        self.lora_r_compressor = lora_r_compressor or lora_r # defaulting to same lora as decoder.
        self.load_adapters = load_adapters # used to load pretrained model: we first load without adapters, and then load them from file.
        self.optimize_mem_tokens = optimize_mem_tokens
        self.different_mem_tokens = different_mem_tokens
        
        self.kbtc_training = kbtc_training
        
        self.device_map = device_map
        
        self.attn_implementation = attn_implementation
        
        if training_form == 'compressor':
            assert compr_model_name is not None and not self.lora


class RheaSessionCache:
    """Cache dialogue state and compressed episodic-memory vectors."""

    def __init__(self):
        self.raw_history: List[str] = []
        self.bot_compressed_embs: List[torch.Tensor] = []
        self.is_instruction_mask: List[int] = []
        self.global_instructions: List[str] = []

    def clear(self):
        self.raw_history = []
        self.bot_compressed_embs = []
        self.is_instruction_mask = []
        self.global_instructions = []

    def __len__(self):
        return len(self.raw_history)

        
class RheaModel(PreTrainedModel):
    config_class = RheaConfig
    def __init__(self, cfg):
        super().__init__(cfg)
        self.decoder_model_name = cfg.decoder_model_name
        self.decoder = self.create_decoder(cfg)
        
        instruction_model_name = os.getenv(
            "RHEA_INSTRUCTION_MODEL", "Qwen/Qwen3-0.6B"
        )
    
        self.i_tokenizer = AutoTokenizer.from_pretrained(instruction_model_name)
        self.i_model = AutoModelForCausalLM.from_pretrained(
            instruction_model_name,
            torch_dtype="auto",
            device_map="auto"
        )

        for param in self.decoder.parameters():
            param.requires_grad = False
        
        self.doc_max_length = cfg.doc_max_length    # 128

        self.compr_model_name = cfg.compr_model_name
        self.training_form = cfg.training_form
        self.lora = cfg.lora
        self.adapter_keys = []

        self.chunk_count = []

        self.compr = None
        # when compr_model_name is not set, then means using a decoder-based compressor, otherwise a bert based compressor
        if cfg.compr_model_name is not None:    # null
            # case bert based compressor
            raise NotImplementedError
            print('Instantiating compressor ', cfg.compr_model_name)
            self.compr = BertCompressor(cfg.compr_model_name, 
                                        cfg.compr_rate, 
                                        doc_max_length=self.doc_max_length,
                                        decoder_hidden_size=self.decoder.config.hidden_size,
                                        mlp_hidden_dim=cfg.compr_mlp_hidden_dim,
                                        compr_n_layers=cfg.compr_n_layers,
                                        compr_every_n_layer=cfg.compr_every_n_layer,
                                        compr_base_model_name=cfg.compr_base_model_name,
                                        compr_rms_norm=cfg.compr_rms_norm,
                                        use_mlp=cfg.compr_use_mlp,
                                        attn_implementation=cfg.attn_implementation)

        # set lora adaptors on decoder model

        if cfg.lora:    
            peft_config = self.get_peft_config(lora_r=cfg.lora_r)

            if cfg.load_adapters:
                self.decoder.add_adapter(peft_config, 'decoder_adapter')
                self.decoder.set_adapter('decoder_adapter')
                self.adapter_keys.append('decoder_adapter')

            # Create separate adapters (if not BERT compressor and training_form == 'both_separately')
            if self.training_form == 'both_separately' and self.compr is None:
                if cfg.load_adapters:
                    self.decoder.add_adapter(peft_config, 'encoder_adapter')
                    self.adapter_keys.append('encoder_adapter')

        # set lora adapters on compressor model:
        if cfg.lora_compressor and self.compr is not None and cfg.load_adapters:    # false
            peft_config = self.get_peft_config(lora_r=cfg.lora_r_compressor)
            self.compr.set_lora(peft_config)
        
        self.decoder_tokenizer = RheaModel.create_decoder_tokenizer(cfg)

        self.bos_id = self.decoder_tokenizer('[/INST]', add_special_tokens=False).input_ids
        self.eos_id = self.decoder_tokenizer('</s>', add_special_tokens=False).input_ids
        self.pad_token_id = self.eos_id

        # resize the tokenizer embedding
        self.decoder.resize_token_embeddings(len(self.decoder_tokenizer))
        self.decoder.generation_config.top_p = None
        self.decoder.generation_config.temperature = None
        self.decoder.generation_config.pad_token_id = self.decoder_tokenizer.pad_token_id

        # other settings
        self.generation_top_k = 1
        self.sep = cfg.sep
        self.compr_rate = cfg.compr_rate
        self.local_rank = os.getenv('LOCAL_RANK', '0')
        
        self.n_mem_tokens = self.doc_max_length // self.compr_rate

        if self.lora:
            for adapter_key in self.adapter_keys:
                self.decoder.set_adapter(adapter_key)
                
            #  We need to activate all adapters so that they are both trained...
            self.set_all_adapters()
        else:
            print(f'Total trainable parameters: {self.num_parameters(only_trainable=True)}')
            
        if self.compr is not None:
            print(f'Compressor number of parameters: {self.compr.model.num_parameters(only_trainable=True)}')

        self.prepare_mem_tokens_optimization()

        self.cache = RheaSessionCache()


    def _sync_and_compress_batch(self, current_history: List[str], current_question: str) -> torch.Tensor:

        cached_len = len(self.cache)
        input_len = len(current_history)

        # Reset the cache if the caller provides a shorter or different history.
        if input_len < cached_len or current_history[:cached_len] != self.cache.raw_history:
            print("[Info] History mismatch. Resetting cache.")
            self.cache.clear()
            cached_len = 0

        new_turns = current_history[cached_len:]
        
        texts_to_compress = [] 
        
        # Process newly observed history turns and collect compressible text.
        for i, txt in enumerate(new_turns):
            global_idx = cached_len + i
            
            if global_idx % 2 == 1:
                # User turns are checked for global-instruction content.
                if self.is_global_instruction(txt):
                    self.cache.is_instruction_mask.append(1)
                    self.cache.global_instructions.append(txt)
                else:
                    self.cache.is_instruction_mask.append(-1)
                    self.cache.global_instructions.append("")
            else:
                # Assistant turns are stored in compressed episodic memory.
                texts_to_compress.append(txt)
                self.cache.is_instruction_mask.append(0)
                self.cache.global_instructions.append("")

        # Compress the current query together with new assistant turns.
        texts_to_compress.append(current_question)
        
        if texts_to_compress:
            input_encoder = self.prepare_encoder_inputs(texts_to_compress, max_length=512)
            device = self.decoder.device
            
            with torch.no_grad():
                batch_embs = self.compress(
                    input_encoder['input_ids'].to(device), 
                    input_encoder['attention_mask'].to(device)
                ) # Shape: (Batch_Size, 8, dim)
            
            curr_q_emb = batch_embs[-1].unsqueeze(0) # (1, 8, dim)
            
            new_bot_embs = batch_embs[:-1]
            for k in range(new_bot_embs.shape[0]):
                self.cache.bot_compressed_embs.append(new_bot_embs[k].unsqueeze(0))
        else:
            raise ValueError("Empty compression batch")

        self.cache.raw_history.extend(new_turns)
        
        return curr_q_emb

    def generate_from_text_w_Retrieval(self, questions: List[str], documents: List[List[str]], mode: int = 2, **kwargs) -> Tuple[List[str], List[str]]:
        """Generate with heuristic retrieval over compressed dialogue memory."""

        assert len(questions) == 1, "Stateful implementation supports single session only."
        current_question = questions[0]
        full_history = documents[0]

        # Step 1: update the cache and obtain the compressed query.
        curr_q_emb = self._sync_and_compress_batch(full_history, current_question)

        # Step 2: prepare the retrieval pool.
        history_embs_tensor = torch.cat(self.cache.bot_compressed_embs, dim=0)

        # Step 3: retrieve history according to query-memory similarity.
        full_embs_tensor = torch.cat([history_embs_tensor, curr_q_emb], dim=0)
        
        if mode == 0:
            result = self.similarity_last_vs_rest(full_embs_tensor)
        elif mode == 1:
            result = self.similarity_last_vs_rest_concat(full_embs_tensor)
        else:
            result = self.similarity_last_vs_rest_max(full_embs_tensor)
            
        # Step 4: reconstruct a hybrid context from raw and compressed memory.
        ind = [[1,3,3]]
        final_compressed_embs = self.cache.bot_compressed_embs[0]


        # 0: discard, 1: compressed, 3: uncompressed.
        if len(result) > 3:
            t1, t2 = 0.5, 0.7 # mode=2
            for i in range(2,len(result)-1):
                if result[i] < t1:
                    ind[0].append(0)
                    ind[0].append(0)
                elif result[i] > t2:
                    ind[0].append(3)
                    ind[0].append(3)
                else:
                    ind[0].append(3)
                    ind[0].append(1)
                    final_compressed_embs = torch.cat([final_compressed_embs, full_embs_tensor[i:i+1]], dim=0)

        global_instructions = [self.cache.global_instructions[i] if self.cache.is_instruction_mask[i]==1 else "" for i in range(len(self.cache.global_instructions))] 
        
        self.generation_top_k = final_compressed_embs.shape[0]

        # Create decoder inputs.
        instr = [self.blend_prompt_and_memory_tokens_with_doc(query=q,doc=documents[i],ind=ind[i][:len(documents[i])],global_instructions=global_instructions) for i,q in enumerate(questions)]
        inp_dec = self.decoder_tokenizer(instr, return_tensors='pt', padding="longest", add_special_tokens=False, truncation=True,  max_length=65536)
        
        # Replace memory-token placeholders with compressed vectors.
        device = self.decoder.device
        inputs_embeds = self.replace_emb1(final_compressed_embs, inp_dec['input_ids'].to(device))

        # Switch adapter if we are training two different ones:
        if 'decoder_adapter' in self.adapter_keys:
            self.decoder.set_adapter('decoder_adapter') 

        output_ids = self.decoder.generate(
            inputs_embeds=inputs_embeds.to("cuda"),
            attention_mask=inp_dec['attention_mask'].to(device),
            **kwargs,
            )

        decoded = self.decoder_tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        return decoded

    
    def blend_prompt_and_memory_tokens_with_doc(self, query: str, doc: List[str], ind: List[int], global_instructions: List[str] = []):
      
        assert len(doc) == len(ind), 'match_1588'
        mem_tokens_str = ''.join(self.decoder_tokenizer.mem_tokens) + self.decoder_tokenizer.sep_token
        

        docs = ""

        for i in range(1,len(doc)):
            if ind[i] == 0:
                pass
            elif ind[i] == 1:
                docs += mem_tokens_str
            elif ind[i] == 2:
                docs += mem_tokens_str
            elif ind[i] == 3:
                docs += doc[i]
        
        global_instruction = ' '.join(global_instructions)
        prompt_user = f"{mem_tokens_str}\n{global_instruction}\nBackground:\n{docs}\n\nQuestion:{query}"

        messages = [{"role": "user", "content": prompt_user.replace(':\\ ', ': ')}]

        # Attempt to apply the system role and catch if it's not supported
        try:
            prompt = self.decoder_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)       
        except TemplateError as e:
            # Catch the error related to system role and handle it (e.g. gemma)
            if "System role not supported" in str(e):
                # Remove system role and proceed with only the user role
                messages = [{"role": "user", "content": messages[0]['content'] + '\n' + messages[1]['content']}]
                # Apply template again without system role
                prompt = self.decoder_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                # Re-raise the exception if it's unrelated to system role
                raise e

        return prompt




    def classify_with_voting(self, text, n=3):
        votes = [self.is_global_instruction(text) for _ in range(n)]
        return sum(votes) >= (n // 2 + 1)

    def is_global_instruction(self, user_input: str) -> bool:
        # Step 1: Quick rule matching
        rule_patterns = [
            r"\ball (of the)? replies\b",
            r"\ball (of the)? answers\b",
            r"\bevery (single )?answer\b",
            r"\bfrom now on\b",
            r"\bin the following replies\b",
            r"\bin subsequent answers\b",
            r"\bin the future responses\b",
            r"\bfor all future answers\b",
        ]
        for pattern in rule_patterns:
            if re.search(pattern, user_input, flags=re.IGNORECASE):
                return True

        # Step 2: LLM classification
        prompt = f"""
            You are a classifier. Determine whether the following user input is a "global instruction" in a multi-turn conversation.
            A global instruction is a directive that affects **all subsequent responses** — such as their style, format, length, or language.
            If the input is a global instruction, answer: YES
            If the input is NOT a global instruction, answer: NO

            Examples of instructions:
            -Input: Explain what is a poem?  Answer: NO
            -Input: All future answers must be less than 30 words.  Answer: YES
            -Input: Can you translate this into French?  Answer: NO
            -Input: Every answer should end with a joke.  Answer: YES

            -Input: {user_input} Answer: 
        """.strip()

        inputs = self.i_tokenizer(prompt, return_tensors="pt").to(self.i_model.device)
        input_length = inputs.input_ids.shape[1]

        outputs = self.i_model.generate(**inputs, max_new_tokens=16)
        generated_ids = outputs[0][input_length:]
        gen_text = self.i_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        if "YES" in gen_text and not "NO" in gen_text:
            return True
        else:
            return False

    def prepare_mem_tokens_optimization(self):
        if self.config.optimize_mem_tokens: # true
            if self.compr is None:
                # Enforcing gradients for input embeddings (even if lora)
                self.decoder.get_input_embeddings().weight.requires_grad = True
                # Applying a hook zero-ing the gradients except for the mem token:
                def hook(grad):
                    mask = torch.zeros_like(grad)
                    mask[self.decoder_tokenizer.mem_token_ids] = 1.0
                    return grad * mask
                self.decoder.get_input_embeddings().weight.register_hook(hook)
                
    def set_all_adapters(self):
        if len(self.adapter_keys) > 0:
            self.decoder.set_adapter(self.adapter_keys)
            
    @staticmethod
    def create_decoder_tokenizer(cfg: RheaConfig):
        decoder_tokenizer = AutoTokenizer.from_pretrained(cfg.decoder_model_name, use_fast=True, padding_side='left')

        # define special tokens
        n_mem_tokens = cfg.doc_max_length // cfg.compr_rate # 128/16 = 8
        if cfg.different_mem_tokens:
            # estimation fo the number of memory tokens needed:
            mem_tokens = ['<MEM' + str(i) + '>' for i in range(n_mem_tokens)]
            decoder_tokenizer.add_special_tokens({'additional_special_tokens': mem_tokens + ['<AE>', '<ENC>', '<SEP>']}) 
            decoder_tokenizer.mem_tokens = mem_tokens
        else:
            decoder_tokenizer.add_special_tokens({'additional_special_tokens': ['<MEM>', '<AE>', '<ENC>', '<SEP>']})
            decoder_tokenizer.mem_tokens = ['<MEM>'] * n_mem_tokens
        
        decoder_tokenizer.mem_token_ids = [decoder_tokenizer.convert_tokens_to_ids(elt) for elt in decoder_tokenizer.mem_tokens]
        decoder_tokenizer.mem_token_ids_pt = torch.LongTensor(decoder_tokenizer.mem_token_ids) # required later on for operations on tensors
        
        decoder_tokenizer.ae_token = '<AE>' # token for autoencoding on decoder side
        decoder_tokenizer.ae_token_id = decoder_tokenizer.convert_tokens_to_ids('<AE>')
        decoder_tokenizer.enc_token = '<ENC>' # token for autoencoding on compressor side
        decoder_tokenizer.sep_token = '<SEP>' # sep token between document
        decoder_tokenizer.sep_token_id = decoder_tokenizer.convert_tokens_to_ids('<SEP>')

        # If kbtc training, we add another one yet
        if cfg.kbtc_training:
            decoder_tokenizer.add_special_tokens({'additional_special_tokens': ['<KBTC>']})
            decoder_tokenizer.kbtc_token = '<KBTC>'
            decoder_tokenizer.kbtc_token_id = decoder_tokenizer.convert_tokens_to_ids('<KBTC>')

        # if pad token exists then use pad token, othrwise bos token
        if decoder_tokenizer.pad_token_id is None:
            decoder_tokenizer.pad_token_id = decoder_tokenizer.bos_token_id

        return decoder_tokenizer

    def get_peft_config(self, lora_r: int) -> LoraConfig:
        """
        Builds the peft config
        """
        return LoraConfig(task_type="CAUSAL_LM", r=lora_r, lora_alpha=2*lora_r, target_modules='all-linear', lora_dropout=0.1)

    def create_decoder(self, cfg):
        """
        Loads the base decoder.
        """
        if torch.cuda.is_available():
            if cfg.quantization == "no":
                return AutoModelForCausalLM.from_pretrained(
                    cfg.decoder_model_name,
                    torch_dtype=torch.bfloat16,
                    attn_implementation=self.config.attn_implementation,
                    device_map=cfg.device_map
                    )
            
            elif cfg.quantization == "int4":
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type='nf4',
                    bnb_4bit_compute_dtype='bfloat16',
                )
                return AutoModelForCausalLM.from_pretrained(
                    cfg.decoder_model_name,
                    quantization_config=quant_config,
                    attn_implementation=self.config.attn_implementation,
                    torch_dtype=torch.bfloat16,
                    resume_download=True,
                    trust_remote_code=True,
                    device_map=cfg.device_map
                )
            elif cfg.quantization == "int8":
                quant_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True,
                    bnb_4bit_compute_dtype='bfloat16',
                )
                return AutoModelForCausalLM.from_pretrained(
                    cfg.decoder_model_name,
                    quantization_config=quant_config,
                    attn_implementation=self.config.attn_implementation,
                    torch_dtype=torch.bfloat16,
                    resume_download=True,
                    trust_remote_code=True,
                    device_map=cfg.device_map
                )
            else:
                raise NotImplementedError()
        else:
            return AutoModelForCausalLM.from_pretrained(
                cfg.decoder_model_name,
                torch_dtype=torch.bfloat16,
                resume_download=True,
                trust_remote_code=True,
                device_map=cfg.device_map
            )
            
    def compress(self, enc_input_ids, enc_attention_mask):
        if self.compr:
            return self.compr(enc_input_ids, enc_attention_mask)
        else:
            return self.compr_decoder_1(enc_input_ids, enc_attention_mask)


    def compr_decoder_1(self, input_ids, attention_mask):
        """Compress inputs while filtering memory-token states per micro-batch."""
        assert input_ids.size() == attention_mask.size(), f"{input_ids.size()} vs {attention_mask.size()}"
        
        # Switch to the encoder adapter when it is available.
        if 'encoder_adapter' in self.adapter_keys:
            self.decoder.set_adapter('encoder_adapter')
    
        self.decoder.eval()
        
        mem_token_ids = self.decoder_tokenizer.mem_token_ids_pt.to(input_ids.device)
        num_mem_tokens = len(mem_token_ids) 
        
        batch_size = input_ids.size(0)
        micro_batch_size = 32
        hidden_list = []

        internal_model = getattr(self.decoder, "model", self.decoder)
        if hasattr(internal_model, "model"): 
            internal_model = internal_model.model
        
        with torch.no_grad():
            for start in range(0, batch_size, micro_batch_size):
                end = min(start + micro_batch_size, batch_size)
                ids_chunk = input_ids[start:end]
                mask_chunk = attention_mask[start:end]
                # Disable KV cache because only hidden states are needed.
                outputs = internal_model(
                    input_ids=ids_chunk,
                    attention_mask=mask_chunk,
                    output_hidden_states=False, 
                    use_cache=False 
                ).last_hidden_state

                # Immediately keep only memory-token hidden states.
                chunk_mask = torch.isin(ids_chunk, mem_token_ids)

                filtered_hidden = outputs[chunk_mask].reshape(ids_chunk.size(0), -1, outputs.size(-1))
                
                hidden_list.append(filtered_hidden)
                
                del outputs
                torch.cuda.empty_cache()
                
        emb = torch.cat(hidden_list, dim=0).to(input_ids.device)
        return emb


    def replace_emb(self, compressed_embs, dec_input_ids):
        """
        Compression logic (either with decoder or with dedicated compressor)
        """
        indices = range(0, compressed_embs.size(0) + 1, self.generation_top_k)            
        input_embeds = self.replace_embeddings(compressed_embs, dec_input_ids, indices)
        return input_embeds

    def replace_emb1(self, compressed_embs, dec_input_ids):
        """
        Compression logic (either with decoder or with dedicated compressor)
        """
        indices = range(0, compressed_embs.size(0) + 1, self.generation_top_k) 

        inputs_embeds = self.decoder.get_input_embeddings()(dec_input_ids)
        num_embs = compressed_embs.size(1)  # 8
        batch_size = inputs_embeds.size(0)

        mask = dec_input_ids == self.decoder_tokenizer.mem_token_ids[0]
        all_mem_token_indices = torch.nonzero(mask, as_tuple=False)
        all_positions = all_mem_token_indices[:, 1].reshape(batch_size, self.generation_top_k)

        # for each example in batch, replace them with compressed embeddings
        for i in range(batch_size):
            for j in range(indices[i], indices[i + 1]):
                start_idx = all_positions[i][j-indices[i]].item()
                assert inputs_embeds[i, start_idx:start_idx + num_embs, :].size() == compressed_embs[j].size(), \
                    f"{inputs_embeds[i, start_idx:start_idx + num_embs, :].size()} VS {compressed_embs[j].size()}"  # 4096
                inputs_embeds[i, start_idx:start_idx + num_embs, :] = compressed_embs[j]
        return inputs_embeds


    def prepare_encoder_inputs_to_decoder(self, texts, max_length, q_texts=None):
        if q_texts is not None:
            texts_to_encode = [self.decoder_tokenizer.enc_token + self.decoder_tokenizer.bos_token + '\nQuery:\n' + query + 'Document:\n' + text + self.decoder_tokenizer.eos_token 
                               for text, query in zip(texts, q_texts)]
            inp_enc = self.decoder_tokenizer(texts_to_encode, return_tensors='pt', padding='max_length', max_length=max_length + 8, truncation=True, add_special_tokens=False)
        else:

            inp_enc = [self.decoder_tokenizer.enc_token + self.decoder_tokenizer.bos_token + text + self.decoder_tokenizer.eos_token for text in texts]
            inp_enc = self.decoder_tokenizer(inp_enc, return_tensors='pt', padding="max_length", max_length=max_length+3, truncation=True, add_special_tokens=False)
        
        num_mem_tokens = self.doc_max_length // self.compr_rate
        assert num_mem_tokens == len(self.decoder_tokenizer.mem_tokens)

        inp_enc['input_ids'], inp_enc['attention_mask'] = add_memory_tokens_to_inputs(inp_enc['input_ids'], 
                                                                                        inp_enc['attention_mask'],
                                                                                        num_mem_tokens, 
                                                                                        tokenizer=self.decoder_tokenizer)
        
        return inp_enc
    
    def prepare_encoder_inputs(self, texts: List[str], max_length: int, q_texts: List[str] = None):
        """
        Create the inputs to the encoder, for compression.
        """
        if q_texts is not None:
            assert len(texts) == len(q_texts), f"{len(texts)} == {len(q_texts)}"

        if self.compr is None:  # Case where the encoder is the decoder with adapter:
            return self.prepare_encoder_inputs_to_decoder(texts, max_length, q_texts)
        else:   # Case where the encoder is a separate network:
            return self.compr.prepare_inputs(texts, max_length, q_texts)

    def replace_embeddings(self, compressed_embs, dec_input_ids, indices):
        """
        Replace memory tokens in the decoder input to with the compressed embeddings
        """
        inputs_embeds = self.decoder.get_input_embeddings()(dec_input_ids)
        num_embs = compressed_embs.size(1)
        if self.sep:
            slot_len = num_embs + 1
        else:
            slot_len = num_embs
        # get first mem_token indices
        first_mem_token_indices = torch.argmax((dec_input_ids == self.decoder_tokenizer.mem_token_ids[0]).int(), dim=1)
        batch_size = inputs_embeds.size(0)
        # for each example in batch, replace them with compressed embeddings
        for i in range(batch_size):
            for j in range(indices[i], indices[i + 1]):
                start_idx = first_mem_token_indices[i].item() + (j-indices[i]) * slot_len
                assert inputs_embeds[i, start_idx:start_idx + num_embs, :].size() == compressed_embs[j].size(), \
                    f"{inputs_embeds[i, start_idx:start_idx + num_embs, :].size()} VS {compressed_embs[j].size()}"
                inputs_embeds[i, start_idx:start_idx + num_embs, :] = compressed_embs[j]
        return inputs_embeds
    

    def forward(self,
                enc_input_ids: torch.LongTensor = None,
                enc_attention_mask: torch.LongTensor = None,
                dec_input_ids: torch.LongTensor = None,
                dec_attention_mask: torch.LongTensor = None,
                labels: torch.LongTensor = None,
                loss_mask = None):
        """
        enc_input_ids: stores the contexts, should be flattened from all queries before input, can be of shape:
            - (batch_size*generation_top_k, enc_token_length)
            - (batch_size, generation_top_k, enc_token_length)
        enc_attention_mask: attention mask of enc_input_ids, same shape as enc_input_ids
        dec_input_ids: stores the prompts (including mem tokens), dimention (batch_size, dec_token_length)
        dec_attention_mask: attention mask of dec_input_ids
        """ 
        assert enc_input_ids.size() == enc_attention_mask.size(), f"{enc_input_ids.size()} vs {enc_attention_mask.size()}"
        
        if len(enc_input_ids.size()) == 3: # likely from bergen: we just flatten all of this to perform encoding in one batch
            batch_size, top_k, seq_length = enc_input_ids.size()
            enc_input_ids = enc_input_ids.view(batch_size * top_k, seq_length)
            enc_attention_mask = enc_attention_mask.view(batch_size * top_k, seq_length)
        
        # Here, we should have top_k times more elements in enc_input_ids than in dec_input_ids
        assert enc_input_ids.size(0) == dec_input_ids.size(0) * self.generation_top_k, \
            f"{enc_input_ids.size(0)} VS {dec_input_ids.size(0)} with generation_top_k={self.generation_top_k}"
            
        # Perform compression with gradient tracking
        compressed_embs = self.compress(enc_input_ids, enc_attention_mask)
        inputs_embeds = self.replace_emb(compressed_embs, dec_input_ids)

        # if training_form is compressor, then detach the inputs_embeds, to make gradient not count in decoder
        if (self.training_form == "compressor") and (self.compr is None):
            inputs_embeds  = inputs_embeds.detach()

        # decoding
        if 'decoder_adapter' in self.adapter_keys:
            self.decoder.set_adapter('decoder_adapter')

        decoder_outputs = self.decoder(inputs_embeds=inputs_embeds, attention_mask=dec_attention_mask, labels=labels)

        # At end of forward, we need to activate all adapters so that they are both trained...
        self.set_all_adapters()

        return {"loss":decoder_outputs.loss, "loss_mask": loss_mask, "logits": decoder_outputs.logits, "label":labels}

    def similarity_last_vs_rest(self, tensor):
        """
        tensor: shape [num_of_doc, 8, 4096]
        return: similarity scores rounded to two decimals.
        """
        # [new_num, 8, 4096]
        tensors = tensor.clone()
        kept_avg = tensors.mean(dim=1)                   # [new_num, 4096]
        last_avg = tensors[-1].mean(dim=0)             # [4096]

        kept_norm = torch.nn.functional.normalize(kept_avg, dim=1)
        last_norm = torch.nn.functional.normalize(last_avg, dim=0)

        sims = torch.matmul(kept_norm, last_norm)     # [new_num]
        retult = [round(float(s), 2) for s in sims]

        return retult
    
    def similarity_last_vs_rest_concat(self, tensor):
        """
        tensor: shape [num_of_doc, 8, 4096]
        return: similarity scores rounded to two decimals.
        """
        # [new_num, 8, 4096]
        tensors = tensor.clone()
        kept_concat = tensors.view(tensors.size(0), -1)      # [new_num, 32768]
        last_concat = tensor[-1].view(-1)            # [4096]

        kept_norm = torch.nn.functional.normalize(kept_concat, dim=1)
        last_norm = torch.nn.functional.normalize(last_concat, dim=0)

        sims = torch.matmul(kept_norm, last_norm)     # [new_num]
        retult = [round(float(s), 2) for s in sims]

        return retult

    def similarity_last_vs_rest_max(self, tensor):
        """
        tensor: shape [num_of_doc, 8, 4096]
        return: similarity scores rounded to two decimals.
        """
        # [new_num, 8, 4096]
        tensors = tensor.clone()
        kept = tensors       
        last = tensors[-1]           

        kept_norm = torch.nn.functional.normalize(kept, dim=-1)
        last_norm = torch.nn.functional.normalize(last, dim=-1)

        sims = torch.matmul(kept_norm, last_norm.T) 
        max_sims, _ = sims.view(sims.size(0), -1).max(dim=1)
        retult = [round(float(s), 2) for s in max_sims]

        return retult

    def generate(self, model_input, max_new_tokens=128, return_doc_embeddings: bool = False, mode: int = 0):

        enc_input_ids, enc_attention_mask, dec_input_ids, dec_attention_mask = model_input['enc_input_ids'], model_input['enc_attention_mask'], model_input['dec_input_ids'], model_input['dec_attention_mask']
        
        assert enc_input_ids.size() == enc_attention_mask.size()
        
        if len(enc_input_ids.size()) == 3: # likely from bergen: we just flatten all of this to perform encoding in one batch
            batch_size, top_k, seq_length = enc_input_ids.size()
            enc_input_ids = enc_input_ids.view(batch_size * top_k, seq_length)
            enc_attention_mask = enc_attention_mask.view(batch_size * top_k, seq_length)
            
        st = time.time()
        compressed_embs = self.compress(enc_input_ids.to('cuda'), enc_attention_mask.to('cuda'))
        used_time = round(time.time()-st, 2)

        inputs_embeds = self.replace_emb1(compressed_embs, dec_input_ids.to('cuda'))

        # Switch adapter if we are training two different ones:
        if 'decoder_adapter' in self.adapter_keys:
            self.decoder.set_adapter('decoder_adapter') 

        output_ids = self.decoder.generate(
            inputs_embeds=inputs_embeds.to("cuda"),
            attention_mask=dec_attention_mask.to("cuda"),
            do_sample=False,
            top_p=None,
            max_new_tokens=max_new_tokens
            )

        decoded = self.decoder_tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        if return_doc_embeddings:  
            assert batch_size is not None
            assert top_k is not None
            compressed_embs = compressed_embs.view(batch_size, top_k, compressed_embs.size(1), compressed_embs.size(2))
            return decoded, compressed_embs
        else:
            return decoded


    def get_all_adapters_state_dict(self):
        """
        Return the state dicts of the adapters
        Used for saving so we go to cpu automatically
        """
        return {key: {k:v.cpu() for k, v in self.decoder.get_adapter_state_dict(key).items()} for key in self.adapter_keys}

    def load_adapter_from_state_dict(self, peft_config: LoraConfig, adapter_name: str, adapter_state_dict: dict) -> None:
        """
        Creates an adapter from the state dict (used to load from pretrained)
        """
        self.decoder.load_adapter(peft_config=peft_config, adapter_name=adapter_name, adapter_state_dict=adapter_state_dict)
        self.adapter_keys.append(adapter_name)
        
    def get_decoder_first_and_last_layer_state_dict(self) -> dict:
        """
        Just getting the first and last layers: the only ones which change when adding tokens
        Used to save the model so we automatically move to cpu.
        """
        out = {}
        for k, v in self.decoder.named_parameters():
            if 'lm_head.weight' in k or 'embed_tokens.weight' in k:
                out[k] = v.cpu()
                
        return out

    def save_pretrained(self, save_directory: str, **kwargs):
        """
        Save only the LoRA adapters and their configurations.
        """
        if self.lora:
            if not os.path.exists(save_directory):
                os.makedirs(save_directory) 

            # Save the LoRA adapter weights
            torch.save(self.get_all_adapters_state_dict(), os.path.join(save_directory, "adapters.pth"))
            
            # Save the first and last layers of decoder (because of diffs with tokens !)
            torch.save(self.get_decoder_first_and_last_layer_state_dict(), os.path.join(save_directory, "decoder_first_last_layers.pth"))
            
            # Save the bert compressor if it exists
            if self.compr_model_name is not None:
                self.compr.save_pretrained(os.path.join(save_directory, 'compressor'))

            # Save the configuration
            self.config.save_pretrained(save_directory)
        else:
            super().save_pretrained(save_directory, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """
        Loading: to take care of checkpoints containing only lora and not base model.
        """
        # Load the configuration
        config = RheaConfig.from_pretrained(pretrained_model_name_or_path)
        config.attn_implementation = kwargs.get('attn_implementation', config.attn_implementation)  
        map_location = torch.device("cpu") if not torch.cuda.is_available() else None
        if config.lora:

            # We need to delay the construction of the adapters (otherwise peft complains)
            config.load_adapters = False

            if 'device_map' in kwargs:
                config.device_map = kwargs['device_map']

            # Initialize the model
            model = cls(config)

            # Loading first and last layers (they might have changed due to extra tokens)
            try:
                # If loading from Hugging Face Hub
                first_and_last_layers_path = hf_hub_download(
                    repo_id=pretrained_model_name_or_path, 
                    filename="decoder_first_last_layers.pth"
                )
            except Exception as e:
                # If loading from a local directory
                first_and_last_layers_path = os.path.join(pretrained_model_name_or_path, "decoder_first_last_layers.pth")

            if os.path.exists(first_and_last_layers_path):
                first_and_last_decoder_state_dict = torch.load(first_and_last_layers_path, map_location=map_location, weights_only=True)
                for key in first_and_last_decoder_state_dict:
                    assert key in model.decoder.state_dict()
                    model.decoder.load_state_dict(first_and_last_decoder_state_dict, strict=False)
            else:
                print('FIRST AND LAST LAYER NOT FOUND (ok for some old models):', first_and_last_layers_path)
            peft_config = model.get_peft_config(lora_r=config.lora_r)
            
            # Load the LoRA adapters (if the file exists)
            try:
                # If loading from Hugging Face Hub
                adapters_path = hf_hub_download(
                    repo_id=pretrained_model_name_or_path, 
                    filename="adapters.pth"
                )
            except Exception as e:
                # If loading from a local directory
                adapters_path = os.path.join(pretrained_model_name_or_path, "adapters.pth")
                
            if os.path.exists(adapters_path):
                adapters_state_dict = torch.load(adapters_path, map_location=map_location, weights_only=True)
            
                for key, val in adapters_state_dict.items():
                    model.load_adapter_from_state_dict(peft_config=peft_config, adapter_name=key, adapter_state_dict=val)
            else:
                warnings.warn(f'I see LoRA enabled for this Rhea model, but {adapters_path} does not exist; this may be normal \
                        for recent versions of transformers, be aware.')

            # If there is a compressor, it's been built: we just need to load the state dict or the adapters:
            if config.compr_model_name is not None: # null
                model.compr.load_pretrained(os.path.join(pretrained_model_name_or_path, 'compressor'), 
                                            lora=config.lora_compressor, 
                                            peft_config=model.get_peft_config(lora_r=config.lora_r_compressor))
            model.set_all_adapters()

            model.config.load_adapters = True

            return model

        else:
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

if __name__ == '__main__':
    cfg = RheaConfig(decoder_model_name='mistralai/Mistral-7B-Instruct-v0.2',
                compr_model_name = "mistral_trimmed",
                compr_rate = 64,
                compr_n_layers = 5,
                compr_mlp_hidden_dim = 8096,
                compr_use_mlp = False, 
                lora = True, # lora on decoder (and decoder as compr)
                lora_compressor = True, # lora only on the compressor if it exists
                training_form = "both",
                load_adapters = True,
                kbtc_training = False,
                optimize_mem_tokens = True,
                different_mem_tokens = True,
                attn_implementation = 'flash_attention_2')
    
    rhea_model = RheaModel(cfg)
    
    rhea_model.save_pretrained('test_ckpt')
    
    del rhea_model
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    rhea_model = RheaModel.from_pretrained('test_ckpt')
