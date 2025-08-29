from imports import *

import random
import json
import torch
import numpy as np
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import torch.nn as nn
from openai import OpenAI
from typing import Any, Optional, Union, List
from tqdm import tqdm
import torch.nn.functional as F
from datasets import Dataset,load_from_disk



def set_seed(seed: int = 42):
    # Python 内置随机库
    random.seed(seed)

    # NumPy 随机种子
    np.random.seed(seed)

    # PyTorch 随机种子
    torch.manual_seed(seed)

    # 如果使用 GPU（CUDA）
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多卡训练下设置所有卡

    # CUDA 加速时的确定性设置（强烈建议）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 环境变量控制 hash 随机性（某些 tokenizer 会用到）
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.set_float32_matmul_precision('high')

def register_forward_hooks(model):
    def make_hook(name):
        def hook_fn(module, input, output):
            try:
                if isinstance(output, torch.Tensor):
                    # Check for NaN or all 0 in the output tensor
                    if torch.isnan(output).any():
                        print(f"[NaN DETECTED] {name} -> Tensor, shape={output.shape}")
                        print(f"    Sample: {output.view(-1)[:10]}")
                    # elif torch.all(output == 0):
                    #     print(f"[ZERO DETECTED] {name} -> Tensor, shape={output.shape}")
                    #     print(f"    Sample: {output.view(-1)[:10]}")
                elif isinstance(output, (tuple, list)):
                    for i, out in enumerate(output):
                        if isinstance(out, torch.Tensor) and torch.isnan(out).any():
                            print(f"[NaN DETECTED] {name}[{i}] -> shape={out.shape}")
                            print(f"    Sample: {out.view(-1)[:10]}")
                        # elif isinstance(out, torch.Tensor) and torch.all(out == 0):
                        #     print(f"[ZERO DETECTED] {name}[{i}] -> shape={out.shape}")
                        #     print(f"    Sample: {out.view(-1)[:10]}")
                elif isinstance(output, dict):
                    for k, v in output.items():
                        if isinstance(v, torch.Tensor) and torch.isnan(v).any():
                            print(f"[NaN DETECTED] {name}['{k}'] -> shape={v.shape}")
                            print(f"    Sample: {v.view(-1)[:10]}")
                        # elif isinstance(v, torch.Tensor) and torch.all(v == 0):
                        #     print(f"[ZERO DETECTED] {name}['{k}'] -> shape={v.shape}")
                        #     print(f"    Sample: {v.view(-1)[:10]}")
            except Exception as e:
                print(f"[HOOK ERROR] in {name}: {e}")
        return hook_fn

    for name, module in model.named_modules():
        if any(isinstance(module, t) for t in [nn.Linear, nn.TransformerEncoderLayer, nn.LayerNorm]):
            module.register_forward_hook(make_hook(name))
def register_backward_hooks(model):
    def make_hook(name):
        def hook_fn(grad):
            if torch.isnan(grad).any():
                print(f"[NaN GRAD] {name} has NaN in gradient!")
                print(f"   grad sample: {grad.view(-1)[:10]}")
            elif torch.isinf(grad).any():
                print(f"[Inf GRAD] {name} has Inf in gradient!")
            else:
                grad_abs_max = grad.abs().max()
                if grad_abs_max > 1e4:  # 可调阈值
                    print(f"[LARGE GRAD] {name} grad max={grad_abs_max:.4e}")
            return grad  # 不修改梯度
        return hook_fn

    for name, param in model.named_parameters():
        if param.requires_grad:
            param.register_hook(make_hook(name))

def print_first_last_grads(model):
    params = list(model.named_parameters())
    
    if not params:
        print("[WARN] Model has no parameters.")
        return

    for idx in [3, -1]:
        name, param = params[idx]
        if param.grad is not None:
            grad = param.grad
            print(f"[GRAD-{ 'FIRST' if idx == 3 else 'LAST' }] {name} mean: {grad.mean().item():.6f}, max: {grad.abs().max().item():.6f}")
        else:
            print(f"[GRAD-{ 'FIRST' if idx == -1 else 'LAST' }] {name} grad is None")


def delta_inject(model, adapter_weights):
    """
    Injects delta weights into the model's layers.
    
    Args:
        model: The model to inject deltas into.
        adapter_weights: A dictionary containing the delta weights.
    """
    # print("num of modouls:", len(adapter_weights))
    modules = set(".".join(k.split(".")[:-2]) for k in adapter_weights.keys())
    for module in modules:
        m = get_attributes(model, module)
        lora_A = adapter_weights[module + ".lora_A.weight"]
        lora_B = adapter_weights[module + ".lora_B.weight"]
        # Calculate delta
        delta = lora_B @ lora_A
        # Set the delta in the module
        setattr(m, "delta", delta)  
        # m.register_buffer("delta", delta) 
        # assert m.delta.requires_grad, f"{module}.delta does not require grad"
        # print(f"{module}.delta grad_fn:", m.delta.grad_fn)

                
def delta_remove(model, adapter_weights):
    """
    Removes delta weights from the model's layers.
    
    Args:
        model: The model to remove deltas from.
        adapter_weights: A dictionary containing the delta weights.
    """
    modules = set(".".join(k.split(".")[:-2]) for k in adapter_weights.keys())
    for module in modules:
        m = get_attributes(model, module)
        delattr(m, "delta") 
                
def get_attributes(x: nn.Module, attributes: str):
    """
    gets a list of period-separated attributes
    i.e get_attributes(model, 'transformer.encoder.layer')
        should return the same as model.transformer.encoder.layer
    """
    for attr in attributes.split("."):
        x = getattr(x, attr)
    return x

class AutoGradTracer:
    def __init__(self, model=None):
        self.model = model
        self.param_hooks = {}
        self.tracked_tensors = []

    def attach(self):
        if self.model is None:
            raise ValueError("No model provided to AutoGradTracer.")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self._add_param_hook(name, param)

    def _add_param_hook(self, name, param):
        def hook_fn(grad):
            print(f"[Param Hook] `{name}` received grad:")
            print(f" - Shape: {grad.shape}")
            print(f" - Mean : {grad.mean().item():.4e}\n")
        self.param_hooks[name] = param.register_hook(hook_fn)

    def track(self, tensor, name="unnamed_tensor"):
        tensor.retain_grad()
        self.tracked_tensors.append((tensor, name))

    def report(self):
        print("\n=== [Tracked Tensors Grad Report] ===")
        for tensor, name in self.tracked_tensors:
            grad = tensor.grad
            if grad is not None:
                print(f"[Tracked] `{name}` grad:")
                print(f" - Shape: {grad.shape}")
                print(f" - Mean : {grad.mean().item():.4e}\n")
            else:
                print(f"[Tracked] `{name}` has NO grad.\n")

# ========== 安全解析函数 ==========
def safe_call_gpt(prompt, client, model="deepseek-v3", temperature=0.7):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[API ERROR] {e}")
        return ""

# 提取 QA JSON 并容错处理
def extract_qa_output(output_str):
    try:
        start = output_str.find("[")
        end = output_str.rfind("]")
        if start != -1 and end != -1:
            raw = output_str[start:end+1]
            qa = json.loads(raw)
            qa = qa[:5]
            for q in qa:
                q.setdefault("question", "")
                q.setdefault("answer", "")
                q.setdefault("full_answer", "")
            return qa
    except Exception as e:
        print(f"[PARSE ERROR] {e}")
    return []

def get_output_path(input_path: str, output_base_dir: str) -> Path:
    input_path = Path(input_path).resolve()
    input_parts = input_path.parts

    # 输入结构应为：.../data_aug_projector/{dataset_name}/{model_name}/{filename}
    dataset_name = input_parts[-3]    # e.g., "2wikimultihopqa"
    file_name = input_parts[-1]       # e.g., "bridge_comparison.json"

    return Path(output_base_dir) / dataset_name / file_name
# ========== 主增强函数 ==========
def reaugment_with_deepseek(input_paths: List[str], output_base_dir: str):
    client = OpenAI(
        api_key="sk-Okf9TYIr0BvKEetyZ7j5vtQyWBmE7EYuMhf5ayS5HSnijONO",
        base_url="https://api.chatanywhere.tech/v1",
    )

    rewrite_prompt_template = """
I will provide a passage. Please rewrite it while preserving all factual content, entities, and key terms. The rewritten version should be semantically faithful but use different wording. Output ONLY the rewritten passage, without any introductions or comments.

Passage:
{passage}

Rewritten Passage:
"""
    qa_prompt_template = "I will provide a passage of text, and you need to generate five different questions based on the content of this passage. Each question should be answerable using the information provided in the passage. Additionally, please provide an appropriate answer for each question derived from the passage.\n\
You need to generate the question and answer in the following format:\n\
[\n\
    {{\n\
        \"question\": \"What is the capital of France?\",\n\
        \"answer\": \"Paris\"\n\
        \"full_answer\": \"The capital of France is Paris.\"\n\
    }}, \n\
]\n\n\
This list should have at least five elements. You only need to output this list in the above format.\n\
Passage:\n\
{passage}"
    for input_path in input_paths:
        with open(input_path, "r") as fin:
            dataset = json.load(fin)

        for item in tqdm(dataset, desc=f"Processing {os.path.basename(input_path)}"):
            if "augment" not in item:
                continue

            for aug in item["augment"]:
                passage = aug.get("passage", "")

                # 1. Rewriting
                prompt = rewrite_prompt_template.format(passage=passage)
                rewrite = safe_call_gpt(prompt,client)
                aug["ds3_rewrite_original_output"] = rewrite  # 保留原始内容
                aug["deepseekv3_rewrite"] = rewrite if rewrite else ""

                # 2. QA Generation
                qa_prompt = qa_prompt_template.format(passage=passage)
                qa_output = safe_call_gpt(qa_prompt,client)
                aug["ds3_qa_original_output"] = qa_output  # 保留原始内容
                qa_parsed = extract_qa_output(qa_output)
                aug["deepseekv3_qa"] = qa_parsed if qa_parsed else []

                # 可选删除旧字段（安全）
                for key in list(aug.keys()):
                    if key.endswith("_rewrite") and not key.startswith("deepseekv3"):
                        del aug[key]
                    if key.endswith("_qa") and not key.startswith("deepseekv3"):
                        del aug[key]
            # break#test
        
        # 构建并创建输出路径
        output_path = get_output_path(input_path, output_base_dir)
        os.makedirs(output_path.parent, exist_ok=True)
        with open(output_path, "w") as fout:
            json.dump(dataset, fout, indent=4, ensure_ascii=False)
        print(f"[Saved] {output_path}")
    


def augment_with_deepseek(input_paths: List[str], output_base_dir: str):
    # global current_dataset, current_output_path
    # setup_signal_handlers()
    saving_steps = 100
    saving_count=0
    client = OpenAI(
        api_key="sk-Okf9TYIr0BvKEetyZ7j5vtQyWBmE7EYuMhf5ayS5HSnijONO",
        base_url="https://api.chatanywhere.tech/v1",
    )

    rewrite_prompt_template = """
I will provide a passage. Please rewrite it while preserving all factual content, entities, and key terms. The rewritten version should be semantically faithful but use different wording. Output ONLY the rewritten passage, without any introductions or comments.

Passage:
{passage}

Rewritten Passage:
"""
    qa_prompt_template = "I will provide a passage of text, and you need to generate two to five different questions based on the content of this passage. Each question should be answerable using the information provided in the passage. Additionally, please provide an appropriate answer for each question derived from the passage.\n\
You need to generate the question and answer in the following format:\n\
[\n\
    {{\n\
        \"question\": \"What is the capital of France?\",\n\
        \"answer\": \"Paris\"\n\
        \"full_answer\": \"The capital of France is Paris.\"\n\
    }}, \n\
]\n\n\
This list should have at least two elements. As long as the passage information is sufficient, you should try to generate five elements as much as possible. You only need to output this list in the above format.\n\
Passage:\n\
{passage}"
    for input_path in input_paths:
        with open(input_path, "r") as fin:
            dataset = json.load(fin)
        output_path = Path(output_base_dir)/Path(input_path).parts[-2]/Path(input_path).parts[-1]
        os.makedirs(output_path.parent, exist_ok=True)

        for item in tqdm(dataset, desc=f"Processing {os.path.basename(input_path)}"):
            passage = item.get("passages", [""])[0]

            augment_entry = {"passage": passage}

            # 1. 重写
            rewrite_prompt = rewrite_prompt_template.format(passage=passage)
            rewrite = safe_call_gpt(rewrite_prompt, client)
            augment_entry["ds3_rewrite_original_output"] = rewrite
            augment_entry["deepseekv3_rewrite"] = rewrite if rewrite else ""

            # 2. QA生成
            qa_prompt = qa_prompt_template.format(passage=passage)
            qa_output = safe_call_gpt(qa_prompt, client)
            augment_entry["ds3_qa_original_output"] = qa_output
            augment_entry["deepseekv3_qa"] = extract_qa_output(qa_output)

            item["augment"] = [augment_entry]
            saving_count+=1
            if saving_count%saving_steps==0:
                with open(output_path, "w") as fout:
                    json.dump(dataset, fout, indent=4, ensure_ascii=False)
            # break#test
        
        with open(output_path, "w") as fout:
            json.dump(dataset, fout, indent=4, ensure_ascii=False)
        print(f"[Saved] {output_path}")

def augment_with_model(input_paths: List[str], output_base_dir: str, model_path: str, max_new_tokens: int = 256, device: str = "cuda:0"):
    def model_generate(model, tokenizer, prompt, max_new_tokens=1024):
        message = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens,pad_token_id=tokenizer.pad_token_id)
        output_str = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return output_str
    saving_steps = 100
    saving_count=0
    llm_model_name = Path(model_path).parts[-1]
    llm_model = AutoModelForCausalLM.from_pretrained(model_path,device_map=device)
    llm_tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    llm_tokenizer.pad_token_id = llm_tokenizer.eos_token_id

    rewrite_prompt_template = """
I will provide a passage. Please rewrite it while preserving all factual content, entities, and key terms. The rewritten version should be semantically faithful but use different wording. Output ONLY the rewritten passage, without any introductions or comments.

Passage:
{passage}

Rewritten Passage:
"""
    qa_prompt_template = "I will provide a passage of text, and you need to generate two to five different questions based on the content of this passage. Each question should be answerable using the information provided in the passage. Additionally, please provide an appropriate answer for each question derived from the passage.\n\
You need to generate the question and answer in the following format:\n\
[\n\
    {{\n\
        \"question\": \"What is the capital of France?\",\n\
        \"answer\": \"Paris\"\n\
        \"full_answer\": \"The capital of France is Paris.\"\n\
    }}, \n\
]\n\n\
This list should have at least two elements. As long as the passage information is sufficient, you should try to generate five elements as much as possible. You only need to output this list in the above format.\n\
Passage:\n\
{passage}"
    for input_path in input_paths:
        with open(input_path, "r") as fin:
            dataset = json.load(fin)
        output_path = Path(output_base_dir)/Path(input_path).parts[-2]/Path(input_path).parts[-1]
        os.makedirs(output_path.parent, exist_ok=True)

        for item in tqdm(dataset, desc=f"Processing {os.path.basename(input_path)}"):
            passage = item.get("passages", [""])[0]

            augment_entry = {"passage": passage}

            # 1. 重写
            rewrite_prompt = rewrite_prompt_template.format(passage=passage)
            rewrite = model_generate(llm_model, llm_tokenizer, rewrite_prompt, max_new_tokens=max_new_tokens)
            augment_entry[f"{llm_model_name}_rewrite_original_output"] = rewrite
            augment_entry[f"{llm_model_name}_rewrite"] = rewrite if rewrite else ""

            # 2. QA生成
            qa_prompt = qa_prompt_template.format(passage=passage)
            qa_output = model_generate(llm_model, llm_tokenizer, qa_prompt, max_new_tokens=max_new_tokens)
            augment_entry[f"{llm_model_name}_qa_original_output"] = qa_output
            augment_entry[f"{llm_model_name}_qa"] = extract_qa_output(qa_output)

            item["augment"] = [augment_entry]
            saving_count+=1
            if saving_count%saving_steps==0:
                with open(output_path, "w") as fout:
                    json.dump(dataset, fout, indent=4, ensure_ascii=False)
            # break#test
        
        with open(output_path, "w") as fout:
            json.dump(dataset, fout, indent=4, ensure_ascii=False)
        print(f"[Saved] {output_path}")

def augment_with_deepseek_multipassage(input_paths: List[str], output_base_dir: str):
    # global current_dataset, current_output_path
    # setup_signal_handlers()
    saving_steps = 100
    saving_count=0
    client = OpenAI(
        api_key="sk-Okf9TYIr0BvKEetyZ7j5vtQyWBmE7EYuMhf5ayS5HSnijONO",
        base_url="https://api.chatanywhere.tech/v1",
    )

    rewrite_prompt_template = """
I will provide a passage. Please rewrite it while preserving all factual content, entities, and key terms. The rewritten version should be semantically faithful but use different wording. Output ONLY the rewritten passage, without any introductions or comments.

Passage:
{passage}

Rewritten Passage:
"""
    qa_prompt_template = (
    "I will provide multiple passages of text. Your task is to generate five different questions that can only be answered by synthesizing information from multiple passages. "
    "Each question must require combining information across all passages. Do not create questions that can be answered using only a single passage.\n\n"
    "To help you think more broadly, here are some types of questions you can consider:\n"
    "- **Comparison questions** (e.g., comparing entities or events mentioned in different passages)\n"
    "- **Cause-and-effect questions** (e.g., identifying how one event described in one passage influences or results in another event in a different passage)\n"
    "- **Temporal reasoning questions** (e.g., constructing a timeline or understanding chronological dependencies)\n"
    "- **Entity synthesis questions** (e.g., asking about a person/organization/concept that appears in multiple passages with different attributes)\n"
    "- **Theme or topic integration** (e.g., identifying common themes, contrasting perspectives, or overarching insights across documents)\n\n"
    "Each question should be answerable using the combined information provided in the set of passages. For each question, provide both a short answer and a full sentence answer.\n\n"
    "You must follow this format strictly:\n"
    "[\n"
    "    {{\n"
    "        \"question\": \"What is the capital of France?\",\n"
    "        \"answer\": \"Paris\",\n"
    "        \"full_answer\": \"The capital of France is Paris.\"\n"
    "    }},\n"
    "    ... (at least five in total)\n"
    "]\n\n"
    "Only output the list in this exact format.\n\n"
    "Passages:\n"
    "{passage}"
)

    for input_path in input_paths:
        with open(input_path, "r") as fin:
            dataset = json.load(fin)
        output_path = Path(output_base_dir)/Path(input_path).parts[-2]/Path(input_path).parts[-1]
        os.makedirs(output_path.parent, exist_ok=True)

        for item in tqdm(dataset, desc=f"Processing {os.path.basename(input_path)}"):
            passages = item.get("passages", [""])
            passage=""
            for idx, passage_ in enumerate(passages):
                passage+=f"Passage {idx+1}:\n"
                passage+=passage_
            augment_entry = {"passage": passage}

            # 1. 重写
            rewrites=[]
            for passage_ in passages:
                rewrite_prompt = rewrite_prompt_template.format(passage=passage_)
                rewrite = safe_call_gpt(rewrite_prompt, client)
                if rewrite:
                    rewrites.append(rewrite)
                else:
                    rewrites.append(passage_)
            augment_entry["ds3_rewrite_original_output"] = rewrites
            augment_entry["deepseekv3_rewrite"] = rewrites

            # 2. QA生成
            qa_prompt = qa_prompt_template.format(passage=passage)
            qa_output = safe_call_gpt(qa_prompt, client)
            augment_entry["ds3_qa_original_output"] = qa_output
            augment_entry["deepseekv3_qa"] = extract_qa_output(qa_output)

            item["augment"] = [augment_entry]
            saving_count+=1
            if saving_count%saving_steps==0:
                with open(output_path, "w") as fout:
                    json.dump(dataset, fout, indent=4, ensure_ascii=False)
            # break#test
        
        with open(output_path, "w") as fout:
            json.dump(dataset, fout, indent=4, ensure_ascii=False)
        print(f"[Saved] {output_path}")

def strip_augment_field(input_paths: List[str], output_base_dir: str):
    for input_path in input_paths:
        input_path = Path(input_path).resolve()

        with open(input_path, "r") as f:
            dataset = json.load(f)

        for item in tqdm(dataset, desc=f"Stripping 'augment' in {input_path.name}"):
            if "augment" in item:
                del item["augment"]

        # 构造输出路径：output_base_dir / dataset / split_dev.json
        dataset_name = input_path.parts[-3]  # e.g., 2wikimultihopqa
        split_name = input_path.stem         # e.g., bridge_comparison
        output_path = Path(output_base_dir) / dataset_name / f"{split_name}_dev.json"

        os.makedirs(output_path.parent, exist_ok=True)
        with open(output_path, "w") as fout:
            json.dump(dataset, fout, indent=4, ensure_ascii=False)

        print(f"[Saved] {output_path}")

def compute_kl_loss(student_hidden_states, teacher_hidden_states, attention_mask):
    total_loss = 0.0
    weight_sum = 0
    for i in range(1, len(student_hidden_states)):
        student = student_hidden_states[i]
        teacher = teacher_hidden_states[i]
        weight = i

        student_log_probs = F.log_softmax(student, dim=-1)
        teacher_probs = F.softmax(teacher.detach(), dim=-1)

        # Mask padding tokens
        mask = attention_mask.unsqueeze(-1).expand_as(student)
        student_log_probs = student_log_probs * mask
        teacher_probs = teacher_probs * mask

        loss_i = F.kl_div(student_log_probs, teacher_probs, reduction="sum") / mask.sum()
        total_loss += weight * loss_i
        weight_sum += weight
    return total_loss / weight_sum
def compute_cosine_similarity_loss(student_hidden_states, teacher_hidden_states, attention_mask):
    """
    计算所有层的 cosine 相似度 loss，加权平均。

    Args:
        student_hidden_states: List of [B, L, D]
        teacher_hidden_states: List of [B, L, D]
        attention_mask: [B, L], 1 表示非 PAD

    Returns:
        Scalar loss
    """
    total_loss = 0.0
    weight_sum = 0
    layer_losses = []
    layer_loss=None
    for i in range(1, len(student_hidden_states)):  # 跳过 embedding 层
        student = student_hidden_states[i]  # [B, L, D]
        teacher = teacher_hidden_states[i].detach()  # [B, L, D]
        weight = i

        cosine_sim = F.cosine_similarity(student, teacher, dim=-1)  # [B, L]
        loss_i = 1 - cosine_sim  # [B, L]

        # mask padding tokens
        mask = attention_mask  # [B, L]
        loss_i = loss_i * mask  # [B, L]
        layer_loss = loss_i.sum() / mask.sum()

        total_loss += weight * layer_loss
        weight_sum += weight
        layer_losses.append(layer_loss.item())
    return layer_loss, layer_losses
    # return total_loss / weight_sum, layer_losses

def compute_combined_loss(student_outputs, teacher_outputs, attention_mask, alpha=1, beta=1000):
    """
    结合余弦相似度与 KL 散度的对齐损失：
    - 前几层（除最后一层）用余弦相似度
    - 最后一层用 KL 散度
    """
    total_cosine_loss = 0.0
    cosine_weight_sum = 0
    student_hidden_states = student_outputs.hidden_states
    teacher_hidden_states = teacher_outputs.hidden_states
    student_logits = student_outputs.logits
    teacher_logits = teacher_outputs.logits.detach()

    # 前几层使用余弦相似度
    layer_losses = []
    for i in range(1, len(student_hidden_states)):  # 不包含最后一层
        student = student_hidden_states[i]
        teacher = teacher_hidden_states[i].detach()

        weight = i  # 层权重可改为非线性或统一

        # mask padding
        mask = attention_mask.unsqueeze(-1).expand_as(student)
        student = student * mask
        teacher = teacher * mask

        # cosine sim loss: 1 - sim
        cos_sim = F.cosine_similarity(student, teacher, dim=-1)
        cosine_loss = 1 - cos_sim  # shape: [batch, seq_len]
        cosine_loss = (cosine_loss * attention_mask).sum() / attention_mask.sum()
        layer_losses.append(cosine_loss.item())
        total_cosine_loss += weight * cosine_loss
        cosine_weight_sum += weight

    avg_cosine_loss = total_cosine_loss / cosine_weight_sum

    mask = attention_mask.unsqueeze(-1).expand_as(student_logits)
    student_log_probs = F.log_softmax(student_logits, dim=-1) * mask
    teacher_probs = F.softmax(teacher_logits, dim=-1) * mask
    kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='sum') / mask.sum()
    # layer_losses.append(kl_loss.item())
    # 总损失
    final_loss = alpha * avg_cosine_loss + beta * kl_loss
    return kl_loss*10000, layer_losses

def compute_multi_loss(
    student_outputs,
    teacher_outputs,
    attention_mask,
    input_ids,
    logits_label_mask=None,
    hidden_loss_type="cosine",  # "cosine" or "mse"
    logits_loss_type="kl",      # "kl" or "mse"
    alpha_zero=1.0,
    alpha=1.0,
    beta=1e5,
    gama = 1.0,
    temperature=1.0
):
    """
    综合损失函数：
    - hidden_loss_type: 对齐 hidden states 使用的损失类型 ("cosine", "mse")
    - logits_loss_type: 对齐 logits 使用的损失类型 ("kl", "mse")
    - return_logits_loss_only: 是否只返回 logits 部分的损失作为训练 loss
    """
    student_hidden_states = student_outputs.hidden_states
    teacher_hidden_states = teacher_outputs.hidden_states
    student_logits = student_outputs.logits
    teacher_logits = teacher_outputs.logits.detach()

    total_hidden_loss = 0.0
    hidden_weight_sum = 0
    layer_losses = []

    # === Hidden States 对齐 ===
    last_layer_loss = None
    for i in range(1, len(student_hidden_states)):  # 通常跳过 embedding 层 0
        student = student_hidden_states[i]
        teacher = teacher_hidden_states[i].detach()

        weight = i

        if hidden_loss_type == "cosine":
            cos_sim = F.cosine_similarity(student, teacher, dim=-1)
            loss = (1 - cos_sim) * attention_mask  # shape: [batch, seq]

        elif hidden_loss_type == "mse":
            loss = F.mse_loss(student, teacher, reduction='none')
            loss = loss.sum(dim=-1)* attention_mask
        else:
            raise ValueError(f"Unsupported hidden_loss_type: {hidden_loss_type}")
        loss = loss.sum() / attention_mask.sum()
        last_layer_loss = loss

        total_hidden_loss += weight * loss
        hidden_weight_sum += weight
        layer_losses.append(loss.item())

    avg_hidden_loss = total_hidden_loss / hidden_weight_sum if hidden_weight_sum > 0 else torch.tensor(0.0)

    # === Logits 对齐 ===
    if logits_label_mask is not None:
        mask_label = logits_label_mask.to(student_logits.device)
    else:
        mask_label = attention_mask
    mask_logits = mask_label * attention_mask
    if logits_loss_type == "kl":
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction="none")
        kl_loss = kl_loss.sum(dim=-1) * mask_logits
        logits_loss = kl_loss.sum() / mask_logits.sum() * (temperature ** 2)
    elif logits_loss_type == "mse":
        mse = F.mse_loss(student_logits, teacher_logits, reduction='none')
        mse = mse.sum(dim=-1) * mask_logits
        logits_loss = mse.sum() / mask_logits.sum()
    elif logits_loss_type == "cosine":
        logits_loss = F.cosine_similarity(student_logits, teacher_logits, dim=-1)
        logits_loss = (1 - logits_loss) * mask_logits
        logits_loss = logits_loss.sum() / mask_logits.sum()
    else:
        raise ValueError(f"Unsupported logits_loss_type: {logits_loss_type}")

    # === Cross-Entropy Loss（仅对mask_logits部分）===
    labels = logits_label_mask.to(student_logits.device) * attention_mask * input_ids
    labels[labels == 0] = -100
    shift_logits = student_logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    logits_flat = shift_logits.view(-1, shift_logits.size(-1))  # [B*T, V]
    labels_flat = shift_labels.view(-1)                         # [B*T]

    # 计算所有token的CE（不reduction）
    celoss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
    logits_ce_loss = celoss_fn(logits_flat, labels_flat) # [B*T]

    # === 组合输出 ===
    backpropagation_loss = alpha_zero*last_layer_loss + alpha * avg_hidden_loss + beta * logits_loss + gama * logits_ce_loss

    return backpropagation_loss, layer_losses, avg_hidden_loss, logits_loss, logits_ce_loss

def compute_multi_loss_nomask(
    student_outputs,
    teacher_outputs,
    attention_mask,
    input_ids,
    teacher_logits_label_mask=None,
    student_logits_label_mask=None,
    hidden_loss_type="cosine",  # "cosine" or "mse"
    logits_loss_type="kl",      # "kl" or "mse"
    alpha_zero=1.0,
    alpha=1.0,
    beta=1e5,
    gama = 1.0,
    temperature=1.0
):
    """
    综合损失函数：
    - hidden_loss_type: 对齐 hidden states 使用的损失类型 ("cosine", "mse")
    - logits_loss_type: 对齐 logits 使用的损失类型 ("kl", "mse")
    - return_logits_loss_only: 是否只返回 logits 部分的损失作为训练 loss
    """
    student_hidden_states = student_outputs.hidden_states
    teacher_hidden_states = teacher_outputs.hidden_states
    student_logits = student_outputs.logits
    teacher_logits = teacher_outputs.logits.detach()
    student_logits_label_mask_flat = student_logits_label_mask.view(-1).bool()
    teacher_logits_label_mask_flat = teacher_logits_label_mask.view(-1).bool()
    
    total_hidden_loss = 0.0
    hidden_weight_sum = 0
    layer_losses = []

    # === Hidden States 对齐 ===
    last_layer_loss = None
    for i in range(1, len(student_hidden_states)):  # 通常跳过 embedding 层 0
        student = student_hidden_states[i]
        teacher = teacher_hidden_states[i].detach()

        student_flat = student.view(-1, student.size(-1))[student_logits_label_mask_flat]
        teacher_flat = teacher.view(-1, teacher.size(-1))[teacher_logits_label_mask_flat]

        weight = i

        if hidden_loss_type == "cosine":
            cos_sim = F.cosine_similarity(student_flat, teacher_flat, dim=-1)
            loss = (1 - cos_sim)

        elif hidden_loss_type == "mse":
            loss = F.mse_loss(student_flat, teacher_flat, reduction='none')
            loss = loss.sum(dim=-1)
        else:
            raise ValueError(f"Unsupported hidden_loss_type: {hidden_loss_type}")
        loss = loss.sum() / student_logits_label_mask_flat.sum()
        last_layer_loss = loss

        total_hidden_loss += weight * loss
        hidden_weight_sum += weight
        layer_losses.append(loss.item())

    avg_hidden_loss = total_hidden_loss / hidden_weight_sum if hidden_weight_sum > 0 else torch.tensor(0.0)

    # === Logits 对齐 ===
    student_logits_flat = student_logits.view(-1, student_logits.size(-1))[student_logits_label_mask_flat]
    teacher_logits_flat = teacher_logits.view(-1, teacher_logits.size(-1))[teacher_logits_label_mask_flat]

    if logits_loss_type == "kl":
        student_log_probs = F.log_softmax(student_logits_flat / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits_flat / temperature, dim=-1)
        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction="none")
        logits_loss = kl_loss.sum(dim=-1)
        logits_loss = logits_loss.sum() / student_logits_label_mask_flat.sum() * (temperature ** 2)
    elif logits_loss_type == "mse":
        mse = F.mse_loss(student_logits_flat, teacher_logits_flat, reduction='none')
        logits_loss = mse.sum(dim=-1)
        logits_loss = logits_loss.sum() / student_logits_label_mask_flat.sum()

    elif logits_loss_type == "cosine":
        logits_loss = F.cosine_similarity(student_logits_flat, teacher_logits_flat, dim=-1)
        logits_loss = (1 - logits_loss).sum() / student_logits_label_mask_flat.sum()
    else:
        raise ValueError(f"Unsupported logits_loss_type: {logits_loss_type}")

    # === Cross-Entropy Loss（仅对mask_logits部分）===
    mask = (student_logits_label_mask == 0) | (attention_mask.to(student_logits_label_mask.device) == 0)
    labels = input_ids.masked_fill(mask.to(input_ids.device), -100)
    shift_logits = student_logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    logits_flat = shift_logits.view(-1, shift_logits.size(-1))  # [B*T, V]
    labels_flat = shift_labels.view(-1)                         # [B*T]

    # 计算所有token的CE（不reduction）
    celoss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
    logits_ce_loss = celoss_fn(logits_flat, labels_flat) # [B*T]

    # === 组合输出 ===
    backpropagation_loss = alpha_zero*last_layer_loss + alpha * avg_hidden_loss + beta * logits_loss + gama * logits_ce_loss
    # 0.23 0.16 4.52 4.46
    return backpropagation_loss, layer_losses, avg_hidden_loss, logits_loss, logits_ce_loss


def stract_data_field(input_paths: List[str], output_dir: str):
    passages=[]
    questions=[]
    answers=[]
    full_answers=[]    
    for input_path in input_paths:
        input_path = Path(input_path).resolve()
        with open(input_path, "r") as f:
            dataset = json.load(f)
        for item in tqdm(dataset, desc=f"stract data in {input_path.name}"):
            for augment in item["augment"]:
                if augment['deepseekv3_rewrite'] == "":
                    print("index:",item["test_id"])
                    continue
                for qa in augment['deepseekv3_qa']:
                    passages.append(augment['deepseekv3_rewrite'])
                    questions.append(qa['question'])
                    answers.append(qa['answer'])
                    full_answers.append(qa['full_answer'])
    
    dataset = Dataset.from_dict({
        "passage": passages,
        "question": questions,
        "answer": answers,
        "full_answer": full_answers,
    })
    dataset.save_to_disk(output_dir)

def init_and_save_tokenizer_embedding(llm_model, tokenizer, doc_mask_token, save_path):
    # 设置 padding
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # 添加新 token
    tokenizer.add_tokens([doc_mask_token])
    llm_model.resize_token_embeddings(len(tokenizer))  # 自动初始化新 embedding

    # 获取 embedding 权重
    token_embeddings = llm_model.get_input_embeddings().weight  # [vocab_size, hidden_dim]


    # 检查统计分布是否一致（使用全部 embedding）
    ref_mean = token_embeddings.mean().item()
    ref_std = token_embeddings.std().item()

    token_id = tokenizer.convert_tokens_to_ids(doc_mask_token)

    with torch.no_grad():
        mean = token_embeddings.mean(dim=0)
        std = token_embeddings.std(dim=0)
        noise = torch.randn_like(mean) * std
        token_embeddings[token_id] = mean + noise

    new_emb = token_embeddings[token_id]
    new_mean = new_emb.mean().item()
    new_std = new_emb.std().item()

    print(f"\n📊 Full embedding matrix: mean = {ref_mean:.4f}, std = {ref_std:.4f}")
    print(f"🆕 New token '{doc_mask_token}': mean = {new_mean:.4f}, std = {new_std:.4f}")

    if not (torch.isfinite(new_emb).all() and torch.isfinite(token_embeddings).all()):
        print("⚠️  Embedding contains NaNs or Infs. Check upstream initialization.")
    else:
        mean_diff = abs(new_mean - ref_mean) / (abs(ref_mean) + 1e-6)
        std_diff = abs(new_std - ref_std) / (ref_std + 1e-6)

        if mean_diff < 0.1 and std_diff < 0.1:
            print("✅ New token embedding looks statistically aligned with full embedding set.\n")
        else:
            print("⚠️  New token embedding might be too different. Consider inspecting further.\n")

    # 保存所有组件
    os.makedirs(save_path, exist_ok=True)
    
    llm_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    if hasattr(llm_model, "generation_config"):
        llm_model.generation_config.save_pretrained(save_path)

    print(f"✅ Model and tokenizer with '{doc_mask_token}' saved to: {save_path}")

def inference_on_dataset(dataset_path: str, model_name="./models/Llama-3.2-1B-Instruct", max_new_tokens=256, batch_size=4):
    '''
    features: ['passage', 'question', 'answer'],
    '''
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda:6", torch_dtype=torch.float16)
    model.generation_config.temperature=None
    model.generation_config.top_p=None
    dataset = load_from_disk(dataset_path)
    field_name = f"answer_{model_name.split('/')[-1]}"
    
    # 先构造 prompts 列表
    prompts = []
    for example in dataset:
        user_msg = {
            "role": "user",
            "content": (
                "You should answer the question by referring to the knowledge provided below and integrating your own knowledge.\n"
                f"Passages:\n{example['passage']}\n\nQuestion: {example['question']}"
            )
        }
        assistant_msg = {"role": "assistant", "content": "Answer:\n"}

        prompt = tokenizer.apply_chat_template(
            [user_msg, assistant_msg],
            tokenize=False,
            add_generation_prompt=False,
        ).removesuffix("<|eot_id|>")
        prompts.append(prompt)

    # 批量推理
    answers = []
    prograss_bar = tqdm(total=len(prompts), desc="Inference")
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        decoded = tokenizer.batch_decode(output_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        answers.extend([ans.strip() for ans in decoded])
        prograss_bar.update(len(batch_prompts))
    prograss_bar.close()

    # 用 map 方式添加新字段
    def add_generated_answer(example, idx):
        example[field_name] = answers[idx]
        return example

    dataset = dataset.map(add_generated_answer, with_indices=True)

    # 保存新数据集
    save_path = f"{dataset_path}_{model_name.split('/')[-1]}"
    dataset.save_to_disk(save_path)
    print(f"✅ 推理完成，保存至：{save_path}")