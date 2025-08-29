import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from imports_104 import *
from utils_104 import *
from train_my_dyprag_104 import ParameterTranslator, CrossAttentionParameterTranslator,CrossAttentionHyperNetworkParameterTranslator

def inference_on_jsondataset_with_mydyprag_masking(
    input_json_path: str,
    output_json_path: str,
    llm_model_path: str = "./models/Llama-3.2-1B-Instruct",
    translator_type: str = "parameter-translator",
    doc_mask_token: str = "<|doc_mask|>",
    max_new_tokens: int = 256,
    batch_size: int = 4,
    device: str = "cuda:0",
    embedding_model_path: str = "./models/long-t5-tglobal-base",
    translator_path: str = "models/Llama-3.2-1B-Instruct-longt5_alignment1/translator_step_30000.safetensors",
):
    
    def _get_doc_embed(passages):
        def _bge_or_snowflake_embed(model, inputs):
            output = model(**inputs)
            embeddings = output[0][:, 0]  # 取 [CLS] token
            return torch.nn.functional.normalize(embeddings, p=2, dim=1)

        def _t5_embed(model, inputs):
            output = model.encoder(**inputs)
            embeddings = output.last_hidden_state
            return torch.mean(embeddings, dim=1)

        with torch.no_grad():
            inputs = embedding_tokenizer(
                passages,
                return_tensors="pt",
                padding="longest",
                max_length=4096,
                truncation=True
            ).to(device)

            model_name = embedding_model.name_or_path.lower()

            if 'bge-large-en-v1.5' in model_name or 'snowflake' in model_name:
                sentence_embeddings = _bge_or_snowflake_embed(embedding_model, inputs)
            elif 't5' in model_name:
                sentence_embeddings = _t5_embed(embedding_model, inputs)
            else:
                raise NotImplementedError(f"[Unsupported Model] {model_name}")

        return sentence_embeddings
    def _get_full_doc_embed(passages):

        def _bge_or_snowflake_embed(model, inputs):
            output = model(**inputs)
            embeddings = output.last_hidden_state #[B, L, D]
            return torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        
        def _default_embed(model, inputs):
            output = model.encoder(**inputs)
            embeddings = output.last_hidden_state #[B, L, D]
            return torch.nn.functional.normalize(embeddings, p=2, dim=-1)

        def _t5_embed(model, inputs):
            output = model.encoder(**inputs)
            embeddings = output.last_hidden_state
            return embeddings #[B, L, D]

        with torch.no_grad():
            inputs = embedding_tokenizer(
                passages,
                return_tensors="pt",
                padding="longest",
                max_length=4096,
                truncation=True
            ).to(device)

            model_name = embedding_model.name_or_path.lower()

            if 't5' in model_name:
                sentence_embeddings = _t5_embed(embedding_model, inputs)
            else:
                sentence_embeddings = _default_embed(embedding_model, inputs)

        return sentence_embeddings, inputs["attention_mask"]


    # 载入模型与tokenizer
    tokenizer = AutoTokenizer.from_pretrained(llm_model_path, padding_side="left")
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eos_token_id
    embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_path)
    llm_model = AutoModelForCausalLM.from_pretrained(llm_model_path, device_map=device)
    llm_model.generation_config.temperature = None
    llm_model.generation_config.top_p = None
    if 'snowflake' in embedding_model_path:
        embedding_model = AutoModel.from_pretrained(embedding_model_path, add_pooling_layer=False, trust_remote_code=True,device_map=device)
    else:
        embedding_model = AutoModel.from_pretrained(embedding_model_path,device_map=device)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=2,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=["down_proj", "up_proj", "gate_proj"],  # 指定 LoRA 注入的模块
    )
    llm_model = get_peft_model(llm_model, peft_config)
    llm_model.config.pad_token_id = tokenizer.pad_token_id
    
    dir_name = os.path.dirname(translator_path)
    step = translator_path.split("_")[-1].split(".")[0]
    doc_mask_token_path = os.path.join(dir_name, f"doc_mask_token_{step}.safetensors")
    if os.path.exists(doc_mask_token_path):
        doc_mask_emb = load_file(doc_mask_token_path, device=device)['doc_mask_token']
        doc_mask_token_id = tokenizer.convert_tokens_to_ids(doc_mask_token)
        with torch.no_grad():
            llm_model.get_input_embeddings().weight[doc_mask_token_id] = doc_mask_emb
    if translator_type == "parameter-translator":
        translator = ParameterTranslator(
            embedding_model=embedding_model,
            llm_model=llm_model,
            lora_rank=2,
            projector_hidden_dim=512
        )#66M
    elif translator_type == "cross-attention-parameter-translator-s":
        translator = CrossAttentionParameterTranslator(
            embedding_model=embedding_model,
            llm_model=llm_model,
            lora_rank=2,
            projector_hidden_dim=1024,
            attn_heads=8,
            attn_ff_dim=1024,
            cross_layers=1,
            encoder_layers=1,
        )#
    elif translator_type == "cross-attention-parameter-translator-l":
        translator = CrossAttentionParameterTranslator(
            embedding_model=embedding_model,
            llm_model=llm_model,
            lora_rank=2,
            projector_hidden_dim=2560,
            attn_heads=8,
            attn_ff_dim=1024,
            cross_layers=4,
            encoder_layers=4,
        )#239M
    elif translator_type == "cross-attention-hyper-parameter-translator":
        translator = CrossAttentionHyperNetworkParameterTranslator(
            embedding_model=embedding_model,
            llm_model=llm_model,
            lora_rank=2,
            projector_hidden_dim=2048,
            attn_heads=8,
            cross_layers=1,
        )
    else:
        raise ValueError(f"Unknown translator type: {translator_type}")
    translator.to(device)
    translator.load_state_dict(load_file(translator_path, device=device))
    # translator.eval()

    # 构造输出字段名
    suffix=Path(translator_path).parts[-2]+"-"+Path(translator_path).parts[-1].removesuffix(".safetensors")
    field_name = f"answer_{suffix}"

    # 读取数据
    with open(input_json_path, "r") as f:
        data = json.load(f)

    # 构造 teacher 和 student inputs（最终推理只用 student）
    teacher_inputs = []
    student_inputs = []

    questions = []
    passages = []
    user_template = "You should answer the question by referring to the knowledge provided below and integrating your own knowledge.\nPassages:\n{passage}\n\nQuestion: {question}"
    # user_template = "You should answer the question by referring to the knowledge provided below and integrating your own knowledge. You must answer in a concise manner without any additional explanation.\nPassages:\n{passage}\n\nQuestion: {question}"
    if "Mistral" in llm_model_path:
        assistant_template = "The answer is"
    else:
        assistant_template = "The answer is "
    for item in data:
        q = item["question"]
        p=""
        for idx,passage in enumerate(item["passages"]):
            p+=f"Passage {idx+1}:\n{passage}\n"

        questions.append(q)
        passages.append(p)

        user_msg = {
            "role": "user",
            "content": user_template.format(passage=p, question=q)
        }
        assistant_msg = {"role": "assistant", "content": assistant_template}

        teacher_inputs.append([user_msg, assistant_msg])

        masked_doc = doc_mask_token * 2
        masked_user_msg = {
            "role": "user",
            "content": user_template.format(passage=masked_doc, question=q)
        }
        student_inputs.append([masked_user_msg, assistant_msg])

    # 初次 token 化比较长度
    teacher_texts = [tokenizer.apply_chat_template(x, tokenize=False).removesuffix("<|eot_id|>").removesuffix("</s>") for x in teacher_inputs]
    student_texts = [tokenizer.apply_chat_template(x, tokenize=False).removesuffix("<|eot_id|>").removesuffix("</s>") for x in student_inputs]

    teacher_tokenized = [tokenizer(teacher_text, return_tensors="pt", truncation=True, max_length=4096, add_special_tokens=False) for teacher_text in teacher_texts]
    student_tokenized = [tokenizer(student_text, return_tensors="pt", truncation=True, max_length=4096, add_special_tokens=False) for student_text in student_texts]
    # 动态扩展 mask token 以保证 student 输入不短于 teacher
    for i in range(len(teacher_texts)):
        t_len = teacher_tokenized[i]["input_ids"].shape[-1]
        s_len = student_tokenized[i]["input_ids"].shape[-1]
        delta = t_len - s_len
        q = data[i]["question"]
        if delta > 0:
            new_mask = doc_mask_token * (2 + delta)
            updated_msg = {
                "role": "user",
                "content": user_template.format(passage=new_mask, question=q)
            }
            student_inputs[i][0] = updated_msg

    # 重新构建 student prompt（用于推理）
    prompts = [tokenizer.apply_chat_template(x, tokenize=False).removesuffix("<|eot_id|>").removesuffix("</s>") for x in student_inputs]

    # 推理过程
    answers_out = []
    pbar = tqdm(total=len(prompts), desc="Running inference")
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False, max_length=2048).to(llm_model.device)
        if translator_type == "parameter-translator":
            doc_embed = _get_doc_embed(passages[i:i + batch_size])
            with torch.no_grad():
                lora_weights = translator(doc_embed)
        elif "cross-attention-parameter-translator" in translator_type:
            doc_embed,attention_mask = _get_full_doc_embed(passages[i:i + batch_size])
            with torch.no_grad():
                lora_weights = translator(doc_embed,attention_mask.to(translator.device))
        elif translator_type == "cross-attention-hyper-parameter-translator":
            doc_embed,attention_mask = _get_full_doc_embed(passages[i:i + batch_size])
            with torch.no_grad():
                lora_weights = translator(doc_embed,attention_mask.to(translator.device))
        else:
            raise ValueError(f"Unknown translator type: {translator_type}")
        delta_inject(llm_model, lora_weights)
        with torch.no_grad():
            output_ids = llm_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        decoded = tokenizer.batch_decode(output_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        answers_out.extend([ans.strip() for ans in decoded])
        pbar.update(len(batch_prompts))
    pbar.close()

    # 写入结果
    for item, pred in zip(data, answers_out):
        item[field_name] = pred

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"✅ 推理完成，保存至：{output_json_path}")
def inference_on_jsondataset_with_mydyprag_masking_concise(
    input_json_path: str,
    output_json_path: str,
    llm_model_path: str = "./models/Llama-3.2-1B-Instruct",
    translator_type: str = "parameter-translator",
    doc_mask_token: str = "<|doc_mask|>",
    max_new_tokens: int = 256,
    batch_size: int = 4,
    device: str = "cuda:0",
    embedding_model_path: str = "./models/long-t5-tglobal-base",
    translator_path: str = "models/Llama-3.2-1B-Instruct-longt5_alignment1/translator_step_30000.safetensors",
):
    
    def _get_doc_embed(passages):
        def _bge_or_snowflake_embed(model, inputs):
            output = model(**inputs)
            embeddings = output[0][:, 0]  # 取 [CLS] token
            return torch.nn.functional.normalize(embeddings, p=2, dim=1)

        def _t5_embed(model, inputs):
            output = model.encoder(**inputs)
            embeddings = output.last_hidden_state
            return torch.mean(embeddings, dim=1)

        with torch.no_grad():
            inputs = embedding_tokenizer(
                passages,
                return_tensors="pt",
                padding="longest",
                max_length=4096,
                truncation=True
            ).to(device)

            model_name = embedding_model.name_or_path.lower()

            if 'bge' in model_name or 'snowflake' in model_name:
                sentence_embeddings = _bge_or_snowflake_embed(embedding_model, inputs)
            elif 't5' in model_name:
                sentence_embeddings = _t5_embed(embedding_model, inputs)
            else:
                raise NotImplementedError(f"[Unsupported Model] {model_name}")

        return sentence_embeddings
    def _get_full_doc_embed(passages):

        def _bge_or_snowflake_embed(model, inputs):
            output = model(**inputs)
            embeddings = output.last_hidden_state #[B, L, D]
            return torch.nn.functional.normalize(embeddings, p=2, dim=-1)

        def _t5_embed(model, inputs):
            output = model.encoder(**inputs)
            embeddings = output.last_hidden_state
            return embeddings #[B, L, D]

        with torch.no_grad():
            inputs = embedding_tokenizer(
                passages,
                return_tensors="pt",
                padding="longest",
                max_length=4096,
                truncation=True
            ).to(device)

            model_name = embedding_model.name_or_path.lower()

            if 'bge' in model_name or 'snowflake' in model_name:
                sentence_embeddings = _bge_or_snowflake_embed(embedding_model, inputs)
            elif 't5' in model_name:
                sentence_embeddings = _t5_embed(embedding_model, inputs)
            else:
                raise NotImplementedError(f"[Unsupported Model] {model_name}")

        return sentence_embeddings, inputs["attention_mask"]


    # 载入模型与tokenizer
    tokenizer = AutoTokenizer.from_pretrained(llm_model_path, padding_side="left")
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eos_token_id
    embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_path)
    llm_model = AutoModelForCausalLM.from_pretrained(llm_model_path, device_map=device)
    llm_model.generation_config.temperature = None
    llm_model.generation_config.top_p = None
    if 'snowflake' in embedding_model_path:
        embedding_model = AutoModel.from_pretrained(embedding_model_path, add_pooling_layer=False, trust_remote_code=True,device_map=device)
    else:
        embedding_model = AutoModel.from_pretrained(embedding_model_path,device_map=device)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=2,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=["down_proj", "up_proj", "gate_proj"],  # 指定 LoRA 注入的模块
    )
    llm_model = get_peft_model(llm_model, peft_config)
    llm_model.config.pad_token_id = tokenizer.pad_token_id
    
    dir_name = os.path.dirname(translator_path)
    step = translator_path.split("_")[-1].split(".")[0]
    doc_mask_token_path = os.path.join(dir_name, f"doc_mask_token_{step}.safetensors")
    if os.path.exists(doc_mask_token_path):
        print(f"Loading doc_mask_token from {doc_mask_token_path}")
        doc_mask_emb = load_file(doc_mask_token_path, device=device)['doc_mask_token']
        doc_mask_token_id = tokenizer.convert_tokens_to_ids(doc_mask_token)
        with torch.no_grad():
            llm_model.get_input_embeddings().weight[doc_mask_token_id] = doc_mask_emb
    if translator_type == "parameter-translator":
        translator = ParameterTranslator(
            embedding_model=embedding_model,
            llm_model=llm_model,
            lora_rank=2,
            projector_hidden_dim=512
        )#66M
    elif translator_type == "cross-attention-parameter-translator-s":
        translator = CrossAttentionParameterTranslator(
            embedding_model=embedding_model,
            llm_model=llm_model,
            lora_rank=2,
            projector_hidden_dim=1024,
            attn_heads=8,
            attn_ff_dim=1024,
            cross_layers=1,
            encoder_layers=1,
        )#
    elif translator_type == "cross-attention-parameter-translator-l":
        translator = CrossAttentionParameterTranslator(
            embedding_model=embedding_model,
            llm_model=llm_model,
            lora_rank=2,
            projector_hidden_dim=2560,
            attn_heads=8,
            attn_ff_dim=1024,
            cross_layers=4,
            encoder_layers=4,
        )#239M
    elif translator_type == "cross-attention-hyper-parameter-translator":
        translator = CrossAttentionHyperNetworkParameterTranslator(
            embedding_model=embedding_model,
            llm_model=llm_model,
            lora_rank=2,
            projector_hidden_dim=2048,
            attn_heads=8,
            cross_layers=1,
        )
    else:
        raise ValueError(f"Unknown translator type: {translator_type}")
    translator.to(device)
    translator.load_state_dict(load_file(translator_path, device=device))
    # translator.eval()

    # 构造输出字段名
    suffix=Path(translator_path).parts[-2]+"-"+Path(translator_path).parts[-1].removesuffix(".safetensors")
    field_name = f"answer_{suffix}_concise"

    # 读取数据
    with open(input_json_path, "r") as f:
        data = json.load(f)

    # 构造 teacher 和 student inputs（最终推理只用 student）
    teacher_inputs = []
    student_inputs = []

    questions = []
    passages = []
    # user_template = "You should answer the question by referring to the knowledge provided below and integrating your own knowledge.\nPassages:\n{passage}\n\nQuestion: {question}"
    user_template = "You should answer the question by referring to the knowledge provided below and integrating your own knowledge. You must answer in a concise manner without any additional explanation.\nPassages:\n{passage}\n\nQuestion: {question}"
    if "Mistral" in llm_model_path:
        assistant_template = "The answer is"
    else:
        assistant_template = "The answer is "
    for item in data:
        q = item["question"]
        p=""
        for idx,passage in enumerate(item["passages"]):
            p+=f"Passage {idx+1}:\n{passage}\n"

        questions.append(q)
        passages.append(p)

        user_msg = {
            "role": "user",
            "content": user_template.format(passage=p, question=q)
        }
        assistant_msg = {"role": "assistant", "content": assistant_template}

        teacher_inputs.append([user_msg, assistant_msg])

        masked_doc = doc_mask_token * 2
        masked_user_msg = {
            "role": "user",
            "content": user_template.format(passage=masked_doc, question=q)
        }
        student_inputs.append([masked_user_msg, assistant_msg])

    # 初次 token 化比较长度
    teacher_texts = [tokenizer.apply_chat_template(x, tokenize=False).removesuffix("<|eot_id|>").removesuffix("</s>") for x in teacher_inputs]
    student_texts = [tokenizer.apply_chat_template(x, tokenize=False).removesuffix("<|eot_id|>").removesuffix("</s>") for x in student_inputs]

    teacher_tokenized = [tokenizer(teacher_text, return_tensors="pt", truncation=True, max_length=4096, add_special_tokens=False) for teacher_text in teacher_texts]
    student_tokenized = [tokenizer(student_text, return_tensors="pt", truncation=True, max_length=4096, add_special_tokens=False) for student_text in student_texts]
    # 动态扩展 mask token 以保证 student 输入不短于 teacher
    for i in range(len(teacher_texts)):
        t_len = teacher_tokenized[i]["input_ids"].shape[-1]
        s_len = student_tokenized[i]["input_ids"].shape[-1]
        delta = t_len - s_len
        q = data[i]["question"]
        if delta > 0:
            new_mask = doc_mask_token * (2 + delta)
            updated_msg = {
                "role": "user",
                "content": user_template.format(passage=new_mask, question=q)
            }
            student_inputs[i][0] = updated_msg

    # 重新构建 student prompt（用于推理）
    prompts = [tokenizer.apply_chat_template(x, tokenize=False).removesuffix("<|eot_id|>").removesuffix("</s>") for x in student_inputs]

    # 推理过程
    answers_out = []
    pbar = tqdm(total=len(prompts), desc="Running inference")
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False, max_length=2048).to(llm_model.device)
        if translator_type == "parameter-translator":
            doc_embed = _get_doc_embed(passages[i:i + batch_size])
            with torch.no_grad():
                lora_weights = translator(doc_embed)
        elif "cross-attention-parameter-translator" in translator_type:
            doc_embed,attention_mask = _get_full_doc_embed(passages[i:i + batch_size])
            with torch.no_grad():
                lora_weights = translator(doc_embed,attention_mask.to(translator.device))
        elif translator_type == "cross-attention-hyper-parameter-translator":
            doc_embed,attention_mask = _get_full_doc_embed(passages[i:i + batch_size])
            with torch.no_grad():
                lora_weights = translator(doc_embed,attention_mask.to(translator.device))
        else:
            raise ValueError(f"Unknown translator type: {translator_type}")
        delta_inject(llm_model, lora_weights)
        with torch.no_grad():
            output_ids = llm_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        decoded = tokenizer.batch_decode(output_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        answers_out.extend([ans.strip() for ans in decoded])
        pbar.update(len(batch_prompts))
    pbar.close()

    # 写入结果
    for item, pred in zip(data, answers_out):
        item[field_name] = pred

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"✅ 推理完成，保存至：{output_json_path}")

def inference_on_jsondataset_with_mydyprag_nomask_concise(
    input_json_path: str,
    output_json_path: str,
    llm_model_path: str = "./models/Llama-3.2-1B-Instruct",
    translator_type: str = "parameter-translator",
    doc_mask_token: str = "",
    max_new_tokens: int = 256,
    batch_size: int = 4,
    device: str = "cuda:0",
    embedding_model_path: str = "./models/long-t5-tglobal-base",
    translator_path: str = "models/Llama-3.2-1B-Instruct-longt5_alignment1/translator_step_30000.safetensors",
):
    
    def _get_doc_embed(passages):
        def _bge_or_snowflake_embed(model, inputs):
            output = model(**inputs)
            embeddings = output[0][:, 0]  # 取 [CLS] token
            return torch.nn.functional.normalize(embeddings, p=2, dim=1)

        def _t5_embed(model, inputs):
            output = model.encoder(**inputs)
            embeddings = output.last_hidden_state
            return torch.mean(embeddings, dim=1)

        with torch.no_grad():
            inputs = embedding_tokenizer(
                passages,
                return_tensors="pt",
                padding="longest",
                max_length=4096,
                truncation=True
            ).to(device)

            model_name = embedding_model.name_or_path.lower()

            if 'bge' in model_name or 'snowflake' in model_name:
                sentence_embeddings = _bge_or_snowflake_embed(embedding_model, inputs)
            elif 't5' in model_name:
                sentence_embeddings = _t5_embed(embedding_model, inputs)
            else:
                raise NotImplementedError(f"[Unsupported Model] {model_name}")

        return sentence_embeddings
    def _get_full_doc_embed(passages):

        def _bge_or_snowflake_embed(model, inputs):
            output = model(**inputs)
            embeddings = output.last_hidden_state #[B, L, D]
            return torch.nn.functional.normalize(embeddings, p=2, dim=-1)

        def _t5_embed(model, inputs):
            output = model.encoder(**inputs)
            embeddings = output.last_hidden_state
            return embeddings #[B, L, D]

        with torch.no_grad():
            inputs = embedding_tokenizer(
                passages,
                return_tensors="pt",
                padding="longest",
                max_length=4096,
                truncation=True
            ).to(device)

            model_name = embedding_model.name_or_path.lower()

            if 'bge' in model_name or 'snowflake' in model_name:
                sentence_embeddings = _bge_or_snowflake_embed(embedding_model, inputs)
            elif 't5' in model_name:
                sentence_embeddings = _t5_embed(embedding_model, inputs)
            else:
                raise NotImplementedError(f"[Unsupported Model] {model_name}")

        return sentence_embeddings, inputs["attention_mask"]


    # 载入模型与tokenizer
    tokenizer = AutoTokenizer.from_pretrained(llm_model_path, padding_side="left")
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eos_token_id
    embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_path)
    llm_model = AutoModelForCausalLM.from_pretrained(llm_model_path, device_map=device)
    llm_model.generation_config.temperature = None
    llm_model.generation_config.top_p = None
    if 'snowflake' in embedding_model_path:
        embedding_model = AutoModel.from_pretrained(embedding_model_path, add_pooling_layer=False, trust_remote_code=True,device_map=device)
    else:
        embedding_model = AutoModel.from_pretrained(embedding_model_path,device_map=device)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=2,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=["down_proj", "up_proj", "gate_proj"],  # 指定 LoRA 注入的模块
    )
    llm_model = get_peft_model(llm_model, peft_config)
    llm_model.config.pad_token_id = tokenizer.pad_token_id
    if translator_type == "parameter-translator":
        translator = ParameterTranslator(
            embedding_model=embedding_model,
            llm_model=llm_model,
            lora_rank=2,
            projector_hidden_dim=512
        )#66M
    elif translator_type == "cross-attention-parameter-translator-s":
        translator = CrossAttentionParameterTranslator(
            embedding_model=embedding_model,
            llm_model=llm_model,
            lora_rank=2,
            projector_hidden_dim=1024,
            attn_heads=8,
            attn_ff_dim=1024,
            cross_layers=1,
            encoder_layers=1,
        )#
    elif translator_type == "cross-attention-parameter-translator-l":
        translator = CrossAttentionParameterTranslator(
            embedding_model=embedding_model,
            llm_model=llm_model,
            lora_rank=2,
            projector_hidden_dim=2560,
            attn_heads=8,
            attn_ff_dim=1024,
            cross_layers=4,
            encoder_layers=4,
        )#239M
    elif translator_type == "cross-attention-hyper-parameter-translator":
        translator = CrossAttentionHyperNetworkParameterTranslator(
            embedding_model=embedding_model,
            llm_model=llm_model,
            lora_rank=2,
            projector_hidden_dim=2048,
            attn_heads=8,
            cross_layers=1,
        )
    else:
        raise ValueError(f"Unknown translator type: {translator_type}")
    translator.to(device)
    translator.load_state_dict(load_file(translator_path, device=device))
    # translator.eval()

    # 构造输出字段名
    suffix=Path(translator_path).parts[-2]+"-"+Path(translator_path).parts[-1].removesuffix(".safetensors")
    field_name = f"answer_{suffix}_concise"

    # 读取数据
    with open(input_json_path, "r") as f:
        data = json.load(f)

    student_inputs = []

    questions = []
    passages = []
    user_template = "You should answer the question by referring to the knowledge provided below and integrating your own knowledge.\nPassages:\n{passage}\n\nQuestion: {question}"
    assistant_template = "The answer is "
    for item in data:
        q = item["question"]
        p=""
        for idx,passage in enumerate(item["passages"]):
            p+=f"Passage {idx+1}:\n{passage}\n"

        questions.append(q)
        passages.append(p)

        user_msg = {
            "role": "user",
            "content": user_template.format(passage="", question=q)
        }
        assistant_msg = {"role": "assistant", "content": assistant_template}
        student_inputs.append([user_msg, assistant_msg])

    prompts = [tokenizer.apply_chat_template(x, tokenize=False, add_special_tokens=False).removesuffix("<|eot_id|>").removesuffix("</s>") for x in student_inputs]

    # 推理过程
    answers_out = []
    pbar = tqdm(total=len(prompts), desc="Running inference")
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False).to(llm_model.device)
        if translator_type == "parameter-translator":
            doc_embed = _get_doc_embed(passages[i:i + batch_size])
            with torch.no_grad():
                lora_weights = translator(doc_embed)
        elif "cross-attention-parameter-translator" in translator_type:
            doc_embed,attention_mask = _get_full_doc_embed(passages[i:i + batch_size])
            with torch.no_grad():
                lora_weights = translator(doc_embed,attention_mask.to(translator.device))
        else:
            raise ValueError(f"Unknown translator type: {translator_type}")
        delta_inject(llm_model, lora_weights)
        with torch.no_grad():
            output_ids = llm_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        decoded = tokenizer.batch_decode(output_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        answers_out.extend([ans.strip() for ans in decoded])
        pbar.update(len(batch_prompts))
    pbar.close()

    # 写入结果
    for item, pred in zip(data, answers_out):
        item[field_name] = pred

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"✅ 推理完成，保存至：{output_json_path}")

def inference_single_sample(
    question: str,
    passages: list[str],
    llm_model_path: str = "./models/Llama-3.2-1B-Instruct",
    doc_mask_token: str = "",
    max_new_tokens: int = 256,
    device: str = "cuda:0",
    embedding_model_path: str = "./models/long-t5-tglobal-base",
    translator_path: str = "models/Llama-3.2-1B-Instruct-longt5_alignment1/translator_step_30000.safetensors",
):
    # 加载模型
    tokenizer = AutoTokenizer.from_pretrained(llm_model_path, padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_path, padding_side="left")

    llm_model = AutoModelForCausalLM.from_pretrained(llm_model_path, device_map=device)
    llm_model.generation_config.temperature = None
    llm_model.generation_config.top_p = None

    if 'snowflake' in embedding_model_path:
        embedding_model = AutoModel.from_pretrained(embedding_model_path, add_pooling_layer=False, trust_remote_code=True, device_map=device)
    else:
        embedding_model = AutoModel.from_pretrained(embedding_model_path, device_map=device)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=2,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=["down_proj", "up_proj", "gate_proj"],
    )
    llm_model = get_peft_model(llm_model, peft_config)
    llm_model.config.pad_token_id = tokenizer.pad_token_id

    translator = ParameterTranslator(
        embedding_model=embedding_model,
        llm_model=llm_model,
        lora_rank=2,
        projector_hidden_dim=512
    ).to(device)
    translator.load_state_dict(load_file(translator_path, device=device))
    # translator.eval()

    # 构建输入
    passage_text = "\n".join(passages)
    user_msg = {
        "role": "user",
        "content": (
            "You should answer the question by referring to the knowledge provided below and integrating your own knowledge.\n"
            f"Passages:\n{passage_text}\n\nQuestion: {question}"
        )
    }
    assistant_msg = {"role": "assistant", "content": "The answer is "}
    teacher_input = [user_msg, assistant_msg]

    masked_user_msg = {
        "role": "user",
        "content": (
            "You should answer the question by referring to the knowledge provided below and integrating your own knowledge.\n"
            f"Passages:\n{doc_mask_token * 2}\n\nQuestion: {question}"
        )
    }
    student_input = [masked_user_msg, assistant_msg]

    # 初次编码
    teacher_prompt = tokenizer.apply_chat_template(teacher_input, tokenize=False).removesuffix("<|eot_id|>").removesuffix("</s>")
    student_prompt = tokenizer.apply_chat_template(student_input, tokenize=False).removesuffix("<|eot_id|>").removesuffix("</s>")

    teacher_tokens = tokenizer(teacher_prompt, return_tensors="pt", truncation=True).to(device)
    student_tokens = tokenizer(student_prompt, return_tensors="pt", truncation=True).to(device)

    # 动态扩展 mask token 长度
    delta = teacher_tokens["input_ids"].shape[1] - student_tokens["input_ids"].shape[1]
    if delta > 0:
        new_mask = doc_mask_token * (2 + delta)
        masked_user_msg["content"] = (
            "You should answer the question by referring to the knowledge provided below and integrating your own knowledge.\n"
            f"Passages:\n{new_mask}\n\nQuestion: {question}"
        )
        student_input[0] = masked_user_msg
        student_prompt = tokenizer.apply_chat_template(student_input, tokenize=False).removesuffix("<|eot_id|>").removesuffix("</s>")
        student_tokens = tokenizer(student_prompt, return_tensors="pt", truncation=True).to(device)

    # 获取文档嵌入
    with torch.no_grad():
        embed_inputs = embedding_tokenizer(passage_text, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(device)
        if 't5' in embedding_model_path:
            doc_embed = torch.mean(embedding_model.encoder(**embed_inputs).last_hidden_state, dim=1)
        else:
            doc_embed = embedding_model(**embed_inputs)[0][:, 0]
        doc_embed = torch.nn.functional.normalize(doc_embed, p=2, dim=1)

    # LoRA 注入并生成
    with torch.no_grad():
        lora_weights = translator(doc_embed)
        delta_inject(llm_model, lora_weights)

        output_ids = llm_model.generate(
            input_ids=student_tokens["input_ids"],
            attention_mask=student_tokens["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    answer = tokenizer.decode(output_ids[0][student_tokens["input_ids"].shape[1]:], skip_special_tokens=True)
    return answer.strip()

def inference_on_jsondataset_with_rag(
    input_json_path: str,
    output_json_path: str,
    llm_model_path: str = "./models/Llama-3.2-1B-Instruct",
    max_new_tokens: int = 256,
    batch_size: int = 4,
    device: str = "cuda:0",
):
    # 加载 tokenizer 和模型
    tokenizer = AutoTokenizer.from_pretrained(llm_model_path, padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(llm_model_path, device_map=device)
    model.eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.config.pad_token_id = tokenizer.pad_token_id

    # 构造输出字段名
    suffix = Path(llm_model_path).parts[-1]
    field_name = f"answer_{suffix}_rag"

    # 加载数据
    with open(input_json_path, "r") as f:
        data = json.load(f)

    prompts = []

    # 构建 prompt（真实 passage）
    for item in data:
        q = item["question"]
        p=""
        for idx,passage in enumerate(item["passages"]):
            p+=f"Passage {idx+1}:\n{passage}\n"
        user_msg = {
            "role": "user",
            # "content": (
            #     "You should answer the question by referring to the knowledge provided below and integrating your own knowledge.\n"
            #     f"Passages:\n{p}\n\nQuestion: {q}"
            # )
            "content": (
                "You should answer the question by referring to the knowledge provided below and integrating your own knowledge.\n"
                "You must give an answer and answer in few words without any additional explanation.\n"
                f"Passages:\n{p}\n\nQuestion: {q}"
            )
            # "content":"You should answer the question by referring to the knowledge provided below and integrating your own knowledge. You must answer in a concise manner without any additional explanation.\nPassages:\n{passage}\n\nQuestion: {question}"
        }
        # system_msg = {
        #     "role": "system",
        #     "content": "You are a helpful assistant. Your task is to extract relevant information from provided documents and to answer to questions as briefly as possible. Your answer must not exceed eight words."
        # }
        assistant_msg = {"role": "assistant", "content": "The answer is "}
        prompt = tokenizer.apply_chat_template([user_msg, assistant_msg], tokenize=False).removesuffix("<|eot_id|>").removesuffix("</s>")
        prompts.append(prompt)

    # 推理
    answers_out = []
    pbar = tqdm(total=len(prompts), desc="Running RAG inference")
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False, max_length=2048).to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        decoded = tokenizer.batch_decode(output_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        answers_out.extend([ans.strip() for ans in decoded])
        pbar.update(len(batch_prompts))
    pbar.close()

    # 写入结果
    for item, pred in zip(data, answers_out):
        item[field_name] = pred

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"✅ RAG 推理完成，保存至：{output_json_path}")

def inference_on_jsondataset_with_cocom(
    input_json_path: str,
    output_json_path: str,
    llm_model_path: str = "./models/cocom-v1-128-mistral-7b",
    max_new_tokens: int = 256,
    batch_size: int = 4,
    device: str = "cuda:0",
):

    model = AutoModel.from_pretrained(llm_model_path, device_map=device, trust_remote_code=True)

    # 构造输出字段名
    suffix = Path(llm_model_path).parts[-1]
    field_name = f"answer_{suffix}"

    # 加载数据
    with open(input_json_path, "r") as f:
        data = json.load(f)

    contexts = []
    questions = []
    # 构建 prompt（真实 passage）
    for item in data:
        contexts.append(item["passages"])
        questions.append(item["question"])

    # 推理
    answers_out = []
    pbar = tqdm(total=len(questions), desc="Running RAG inference")
    for i in range(0, len(questions), batch_size):
        documents_batch = contexts[i:i + batch_size]
        questions_batch = questions[i:i + batch_size]
        answers = model.generate_from_text(questions=questions_batch, documents=documents_batch, max_new_tokens=max_new_tokens)
        answers_out.extend(answers)
        pbar.update(len(questions_batch))
    pbar.close()

    # 写入结果
    for item, pred in zip(data, answers_out):
        item[field_name] = pred

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"✅ RAG 推理完成，保存至：{output_json_path}")

def inference_on_jsondataset_rag_summary(
    input_json_path: str,
    output_json_path: str,
    llm_model_path: str = "./models/Llama-3.2-1B-Instruct",
    max_new_tokens: int = 256,
    batch_size: int = 4,
    device: str = "cuda:0",
):
    # 加载 tokenizer 和模型
    tokenizer = AutoTokenizer.from_pretrained(llm_model_path, padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(llm_model_path, device_map=device)
    model.eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.config.pad_token_id = tokenizer.pad_token_id

    # 构造输出字段名
    suffix = Path(llm_model_path).parts[-1]
    field_name = f"answer_{suffix}"

    # 加载数据
    with open(input_json_path, "r") as f:
        data = json.load(f)

    prompts = []
    user_template = """Summarize the following news article in a concise and informative paragraph:

Article:
{article}

Summary:
"""
    # 构建 prompt（真实 passage）
    for item in data:
        ar = item["article"]
        user_msg = {
            "role": "user",
            "content": user_template.format(article=ar)
        }
        prompt = tokenizer.apply_chat_template([user_msg], tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)

    # 推理
    answers_out = []
    pbar = tqdm(total=len(prompts), desc="Running RAG inference")
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        decoded = tokenizer.batch_decode(output_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        answers_out.extend([ans.strip() for ans in decoded])
        pbar.update(len(batch_prompts))
    pbar.close()

    # 写入结果
    for item, pred in zip(data, answers_out):
        item[field_name] = pred

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"✅ RAG 推理完成，保存至：{output_json_path}")

def inference_on_jsondataset_no_rag(
    input_json_path: str,
    output_json_path: str,
    llm_model_path: str = "./models/Llama-3.2-1B-Instruct",
    max_new_tokens: int = 256,
    batch_size: int = 4,
    device: str = "cuda:0",
    adapter_path: str = None,
):
    # 加载 tokenizer 和模型
    tokenizer = AutoTokenizer.from_pretrained(llm_model_path, padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(llm_model_path, device_map=device)
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.config.pad_token_id = tokenizer.pad_token_id

    # 构造输出字段名
    if adapter_path is not None:
        suffix = Path(adapter_path).parts[-1]
    else:
        suffix = Path(llm_model_path).parts[-1]
    field_name = f"answer_{suffix}_norag"

    # 加载数据
    with open(input_json_path, "r") as f:
        data = json.load(f)

    prompts = []

    # 构建 prompt（真实 passage）
    for item in data:
        q = item["question"]
        user_msg = {
            "role": "user",
            "content": (
                "You should answer the question by integrating your own knowledge.\n"
                f"\n\nQuestion: {q}"
            )
        }
        assistant_msg = {"role": "assistant", "content": "The answer is "}
        prompt = tokenizer.apply_chat_template([user_msg, assistant_msg], tokenize=False).removesuffix("<|eot_id|>").removesuffix("</s>")
        prompts.append(prompt)

    # 推理
    answers_out = []
    pbar = tqdm(total=len(prompts), desc="Running no RAG inference")
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        decoded = tokenizer.batch_decode(output_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        answers_out.extend([ans.strip() for ans in decoded])
        pbar.update(len(batch_prompts))
    pbar.close()

    # 写入结果
    for item, pred in zip(data, answers_out):
        item[field_name] = pred

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"✅ RAG 推理完成，保存至：{output_json_path}")

def inference_on_jsondataset_with_masking_summary(
    input_json_path: str,
    output_json_path: str,
    llm_model_path: str = "./models/Llama-3.2-1B-Instruct",
    translator_type: str = "parameter-translator",
    doc_mask_token: str = "",
    max_new_tokens: int = 256,
    batch_size: int = 4,
    device: str = "cuda:0",
    embedding_model_path: str = "./models/long-t5-tglobal-base",
    translator_path: str = "models/Llama-3.2-1B-Instruct-longt5_alignment1/translator_step_30000.safetensors",
):
    
    def _get_doc_embed(passages):
        def _bge_or_snowflake_embed(model, inputs):
            output = model(**inputs)
            embeddings = output[0][:, 0]  # 取 [CLS] token
            return torch.nn.functional.normalize(embeddings, p=2, dim=1)

        def _t5_embed(model, inputs):
            output = model.encoder(**inputs)
            embeddings = output.last_hidden_state
            return torch.mean(embeddings, dim=1)

        with torch.no_grad():
            inputs = embedding_tokenizer(
                passages,
                return_tensors="pt",
                padding="longest",
                max_length=4096,
                truncation=True
            ).to(device)

            model_name = embedding_model.name_or_path.lower()

            if 'bge-large-en-v1.5' in model_name or 'snowflake' in model_name:
                sentence_embeddings = _bge_or_snowflake_embed(embedding_model, inputs)
            elif 't5' in model_name:
                sentence_embeddings = _t5_embed(embedding_model, inputs)
            else:
                raise NotImplementedError(f"[Unsupported Model] {model_name}")

        return sentence_embeddings
    def _get_full_doc_embed(passages):

        def _bge_or_snowflake_embed(model, inputs):
            output = model(**inputs)
            embeddings = output.last_hidden_state #[B, L, D]
            return torch.nn.functional.normalize(embeddings, p=2, dim=-1)

        def _t5_embed(model, inputs):
            output = model.encoder(**inputs)
            embeddings = output.last_hidden_state
            return embeddings #[B, L, D]

        with torch.no_grad():
            inputs = embedding_tokenizer(
                passages,
                return_tensors="pt",
                padding="longest",
                max_length=4096,
                truncation=True
            ).to(device)

            model_name = embedding_model.name_or_path.lower()

            if 'bge-large-en-v1.5' in model_name or 'snowflake' in model_name:
                sentence_embeddings = _bge_or_snowflake_embed(embedding_model, inputs)
            elif 't5' in model_name:
                sentence_embeddings = _t5_embed(embedding_model, inputs)
            else:
                raise NotImplementedError(f"[Unsupported Model] {model_name}")

        return sentence_embeddings, inputs["attention_mask"]


    # 载入模型与tokenizer
    tokenizer = AutoTokenizer.from_pretrained(llm_model_path, padding_side="left")
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eos_token_id
    embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_path)
    llm_model = AutoModelForCausalLM.from_pretrained(llm_model_path, device_map=device)
    llm_model.generation_config.temperature = None
    llm_model.generation_config.top_p = None
    if 'snowflake' in embedding_model_path:
        embedding_model = AutoModel.from_pretrained(embedding_model_path, add_pooling_layer=False, trust_remote_code=True,device_map=device)
    else:
        embedding_model = AutoModel.from_pretrained(embedding_model_path,device_map=device)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=2,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=["down_proj", "up_proj", "gate_proj"],  # 指定 LoRA 注入的模块
    )
    llm_model = get_peft_model(llm_model, peft_config)
    llm_model.config.pad_token_id = tokenizer.pad_token_id
    if translator_type == "parameter-translator":
        translator = ParameterTranslator(
            embedding_model=embedding_model,
            llm_model=llm_model,
            lora_rank=2,
            projector_hidden_dim=512
        )#66M
    elif translator_type == "cross-attention-parameter-translator":
        # translator = CrossAttentionParameterTranslator(
        #     embedding_model=embedding_model,
        #     llm_model=llm_model,
        #     lora_rank=2,
        #     projector_hidden_dim=1024,
        #     attn_heads=8,
        #     attn_ff_dim=1024,
        #     cross_layers=3,
        #     encoder_layers=3,
        # )#122M
        translator = CrossAttentionParameterTranslator(
            embedding_model=embedding_model,
            llm_model=llm_model,
            lora_rank=2,
            projector_hidden_dim=2560,
            attn_heads=8,
            attn_ff_dim=1024,
            cross_layers=4,
            encoder_layers=4,
        )#239M
    elif translator_type == "cross-attention-hyper-parameter-translator":
        translator = CrossAttentionHyperNetworkParameterTranslator(
            embedding_model=embedding_model,
            llm_model=llm_model,
            lora_rank=2,
            projector_hidden_dim=2048,
            attn_heads=8,
            cross_layers=1,
        )
    else:
        raise ValueError(f"Unknown translator type: {translator_type}")
    translator.to(device)
    translator.load_state_dict(load_file(translator_path, device=device))
    # translator.eval()

    # 构造输出字段名
    suffix=Path(translator_path).parts[-2]+"-"+Path(translator_path).parts[-1].removesuffix(".safetensors")
    field_name = f"answer_{suffix}"

    # 读取数据
    with open(input_json_path, "r") as f:
        data = json.load(f)

    # 构造 teacher 和 student inputs（最终推理只用 student）
    teacher_inputs = []
    student_inputs = []

    articles = []
    user_template = """Summarize the following news article in a concise and informative paragraph:

Article:
{article}

Summary:
"""
    
    for item in data:
        ar = item["article"]

        articles.append(ar)

        user_msg = {
            "role": "user",
            "content": user_template.format(article=ar)
        }
        # assistant_msg = {"role": "assistant", "content": assistant_template}

        # teacher_inputs.append([user_msg, assistant_msg])
        teacher_inputs.append([user_msg])

        masked_doc = doc_mask_token * 2
        masked_user_msg = {
            "role": "user",
            "content": user_template.format(article=masked_doc)
        }
        # student_inputs.append([masked_user_msg, assistant_msg])
        student_inputs.append([masked_user_msg])

    # 初次 token 化比较长度
    teacher_texts = [tokenizer.apply_chat_template(x, tokenize=False).removesuffix("<|eot_id|>").removesuffix("</s>") for x in teacher_inputs]
    student_texts = [tokenizer.apply_chat_template(x, tokenize=False).removesuffix("<|eot_id|>").removesuffix("</s>") for x in student_inputs]

    teacher_tokenized = [tokenizer(teacher_text, return_tensors="pt", truncation=True, max_length=4096, add_special_tokens=False) for teacher_text in teacher_texts]
    student_tokenized = [tokenizer(student_text, return_tensors="pt", truncation=True, max_length=4096, add_special_tokens=False) for student_text in student_texts]
    # 动态扩展 mask token 以保证 student 输入不短于 teacher
    for i in range(len(teacher_texts)):
        t_len = teacher_tokenized[i]["input_ids"].shape[-1]
        s_len = student_tokenized[i]["input_ids"].shape[-1]
        delta = t_len - s_len
        if delta > 0:
            new_mask = doc_mask_token * (2 + delta)
            updated_msg = {
                "role": "user",
                "content": user_template.format(article=new_mask)
            }
            student_inputs[i][0] = updated_msg

    # 重新构建 student prompt（用于推理）
    prompts = [tokenizer.apply_chat_template(x, tokenize=False, add_generation_prompt=True) for x in student_inputs]

    # 推理过程
    answers_out = []
    pbar = tqdm(total=len(prompts), desc="Running inference")
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False).to(llm_model.device)
        if translator_type == "parameter-translator":
            doc_embed = _get_doc_embed(articles[i:i + batch_size])
            with torch.no_grad():
                lora_weights = translator(doc_embed)
        elif translator_type == "cross-attention-parameter-translator":
            doc_embed,attention_mask = _get_full_doc_embed(articles[i:i + batch_size])
            with torch.no_grad():
                lora_weights = translator(doc_embed,attention_mask.to(translator.device))
        else:
            raise ValueError(f"Unknown translator type: {translator_type}")
        delta_inject(llm_model, lora_weights)
        with torch.no_grad():
            output_ids = llm_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        decoded = tokenizer.batch_decode(output_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        answers_out.extend([ans.strip() for ans in decoded])
        pbar.update(len(batch_prompts))
    pbar.close()

    # 写入结果
    for item, pred in zip(data, answers_out):
        item[field_name] = pred

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"✅ 推理完成，保存至：{output_json_path}")

def normalize_text(text: str) -> str:
    """Normalize text with lowercasing, removing articles, and punctuation."""

    # 定义一个函数，用于移除文本中的冠词
    def remove_articles(text: str) -> str:
        # 使用正则表达式，将文本中的冠词替换为空格
        return re.sub(r"\b(a|an|the)\b", " ", text)

    # 定义一个函数，用于去除字符串中的多余空格
    def white_space_fix(text: str) -> str:
        # 使用split()方法将字符串按空格分割成一个列表
        # 使用join()方法将列表中的元素用空格连接成一个字符串
        return " ".join(text.split())

    # 定义一个函数，用于移除字符串中的标点符号
    def remove_punc(text: str) -> str:
        # 定义一个集合，包含所有标点符号
        exclude = set(string.punctuation)
        # 返回一个新的字符串，其中不包含标点符号
        return "".join(ch for ch in text if ch not in exclude)

    # 定义一个函数，将输入的字符串转换为小写
    def lower(text: str) -> str:
        # 返回转换后的小写字符串
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def calc_unigram_f1(text: str, answers: list[str], field: str = "f1") -> float:
    norm_pred = normalize_text(text).split()
    norm_answers = [normalize_text(ans).split() for ans in answers]

    score_list = []
    for norm_ans in norm_answers:
        common = Counter(norm_pred) & Counter(norm_ans)
        num_same = sum(common.values())
        if num_same == 0:
            score_list.append(0.0)
            continue
        p = num_same / len(norm_pred)
        r = num_same / len(norm_ans)
        if field == "precision":
            score_list.append(p)
        elif field == "recall":
            score_list.append(r)
        elif field == "f1":
            f1 = 2 * p * r / (p + r) if (p + r) else 0.0
            score_list.append(f1)
        else:
            raise ValueError(f"Unknown field: {field}")
    return max(score_list)


def calc_span_em(text: str, answers: list[str]) -> float:
    norm_pred = normalize_text(text)
    norm_answers = [normalize_text(ans) for ans in answers]
    em = [1.0 if ((norm_ans in norm_pred) or (norm_pred in norm_ans)) else 0.0 for norm_ans in norm_answers]
    return max(em)

def cal_one_metric(generated_answer, given_answer, metric="f1"):
    scores = []
    for text, answers in zip(generated_answer, given_answer):
        if metric in ["f1", "precision", "recall"]:
            scores.append(calc_unigram_f1(text, answers, field=metric))
        elif metric == "span_EM":
            scores.append(calc_span_em(text, answers))
    # print(sum(scores) / len(scores))
    return sum(scores) / len(scores)

def cal_matric(inference_file, answer_field, metrics=["f1","span_EM"]):
    with open(inference_file, "r") as f:
        dataset = json.load(f)
    generated_answer = [data[answer_field] for data in dataset]
    try:
        if isinstance(dataset[0]["answer"], list):
            given_answer = [data["answer"] for data in dataset]
        else:
            given_answer = [[data["answer"]] for data in dataset]
    except:
        given_answer = [[data["highlights"]] for data in dataset]
    metric_scores=[]
    for metric in metrics:
        metric_score = cal_one_metric(generated_answer, given_answer, metric=metric)
        print(f"{Path(inference_file).parts[-2]},{Path(inference_file).parts[-1]},{metric}:", metric_score)
        metric_scores.append(metric_score)
    return metric_scores





if __name__ == '__main__':
    # 设置参数
    '''
    input_json_path: str,
    output_json_path: str,
    llm_model_path: str = "./models/Llama-3.2-1B-Instruct",
    doc_mask_token: str = "",
    max_new_tokens: int = 256,
    batch_size: int = 4,
    device: str = "cuda:0",
    embedding_model_path: str = "./models/long-t5-tglobal-base",
    translator_path: str = "models/Llama-3.2-1B-Instruct-longt5_alignment1/translator_step_119000.safetensors",
    '''
    input_json_paths=[
        "./data_dev_inference_104/2wikimultihopqa/comparison_dev.json",
        "./data_dev_inference_104/2wikimultihopqa/bridge_comparison_dev.json",
        "./data_dev_inference_104/2wikimultihopqa/inference_dev.json",
        "./data_dev_inference_104/2wikimultihopqa/compositional_dev.json",
        "./data_dev_inference_104/hotpotqa/bridge_dev.json",
        "./data_dev_inference_104/hotpotqa/comparison_dev.json",
        "./data_dev_inference_104/popqa/total_dev.json",
        "./data_dev_inference_104/complexwebquestions/total_dev.json",
        # "./data_dev_inference_104/iirc/total_dev.json",
        # "./data_dev_inference_104/strategyqa/total_dev.json",
        # "./data_dev_inference_104/ragtruth/total_dev.json",
        # "./data_dev_inference/cnn_dailymail/test_100.json",
    ]
    # llm_model_path = "./models/pisco-mistral"
    llm_model_path = "./models/Mistral-7B-Instruct-v0.2"
    # llm_model_path = "./models/Meta-Llama-3-8B-Instruct-Doc_mask"
    # llm_model_path = "./models/Llama-3.2-1B-Instruct-Doc_mask"
    embedding_model_path = "./models/long-t5-tglobal-base"
    doc_mask_token = "<|doc_mask|>"
    device = "cuda:0"
    translator_type = "cross-attention-parameter-translator-s"#"parameter-translator","cross-attention-parameter-translator"
    # translator_path = "./models/Llama-3.2-1B-Instruct-Doc_mask-longt5_capt_70/translator_step_72270.safetensors"
    translator_path="./models/Meta-Llama-3-8B-Instruct-Doc_mask-longt5_capt_5/translator_step_72270.safetensors"
    # translator_paths = [

    # ]
    # input_json_path = "./data_dev_inference_104/2wikimultihopqa/comparison_dev.json"
    for input_json_path in input_json_paths:
        if "ragtruth" in input_json_path:
            max_new_tokens = 256
        else:
            max_new_tokens = 128
        # inference_on_jsondataset_with_mydyprag_masking_concise(
        #     input_json_path, 
        #     input_json_path, 
        #     llm_model_path, 
        #     translator_type,
        #     doc_mask_token, 
        #     max_new_tokens=50, 
        #     batch_size=4, 
        #     device=device, 
        #     embedding_model_path=embedding_model_path, 
        #     translator_path=translator_path
        # )
        # inference_on_jsondataset_with_masking_summary(
        #     input_json_path, 
        #     input_json_path, 
        #     llm_model_path, 
        #     translator_type,
        #     doc_mask_token, 
        #     max_new_tokens=max_new_tokens, 
        #     batch_size=2, 
        #     device=device, 
        #     embedding_model_path=embedding_model_path, 
        #     translator_path=translator_path
        # )
        # inference_on_jsondataset_rag_summary(
        #     input_json_path,
        #     input_json_path,
        #     llm_model_path,
        #     max_new_tokens=max_new_tokens,
        #     batch_size=2,
        #     device=device,
        # )
        inference_on_jsondataset_with_rag(
            input_json_path, 
            input_json_path, 
            llm_model_path, 
            max_new_tokens=12, 
            batch_size=1, 
            device=device, 
        )
        # inference_on_jsondataset_with_cocom(
        #     input_json_path, 
        #     input_json_path, 
        #     llm_model_path, 
        #     max_new_tokens=12, 
        #     batch_size=1, 
        #     device=device, 
        # )
        # suffix = f"{Path(translator_path).parts[-2]}-{Path(translator_path).parts[-1]}".removesuffix(".safetensors")
        # answer_field = f"answer_{suffix}_concise"
        answer_field = f"answer_{Path(llm_model_path).parts[-1]}_rag"
        # answer_field = "answer_Llama-3.2-1B-Instruct-Doc_mask-longt5_alignment5-translator_step_29000"
        # print(answer_field)
        cal_matric(
            input_json_path, answer_field, metrics=[
                "f1",
                "span_EM",
                ]
        )

        # inference_on_jsondataset_no_rag(
        #     input_json_path,
        #     input_json_path,
        #     llm_model_path,
        #     max_new_tokens=256,
        #     batch_size=8,
        #     device=device,
        # )
    # answer_field = f"answer_{Path(translator_path).parts[-2]}-{Path(translator_path).parts[-1]}".removesuffix(".safetensors")
    # # answer_field = "answer_Llama-3.2-1B-Instruct-Doc_mask_norag"
    # print(answer_field)
    # for inference_file in input_json_paths:
    #     # answer_field = "answer_Llama-3.2-1B-Instruct-Doc_mask"
    #     # answer_field = "answer_Llama-3.2-1B-Instruct-Doc_mask_norag"
    #     # answer_field = "answer_Llama-3.2-1B-Instruct-Doc_mask-longt5_alignment5-translator_step_29000"
    #     cal_matric(
    #         inference_file, answer_field, metrics=[
    #             "f1",
    #             "span_EM",
    #             ]
    #     )

    # question = "Who is the 47th President of the United States?"
    # passages = [
    #     "Donald Trump took the oath of office as the nation’s 47th president at 12:02 p.m. on Monday, marking a historic comeback for a president who has promised to disrupt Washington even more so than he did during his first term. With four predecessors, several supportive billionaires and scores of elected officials looking on, Trump became president for a second time inside the same Capitol building his supporters stormed four years ago in an effort to halt Congress’ ratification of his defeat. It was the first time in more than a century that a former president has taken the oath for a second time after leaving office, with the 45th and now 47th president following in the footsteps of Grover Cleveland, the only other president to serve nonconsecutive terms.",
    # ]
    # answer = inference_single_sample(
    #     question, 
    #     passages, 
    #     llm_model_path="./models/Llama-3.2-1B-Instruct-Doc_mask", 
    #     doc_mask_token="<|doc_mask|>", 
    #     max_new_tokens=256, 
    #     device="cuda:6",
    #     embedding_model_path="./models/long-t5-tglobal-base",
    #     translator_path="./models/Llama-3.2-1B-Instruct-longt5_alignment1/translator_step_119000.safetensors"
    #     )
    # print(answer)

