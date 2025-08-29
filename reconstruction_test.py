from train_my_dyprag_104 import *
from utils_104 import *
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import T5Tokenizer, T5ForConditionalGeneration
from safetensors.torch import save_file, load_file
from typing import List, Dict
from reconstruction_train import *
from evaluate import load
import sacrebleu
from rouge_score import rouge_scorer
from sklearn.metrics import precision_recall_fscore_support


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

def normalize_text(text: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))




if __name__ == "__main__":
    embedding_model_path = "./models/long-t5-tglobal-base"
    translator_path = "./models/Llama-3.2-1B-Instruct-Doc_mask-longt5_capt_57/translator_step_72270.safetensors"
    device = "cuda:0"
    embedding_model = AutoModel.from_pretrained(embedding_model_path,device_map=device, torch_dtype=torch.bfloat16)
    embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_path)
    llm_model_path = "./models/Llama-3.2-1B-Instruct-Doc_mask"
    llm_model = AutoModelForCausalLM.from_pretrained(llm_model_path, device_map="cpu")
    translator = CrossAttentionParameterTranslator(
                embedding_model=embedding_model,
                llm_model=llm_model,
                lora_rank=2,
                projector_hidden_dim=1024,
                attn_heads=8,
                attn_ff_dim=1024,
                cross_layers=1,
                encoder_layers=1,
            )
    translator.to(device)
    translator.load_state_dict(load_file(translator_path, device=device))
    translator = translator.to(torch.bfloat16)
    translator.eval()

    checkpoint=22425
    reconstructor_path = f"./models/reconstruct_model/{checkpoint}"
    reconstructor = LoRAReconstructor.load_from_checkpoint(reconstructor_path).to(torch.bfloat16)
    reconstructor.eval().cuda()

    test_dataset_path = "./data_dev_inference_104/reconstruct_passage_test.json"
    test_dataset = json.load(open(test_dataset_path, "r"))
    passages = []
    for data in test_dataset:
        passages.append(data["passage"])
    
    batch_size = 4
    preds, refs = [], []

    for i in tqdm(range(0, len(passages), batch_size)):
        if f'rc_passage_{checkpoint}' in test_dataset[i]:
            break
        batch_passages = passages[i:i+batch_size]
        with torch.no_grad():
            doc_embed, attn_mask = _get_full_doc_embed(batch_passages)
            lora_dict = translator(doc_embed, attn_mask.to(device))
            lora_tensor = extract_lora_features(lora_dict).cuda()
            outputs = reconstructor.infer(lora_tensor)
            preds.extend(outputs)
            refs.extend(batch_passages)

    for d, p in zip(test_dataset, preds):
        d[f"rc_passage_{checkpoint}"] = p

    json.dump(test_dataset, open(test_dataset_path, "w"), indent=2, ensure_ascii=False)

    # Normalize for evaluation
    norm_preds = [normalize_text(p) for p in preds]
    norm_refs = [normalize_text(r) for r in refs]

    # rouge = load("rouge", download_mode="force_local")
    # bleu = load("bleu")
    # rouge_scores = rouge.compute(predictions=norm_preds, references=norm_refs, rouge_types=["rouge1"])
    # bleu_score = bleu.compute(predictions=norm_preds, references=[[r] for r in norm_refs])

    # print(f"\nCheckpoint {checkpoint} Evaluation Results:")
    # print(f"ROUGE-1 F1: {rouge_scores['rouge1'].mid.fmeasure:.4f}")
    # print(f"BLEU: {bleu_score['bleu']:.4f}")

    # 1. 计算 BLEU
    bleu = sacrebleu.corpus_bleu(norm_preds, [norm_refs])
    print(f"BLEU: {bleu.score:.4f}")

    # 2. 计算 ROUGE-1 F1
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    rouge1_f1_scores = [scorer.score(r, p)['rouge1'].fmeasure for r, p in zip(norm_refs, norm_preds)]
    avg_rouge1_f1 = sum(rouge1_f1_scores) / len(rouge1_f1_scores)
    print(f"ROUGE-1 F1: {avg_rouge1_f1:.4f}")

    # 3. 自定义 F1：逐 token 精确度/召回/F1
    all_preds_tokens = [p.split() for p in norm_preds]
    all_refs_tokens = [r.split() for r in norm_refs]

    flatten_preds = [tok for pred in all_preds_tokens for tok in pred]
    flatten_refs = [tok for ref in all_refs_tokens for tok in ref]

    precision, recall, f1, _ = precision_recall_fscore_support(
        flatten_refs, flatten_preds, average='micro', zero_division=0
    )
    print(f"Token-level F1: {f1:.4f} (Precision: {precision:.4f}, Recall: {recall:.4f})")




