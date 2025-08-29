from train_my_dyprag_104 import *
from utils_104 import *
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration
from safetensors.torch import save_file, load_file
from typing import List, Dict




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

def extract_lora_features(lora_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # Convert dict to tensor shape [B, 16, 61440]
    # batch_size = list(lora_dict.values())[0].shape[0]
    layers = []
    for i in range(16):
        prefix = f"base_model.model.model.layers.{i}.mlp"
        keys = [
            f"{prefix}.gate_proj.lora_A.weight",
            f"{prefix}.gate_proj.lora_B.weight",
            f"{prefix}.up_proj.lora_A.weight",
            f"{prefix}.up_proj.lora_B.weight",
            f"{prefix}.down_proj.lora_A.weight",
            f"{prefix}.down_proj.lora_B.weight",
        ]
        matrices = [lora_dict[k] for k in keys]
        flat = torch.cat([mat.flatten(1) for mat in matrices], dim=-1)  # [B, 61440]
        layers.append(flat)
    return torch.stack(layers, dim=1)  # [B, 16, 61440]



class LoRAReconstructor(nn.Module):
    def __init__(self, encoder_dim=768):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(61440, encoder_dim),
            nn.ReLU(),
            nn.LayerNorm(encoder_dim)
        )
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=encoder_dim, nhead=8, dim_feedforward=2048),
            num_layers=1
        )
        # self.decoder = T5ForConditionalGeneration.from_pretrained("./models/long-t5-tglobal-base")
        config = T5Config.from_pretrained("./models/long-t5-tglobal-base")
        self.decoder = T5ForConditionalGeneration(config)
        self.tokenizer = T5Tokenizer.from_pretrained("./models/long-t5-tglobal-base")

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, lora_input, attention_mask, labels):
        fused = self.proj(lora_input)  # [B, 16, 1024]
        memory = self.encoder(fused.permute(1, 0, 2)).permute(1, 0, 2)  # [B, 16, 1024]
        outputs = self.decoder(
            inputs_embeds=memory,                 # [B, seq_len, 768]
            decoder_attention_mask=attention_mask,
            labels=labels
        )
        return outputs

    def infer(self, lora_tensor: torch.Tensor) -> List[str]:
        self.eval()
        with torch.no_grad():
            if lora_tensor.dim() == 2:
                lora_tensor = lora_tensor.unsqueeze(0)  # [1, 16, 61440]

            fused = self.proj(lora_tensor)  # [B, 16, 1024]
            memory = self.encoder(fused.permute(1, 0, 2)).permute(1, 0, 2)  # [B, 16, 1024]
    
            start_token_id = self.tokenizer.pad_token_id  # T5 的起始 token 通常是 <pad>
            input_ids = torch.full((memory.shape[0], 1), start_token_id, dtype=torch.long).to(memory.device)

            generated = self.decoder.generate(
                inputs_embeds=memory,
                decoder_input_ids=input_ids,
                max_new_tokens=1024
            )
            return self.tokenizer.batch_decode(generated, skip_special_tokens=True)

    def get_encoder_state_dict(self):
        return {k: v for k, v in self.state_dict().items() if not k.startswith("decoder") and not k.startswith("tokenizer")}

    def save(self, path):
        self.decoder.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        save_file(self.get_encoder_state_dict(), f"{path}/proj_encoder.safetensors")
 
    @classmethod
    def load_from_checkpoint(cls, path):
        model = cls()
        model.decoder = T5ForConditionalGeneration.from_pretrained(path)
        model.tokenizer = T5Tokenizer.from_pretrained(path)
        encoder_weights = load_file(f"{path}/proj_encoder.safetensors")
        model.load_state_dict(encoder_weights, strict=False)
        return model

def train(model: LoRAReconstructor, passages: List[str], translator, embedder, batch_size=4, epochs=3, lr=5e-5, save_path="./models/reconstruct_init_model/{checkpoint}"):
    import wandb
    name=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    wandb.init(project="end2end-DyPRAG-reconstruction",name=name, notes=save_path)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    tokenizer = model.tokenizer
    model.train()
    step=0
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        pbar = tqdm(range(0, len(passages), batch_size))
        for i in range(0, len(passages), batch_size):
            batch_passages = passages[i:i+batch_size]
            encodings = tokenizer(batch_passages, return_tensors='pt', padding=True, truncation=True, add_special_tokens=True, max_length=2048)
            input_ids = encodings.input_ids.cuda()
            attention_mask = encodings.attention_mask.cuda()

            with torch.no_grad():
                doc_embed, attn_mask = embedder(batch_passages)
                lora_dict = translator(doc_embed, attn_mask.to(translator.device))
                lora_input = extract_lora_features(lora_dict).to(model.device) #(4,16,61440)

            out = model(lora_input, attention_mask, input_ids)
            loss = out.loss
            wandb.log({"loss":loss.item()})
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.update(1)
            pbar.set_description(f"Step {i//batch_size+1}, Loss: {loss.item():.4f}")
            step+=1
            # print(f"Epoch {epoch+1}, Step {i//batch_size+1}, Loss: {loss.item():.4f}")
            if step % 3700 == 0:
                model.save(save_path.format(checkpoint=step))
    model.save(save_path.format(checkpoint=step))



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



    passages=[]
    train_dataset_path="./data_aug_deepseek-v3/2wikimultihopqa/reconstruct_passage_train.json"
    dataset = json.load(open(train_dataset_path, "r"))
    passages = [item["passage"] for item in dataset]
    rc_model = LoRAReconstructor()
    rc_model.to(device).to(torch.bfloat16)
    save_path = "././models/reconstruct_init_model/{checkpoint}"
    train(rc_model,passages,translator,_get_full_doc_embed,save_path=save_path)

