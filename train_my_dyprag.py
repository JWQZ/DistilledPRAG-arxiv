from imports_104 import *
from utils_104 import *


class PositionalEncoding(nn.Module):
    """
    正弦位置编码，每一层一个独立向量，形状为 [1, num_layers, dim]。
    """
    def __init__(self, num_layers, d_model):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.register_buffer("pe", self._build_encoding())

    def _build_encoding(self):
        position = torch.arange(self.num_layers).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * -(math.log(10000.0) / self.d_model))
        pe = torch.zeros(self.num_layers, self.d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1, num_layers, d_model]

    def forward(self, device):
        return self.pe.to(device)  # [1, num_layers, d_model]

class TransformerProjector(nn.Module):
    """
    使用 Transformer 架构对所有层的 LoRA 参数生成进行建模。
    每一层作为一个 token，输入为文档嵌入加 token embedding 加位置编码。
    """
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, num_heads=4):
        super().__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        # self.token_embeddings = nn.Parameter(torch.randn(num_layers, input_dim))  # 可学习 token 表示
        self.pos_encoding = PositionalEncoding(num_layers, input_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim,bias=False,dropout=0.0)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.output_layer = nn.Sequential(
            nn.Linear(input_dim, output_dim,bias=False),
            nn.LayerNorm(output_dim, eps=1e-6)
            )
        # nn.init.constant_(self.output_layer[1].weight, 1e-4)
        
        # self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight)


    def forward(self, doc_embed):
        # doc_embed: [1, input_dim]
        batch_size = doc_embed.size(0)
        # tokens = self.token_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  # [B, L, D]
        doc_expand = doc_embed.unsqueeze(1).expand(-1, self.num_layers, -1)     # [B, L, D]
        pos_embed = self.pos_encoding(doc_embed.device).expand(batch_size, -1, -1)  # [B, L, D]

        # x = tokens + doc_expand + pos_embed  # [B, L, D]
        x = doc_expand + pos_embed  # [B, L, D]
        # print("[DEBUG] input to transformer:", x.abs().max())

        out = self.transformer(x.transpose(0, 1)).transpose(0, 1)  # [B, L, D]
        # print("[DEBUG] output from transformer:", out.abs().max())

        return self.output_layer(out)  # [B, L, output_dim]


class ParameterTranslator(nn.Module):
    """
    参数生成模块：输入文档 → 动态生成各层 MLP 模块的 LoRA 参数（使用 Transformer）
    支持位置编码按位加法，模型初始化自动读取维度参数。
    """
    def __init__(self, embedding_model: PreTrainedModel, llm_model: PreTrainedModel, 
                 lora_rank=2, projector_hidden_dim=512):
        super().__init__()
        
        self.input_dim = embedding_model.config.hidden_size if 'hidden_size' in embedding_model.config else embedding_model.config.d_model
        self.doc_norm = nn.LayerNorm(self.input_dim)
        self.num_layers = llm_model.config.num_hidden_layers
        self.hidden_size = llm_model.config.hidden_size
        self.ff_hidden_dim = llm_model.config.intermediate_size
        self.lora_rank = lora_rank
        self._scale_param = nn.Parameter(torch.tensor(-8.0))

        self.module_names = ["down_proj", "up_proj", "gate_proj"]

        self.projectors = nn.ModuleList([
            TransformerProjector(
                num_layers=self.num_layers,
                input_dim=self.input_dim,
                hidden_dim=projector_hidden_dim,
                output_dim=self._get_lora_dim(m),
                num_heads=4
            )
            for m in self.module_names
        ])

    def _get_lora_dim(self, module_name):

        return self.lora_rank * (self.hidden_size + self.ff_hidden_dim)

    @property
    def device(self):
        return next(self.parameters()).device

    def get_lorab_scale(self):
        return 0.0001 + 0.9999 * torch.sigmoid(self._scale_param)

    def forward(self, doc_embed):
        """
        输入：doc_embed [1, input_dim]
        输出：字典形式 LoRA 参数（每层 × 每模块）
        """
        outputs = defaultdict(list)
        # print("doc_embed",doc_embed)
        # print("[DEBUG] doc_embed max before norm:", doc_embed.abs().max().item())
        doc_embed = self.doc_norm(doc_embed)  # [1, input_dim]
        # print("doc_embed",doc_embed)
        # print("[DEBUG] doc_embed max:", doc_embed.abs().max().item())

        for i, module_name in enumerate(self.module_names):

            lora_matrix = self.projectors[i](doc_embed)  # [batch_size, num_layers, output_dim]
            # lora_matrix = lora_matrix.squeeze(0)  # [num_layers, output_dim]
            # if self.training:
            #     assert lora_matrix.requires_grad, "LORA output has no grad!"

            # print("[DEBUG] lora_matrix max:", lora_matrix.abs().max().item())
            

            batch_size = lora_matrix.size(0)
            for layer_idx in range(self.num_layers):
                lora_out = lora_matrix[:, layer_idx, :]  # [batch_size, output_dim]

                if module_name == "down_proj":
                    A = lora_out[:, :self.lora_rank * self.ff_hidden_dim].view(batch_size, self.lora_rank, self.ff_hidden_dim)
                    B = lora_out[:, self.lora_rank * self.ff_hidden_dim:].view(batch_size, self.hidden_size, self.lora_rank)*self.get_lorab_scale()
                else:
                    A = lora_out[:, :self.lora_rank * self.hidden_size].view(batch_size, self.lora_rank, self.hidden_size)
                    B = lora_out[:, self.lora_rank * self.hidden_size:].view(batch_size, self.ff_hidden_dim, self.lora_rank)*self.get_lorab_scale()

                A_key = f"base_model.model.model.layers.{layer_idx}.mlp.{module_name}.lora_A.weight"
                B_key = f"base_model.model.model.layers.{layer_idx}.mlp.{module_name}.lora_B.weight"

                outputs[A_key].append(A)
                outputs[B_key].append(B)

        # 最后，stack layer维
        for key in outputs.keys():
            outputs[key] = torch.cat(outputs[key], dim=1)  # [batch_size, total_lora_dim, hidden_size or vice versa]

        return outputs

class CrossAttentionProjector(nn.Module):
    def __init__(self, num_queries, input_dim, hidden_dim, output_dim, num_heads=8, ff_dim=1024, cross_layers=3, encoder_layers=3):
        super().__init__()
        self.query_embed = nn.Parameter(torch.randn(num_queries, input_dim))  # [num_queries, D]

        self.cross_attns = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=input_dim, 
                num_heads=num_heads, 
                batch_first=True, 
                )
            for _ in range(cross_layers)
        ])

        # 加一层 TransformerEncoderLayer（作用在 query 维度）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=encoder_layers,
            enable_nested_tensor=False,
        )

        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=False),
            # nn.LayerNorm(output_dim)#!!!!!!

        )

    def forward(self, token_embed, attention_mask=None):
        """
        token_embed: [B, L, input_dim]
        attention_mask: [B, L] → 1: keep, 0: pad
        """
        B = token_embed.size(0)
        x = self.query_embed.unsqueeze(0).expand(B, -1, -1)  # [B, num_queries, D]

        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None  # [B, L]

        for attn in self.cross_attns:
            x, _ = attn(x, token_embed, token_embed, key_padding_mask=key_padding_mask)

        # Transformer encoder 层作用在每个 query 上（允许 query 之间交互）
        refined = self.encoder(x)  # [B, num_queries, D]

        return self.ffn(refined)  # [B, num_queries, output_dim]


class CrossAttentionParameterTranslator(nn.Module):
    """
    参数生成模块：输入 token 级嵌入和 attention_mask → 生成每层每模块 LoRA 参数
    使用 CrossAttention + TransformerEncoder（增强表达能力）
    """
    def __init__(self, embedding_model: nn.Module, llm_model: nn.Module, 
                 lora_rank=2, projector_hidden_dim=512,
                 attn_heads=4, attn_ff_dim=1024, cross_layers=3, encoder_layers=3):
        super().__init__()

        self.input_dim = embedding_model.config.hidden_size if 'hidden_size' in embedding_model.config else embedding_model.config.d_model
        self.doc_norm = nn.LayerNorm(self.input_dim)

        self.num_layers = llm_model.config.num_hidden_layers
        self.hidden_size = llm_model.config.hidden_size
        self.ff_hidden_dim = llm_model.config.intermediate_size
        self.lora_rank = lora_rank
        self._scale_param = nn.Parameter(torch.tensor(-8.0))

        self.module_names = ["down_proj", "up_proj", "gate_proj"]

        self.projectors = nn.ModuleList([
            CrossAttentionProjector(
                num_queries=self.num_layers,
                input_dim=self.input_dim,
                hidden_dim=projector_hidden_dim,
                output_dim=self._get_lora_dim(m),
                num_heads=attn_heads,
                ff_dim=attn_ff_dim,
                cross_layers=cross_layers,
                encoder_layers=encoder_layers,
            )
            for m in self.module_names
        ])

    def _get_lora_dim(self, module_name):
        return self.lora_rank * (self.hidden_size + self.ff_hidden_dim)

    @property
    def device(self):
        return next(self.parameters()).device

    def get_lorab_scale(self):
        return 0.0001 + 0.9999 * torch.sigmoid(self._scale_param)

    def forward(self, doc_embed, attention_mask):
        """
        输入：
            doc_embed: [B, L, input_dim]
            attention_mask: [B, L]，1 = keep, 0 = pad
        输出：
            dict[str → Tensor]，每层每模块的 LoRA 参数
        """
        outputs = defaultdict(list)
        doc_embed = self.doc_norm(doc_embed)  # [B, L, input_dim]

        for i, module_name in enumerate(self.module_names):
            lora_matrix = self.projectors[i](doc_embed, attention_mask=attention_mask)  # [B, num_layers, output_dim]
            batch_size = lora_matrix.size(0)

            for layer_idx in range(self.num_layers):
                lora_out = lora_matrix[:, layer_idx, :]  # [B, output_dim]

                if module_name == "down_proj":
                    A = lora_out[:, :self.lora_rank * self.ff_hidden_dim].view(batch_size, self.lora_rank, self.ff_hidden_dim)
                    B = lora_out[:, self.lora_rank * self.ff_hidden_dim:].view(batch_size, self.hidden_size, self.lora_rank) * self.get_lorab_scale()
                else:
                    A = lora_out[:, :self.lora_rank * self.hidden_size].view(batch_size, self.lora_rank, self.hidden_size)
                    B = lora_out[:, self.lora_rank * self.hidden_size:].view(batch_size, self.ff_hidden_dim, self.lora_rank) * self.get_lorab_scale()

                A_key = f"base_model.model.model.layers.{layer_idx}.mlp.{module_name}.lora_A.weight"
                B_key = f"base_model.model.model.layers.{layer_idx}.mlp.{module_name}.lora_B.weight"

                outputs[A_key].append(A)
                outputs[B_key].append(B)

        for key in outputs:
            outputs[key] = torch.cat(outputs[key], dim=1)

        return outputs

class CrossAttentionHyperNetworkProjector(nn.Module):
    def __init__(self, num_queries, input_dim, hidden_dim, output_dim,
                 num_heads=8, cross_layers=3):
        super().__init__()
        self.query_embed = nn.Parameter(torch.randn(num_queries, input_dim))  # [num_queries, D]

        self.cross_attns = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=input_dim,
                num_heads=num_heads,
                batch_first=True,
            )
            for _ in range(cross_layers)
        ])

        # nput_dim → hidden_dim → output_dim
        self.hyper_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),  # [B, Q, D] → [B, Q, H]
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim, bias=False),  # [B, Q, H] → [B, Q, output_dim]
        )

    def forward(self, token_embed, attention_mask=None):
        """
        token_embed: [B, L, input_dim]
        attention_mask: [B, L], 1: keep, 0: pad
        Returns: [B, num_queries, output_dim]
        """
        B, L, D = token_embed.size()
        # Q = self.query_embed.size(0)  # num_queries

        # Expand queries across batch
        x = self.query_embed.unsqueeze(0).expand(B, -1, -1)  # [B, Q, D]

        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None  # [B, L]

        # Cross-Attention block: query attends to token embeddings
        for attn in self.cross_attns:
            x, _ = attn(x, token_embed, token_embed, key_padding_mask=key_padding_mask)  # [B, Q, D]

        out = self.hyper_mlp(x)  # [B, Q, output_dim]
        return out

class CrossAttentionHyperNetworkParameterTranslator(nn.Module):
    def __init__(self, embedding_model: nn.Module, llm_model: nn.Module,
                 lora_rank=2, projector_hidden_dim=512,
                 attn_heads=4, cross_layers=3):
        super().__init__()

        self.input_dim = getattr(embedding_model.config, 'hidden_size', embedding_model.config.d_model)
        self.doc_norm = nn.LayerNorm(self.input_dim)

        self.num_layers = llm_model.config.num_hidden_layers
        self.hidden_size = llm_model.config.hidden_size
        self.ff_hidden_dim = llm_model.config.intermediate_size
        self.lora_rank = lora_rank
        self._scale_param = nn.Parameter(torch.tensor(-8.0))

        self.module_names = ["down_proj", "up_proj", "gate_proj"]
        self.num_modules = len(self.module_names)
        self.total_queries = self.num_layers * self.num_modules

        self.projector = CrossAttentionHyperNetworkProjector(
            num_queries=self.total_queries,
            input_dim=self.input_dim,
            hidden_dim=projector_hidden_dim,
            output_dim=self._get_lora_dim_per_module(),
            num_heads=attn_heads,
            cross_layers=cross_layers,
        )

    def _get_lora_dim_per_module(self):
        return self.lora_rank * (self.hidden_size + self.ff_hidden_dim)

    @property
    def device(self):
        return next(self.parameters()).device

    def get_lorab_scale(self):
        return 0.0001 + 0.9999 * torch.sigmoid(self._scale_param)

    def forward(self, doc_embed, attention_mask):
        """
        doc_embed: [B, L, input_dim]
        attention_mask: [B, L] (1 = keep, 0 = pad)

        Return: dict[str → Tensor], LoRA weights per layer & module
        """
        outputs = defaultdict(list)

        B, L, D = doc_embed.size()

        doc_embed = self.doc_norm(doc_embed)  # [B, L, input_dim]

        # [B, total_queries, output_dim]
        lora_matrix = self.projector(doc_embed, attention_mask=attention_mask)

        # Reshape to: [B, num_layers, num_modules, output_dim]
        lora_matrix = lora_matrix.view(B, self.num_layers, self.num_modules, -1)

        for layer_idx in range(self.num_layers):
            for module_idx, module_name in enumerate(self.module_names):
                lora_out = lora_matrix[:, layer_idx, module_idx, :]  # [B, output_dim]

                if module_name == "down_proj":
                    A = lora_out[:, :self.lora_rank * self.ff_hidden_dim].view(B, self.lora_rank, self.ff_hidden_dim)
                    B_mat = lora_out[:, self.lora_rank * self.ff_hidden_dim:].view(B, self.hidden_size, self.lora_rank)
                else:
                    A = lora_out[:, :self.lora_rank * self.hidden_size].view(B, self.lora_rank, self.hidden_size)
                    B_mat = lora_out[:, self.lora_rank * self.hidden_size:].view(B, self.ff_hidden_dim, self.lora_rank)

                # Apply learned scale to B
                B_mat = B_mat * self.get_lorab_scale()

                A_key = f"base_model.model.model.layers.{layer_idx}.mlp.{module_name}.lora_A.weight"
                B_key = f"base_model.model.model.layers.{layer_idx}.mlp.{module_name}.lora_B.weight"

                outputs[A_key].append(A)       # shape: [B, rank, D]
                outputs[B_key].append(B_mat)   # shape: [B, D, rank]

        # Concatenate over layer dim: [B, num_layers, ...]
        for key in outputs:
            outputs[key] = torch.cat(outputs[key], dim=1)

        return outputs
class DyPRAGTrainer:
    def __init__(self, 
                 translator,            # ParameterTranslator
                 embedding_model,       # HuggingFace encoder (e.g., T5, BERT)
                 llm_model,             # HuggingFace CausalLM + PEFT
                 llm_tokenizer,             # Tokenizer for LLM input
                 embedding_tokenizer,    # Tokenizer for embedding model input
                 lr=1e-5,
                 scheduler_type=None,     # lr_scheduler type
                 log_tool=None,         # 
                 log_description=None,
                 devices=None,
                #  {              # 控制每个模型的设备
                #      'translator': 'cuda:0',
                #      'embedding_model': 'cuda:0',
                #      'llm_model': 'cuda:0'
                #  },
                 log_steps=10,          # 每多少步打印一次 loss
                 saving_steps=100,      # 每多少步保存一次翻译器参数
                 save_path="./models/model_a",  # 保存路径
                 args=None
                 ):
        if devices is not None:
            self.translator = translator.to(devices['translator'])
            self.embedding_model = embedding_model.to(devices['embedding_model'])
            self.llm_model = llm_model.to(devices['llm_model'])
            self.devices = devices
        else:
            self.translator = translator
            self.embedding_model = embedding_model
            self.llm_model = llm_model
            self.devices = {"translator": translator.device,
                            "embedding_model": embedding_model.device,
                            "llm_model": llm_model.device}
        self.embedding_tokenizer = embedding_tokenizer
        self.embedding_tokenizer.padding_side = "right"
        self.llm_tokenizer = llm_tokenizer
        self.lr = lr
        self.scheduler_type = scheduler_type
        self.optimizer = None
        self.scheduler = None
        self.args = args
        # self.optimizer = optim.AdamW(self.translator.parameters(), lr=lr)
        # if scheduler_type == "cosine":
        #     self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        # elif scheduler_type == "linear":
        #     self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=0.01, end_factor=0.001, total_iters=100)
        # else:
        #     self.scheduler = None
        self.log_tool = log_tool
        self.log_steps = log_steps
        self.saving_steps = saving_steps
        self.save_path = save_path

        self.global_step = 0
        self.sample_step = 0

        if self.log_tool == "wandb":
            import wandb
            name=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+log_description
            wandb.init(project="end2end-DyPRAG",name=name, config=vars(args), notes=log_description)
        elif self.log_tool == "swanlab":
            import swanlab
            name=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+log_description
            swanlab.init(project="end2end-DyPRAG",name=name, config=vars(args), description=log_description)
        elif self.log_tool == "tensorboard":
            name=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+log_description
            self.tb_writer = SummaryWriter(log_dir=f"runs/{name}")
            for key, value in vars(args).items():
                if isinstance(value, (int, float, str)):
                    self.tb_writer.add_text(f"args/{key}", str(value), 0)
        # register_forward_hooks(self.translator)
        # register_forward_hooks(self.llm_model)
        # register_backward_hooks(self.translator)
    
    def init_optim_lrscheduler(self, 
                               lr=1e-5, 
                               optim_type="adamw", 
                               scheduler_type=None,
                               optim_config={},
                               lrscheduler_config={"T_max": 4, "eta_min": 1e-7},
                               ):
        new_token_id = self.llm_tokenizer.convert_tokens_to_ids("<|doc_mask|>")
        if optim_type == "adamw":
            if self.args.train_token:
                with torch.no_grad():
                    init_embed = self.llm_model.get_input_embeddings().weight[new_token_id].clone()
                self.docmask_embed = nn.Parameter(init_embed)
                self.optimizer = optim.AdamW(list(self.llm_model.parameters())+[self.docmask_embed], lr=lr, **optim_config)
            else:
                self.optimizer = optim.AdamW(self.translator.parameters(), lr=lr, **optim_config)
        elif optim_type == "adam":
            if self.args.train_token:
                with torch.no_grad():
                    init_embed = self.llm_model.get_input_embeddings().weight[new_token_id].clone()
                self.docmask_embed = nn.Parameter(init_embed)
                self.optimizer = optim.Adam(list(self.llm_model.parameters())+[self.docmask_embed], lr=lr, **optim_config)
            else:
                self.optimizer = optim.Adam(self.translator.parameters(), lr=lr, **optim_config)
        else:
            raise ValueError(f"Unsupported optimizer type: {optim_type}, please add it in the code or use 'adamw' or 'adam")
        if scheduler_type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **lrscheduler_config)
        elif scheduler_type == "linear":
            # self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer, **lrscheduler_config)
            self.scheduler = get_polynomial_decay_schedule_with_warmup(self.optimizer, **lrscheduler_config)
        else:
            self.scheduler = None

    def _clip_gradients(self, model, max_norm=1.0, max_value=10.0, verbose=True, print_first_only=True):
        """
        内部梯度裁剪函数，防止梯度爆炸：
        - 限制总 L2 范数（clip_grad_norm_）
        - 限制单个参数最大值（clip_grad_value_）
        - 检查 NaN/Inf，异常警告
        """
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # if verbose:
        #     print(f"[GradClip] Total grad norm after clip: {total_norm:.2e}")

        for name, param in model.named_parameters():
            if param.grad is None:
                continue

            grad = param.grad

            if torch.isnan(grad).any() or torch.isinf(grad).any():
                print(f"[GradClip][ERROR] {name} contains NaN or Inf in gradient!")

            max_grad_val = grad.abs().max().item()
            # print(f"[GradClip] {name} grad max: {max_grad_val:.2e}")
            if max_grad_val > max_value:
                grad.clamp_(min=-max_value, max=max_value)
                if verbose:
                    print(f"[GradClip][CLAMPED] {name} grad max={max_grad_val:.2e} clipped to ±{max_value:.1f}")
                    if print_first_only:
                        break


    def _get_doc_embed(self, passages):
        with torch.no_grad():
            # print(passage)
            inputs = self.embedding_tokenizer(passages, return_tensors="pt",padding="longest", max_length=4096, truncation=True).to(self.devices['embedding_model'])
            
            if 'bge-large-en-v1.5' in embedding_model.name_or_path or "snowflake" in embedding_model.name_or_path:
                output = self.embedding_model(**inputs)
                sentence_embeddings = output[0][:,0]
                sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            elif 't5' in embedding_model.name_or_path:
                output = embedding_model.encoder(**inputs)
                sentence_embeddings = output.last_hidden_state
                sentence_embeddings = torch.mean(sentence_embeddings, dim=1)
            else:
                raise NotImplementedError("Embedding model not implemented")
            
        return sentence_embeddings
    def _get_full_doc_embed(self, passages):
        """
        获取每个文档的完整 token 级嵌入及 attention_mask。
        输出：
            token_embeddings: [B, L, hidden]
            attention_mask: [B, L]，1 表示有效 token，0 表示 padding
        """
        with torch.no_grad():
            inputs = self.embedding_tokenizer(
                passages,
                return_tensors="pt",
                padding="longest",
                max_length=4096,
                truncation=True
            ).to(self.devices['embedding_model'])

            attention_mask = inputs["attention_mask"]  # [B, L]
            model_name = self.embedding_model.name_or_path.lower()

            if 'bge-large-en-v1.5' in model_name or "snowflake" in model_name:
                output = self.embedding_model(**inputs)
                sentence_embeddings = output.last_hidden_state  # [B, L, hidden]
                sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=-1)
            elif 't5' in model_name:
                encoder_outputs = self.embedding_model.encoder(**inputs)
                sentence_embeddings = encoder_outputs.last_hidden_state  # [B, L, hidden]
            else:
                raise NotImplementedError(f"Embedding model {model_name} not supported.")

        return sentence_embeddings, attention_mask

    def _get_logits_label_mask(self, input_ids, pad_token_id=None):
        """
        定位 assistant 回复开始后的 token，用于对 logits 部分计算 loss。
        支持 LLaMA / Qwen chat 模型，其它模型返回 None 并警告。
        """
        chat_template = getattr(self.llm_tokenizer, "chat_template", "")
        if "<|start_header_id|>assistant<|end_header_id|>" not in chat_template:
            print("⚠️ Warning: The tokenizer does not use '<|start_header_id|>assistant<|end_header_id|>' in its chat_template. Logits label cannot be inferred.")
            return None

        start_tokens = self.llm_tokenizer("<|start_header_id|>assistant<|end_header_id|>", add_special_tokens=False)["input_ids"]
        labels_mask = torch.zeros_like(input_ids)

        for b in range(input_ids.size(0)):
            ids = input_ids[b].tolist()
            found = False
            for i in range(len(ids) - len(start_tokens), -1, -1):  # 从后往前找
                if ids[i:i+len(start_tokens)] == start_tokens:
                    labels_mask[b, i + len(start_tokens):] = 1
                    found = True
                    break
            if not found:
                warnings.warn(f"[WARN] Start token not found in sample {b}. Skipping entire batch logits loss.")
                return None  # 有一个找不到就全体作废
        if pad_token_id is not None:
            labels_mask[input_ids == pad_token_id] = 0

        return labels_mask


    def _construct_lm_inputs(self, batch_question, batch_answer, batch_passage, doc_mask_token):
        teacher_inputs = []
        student_inputs = []
        user_template = "You should answer the question by referring to the knowledge provided below and integrating your own knowledge.\nPassages:\n{passage}\n\nQuestion: {question}"
        assistant_template = "The answer is {answer}"
        for q, a, p in zip(batch_question, batch_answer, batch_passage):
            user_msg = {"role": "user", "content": user_template.format(passage=p, question=q)}
            assistant_msg = {"role": "assistant", "content": assistant_template.format(answer=a)}
            teacher_inputs.append([user_msg, assistant_msg])

            # doc mask 替换
            # tokenized = self.llm_tokenizer(p, return_tensors="pt",add_special_tokens=False)
            # token_len = tokenized["input_ids"].shape[1]
            masked_doc = doc_mask_token * 2

            masked_msg = {"role": "user", "content": user_template.format(passage=masked_doc, question=q)}
            student_inputs.append([masked_msg, assistant_msg])

        teacher_texts = [self.llm_tokenizer.apply_chat_template(x, tokenize=False) for x in teacher_inputs]
        student_texts = [self.llm_tokenizer.apply_chat_template(x, tokenize=False) for x in student_inputs]

        # 处理长度对齐（再判断差异）
        teacher_tokenized = [self.llm_tokenizer(x, return_tensors="pt", max_length=4096, truncation=True,add_special_tokens=False) for x in teacher_texts]
        student_tokenized = [self.llm_tokenizer(x, return_tensors="pt", max_length=4096, truncation=True,add_special_tokens=False) for x in student_texts]
        
        for i in range(len(teacher_texts)):
            teacher_len = teacher_tokenized[i]["input_ids"].shape[-1]
            student_len = student_tokenized[i]["input_ids"].shape[-1]
            delta = teacher_len - student_len
            if delta > 0:
                masked_doc = doc_mask_token * (2 + delta)
                student_inputs[i][0]["content"] = user_template.format(passage=masked_doc, question=batch_question[i])

        student_texts = [self.llm_tokenizer.apply_chat_template(x, tokenize=False) for x in student_inputs]
        student_tokenized = self.llm_tokenizer(student_texts, return_tensors="pt", padding=True,max_length=4096, truncation=True,add_special_tokens=False)
        teacher_tokenized = self.llm_tokenizer(teacher_texts, return_tensors="pt", padding=True,max_length=4096, truncation=True,add_special_tokens=False)
        teacher_logits_label_mask = self._get_logits_label_mask(teacher_tokenized["input_ids"], pad_token_id=self.llm_tokenizer.pad_token_id)
        student_logits_label_mask = self._get_logits_label_mask(student_tokenized["input_ids"], pad_token_id=self.llm_tokenizer.pad_token_id)
        return teacher_tokenized, student_tokenized, teacher_logits_label_mask, student_logits_label_mask
    def _construct_lm_inputs_nomask(self, batch_question, batch_answer, batch_passage, doc_mask_token):
        teacher_inputs = []
        student_inputs = []
        user_template = "You should answer the question by referring to the knowledge provided below and integrating your own knowledge.\nPassages:\n{passage}\n\nQuestion: {question}"
        assistant_template = "The answer is {answer}"
        for q, a, p in zip(batch_question, batch_answer, batch_passage):
            user_msg = {"role": "user", "content": user_template.format(passage=p, question=q)}
            assistant_msg = {"role": "assistant", "content": assistant_template.format(answer=a)}
            teacher_inputs.append([user_msg, assistant_msg])
            student_msg = {"role": "user", "content": user_template.format(passage="", question=q)}
            student_inputs.append([student_msg, assistant_msg])

        teacher_texts = [self.llm_tokenizer.apply_chat_template(x, tokenize=False) for x in teacher_inputs]
        student_texts = [self.llm_tokenizer.apply_chat_template(x, tokenize=False) for x in student_inputs]
        student_tokenized = self.llm_tokenizer(student_texts, return_tensors="pt", padding=True,max_length=4096, truncation=True,add_special_tokens=False)
        teacher_tokenized = self.llm_tokenizer(teacher_texts, return_tensors="pt", padding=True,max_length=4096, truncation=True,add_special_tokens=False)
        teacher_logits_label_mask = self._get_logits_label_mask(teacher_tokenized["input_ids"], pad_token_id=self.llm_tokenizer.pad_token_id)
        student_logits_label_mask = self._get_logits_label_mask(student_tokenized["input_ids"], pad_token_id=self.llm_tokenizer.pad_token_id)
        return teacher_tokenized, student_tokenized, teacher_logits_label_mask, student_logits_label_mask
    def _construct_lm_inputs_fullanswer(self, batch_question, batch_answer, batch_passage, doc_mask_token):
        teacher_inputs = []
        student_inputs = []
        user_template = "You should answer the question by referring to the knowledge provided below and integrating your own knowledge.\nPassages:\n{passage}\n\nQuestion: {question}"
        assistant_template = "The answer is {answer}"
        for q, a, p in zip(batch_question, batch_answer, batch_passage):
            user_msg = {"role": "user", "content": user_template.format(passage=p, question=q)}
            assistant_msg = {"role": "assistant", "content": assistant_template.format(answer=a)}
            teacher_inputs.append([user_msg, assistant_msg])

            # doc mask 替换
            # tokenized = self.llm_tokenizer(p, return_tensors="pt",add_special_tokens=False)
            # token_len = tokenized["input_ids"].shape[1]
            masked_doc = doc_mask_token * 2

            masked_msg = {"role": "user", "content": user_template.format(passage=masked_doc, question=q)}
            student_inputs.append([masked_msg, assistant_msg])

        teacher_texts = [self.llm_tokenizer.apply_chat_template(x, tokenize=False) for x in teacher_inputs]
        student_texts = [self.llm_tokenizer.apply_chat_template(x, tokenize=False) for x in student_inputs]

        # 处理长度对齐（再判断差异）
        teacher_tokenized = [self.llm_tokenizer(x, return_tensors="pt", max_length=4096, truncation=True,add_special_tokens=False) for x in teacher_texts]
        student_tokenized = [self.llm_tokenizer(x, return_tensors="pt", max_length=4096, truncation=True,add_special_tokens=False) for x in student_texts]
        
        for i in range(len(teacher_texts)):
            teacher_len = teacher_tokenized[i]["input_ids"].shape[-1]
            student_len = student_tokenized[i]["input_ids"].shape[-1]
            delta = teacher_len - student_len
            if delta > 0:
                masked_doc = doc_mask_token * (2 + delta)
                student_inputs[i][0]["content"] = user_template.format(passage=masked_doc, question=batch_question[i])

        student_texts = [self.llm_tokenizer.apply_chat_template(x, tokenize=False) for x in student_inputs]
        student_tokenized = self.llm_tokenizer(student_texts, return_tensors="pt", padding=True,max_length=4096, truncation=True,add_special_tokens=False)
        teacher_tokenized = self.llm_tokenizer(teacher_texts, return_tensors="pt", padding=True,max_length=4096, truncation=True,add_special_tokens=False)
        logits_label_mask = self._get_logits_label_mask(student_tokenized["input_ids"])
        return teacher_tokenized, student_tokenized, logits_label_mask

    def safe_save_model(self, model, path):
        dir_name = os.path.dirname(path)
        os.makedirs(dir_name, exist_ok=True)
        save_file(model.state_dict(), path)
        if self.args.train_token:
            step = path.split("_")[-1].split(".")[0]
            save_file(
                {"doc_mask_token":self.docmask_embed.detach().cpu()},
                os.path.join(dir_name, f"doc_mask_token_{step}.safetensors")
            )

    def train_step_qa_alignment(self, batch, doc_mask_token='<|doc_mask|>'):
        questions = batch['question']
        answers = batch['answer']
        passages = batch['passage']
        full_answers  = batch['full_answer']

        filter_by_len=[]
        for i in range(len(passages)):
            if len(passages[i].split(' ')) < 2000:
                filter_by_len.append(i)
        questions = [questions[i] for i in filter_by_len]
        answers = [answers[i] for i in filter_by_len]
        passages = [passages[i] for i in filter_by_len]
        full_answers = [full_answers[i] for i in filter_by_len]
        if len(questions) == 0:
            return None, None

        if isinstance(self.translator, ParameterTranslator):
            doc_embed = self._get_doc_embed(passages)
            lora_weights = self.translator(doc_embed.to(self.devices['translator']))
        elif isinstance(self.translator, CrossAttentionParameterTranslator):
            doc_embed, attention_mask = self._get_full_doc_embed(passages)
            lora_weights = self.translator(doc_embed.to(self.devices['translator']), attention_mask=attention_mask.to(self.devices['translator']))
        elif isinstance(self.translator, CrossAttentionHyperNetworkParameterTranslator):
            doc_embed, attention_mask = self._get_full_doc_embed(passages)
            lora_weights = self.translator(doc_embed.to(self.devices['translator']), attention_mask=attention_mask.to(self.devices['translator']))
        else:
            raise ValueError(f"Unsupported translator type: {type(self.translator)}")

        teacher_tok, student_tok, teacher_logits_label_mask, student_logits_label_mask = self._construct_lm_inputs(questions, answers, passages, doc_mask_token)
        # teacher_tok, student_tok, logits_label_mask = self._construct_lm_inputs_fullanswer(questions, full_answers, passages, doc_mask_token)
        # teacher_tok, student_tok, teacher_logits_label_mask, student_logits_label_mask = self._construct_lm_inputs_nomask(questions, answers, passages, doc_mask_token)
        teacher_tok = {k: v.to(self.devices['llm_model']) for k, v in teacher_tok.items()}
        student_tok = {k: v.to(self.devices['llm_model']) for k, v in student_tok.items()}

        with torch.no_grad():
            teacher_outputs = self.llm_model(
                input_ids=teacher_tok['input_ids'],
                attention_mask=teacher_tok['attention_mask'],
                output_hidden_states=True,
            )
        labels = student_logits_label_mask.to(self.devices['llm_model']) * student_tok['attention_mask'] * student_tok['input_ids']
        labels[labels == 0] = -100

        delta_inject(self.llm_model, lora_weights)
        if self.args.train_token:
            inputs_embeds = self.llm_model.get_input_embeddings()(student_tok['input_ids'])
            doc_mask_token_id = self.llm_tokenizer.convert_tokens_to_ids("<|doc_mask|>")
            mask_positions = (student_tok['input_ids'] == doc_mask_token_id).nonzero(as_tuple=False)
            for batch_idx, seq_idx in mask_positions:
                inputs_embeds[batch_idx, seq_idx] = self.docmask_embed  # 替换向量，确保 self.docmask_embed 是 nn.Parameter
            student_outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=student_tok['attention_mask'],
                output_hidden_states=True,
                labels=labels,
            )
        else:
            student_outputs = self.llm_model(
                input_ids=student_tok['input_ids'],
                attention_mask=student_tok['attention_mask'],
                output_hidden_states=True,
                labels=labels,
            )
        # delta_remove(self.llm_model, lora_weights)
        total_loss, layer_losses, avg_hidden_loss, logits_loss, logits_ce_loss = compute_multi_loss_nomask(
            student_outputs,
            teacher_outputs,
            student_tok['attention_mask'],
            student_tok['input_ids'],
            teacher_logits_label_mask,
            student_logits_label_mask,
            hidden_loss_type=args.hidden_loss_type,
            logits_loss_type=args.logits_loss_type,
            alpha_zero=args.alpha_zero,
            alpha=args.alpha,
            beta=args.beta,
            gama=args.gama,
            temperature=args.kl_temperature,
        )
        # total_loss, layer_losses, avg_hidden_loss, logits_loss, logits_ce_loss = compute_multi_loss(
        #     student_outputs,
        #     teacher_outputs,
        #     student_tok['attention_mask'],
        #     student_tok['input_ids'],
        #     student_logits_label_mask,
        #     hidden_loss_type=args.hidden_loss_type,
        #     logits_loss_type=args.logits_loss_type,
        #     alpha_zero=args.alpha_zero,
        #     alpha=args.alpha,
        #     beta=args.beta,
        #     gama=args.gama,
        #     temperature=args.kl_temperature,
        # )

        return total_loss,layer_losses, avg_hidden_loss, logits_loss, logits_ce_loss, lora_weights
    def train_on_dataloader(self, dataloader, epochs=1):
        self.translator.train()
        self.embedding_model.eval()
        self.llm_model.eval()
        num_training_steps = len(dataloader) * epochs
        num_warmup_steps = int(num_training_steps * 0.1)
        self.init_optim_lrscheduler(
            lr=self.lr,
            optim_type="adamw",
            scheduler_type=self.scheduler_type,
            optim_config={},
            # lrscheduler_config={'total_iters': len(dataloader) * epochs, 'start_factor': 1.0, 'end_factor': 0.5, 'verbose': False}
            lrscheduler_config={
                'num_warmup_steps':num_warmup_steps,
                'num_training_steps':num_training_steps,
                'lr_end':1e-6,         # 最低学习率
                'power':1.0,# 学习率指数衰减的指数  
            }
        )

        for epoch in range(epochs):
            step_count = 0
            with tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}") as tbar:
                for batch in tbar:


                    # torch.cuda.empty_cache()
                    # torch.cuda.reset_peak_memory_stats()

                    self.optimizer.zero_grad(set_to_none=True)

                    # loss,layer_losses,logits_loss,logits_ce_loss = self.train_step_qa_alignment(batch)
                    step_count += 1
                    try:
                        loss,layer_losses, avg_hidden_loss,logits_loss,logits_ce_loss,lora_weights = self.train_step_qa_alignment(batch)
                    except Exception as e:
                        print(f"Error in train_step_qa_alignment: {e}")
                        print(f"Batch: {batch}")
                        print(f"step_count: {step_count}")
                        continue
                    if loss is None:
                        self.global_step += 1
                        continue

                    assert loss is not None and not torch.isnan(loss), "Loss is None or NaN!"
                    loss.backward()
                    # self._clip_gradients(self.translator, max_norm=1.0, max_value=10.0, verbose=True)
                    self.optimizer.step()
                    if self.scheduler:
                        self.scheduler.step()
                    delta_remove(self.llm_model, lora_weights)

                    self.global_step += 1
                    if self.log_tool == "wandb":
                        import wandb
                        if self.global_step % self.log_steps == 0:
                            current_lr = self.optimizer.param_groups[0]['lr']
                            wandb.log({
                                "backpropagation loss": loss.item(),
                                "avg_hidden_loss": avg_hidden_loss,
                                "logits_alignment_loss": logits_loss,
                                "logits_ce_loss": logits_ce_loss,
                                "learning_rate": current_lr,
                                **{f"layer_loss/{i}": layer_loss for i, layer_loss in enumerate(layer_losses)}
                            }, step=self.global_step)
                    elif self.log_tool == "swanlab":
                        import swanlab
                        if self.global_step % self.log_steps == 0:
                            current_lr = self.optimizer.param_groups[0]['lr']
                            swanlab.log({
                                "backpropagation loss": loss.item(),
                                "avg_hidden_loss": avg_hidden_loss,
                                "logits_alignment_loss": logits_loss,
                                "logits_ce_loss": logits_ce_loss,
                                "learning_rate": current_lr,
                                **{f"layer_loss/{i}": layer_loss for i, layer_loss in enumerate(layer_losses)}
                            }, step=self.global_step)
                    elif self.log_tool == "tensorboard":
                        if self.global_step % self.log_steps == 0:
                            current_lr = self.optimizer.param_groups[0]['lr']
                            self.tb_writer.add_scalar("backpropagation loss", loss.item(), self.global_step)
                            self.tb_writer.add_scalar("avg_hidden_loss", avg_hidden_loss, self.global_step)
                            self.tb_writer.add_scalar("logits_alignment_loss", logits_loss, self.global_step)
                            self.tb_writer.add_scalar("logits_ce_loss", logits_ce_loss, self.global_step)
                            self.tb_writer.add_scalar("learning_rate", current_lr, self.global_step)
                            for i, layer_loss in enumerate(layer_losses):
                                self.tb_writer.add_scalar(f"layer_loss/{i}", layer_loss, self.global_step)
                    else:
                        if self.global_step % self.log_steps == 0:
                            print(f"[Step {self.global_step}] loss: {loss.item():.4f}")

                    if self.global_step % self.saving_steps == 0:
                        self.safe_save_model(self.translator, os.path.join(self.save_path, f"translator_step_{self.global_step}.safetensors"))

                    # torch.cuda.empty_cache()
        self.safe_save_model(self.translator, os.path.join(self.save_path, f"translator_step_{self.global_step}.safetensors"))
        if self.log_tool == "wandb":
            import wandb
            wandb.finish()
        elif self.log_tool == "swanlab":
            import swanlab
            swanlab.finish()
class ChainLoader:
    """
    拼接多个 DataLoader，行为与 PyTorch 的 DataLoader 类似：
    - 每轮 epoch 都可以重复遍历；
    - 支持 len()；
    - 保持各个子 DataLoader 的原有配置（如 shuffle、batch_size）。
    """

    def __init__(self, *loaders):
        """
        初始化 ChainLoader。

        参数：
            *loaders: 任意数量的 PyTorch DataLoader 对象。
        """
        self.loaders = loaders

    def __iter__(self):
        """
        每次迭代重新构建迭代器，确保支持多轮 epoch。
        """
        return chain(*self.loaders)

    def __len__(self):
        """
        返回所有子 DataLoader 的 batch 总数之和。
        """
        return sum(len(loader) for loader in self.loaders)
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--embedding_model_path", type=str, default="./models/long-t5-tglobal-base")
    parser.add_argument("--llm_model_path", type=str, default="./models/Llama-3.2-1B-Instruct-Doc_mask")
    parser.add_argument("--save_path", type=str, default="./models/Llama-3.2-1B-Instruct-Doc_mask-longt5_temp")
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--dataset2_path", type=str, default="")
    parser.add_argument("--translator_type", type=str, default="cross-attention-hyper-parameter-translator")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_loss_type", type=str, default="cosine", choices=["mse", "cosine"])
    parser.add_argument("--logits_loss_type", type=str, default="kl", choices=["mse", "kl"])
    parser.add_argument("--alpha_zero", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--gama", type=float, default=1.0)
    parser.add_argument("--kl_temperature", type=float, default=2.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--scheduler_type", type=str, default="linear", choices=[None, "cosine", "linear"])
    parser.add_argument("--log_tool", type=str, default=None, choices=[None, "wandb",  "tensorboard", "swanlab"])
    parser.add_argument("--log_description", type=str, default=None)
    parser.add_argument("--log_steps", type=int, default=1)
    parser.add_argument("--saving_steps", type=int, default=100)
    parser.add_argument("--device_translator", type=str, default="cuda:5")
    parser.add_argument("--device_embedding", type=str, default="cuda:5")
    parser.add_argument("--device_llm", type=str, default="cuda:5")
    parser.add_argument("--train_token", type=int, default=0)
    return parser.parse_args()
if __name__ == "__main__":

    set_seed(42)
    args = get_args()

    print("================= Training Configuration =================")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("==========================================================")
    pid = os.getpid()
    print(f"[INFO] Python process PID: {pid}")
    TRAIN_TYPE=torch.float32

    # while True:
    #     if not psutil.pid_exists(2896710):
    #         break
    #     time.sleep(1)    

    if 'snowflake' in args.embedding_model_path:
        embedding_model = AutoModel.from_pretrained(args.embedding_model_path, add_pooling_layer=False, trust_remote_code=True,device_map=args.device_embedding)
    else:
        embedding_model = AutoModel.from_pretrained(args.embedding_model_path,device_map=args.device_embedding)
    embedding_tokenizer = AutoTokenizer.from_pretrained(args.embedding_model_path)
    llm_model = AutoModelForCausalLM.from_pretrained(args.llm_model_path,device_map=args.device_llm)
    llm_tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path)
    llm_tokenizer.padding_side = "right"
    llm_tokenizer.pad_token = llm_tokenizer.eos_token

    doc_mask_token = "<|doc_mask|>"
    # llm_tokenizer.chat_template=llm_tokenizer.chat_template.replace("Cutting Knowledge Date: December 2023\\n","Cutting Knowledge Date: December 2023\\nBut you have additional document knowledge in your lora weights which may contain the latest knowledge.\\n")

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=2,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=["down_proj", "up_proj", "gate_proj"],  # 指定 LoRA 注入的模块
    )
    llm_model = get_peft_model(llm_model, peft_config)
    print(type(llm_model.base_model.model.model.layers[0].mlp.up_proj))

    llm_model.config.pad_token_id = llm_tokenizer.pad_token_id

    embedding_model.eval()
    llm_model.eval()
    if args.train_token:
        for param in llm_model.parameters():
            param.requires_grad = False
    try:
        dataset_path = args.dataset_path
        dataset1 = load_from_disk(dataset_path)
    except:
        pass
    try:
        dataset2_path = args.dataset2_path
        dataset2 = load_from_disk(dataset2_path)
    except:
        pass
        # dataset = concatenate_datasets([dataset1, dataset2])
    # else:
    #     passages=[              
    #         # "Donald Trump took the oath of office as the nation’s 47th president at 12:02 p.m. on Monday, marking a historic comeback for a president who has promised to disrupt Washington even more so than he did during his first term. With four predecessors, several supportive billionaires and scores of elected officials looking on, Trump became president for a second time inside the same Capitol building his supporters stormed four years ago in an effort to halt Congress’ ratification of his defeat. It was the first time in more than a century that a former president has taken the oath for a second time after leaving office, with the 45th and now 47th president following in the footsteps of Grover Cleveland, the only other president to serve nonconsecutive terms."
    #         # "Donald Trump took the oath of office as the nation’s 47th president at 12:02 p.m. on Monday.",
    #         # "He is the son of actress Magorzata Braunek and director Andrzej."
    #         # "included, yet it carries a pervasive bitterness toward war that resonates until the impactful closing title, mirroring Sinatra's progressive stance during that period. Horton notes that while Clint Eastwood was widely praised for directing two films depicting World War II from both American and Japanese perspectives (\"Flags of Our Fathers\" and \"Letters from Iwo Jima\"), Sinatra had essentially achieved the same feat earlier in a single film. *None but the Brave*, released in 1965 and also titled differently in Japan, is a war movie starring Frank Sinatra.",
    #         "one two three.",
    #         "one two three four."
    #     ]
    #     questions =[
    #         "What is the main theme of the movie 'None but the Brave' as described in the passage?",
    #         "What is the main theme of the movie 'None but the Brave'?",
    #     ]
    #     answers=[
    #         "Bitterness about war one two three four.",
    #         "Bitterness about war"
    #     ]
    #     full_answers=[
    #         "Bitterness about war",
    #         "Bitterness about war"
    #     ]
    #     dataset = Dataset.from_dict({
    #         "passage": passages,
    #         "question": questions,
    #         "answer": answers,
    #         "full_answer": full_answers,
    #     })
    dataset_paths = [
        "./data_aug_projector/2wikimultihopqa/llama3.2-1b-instruct/bridge_comparison.json",
        "./data_aug_projector/2wikimultihopqa/llama3.2-1b-instruct/comparison.json",
        "./data_aug_projector/2wikimultihopqa/llama3.2-1b-instruct/compositional.json",
        "./data_aug_projector/2wikimultihopqa/llama3.2-1b-instruct/inference.json",
        "./data_aug_projector/complexwebquestions/llama3.2-1b-instruct/total.json",
        "./data_aug_projector/hotpotqa/llama3.2-1b-instruct/bridge.json",
        "./data_aug_projector/hotpotqa/llama3.2-1b-instruct/comparison.json",
        "./data_aug_projector/popqa/llama3.2-1b-instruct/total.json"
    ]
    questions = []
    answers = []
    full_answers = []
    passages = []
    for dataset_path in dataset_paths:
        with  open(dataset_path, "r") as f:
            dataset = json.load(f)
        for item in dataset:
            for augment in item['augment']:
                qa_cnt = (len(augment['llama3.2-1b-instruct_qa'])+1)//2
                for idx,qa in enumerate(augment['llama3.2-1b-instruct_qa']):
                    if idx < qa_cnt:
                        questions.append(qa['question'])
                        answers.append(qa['answer'])
                        full_answers.append(qa['full_answer'])
                        passages.append(augment['llama3.2-1b-instruct_rewrite'])
                    questions.append(qa['question'])
                    answers.append(qa['answer'])
                    full_answers.append(qa['full_answer'])
                    passages.append(augment['passage'])
    dataset = Dataset.from_dict({
        "passage": passages,
        "question": questions,
        "answer": answers,
        "full_answer": full_answers,
    })


    # print("dataset length: ",len(dataset))
    # dataloader = DataLoader(
    #     dataset,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    # )
    try:
        # dataloader1= DataLoader(
        #     dataset1,
        #     batch_size=args.batch_size,
        #     shuffle=True,
        # )
        # dataloader2 = DataLoader(
        #     dataset2,
        #     batch_size=args.batch_size,
        #     shuffle=True,
        # )
        # dataloader = ChainLoader(dataloader1, dataloader2)
        try:
            dataset = concatenate_datasets([dataset1, dataset2])
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=True,
            )
        except:
            dataloader = DataLoader(
                dataset1,
                batch_size=args.batch_size,
                shuffle=True,
            )
    except:
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
        )
    print("dataloader length: ",len(dataloader))
    # Initialize translator
    if args.translator_type == "parameter-translator":
        translator = ParameterTranslator(
            embedding_model=embedding_model,
            llm_model=llm_model,
            lora_rank=2,
            projector_hidden_dim=512
        )#66M
    elif args.translator_type == "cross-attention-parameter-translator-s":
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
    elif args.translator_type == "cross-attention-parameter-translator-l":
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
    elif args.translator_type == "cross-attention-hyper-parameter-translator":
        translator = CrossAttentionHyperNetworkParameterTranslator(
            embedding_model=embedding_model,
            llm_model=llm_model,
            lora_rank=2,
            projector_hidden_dim=2048,
            attn_heads=8,
            cross_layers=1,
        )
    else:
        raise ValueError(f"Unknown translator type: {args.translator_type}")
    print("translator:\n",translator)
    translator=translator.to(args.device_translator).to(TRAIN_TYPE)
    # print(translator)
    translator.train()
    trainer = DyPRAGTrainer(
        translator=translator,
        embedding_model=embedding_model,
        llm_model=llm_model,
        llm_tokenizer=llm_tokenizer,
        embedding_tokenizer=embedding_tokenizer,
        scheduler_type=args.scheduler_type,
        lr = args.lr,
        log_steps=args.log_steps,
        saving_steps=args.saving_steps,
        save_path=args.save_path,
        log_tool=args.log_tool,
        log_description=args.log_description,
        args=args,
    )
    trainer.train_on_dataloader(dataloader, epochs=args.epochs)