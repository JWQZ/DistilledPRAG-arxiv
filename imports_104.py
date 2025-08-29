import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["WANDB_MODE"] = "offline"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft.tuners.lora.layer import Linear as PeftLinear
from typing import Any, Optional, Union, List
class DynamicLinear(PeftLinear):
    """
    Linear layer with optional delta override.
    If `self.delta` is set (e.g., injected from Translator), this overrides the standard lora_A -> lora_B logic.
    """

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:

            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype

            lora_A_keys = self.lora_A.keys()
            # 🔥 Our injected path
            if (hasattr(self, "delta") or hasattr(self, "delta_lora_a")):  
                if hasattr(self, "delta_lora_a"):
                    delta_lora_a = self.delta_lora_a.to(result.device)
                    delta_lora_b = self.delta_lora_b.to(result.device)
                    dropout = self.lora_dropout['default']
                    x = self._cast_input_dtype(x, delta_lora_a.dtype)
                else:
                    delta = self.delta.to(result.device)
                    dropout = self.lora_dropout['default']
                    x = self._cast_input_dtype(x, delta.dtype)
                if x.dim() == 3:  # [B, L, D]
                    if hasattr(self, "delta_lora_a"):
                        delta_result = torch.matmul(x, (delta_lora_b @ delta_lora_a).transpose(-1, -2))
                    else:
                        delta_result = torch.matmul(x, delta.transpose(-1, -2))  # [B, L, D]
                    # print("delta_result mean:", delta_result.mean().item())
                    # print("delta_result std:", delta_result.std().item())
                    result += delta_result
                elif x.dim() == 2:  # [B, D]
                    if hasattr(self, "delta_lora_a"):
                        delta_result = torch.matmul(x, (delta_lora_b @ delta_lora_a).transpose(-1, -2))
                    else:
                        delta_result = torch.matmul(x, delta.transpose(-1, -2))
                    # print("delta_result mean:", delta_result.mean().item())
                    # print("delta_result std:", delta_result.std().item())
                    result += delta_result
                else:
                    raise ValueError(f"[DynamicLoRA] Unsupported x shape: {x.shape}")
            else:
                for active_adapter in self.active_adapters:
                    if active_adapter not in lora_A_keys:
                        continue

                    lora_A = self.lora_A[active_adapter]
                    lora_B = self.lora_B[active_adapter]
                    dropout = self.lora_dropout[active_adapter]
                    scaling = self.scaling[active_adapter]
                    x = self._cast_input_dtype(x, lora_A.weight.dtype)

                    if not self.use_dora[active_adapter]:
                        result = result + lora_B(lora_A(dropout(x))) * scaling
                    else:
                        if isinstance(dropout, nn.Identity) or not self.training:
                            base_result = result
                        else:
                            x = dropout(x)
                            base_result = None

                        result = result + self.lora_magnitude_vector[active_adapter](
                            x,
                            lora_A=lora_A,
                            lora_B=lora_B,
                            scaling=scaling,
                            base_layer=self.get_base_layer(),
                            base_result=base_result,
                        )
            result = result.to(torch_result_dtype)

        return result
from peft import tuners,PeftModel
tuners.lora.layer.Linear = DynamicLinear
from peft import LoraConfig, TaskType, get_peft_model
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForCausalLM,T5Tokenizer,T5EncoderModel,PreTrainedModel,AutoModel,LongT5Model,LongT5EncoderModel
from tqdm import tqdm
import psutil
import time

from datasets import Dataset,load_dataset,load_from_disk,concatenate_datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import math
from collections import defaultdict
from utils_104 import *
from safetensors.torch import save_file,load_file
import random
import numpy as np
import json
import Levenshtein
import concurrent.futures
from openai import OpenAI   
import argparse
import re
from collections import Counter
import pandas as pd
import signal
import threading
import sys
import time
import warnings
import string
from itertools import chain
