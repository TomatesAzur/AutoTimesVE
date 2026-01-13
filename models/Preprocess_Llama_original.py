import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaTokenizer

class Model(nn.Module):
    """
    入力: ["This is the series of FOODS_1", "This is the series of FOODS_2", ...]
    出力: VE [C, D] (D = llama.hidden_size, 例: 4096)
    """
    def __init__(self, configs):
        super().__init__()
        self.llama = LlamaForCausalLM.from_pretrained(
            configs.llm_ckp_dir,
            device_map="auto",
            torch_dtype=torch.float16,
        ).eval()
        self.tokenizer = LlamaTokenizer.from_pretrained(configs.llm_ckp_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        for p in self.llama.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, series_texts):
        texts = list(series_texts)
        toks = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        toks = {k: v.to(self.llama.device) for k, v in toks.items()}

        # Transformer を回さずに語彙埋め込みだけ
        tok_emb = self.llama.model.embed_tokens(toks["input_ids"])  # [C, L, D]

        # PAD を除いた平均
        mask = toks["attention_mask"].unsqueeze(-1)                 # [C, L, 1]
        tok_emb = tok_emb * mask
        denom = mask.sum(dim=1).clamp_min(1)                        # [C, 1]
        ve = tok_emb.sum(dim=1) / denom                             # [C, D]

        # 安定化（後で足し算するなら特に有効）
        ve = torch.nn.functional.normalize(ve, dim=-1)

        return ve.detach().to("cpu")  # [C, D]
    
    # @torch.no_grad()
    # def forward(self, series_texts):
    #     toks = self.tokenizer(series_texts, return_tensors="pt", padding=True, truncation=True)
    #     toks = {k: v.to(self.llama.device) for k, v in toks.items()}
    #     tok_emb = self.llama.model.embed_tokens(toks["input_ids"])  # [C, L, D]
    #     mask = toks["attention_mask"]                                # [C, L]
    #     # 各系列の「最後の非PAD位置」を求める
    #     last_idx = mask.sum(dim=1) - 1                               # [C]
    #     # 末尾トークンの埋め込みを取り出す
    #     ve = tok_emb[torch.arange(tok_emb.size(0), device=tok_emb.device), last_idx]  # [C, D]
    #     ve = torch.nn.functional.normalize(ve, dim=-1)
    #     return ve.detach().to("cpu")

    # @torch.no_grad()
    # def forward(self, series_texts):
    #     toks = self.tokenizer(series_texts, return_tensors="pt", padding=True, truncation=True)
    #     toks = {k: v.to(self.llama.device) for k, v in toks.items()}
    #     # LLaMA を1段通す（本体は凍結済み）
    #     out = self.llama.model(**toks).last_hidden_state            # [C, L, D]
    #     mask = toks["attention_mask"]                               # [C, L]
    #     last_idx = mask.sum(dim=1) - 1                              # [C]
    #     ve = out[torch.arange(out.size(0), device=out.device), last_idx]   # [C, D]
    #     ve = torch.nn.functional.normalize(ve, dim=-1)
    #     return ve.detach().to("cpu")
