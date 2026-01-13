# Preprocess_Llama.py  ——  EOS Hidden pooling (default) + optional PC1 removal
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Literal, Optional
from transformers import LlamaForCausalLM, LlamaTokenizer

PoolType = Literal["eos", "eos_embed", "mean"]

class Model(nn.Module):
    """
    入力: List[str]  （各系列の説明テキスト。末尾に固有アンカー [SER=...] を置くのが推奨）
    出力: ve [C, D]  （各系列ベクトル；行方向L2正規化済み）

    既定：
      - プーリング: "eos"（Transformerを通した末尾トークン＝EOSの隠れ状態）
      - LLaMA本体: 凍結・推論のみ（前処理なのでOK）
    切替：
      - configs.ve_pooling in {"eos","eos_embed","mean"}
      - configs.ve_use_transformer in {True, False}
         * "eos" で use_transformer=False の場合は、embed_tokens の末尾トークン埋め込み（軽量）
         * "mean" は embed_tokens の PAD 除去平均（旧実装）
    """
    def __init__(self, configs):
        super().__init__()
        self.pooling: PoolType = getattr(configs, "ve_pooling", "eos")
        # "eos" では True 推奨（文脈を通す）
        self.use_transformer: bool = getattr(
            configs, "ve_use_transformer",
            True if self.pooling == "eos" else False
        )

        # モデル読み込み
        self.llama = LlamaForCausalLM.from_pretrained(
            getattr(configs, "llm_ckp_dir", "./llama"),
            device_map="auto",
            torch_dtype=(torch.float16 if torch.cuda.is_available() else torch.float32),
        ).eval()

        self.tokenizer = LlamaTokenizer.from_pretrained(getattr(configs, "llm_ckp_dir", "./llama"))
        if self.tokenizer.pad_token is None:
            # LLaMA系は pad_token が未定義のことがあるので、EOS をPADとして使う
            self.tokenizer.pad_token = self.tokenizer.eos_token

        for p in self.llama.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, series_texts: List[str]) -> torch.Tensor:
        """
        series_texts: 各系列用の短い固有テキスト（末尾に [SER=xxx] アンカー推奨）
        return: ve [C, D]  （行方向L2）
        """
        # トークナイズ
        toks = self.tokenizer(
            list(series_texts),
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        device = next(self.llama.parameters()).device
        toks = {k: v.to(device) for k, v in toks.items()}
        attn = toks["attention_mask"]                  # [C, L]
        C, L = attn.shape
        last_idx = attn.sum(dim=1) - 1                 # [C] 末尾の非PAD位置
        last_idx = last_idx.clamp_min(0)

        # プーリング分岐
        if self.pooling == "eos":
            if self.use_transformer:
                # 推奨：Transformerを通して EOS の隠れ状態を取る
                hidden = self.llama.model(**toks).last_hidden_state   # [C, L, D]
                ve = hidden[torch.arange(C, device=device), last_idx] # [C, D]
            else:
                # 軽量：embed_tokens の末尾トークン埋め込み
                tok_emb = self.llama.model.embed_tokens(toks["input_ids"])  # [C, L, D]
                ve = tok_emb[torch.arange(C, device=device), last_idx]       # [C, D]

        elif self.pooling == "eos_embed":
            # 明示的に“末尾embed”を選ぶ場合
            tok_emb = self.llama.model.embed_tokens(toks["input_ids"])      # [C, L, D]
            ve = tok_emb[torch.arange(C, device=device), last_idx]          # [C, D]

        elif self.pooling == "mean":
            # 旧: PAD 除去平均（共通語の影響が強くなりやすい）
            tok_emb = self.llama.model.embed_tokens(toks["input_ids"])      # [C, L, D]
            ve = (tok_emb * attn.unsqueeze(-1)).sum(dim=1) / attn.sum(dim=1, keepdim=True).clamp_min(1.0)

        else:
            raise ValueError(f"unknown pooling: {self.pooling}")

        # 行方向L2正規化
        ve = F.normalize(ve, dim=-1)
        return ve.detach().to("cpu")  # [C, D]

    # --------------------------- optional utils --------------------------- #
    @staticmethod
    @torch.no_grad()
    def remove_pc1(ve: torch.Tensor) -> torch.Tensor:
        """
        共通方向（第1主成分）を除去して再L2正規化。
        入力: ve [C, D]（行方向L2済み推奨）
        出力: ve_denoised [C, D]
        """
        assert ve.dim() == 2, f"ve must be 2D [C, D], got {ve.shape}"
        device = ve.device
        mu = ve.mean(dim=0, keepdim=True)
        Xc = ve - mu
        # SVD（PCA）
        U, S, Vt = torch.linalg.svd(Xc, full_matrices=False)
        pc1 = Vt[0:1, :]                             # [1, D]
        proj = (Xc @ pc1.T) @ pc1                    # [C, D]
        ve_denoised = Xc - proj
        ve_denoised = F.normalize(ve_denoised, dim=-1)
        return ve_denoised.to(device)
