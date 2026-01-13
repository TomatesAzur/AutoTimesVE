# models/AutoTimes_Llama.py
import torch
import torch.nn as nn
from transformers import LlamaForCausalLM
from layers.mlp import MLP


class Model(nn.Module):
    """
    TE風の合成に置換:
      - 数値 x_enc [B,L,C] を z-score -> encoder(C->D) -> times_embeds [B,L,D]
      - 構造VE x_mark_enc [B,L,D]（前処理で作った1文ベクトルをトークンにブロードキャスト）
      - 両者を正規化して加算（learnable scale 付き）
      - LLaMA(凍結) -> decoder(D->C) -> [B,L,C]
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        # device
        if configs.use_multi_gpu:
            self.device = f"cuda:{configs.local_rank}"
        else:
            self.device = f"cuda:{configs.gpu}"
        print(self.device)

        # LLaMA 読み込み（本体は凍結）
        self.llama = LlamaForCausalLM.from_pretrained(
            configs.llm_ckp_dir,
            device_map=self.device,
            torch_dtype=torch.float16 if configs.use_amp else torch.float32,
        )
        for p in self.llama.parameters():
            p.requires_grad = False

        # hidden 次元
        self.hidden_dim_of_llama = getattr(self.llama.config, "hidden_size", 4096)

        # ハイパラ保存
        self._use_mlp           = (configs.mlp_hidden_layers != 0)
        self._mlp_hidden_dim    = configs.mlp_hidden_dim
        self._mlp_hidden_layers = configs.mlp_hidden_layers
        self._mlp_activation    = configs.mlp_activation
        self._dropout           = configs.dropout

        # encoder/decoder を遅延構築
        self.encoder = None
        self.decoder = None
        self._c_dim  = getattr(configs, "c_dim", None)  # Cが分かっていれば先に作る
        if self._c_dim is not None:
            self._lazy_build_tokennets(self._c_dim)

        # TE風の加算を有効化
        self.mix = True if getattr(configs, "mix_embeds", True) else False
        if self.mix:
            p = next(self.llama.parameters())
            # learnable scale（FP32固定）
            self.add_scale = nn.Parameter(torch.ones([], device=p.device, dtype=torch.float32))

    def _lazy_build_tokennets(self, c_dim: int):
        """ 数値C<->D の写像（学習対象）を FP32 で作る """
        if (self.encoder is not None) and (self._c_dim == c_dim):
            return
        self._c_dim = c_dim
        if self._use_mlp:
            print("use mlp as tokenizer and detokenizer (lazy build)")
            self.encoder = MLP(self._c_dim, self.hidden_dim_of_llama,
                               self._mlp_hidden_dim, self._mlp_hidden_layers,
                               self._dropout, self._mlp_activation)
            self.decoder = MLP(self.hidden_dim_of_llama, self._c_dim,
                               self._mlp_hidden_dim, self._mlp_hidden_layers,
                               self._dropout, self._mlp_activation)
        else:
            print("use linear as tokenizer and detokenizer (lazy build)")
            self.encoder = nn.Linear(self._c_dim, self.hidden_dim_of_llama)
            self.decoder = nn.Linear(self.hidden_dim_of_llama, self._c_dim)

        llama_dev = next(self.llama.parameters()).device
        self.encoder.to(device=llama_dev, dtype=torch.float32)
        self.decoder.to(device=llama_dev, dtype=torch.float32)

    @torch.no_grad()
    def _zscore(self, x: torch.Tensor):
        """
        各変量ごと（C軸）に時刻方向 L で z-score
        x: [B,L,C] -> (x_norm, means[B,1,C], stdev[B,1,C])
        """
        means = x.mean(dim=1, keepdim=True)
        var   = x.var(dim=1, keepdim=True, unbiased=False)
        stdev = torch.sqrt(var + 1e-5)
        return (x - means) / stdev, means, stdev

    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        """
        期待する引数:
          x_enc:      [B,L,C]
          x_mark_enc: [B,L,D]  ← 構造VE（1文ベクトルを各トークンにブロードキャスト）
        出力:
          dec_out:    [B,L,C]
        """
        assert x_mark_enc is not None and x_mark_enc.dim() == 3, \
            f"x_mark_enc must be [B,L,D], got {None if x_mark_enc is None else x_mark_enc.shape}"

        # 標準化（係数の安定化）
        x, means, stdev = self._zscore(x_enc)  # [B,L,C]
        B, L, C = x.shape

        # encoder/decoder をCに合わせて用意
        self._lazy_build_tokennets(C)

        # 数値 -> D
        enc_param  = next(self.encoder.parameters())  # FP32
        x = x.to(device=enc_param.device, dtype=enc_param.dtype)
        times_embeds = self.encoder(x.reshape(B * L, C)).reshape(B, L, -1)  # [B,L,D]

        # 構造VE（x_mark_enc）と加算（TEと同じ正規化＋learnable scale）
        if self.mix:
            x_mark_enc = x_mark_enc.to(device=times_embeds.device, dtype=enc_param.dtype)
            times_embeds = times_embeds / (times_embeds.norm(dim=2, keepdim=True) + 1e-6)
            x_mark_enc   = x_mark_enc   / (x_mark_enc.norm(dim=2, keepdim=True) + 1e-6)
            # learnable scale は FP32 のまま
            times_embeds = times_embeds + self.add_scale * x_mark_enc

        # LLaMA 本体へ（dtype は LLaMA に合わせる）
        llama_param  = next(self.llama.parameters())
        times_embeds = times_embeds.to(device=llama_param.device, dtype=llama_param.dtype)
        hidden = self.llama.model(inputs_embeds=times_embeds)[0]  # [B,L,D]

        # D -> C （学習対象はFP32）
        dec_param = next(self.decoder.parameters())
        hidden = hidden.to(device=dec_param.device, dtype=dec_param.dtype)
        dec_out = self.decoder(hidden.reshape(B * L, -1)).reshape(B, L, C)
        return dec_out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
