# models/AutoTimes_Llama.py
import torch
import torch.nn as nn
from transformers import LlamaForCausalLM
from layers.mlp import MLP
import torch.nn.functional as F  # まだ無ければ
# --- add: small helper ---
def _l2norm(x, dim=-1, eps=1e-12):
    return x / (x.norm(dim=dim, keepdim=True).clamp_min(eps))

# --- add: FiLM adapter (最小) ---
class FiLMAdapter(nn.Module):
    """
    FiLM: h <- (1 + eps * gamma(ctx)) ⊙ norm(h) + eps * beta(ctx)
    h:   [B,L,D]  （LLaMAへ渡す直前の埋め込み）
    ctx: [B,D] or [B,L,D]（static なら [B,D], dynamic なら [B,L,D]）
    """
    def __init__(self, d_model: int, hidden: int = 0, eps_init: float = 0.1, use_tanh: bool = False, dropout: float = 0.0):
        super().__init__()
        self.use_tanh = use_tanh
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if hidden > 0:
            self.net = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, hidden), nn.SiLU(),
                nn.Linear(hidden, 2*d_model),
            )
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
        else:
            self.net = nn.Linear(d_model, 2*d_model)
            nn.init.zeros_(self.net.weight)
            nn.init.zeros_(self.net.bias)
        # 変調強度は学習可能（小さめから）
        self.eps = nn.Parameter(torch.tensor(float(eps_init), dtype=torch.float32))

    def forward(self, h: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        B, L, D = h.shape
        h32 = _l2norm(h.to(torch.float32), dim=-1)
        if ctx.dim() == 2:  # [B,D] → 各時刻へブロードキャスト
            gb = self.net(ctx.to(torch.float32))      # [B,2D]
            gamma, beta = gb.chunk(2, dim=-1)
            if self.use_tanh:
                gamma, beta = torch.tanh(gamma), torch.tanh(beta)
            out = (1.0 + self.eps * gamma).unsqueeze(1) * h32 + self.eps * beta.unsqueeze(1)
        elif ctx.dim() == 3:  # [B,L,D] → 時刻ごと
            gb = self.net(ctx.reshape(B*L, D).to(torch.float32))  # [B*L,2D]
            gamma, beta = gb.chunk(2, dim=-1)
            if self.use_tanh:
                gamma, beta = torch.tanh(gamma), torch.tanh(beta)
            out = (1.0 + self.eps * gamma.reshape(B,L,D)) * h32 + self.eps * beta.reshape(B,L,D)
        else:
            raise ValueError(f"ctx dim must be 2 or 3, got {ctx.dim()}")
        return self.drop(out).to(h.dtype)


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

        # --- add: FiLM flags (後方互換: 指定なければ全てデフォルトOFF/無影響) ---
        self.use_film    = bool(getattr(configs, "film", False))
        self.film_mode   = str(getattr(configs, "film_mode", "static"))   # "static" | "dynamic"
        self.film_hidden = int(getattr(configs, "film_hidden", 0))        # 0: 線形, >0: 小MLP
        self.film_eps0   = float(getattr(configs, "film_eps", 0.1))       # 0.05〜0.2 推奨
        self.film_tanh   = bool(getattr(configs, "film_tanh", False))
        self.film_dropout= float(getattr(configs, "film_dropout", 0.0))

        if self.use_film:
            # FiLM有効時は従来の「加算ミックス」を無効化するのが安全（二重注入を避ける）
            self.mix = False
            # llama の埋め込み次元を取得（既存の self.llama を使う）
            d_model = int(self.llama.model.embed_tokens.weight.shape[1])
            self.film = FiLMAdapter(
                d_model=d_model,
                hidden=self.film_hidden,
                eps_init=self.film_eps0,
                use_tanh=self.film_tanh,
                dropout=self.film_dropout,
            )
        self.film_gate = nn.Parameter(torch.tensor(0.0))

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
        
        # --- add: FiLM 分岐（最小） ---
        if self.use_film:
            # ctx の作り方：
            #   - static: 系列ごとの定常ベクトル（x_mark_enc の L 平均 or 先頭）
            #   - dynamic: 各時刻の文脈（x_mark_enc そのもの）
            if self.film_mode == "static":
                # [B,L,D] -> [B,D]（L 平均が無難。先頭でも可）
                ctx = _l2norm(x_mark_enc.mean(dim=1).to(torch.float32), dim=-1)
            elif self.film_mode == "dynamic":
                # [B,L,D] のまま（未来での更新は非推奨。まずは static を使う）
                ctx = _l2norm(x_mark_enc.to(torch.float32), dim=-1)
            else:
                raise ValueError(f"unknown film_mode: {self.film_mode}")
            

            h_before = F.normalize(times_embeds.detach().to(torch.float32), dim=-1)
            # FiLM で変調（内部で FP32→L2→元dtype に戻します）
            # FiLM 入力は正規化してから
            h_base = F.normalize(times_embeds, dim=-1)
            h_film = self.film(h_base, ctx)

            # 学習可能ゲート（init 0.0）を __init__ に用意
            # self.film_gate = nn.Parameter(torch.tensor(0.0))

            gate = torch.sigmoid(self.film_gate)  # 0→1
            times_embeds = h_base + gate * (h_film - h_base)



            with torch.no_grad():
                h_after = F.normalize(times_embeds.detach().to(torch.float32), dim=-1)
                cos_mean = (h_before * h_after).sum(dim=-1).mean().item()
            print(f"[FiLM] cos(before,after)≈{cos_mean:.3f}")

        else:
            # 従来どおり：正規化（＋必要なら加算ミックス）
            if self.mix:
                times_embeds = _l2norm(times_embeds, -1) + self.add_scale * _l2norm(x_mark_enc, -1)
            else:
                times_embeds = _l2norm(times_embeds, -1)


        # # 構造VE（x_mark_enc）と加算（TEと同じ正規化＋learnable scale）
        # if self.mix:
        #     x_mark_enc = x_mark_enc.to(device=times_embeds.device, dtype=enc_param.dtype)
        #     times_embeds = times_embeds / (times_embeds.norm(dim=2, keepdim=True) + 1e-6)
        #     x_mark_enc   = x_mark_enc   / (x_mark_enc.norm(dim=2, keepdim=True) + 1e-6)
        #     # learnable scale は FP32 のまま
        #     times_embeds = times_embeds + self.add_scale * x_mark_enc

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
