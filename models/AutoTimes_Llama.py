# models/AutoTimes_Llama_Inner_FiLM.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM
from layers.mlp import MLP


def _l2norm(x, dim=-1, eps=1e-12):
    return x / (x.norm(dim=dim, keepdim=True).clamp_min(eps))


class CondProjector(nn.Module):
    """
    数値特徴 x:[B,L,C] を D(=LLM hidden) に投影する際、
    中間 H 次元で ctx:[B,D] による FiLM を挟むプロジェクタ。
      - 初期は恒等（ctx_to_gb の最終層ゼロ初期化、eps 小）
      - 出力は L2 正規化して LLM へ渡す
    """
    def __init__(self, d_in: int, d_hid: int, d_out: int,
                 eps: float = 0.08, dropout: float = 0.1, verbose: bool = False):
        super().__init__()
        self.ln_in = nn.LayerNorm(d_in)
        self.fc1   = nn.Linear(d_in, d_hid)     # C -> H
        self.ln_h  = nn.LayerNorm(d_hid)

        # ctx:[B,D] -> (gamma,beta):[B,2H]    ゼロ初期化で恒等スタート
        self.ctx_to_gb = nn.Sequential(
            nn.LayerNorm(d_out),
            nn.Linear(d_out, 512), nn.SiLU(),
            nn.Linear(512, 2 * d_hid)
        )
        nn.init.zeros_(self.ctx_to_gb[-1].weight)
        nn.init.zeros_(self.ctx_to_gb[-1].bias)

        self.eps  = nn.Parameter(torch.tensor(float(eps), dtype=torch.float32))
        self.fc2  = nn.Linear(d_hid, d_out)     # H -> D
        self.skip = nn.Linear(d_in, d_out)      # 入力スキップ（次元合わせ）
        self.drop = nn.Dropout(dropout)
        self.verbose = verbose

        # ログ用（任意）
        self.last_cos = None
        self.last_eps_gamma_mean = None
        self.last_eps_beta_mean  = None

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """
        x:   [B,L,C]  （z-score 後）
        ctx: [B,D]    （static；x_mark_enc の L 平均など）
        """
        B, L, _ = x.shape
        z = self.ln_in(x)
        h = self.fc1(z)               # [B,L,H]
        h = self.ln_h(h)

        # FiLM in H
        gb = self.ctx_to_gb(ctx.to(torch.float32))   # [B,2H]
        gamma, beta = gb.chunk(2, dim=-1)            # [B,H], [B,H]
        h_before = F.normalize(h.detach().to(torch.float32), dim=-1)

        h = (1.0 + self.eps * gamma).unsqueeze(1) * h + self.eps * beta.unsqueeze(1)

        # 出力へ（残差＋正規化）
        y = self.fc2(F.gelu(h))          # [B,L,D]
        y = self.drop(y) + self.skip(x)  # [B,L,D]
        y = F.normalize(y, dim=-1)

        # ログ（任意）
        if self.verbose:
            with torch.no_grad():
                h_after = F.normalize(h.detach().to(torch.float32), dim=-1)
                cos_mean = (h_before * h_after).sum(dim=-1).mean().item()
                self.last_cos = cos_mean
                self.last_eps_gamma_mean = float((self.eps * gamma).abs().mean().item())
                self.last_eps_beta_mean  = float((self.eps * beta ).abs().mean().item())
                print(f"[InnerFiLM] cos(H before, after)≈{cos_mean:.3f} "
                      f"|eps*γ|≈{self.last_eps_gamma_mean:.3f} |eps*β|≈{self.last_eps_beta_mean:.3f}")
        return y


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        # device
        if getattr(configs, "use_multi_gpu", False):
            self.device = f"cuda:{getattr(configs, 'local_rank', 0)}"
        else:
            self.device = f"cuda:{getattr(configs, 'gpu', 0)}"
        print(self.device)

        # LLaMA 読み込み（本体は凍結）
        self.llama = LlamaForCausalLM.from_pretrained(
            configs.llm_ckp_dir,
            device_map=self.device,
            torch_dtype=torch.float16 if getattr(configs, "use_amp", False) else torch.float32,
        )
        for p in self.llama.parameters():
            p.requires_grad = False

        # 次元・ハイパ
        self.d_model = int(getattr(self.llama.config, "hidden_size", 4096))
        self.hid_proj = int(getattr(configs, "mlp_hidden_dim", 512))     # H 次元
        self.n_layers_proj = int(getattr(configs, "mlp_hidden_layers", 2))  # decoder MLP 用
        self.drop = float(getattr(configs, "dropout", 0.1))
        self.act  = str(getattr(configs, "mlp_activation", "gelu"))
        self.verbose = bool(getattr(configs, "verbose", False))
        self.proj_film_eps = float(getattr(configs, "proj_film_eps", 0.08))

        # encoder/decoder は C に依存するので遅延構築
        self.encoder = None
        self.decoder = None
        self._c_dim  = getattr(configs, "c_dim", None)
        if self._c_dim is not None:
            self._lazy_build_tokennets(self._c_dim)

    def _lazy_build_tokennets(self, c_dim: int):
        if (self.encoder is not None) and (self._c_dim == c_dim):
            return
        self._c_dim = c_dim

        # 内側FiLM付き CondProjector（encoder）
        self.encoder = CondProjector(
            d_in=c_dim, d_hid=self.hid_proj, d_out=self.d_model,
            eps=self.proj_film_eps, dropout=self.drop, verbose=self.verbose
        )
        # decoder は既存の MLP（D -> C）
        self.decoder = MLP(self.d_model, c_dim, self.hid_proj, self.n_layers_proj, self.drop, self.act)

        llama_dev = next(self.llama.parameters()).device
        self.encoder.to(device=llama_dev, dtype=torch.float32)  # 安定のため FP32
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
        x_enc:      [B,L,C]
        x_mark_enc: [B,L,D]  ← VE（系列名など）を各トークンにブロードキャスト済みを想定
        出力:       [B,L,C]
        """
        assert x_mark_enc is not None and x_mark_enc.dim() == 3, \
            f"x_mark_enc must be [B,L,D], got {None if x_mark_enc is None else x_mark_enc.shape}"

        # 標準化
        x, means, stdev = self._zscore(x_enc)  # [B,L,C]
        B, L, C = x.shape

        # C に合わせてネット構築
        self._lazy_build_tokennets(C)

        # ctx 構築（static）：x_mark_enc の L 平均 → [B,D]
        ctx = F.normalize(x_mark_enc.mean(dim=1).to(torch.float32), dim=-1)

        # 数値 -> D（プロジェクタ“内”で FiLM）
        enc_param = next(self.encoder.parameters())
        x = x.to(device=enc_param.device, dtype=enc_param.dtype)
        times_embeds = self.encoder(x, ctx)  # [B,L,D], L2 norm 済み

        # LLaMA 本体へ
        llama_param  = next(self.llama.parameters())
        times_embeds = times_embeds.to(device=llama_param.device, dtype=llama_param.dtype)
        hidden = self.llama.model(inputs_embeds=times_embeds)[0]  # [B,L,D]

        # D -> C
        dec_param = next(self.decoder.parameters())
        hidden = hidden.to(device=dec_param.device, dtype=dec_param.dtype)
        dec_out = self.decoder(hidden.reshape(B * L, -1)).reshape(B, L, C)
        return dec_out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
