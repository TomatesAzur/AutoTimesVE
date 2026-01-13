
import argparse, os, sys, json
from typing import List
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

try:
    from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer
except Exception as e:
    print("[ERROR] transformers not available. Please install transformers.", e)
    sys.exit(1)


def read_series_names(csv_path: str) -> List[str]:
    df = pd.read_csv(csv_path)
    cols = [c for c in df.columns if c.lower() not in ["date","time","timestamp"]]
    return cols


def make_text(name: str, template: str) -> str:
    parts = name.split("_")
    cat = parts[0] if len(parts) > 0 else name
    idx = parts[1] if len(parts) > 1 else "X"

    if template == "short_anchor":
        # Recommended: end on a unique token (no bracket or period at the end)
        return f"CAT={cat} IDX={idx} SER={name}"
    elif template == "anchor_only":
        return f"SER={name}"
    elif template == "name_only":
        return name
    elif template == "long_sentence":
        # Avoid using this for VE; it's here for comparison only
        return f"This time series is daily unit sales of {name}"
    else:
        # pass-through (user-provided literal)
        return template.replace("{name}", name).replace("{cat}", cat).replace("{idx}", idx)


def build_texts(series_names: List[str], template: str, max_series: int = None) -> List[str]:
    names = series_names[:max_series] if (max_series and max_series > 0) else series_names
    return [make_text(n, template) for n in names]


def pool_ve(texts: List[str], llm_ckp_dir: str, pooling: str, use_transformer: bool):
    """
    pooling: 'eos' | 'eos_embed' | 'mean'
    - eos: last non-pad token hidden state (Transformer forward) if use_transformer=True,
           else last non-pad token's input embedding (no Transformer forward)
    - eos_embed: always last non-pad token's input embedding
    - mean: PAD-masked mean of input embeddings (classic/old, not recommended)
    """
    # load model/tokenizer
    model = LlamaForCausalLM.from_pretrained(
        llm_ckp_dir,
        device_map="auto",
        torch_dtype=(torch.float16 if torch.cuda.is_available() else torch.float32),
    ).eval()
    try:
        tok = LlamaTokenizer.from_pretrained(llm_ckp_dir)
    except Exception:
        tok = AutoTokenizer.from_pretrained(llm_ckp_dir, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    for p in model.parameters():
        p.requires_grad = False

    device = next(model.parameters()).device

    toks = tok(texts, return_tensors="pt", padding=True, truncation=True)
    toks = {k: v.to(device) for k, v in toks.items()}
    attn = toks["attention_mask"]            # [C, L]
    C, L = attn.shape
    last_idx = attn.sum(dim=1) - 1
    last_idx = last_idx.clamp_min(0)

    # capture last token id & token string for debug
    last_token_ids = toks["input_ids"][torch.arange(C, device=device), last_idx]
    last_token_ids_cpu = last_token_ids.detach().cpu().tolist()
    last_token_strs = tok.convert_ids_to_tokens(last_token_ids_cpu)

    if pooling == "eos":
        if use_transformer:
            hidden = model.model(**toks).last_hidden_state       # [C, L, D]
            ve = hidden[torch.arange(C, device=device), last_idx]# [C, D]
        else:
            tok_emb = model.model.embed_tokens(toks["input_ids"])# [C, L, D]
            ve = tok_emb[torch.arange(C, device=device), last_idx]
    elif pooling == "eos_embed":
        tok_emb = model.model.embed_tokens(toks["input_ids"])
        ve = tok_emb[torch.arange(C, device=device), last_idx]
    elif pooling == "mean":
        tok_emb = model.model.embed_tokens(toks["input_ids"])
        ve = (tok_emb * attn.unsqueeze(-1)).sum(dim=1) / attn.sum(dim=1, keepdim=True).clamp_min(1.0)
    else:
        raise ValueError(f"Unknown pooling: {pooling}")

    # Force FP32 normalize -> CPU
    ve = ve.to(torch.float32)
    ve = F.normalize(ve, dim=-1)
    ve_cpu = ve.detach().cpu()

    debug = {
        "last_token_ids": last_token_ids_cpu,
        "last_token_strs": last_token_strs,
        "input_ids": toks["input_ids"].detach().cpu().numpy().tolist(),
        "attention_mask": attn.detach().cpu().numpy().tolist(),
    }
    return ve_cpu, debug


def remove_topk_pc(ve: torch.Tensor, k: int = 1) -> torch.Tensor:
    if k <= 0:
        return ve
    # Expect ve on CPU float32
    assert ve.dtype == torch.float32 and ve.device.type == "cpu"
    X = ve - ve.mean(dim=0, keepdim=True)
    U, S, Vt = torch.linalg.svd(X, full_matrices=False)
    V = Vt[:k, :]                     # [k, D]
    proj = (X @ V.T) @ V              # [C, D]
    Y = X - proj
    Y = F.normalize(Y, dim=-1)
    return Y


def cosine_stats(ve: torch.Tensor):
    # ve: [C, D] CPU float32 L2 normalized rows
    C = ve.shape[0]
    sims = ve @ ve.T
    diag = torch.diag(sims).mean().item()
    off = (sims.sum() - sims.trace()) / (C*C - C)
    return diag, off.item(), sims.numpy()


def save_fig_hist_offdiag(sims: np.ndarray, out_png: str):
    C = sims.shape[0]
    mask = ~np.eye(C, dtype=bool)
    vals = sims[mask].ravel()
    plt.figure()
    plt.hist(vals, bins=50)
    plt.title("Histogram of cosine similarities (off-diagonal)")
    plt.xlabel("cosine similarity")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def save_fig_heatmap(sims: np.ndarray, out_png: str):
    plt.figure()
    plt.imshow(sims, aspect="auto")
    plt.colorbar()
    plt.title("Cosine similarity matrix")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def main():
    ap = argparse.ArgumentParser("VE Diagnostics")
    ap.add_argument("--llm_ckp_dir", type=str, required=True)
    ap.add_argument("--root_path", type=str, required=True)
    ap.add_argument("--data_path", type=str, required=True)
    ap.add_argument("--pooling", type=str, default="eos", choices=["eos","eos_embed","mean"])
    ap.add_argument("--use_transformer", type=str, default="true", choices=["true","false"])
    ap.add_argument("--template", type=str, default="short_anchor",
                    help="short_anchor | anchor_only | name_only | long_sentence | custom literal with {name},{cat},{idx}")
    ap.add_argument("--max_series", type=int, default=0, help="0 means all")
    ap.add_argument("--apply_pc", type=int, default=0, help="remove top-k PCs after pooling, then re-normalize")
    ap.add_argument("--save_pt", type=str, default="")
    ap.add_argument("--out_dir", type=str, default="./ve_diag_out")
    args = ap.parse_args()

    use_tf = (args.use_transformer.lower() == "true")

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.root_path, args.data_path)
    series_names = read_series_names(csv_path)
    texts = build_texts(series_names, args.template, max_series=args.max_series)

    # Save the texts we actually used
    pd.DataFrame({"series_names": series_names[:len(texts)], "text": texts}).to_csv(
        os.path.join(args.out_dir, "texts.csv"), index=False, encoding="utf-8"
    )

    print(f"[INFO] num series = {len(texts)} (from {csv_path})")
    print(f"[INFO] pooling={args.pooling}, use_transformer={use_tf}, template={args.template}")
    print("[INFO] first 3 texts:")
    for t in texts[:3]:
        print("  ", t)

    # Build VE
    ve_cpu, debug = pool_ve(texts, args.llm_ckp_dir, args.pooling, use_tf)

    # Save last-token info for forensic
    last_tok_df = pd.DataFrame({
        "series_name": series_names[:len(texts)],
        "last_token_id": debug["last_token_ids"],
        "last_token_str": debug["last_token_strs"],
    })
    last_tok_df.to_csv(os.path.join(args.out_dir, "last_tokens.csv"), index=False, encoding="utf-8")
    print("[INFO] Saved last token info -> last_tokens.csv")
    unique_last = last_tok_df["last_token_str"].nunique()
    print(f"[CHECK] unique last tokens: {unique_last}/{len(texts)}")

    # Stats before PC removal
    diag0, off0, sims0 = cosine_stats(ve_cpu)
    print(f"[STATS] BEFORE  PC: diag={diag0:.6f}, off-diag={off0:.6f}")
    save_fig_hist_offdiag(sims0, os.path.join(args.out_dir, "hist_offdiag_before.png"))
    save_fig_heatmap(sims0, os.path.join(args.out_dir, "sims_before.png"))

    # Optional PC removal
    if args.apply_pc > 0:
        ve_pc = remove_topk_pc(ve_cpu.to(torch.float32), k=args.apply_pc)
        diag1, off1, sims1 = cosine_stats(ve_pc)
        print(f"[STATS] AFTER   PC({args.apply_pc}): diag={diag1:.6f}, off-diag={off1:.6f}")
        save_fig_hist_offdiag(sims1, os.path.join(args.out_dir, "hist_offdiag_after.png"))
        save_fig_heatmap(sims1, os.path.join(args.out_dir, "sims_after.png"))
        ve_final = ve_pc
    else:
        ve_final = ve_cpu

    # Save VE.pt if requested
    if args.save_pt:
        out_obj = {"ve": ve_final.contiguous(), "series_names": series_names[:len(texts)]}
        # ensure float32
        out_obj["ve"] = out_obj["ve"].to(torch.float32)
        torch.save(out_obj, args.save_pt)
        print(f"[INFO] Saved VE.pt -> {args.save_pt} | shape={tuple(out_obj['ve'].shape)}")
        print(json.dumps({
            "save_pt": args.save_pt,
            "C": int(out_obj["ve"].shape[0]),
            "D": int(out_obj["ve"].shape[1]),
            "diag": float((out_obj["ve"]*out_obj["ve"]).sum(dim=1).mean().item()),
        }, ensure_ascii=False))

    print(f"[DONE] Outputs saved to: {os.path.abspath(args.out_dir)}")

if __name__ == "__main__":
    main()
