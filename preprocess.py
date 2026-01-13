import argparse, os, torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.Preprocess_Llama import Model
from data_provider.data_loader import Dataset_Preprocess
import torch.nn.functional as F

def main():
    parser = argparse.ArgumentParser(description='AutoTimes Preprocess (VE only, M5 CA_1)')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--llm_ckp_dir', type=str, default='./llama')
    parser.add_argument('--root_path', type=str, default='./dataset/m5/')
    parser.add_argument('--data_path', type=str, default='CA_1.csv')
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()

    print(f"VE preprocess for: {os.path.join(args.root_path, args.data_path)}")

    torch.set_grad_enabled(False)
    model = Model(args).eval()   # LLaMA で系列名→埋め込み

    # “系列名の収集” 用：系列数ぶんを回すだけなのでサイズはダミーでOK
    seq_len, label_len, pred_len = 1, 0, 0
    data_set = Dataset_Preprocess(
        root_path=args.root_path, flag='train',
        size=[seq_len, label_len, pred_len],
        data_path=args.data_path
    )
    loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, collate_fn=list)

    outputs = []
    for _, texts in tqdm(enumerate(loader), total=len(loader)):
        ve = model(texts)                 # [c_batch, D]
        outputs.append(ve.detach().cpu())

    ve_all = torch.cat(outputs, dim=0)    # [C, D]
    ve_all = F.normalize(ve_all, dim=-1)

    out_obj = {"ve": ve_all, "series_names": data_set.series_names}

    base = os.path.splitext(args.data_path)[0]   # "CA_1"
    out_path = os.path.join(args.root_path, f"{base}_VE.pt")
    torch.save(out_obj, out_path)

    # 軽い健診
    with torch.no_grad():
        sims = (ve_all @ ve_all.T).cpu().numpy()
        off = (sims.sum()-sims.diagonal().sum())/(sims.size - len(sims))
        print("cos diag≈", sims.diagonal().mean(), " off-diag≈", off)
    print("VE shape:", tuple(ve_all.shape))
    print("first 5 names:", out_obj["series_names"][:5])
    print(f"Saved: {out_path}")

if __name__ == '__main__':
    main()
# import argparse, os, torch
# from torch.utils.data import DataLoader
# from transformers import LlamaForCausalLM, AutoTokenizer
# import torch.nn.functional as F
# import pandas as pd

# def main():
#     p = argparse.ArgumentParser(description="Make structure VE (one sentence) like TE")
#     p.add_argument('--gpu', type=int, default=0)
#     p.add_argument('--llm_ckp_dir', type=str, default='./llama')
#     p.add_argument('--root_path', type=str, default='./dataset/m5/')
#     p.add_argument('--data_path', type=str, default='CA_1.csv')
#     args = p.parse_args()

#     csv_path = os.path.join(args.root_path, args.data_path)
#     df = pd.read_csv(csv_path)
#     series_names = [c for c in df.columns if c.lower() not in ['date','time','timestamp']]

#     # 1文で“構造”を指定（順序が重要なのでそのまま列順で列挙）
#     s_list = ", ".join(series_names)
#     text = f"This time series represents daily sales of {s_list}."

#     device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
#     llm = LlamaForCausalLM.from_pretrained(args.llm_ckp_dir,
#                                            device_map=device,
#                                            torch_dtype=torch.float16 if 'cuda' in device else torch.float32)
#     tok = AutoTokenizer.from_pretrained(args.llm_ckp_dir, use_fast=True)

#     with torch.no_grad():
#         enc = tok(text, return_tensors='pt')
#         for k in enc: enc[k] = enc[k].to(device)
#         # “TE風”に、Transformerは通さず embed_tokens の平均で文ベクトル化
#         emb = llm.get_input_embeddings()(enc['input_ids'])  # [1, T, D]
#         # PAD 除外（存在しない場合もあるので安全に処理）
#         if 'attention_mask' in enc:
#             mask = enc['attention_mask'].unsqueeze(-1)      # [1, T, 1]
#             emb = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)   # [1, D]
#         else:
#             emb = emb.mean(dim=1)
#         ve_struct = F.normalize(emb.squeeze(0), dim=-1)     # [D], L2 normalize

#     base = os.path.splitext(args.data_path)[0]              # e.g., "CA_1"
#     out_path = os.path.join(args.root_path, f"{base}_VE.pt")
#     torch.save({"ve_struct": ve_struct.float().cpu(),
#                 "series_names": series_names}, out_path)

#     print("Saved:", out_path, " shape:", tuple(ve_struct.shape))

# if __name__ == '__main__':
#     main()
