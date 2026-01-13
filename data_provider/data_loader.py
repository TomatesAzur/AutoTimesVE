import os
import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from utils.tools import convert_tsf_to_dataframe
import warnings

warnings.filterwarnings('ignore')

class Dataset_Preprocess(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 data_path='CA_1.csv', scale=True, seasonal_patterns=None):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.token_len = self.seq_len - self.label_len
        self.token_num = self.seq_len // self.token_len
        self.flag = flag
        self.data_set_type = data_path.split('.')[0]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.tot_len = len(self.series_names)

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        self.series_names = [c for c in df_raw.columns if c.lower() not in ["date", "timestamp", "time"]]

    def __getitem__(self, index):
        name = self.series_names[index % len(self.series_names)]
        # 例1: 種別や役割を含める（カテゴリが分かるなら）
        # "This time series is daily sales for the grocery category FOODS_3 in California."
        # 例2: 関係を示唆（競合・補完などが分かるなら）
        # "FOODS_3 often competes with FOODS_1 and FOODS_2 in promotions."
        return f"This time series represents daily sales of {name}."


    def __len__(self):
        return len(self.series_names)

# ==== Var-tokens loader: 1 token = values of all variables at time t ====
class VarTokensDataset(Dataset):
    """
    CSV: 先頭列 date、以降が変数列（FOODS_1,...,HOUSEHOLD_2）
    返り値: x_enc [L, C], series_names（並び合わせ用）
    """
    def __init__(self, csv_path: str, seq_len: int, label_len: int, pred_len: int,
                 stride: int = None, dtype=np.float32):
        if seq_len != label_len + pred_len:
            print(
                f"[VarTokensDataset][Warning] "
                f"seq_len({seq_len}) != label_len({label_len}) + pred_len({pred_len})"
            )
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.stride = stride if stride is not None else 1

        if not os.path.exists(csv_path):
            raise FileNotFoundError(csv_path)
        df = pd.read_csv(csv_path)

        self.series_names = [c for c in df.columns if c.lower() not in ("date","timestamp","time")]
        vals = df[self.series_names].values.astype(dtype)  # [T, C]
        self.X = torch.from_numpy(vals)                   # [T, C]
        self.T, self.C = self.X.shape

        self.starts = list(range(0, self.T - self.seq_len + 1, self.stride))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]; e = s + self.seq_len
        x_enc = self.X[s:e, :]          # [L, C]
        return x_enc, self.series_names

# ==== x_mark の生成（構造VE or 系列VE） =====================================

def build_x_mark_struct(B: int, L: int, ve_struct: torch.Tensor, device: torch.device):
    """
    構造VE（1文ベクトル）を TE と同様に各トークンへブロードキャストする。
    ve_struct: [D]
    return   : [B, L, D]
    """
    ve_struct = ve_struct.to(device=device, dtype=torch.float32)
    D = int(ve_struct.shape[0])
    x_mark = ve_struct.view(1, 1, D).expand(B, L, D).contiguous()
    # （任意）正規化：前処理側で normalize 済みなら不要
    x_mark = x_mark / (x_mark.norm(dim=2, keepdim=True) + 1e-6)
    return x_mark


def build_x_mark_enc_from_VE_table(x_enc_batch, ve_table, tau=1.0):
    """
    従来（系列VEテーブル）の方式：各時点 t の |x| を softmax(温度 tau) で重み w_t にし、
    VE を重み付き合成して [B, L, D] を得る。
    x_enc_batch: [B, L, C]
    ve_table   : [C, D]
    return     : [B, L, D]
    """
    B, L, C = x_enc_batch.shape
    C_ve, D = ve_table.shape
    assert C == C_ve, f"VEのC({C_ve})とデータのC({C})が一致しません。"

    with torch.no_grad():
        w = torch.softmax((x_enc_batch.abs() / float(tau)), dim=2)   # [B, L, C]
        x_mark_enc = torch.einsum("blc,cd->bld", w, ve_table.to(x_enc_batch.dtype))  # [B, L, D]
        x_mark_enc = x_mark_enc / (x_mark_enc.norm(dim=2, keepdim=True) + 1e-6)
    return x_mark_enc


def collate_var_tokens(
    batch,
    ve_table: torch.Tensor = None,
    ve_struct: torch.Tensor = None,
    tau: float = 1.0
):
    """
    returns:
      batch_x      : [B, L, C]
      batch_y      : [B, L, C]   ← まずは復元タスクとして同じものをターゲットに
      batch_x_mark : [B, L, D]   ← 構造VE or 系列VE合成
      batch_y_mark : [B, L, D]   ← 同じでOK
    """
    x_list = [b[0] for b in batch]
    x_enc = torch.stack(x_list, dim=0)               # [B, L, C]
    device = x_enc.device

    if ve_struct is not None:
        # 構造VE（1文ベクトル）を TE と同様に各トークンへ
        B, L, _ = x_enc.shape
        x_mark_enc = build_x_mark_struct(B, L, ve_struct, device=device)   # [B, L, D]
    else:
        # 従来の系列VEテーブルで重み付き合成
        x_mark_enc = build_x_mark_enc_from_VE_table(x_enc, ve_table.to(device), tau=tau)  # [B, L, D]

    batch_x      = x_enc
    batch_y      = x_enc.clone()
    batch_x_mark = x_mark_enc
    batch_y_mark = x_mark_enc.clone()
    return batch_x, batch_y, batch_x_mark, batch_y_mark


def make_dataloader_var_tokens(csv_path: str,
                               ve_pt_path: str,
                               seq_len: int, label_len: int, pred_len: int,
                               batch_size: int = 16,
                               shuffle: bool = True,
                               stride: int = 1,
                               tau: float = 1.0,
                               num_workers: int = 0):
    """
    VE をロードして DataLoader を返す。
    - 構造VE（TE風）: .pt が {"ve_struct": [D]} を含む → [B, L, D] をブロードキャスト
    - 系列VEテーブル : .pt が {"ve": [C, D]}        → 従来どおり重み付き合成で [B, L, D]
    """
    ve_obj = torch.load(ve_pt_path)
    ve_table, ve_struct, ve_names = None, None, None

    if isinstance(ve_obj, dict) and ("ve_struct" in ve_obj):
        # 構造VE（1文）モード
        ve_struct = ve_obj["ve_struct"].float()      # [D]
        ve_names  = ve_obj.get("series_names", None) # ログ用途（列順の材料）
    elif isinstance(ve_obj, dict) and ("ve" in ve_obj):
        # 従来の系列VEテーブル
        ve_table = ve_obj["ve"].float()              # [C, D]
        ve_names = ve_obj.get("series_names", None)
    else:
        # 生テンソルが来た場合の後方互換
        if ve_obj.dim() == 1:
            ve_struct = ve_obj.float()               # [D]
        else:
            ve_table  = ve_obj.float()               # [C, D]

    ds = VarTokensDataset(csv_path=csv_path,
                          seq_len=seq_len, label_len=label_len, pred_len=pred_len,
                          stride=stride)

    # 系列VEテーブルの場合のみ、列順を合わせる
    if (ve_table is not None) and (ve_names is not None) and (list(ve_names) != list(ds.series_names)):
        name2idx = {n: i for i, n in enumerate(ve_names)}
        order = [name2idx[n] for n in ds.series_names]
        ve_table = ve_table[torch.tensor(order, dtype=torch.long)]

    # collate の切替（構造VE優先）
    if ve_struct is not None:
        collate_fn = lambda b: collate_var_tokens(b, ve_struct=ve_struct, tau=tau)
    else:
        collate_fn = lambda b: collate_var_tokens(b, ve_table=ve_table, tau=tau)

    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        collate_fn=collate_fn
    )
    return loader
