from data_provider.data_loader import VarTokensDataset, collate_var_tokens
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
import os
from functools import partial

data_dict = {
    'var_tokens': VarTokensDataset,
}

def data_provider(args, flag):
    """
    公式実装と同様に:
      - x_mark を DataLoader 側で作って [B,L,D] をモデルへ渡す
      - TE相当の x_mark は「構造VE（1文）ve_struct」か「系列VE表 ve」のどちらでもOK
         * ve_struct: {"ve_struct":[D], "series_names":[...]} の .pt
         * ve       : {"ve":[C,D], "series_names":[...]} の .pt
    """
    Data = data_dict[args.data]

    # === バッチ設定 ===
    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        seq_len, label_len, pred_len = args.test_seq_len, args.test_label_len, args.test_pred_len
    elif flag == 'val':
        shuffle_flag = args.val_set_shuffle
        drop_last = False
        batch_size = args.batch_size
        seq_len, label_len, pred_len = args.seq_len, args.label_len, args.token_len  # valiは未来=token_len
    else:  # train
        shuffle_flag = True
        drop_last = args.drop_last
        batch_size = args.batch_size
        seq_len, label_len, pred_len = args.seq_len, args.label_len, args.token_len  # trainも未来=token_len

    # === var_tokens モード（今回の例） ===
    if args.data == 'var_tokens':
        csv_path = os.path.join(args.root_path, args.data_path)

        # Dataset は数値 [L,C] のみ返す。x_mark([B,L,D]) は collate で作る。
        data_set = Data(
            csv_path=csv_path,
            seq_len=seq_len, label_len=label_len, pred_len=pred_len,
            stride=getattr(args, 'stride', 1),
        )

        if (args.use_multi_gpu and args.local_rank == 0) or (not args.use_multi_gpu):
            print(flag, len(data_set))

        # --- VE .pt をここでは開かず、collate に渡す形に統一するなら、
        #     data_loader 側（collate_var_tokens内部）で torch.load してもよいが、
        #     ここでは .pt を読み、どちらのキーかを判別して渡す実装にしておく ---
        ve_obj = torch.load(args.ve_pt_path)
        ve_struct, ve_table, ve_names = None, None, None
        if isinstance(ve_obj, dict) and ("ve_struct" in ve_obj):
            ve_struct = ve_obj["ve_struct"].float()     # [D]
            ve_names  = ve_obj.get("series_names", None)
        elif isinstance(ve_obj, dict) and ("ve" in ve_obj):
            ve_table = ve_obj["ve"].float()             # [C,D]
            ve_names = ve_obj.get("series_names", None)
        else:
            t = ve_obj.float()
            if t.dim() == 1:  # [D]
                ve_struct = t
            else:             # [C,D]
                ve_table = t

        # 系列VE表を使う場合のみ、CSV列順に並べ替え（構造VEは1ベクトルなので不要）
        if (ve_table is not None) and (ve_names is not None) and (list(ve_names) != list(data_set.series_names)):
            name2idx = {n: i for i, n in enumerate(ve_names)}
            order = [name2idx[n] for n in data_set.series_names]
            ve_table = ve_table[torch.tensor(order, dtype=torch.long)]

        # collate: 構造VEを優先。なければ系列VE表を使用。
        if ve_struct is not None:
            collate = partial(
                collate_var_tokens,
                ve_struct=ve_struct,    # [D] を各トークンへブロードキャスト（TEと同じ）
                tau=getattr(args, 've_tau', 1.0),
            )
        else:
            collate = partial(
                collate_var_tokens,
                ve_table=ve_table,      # [C,D] から重み付き合成で [B,L,D] を作成
                tau=getattr(args, 've_tau', 1.0),
            )

        if args.use_multi_gpu:
            sampler = DistributedSampler(data_set, shuffle=shuffle_flag)
            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=args.num_workers,
                persistent_workers=True,
                pin_memory=True,
                drop_last=drop_last,
                collate_fn=collate,
            )
        else:
            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last,
                collate_fn=collate,
            )
        return data_set, data_loader

    # === それ以外（後方互換）：従来ルート ===
    if flag in ['train', 'val']:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[seq_len, label_len, pred_len],
            seasonal_patterns=args.seasonal_patterns,
            drop_short=args.drop_short,
        )
    else:  # test
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[seq_len, label_len, pred_len],
            seasonal_patterns=args.seasonal_patterns,
            drop_short=args.drop_short,
        )

    if (args.use_multi_gpu and args.local_rank == 0) or (not args.use_multi_gpu):
        print(flag, len(data_set))

    if args.use_multi_gpu:
        train_datasampler = DistributedSampler(data_set, shuffle=shuffle_flag)
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            sampler=train_datasampler,
            num_workers=args.num_workers,
            persistent_workers=True,
            pin_memory=True,
            drop_last=drop_last,
        )
    else:
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last
        )
    return data_set, data_loader
