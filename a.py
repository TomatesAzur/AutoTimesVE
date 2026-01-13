import pandas as pd
from pathlib import Path

# 元CSVのパス
root = Path("./dataset/M5")
src_path = root / "CA_1.csv"

df = pd.read_csv(src_path)

# 単変量化したい列名をここに書く
target_cols = ["FOODS_1","FOODS_2","FOODS_3","HOBBIES_1","HOBBIES_2","HOUSEHOLD_1","HOUSEHOLD_2",]

for col in target_cols:
    # date + 対象列だけ残す
    out_df = df[["date", col]].copy()

    # 出力ファイル名： CA_1_FOODS_3.csv みたいな感じ
    out_path = root / f"CA_1_{col}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"saved: {out_path}")
