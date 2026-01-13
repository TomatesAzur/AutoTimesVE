import pandas as pd
# 元のCSVファイルを読み込み
df = pd.read_csv('dataset/ETT-small/ETTh1.csv')

# カラムの順序を変更（例：'date' を先頭に）
df = df[['date', 'OT', 'HUFL', 'HULL', 'MUFL', 'LUFL', 'LULL', 'MULL']]

# 新しいCSVとして保存（上書きしてもいいなら 'ETTh1.csv' にしてもOK）
df.to_csv('dataset/ETT-small/ETTh1_reordered.csv', index=False)