#!/bin/bash

# ==== データセット定義 ====
datasets=(
    "date_FOODS_1.csv"
    "date_FOODS_2.csv"
    "date_FOODS_3.csv"
    "date_HOBBIES_1.csv"
    "date_HOBBIES_2.csv"
    "date_HOUSEHOLD_1.csv"
    "date_HOUSEHOLD_2.csv"
)

# ==== VE Path ====
ve_path="./dataset/M5/CA_1_VE.pt"

# ==== 共通引数 ====
common_args="--task_name long_term_forecast \
    --is_training 1 \
    --model AutoTimes_Llama \
    --data var_tokens \
    --root_path ./dataset/M5 \
    --seq_len 28 \
    --label_len 21 \
    --token_len 7 \
    --test_seq_len 28 \
    --test_label_len 21 \
    --test_pred_len 7 \
    --c_dim 1 \
    --mlp_hidden_dim 512 \
    --mlp_hidden_layers 2 \
    --dropout 0.1 \
    --mlp_activation gelu \
    --batch_size 16 \
    --learning_rate 5e-4 \
    --train_epochs 10 \
    --use_amp \
    --drop_last \
    --proj_film_eps 0.05 \
    --gpu 0"

# ==== ループして実行 ====
for d in "${datasets[@]}"; do
    model_id="AutoTimes_FiLM_${d%.csv}"  # 拡張子除去

    echo "=============================="
    echo " Training on: $d"
    echo " Model ID   : $model_id"
    echo "=============================="

    python -u run.py \
        $common_args \
        --data_path "$d" \
        --ve_pt_path "$ve_path" \
        --model_id "$model_id"

    echo "✔ Finished: $d"
    echo ""
done

echo "Finished all datasets."
