#!/bin/bash

# ==== 実験設定 ====
seeds=(42 100 2025 777 1234)

# ==== 共通実行設定 ====
COMMON_ARGS="\
    --task_name long_term_forecast \
    --is_training 1 \
    --model AutoTimes_Llama \
    --model_id CA1_28_7_AutoTimes_Llama_FiLM \
    --data var_tokens \
    --root_path ./dataset/M5 \
    --data_path CA_1.csv \
    --ve_pt_path ./dataset/M5/CA_1_VE.pt \
    --seq_len 28 \
    --label_len 21 \
    --token_len 7 \
    --test_seq_len 28 \
    --test_label_len 21 \
    --test_pred_len 7 \
    --c_dim 1 \
    --mlp_hidden_dim 512 \
    --mlp_hidden_layers 0 \
    --dropout 0.1 \
    --mlp_activation gelu \
    --batch_size 16 \
    --learning_rate 5e-4 \
    --train_epochs 10 \
    --use_amp \
    --drop_last \
    --proj_film_eps 0.05 \
    --gpu 0 \
"

# ==== ループ実行 ====
for seed in "${seeds[@]}"; do
    echo "=============================="
    echo " Running experiment with SEED = $seed"
    echo "=============================="

    python -u run.py \
        $COMMON_ARGS \
        --seed "$seed" \
        --model_id "CA1_FiLM_seed${seed}"

    echo "Finished SEED $seed"
    echo
done

echo "ALL SEED RUNS COMPLETED"
