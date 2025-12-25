#!/bin/bash

model_name=RAFT
seq_len=192
e_layers=2
down_sampling_layers=3
down_sampling_window=4
learning_rate=0.001
d_model=32
d_ff=64
train_epochs=10
patience=10
batch_size=64
label_len=0
enc_in=8

for pred_len in 6 24 48; do
  echo "Running $model_name with pred_len=${pred_len}"

  python -u run.py \
    --task_name forecast \
    --is_training 1 \
    --root_path ./dataset/argyle-square/ \
    --data_path hourly_argyle_square.csv \
    --model_id argyle-square_${model_name}_${pred_len} \
    --model $model_name \
    --data melbourne \
    --features M \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers $e_layers \
    --n_period 2 \
    --enc_in $enc_in \
    --c_out $enc_in \
    --des 'Exp' \
    --itr 1 \
    --d_model $d_model \
    --d_ff $d_ff \
    --learning_rate $learning_rate \
    --train_epochs $train_epochs \
    --patience $patience \
    --batch_size $batch_size \
    --down_sampling_layers $down_sampling_layers \
    --down_sampling_method avg \
    --down_sampling_window $down_sampling_window \
    --use_retrieval 1
  echo "Finished pred_len=${pred_len}"
  echo "-----------------------------"
done
