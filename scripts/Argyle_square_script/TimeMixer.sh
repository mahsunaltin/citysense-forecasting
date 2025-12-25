#!/bin/bash

model_name=TimeMixer

# TimeMixer requires seq_len divisible by down_sampling_window^down_sampling_layers (4^3=64)
# 192 satisfies this constraint (192/64=3) and keeps the multi-scale tensors aligned.
seq_len=192
label_len=0
e_layers=3
down_sampling_layers=3
down_sampling_window=4
d_model=32
d_ff=64
learning_rate=0.001
train_epochs=10
patience=10
batch_size=64
enc_in=8

for pred_len in 6 24 48; do
  echo "Running $model_name with pred_len=${pred_len}"
  python -u run.py \
    --task_name forecast \
    --is_training 1 \
    --root_path ./dataset/argyle-square/ \
    --data_path hourly_argyle_square.csv \
    --model_id argyle-square_${seq_len}_${pred_len} \
    --model $model_name \
    --data melbourne \
    --features M \
    --target 35 \
    --freq h \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers $e_layers \
    --d_layers 1 \
    --factor 3 \
    --enc_in $enc_in \
    --dec_in $enc_in \
    --c_out $enc_in \
    --d_model $d_model \
    --d_ff $d_ff \
    --batch_size $batch_size \
    --learning_rate $learning_rate \
    --train_epochs $train_epochs \
    --patience $patience \
    --down_sampling_layers $down_sampling_layers \
    --down_sampling_method avg \
    --down_sampling_window $down_sampling_window \
    --embed timeF \
    --num_workers 4
  echo "Finished pred_len=${pred_len}"
  echo "-----------------------------"
done
