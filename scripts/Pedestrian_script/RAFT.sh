#!/bin/bash

model_name=RAFT
seq_len=192
label_len=0
e_layers=2
down_sampling_layers=3
down_sampling_window=4
d_model=32
d_ff=64
learning_rate=0.001
train_epochs=10
patience=10
batch_size=64
enc_in=54

for pred_len in 6 12 24 48; do
  echo "Running $model_name with pred_len=${pred_len}"
  python -u run.py \
    --task_name forecast \
    --is_training 1 \
    --root_path ./dataset/pedestrian/ \
    --data_path hourly_pedestrian.csv \
    --model_id pedestrian_${model_name}_${pred_len} \
    --model $model_name \
    --data melbourne \
    --features M \
    --target 35 \
    --freq h \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers $e_layers \
    --n_period 2 \
    --enc_in $enc_in \
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
    --use_retrieval 1 \
    --num_workers 4
  echo "Finished pred_len=${pred_len}"
  echo "-----------------------------"
done
