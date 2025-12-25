#!/bin/bash

model_name=WPMixer
seq_len=192
label_len=0
enc_in=54
d_model=128
learning_rate=0.001
patch_len=16
batch_size=64
train_epochs=10
patience=12
dropout=0.3
lradj=type3

for pred_len in 6 12 24 48; do
  echo "Running $model_name with pred_len=${pred_len}"
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/pedestrian/ \
    --data_path hourly_pedestrian.csv \
    --model_id wpmixer_pedestrian_${pred_len} \
    --model $model_name \
    --task_name forecast \
    --data melbourne \
    --features M \
    --target 35 \
    --freq h \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --label_len $label_len \
    --enc_in $enc_in \
    --c_out $enc_in \
    --d_model $d_model \
    --patch_len $patch_len \
    --batch_size $batch_size \
    --learning_rate $learning_rate \
    --lradj $lradj \
    --dropout $dropout \
    --train_epochs $train_epochs \
    --patience $patience \
    --use_amp \
    --num_workers 4
  echo "Finished pred_len=${pred_len}"
  echo "-----------------------------"
done
