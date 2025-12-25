#!/bin/bash

model_name=WPMixer
label_len=0
enc_in=36
d_model=128
learning_rate=0.001
patch_len=16
batch_size=64
train_epochs=10
patience=12
dropout=0.3
lradj=type3

# First set: seq_len=128, pred_len=[12, 24]
seq_len=128
for pred_len in 12 24; do
  echo "Running $model_name with seq_len=${seq_len}, pred_len=${pred_len}"
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/microclimate-sensors/ \
    --data_path minutes_microclimate_sensors.csv \
    --model_id wpmixer_microclimate-sensors_${seq_len}_${pred_len} \
    --model $model_name \
    --task_name forecast \
    --data melbourne \
    --features M \
    --target 35 \
    --freq t \
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
  echo "Finished seq_len=${seq_len}, pred_len=${pred_len}"
  echo "-----------------------------"
done

# Second set: seq_len=640, pred_len=96
seq_len=640
pred_len=96
echo "Running $model_name with seq_len=${seq_len}, pred_len=${pred_len}"
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/microclimate-sensors/ \
  --data_path minutes_microclimate_sensors.csv \
  --model_id wpmixer_microclimate-sensors_${seq_len}_${pred_len} \
  --model $model_name \
  --task_name forecast \
  --data melbourne \
  --features M \
  --target 35 \
  --freq t \
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
echo "Finished seq_len=${seq_len}, pred_len=${pred_len}"
echo "-----------------------------"
