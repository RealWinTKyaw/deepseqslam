#!/bin/sh

SEQ_LENGHT=4

BATCH_SIZE=32

EPOCHS=100

NGPUS=1

SEQ1='day_left'

SEQ2='day_right'

SEQ3='night_right'

CNN='vgg16'

SEQ_MODEL='lstm'

LR='0.01'

MODEL_NAME="gp_${CNN}_${SEQ_MODEL}"

python run.py train \
    --model_name $MODEL_NAME \
    --ngpus $NGPUS \
    --batch_size $BATCH_SIZE \
    --seq_len $SEQ_LENGHT \
    --epochs $EPOCHS \
    --val_set $SEQ2 \
    --cnn_arch $CNN \
    --sequence_model $SEQ_MODEL \
    --learning_rate $LR

for i in $SEQ1 $SEQ2 $SEQ3
do
    python run.py val \
    --model_name $MODEL_NAME \
    --ngpus $NGPUS \
    --batch_size $BATCH_SIZE \
    --seq_len $SEQ_LENGHT \
    --val_set $i \
    --cnn_arch $CNN \
    --sequence_model $SEQ_MODEL
done
