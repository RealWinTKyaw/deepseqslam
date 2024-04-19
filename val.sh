#!/bin/sh

SEQ_LENGHT=10

BATCH_SIZE=8

NGPUS=1

SEQ1='day_left'

SEQ2='day_right'

SEQ3='night_right'

CNN='squeezenet1_0'

SEQ_MODEL='lstm'

MODEL_NAME="gp_${CNN}_${SEQ_MODEL}"

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
