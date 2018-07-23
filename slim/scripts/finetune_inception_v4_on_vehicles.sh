#!/bin/bash
# Usage:
# cd slim
# ./slim/scripts/finetune_inception_v4.sh
set -e

# Where the pre-trained InceptionV4 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=../checkpoints

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=../train

EVAL_DIR=../eval

# Where the dataset is saved to.
DATASET_DIR=../dataset

# Download the pre-trained checkpoint.
if [ ! -d "$PRETRAINED_CHECKPOINT_DIR" ]; then
  mkdir ${PRETRAINED_CHECKPOINT_DIR}
fi
if [ ! -f ${PRETRAINED_CHECKPOINT_DIR}/inception_v4.ckpt ]; then
  wget http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz
  tar -xvf inception_v4_2016_09_09.tar.gz
  mv inception_v4.ckpt ${PRETRAINED_CHECKPOINT_DIR}/inception_v4.ckpt
  rm inception_v4_2016_09_09.tar.gz
fi

# Convert the dataset
python download_and_convert_data.py \
  --dataset_name=vehicles \
  --dataset_dir=${DATASET_DIR}

# Fine-tune only the new layers for 1000 steps.
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=vehicles \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v4 \
  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/inception_v4.ckpt \
  --checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
  --trainable_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
  --max_number_of_steps=1000 \
  --batch_size=32 \
  --learning_rate=0.01 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=120 \
  --save_summaries_secs=120 \
  --log_every_n_steps=100 \
  --optimizer=rmsprop \
  --weight_decay=0.00004

# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${EVAL_DIR} \
  --dataset_name=vehicles \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v4

# Fine-tune all the new layers for 500 steps.
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR}/all \
  --dataset_name=vehicles \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v4 \
  --checkpoint_path=${TRAIN_DIR} \
  --max_number_of_steps=500 \
  --batch_size=32 \
  --learning_rate=0.0001 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=120 \
  --save_summaries_secs=120 \
  --log_every_n_steps=10 \
  --optimizer=rmsprop \
  --weight_decay=0.00004

# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR}/all \
  --eval_dir=${EVAL_DIR}/all \
  --dataset_name=vehicles \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v4
