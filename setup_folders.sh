#!/usr/bin/env bash

SCRATCH_DIR=$1

cd $SCRATCH_DIR

mkdir imagenet
mkdir imagenet/data
mkdir imagenet/data/train
mkdir imagenet/data/validation
mkdir imagenet/out
mkdir imagenet/out/validation
mkdir imagenet/out/validation/adversarial
mkdir imagenet/out/validation/preprocessed
mkdir imagenet/out/validation/eval
mkdir imagenet/out/validation/l2norm
mkdir imagenet/out/validation/top_k
mkdir imagenet/training
