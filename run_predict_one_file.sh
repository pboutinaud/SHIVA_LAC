#!/bin/bash
# Example script to run segmentation prediction for LAC (Lacune)
# Supports new SavedModel (v0) model format.
#
# Adjust MODEL_DIR, IMAGE_DIR and output path to your setup.
# --batch_size: number of slices per batch (increase for GPU, default 1 for CPU)
# --gpu: GPU index to use (-1 for CPU)
#
# @author : Philippe Boutinaud - Fealinx

# ---- T1+FLAIR models (v0, SavedModel, 3 folds) ----
MODEL_DIR=./T1.FLAIR-LAC
IMAGE_DIR=./images

python ./predict_one_file.py \
    --verbose --gpu 0 --batch_size 1 \
    -m $MODEL_DIR/20250613-114524_ResUnet3D-8.9.2-1.5-T1_FLAIR.LAC_1st_fold_0_bestvalloss.tf_inference \
    -m $MODEL_DIR/20250613-114605_ResUnet3D-8.9.2-1.5-T1_FLAIR.LAC_1st_fold_1_bestvalloss.tf_inference \
    -m $MODEL_DIR/20250613-114524_ResUnet3D-8.9.2-1.5-T1_FLAIR.LAC_1st_fold_2_bestvalloss.tf_inference \
    -i $IMAGE_DIR/test_T1_Axial_resampled_111_cropped_intensity_normed.nii.gz \
    -i $IMAGE_DIR/test_FLAIR_Axial_resampled_111_cropped_intensity_normed.nii.gz \
    -o ./predicted/test_lac.nii.gz
