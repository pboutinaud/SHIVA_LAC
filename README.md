# Lacune (LAC) segmentation with a 3D Unet

This repository contains the trained tensorflow models for the 3D Segmentation of Lacunes (LAC) from multi-modal T1-Weighted + FLAIR MR Images with a 3D U-Shaped Neural Network (U-net).

![Gif Image](https://github.com/pboutinaud/SHIVA_PVS/blob/main/docs/Images/SHIVA_BrainTools_small2.gif)

## IP, Licencing & Usage

**The inferences created by these models should not be used for clinical purposes.**

The segmentation models in this repository are provided under the Creative Common Licence [BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## The segmentation models

For multi-modal models trained with T1 + FLAIR images, the models were trained with FLAIR images coregistered to the T1 and added as a second channel: 160 × 214 × 176 x 2 voxels.

The segmentation can be computed as the average of the inference of several models (depending on the number of folds used in the training for a particular model). The resulting segmentation is an image with voxels values in [0, 1] (proxy for the probability of detection of LAC) that must be thresholded to get the actual segmentation. A threshold of 0.5 has been used successfully but that depends on the preferred balance between precision and sensitivity.

To access the models :
* **v0/T1+FLAIR-LAC**: Production models (T1 + FLAIR) based on the ResUnet3D architecture, trained with Keras 3 / TensorFlow ≥ 2.17. Models are stored in TensorFlow SavedModel format (3 folds).
    * Download: [cloud.efixia.com](https://cloud.efixia.com/sharing/d2jPsn1Ev)
    * SHA256 checksum : ADB14247FE2A3CA87EF7B44933FF25151363E98E5EB0305E94756AB7F5458DCB
    * JSON file for SHiVAi pipeline: [model_info_t1-flair-lac-v0.json](model_info_t1-flair-lac-v0.json)

## Requirements

### For new models (v0, SavedModel format)
The models require TensorFlow ≥ 2.17 and were tested with Python 3.12 and TensorFlow 2.20. They are stored in the TensorFlow SavedModel format. A NVIDIA GPU with at least 9 GB of VRAM is recommended for inference (CPU inference is also supported but slower).

### Python dependencies
To run the `predict_one_file.py` script, you will need a python environment with the following libraries:
- tensorflow >= 2.17
- numpy
- nibabel

If you don't know anything about python environment and libraries, you can find some documentation and installers on the [Anaconda website](https://docs.anaconda.com/). We recommend using the lightweight [Miniconda](https://docs.anaconda.com/miniconda/).

## Usage
**These models can be used with the [SHiVAi](https://github.com/pboutinaud/SHiVAi) preprocessing and deep learning segmentation workflow.**

### Step-by-step process to run the model without SHiVAi
1. Download the `predict_one_file.py` from the repository (click the "<> Code" button on the GitHub interface and download the zip file, or clone the repository)
2. Download and unzip the trained models (see [above](#the-segmentation-models))
3. Preprocess the input data (T1 and FLAIR images) to the proper x-y-z volume (160 × 214 × 176). If the resolution is close to 1mm isotropic voxels, a simple cropping is enough. Otherwise, you will have to resample the images to 1mm isotropic voxels.
4. Run the `predict_one_file.py` script as described below

To run `predict_one_file.py` in your python environment you can check the help with the command `python predict_one_file.py -h` (replace "predict_one_file.py" with the full path to the script if it is not in the working directory).

Here is an example of usage of the script with the new SavedModel models:
- The `predict_one_file.py` script stored in `/myhome/my_scripts/`
- Preprocessed Nifti images (volume shape must be 160 × 214 × 176 and voxel values between 0 and 1) stored (for the example) in the folder `/myhome/mydata/`
- The LAC AI models stored (for the example) in `/myhome/lac_models/v0`
- The output folder (for the example) `/myhome/my_results` needs to exist at launch

```bash
# T1+FLAIR SavedModel models (v0)
python /myhome/my_scripts/predict_one_file.py \
    -i /myhome/mydata/t1_image.nii.gz \
    -i /myhome/mydata/flair_image.nii.gz \
    -b /myhome/mydata/input_brainmask.nii.gz \
    -o /myhome/my_results/lac_segmentation.nii.gz \
    --batch_size 1 --gpu 0 \
    -m /myhome/lac_models/v0/20250613-114524_ResUnet3D-8.9.2-1.5-T1_FLAIR.LAC_1st_fold_0_bestvalloss.tf_inference \
    -m /myhome/lac_models/v0/20250613-114605_ResUnet3D-8.9.2-1.5-T1_FLAIR.LAC_1st_fold_1_bestvalloss.tf_inference \
    -m /myhome/lac_models/v0/20250613-114524_ResUnet3D-8.9.2-1.5-T1_FLAIR.LAC_1st_fold_2_bestvalloss.tf_inference
```
>Note that the brain mask input here with `-b /myhome/mydata/input_brainmask.nii.gz` is optional

### Building your own script
The provided python script `predict_one_file.py` can be used as is for running the model or can be used as an example to build your own script.

Here is the main part of the script for SavedModel models, assuming that the images are in a numpy array with the correct shape (*nb of images*, 160, 214, 176, *number of modality to use for this model*):
```python
import tensorflow as tf
import numpy as np

# Load models & predict
predictions = []
for model_dir in model_dirs:  # model_dirs is the list of SavedModel directory paths
    model = tf.saved_model.load(model_dir)
    batch = tf.constant(images, dtype=tf.float32)
    prediction = model.serve(batch).numpy()
    predictions.append(prediction)

# Average all predictions
predictions = np.mean(predictions, axis=0)
```

## Acknowledgements
This work has been done in collaboration between the [Fealinx](http://www.fealinx-biomedical.com/en/) company and the [GIN](https://www.gin.cnrs.fr/en/) laboratory (Groupe d'Imagerie Neurofonctionelle, UMR5293, IMN, Univ. Bordeaux, CEA , CNRS) with grants from the Agence Nationale de la Recherche (ANR) with the projects [GinesisLab](http://www.ginesislab.fr/) (ANR 16-LCV2-0006-01) and [SHIVA](https://rhu-shiva.com/en/) (ANR-18-RHUS-0002)

|<img src="./docs/logos/shiva_blue.png" width="100" height="100" />|<img src="./docs/logos/fealinx.jpg" height="200" />|<img src="./docs/logos/Logo-Gin.png" height="200" />|<img src="./docs/logos/logo_ginesis-1.jpeg" height="100" />|<img src="./docs/logos/logo_anr.png" height="50" />|
|---|---|---|---|---|

## Publication
forthcoming.
