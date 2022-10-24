# HEIP

[![Generic badge](https://img.shields.io/badge/License-MIT-<COLOR>.svg?style=for-the-badge)](https://github.com/ValeAri/HEIP_HE-image-analysis-pipeline/blob/main/LICENSE) [![Python - Version](https://img.shields.io/badge/PYTHON-3.7+-red?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

![schema](./images/pipeline_flow.png)

**HEIP (HE-image-analysis-pipeline) is a set of tools to run end-to-end HE-image analysis.**

## Introduction

HEIP is a fully fledged HE-image pipeline that consists of different tools ranging from HE-image patching, pre-processing, segmentation, and downstream feature extraction, segmentation tile merging etc. TODO: More text

## Set Up

1. Clone the repository

```shell
git clone git@github.com:ValeAri/HEIP_HE-image-analysis-pipeline.git
```

2. cd to the repository `cd <path>/HEIP_HE-image-analysis-pipeline/`

3. Create environment (optional but recommended)

```
conda create --name HEIP python
conda activate HEIP
```

or

```
python3 -m venv HEIP
source HEIP/bin/activate
pip install -U pip
```

4. Install dependencies

```
pip install -r requirements.txt
```

## Notebook examples

1. [Patching HE images](https://github.com/ValeAri/HEIP_HE-image-analysis-pipeline/blob/main/examples/1_patching.ipynb).
2. [Pre-process HE images](https://github.com/ValeAri/HEIP_HE-image-analysis-pipeline/blob/main/examples/2_pre-proc.ipynb).
3. [Train a segmentation model for HE images with a training set](https://github.com/ValeAri/HEIP_HE-image-analysis-pipeline/blob/main/examples/3_train_seg_model.ipynb).
4. [Run inference with the segmentation model](https://github.com/ValeAri/HEIP_HE-image-analysis-pipeline/blob/main/examples/4_inference.ipynb).

#### Run instance segmentation training and inference with SLURM from CLI

**NOTE** You might want to modify the batchscripts to your needs (setting up right paths etc).

**Training**

```shell
cd HEIP_HE-image-analysis-pipeline/
sbatch ./batchscripts/train.sh
```

**Inference**

```shell
cd HEIP_HE-image-analysis-pipeline/
sbatch ./batchscripts/infer_wsi.sh
```
