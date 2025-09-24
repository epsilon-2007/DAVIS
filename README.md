# DAVIS : Dominant Activations and Variance for Improved Separation
In this paper, we make two additional key observations: (i) Most OOD samples exhibit a more uniform distribution within each channel compared to ID samples, i.e., the within-channel variance is typically higher for ID samples. (ii) The dominant (maximum) values within each channel are generally higher for ID samples than for OOD samples.

## Models

### Models on CIFAR Benchmark
The model used for ResNet-18, ResNet-34 and DenseNet-101 in this project are already provided as checkpoints inside `models/checkpoints/resnet18` , `models/checkpoints/resnet34`, and `models/checkpoints/densenet101`.
### Pre-trained Model on ImageNet Benchmark
We use pre-trained models — ResNet-34, ResNet-50, and MobileNet-v2 — provided by PyTorch. These models are automatically downloaded at the start of the evaluation process. Additionally, all necessary checkpoints required for successful evaluation have been uploaded and are readily available.



## Datasets Preparation

### 1. CIFAR Benchmark Experiment
#### In-distribution dataset
The downloading process will start immediately upon running the training or evaluation module. You can download CIFAR-10 and CIFAR-100 manually using following links:

```
mkdir -p datasets/in/
tar -xvzf cifar-10-python.tar.gz -C datasets/in/
tar -xvzf cifar-100-python.tar.gz -C datasets/in/
```
Download and extract the following datasets into `datasets/in/`
#### Out-of-distribution dataset
Similar to [DICE](https://github.com/deeplearning-wisc/dice ) following links can be used to download each dataset:

* [SVHN](http://ufldl.stanford.edu/housenumbers/test_32x32.mat): download it and place it in the folder of `datasets/ood_datasets/svhn`. Then run `python select_svhn_data.py` to generate test subset.
* [Textures](https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz): download it and place it in the folder of `datasets/ood_datasets/dtd`.
* [Places365](http://data.csail.mit.edu/places/places365/test_256.tar): download it and place it in the folder of `datasets/ood_datasets/places365/test_subset`. We randomly sample 10,000 images from the original test dataset. 
* [LSUN-C](https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz): download it and place it in the folder of `datasets/ood_datasets/LSUN`.
* [LSUN-R](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz): download it and place it in the folder of `datasets/ood_datasets/LSUN_resize`.
* [iSUN](https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz): download it and place it in the folder of `datasets/ood_datasets/iSUN`.

For example, run the following commands in the **root** directory to download **LSUN-C**:
```
cd datasets/ood_datasets
wget https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz
tar -xvzf LSUN.tar.gz
```
Once all the out-distribution datasets are downloaded, places them inside `datasets/ood`. 

### 2. ImageNet Benchmark Experiment

#### In-distribution dataset
Please download [ImageNet-1k](http://www.image-net.org/challenges/LSVRC/2012/index) and place the validation data inside `datasets/in-imagenet`. We only need the validation set to test DAVIS and existing approaches. 
#### Out-of-distribution dataset
The curated 4 OOD datasets from  [iNaturalist](https://arxiv.org/pdf/1707.06642.pdf),  [SUN](https://vision.princeton.edu/projects/2010/SUN/paper.pdf),  [Places](http://places2.csail.mit.edu/PAMI_places.pdf),  and [Textures](https://arxiv.org/pdf/1311.3618.pdf), and de-duplicated concepts overlapped with ImageNet-1k by [ReAct](https://github.com/deeplearning-wisc/react)

For Textures, we use the entire dataset, which can be downloaded from their [original website](https://www.robots.ox.ac.uk/~vgg/data/dtd/). For iNaturalist, SUN, and Places, we have sampled 10,000 images from the selected concepts for each dataset, which can be download via the following links:

```bash
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/iNaturalist.tar.gz
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/SUN.tar.gz
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/Places.tar.gz
```
Places all the dataset into `datasets/ood-imagenet/`.

Overall the dataset directory should look like this:
```
datasets/
├── in/
│   ├── cifar-10-batches-py/
│   ├── cifar-100-python/
│   └── cifar-100-python.tar.gz
├── in-imagenet/
│   └── val/
├── ood/
│   ├── dtd/
│   ├── iSUN/
│   ├── LSUN/
│   ├── LSUN_resize/
│   ├── places365/
│   └── SVHN/
└── ood-imagenet/
    ├── imagenet_dtd/
    ├── iNaturalist/
    ├── Places/
    └── SUN/
```




## Evaluation
### CIFAR Benchmark
To evaluate OOD detection on CIFAR-10, run the following script:
``` sh ./scripts/eval.sh ```
This script internally executes, the model and evaluation methods can be modified as needed.
```bash
python3 eval_ood.py \
    --p 85 \
    --pool avg \
    --score energy \
    --model densenet101 \
    --embedding_dim 342 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-10 \
    --ood_loc datasets/ood/ \
    --ood_eval_method DomAct
``` 
### ImageNet Benchmark
To evaluate OOD detection on ImageNet, run: `sh ./scripts/eval_imagenet.sh` which internally executes:  
```bash
python3 eval_ood.py \
    --ash_p 90 \
    --pool avg \
    --score energy \
    --batch-size 64 \
    --model resnet_imagenet50 \
    --embedding_dim 2048 \
    --id_loc datasets/in-imagenet/val \
    --in-dataset ImageNet-1K \
    --ood_loc datasets/ood-imagenet/ \
    --ood_eval_method DomAct
```
Model and evaluation methods can be updated accordingly. `DomAct` represents our methods that can be combined with existing techniques as outline in paper. Details of the hyper-parameter is detailed in the paper. In this repo, `experiment/` has details of all the experiment we run for OOD detection.
