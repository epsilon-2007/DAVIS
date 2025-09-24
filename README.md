# DAVIS: Out-of-distribution Detection via Dominant Activation and Variance for Increased Separation

The DAVIS (Dominant Activation and Variance for Increased Separation) method is based on two simple yet powerful observations about how neural networks process in-distribution (ID) vs. out-of-distribution (OOD) data at the feature level:
 - Higher Maximums for ID Samples: The peak (maximum) activation value within a feature map is consistently higher for in-distribution images than for out-of-distribution ones.
 - Higher Variance for ID Samples: Activations for ID images are often more "spiky" and concentrated, resulting in higher channel-wise variance. OOD images tend to produce flatter, more uniform activations with lower variance.

DAVIS leverages these two statistical signals—maximum and variance—to create a feature representation that dramatically improves OOD detection.

## Models

### Models on CIFAR Benchmark
The model used for ResNet-18, ResNet-34, DenseNet-101 and MobileNet-v2 in this project are already provided as checkpoints inside `experiments/checkpoints/resnet18` , `experiments/checkpoints/resnet34`, `experiments/checkpoints/densenet101` and `experiments/checkpoints/mobilenetv2`.
### Pre-trained Model on ImageNet Benchmark
We use pre-trained models — DenseNet-121, ResNet-50, MobileNet-v2, EfficientNet-b0 — provided by PyTorch. These models are automatically downloaded at the start of the evaluation process, when the parameter `pre-trained` is set to `False`.



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
Before running the evaluation make sure to run following scripts for respective models and dataset pair. These scripts are inside `scripts/statistics.sh` and `scripts/precompute.sh`
```bash
python3 Statistics.py \
 --batch-size 256 \
 --in-dataset ImageNet-1K \
 --id_loc datasets/in-imagenet/val \
 --ood_loc datasets/ood-imagenet/ \
 --model densenet121_imagenet \
```
```bash
python3 precompute.py \
 --in-dataset ImageNet-1K  \
 --model densenet121_imagenet \
 --id_loc datasets/in-imagenet/val \
```
To evaluate OOD detection run the following script:
``` sh ./scripts/eval.sh ```
This script internally executes, the model and evaluation methods can be modified as needed.
```bash
python3 eval_ood.py \
 --score energy \
 --batch-size 64 \
 --model densenet121_imagenet \
 --id_loc datasets/in-imagenet/val \
 --in-dataset ImageNet-1K \
 --ood_loc datasets/ood-imagenet/ \
 --ood_eval_type <eval_type> \
 --ash_p 90 \
 --std 0.5 \
 --ood_eval_method <eval_method>
``` 
			

## Command Line Arguments

| Argument           | Options                   | Description                                                                 |
|--------------------|---------------------------|-----------------------------------------------------------------------------|
| `--eval_type`  | `avg`, `max`, `avg+std`  | Specifies the feature type. `avg` is the baseline (GAP). `max` and `avg+std` are our DAVIS variants. |
| `--eval_method`| `baseline/energy`,`SCALE`, `ReAct`, `ASH`, `DICE` | The post-hoc baseline method to apply on top of the features from `--ood_eval_type`. |
| `--std`            | *(float)*                | The hyperparameter γ for our `avg+std` method. Only used when `--eval_type` is `avg+std`. |
| `--ash_p`          | *(int)*                  | The percentile hyperparameter for the ASH baseline. Only used when `--eval_method` is `ash`. |


All experiments details are inside `scripts\<model_name>`