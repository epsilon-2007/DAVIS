python3 Statistics.py \
 --batch-size 1000 \
 --in-dataset CIFAR-10 \
 --id_loc datasets/in/ \
 --ood_loc datasets/ood/ \
 --model resnet18 \

python3 Statistics.py \
 --batch-size 1000 \
 --in-dataset CIFAR-100 \
 --id_loc datasets/in/ \
 --ood_loc datasets/ood/ \
 --model resnet18 \

python3 Statistics.py \
 --batch-size 1000 \
 --in-dataset CIFAR-10 \
 --id_loc datasets/in/ \
 --ood_loc datasets/ood/ \
 --model resnet34 \

python3 Statistics.py \
 --batch-size 1000 \
 --in-dataset CIFAR-100 \
 --id_loc datasets/in/ \
 --ood_loc datasets/ood/ \
 --model resnet34 \

python3 Statistics.py \
 --batch-size 1000 \
 --in-dataset CIFAR-10 \
 --id_loc datasets/in/ \
 --ood_loc datasets/ood/ \
 --model densenet101 \

python3 Statistics.py \
 --batch-size 1000 \
 --in-dataset CIFAR-100 \
 --id_loc datasets/in/ \
 --ood_loc datasets/ood/ \
 --model densenet101 \

python3 Statistics.py \
 --batch-size 1000 \
 --in-dataset CIFAR-10 \
 --id_loc datasets/in/ \
 --ood_loc datasets/ood/ \
 --model mobilenetv2 \

python3 Statistics.py \
 --batch-size 1000 \
 --in-dataset CIFAR-100 \
 --id_loc datasets/in/ \
 --ood_loc datasets/ood/ \
 --model mobilenetv2 \

# python3 Statistics.py \
#  --batch-size 256 \
#  --in-dataset ImageNet-1K \
#  --id_loc datasets/in-imagenet/val \
#  --ood_loc datasets/ood-imagenet/ \
#  --model mobilenetv2_imagenet \

# python3 Statistics.py \
#  --batch-size 256 \
#  --in-dataset ImageNet-1K \
#  --id_loc datasets/in-imagenet/val \
#  --ood_loc datasets/ood-imagenet/ \
#  --model resnet50_imagenet \

# python3 Statistics.py \
#  --batch-size 256 \
#  --in-dataset ImageNet-1K \
#  --id_loc datasets/in-imagenet/val \
#  --ood_loc datasets/ood-imagenet/ \
#  --model densenet121_imagenet \

# python3 Statistics.py \
#  --batch-size 256 \
#  --in-dataset ImageNet-1K \
#  --id_loc datasets/in-imagenet/val \
#  --ood_loc datasets/ood-imagenet/ \
#  --model efficientnetb0_imagenet \

# python3 Statistics.py \
#  --batch-size 256 \
#  --in-dataset ImageNet-1K \
#  --id_loc datasets/in-imagenet/val \
#  --ood_loc datasets/ood-imagenet/ \
#  --model swinB_imagenet \

# python3 Statistics.py \
#  --batch-size 256 \
#  --in-dataset ImageNet-1K \
#  --id_loc datasets/in-imagenet/val \
#  --ood_loc datasets/ood-imagenet/ \
#  --model swinT_imagenet \